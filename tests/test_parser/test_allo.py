"""Allo syntax/structure tests require neither Allo nor its MLIR bindings."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cadl_frontend import cadl_ast as c, parse_proc
from cadl_frontend.allo import check_source, reject_allo_backend
from cadl_frontend.parser import CADLParseError

ROOT = Path(__file__).resolve().parents[2]


def declaration(name="kernel", args="X: int32[16], factor: int32", returns="", body="pass", extra=""):
    return f'allo {name} with s """\ndef {name}({args}){returns}:\n    {body}\ns = None\n{extra}\n""";\n'


def program(call, decl=None, prefix="", body_prefix=""):
    return (decl or declaration()) + "static buf: [i32; 64];\n" + prefix + (
        "rtype command(rs1: u5, rd: u5) {\n" + body_prefix + call + "\n}")


def test_preserves_python_and_positions_without_running_it(tmp_path):
    sentinel = tmp_path / "executed"
    code = '''
    import does_not_exist
    note = {'url': "https://host/a", 'text': '/* not a CADL comment */'}
    # #[opcode(3)] and } must remain Python text
    def kernel(X: int32[16], factor: int32):
        ''' + "'''Python docstring with \"quotes\".'''" + '''
        return None
    s = None
    ''' + f'open({str(sentinel)!r}, "w").write("bad")\n'
    # Indent the last statement just like the rest of the embedded module.
    source = 'allo kernel with s """' + code + '""";'
    proc = check_source(source)
    block = proc.allo_kernels['kernel'].source
    assert block.raw == '"""' + code + '"""'
    assert block.text.startswith('\nimport does_not_exist')
    assert not sentinel.exists()
    assert source[block.span.start_pos:block.span.end_pos] == block.raw
    assert block.source_position(2, 1) == (2, 5)
    assert "Allo Kernels:" in proc.pretty_print()


def test_single_quote_delimiter_and_inline_source():
    source = "allo k with s '''def k(): pass\ns = None\n'''; rtype run() { invoke k(); }"
    proc = check_source(source)
    assert proc.allo_kernels['k'].source.text.startswith('def k')
    assert proc.allo_kernels['k'].source.source_position(1, 1) == (1, source.index('def') + 1)
    assert isinstance(proc.flows['run'].body[0], c.InvokeStmt)


@pytest.mark.parametrize("name,length", [("vector_add", 16), ("crc_words", 32), ("different_name", 8)])
def test_generic_named_bindings_and_void_output_parameters(name, length):
    decl = declaration(name, f"src: int32[{length}], scale: int32, dst: int32[{length}]", body="dst[0] = src[0] * scale")
    source = program(f"invoke {name}(dst = buf[32 +: {length}], scale = _irf[rs1], src = buf[0 +: {length}],);", decl)
    proc = check_source(source)
    call = proc.flows['command'].body[0]
    assert call.kernel == name
    assert [arg.name for arg in call.bindings] == ['dst', 'scale', 'src']
    assert call.results == []
    assert isinstance(call.bindings[0].value, c.RangeSliceExpr)


@pytest.mark.parametrize("returns,target,result_types", [
    (" -> uint32", "sum: u32", [c.AlloScalarResult]),
    (" -> int32[4, 4]", "buf[16 +: 16]", [c.AlloArrayResult]),
    (" -> (int32[4, 4], uint32)", "(buf[16 +: 16], status: u32,)", [c.AlloArrayResult, c.AlloScalarResult]),
    (" -> tuple[int32[16], uint32]", "(buf[16 +: 16], status: u32)", [c.AlloArrayResult, c.AlloScalarResult]),
])
def test_scalar_array_and_tuple_returns(returns, target, result_types):
    source = program(f"invoke kernel(X = buf[0 +: 16], factor = 2) -> {target};", declaration(returns=returns))
    call = check_source(source).flows['command'].body[0]
    assert [type(result) for result in call.results] == result_types
    assert "invoke kernel" in str(call)


def test_result_locals_can_feed_later_calls_and_user_controls_rd():
    decl = declaration(returns=" -> uint32", body="return factor")
    source = program('''invoke kernel(X = buf[0 +: 16], factor = 2) -> value: u32;
invoke kernel(X = buf[0 +: 16], factor = value) -> second: u32;
_irf[rd] = second + 1;''', decl)
    proc = check_source(source)
    assert isinstance(proc.flows['command'].body[-1], c.AssignStmt)


@pytest.mark.parametrize("call,message", [
    ("invoke absent();", "Unknown Allo kernel"),
    ("invoke kernel(X = buf[0 +: 16], X = buf[0 +: 16], factor = 1);", "Duplicate Allo parameter"),
    ("invoke kernel(typo = 0, factor = 1);", "Unknown parameter"),
    ("invoke kernel(X = buf[0 +: 16]);", "Missing Allo parameter"),
    ("invoke kernel(X = _mem[0 +: 16], factor = 1);", "global static SPM"),
    ("invoke kernel(X = nonexistent[0 +: 16], factor = 1);", "global static SPM"),
    ("invoke kernel(X = buf[0 +: ], factor = 1);", "positive constant"),
    ("invoke kernel(X = buf[0 +: 0], factor = 1);", "positive constant"),
    ("invoke kernel(X = buf[0 +: -1], factor = 1);", "positive constant"),
    ("invoke kernel(X = buf[0 +: rs1], factor = 1);", "positive constant"),
    ("invoke kernel(X = buf[-1 +: 16], factor = 1);", "exceeds"),
    ("invoke kernel(X = buf[50 +: 16], factor = 1);", "exceeds"),
    ("invoke kernel(X = buf[rs1 +: 65], factor = 1);", "exceeds"),
    ("invoke kernel(X = buf, factor = 1);", "explicit SPM range"),
    ("invoke kernel(X = buf[missing +: 16], factor = 1);", "Unknown CADL value"),
    ("invoke kernel(X = buf[0 +: 16], factor = missing);", "Unknown CADL value"),
    ("invoke kernel(X = buf[0 +: 16], factor = 1) -> result: u32;", "0 return values"),
    ("spawn { invoke kernel(X = buf[0 +: 16], factor = 1); }", "inside spawn"),
])
def test_invalid_bindings(call, message):
    with pytest.raises(CADLParseError, match=message) as error:
        check_source(program(call), "binding.cadl")
    assert error.value.filename == "binding.cadl"
    assert error.value.line > 0


@pytest.mark.parametrize("target", ["rs1: u32", "buf: u32", "_irf: u32", "(n: u32, n: u32)"])
def test_scalar_result_must_be_new_local(target):
    returns = " -> (uint32, uint32)" if target.startswith('(') else " -> uint32"
    with pytest.raises(CADLParseError, match="new local"):
        check_source(program(f"invoke kernel(X = buf[0 +: 16], factor = 1) -> {target};", declaration(returns=returns)))


def test_inputs_and_result_offsets_cannot_reference_results_from_same_call():
    decl = declaration(returns=" -> (uint32, int32[16])")
    with pytest.raises(CADLParseError, match="Unknown CADL value"):
        check_source(program("invoke kernel(X=buf[0 +: 16], factor=1) -> (n:u32, buf[n +: 16]);", decl))


def test_unresolved_return_alias_is_deferred_to_allo():
    decl = declaration(returns=" -> types.ResultPair")
    check_source(program("invoke kernel(X=buf[0 +: 16], factor=1) -> (a:u32, b:u32);", decl))


def test_zero_argument_scalar_kernel_and_multidimensional_spm():
    decl = declaration("constant", args="", returns=" -> int32", body="return 7")
    check_source(decl + "rtype run(rd:u5) { invoke constant() -> value:i32; _irf[rd] = value; }")
    source = program("invoke kernel(X=buf[32 +: 4 * 4], factor=1);",
                     declaration(args="X: int32[4,4], factor: int32"))
    check_source(source.replace("[i32; 64]", "[i32; 4; 16]"))


def test_nested_calls_and_scope():
    call = "invoke kernel(X=buf[row +: 16], factor=1);"
    check_source(program(f"with row: u32 = (0, next) do {{ if row < 4 {{ {call} }} let next:u32=row+1; }} while(next<4);"))
    with pytest.raises(CADLParseError, match="unshadowed"):
        check_source(program(call, body_prefix="let buf:u32=0; let row:u32=0;"))


def test_duplicate_kernel_and_missing_exports():
    with pytest.raises(CADLParseError, match="Duplicate Allo kernel"):
        parse_proc(declaration() + declaration())
    with pytest.raises(CADLParseError, match="top-level Python def"):
        check_source('allo nope with s """s = None""";')
    with pytest.raises(CADLParseError, match="Schedule name"):
        check_source('allo kernel with missing """def kernel(): pass""";')
    with pytest.raises(CADLParseError, match="Schedule name"):
        check_source('allo kernel with s """def kernel():\n    s = None\n""";')


def test_python_error_reports_cadl_line_and_column():
    source = 'static buf:[u32;4];\nallo broken with s """\n    def broken(:\n        pass\n    s = None\n""";'
    with pytest.raises(CADLParseError, match="Invalid embedded Python") as error:
        check_source(source, "python.cadl")
    assert error.value.line == 3
    assert error.value.column == source.splitlines()[2].index(':') + 1


@pytest.mark.parametrize("bad", [
    'allo k with s """def k(): pass',
    'rtype r() { invoke kernel(X=0) -> _irf[rd]; }',
    'rtype r() { invoke kernel(X=0) -> n: [u32;4]; }',
    'rtype r() { invoke kernel(X=0) -> buf[0 +: ]; }',
])
def test_malformed_syntax(bad):
    with pytest.raises(CADLParseError):
        parse_proc(bad)


def run_cli(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, str(ROOT/'tools/aps-frontend'), *map(str, args)],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def test_full_gemm_example_and_instruction_encodings():
    example = ROOT/'examples/allo/gemm.cadl'
    proc = check_source(example.read_text(), str(example))
    assert len(proc.allo_kernels) == 1 and len(proc.flows) == 4
    assert [p.ty.dimensions for p in proc.statics.values()] == [[131072]] * 3
    result = run_cli('check', example)
    assert result.returncode == 0, result.stderr
    assert "not executed" in result.stdout
    result = run_cli('encoding', example)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        'gemm8x8_fp32': {'opcode': '0xB', 'funct7': '0x0'},
        'loadmat_a_8tile': {'opcode': '0x7B', 'funct7': '0x0'},
        'loadmat_b_8tile': {'opcode': '0x7B', 'funct7': '0x1'},
        'storemat_c_8tile': {'opcode': '0x7B', 'funct7': '0x2'},
    }


@pytest.mark.parametrize('backend', ['mlir', 'cadl2c'])
def test_legacy_cli_refuses_allo_before_writing_output(tmp_path, backend):
    output = tmp_path/'result'
    output.write_text('preserve existing output')
    result = run_cli(backend, ROOT/'examples/allo/gemm.cadl', '-o', output)
    assert result.returncode == 1
    assert 'independent Allo + Vitis wrapper backend is not implemented' in result.stderr
    assert output.read_text() == 'preserve existing output'


def test_legacy_guard_also_finds_nested_undeclared_invocations():
    proc = parse_proc('rtype r() { if true { invoke unknown(); } }')
    with pytest.raises(NotImplementedError, match='Aquas HLS'):
        reject_allo_backend(proc, 'mlir')


def test_c_api_rejects_unused_allo_declaration(tmp_path):
    from cadl_frontend.to_c import CTranspiler
    source = tmp_path/'kernel.cadl'
    source.write_text(declaration())
    with pytest.raises(NotImplementedError, match='Allo'):
        CTranspiler().transpile(source)


def test_cli_check_has_no_allo_circt_or_compiler_dependency(monkeypatch, capsys):
    import runpy
    import importlib.abc

    class NoCompilerImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'allo', 'circt'} or fullname.startswith('cadl_frontend.to_'):
                raise AssertionError(f"check must not load {fullname}")

    # Do this in a fresh import context even if other tests loaded the compilers.
    with monkeypatch.context() as patch:
        for name in list(sys.modules):
            if name.split('.')[0] in {'allo', 'circt'} or name.startswith('cadl_frontend.to_'):
                patch.delitem(sys.modules, name)
        patch.setattr(sys, 'meta_path', [NoCompilerImports(), *sys.meta_path])
        patch.setattr(sys, 'argv', ['aps-frontend', 'check', str(ROOT/'examples/allo/gemm.cadl')])
        with pytest.raises(SystemExit) as result:
            runpy.run_path(str(ROOT/'tools/aps-frontend'), run_name='__main__')
        assert result.value.code == 0
    assert 'structural check PASS' in capsys.readouterr().out


def test_mlir_api_rejects_unused_allo_declaration():
    pytest.importorskip('circt')
    from cadl_frontend.to_mlir import convert_cadl_to_mlir
    with pytest.raises(NotImplementedError, match='Allo'):
        convert_cadl_to_mlir(parse_proc(declaration()))
