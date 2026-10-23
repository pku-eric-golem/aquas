import json
import re
import sys
from pathlib import Path

import pytest

allo = pytest.importorskip("allo")
from allo.ir.types import float32, float64, int32, uint32  # noqa: E402

from cadl_frontend.allo_kernel.hls import emit_project  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/allo_manual"))
from schedule import build_schedule  # noqa: E402


def test_gemm_native_interface(tmp_path):
    project = emit_project(build_schedule(), tmp_path)
    assert [(p.name, p.kind) for p in project.params] == [
        ("A", "ap_memory"), ("BT", "ap_memory"), ("C", "ap_memory"),
        ("use_c", "ap_none"), ("D", "ap_memory")]
    assert project.float_calls == {"hf_add_f32": 2, "hf_mul_f32": 1}

    code = (tmp_path / "kernel.cpp").read_text()
    assert "m_axi" not in code
    assert "float A[64]" in code and "uint8_t use_c" in code
    assert code.count("hf_add_f32(") == 2 and code.count("hf_mul_f32(") == 1
    assert not re.search(r"float v\d+ = v\d+ [-+*/] v\d+;", code)

    tcl = (tmp_path / "run.tcl").read_text()
    assert "set_directive_interface -mode ap_memory -latency 1 gemm_fp32 D" in tcl
    assert "set_directive_interface -mode ap_none gemm_fp32 use_c" in tcl

    configs = [json.loads((tmp_path / f"blackbox/{n}.json").read_text())
               for n in ("hf_add_f32", "hf_mul_f32")]
    bundled = [c for c in configs if any(f.endswith("HardFloatNative.v") for f in c["rtl_files"])]
    assert len(bundled) == 1
    assert [c["c_function_name"] for c in configs] == ["hf_add_f32_rtl", "hf_mul_f32_rtl"]
    assert all(c["c_return"]["rtl_ports"]["data_write_out"] == "result" for c in configs)


def test_unsupported_float_op_rejected(tmp_path):
    def div(A: float32[4], B: float32[4]):
        for i in range(4):
            B[i] = A[i] / 3.0

    def wide(A: float64[4], B: float64[4]):
        for i in range(4):
            B[i] = A[i] + A[i]

    for kernel, error in ((div, "no HardFloat blackbox for .*arith.divf"),
                          (wide, r"arith.addf on \['f64', 'f64', 'f64'\]")):
        s = allo.customize(kernel)
        s.pipeline("i")
        with pytest.raises(NotImplementedError, match=error):
            emit_project(s, tmp_path / kernel.__name__)


def test_float_ops_lowered_to_blackboxes(tmp_path):
    def ops(A: float32[4], B: float32[4], I: int32[4], U: uint32[4], C: int32[4], F: float32[4]):
        for i in range(4):
            c: int32 = int(A[i] - B[i])
            if A[i] != B[i]:  # Allo emits cmpf one (false for NaN), not Python's une
                c = c + 1
            C[i] = c
            F[i] = float(I[i]) * float(U[i]) + (-A[i])

    s = allo.customize(ops)
    s.pipeline("i")
    project = emit_project(s, tmp_path)
    assert project.float_calls == {"hf_add_f32": 1, "hf_sub_f32": 1, "hf_mul_f32": 1,
                                   "hf_sitofp_f32": 1, "hf_uitofp_f32": 1, "hf_fptosi_f32": 1,
                                   "hf_neg_f32": 1, "hf_cmp_one_f32": 1}
    code = (tmp_path / "kernel.cpp").read_text()
    assert "hf_cmp_one_f32(" in code and " - " not in code.split("void ops(")[1]
    assert json.loads((tmp_path / "blackbox/hf_cmp_one_f32.json").read_text())[
        "rtl_performance"] == {"latency": "1", "II": "1"}


def test_float_op_needs_pipelined_loop(tmp_path):
    def add(A: float32[4], B: float32[4]):
        for i in range(4):
            B[i] = A[i] + 1.0

    with pytest.raises(NotImplementedError, match="outside a pipelined loop"):
        emit_project(allo.customize(add), tmp_path / "plain")
    s = allo.customize(add)
    s.pipeline("i")
    assert emit_project(s, tmp_path / "pipelined").float_calls == {"hf_add_f32": 1}


def test_constant_conversions_folded(tmp_path):
    # `v: float32 = 0` is sitofp of a constant (allo.library.systolic's PEs); no loop is pipelined.
    def init(B: float32[2]):
        v: float32 = 0
        w: float32 = 16777217  # 2**24 + 1 rounds to even
        B[0] = v
        B[1] = w

    project = emit_project(allo.customize(init), tmp_path)
    assert project.float_calls == {}
    code = (tmp_path / "kernel.cpp").read_text()
    assert "hf_sitofp_f32(" not in code and "16777216.000000" in code
