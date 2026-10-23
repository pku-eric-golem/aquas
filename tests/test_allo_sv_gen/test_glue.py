"""CADL glue beyond straight-line code (cadl-allo-integration.md 5.C): `if`,
SPM element reads and writes, and scalar statics kept across instructions.

The instructions run in one RoCC-level simulation, in order, against a Python
model that carries the same state from one instruction to the next.
"""
import shutil
import subprocess

import pytest

from cadl_frontend.allo_sv_gen.fsm import build_fsm, generate_fsm
from cadl_frontend.allo_sv_gen.main import generate_main
from cadl_frontend.allo_sv_gen.ports import assemble_module, load_ports
from cadl_frontend.allo_sv_gen.spm import build_layout, generate_spm
from cadl_frontend.parser import parse_proc
from test_accept import REDUCE, _reduce_instance
from test_fsm import _testbench

SOURCE = """
static buf: [i32; 16];
static count: u32 = 5;
static last: i32 = -7;

#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype put(rs1: u5, rs2: u5, rd: u5) {
    let i: u32 = _irf[rs1] & 15;
    let v: i32 = _irf[rs2];
    buf[i] = v;
    count = count + 1;
    _irf[rd] = count;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000001)]
rtype add2(rs1: u5, rs2: u5, rd: u5) {
    let i: u32 = _irf[rs1] & 15;
    let j: u32 = _irf[rs2] & 15;
    _irf[rd] = buf[i] + buf[j];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000010)]
rtype absval(rs1: u5, rs2: u5, rd: u5) {
    let i: u32 = _irf[rs1] & 15;
    let v: i32 = buf[i];
    if v < $signed(0) {
        _irf[rd] = 0 - v;
    } else {
        last = v;
        _irf[rd] = v;
    }
}

#[opcode(7'b0001011)]
#[funct7(7'b0000011)]
rtype getlast(rs1: u5, rs2: u5, rd: u5) {
    _irf[rd] = last;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000100)]
rtype sumif(rs1: u5, rs2: u5, rd: u5) {
    let n: u32 = _irf[rs1];
    let acc: i32 = 0;
    if n > 0 {
        with
            k: u32 = (0, k_next)
            a: i32 = (0, a_next)
        do {
            let a_next: i32 = a + buf[k];
            let k_next: u32 = k + 1;
        } while (k_next < n);
        acc = a;
    }
    _irf[rd] = acc * 2;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000101)]
rtype indirect(rs1: u5, rs2: u5, rd: u5) {
    let i: u32 = _irf[rs1] & 15;
    _irf[rd] = buf[buf[i] & 15];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000110)]
rtype nested(rs1: u5, rs2: u5, rd: u5) {
    let i: u32 = _irf[rs1];
    let x: u32 = 0;
    if i == 0 {
        x = 100;
    } else {
        if i == 1 {
            x = 200 + count;
        } else {
            x = 300 + count;
        }
    }
    _irf[rd] = x;
}
"""


class Model:
    def __init__(self):
        self.buf, self.count, self.last = [0] * 16, 5, -7

    def put(self, i, v):
        self.buf[i & 15] = v
        self.count += 1
        return self.count

    def add2(self, i, j):
        return self.buf[i & 15] + self.buf[j & 15]

    def absval(self, i, _):
        v = self.buf[i & 15]
        if v < 0:
            return -v
        self.last = v
        return v

    def getlast(self, *_):
        return self.last

    def sumif(self, n, _):
        return 2 * sum(self.buf[:n]) if n > 0 else 0

    def indirect(self, i, _):
        return self.buf[self.buf[i & 15] & 15]

    def nested(self, i, _):
        return 100 if i == 0 else (200 if i == 1 else 300) + self.count


FUNCT = {"put": 0, "add2": 1, "absval": 2, "getlast": 3, "sumif": 4, "indirect": 5, "nested": 6}
SEQUENCE = [("put", 3, -5), ("put", 4, 9), ("put", 0, 4), ("put", 1, -2), ("add2", 3, 4),
            ("absval", 3, 0), ("getlast", 0, 0), ("absval", 4, 0), ("getlast", 0, 0),
            ("sumif", 5, 0), ("sumif", 0, 0), ("indirect", 0, 0),
            ("nested", 0, 0), ("nested", 1, 0), ("nested", 7, 0)]


def _cases():
    model, cases = Model(), []
    for name, a, b in SEQUENCE:
        want = getattr(model, name)(a, b)
        cases.append((0x0b, FUNCT[name], a & 0xFFFFFFFF, b & 0xFFFFFFFF, want & 0xFFFFFFFF))
    return cases


def _main(source: str) -> str:
    proc = parse_proc(source)
    layout, ports = build_layout(proc), load_ports()
    fsm = build_fsm(proc, {}, layout)
    return assemble_module(ports, generate_spm(layout) + generate_fsm(fsm, {p.name for p in ports}))


def test_join_and_spm_read_states():
    fsm = build_fsm(parse_proc(SOURCE), {}, build_layout(parse_proc(SOURCE)))
    names = [s.name for s in fsm.states]
    assert sum(n.startswith("S_add2_spm_read") for n in names) == 2
    assert sum(n.startswith("S_indirect_spm_read") for n in names) == 2  # inner index first
    assert sum(n.startswith("S_nested_join") for n in names) == 2
    assert not any(n.startswith("S_absval_join") for n in names)  # both arms answer
    assert [(v.reg, init) for v, init in fsm.persistent] == [("static_count", 5), ("static_last", -7)]


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_glue_simulation(tmp_path):
    cases = _cases()
    (tmp_path / "main.sv").write_text(_main(SOURCE))
    (tmp_path / "tb.sv").write_text(_testbench(cases))
    subprocess.run(["verilator", "--binary", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv", "main.sv"],
                   cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    assert f"ROCC PASS cases={len(cases)}" in out, out


def test_csr_register_rejected():
    source = SOURCE.replace("_irf[rd] = last;", "_irf[rd] = gain;")
    source = "#[csr_address(0x801)]\nregister gain: u32;\n" + source
    with pytest.raises(NotImplementedError, match="CSR register gain"):
        _main(source)


def test_partitioned_spm_rejected_for_invoke():
    source = REDUCE.replace("static red_x", "#[partition_dim_array([0])]\nstatic red_x", 1)
    with pytest.raises(NotImplementedError, match="red_x is partitioned"):
        generate_main(parse_proc(source), _reduce_instance())
