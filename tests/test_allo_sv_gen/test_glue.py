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
from cadl_frontend.allo_sv_gen.spm import build_layout, generate_spm, generate_spm_writes
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
    return assemble_module(ports, generate_spm(layout) + generate_fsm(fsm, {p.name for p in ports})
                           + generate_spm_writes(layout, fsm.writes()))


def test_join_and_spm_read_states():
    fsm = build_fsm(parse_proc(SOURCE), {}, build_layout(parse_proc(SOURCE)))
    names = [s.name for s in fsm.states]
    assert sum(n.startswith("S_add2_spm_read") for n in names) == 2
    assert sum(n.startswith("S_indirect_spm_read") for n in names) == 2  # inner index first
    assert sum(n.startswith("S_nested_join") for n in names) == 2
    assert not any(n.startswith("S_absval_join") for n in names)  # both arms answer
    assert [(v.reg, init) for v, init in fsm.persistent] == [("static_count", 5), ("static_last", -7)]


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
@pytest.mark.parametrize("banks", ["", "4, 1", "2, 0"], ids=["flat", "cyclic4", "block2"])
def test_glue_simulation(tmp_path, banks):
    cases = _cases()
    source = SOURCE
    if banks:  # the same sequence on a banked buf (spm.py)
        factor, cyclic = banks.split(", ")
        source = source.replace("static buf", f"#[partition_dim_array([0])]\n"
                                f"#[partition_factor_array([{factor}])]\n"
                                f"#[partition_cyclic_array([{cyclic}])]\nstatic buf", 1)
        assert f"spm_buf [0:{int(factor) - 1}][" in _main(source)
    (tmp_path / "main.sv").write_text(_main(source))
    (tmp_path / "tb.sv").write_text(_testbench(cases))
    subprocess.run(["verilator", "--binary", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv", "main.sv"],
                   cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    assert f"ROCC PASS cases={len(cases)}" in out, out



def test_incomplete_partition_rejected():
    source = REDUCE.replace("static red_x", "#[partition_dim_array([0])]\nstatic red_x", 1)
    with pytest.raises(ValueError, match="give all of partition_dim_array"):
        generate_main(parse_proc(source), _reduce_instance())


# 8- and 16-bit SPMs, written and read by the FSM only (spm.py), in the
# packed-word style of the vla_testbench engines; bitcast changes signedness.

NARROW = """
static n8: [i8; 16];
static n16: [i16; 8];
static nu: [u8; 4];
static wide: [i32; 4];

// rs1: word address. rs2: packed value. rd: 0 accepted, 1 bad address.
#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype nwrite(rs1: u5, rs2: u5, rd: u5) {
    let address: u32 = _irf[rs1];
    let value: u32 = _irf[rs2];
    let accepted: u32 = 0;
    if (address >= 0) && (address < 4) { n8[address * 4 + 0] = (value >> 0) & 255; accepted = 1; }
    if (address >= 0) && (address < 4) { n8[address * 4 + 1] = (value >> 8) & 255; accepted = 1; }
    if (address >= 0) && (address < 4) { n8[address * 4 + 2] = (value >> 16) & 255; accepted = 1; }
    if (address >= 0) && (address < 4) { n8[address * 4 + 3] = (value >> 24) & 255; accepted = 1; }
    if (address >= 4) && (address < 8) { n16[(address - 4) * 2 + 0] = (value >> 0) & 65535; accepted = 1; }
    if (address >= 4) && (address < 8) { n16[(address - 4) * 2 + 1] = (value >> 16) & 65535; accepted = 1; }
    if (address == 8) { nu[0] = value; wide[0] = bitcast<i32>(value); accepted = 1; }
    _irf[rd] = if accepted != 0 {0} else {1};
}

// SPM writes put wait states between the reads of rs1 and rs2, as in the run
// instructions of the vla_testbench engines (latched at command fire, fsm.py).
#[opcode(7'b0001011)]
#[funct7(7'b0000010)]
rtype late(rs1: u5, rs2: u5, rd: u5) {
    wide[1] = _irf[rs1];
    wide[2] = _irf[rs2] + wide[1];
    wide[3] = wide[2] + 1;
    _irf[rd] = _irf[rs1] ^ wide[3];
}

// rs1: word address. rd: the packed word, a sign- or zero-extended element,
// or 0 for an unknown address. Element reads sit in the branches.
#[opcode(7'b0001011)]
#[funct7(7'b0000001)]
rtype nread(rs1: u5, rs2: u5, rd: u5) {
    let address: u32 = _irf[rs1];
    let bits: u32 = 0;
    if (address >= 0) && (address < 4) {
        let p0: i32 = n8[address * 4 + 0]; bits = bits | ((bitcast<u32>(p0) & 255) << 0);
        let p1: i32 = n8[address * 4 + 1]; bits = bits | ((bitcast<u32>(p1) & 255) << 8);
        let p2: i32 = n8[address * 4 + 2]; bits = bits | ((bitcast<u32>(p2) & 255) << 16);
        let p3: i32 = n8[address * 4 + 3]; bits = bits | ((bitcast<u32>(p3) & 255) << 24);
    }
    if (address >= 16) && (address < 32) { let e8: i32 = n8[address - 16]; bits = bitcast<u32>(e8); }
    if (address >= 32) && (address < 40) { let e16: i32 = n16[address - 32]; bits = bitcast<u32>(e16); }
    if (address == 40) { let eu: u32 = nu[0]; bits = eu; }
    if (address == 41) { bits = bitcast<u32>(wide[0]); }
    _irf[rd] = bits;
}
"""


class NarrowModel:
    def __init__(self):
        self.n8, self.n16, self.nu, self.wide = [0] * 16, [0] * 8, 0, 0

    def nwrite(self, address, value):
        if address < 4:
            for k in range(4):
                self.n8[address * 4 + k] = (value >> (8 * k)) & 255
        elif address < 8:
            for k in range(2):
                self.n16[(address - 4) * 2 + k] = (value >> (16 * k)) & 65535
        elif address == 8:
            self.nu, self.wide = value & 255, value
        else:
            return 1
        return 0

    @staticmethod
    def _signed(value, bits):
        return value - (1 << bits) if value >> (bits - 1) else value

    def late(self, a, b):
        return a ^ ((b + a + 1) & 0xFFFFFFFF)

    def nread(self, address, _):
        if address < 4:
            return sum(self.n8[address * 4 + k] << (8 * k) for k in range(4))
        if 16 <= address < 32:
            return self._signed(self.n8[address - 16], 8)
        if 32 <= address < 40:
            return self._signed(self.n16[address - 32], 16)
        if address == 40:
            return self.nu
        if address == 41:
            return self.wide
        return 0


NARROW_SEQUENCE = [
    ("nwrite", 0, 0x80FF7F01), ("nwrite", 3, 0x12345678), ("nwrite", 4, 0x8000FFFF),
    ("nwrite", 7, 0x7FFF0001), ("nwrite", 8, 0xFFFFFF9C), ("nwrite", 9, 5),
    ("nread", 0, 0), ("nread", 3, 0), ("nread", 1, 0),
    ("nread", 16, 0), ("nread", 17, 0), ("nread", 18, 0), ("nread", 19, 0), ("nread", 31, 0),
    ("nread", 32, 0), ("nread", 33, 0), ("nread", 38, 0), ("nread", 39, 0),
    ("nread", 40, 0), ("nread", 41, 0), ("nread", 1000, 0),
    ("late", 0x1234, 0xFFFF0000), ("late", 7, 9),
]


def test_narrow_spm_layout():
    proc = parse_proc(NARROW)
    layout = build_layout(proc)
    assert [(s.name, s.elem_bits, s.signed) for s in layout.spms] == [
        ("n8", 8, True), ("n16", 16, True), ("nu", 8, False), ("wide", 32, True)]
    assert [s.name for s in layout.dma_spms] == ["wide"]
    sv = generate_spm(layout)
    assert "reg [7:0] spm_n8 [0:15];" in sv and "reg [15:0] spm_n16 [0:7];" in sv


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_narrow_spm_simulation(tmp_path):
    model, cases = NarrowModel(), []
    for name, a, b in NARROW_SEQUENCE:
        want = getattr(model, name)(a, b)
        cases.append((0x0b, {"nwrite": 0, "nread": 1, "late": 2}[name], a, b, want & 0xFFFFFFFF))
    (tmp_path / "main.sv").write_text(_main(NARROW))
    (tmp_path / "tb.sv").write_text(_testbench(cases))
    subprocess.run(["verilator", "--binary", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv", "main.sv"],
                   cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    assert f"ROCC PASS cases={len(cases)}" in out, out


def test_dma_to_narrow_spm_rejected():
    source = REDUCE.replace("static red_x", "static red_h: [i16; 16];\nstatic red_x", 1) + """
#[opcode(7'b1111011)]
#[funct7(7'b0000111)]
rtype load_h(rs1: u5, rs2: u5, rd: u5) {
    let dram: u32 = _irf[rs1];
    red_h[0 +: ] = _mem[dram +: 16];
    _irf[rd] = 0;
}
"""
    with pytest.raises(NotImplementedError, match="only 32-bit SPMs are reached by DMA"):
        generate_main(parse_proc(source), _reduce_instance())


def test_unsupported_spm_width_rejected():
    with pytest.raises(NotImplementedError, match="SPM elements are 8, 16, 32 bits"):
        build_layout(parse_proc(NARROW.replace("static nu: [u8; 4];", "static nu: [u4; 4];")))
