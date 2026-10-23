"""Runtime checks and the error ABI (cadl-allo-integration.md 5.C, 5.D.3).

A failing check aborts the instruction with rd = 0 and ORs its cause into a
sticky error register; opcode 0x7b funct7 0x7f returns and clears it. Aborted
accesses must not reach memory, the DMA engine or the core.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from cadl_frontend.allo_sv_gen.fsm import ERRORS
from cadl_frontend.allo_sv_gen.main import generate_main
from cadl_frontend.parser import parse_proc
from test_dma import MANUAL, instance
from test_mem import TB, WORDS

ROOT = Path(__file__).resolve().parents[2]
GEMM = (ROOT / "examples/allo/gemm.cadl").read_text()
SPM = 131072  # elements of each GEMM scratchpad
SOURCE = GEMM[:GEMM.index("#[opcode")] + """
#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype elem_rd(rs1: u5, rs2: u5, rd: u5) {
    _irf[rd] = matA[_irf[rs1]];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000001)]
rtype elem_wr(rs1: u5, rs2: u5, rd: u5) {
    let i: u32 = _irf[rs1];
    let v: u32 = _irf[rs2];
    matA[i] = v;
    _irf[rd] = 1;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000010)]
rtype ldw(rs1: u5, rs2: u5, rd: u5) {
    _irf[rd] = _mem[_irf[rs1]];
}

#[opcode(7'b0001011)]
#[funct7(7'b0000011)]
rtype run(rs1: u5, rs2: u5, rd: u5) {
    let a: u32 = _irf[rs1];
    invoke gemm_fp32(A = matA[a +: 64], BT = matBT[0 +: 64], C = matCD[0 +: 64],
                     use_c = 0, D = matCD[64 +: 64]);
    _irf[rd] = 1;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000100)]
rtype dma(rs1: u5, rs2: u5, rd: u5) {
    let src: u32 = _irf[rs1];
    let dst: u32 = _irf[rs2];
    matA[dst +: ] = _mem[src +: 32];
    _irf[rd] = 1;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000101)]
rtype row_rd(rs1: u5, rs2: u5, rd: u5) {
    _irf[rd] = matA[_irf[rs1] * 16];
}
"""

OP, STATUS = 0x0b, (0x7b, 0x7f)
E = ERRORS
INIT = [(i * 37 + 11) & 0xFFFFFFFF for i in range(WORDS)]
SEQUENCE = [  # (opcode, funct7, rs1, rs2, rd)
    (*STATUS, 0, 0, 0),
    (OP, 1, 5, 0x40490FDB, 1), (OP, 0, 5, 0, 0x40490FDB),
    (OP, 0, SPM, 0, 0), (*STATUS, 0, 0, E["spm_index"]), (*STATUS, 0, 0, 0),
    (OP, 1, SPM + 3, 7, 0), (*STATUS, 0, 0, E["spm_index"]),
    (OP, 2, 0x103, 0, 0), (*STATUS, 0, 0, E["mem_align"]), (OP, 2, 0x100, 0, INIT[0x40]),
    (OP, 3, SPM - 32, 0, 0), (*STATUS, 0, 0, E["invoke_range"]), (OP, 3, SPM - 64, 0, 1),
    (OP, 4, 0x100, SPM - 16, 0), (*STATUS, 0, 0, E["dma_range"]),
    (OP, 0, 1 << 20, 0, 0), (OP, 2, 0x101, 0, 0), (*STATUS, 0, 0, E["spm_index"] | E["mem_align"]),
    (OP, 0, 5, 0, 0x40490FDB),  # state survives the aborted instructions
    # The index arithmetic must not wrap: 2**29 * 16 is 2**33, not element 0.
    (OP, 1, 0, 0x3F800000, 1), (OP, 5, 1 << 29, 0, 0), (OP, 5, 1 << 28, 0, 0),
    (*STATUS, 0, 0, E["spm_index"]), (OP, 5, 0, 0, 0x3F800000),
]
EDITS = [
    ("  task automatic issue(input [6:0] f,", "  task automatic issue(input [6:0] op, input [6:0] f,"),
    ("opcode = 7'h0b; funct = f;", "opcode = op; funct = f;"),
    ("  wire cmd_ready, resp_enable;\n",
     "  wire cmd_ready, resp_enable, dma_enable;\n  integer dma_requests = 0;\n"),
    ("    .hella_resp_enable(r_en),",
     "    .dma_cpu_to_isax_ch0_enable(dma_enable),\n    .hella_resp_enable(r_en),"),
    ("    r_en <= 0;\n", "    r_en <= 0;\n    if (dma_enable) dma_requests = dma_requests + 1;\n"),
    ('$display("MEM PASS requests=%0d", requests);',
     '$display("MEM PASS requests=%0d dma=%0d", requests, dma_requests);'),
]


def _main() -> str:
    return generate_main(parse_proc(SOURCE), instance())


def test_reserved_status_encoding():
    clash = SOURCE.replace("#[funct7(7'b0000100)]\nrtype dma", "#[funct7(7'b1111111)]\nrtype dma")
    clash = clash.replace("#[opcode(7'b0001011)]\n#[funct7(7'b1111111)]",
                          "#[opcode(7'b1111011)]\n#[funct7(7'b1111111)]")
    with pytest.raises(ValueError, match="reserved error status instruction"):
        generate_main(parse_proc(clash), instance())


def test_constant_index_checked_at_build_time():
    with pytest.raises(ValueError, match=r"matA index \[131072, 131073\) outside matA"):
        generate_main(parse_proc(SOURCE.replace("matA[i] = v;", f"matA[{SPM}] = v;")), instance())
    with pytest.raises(ValueError, match="not 4-byte aligned"):
        generate_main(parse_proc(SOURCE.replace("_mem[_irf[rs1]]", "_mem[258]")), instance())


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_errors_simulation(tmp_path):
    tb = TB
    for old, new in EDITS:
        assert tb.count(old) == 1, old
        tb = tb.replace(old, new)
    steps = [f"    issue(7'h{op:02x}, 7'h{f:02x}, 32'h{a:08x}, 32'h{b:08x}, 32'h{want:08x});"
             for op, f, a, b, want in SEQUENCE]
    (tmp_path / "tb.sv").write_text(tb.replace("@WORDS@", str(WORDS)).replace("@STEPS@", "\n".join(steps)))
    (tmp_path / "main.sv").write_text(_main())
    (tmp_path / "mem.hex").write_text("\n".join(f"{w:08x}" for w in INIT) + "\n")
    rtl = sorted(str(p) for p in (MANUAL / "rtl").glob("*.v"))
    subprocess.run(["verilator", "--binary", "-j", "4", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--assert", "--top-module", "tb", "-Mdir", "obj", "tb.sv",
                    "main.sv", *rtl], cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    # Only the aligned load reaches memory; the out-of-range DMA is never issued.
    assert "MEM PASS requests=1 dma=0" in out, out
