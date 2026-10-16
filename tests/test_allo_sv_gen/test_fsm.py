"""Subtask B: decode and FSM translation, checked by RoCC-level simulation."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cadl_frontend.allo_sv_gen.fsm import build_fsm, generate_fsm
from cadl_frontend.allo_sv_gen.ports import assemble_module, load_ports
from cadl_frontend.parser import parse_proc

ROOT = Path(__file__).resolve().parents[2]

# Exercises let chaining with truncation, loop-carried values, nested loops,
# a post-loop read of the final binding and an outer variable updated in a loop.
SOURCE = """
#[opcode(7'b0001011)]
#[funct7(7'b0000001)]
rtype fields(rs1: u5, rs2: u5, rd: u5) {
    let r1: u32 = _irf[rs1];
    let hi: u16 = r1 >> 16;
    let lo: u8 = r1 + 1;
    _irf[rd] = lo * 65536 + hi;
}

#[opcode(7'b0001011)]
#[funct7(7'b0000010)]
rtype accum(rs1: u5, rs2: u5, rd: u5) {
    let n: u32 = _irf[rs1];
    let x: u32 = _irf[rs2];
    with
        i: u32 = (0, i_next)
        acc: u32 = (0, acc_next)
    do {
        let acc_next: u32 = acc + x;
        let i_next: u32 = i + 1;
    } while (i_next < n);
    _irf[rd] = acc + 1;
}

#[opcode(7'b1111011)]
#[funct7(7'b0000011)]
rtype tri(rs1: u5, rs2: u5, rd: u5) {
    let n: u32 = _irf[rs1];
    let calls: u32 = 0;
    with
        i: u32 = (0, i_next)
        total: u32 = (0, total_next)
    do {
        with
            j: u32 = (0, j_next)
            cnt: u32 = (total, cnt_next)
        do {
            let cnt_next: u32 = cnt + 1;
            let j_next: u32 = j + 1;
        } while (j_next <= i);
        let total_next: u32 = cnt;
        calls = calls + 1;
        let i_next: u32 = i + 1;
    } while (i_next < n);
    _irf[rd] = total * 256 + calls;
}
"""


def fields(r1, r2):
    return (((r1 + 1) & 0xFF) * 65536 + (r1 >> 16)) & 0xFFFFFFFF


def accum(n, x):
    acc, i = 0, 0
    while True:
        acc, i = (acc + x) & 0xFFFFFFFF, i + 1
        if not i < n:
            return (acc + 1) & 0xFFFFFFFF


def tri(n, _):
    total, calls, i = 0, 0, 0
    while True:
        cnt, j = total, 0
        while True:
            cnt, j = cnt + 1, j + 1
            if not j <= i:
                break
        total, calls, i = cnt, calls + 1, i + 1
        if not i < n:
            return total * 256 + calls


CASES = [  # opcode, funct7, rs1data, rs2data, expected
    (0x0b, 1, 0xABCD0141, 0, fields(0xABCD0141, 0)),
    (0x0b, 1, 0x000000FF, 0, fields(0x000000FF, 0)),
    (0x0b, 2, 4, 7, accum(4, 7)),
    (0x0b, 2, 1, 3, accum(1, 3)),
    (0x7b, 3, 5, 0, tri(5, 0)),
    (0x7b, 3, 1, 0, tri(1, 0)),
    (0x7b, 0x7f, 0, 0, 1),  # unknown instruction
    (0x0b, 2, 3, 0x10000, accum(3, 0x10000)),
]


def _stub(tag):
    def handler(builder, stmt, directives):
        builder.wait(tag, "1'b1")
    return handler


def test_gemm_decode_and_states():
    proc = parse_proc((ROOT / "examples/allo/gemm.cadl").read_text())
    fsm = build_fsm(proc, {"invoke": _stub("invoke"), "dma": _stub("dma")})
    assert [(d.opcode, d.funct7) for d in fsm.decode] == [(0x0b, 0), (0x7b, 0), (0x7b, 1), (0x7b, 2)]
    names = [s.name for s in fsm.states]
    assert names[:3] == ["S_IDLE", "S_RESPONSE", "S_gemm8x8_fp32_invoke1"]
    assert names[3:6] == ["S_loadmat_a_8tile_loop1", "S_loadmat_a_8tile_dma2",
                          "S_loadmat_a_8tile_dma3"]
    entry = "\n".join(fsm.decode[0].entry.actions)
    assert ("gemm8x8_fp32__tile_a <= 16'(((32'(rocc_cmd_rocc_cmd_rs1data) & 32'd4026531839)"
            " >> 32'd16));") in entry
    assert "gemm8x8_fp32__use_c <= 8'(" in entry


def test_irf_read_after_wait_rejected():
    source = SOURCE.replace("_irf[rd] = acc + 1;", "_irf[rd] = _irf[rs2];")
    with pytest.raises(ValueError, match="only valid before the first wait state"):
        build_fsm(parse_proc(source), {})


def test_missing_handler_rejected():
    proc = parse_proc((ROOT / "examples/allo/gemm.cadl").read_text())
    with pytest.raises(NotImplementedError, match="no handler for invoke"):
        build_fsm(proc, {})


def _testbench(cases) -> str:
    calls = "\n".join(f"    issue(7'h{op:02x}, 7'h{f:02x}, 32'h{a:08x}, 32'h{b:08x}, 32'h{want:08x}, {i % 32});"
                      for i, (op, f, a, b, want) in enumerate(cases))
    return f"""module tb;
  reg clk = 0, rst = 1;
  always #1 clk = ~clk;
  reg cmd_enable = 0;
  reg [6:0] opcode = 0, funct = 0;
  reg [4:0] rd = 0;
  reg [31:0] rs1data = 0, rs2data = 0;
  wire cmd_ready, resp_enable;
  wire [4:0] resp_rd;
  wire [31:0] resp_data;
  main dut(
    .clk(clk), .rst(rst),
    .rocc_cmd_enable(cmd_enable), .rocc_cmd_ready(cmd_ready),
    .rocc_cmd_rocc_cmd_opcode(opcode), .rocc_cmd_rocc_cmd_funct(funct),
    .rocc_cmd_rocc_cmd_rd(rd), .rocc_cmd_rocc_cmd_xd(1'b1),
    .rocc_cmd_rocc_cmd_rs1data(rs1data), .rocc_cmd_rocc_cmd_rs2data(rs2data),
    .rocc_resp_rocc_resp_to_bus_enable(resp_enable),
    .rocc_resp_rocc_resp_to_bus_ready(1'b1),
    .rocc_resp_rocc_resp_to_bus_result_rd(resp_rd),
    .rocc_resp_rocc_resp_to_bus_result_rddata(resp_data));
  integer errors = 0, cycles;
  task automatic issue(input [6:0] op, input [6:0] f, input [31:0] a, input [31:0] b,
                       input [31:0] want, input [4:0] dest);
    begin
      while (!cmd_ready) @(negedge clk);
      cmd_enable = 1; opcode = op; funct = f; rs1data = a; rs2data = b; rd = dest;
      @(negedge clk);
      cmd_enable = 0; rs1data = 32'hdead_beef; rs2data = 32'hdead_beef;
      cycles = 0;
      while (!resp_enable && cycles < 10000) begin @(negedge clk); cycles = cycles + 1; end
      if (!resp_enable || resp_data !== want || resp_rd !== dest) begin
        $display("ROCC FAIL op=%h f=%h got=%h want=%h rd=%0d", op, f, resp_data, want, resp_rd);
        errors = errors + 1;
      end
      @(negedge clk);
    end
  endtask
  initial begin
    repeat (3) @(negedge clk);
    rst = 0;
{calls}
    if (errors == 0) $display("ROCC PASS cases=%0d", {len(cases)});
    $finish;
  end
endmodule
"""


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_fsm_simulation(tmp_path):
    ports = load_ports()
    fsm = build_fsm(parse_proc(SOURCE), {})
    (tmp_path / "main.sv").write_text(assemble_module(ports, generate_fsm(fsm, {p.name for p in ports})))
    (tmp_path / "tb.sv").write_text(_testbench(CASES))
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    subprocess.run(["verilator", "--binary", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv", "main.sv"],
                   cwd=tmp_path, env=env, check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, env=env, check=True,
                         capture_output=True, text=True).stdout
    assert f"ROCC PASS cases={len(CASES)}" in out, out
