"""Subtask C: invoke a real HLS GEMM core through the generated wrapper."""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cadl_frontend.allo_sv_gen.instantiate import KernelParam, classify, load_rtl_ports
from cadl_frontend.allo_sv_gen.main import generate_main
from cadl_frontend.parser import parse_proc

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "examples/allo_manual"
GEMM = (ROOT / "examples/allo/gemm.cadl").read_text()
# DMA instructions belong to subtask D; keep statics, the kernel and gemm8x8_fp32.
GEMM_ONLY = GEMM[:GEMM.index("// rs1: DRAM byte address")]
PARAMS = [KernelParam("A", "ap_memory"), KernelParam("BT", "ap_memory"),
          KernelParam("C", "ap_memory"), KernelParam("use_c", "ap_none"),
          KernelParam("D", "ap_memory")]
BASES = {"A": 0, "BT": 0x80000, "CD": 0x100000}
# (tile_a, tile_b, tile_c, tile_d, use_c, reserved bit 28 set)
CALLS = [(0, 1, 3, 9, 0, False), (5, 7, 3, 3, 1, False), (2047, 2047, 9, 2047, 1, True)]


def instance():
    return classify("gemm_fp32", load_rtl_ports(MANUAL / "rtl/gemm_fp32.v"), PARAMS)


def reference_gemm(a, bt, c, use_c):
    """Sequential FP32 order of the Allo kernel: no fusion, k ascending."""
    d = np.zeros(64, np.float32)
    for i in range(8):
        for j in range(8):
            acc = np.float32(0)
            for k in range(8):
                acc = np.float32(acc + np.float32(a[i * 8 + k] * bt[j * 8 + k]))
            d[i * 8 + j] = np.float32(acc + (c[i * 8 + j] if use_c else np.float32(0)))
    return d


def _beats(values):
    bits = values.view(np.uint32)
    return [(int(bits[n + 1]) << 32) | int(bits[n]) for n in range(0, 64, 2)]


def _testbench():
    rng = np.random.default_rng(7)
    tiles = {("A", t): rng.standard_normal(64).astype(np.float32) for t in (0, 5, 2047)}
    tiles |= {("BT", t): rng.standard_normal(64).astype(np.float32) for t in (1, 7, 2047)}
    tiles |= {("CD", t): rng.standard_normal(64).astype(np.float32) for t in (3, 9, 2047)}
    lines = []

    def write(region, tile, values):
        for n, beat in enumerate(_beats(values)):
            lines.append(f"    write_beat(32'h{BASES[region] + tile * 256 + n * 8:08x}, 64'h{beat:016x});")

    for (region, tile), values in tiles.items():
        write(region, tile, values)
    for ta, tb, tc, td, use_c, reserved in CALLS:
        rs1 = (ta << 16) | (use_c << 15) | tb | (1 << 28 if reserved else 0)
        rs2 = (tc << 16) | td
        lines.append(f"    gemm(32'h{rs1:08x}, 32'h{rs2:08x});")
        d = reference_gemm(tiles["A", ta], tiles["BT", tb], tiles["CD", tc], use_c)
        tiles["CD", td] = d
        for n, beat in enumerate(_beats(d)):
            lines.append(f"    read_check(32'h{BASES['CD'] + td * 256 + n * 8:08x}, 64'h{beat:016x});")
    return TB.replace("STEPS", "\n".join(lines)).replace("NCALLS", str(len(CALLS)))


TB = """module tb;
  reg clk = 0, rst = 1;
  always #1 clk = ~clk;
  reg cmd_enable = 0;
  reg [31:0] rs1data = 0, rs2data = 0;
  reg burst_write_enable = 0, burst_read_0_enable = 0;
  reg [31:0] burst_write_addr = 0, burst_read_0_addr = 0;
  reg [63:0] burst_write_data = 0;
  wire [63:0] burst_read_1_res0;
  wire cmd_ready, resp_enable;
  wire [31:0] resp_data;
  main dut(
    .clk(clk), .rst(rst),
    .rocc_cmd_enable(cmd_enable), .rocc_cmd_ready(cmd_ready),
    .rocc_cmd_rocc_cmd_opcode(7'h0b), .rocc_cmd_rocc_cmd_funct(7'h00),
    .rocc_cmd_rocc_cmd_rd(5'd3), .rocc_cmd_rocc_cmd_xd(1'b1),
    .rocc_cmd_rocc_cmd_rs1data(rs1data), .rocc_cmd_rocc_cmd_rs2data(rs2data),
    .rocc_resp_rocc_resp_to_bus_enable(resp_enable),
    .rocc_resp_rocc_resp_to_bus_ready(1'b1),
    .rocc_resp_rocc_resp_to_bus_result_rddata(resp_data),
    .burst_write_enable(burst_write_enable), .burst_write_addr(burst_write_addr),
    .burst_write_data(burst_write_data),
    .burst_read_0_enable(burst_read_0_enable), .burst_read_0_addr(burst_read_0_addr),
    .burst_read_1_enable(1'b1), .burst_read_1_res0(burst_read_1_res0));
  integer errors = 0, cycles, calls = 0;
  task automatic write_beat(input [31:0] addr, input [63:0] data);
    begin
      burst_write_enable = 1; burst_write_addr = addr; burst_write_data = data;
      @(negedge clk);
      burst_write_enable = 0;
    end
  endtask
  task automatic read_check(input [31:0] addr, input [63:0] want);
    begin
      burst_read_0_enable = 1; burst_read_0_addr = addr;
      @(negedge clk);
      burst_read_0_enable = 0;
      if (burst_read_1_res0 !== want) begin
        if (errors < 8) $display("GEMM FAIL addr=%h got=%h want=%h", addr, burst_read_1_res0, want);
        errors = errors + 1;
      end
    end
  endtask
  task automatic gemm(input [31:0] a, input [31:0] b);
    begin
      while (!cmd_ready) @(negedge clk);
      cmd_enable = 1; rs1data = a; rs2data = b;
      @(negedge clk);
      cmd_enable = 0; rs1data = 32'hdead_beef; rs2data = 32'hdead_beef;
      cycles = 0;
      while (!resp_enable && cycles < 20000) begin @(negedge clk); cycles = cycles + 1; end
      if (!resp_enable || resp_data !== 0) begin
        $display("GEMM FAIL: no response for %h %h", a, b);
        errors = errors + 1;
      end
      calls = calls + 1;
      $display("gemm call %0d: %0d cycles to response", calls, cycles);
      @(negedge clk);
    end
  endtask
  initial begin
    repeat (3) @(negedge clk);
    rst = 0;
STEPS
    if (errors == 0) $display("GEMM PASS calls=NCALLS");
    $finish;
  end
endmodule
"""


def _simulate(work: Path, main_sv: str) -> str:
    work.mkdir()
    (work / "main.sv").write_text(main_sv)
    (work / "tb.sv").write_text(_testbench())
    rtl = sorted(str(p) for p in (MANUAL / "rtl").glob("*.v"))
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    subprocess.run(["verilator", "--binary", "-j", "4", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv", "main.sv",
                    *rtl], cwd=work, env=env, check=True, capture_output=True)
    return subprocess.run([str(work / "obj/Vtb")], cwd=work, env=env, check=True,
                          capture_output=True, text=True).stdout


def test_invoke_binding_errors():
    proc = parse_proc(GEMM_ONLY.replace("use_c = use_c,", ""))
    with pytest.raises(ValueError, match="kernel takes"):
        generate_main(proc, instance())
    proc = parse_proc(GEMM_ONLY.replace("A = matA[tile_a * 64 +: 64]", "A = tile_a"))
    with pytest.raises(ValueError, match="must bind an SPM range"):
        generate_main(proc, instance())


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_gemm_invoke_matches_reference_and_golden(tmp_path):
    generated = _simulate(tmp_path / "generated", generate_main(parse_proc(GEMM_ONLY), instance()))
    golden = _simulate(tmp_path / "golden", (MANUAL / "main.sv").read_text())
    for out in (generated, golden):
        assert f"GEMM PASS calls={len(CALLS)}" in out, out
    cycles = lambda out: [l.split(": ")[1] for l in out.splitlines() if l.startswith("gemm call")]  # noqa: E731
    assert cycles(generated) == cycles(golden)
