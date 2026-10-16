"""Subtask D: the full GEMM ISAX (DMA load -> invoke -> DMA store) end to end.

The scenario mirrors examples/allo_manual/software/test_gemm.c. The bench's
DMA model follows TlDmaMultichannel.scala as wired by ApsWrapper.scala:
ready and poll_for_idle are the channel idle flag, 8-byte beats, and the
stride_x/stride_y tile jump on the SPM side.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cadl_frontend.allo_sv_gen.instantiate import KernelParam, classify, load_rtl_ports
from cadl_frontend.allo_sv_gen.main import generate_main
from cadl_frontend.parser import parse_proc
from test_invoke import reference_gemm

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "examples/allo_manual"
GEMM = (ROOT / "examples/allo/gemm.cadl").read_text()
PARAMS = [KernelParam("A", "ap_memory"), KernelParam("BT", "ap_memory"),
          KernelParam("C", "ap_memory"), KernelParam("use_c", "ap_none"),
          KernelParam("D", "ap_memory")]
DEAD = 0xDEADBEEF
# Byte addresses in the bench DRAM. The image is loaded once, so each batch
# gets its own A/B region (the C test refills one buffer at run time).
D_BUF = 0x4000
D_DATA = D_BUF + 256                              # 256-byte guard on each side
D_WORDS = 64 + 8 * 128 + 64                       # 32-bit words incl. guards
DRAM_WORDS = 0x6000 // 8                          # 64-bit words
BATCHES = [  # row words, a/b/c tile, pass-1 result goes to c+16 or in place
    dict(row_words=64, a=8, b=24, c=40, separate=True),
    dict(row_words=128, a=1008, b=1016, c=1000, separate=False),
]


def instance():
    return classify("gemm_fp32", load_rtl_ports(MANUAL / "rtl/gemm_fp32.v"), PARAMS)


def _dma_args(tile, row_words):
    return ((tile * 64) << 16) | (row_words * 4)


def _scenario(rng):
    """DRAM image, command list and the expected D buffer after each store."""
    dram = np.full(DRAM_WORDS * 2, DEAD, np.uint32)
    commands, expected = [], []
    for index, batch in enumerate(BATCHES):
        rw = batch["row_words"]
        A_DATA, B_DATA = 0x2000 * index, 0x2000 * index + 0x1000
        a = [rng.standard_normal(64).astype(np.float32) for _ in range(8)]
        bt = [rng.standard_normal(64).astype(np.float32) for _ in range(8)]
        for t in range(8):
            for r in range(8):
                for k in range(8):
                    dram[A_DATA // 4 + r * rw + t * 8 + k] = a[t][r * 8 + k].view(np.uint32)
                    dram[B_DATA // 4 + r * rw + t * 8 + k] = bt[t][r * 8 + k].view(np.uint32)
        commands.append(("load_a", 0x7B, 0, A_DATA, _dma_args(batch["a"], rw)))
        commands.append(("load_b", 0x7B, 1, B_DATA, _dma_args(batch["b"], rw)))
        first = [reference_gemm(a[t], bt[t], None, 0) for t in range(8)]
        results = [first, [reference_gemm(a[t], bt[t], first[t], 1) for t in range(8)]]
        for pass_ in (0, 1):
            d = batch["c"] + (16 if pass_ and batch["separate"] else 0)
            for t in range(8):
                rs1 = ((batch["a"] + t) << 16) | (batch["b"] + t) | (pass_ << 15)
                rs2 = ((batch["c"] + t) << 16) | (d + t)
                commands.append(("gemm", 0x0B, 0, rs1, rs2))
            commands.append(("store_c", 0x7B, 2, D_DATA, _dma_args(d, rw)))
            want = np.full(D_WORDS, DEAD, np.uint32)
            for r in range(8):
                for col in range(64):
                    want[64 + r * rw + col] = results[pass_][col // 8][r * 8 + col % 8].view(np.uint32)
            expected.append(want)
    return dram, commands, expected


TB = """module tb;
  reg clk = 0, rst = 1;
  always #1 clk = ~clk;
  localparam integer DRAM_WORDS = @DRAM_WORDS@;
  reg [63:0] dram [0:DRAM_WORDS-1];

  reg cmd_enable = 0;
  reg [6:0] opcode = 0, funct = 0;
  reg [31:0] rs1data = 0, rs2data = 0;
  wire cmd_ready, resp_enable;
  wire [31:0] resp_data;

  wire [31:0] l_cpu, l_isax, s_cpu, s_isax;
  wire [3:0] l_len, s_len;
  wire [7:0] l_sx, l_sy, s_sx, s_sy;
  wire l_en, s_en;
  reg burst_write_enable = 0, burst_read_0_enable = 0;
  reg [31:0] burst_addr = 0;
  reg [63:0] burst_write_data = 0;
  wire [63:0] burst_res0;
  reg [2:0] dstate = 0;
  wire idle = dstate == 0;

  main dut(
    .clk(clk), .rst(rst),
    .rocc_cmd_enable(cmd_enable), .rocc_cmd_ready(cmd_ready),
    .rocc_cmd_rocc_cmd_opcode(opcode), .rocc_cmd_rocc_cmd_funct(funct),
    .rocc_cmd_rocc_cmd_rd(5'd1), .rocc_cmd_rocc_cmd_xd(1'b1),
    .rocc_cmd_rocc_cmd_rs1data(rs1data), .rocc_cmd_rocc_cmd_rs2data(rs2data),
    .rocc_resp_rocc_resp_to_bus_enable(resp_enable),
    .rocc_resp_rocc_resp_to_bus_ready(1'b1),
    .rocc_resp_rocc_resp_to_bus_result_rddata(resp_data),
    .dma_cpu_to_isax_ch0_cpu_addr(l_cpu), .dma_cpu_to_isax_ch0_isax_addr(l_isax),
    .dma_cpu_to_isax_ch0_length(l_len), .dma_cpu_to_isax_ch0_stride_x(l_sx),
    .dma_cpu_to_isax_ch0_stride_y(l_sy), .dma_cpu_to_isax_ch0_enable(l_en),
    .dma_cpu_to_isax_ch0_ready(idle),
    .dma_isax_to_cpu_ch0_cpu_addr(s_cpu), .dma_isax_to_cpu_ch0_isax_addr(s_isax),
    .dma_isax_to_cpu_ch0_length(s_len), .dma_isax_to_cpu_ch0_stride_x(s_sx),
    .dma_isax_to_cpu_ch0_stride_y(s_sy), .dma_isax_to_cpu_ch0_enable(s_en),
    .dma_isax_to_cpu_ch0_ready(idle),
    .dma_poll_for_idle_ch0_ready(idle), .dma_poll_for_idle_ch0_res0(1'b1),
    .burst_write_enable(burst_write_enable), .burst_write_addr(burst_addr),
    .burst_write_data(burst_write_data),
    .burst_read_0_enable(burst_read_0_enable), .burst_read_0_addr(burst_addr),
    .burst_read_1_enable(1'b1), .burst_read_1_res0(burst_res0));

  // DMA model: 0 idle, 1 latency, 2 load beat, 3/4/5 store read+capture, 6 done.
  reg store;
  reg [31:0] cpu, isax;
  reg [8:0] remaining;
  reg [7:0] sx;
  reg [15:0] sxy, tileoff;
  integer lat;
  task automatic advance;
    begin
      if (sx != 0 && tileoff + 8 >= sx) begin
        isax <= isax + sxy - tileoff;
        tileoff <= 0;
      end else begin
        isax <= isax + 8;
        tileoff <= tileoff + 8;
      end
      cpu <= cpu + 8;
      remaining <= remaining - 8;
    end
  endtask
  always @(posedge clk) begin
    burst_write_enable <= 0;
    burst_read_0_enable <= 0;
    if (rst) dstate <= 0;
    else case (dstate)
      0: if (l_en || s_en) begin
        if (l_en && s_en) $fatal(1, "both DMA directions enabled");
        store <= s_en;
        cpu <= s_en ? s_cpu : l_cpu;
        isax <= s_en ? s_isax : l_isax;
        remaining <= 9'd1 << (s_en ? s_len : l_len);
        sx <= s_en ? s_sx : l_sx;
        sxy <= s_en ? s_sx * s_sy : l_sx * l_sy;
        tileoff <= 0;
        lat <= 4;
        dstate <= 1;
      end
      1: if (lat == 0) dstate <= store ? 3 : 2; else lat <= lat - 1;
      2: begin
        burst_write_enable <= 1;
        burst_addr <= isax;
        burst_write_data <= dram[cpu >> 3];
        advance();
        if (remaining == 8) dstate <= 6;
      end
      3: begin burst_read_0_enable <= 1; burst_addr <= isax; dstate <= 4; end
      4: dstate <= 5;
      5: begin
        dram[cpu >> 3] <= burst_res0;
        advance();
        dstate <= remaining == 8 ? 6 : 3;
      end
      6: dstate <= 0;
      default: dstate <= 0;
    endcase
  end

  integer errors = 0, cycles, n;
  task automatic rocc(input [6:0] op, input [6:0] f, input [31:0] a, input [31:0] b,
                      input [8*8-1:0] name);
    begin
      while (!cmd_ready) @(negedge clk);
      cmd_enable = 1; opcode = op; funct = f; rs1data = a; rs2data = b;
      @(negedge clk);
      cmd_enable = 0; rs1data = 32'hdead_beef; rs2data = 32'hdead_beef;
      cycles = 0;
      while (!resp_enable && cycles < 100000) begin @(negedge clk); cycles = cycles + 1; end
      if (!resp_enable || resp_data !== 0) begin
        $display("DMA FAIL: %0s got response %h", name, resp_data);
        errors = errors + 1;
      end
      $display("cycles %0s %0d", name, cycles);
      @(negedge clk);
    end
  endtask
  task automatic reset_output;
    for (n = @D_BUF@ / 8; n < (@D_BUF@ + @D_BYTES@) / 8; n = n + 1) dram[n] = 64'hdeadbeef_deadbeef;
  endtask
  initial begin
    $readmemh("dram.hex", dram);
    repeat (3) @(negedge clk);
    rst = 0;
@STEPS@
    if (errors == 0) $display("DMA BENCH DONE");
    $finish;
  end
endmodule
"""


def _bench(work: Path, commands) -> None:
    steps, dump = [], 0
    for name, op, funct, rs1, rs2 in commands:
        if name == "store_c":
            steps.append("    reset_output();")
        steps.append(f'    rocc(7\'h{op:02x}, 7\'h{funct:02x}, 32\'h{rs1:08x}, 32\'h{rs2:08x}, "{name}");')
        if name == "store_c":
            steps.append(f'    $writememh("out{dump}.hex", dram, {D_BUF // 8}, {(D_BUF + D_WORDS * 4) // 8 - 1});')
            dump += 1
    text = (TB.replace("@STEPS@", "\n".join(steps)).replace("@DRAM_WORDS@", str(DRAM_WORDS))
            .replace("@D_BUF@", str(D_BUF)).replace("@D_BYTES@", str(D_WORDS * 4)))
    (work / "tb.sv").write_text(text)


def _run(work: Path, main_sv: str, dram, commands) -> tuple[str, list[np.ndarray]]:
    work.mkdir()
    (work / "main.sv").write_text(main_sv)
    words64 = dram.view(np.uint64)
    (work / "dram.hex").write_text("\n".join(f"{int(w):016x}" for w in words64) + "\n")
    _bench(work, commands)
    rtl = sorted(str(p) for p in (MANUAL / "rtl").glob("*.v"))
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    subprocess.run(["verilator", "--binary", "-j", "4", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv", "main.sv",
                    *rtl], cwd=work, env=env, check=True, capture_output=True)
    out = subprocess.run([str(work / "obj/Vtb")], cwd=work, env=env, check=True,
                         capture_output=True, text=True).stdout
    dumps = []
    for path in sorted(work.glob("out*.hex"), key=lambda p: int(p.stem[3:])):
        words = [int(w, 16) for w in path.read_text().split() if not w.startswith("@")]
        dumps.append(np.array(words, np.uint64).view(np.uint32))
    return out, dumps


def test_dma_encoding():
    sv = generate_main(parse_proc(GEMM), instance())
    assert sv.count("dma_length <= 4'd7;") == 6  # two 128-byte bursts in each of 3 instructions
    assert sv.count("dma_stride_x <= 8'd32;") == 6 and sv.count("dma_stride_y <= 8'd8;") == 6
    assert "dma_isax_addr <= 32'(32'h80000 + " in sv  # matBT
    assert "dma_isax_addr <= 32'(32'h100000 + " in sv  # matCD
    assert "assign dma_cpu_to_isax_ch0_enable = !rst && (" in sv
    assert "assign dma_isax_to_cpu_ch1_enable = '0;" in sv


def test_dma_rejects_non_power_of_two_burst():
    proc = parse_proc(GEMM.replace("_mem[dram_addr +: 32]", "_mem[dram_addr +: 24]", 1))
    with pytest.raises(ValueError, match="power of two"):
        generate_main(proc, instance())


def test_dma_rejects_unaligned_stride():
    proc = parse_proc(GEMM.replace("[[stride_x(32)]]", "[[stride_x(12)]]", 1))
    with pytest.raises(ValueError, match="stride_x must be a multiple of 8"):
        generate_main(proc, instance())


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_gemm_isax_end_to_end(tmp_path):
    dram, commands, expected = _scenario(np.random.default_rng(11))
    results = {}
    for name, main_sv in (("generated", generate_main(parse_proc(GEMM), instance())),
                          ("golden", (MANUAL / "main.sv").read_text())):
        out, dumps = _run(tmp_path / name, main_sv, dram, commands)
        assert "DMA BENCH DONE" in out, out
        assert len(dumps) == len(expected)
        for i, (got, want) in enumerate(zip(dumps, expected)):
            bad = np.nonzero(got != want)[0]
            assert not len(bad), (f"{name} store {i}: {len(bad)} words differ, first at word "
                                  f"{bad[0]}: got {got[bad[0]]:08x} want {want[bad[0]]:08x}")
        results[name] = re.findall(r"cycles (\w+) (\d+)", out)
    print(results)
