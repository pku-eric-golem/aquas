"""Robustness of the GEMM ISAX (cadl-allo-integration.md 5.D.3): command and
response backpressure, random DMA latency, and reset while an instruction runs.

The bench is test_dma's, changed so that the response `ready` is random (rd and
data must hold while it is low), a bogus command is offered the whole time an
instruction runs (it must not be taken), the DMA latency is random, and reset
hits the first DMA load, GEMM and DMA store partway, each issued again at once.
"""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cadl_frontend.allo_sv_gen.main import generate_main
from cadl_frontend.parser import parse_proc
from test_dma import D_BUF, D_WORDS, DRAM_WORDS, GEMM, MANUAL, TB, _scenario, instance

# (old, new, count) edits of the test_dma bench.
EDITS = [
    ("    .rocc_resp_rocc_resp_to_bus_ready(1'b1),\n",
     "    .rocc_resp_rocc_resp_to_bus_ready(resp_ready),\n"
     "    .rocc_resp_rocc_resp_to_bus_result_rd(resp_rd),\n", 1),
    ("  wire [31:0] resp_data;\n",
     "  wire [31:0] resp_data;\n"
     "  wire [4:0] resp_rd;\n"
     "  reg [15:0] lfsr = 16'h1d0f;\n"
     "  always @(posedge clk) lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};\n"
     "  wire resp_ready = lfsr[1] & lfsr[4];\n"
     "  reg held = 0;\n"
     "  reg [31:0] held_data;\n"
     "  reg [4:0] held_rd;\n", 1),
    ("  integer errors = 0, cycles, n;\n",
     "  integer errors = 0, cycles, n;\n"
     "  integer fires = 0, issued = 0;\n"
     "  always @(posedge clk) begin\n"
     "    if (!rst && cmd_enable && cmd_ready) fires = fires + 1;\n"
     "    if (held && (!resp_enable || resp_data !== held_data || resp_rd !== held_rd)) begin\n"
     "      $display(\"DMA FAIL: response changed while held\");\n"
     "      errors = errors + 1;\n"
     "    end\n"
     "    held <= !rst && resp_enable && !resp_ready;\n"
     "    held_data <= resp_data;\n"
     "    held_rd <= resp_rd;\n"
     "  end\n", 1),
    ("        lat <= 4;\n", "        lat <= lfsr[2:0];\n", 1),
    # Keep offering a bogus command while the instruction runs, then wait for the handshake.
    ("      while (!resp_enable && cycles < 100000) begin @(negedge clk); cycles = cycles + 1; end\n",
     "      issued = issued + 1;\n"
     "      cmd_enable = 1; opcode = 7'h0b; funct = 7'h55;\n"
     "      while (!(resp_enable && resp_ready) && cycles < 100000) begin\n"
     "        @(negedge clk); cycles = cycles + 1;\n"
     "      end\n"
     "      cmd_enable = 0;\n", 1),
    ("  task automatic reset_output;\n",
     "  task automatic interrupt(input [6:0] op, input [6:0] f, input [31:0] a, input [31:0] b,\n"
     "                           input integer after);\n"
     "    begin\n"
     "      while (!cmd_ready) @(negedge clk);\n"
     "      cmd_enable = 1; opcode = op; funct = f; rs1data = a; rs2data = b;\n"
     "      @(negedge clk);\n"
     "      cmd_enable = 0; issued = issued + 1;\n"
     "      repeat (after) @(negedge clk);\n"
     "      if (resp_enable) begin $display(\"DMA FAIL: done before the reset\"); errors = errors + 1; end\n"
     "      rst = 1; repeat (3) @(negedge clk); rst = 0;\n"
     "      $display(\"reset after %0d cycles\", after);\n"
     "    end\n"
     "  endtask\n"
     "  task automatic reset_output;\n", 1),
    ("    if (errors == 0) $display(\"DMA BENCH DONE\");\n",
     "    if (fires != issued) begin\n"
     "      $display(\"DMA FAIL: %0d commands taken, %0d issued\", fires, issued);\n"
     "      errors = errors + 1;\n"
     "    end\n"
     "    if (errors == 0) $display(\"DMA BENCH DONE\");\n", 1),
]


def _bench(commands) -> str:
    text = TB
    for old, new, count in EDITS:
        assert text.count(old) == count, old
        text = text.replace(old, new)
    # The first load, GEMM and store are each run once and reset partway, then issued again
    # at once: a core still busy with the old job would answer the new one too early.
    interrupts = {"load_a": 20, "gemm": 150, "store_c": 30}
    steps, dump = [], 0
    for name, op, funct, rs1, rs2 in commands:
        if name in interrupts:
            after = interrupts.pop(name)
            steps.append(f"    interrupt(7'h{op:02x}, 7'h{funct:02x}, 32'h{rs1:08x}, 32'h{rs2:08x}, {after});")
        if name == "store_c":
            steps.append("    reset_output();")
        steps.append(f'    rocc(7\'h{op:02x}, 7\'h{funct:02x}, 32\'h{rs1:08x}, 32\'h{rs2:08x}, "{name}");')
        if name == "store_c":
            steps.append(f'    $writememh("out{dump}.hex", dram, {D_BUF // 8}, {(D_BUF + D_WORDS * 4) // 8 - 1});')
            dump += 1
    return (text.replace("@STEPS@", "\n".join(steps)).replace("@DRAM_WORDS@", str(DRAM_WORDS))
            .replace("@D_BUF@", str(D_BUF)).replace("@D_BYTES@", str(D_WORDS * 4)))


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_backpressure_and_reset(tmp_path):
    dram, commands, expected = _scenario(np.random.default_rng(11))
    (tmp_path / "main.sv").write_text(generate_main(parse_proc(GEMM), instance()))
    (tmp_path / "dram.hex").write_text("\n".join(f"{int(w):016x}" for w in dram.view(np.uint64)) + "\n")
    (tmp_path / "tb.sv").write_text(_bench(commands))
    rtl = sorted(str(p) for p in (MANUAL / "rtl").glob("*.v"))
    subprocess.run(["verilator", "--binary", "-j", "4", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--assert", "--top-module", "tb", "-Mdir", "obj", "tb.sv",
                    "main.sv", *rtl], cwd=tmp_path, check=True, capture_output=True)
    out = subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout
    assert "DMA BENCH DONE" in out, out
    assert len(re.findall(r"^reset after", out, re.M)) == 3
    dumps = sorted(Path(tmp_path).glob("out*.hex"), key=lambda p: int(p.stem[3:]))
    assert len(dumps) == len(expected)
    for i, (path, want) in enumerate(zip(dumps, expected)):
        words = [int(w, 16) for w in path.read_text().split() if not w.startswith("@")]
        got = np.array(words, np.uint64).view(np.uint32)
        bad = np.nonzero(got != want)[0]
        assert not len(bad), f"store {i}: {len(bad)} words differ, first at word {bad[0]}"
