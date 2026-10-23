"""HardFloat blackbox RTL against IEEE binary32 (cadl-allo-integration.md 5.D.2):
special values, clock-enable stalls and reset.

hf_add_f32_rtl and hf_mul_f32_rtl (blackbox.py) are simulated cycle by cycle.
The model is a 3-stage pipeline that advances only on clock edges with ap_ce
high and ap_rst low, and whose output is 0 during reset; results are NumPy
float32 (round to nearest even, subnormals kept) with NaNs made canonical, as
the RISC-V specialization of HardFloat returns them.
"""
import shutil
import subprocess

import numpy as np
import pytest

from cadl_frontend.allo_kernel.blackbox import LATENCY, build_blackboxes

SPECIAL = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000,
           0x7FC00001, 0x7F800001, 0xFFC00000,  # quiet, signaling and negative NaN
           0x00000001, 0x807FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF,  # subnormal, normal edges
           0x3F000000, 0x3FC00000, 0x00C00000, 0x7F000000, 0x33800000, 0x4B800000]
CANONICAL_NAN = 0x7FC00000

TB = """module tb;
  reg clk = 0;
  always #1 clk = ~clk;
  reg rst = 1, ce = 0;
  reg [31:0] a = 0, b = 0;
  wire [31:0] sum, product;
  hf_add_f32_rtl add(.ap_clk(clk), .ap_rst(rst), .ap_ce(ce), .a(a), .b(b), .result(sum));
  hf_mul_f32_rtl mul(.ap_clk(clk), .ap_rst(rst), .ap_ce(ce), .a(a), .b(b), .result(product));
  reg [65:0] stimulus [0:@N@-1];  // {rst, ce, a, b}
  integer i, out;
  initial begin
    $readmemh("stimulus.hex", stimulus);
    out = $fopen("out.txt", "w");
    for (i = 0; i < @N@; i = i + 1) begin
      @(negedge clk);
      if (i > 0) $fwrite(out, "%h %h\\n", sum, product);
      {rst, ce, a, b} = stimulus[i];
    end
    @(negedge clk);
    $fwrite(out, "%h %h\\n", sum, product);
    $fclose(out);
    $finish;
  end
endmodule
"""


def _ieee(op, a: int, b: int) -> int:
    x, y = (np.array([v], np.uint32).view(np.float32)[0] for v in (a, b))
    with np.errstate(all="ignore"):
        r = np.float32(x + y) if op == "add" else np.float32(x * y)
    return CANONICAL_NAN if np.isnan(r) else int(np.array([r], np.float32).view(np.uint32)[0])


def _stimulus(rng):
    pairs = [(a, b) for a in SPECIAL for b in SPECIAL]
    pairs += [tuple(int(v) for v in rng.integers(0, 1 << 32, 2, dtype=np.uint64)) for _ in range(1500)]
    vectors = [(1, 0, 0, 0)] * 3  # reset first
    for i, (a, b) in enumerate(pairs):
        if i in (400, 1100):
            vectors += [(1, int(rng.integers(0, 2)), a, b)] * 3  # reset mid-stream, CE either way
        ce = int(rng.random() < 0.7)
        vectors.append((0, ce, a, b))
        while not ce:  # hold the operands until they are taken
            ce = int(rng.random() < 0.7)
            vectors.append((0, ce, a, b))
    return vectors


def _expected(vectors):
    """Per vector, the (sum, product) after its clock edge; None while unknown."""
    stages, want = [None] * LATENCY, []
    for rst, ce, a, b in vectors:
        if ce and not rst:
            stages = [(a, b)] + stages[:-1]
        if rst:
            want.append((0, 0))
        elif stages[-1] is None:
            want.append(None)
        else:
            want.append((_ieee("add", *stages[-1]), _ieee("mul", *stages[-1])))
    return want


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_blackboxes_special_values_stalls_reset(tmp_path):
    rtl = tmp_path / "rtl"
    build_blackboxes(rtl, ["hf_add_f32", "hf_mul_f32"])
    vectors = _stimulus(np.random.default_rng(3))
    (tmp_path / "stimulus.hex").write_text(
        "\n".join(f"{(r << 65) | (c << 64) | (a << 32) | b:017x}" for r, c, a, b in vectors) + "\n")
    (tmp_path / "tb.sv").write_text(TB.replace("@N@", str(len(vectors))))
    sources = [str(rtl / f) for f in ("HardFloatNative.v", "hf_add_f32.v", "hf_mul_f32.v")]
    subprocess.run(["verilator", "--binary", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "--top-module", "tb", "-Mdir", "obj", "tb.sv", *sources],
                   cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True, capture_output=True)
    got = [tuple(int(w, 16) for w in line.split())
           for line in (tmp_path / "out.txt").read_text().splitlines()]
    want = _expected(vectors)
    assert len(got) == len(want)
    checked = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if w is not None]
    bad = [(i, f"{g[0]:08x} {g[1]:08x}", f"{w[0]:08x} {w[1]:08x}", vectors[i]) for i, g, w in checked if g != w]
    assert not bad, f"{len(bad)} of {len(checked)} cycles differ, first: {bad[0]}"
    stalls = sum(1 for r, c, *_ in vectors if not c and not r)
    assert len(checked) > len(SPECIAL) ** 2 + 1500 and stalls > 500  # every pair is seen
