"""HardFloat blackbox RTL against IEEE binary32 (cadl-allo-integration.md 5.D.2):
special values, clock-enable stalls and reset, for every blackbox.OPERATORS entry.

All blackboxes run side by side on the same operands, cycle by cycle. Each is
modelled as a pipeline of its own latency that advances only on clock edges
with ap_ce high and ap_rst low, and whose output is 0 during reset. Results:
NumPy float32 arithmetic (round to nearest even, subnormals kept) with NaNs
made canonical, as the RISC-V specialization of HardFloat returns them;
comparisons with IEEE unordered semantics; conversions as RISC-V fcvt does
(integer to float rounds to nearest even, float to integer truncates and
saturates, NaN converting to the largest integer); neg and abs flip or clear
the sign bit only.
"""
import math
import shutil
import subprocess

import numpy as np
import pytest

from cadl_frontend.allo_kernel.blackbox import OPERATORS, build_blackboxes

SPECIAL = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000,
           0x7FC00001, 0x7F800001, 0xFFC00000,  # quiet, signaling and negative NaN
           0x00000001, 0x807FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF,  # subnormal, normal edges
           0x3F000000, 0x3FC00000, 0x00C00000, 0x7F000000, 0x33800000, 0x4B800000,
           0x4F000000, 0xCF000000, 0x4F800000, 0xBF000000]  # 2^31, -2^31, 2^32, -0.5
INTS = [0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x01000001, 0x00FFFFFF, 0xFEFFFFFF]
CANONICAL_NAN = 0x7FC00000
NAMES = sorted(OPERATORS)


def _f(bits: int) -> np.float32:
    return np.array([bits], np.uint32).view(np.float32)[0]


def _bits(value) -> int:
    value = np.float32(value)
    return CANONICAL_NAN if np.isnan(value) else int(np.array([value], np.float32).view(np.uint32)[0])


def _to_int(x: np.float32, signed: bool) -> int:
    lo, hi = (-2**31, 2**31 - 1) if signed else (0, 2**32 - 1)
    if np.isnan(x):
        return hi & 0xFFFFFFFF
    value = math.trunc(float(x)) if np.isfinite(x) else (hi if x > 0 else lo)
    return min(max(value, lo), hi) & 0xFFFFFFFF


def expected(name: str, a: int, b: int) -> int:
    x, y = _f(a), _f(b)
    with np.errstate(all="ignore"):
        if name.startswith("hf_cmp_"):
            pred, unordered = name[7:-4], bool(np.isnan(x) or np.isnan(y))
            lt, eq, gt = (not unordered and x < y), (not unordered and x == y), (not unordered and x > y)
            ordered = {"oeq": eq, "ogt": gt, "oge": gt or eq, "olt": lt, "ole": lt or eq,
                       "one": lt or gt, "ord": not unordered}
            if pred in ordered:
                return int(ordered[pred])
            return int(unordered or {"ueq": eq, "ugt": gt, "uge": gt or eq, "ult": lt, "ule": lt or eq,
                                     "une": not eq, "uno": False}[pred])
        return {
            "hf_add_f32": lambda: _bits(x + y), "hf_sub_f32": lambda: _bits(x - y),
            "hf_mul_f32": lambda: _bits(x * y),
            "hf_sitofp_f32": lambda: _bits(np.float32(float(a - (a >> 31 << 32)))),
            "hf_uitofp_f32": lambda: _bits(np.float32(float(a))),
            "hf_fptosi_f32": lambda: _to_int(x, True), "hf_fptoui_f32": lambda: _to_int(x, False),
            "hf_neg_f32": lambda: a ^ 0x80000000, "hf_abs_f32": lambda: a & 0x7FFFFFFF,
        }[name]()


def _testbench(n: int) -> str:
    wires, units, outs = [], [], []
    for i, name in enumerate(NAMES):
        op = OPERATORS[name]
        bits = op.result_bits
        wires.append(f"  wire [{bits - 1}:0] r{i};")
        inputs = ".a(a), .b(b)" if len(op.args) == 2 else ".a(a)"
        units.append(f"  {name}_rtl u{i}(.ap_clk(clk), .ap_rst(rst), .ap_ce(ce), {inputs}, .result(r{i}));")
        outs.append(f"r{i}")
    fmt = " ".join(["%h"] * len(outs))
    return f"""module tb;
  reg clk = 0;
  always #1 clk = ~clk;
  reg rst = 1, ce = 0;
  reg [31:0] a = 0, b = 0;
{chr(10).join(wires)}
{chr(10).join(units)}
  reg [65:0] stimulus [0:{n}-1];  // {{rst, ce, a, b}}
  integer i, out;
  initial begin
    $readmemh("stimulus.hex", stimulus);
    out = $fopen("out.txt", "w");
    for (i = 0; i < {n}; i = i + 1) begin
      @(negedge clk);
      if (i > 0) $fwrite(out, "{fmt}\\n", {", ".join(outs)});
      {{rst, ce, a, b}} = stimulus[i];
    end
    @(negedge clk);
    $fwrite(out, "{fmt}\\n", {", ".join(outs)});
    $fclose(out);
    $finish;
  end
endmodule
"""


def _stimulus(rng):
    values = SPECIAL + INTS
    pairs = [(a, b) for a in values for b in values]
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
    return vectors, len(pairs)


def _expected(vectors, name):
    """Per vector, `name`'s output after its clock edge; None while unknown."""
    stages, want = [None] * OPERATORS[name].latency, []
    for rst, ce, a, b in vectors:
        if ce and not rst:
            stages = [(a, b)] + stages[:-1]
        if rst:
            want.append(0)
        else:
            want.append(None if stages[-1] is None else expected(name, *stages[-1]))
    return want


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_blackboxes_special_values_stalls_reset(tmp_path):
    rtl = tmp_path / "rtl"
    build_blackboxes(rtl, NAMES)
    vectors, pairs = _stimulus(np.random.default_rng(3))
    (tmp_path / "stimulus.hex").write_text(
        "\n".join(f"{(r << 65) | (c << 64) | (a << 32) | b:017x}" for r, c, a, b in vectors) + "\n")
    (tmp_path / "tb.sv").write_text(_testbench(len(vectors)))
    sources = [str(rtl / "HardFloatNative.v")] + [str(rtl / f"{name}.v") for name in NAMES]
    subprocess.run(["verilator", "--binary", "-j", "8", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "--top-module", "tb", "-Mdir", "obj", "tb.sv", *sources],
                   cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([str(tmp_path / "obj/Vtb")], cwd=tmp_path, check=True, capture_output=True)
    got = [[int(w, 16) for w in line.split()] for line in (tmp_path / "out.txt").read_text().splitlines()]
    assert len(got) == len(vectors)
    for column, name in enumerate(NAMES):
        want = _expected(vectors, name)
        checked = [(i, row[column], w) for i, (row, w) in enumerate(zip(got, want)) if w is not None]
        bad = [(i, f"{g:08x}", f"{w:08x}", vectors[i]) for i, g, w in checked if g != w]
        assert not bad, f"{name}: {len(bad)} of {len(checked)} cycles differ, first: {bad[0]}"
        assert len(checked) > pairs  # every operand pair is seen
    assert sum(1 for r, c, *_ in vectors if not c and not r) > 500
