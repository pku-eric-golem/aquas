"""Route 1 acceptance (cadl-allo-integration.md 5.D.1): vector inputs and
outputs, a scalar reduction, fixed 2-D arrays, array and mixed returns, each
kernel invoked more than once and none relying on GEMM parameter names.

Each example in examples/allo/accept is synthesized, wrapped and simulated:
DMA loads, two invokes on different tiles, DMA stores. The expected memory and
rd values come from Allo's own LLVM build of the same embedded code (NumPy
for relu_count, whose mixed return that build does not support). fp_ops pushes
FP32 special values through the HardFloat blackboxes (5.D.2).
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cadl_frontend.allo_kernel.interface import HlsPort, KernelParam, classify
from cadl_frontend.allo_sv_gen.main import generate_main
from cadl_frontend.parser import parse_proc
from test_dma import TB

ROOT = Path(__file__).resolve().parents[2]
ACCEPT = ROOT / "examples/allo/accept"
DEAD = 0xDEADBEEF
DRAM_WORDS = 0x6000 // 8          # 64-bit words
IN_DATA = 0x1000                  # inputs, packed one after another
OUT_BUF = 0x4000                  # outputs, with a 64-byte guard each side
OUT_DATA = OUT_BUF + 64
LOAD, STORE, COMPUTE = 0x7B, 0x7B, 0x0B

# The bench checks that every instruction answers; rd values are compared here.
RESPONSE_CHECK = """      if (!resp_enable || resp_data !== 0) begin
        $display("DMA FAIL: %0s got response %h", name, resp_data);
        errors = errors + 1;
      end
"""
RESPONSE_LOG = """      if (!resp_enable) begin
        $display("DMA FAIL: %0s got no response", name);
        errors = errors + 1;
      end
      $display("resp %0s %h", name, resp_data);
"""


def _u32(value) -> int:
    return int(np.int64(value)) & 0xFFFFFFFF


class Scenario:
    """Loads, then invokes with their expected rd, then stores with expected data."""

    def __init__(self):
        self.loads, self.invokes, self.stores = [], [], []
        self.next_in, self.next_out = IN_DATA, OUT_DATA

    def load(self, name, funct, spm_offset, data):
        self.loads.append((name, funct, self.next_in, spm_offset, np.asarray(data, np.int32)))
        self.next_in += data.size * 4

    def invoke(self, name, rs1, rs2, rd=0):
        self.invokes.append((name, rs1, rs2, _u32(rd)))

    def store(self, name, funct, spm_offset, want):
        self.stores.append((name, funct, self.next_out, spm_offset, np.asarray(want, np.int32)))
        self.next_out += want.size * 4

    def commands(self):
        for name, funct, dram, spm, _ in self.loads:
            yield name, LOAD, funct, dram, spm, 0
        for name, rs1, rs2, rd in self.invokes:
            yield name, COMPUTE, 0, rs1, rs2, rd
        for name, funct, dram, spm, _ in self.stores:
            yield name, STORE, funct, dram, spm, 0

    def dram(self):
        image = np.full(DRAM_WORDS * 2, DEAD, np.uint32)
        for *_, dram, _, data in self.loads:
            image[dram // 4:dram // 4 + data.size] = data.view(np.uint32)
        return image

    def out_words(self):
        return (self.next_out + 64 - OUT_BUF) // 4

    def expected_out(self):
        want = np.full(self.out_words(), DEAD, np.uint32)
        for *_, dram, _, data in self.stores:
            start = (dram - OUT_BUF) // 4
            want[start:start + data.size] = data.view(np.uint32)
        return want


def _model(name, tmp_path):
    """Allo's LLVM build of the embedded kernel."""
    from cadl_frontend.allo import check_source
    from cadl_frontend.allo_kernel.schedule import load_schedule
    cadl = ACCEPT / f"{name}.cadl"
    kernel = check_source(cadl.read_text(), str(cadl)).allo_kernels[name]
    (tmp_path / "model").mkdir()
    return load_schedule(kernel, tmp_path / "model", cadl).build()


def _ints(rng, *shape):
    return rng.integers(-1000, 1000, shape, dtype=np.int32)


def vec_add(model, rng):
    s = Scenario()
    for a, b, c in ((3, 0, 5), (15, 7, 15)):
        x, y = _ints(rng, 16), _ints(rng, 16)
        s.load("load_a", 0, a * 16, x)
        s.load("load_b", 1, b * 16, y)
        out = np.zeros(16, np.int32)
        model(x, y, out)
        s.invoke("vadd", (a << 16) | b, c)
        s.store("store_c", 2, c * 16, out)
    return s


def mat_scale(model, rng):
    s = Scenario()
    for a, b, k in ((1, 0, 3), (7, 7, -2)):
        x = _ints(rng, 4, 8)
        s.load("load_a", 0, a * 32, x.reshape(-1))
        out = np.zeros((4, 8), np.int32)
        model(x, k, out)
        s.invoke("mscale", (a << 16) | b, _u32(k))
        s.store("store_b", 2, b * 32, out.reshape(-1))
    return s


def reduce_sum(model, rng):
    s = Scenario()
    for tile in (2, 15):
        x = _ints(rng, 16)
        s.load("load_x", 0, tile * 16, x)
        s.invoke("rsum", tile, 0, model(x))
    return s


def prefix_sum(model, rng):
    s = Scenario()
    for x_tile, y_tile in ((4, 9), (15, 0)):
        x = _ints(rng, 16)
        s.load("load_x", 0, x_tile * 16, x)
        s.invoke("psum", x_tile, y_tile)
        s.store("store_y", 2, y_tile * 16, model(x))
    return s


def relu_count(model, rng):
    # Allo's LLVM backend returns several values only if all are tensors, so
    # the expected values of this mixed return are computed with NumPy.
    s = Scenario()
    for x_tile, y_tile in ((1, 15), (15, 3)):
        x = _ints(rng, 16)
        s.load("load_x", 0, x_tile * 16, x)
        y, positive = np.maximum(x, 0), int(np.count_nonzero(x > 0))
        assert 0 < positive < 16  # both paths of the kernel's `if` are taken
        s.invoke("relu", x_tile, y_tile, positive)
        s.store("store_y", 2, y_tile * 16, y)
    return s


SPECIAL = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000,
           0x7FC00001, 0x7F800001, 0xFFC00000, 0x00000001, 0x807FFFFF, 0x00800000,
           0x7F7FFFFF, 0xFF7FFFFF, 0x3F000000, 0x3FC00000, 0x00C00000, 0x7F000000,
           0x33800000, 0x4B800000]
# inf - inf, 0 * inf, signed zeros, NaN kinds, subnormals, underflow, overflow, ties to even.
KEY_PAIRS = [(0x7F800000, 0xFF800000), (0x00000000, 0x7F800000), (0x80000000, 0x80000000),
             (0x00000000, 0x80000000), (0x7F800001, 0x3F800000), (0x3F800000, 0x7FC00001),
             (0xFFC00000, 0xFF800000), (0x7F7FFFFF, 0x7F7FFFFF), (0x00000001, 0x3F000000),
             (0x00000001, 0xBF800000), (0x807FFFFF, 0x00800000), (0x00C00000, 0x3F000000),
             (0x3FC00000, 0x7F000000), (0xFF7FFFFF, 0xFF7FFFFF), (0x33800000, 0x3F800000),
             (0x4B800000, 0x3F800000)]


def _ieee(op, a: int, b: int) -> int:
    """binary32 a+b or a*b, round to nearest even, NaNs as RISC-V's canonical NaN."""
    x, y = (np.array([v], np.uint32).view(np.float32)[0] for v in (a, b))
    with np.errstate(all="ignore"):
        r = np.float32(x + y) if op == "add" else np.float32(x * y)
    return 0x7FC00000 if np.isnan(r) else int(np.array([r], np.float32).view(np.uint32)[0])


def fp_ops(model, rng):
    # The HardFloat blackboxes, not the host FPU behind Allo's LLVM model, set the
    # NaN encoding, so the expected values come from _ieee.
    pairs = KEY_PAIRS + [(SPECIAL[i], SPECIAL[j]) for i, j in rng.integers(0, len(SPECIAL), (48, 2))]
    s = Scenario()
    for call, (a, b, sum_, prod) in enumerate(((0, 1, 2, 3), (5, 6, 7, 8), (10, 11, 12, 9), (15, 14, 13, 4))):
        chunk = pairs[call * 16:call * 16 + 16]
        bits = lambda vs: np.array(vs, np.uint32).view(np.int32)  # noqa: E731
        s.load("load_a", 0, a * 16, bits([x for x, _ in chunk]))
        s.load("load_b", 1, b * 16, bits([y for _, y in chunk]))
        s.invoke("fops", (a << 16) | b, (sum_ << 16) | prod)
        s.store("store_s", 2, sum_ * 16, bits([_ieee("add", x, y) for x, y in chunk]))
        s.store("store_p", 3, prod * 16, bits([_ieee("mul", x, y) for x, y in chunk]))
    return s


SCENARIOS = [vec_add, mat_scale, reduce_sum, prefix_sum, relu_count, fp_ops]


def _simulate(work: Path, out: Path, s: Scenario) -> tuple[str, np.ndarray]:
    work.mkdir()
    (work / "dram.hex").write_text(
        "\n".join(f"{int(w):016x}" for w in s.dram().view(np.uint64)) + "\n")
    steps = ["    reset_output();"]
    for name, op, funct, rs1, rs2, _ in s.commands():
        steps.append(f'    rocc(7\'h{op:02x}, 7\'h{funct:02x}, 32\'h{rs1:08x}, 32\'h{rs2:08x}, "{name}");')
    out_bytes = s.out_words() * 4
    steps.append(f'    $writememh("out.hex", dram, {OUT_BUF // 8}, {(OUT_BUF + out_bytes) // 8 - 1});')
    assert TB.count(RESPONSE_CHECK) == 1
    (work / "tb.sv").write_text(
        TB.replace(RESPONSE_CHECK, RESPONSE_LOG).replace("@STEPS@", "\n".join(steps))
        .replace("@DRAM_WORDS@", str(DRAM_WORDS)).replace("@D_BUF@", str(OUT_BUF))
        .replace("@D_BYTES@", str(out_bytes)))
    rtl = sorted(str(p) for p in (out / "rtl").glob("*.v"))
    subprocess.run(["verilator", "--binary", "-j", "4", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--assert", "--top-module", "tb", "-Mdir", "obj", "tb.sv",
                    str(out / "main.sv"), *rtl], cwd=work, check=True, capture_output=True)
    log = subprocess.run([str(work / "obj/Vtb")], cwd=work, check=True,
                         capture_output=True, text=True, timeout=600).stdout
    words = [int(w, 16) for w in (work / "out.hex").read_text().split() if not w.startswith("@")]
    return log, np.array(words, np.uint64).view(np.uint32)


@pytest.mark.skipif(shutil.which("vitis_hls") is None, reason="vitis_hls not on PATH")
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
@pytest.mark.skipif("LLVM_BUILD_DIR" not in os.environ,
                    reason="LLVM_BUILD_DIR not set (Allo's LLVM model gives the expected values)")
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda f: f.__name__)
def test_accept_end_to_end(scenario, tmp_path):
    pytest.importorskip("allo")
    from cadl_frontend.allo_sv_gen.build import build
    name = scenario.__name__
    result = build(ACCEPT / f"{name}.cadl", tmp_path / "build")
    s = scenario(_model(name, tmp_path), np.random.default_rng(7))
    log, got = _simulate(tmp_path / "sim", result.output, s)
    assert "DMA BENCH DONE" in log, log
    responses = re.findall(r"^resp (\w+) ([0-9a-f]{8})$", log, re.M)
    assert [(n, int(v, 16)) for n, v in responses] == [(c[0], c[5]) for c in s.commands()]
    want = s.expected_out()
    bad = np.nonzero(got != want)[0]
    assert not len(bad), (f"{len(bad)} words differ, first at word {bad[0]}: "
                          f"got {got[bad[0]]:08x} want {want[bad[0]]:08x}")


# Manifest checks on invoke, against a hand-written reduce_sum port list.

REDUCE = (ACCEPT / "reduce_sum.cadl").read_text()


def _reduce_instance():
    ports = [HlsPort("ap_clk", "input", 1), HlsPort("ap_rst", "input", 1),
             HlsPort("ap_start", "input", 1), HlsPort("ap_done", "output", 1),
             HlsPort("ap_idle", "output", 1), HlsPort("ap_ready", "output", 1),
             HlsPort("X_address0", "output", 4), HlsPort("X_ce0", "output", 1),
             HlsPort("X_q0", "input", 32), HlsPort("v1", "output", 32),
             HlsPort("v1_ap_vld", "output", 1)]
    return classify("reduce_sum", ports, [KernelParam("X", "ap_memory", "i32", 32, (16,))],
                    [KernelParam("v1", "ap_vld", "i32", 32)])


def test_scalar_result_capture():
    sv = generate_main(parse_proc(REDUCE), _reduce_instance())
    assert "if (!rst && hls_v1_ap_vld) hls_v1_value <= hls_v1;" in sv
    assert "rsum__total <= $signed(32'(hls_v1_ap_vld ? hls_v1 : hls_v1_value));" in sv
    assert "assert(hls_v1_seen || hls_v1_ap_vld);" in sv


@pytest.mark.parametrize("old, new, error", [
    ("red_x[tile * 16 +: 16]", "red_x[tile * 16 +: 8]", "binds 8 elements of red_x, the kernel's X has shape"),
    ("static red_x: [i32; 256];", "static red_x: [f32; 256];", "is i32, red_x holds 32-bit floats"),
    ("-> total: i32;\n    _irf[rd] = total;", "-> total: i16;\n    _irf[rd] = total;", "total is 16 bits"),
    ("-> total: i32;\n    _irf[rd] = total;", "-> red_x[0 +: 16];\n    _irf[rd] = 0;", "is a scalar"),
    ("-> total: i32;\n    _irf[rd] = total;", ";\n    _irf[rd] = 0;", "binds 0 results"),
])
def test_invoke_checked_against_manifest(old, new, error):
    assert REDUCE.count(old) == 1
    with pytest.raises(ValueError, match=re.escape(error)):
        generate_main(parse_proc(REDUCE.replace(old, new)), _reduce_instance())
