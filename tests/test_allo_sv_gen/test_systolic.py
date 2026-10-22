"""The Allo library systolic array (examples/allo/systolic.cadl) as an ISAX.

The PE grid comes from allo.library.systolic; its schedule partitions only the
local tile buffers, so the HLS top keeps plain A/B/C arrays that the existing
SPM bridge serves. The end-to-end bench reuses the DMA model of test_dma and
runs the Vitis RTL with no pipeline-stall repair.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("allo")

from cadl_frontend.allo_sv_gen.build import build  # noqa: E402
from test_dma import TB  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SYSTOLIC = ROOT / "examples/allo/systolic.cadl"
DEAD = 0xDEADBEEF
DRAM_WORDS = 0x6000 // 8          # 64-bit words
A_DATA, B_DATA = 0x1000, 0x2000   # A: 8x4 (pitch 16 B), B: 4x8 (pitch 32 B)
C_BUF = 0x4000                    # C: 8x8 (pitch 32 B) with a 256-byte guard each side
C_DATA = C_BUF + 256
C_WORDS = 64 + 64 + 64


def reference_tile(a, b):
    """Sequential FP32 order of each PE: no fusion, k ascending."""
    c = np.zeros((4, 4), np.float32)
    for i in range(4):
        for j in range(4):
            v = np.float32(0)
            for k in range(4):
                v = np.float32(v + np.float32(a[i, k] * b[k, j]))
            c[i, j] = v
    return c


def _scenario(rng):
    """C[8x8] = A[8x4] * B[4x8] as four 4x4 output tiles."""
    a = rng.standard_normal((8, 4)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    dram = np.full(DRAM_WORDS * 2, DEAD, np.uint32)
    dram[A_DATA // 4:A_DATA // 4 + 32] = a.reshape(-1).view(np.uint32)
    dram[B_DATA // 4:B_DATA // 4 + 32] = b.reshape(-1).view(np.uint32)
    # (name, opcode, funct, rs1, rs2); SPM tiles are chosen away from 0 and at the top.
    tiles_a, tiles_b, tiles_c = (5, 63), (1, 62), ((0, 7), (40, 63))
    commands = []
    for t, row in zip(tiles_a, (0, 4)):
        commands.append(("load_a", 0x7B, 0, A_DATA + row * 16, ((t * 16) << 16) | 16))
    for t, col in zip(tiles_b, (0, 4)):
        commands.append(("load_b", 0x7B, 1, B_DATA + col * 4, ((t * 16) << 16) | 32))
    want = np.full(C_WORDS, DEAD, np.uint32)
    for bi, ta in enumerate(tiles_a):
        for bj, tb in enumerate(tiles_b):
            tc = tiles_c[bi][bj]
            commands.append(("systolic", 0x0B, 0, (ta << 16) | tb, tc))
            commands.append(("store_c", 0x7B, 2, C_DATA + bi * 4 * 32 + bj * 16,
                             ((tc * 16) << 16) | 32))
            tile = reference_tile(a[bi * 4:bi * 4 + 4], b[:, bj * 4:bj * 4 + 4])
            for r in range(4):
                start = 64 + (bi * 4 + r) * 8 + bj * 4
                want[start:start + 4] = tile[r].view(np.uint32)
    return dram, commands, want


def _run(work: Path, out: Path, dram, commands) -> tuple[str, np.ndarray]:
    work.mkdir()
    (work / "dram.hex").write_text("\n".join(f"{int(w):016x}" for w in dram.view(np.uint64)) + "\n")
    steps = ["    reset_output();"]
    steps += [f'    rocc(7\'h{op:02x}, 7\'h{funct:02x}, 32\'h{rs1:08x}, 32\'h{rs2:08x}, "{name}");'
              for name, op, funct, rs1, rs2 in commands]
    steps.append(f'    $writememh("out.hex", dram, {C_BUF // 8}, {(C_BUF + C_WORDS * 4) // 8 - 1});')
    (work / "tb.sv").write_text(
        TB.replace("@STEPS@", "\n".join(steps)).replace("@DRAM_WORDS@", str(DRAM_WORDS))
        .replace("@D_BUF@", str(C_BUF)).replace("@D_BYTES@", str(C_WORDS * 4)))
    rtl = sorted(str(p) for p in (out / "rtl").glob("*.v"))
    subprocess.run(["verilator", "--binary", "-j", "4", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                    "-Wno-PINMISSING", "--top-module", "tb", "-Mdir", "obj", "tb.sv",
                    str(out / "main.sv"), *rtl], cwd=work, check=True, capture_output=True)
    log = subprocess.run([str(work / "obj/Vtb")], cwd=work, check=True,
                         capture_output=True, text=True, timeout=600).stdout
    words = [int(w, 16) for w in (work / "out.hex").read_text().split() if not w.startswith("@")]
    return log, np.array(words, np.uint64).view(np.uint32)


def test_library_kernel_keeps_plain_array_top(tmp_path):
    result = build(SYSTOLIC, tmp_path / "hls_only", csynth=False)
    code = (result.hls.directory / "kernel.cpp").read_text()
    assert re.search(r"void systolic4x4_fp32\(\s*float A\[4\]\[4\],\s*float B\[4\]\[4\],"
                     r"\s*float C\[4\]\[4\]\s*\)", code)
    assert [(p.name, p.kind) for p in result.hls.params] == [
        ("A", "ap_memory"), ("B", "ap_memory"), ("C", "ap_memory")]
    assert result.hls.float_calls == {"hf_add_f32": 16, "hf_mul_f32": 16}  # one MAC per PE


@pytest.fixture(scope="module")
def full_build(tmp_path_factory):
    if shutil.which("vitis_hls") is None:
        pytest.skip("vitis_hls not on PATH")
    return build(SYSTOLIC, tmp_path_factory.mktemp("systolic") / "full")


def test_full_build_needs_no_stall_repair(full_build):
    # rtl_fixups would raise on a blackbox completion stall; only pins are dropped.
    fixups = json.loads((full_build.hls.directory / "rtl_fixups.json").read_text())
    assert {f["fix"] for f in fixups} <= {"drop dangling ap_ready pin"}
    assert all("_PE_kernel_" in f["file"] for f in fixups)
    ports = (full_build.output / "rtl/systolic4x4_fp32.v").read_text()
    assert "A_address1" in ports and "C_we1" in ports  # Vitis uses both ports of each array


def test_full_build_manifest(full_build):
    interface = json.loads((full_build.output / "manifest.json").read_text())["interface"]
    assert interface == json.loads((full_build.hls.directory / "kernel.json").read_text())
    params = {p["name"]: p for p in interface["params"]}
    assert {n: (p["direction"], p["shape"], p["ports"]) for n, p in params.items()} == {
        "A": ("in", [4, 4], 2), "B": ("in", [4, 4], 2), "C": ("out", [4, 4], 2)}
    latency = interface["latency"]
    assert isinstance(latency["best"], int) and latency["best"] <= latency["worst"]


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_systolic_isax_end_to_end(full_build, tmp_path):
    dram, commands, want = _scenario(np.random.default_rng(5))
    log, got = _run(tmp_path / "sim", full_build.output, dram, commands)
    assert "DMA BENCH DONE" in log, log
    bad = np.nonzero(got != want)[0]
    assert not len(bad), (f"{len(bad)} words differ, first at word {bad[0]}: "
                          f"got {got[bad[0]]:08x} want {want[bad[0]]:08x}")
    print(re.findall(r"cycles (\w+) (\d+)", log))
