"""tools/allo-gen: one command from CADL with an embedded Allo kernel to Chipyard inputs."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("allo")

from cadl_frontend.allo_sv_gen.build import build  # noqa: E402
from cadl_frontend.allo_sv_gen.hls import emit_project  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "examples/allo_manual"
GEMM = ROOT / "examples/allo/gemm.cadl"
sys.path.insert(0, str(MANUAL))
from schedule import build_schedule  # noqa: E402


def test_embedded_kernel_matches_manual_schedule(tmp_path):
    result = build(GEMM, tmp_path / "cadl", csynth=False)
    assert result.main_sv is None
    manual = emit_project(build_schedule(), tmp_path / "manual")
    assert (tmp_path / "cadl/hls/kernel.cpp").read_text() == (manual.directory / "kernel.cpp").read_text()
    assert result.hls.float_calls == manual.float_calls
    module = (tmp_path / "cadl/hls/gemm_fp32_kernel.py").read_text().splitlines()
    source = GEMM.read_text().splitlines()
    line = next(i for i, text in enumerate(source) if text.startswith("def gemm_fp32("))
    assert module[line] == source[line]  # tracebacks point at .cadl lines


def test_bundle_with_given_rtl(tmp_path):
    out = tmp_path / "out"
    result = build(GEMM, out, rtl=MANUAL / "rtl")
    assert "gemm8x8_fp32__tile_a" in result.main_sv.read_text()
    config = json.loads((out / "aps_config.yaml").read_text())
    assert config["vsrc"][0] == str(out / "main.sv")
    assert {Path(f).name for f in config["vsrc"][1:]} == {p.name for p in (MANUAL / "rtl").glob("*.v")}
    assert all(Path(f).parent == out / "rtl" for f in config["vsrc"][1:])
    encoding = json.loads((out / "encoding.json").read_text())
    assert encoding["gemm8x8_fp32"] == {"opcode": "0xB", "funct7": "0x0"}
    assert encoding["storemat_c_8tile"] == {"opcode": "0x7B", "funct7": "0x2"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert [s["dma_base"] for s in manifest["spm"]] == ["0x0", "0x80000", "0x100000"]
    assert manifest["rtl"]["provenance"] == "given"


def test_cli(tmp_path):
    out = tmp_path / "cli"
    run = subprocess.run([sys.executable, str(ROOT / "tools/allo-gen"), str(GEMM), "-o", str(out),
                          "--rtl", str(MANUAL / "rtl")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert (out / "main.sv").exists() and "ISAX wrapper:" in run.stdout
    bad = tmp_path / "bad.cadl"
    bad.write_text(GEMM.read_text().replace("s = allo.customize", "t = allo.customize")
                   .replace("s.", "t."))
    run = subprocess.run([sys.executable, str(ROOT / "tools/allo-gen"), str(bad), "-o",
                          str(tmp_path / "bad"), "--no-csynth"], capture_output=True, text=True)
    assert run.returncode == 1 and "Schedule name s is not declared" in run.stderr


@pytest.mark.skipif(shutil.which("vitis_hls") is None, reason="vitis_hls not on PATH")
def test_full_build_with_vitis(tmp_path):
    result = build(GEMM, tmp_path / "full")
    manifest = json.loads((tmp_path / "full/manifest.json").read_text())
    assert manifest["rtl"]["provenance"] == "vitis_hls + rtl_fixups"
    fixups = json.loads((result.hls.directory / "rtl_fixups.json").read_text())
    assert len(fixups) == 8
    assert "HLS 200-880" not in (result.hls.directory / "csynth.log").read_text()
