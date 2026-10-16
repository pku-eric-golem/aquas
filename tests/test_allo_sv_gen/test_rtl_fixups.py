"""rtl_fixups reproduces the validated allo_manual repair of raw Vitis output."""
from pathlib import Path

import pytest

from cadl_frontend.allo_sv_gen.rtl_fixups import fix_rtl

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "examples/allo_manual"
RAW = MANUAL / "hls_native/out.prj/solution1/syn/verilog"


def test_matches_manual_repair(tmp_path):
    record = fix_rtl(RAW, tmp_path)
    assert sorted((r["file"], r["fix"]) for r in record).count(
        ("gemm_fp32_gemm_fp32_Pipeline_l_S_k_0_k_l_j.v", "clear ap_ctrl_none completion stall")) == 4
    assert [r["signal"] for r in record if r["fix"].startswith("drop")] == [
        "grp_hf_add_f32_fu_634_ap_ready"]
    for path in sorted(tmp_path.glob("*.v")):
        assert path.read_text() == (MANUAL / "rtl" / path.name).read_text(), path.name


def test_real_handshake_is_not_patched(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    name = "gemm_fp32_gemm_fp32_Pipeline_l_outputs_u_l_v.v"
    text = (RAW / name).read_text().replace(
        "ap_block_pp0_stage0_subdone = (", "ap_block_pp0_stage0_subdone = (mem_ready_n | ", 1)
    (src / name).write_text(text)
    with pytest.raises(ValueError, match="unexpected stall dependency"):
        fix_rtl(src, tmp_path / "out")
