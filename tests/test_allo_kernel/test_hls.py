import json
import re
import sys
from pathlib import Path

import pytest

allo = pytest.importorskip("allo")
from allo.ir.types import float32  # noqa: E402

from cadl_frontend.allo_kernel.hls import emit_project  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/allo_manual"))
from schedule import build_schedule  # noqa: E402


def test_gemm_native_interface(tmp_path):
    project = emit_project(build_schedule(), tmp_path)
    assert [(p.name, p.kind) for p in project.params] == [
        ("A", "ap_memory"), ("BT", "ap_memory"), ("C", "ap_memory"),
        ("use_c", "ap_none"), ("D", "ap_memory")]
    assert project.float_calls == {"hf_add_f32": 2, "hf_mul_f32": 1}

    code = (tmp_path / "kernel.cpp").read_text()
    assert "m_axi" not in code
    assert "float A[64]" in code and "uint8_t use_c" in code
    assert code.count("hf_add_f32(") == 2 and code.count("hf_mul_f32(") == 1
    assert not re.search(r"float v\d+ = v\d+ [-+*/] v\d+;", code)

    tcl = (tmp_path / "run.tcl").read_text()
    assert "set_directive_interface -mode ap_memory -latency 1 gemm_fp32 D" in tcl
    assert "set_directive_interface -mode ap_none gemm_fp32 use_c" in tcl

    configs = [json.loads((tmp_path / f"blackbox/{n}.json").read_text())
               for n in ("hf_add_f32", "hf_mul_f32")]
    bundled = [c for c in configs if any(f.endswith("HardFloatNative.v") for f in c["rtl_files"])]
    assert len(bundled) == 1
    assert [c["c_function_name"] for c in configs] == ["hf_add_f32_rtl", "hf_mul_f32_rtl"]
    assert all(c["c_return"]["rtl_ports"]["data_write_out"] == "result" for c in configs)


def test_unsupported_float_op_rejected(tmp_path):
    def sub(A: float32[4], B: float32[4]):
        for i in range(4):
            B[i] = A[i] - 1.0

    with pytest.raises(NotImplementedError, match="arith.subf"):
        emit_project(allo.customize(sub), tmp_path)
