"""build_kernel writes an interface manifest a wrapper can be checked against."""
import json
import re
import sys
from pathlib import Path

import pytest

allo = pytest.importorskip("allo")
from allo.ir.types import float32, int32, uint8  # noqa: E402

from cadl_frontend.allo_kernel.build import build_kernel  # noqa: E402
from cadl_frontend.allo_kernel.manifest import read_latency  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "examples/allo_manual"
sys.path.insert(0, str(MANUAL))
from schedule import build_schedule  # noqa: E402


def test_gemm_manifest_from_given_rtl(tmp_path):
    built = build_kernel(build_schedule(), tmp_path, rtl=MANUAL / "rtl")
    manifest = built.manifest
    assert json.loads((tmp_path / "kernel.json").read_text()) == manifest
    assert (manifest["top"], manifest["control"]) == ("gemm_fp32", "ap_ctrl_hs")
    assert manifest["latency"] is None  # no Vitis report for given RTL
    assert manifest["rtl"] == {"source": str(MANUAL / "rtl"), "provenance": "given"}
    params = {p["name"]: p for p in manifest["params"]}
    assert params["A"] == {"name": "A", "interface": "ap_memory", "direction": "in",
                           "dtype": "f32", "elem_bits": 32, "shape": [64], "elements": 64,
                           "layout": "row-major", "ports": 1, "read_latency": 1}
    assert [params[n]["direction"] for n in ("BT", "C", "D")] == ["in", "in", "out"]
    assert params["use_c"] == {"name": "use_c", "interface": "ap_none", "direction": "in",
                               "dtype": "ui8", "elem_bits": 8, "shape": []}


def test_manifest_without_rtl(tmp_path):
    built = build_kernel(build_schedule(), tmp_path, csynth=False)
    assert built.rtl_dir is None and built.interface is None
    assert built.manifest["rtl"] is None
    assert all(p["direction"] is None for p in built.manifest["params"])
    assert {p.get("ports") for p in built.manifest["params"]} == {None}


def test_partitioned_top_argument_rejected(tmp_path):
    def copy(A: float32[4], B: float32[4]):
        for i in range(4):
            B[i] = A[i]

    s = allo.customize(copy)
    s.partition("copy:A")
    with pytest.raises(NotImplementedError, match=r"partitioned top arguments \['A'\]"):
        build_kernel(s, tmp_path, csynth=False)


def test_returns_named_in_order(tmp_path):
    def stats(X: int32[16], k: uint8) -> (int32[16], uint8):
        Y: int32[16] = 0
        for i in range(16):
            Y[i] = X[i] * k
        n: uint8 = k + 1
        return Y, n

    built = build_kernel(allo.customize(stats), tmp_path, csynth=False)
    ret0, ret1 = built.manifest["returns"]
    assert ret0 == {"name": "ret0", "port": ret0["port"], "interface": "ap_memory",
                    "direction": None, "dtype": "i32", "elem_bits": 32, "shape": [16],
                    "elements": 16, "layout": "row-major", "ports": None, "read_latency": 1}
    assert ret1 == {"name": "ret1", "port": ret1["port"], "interface": "ap_vld",
                    "direction": None, "dtype": "ui8", "elem_bits": 8, "shape": []}
    # Allo names the output arguments itself; they trail the inputs in kernel.cpp.
    code = (tmp_path / "kernel.cpp").read_text()
    assert re.search(rf"uint8_t k,\s*int32_t {ret0['port']}\[16\],\s*uint8_t \*{ret1['port']}\s*\)", code)
    tcl = (tmp_path / "run.tcl").read_text()
    assert f"set_directive_interface -mode ap_memory -latency 1 stats {ret0['port']}" in tcl
    assert f"set_directive_interface -mode ap_vld stats {ret1['port']}" in tcl


def test_returned_argument_rejected(tmp_path):
    # Allo emits no output argument for a returned input, so the count is off.
    def echo(X: int32[4]) -> int32[4]:
        return X

    with pytest.raises(ValueError, match="1 HLS arguments for 1 inputs and 1 return values"):
        build_kernel(allo.customize(echo), tmp_path, csynth=False)


def test_read_latency(tmp_path):
    report = tmp_path / "csynth.xml"
    assert read_latency(report) is None
    report.write_text("<profile><PerformanceEstimates><SummaryOfOverallLatency>"
                      "<Best-caseLatency>88</Best-caseLatency>"
                      "<Worst-caseLatency>undef</Worst-caseLatency>"
                      "</SummaryOfOverallLatency></PerformanceEstimates></profile>")
    assert read_latency(report) == {"best": 88, "worst": None}
