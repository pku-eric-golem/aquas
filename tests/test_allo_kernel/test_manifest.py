"""build_kernel writes an interface manifest a wrapper can be checked against."""
import json
import sys
from pathlib import Path

import pytest

allo = pytest.importorskip("allo")
from allo.ir.types import float32  # noqa: E402

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


def test_read_latency(tmp_path):
    report = tmp_path / "csynth.xml"
    assert read_latency(report) is None
    report.write_text("<profile><PerformanceEstimates><SummaryOfOverallLatency>"
                      "<Best-caseLatency>88</Best-caseLatency>"
                      "<Worst-caseLatency>undef</Worst-caseLatency>"
                      "</SummaryOfOverallLatency></PerformanceEstimates></profile>")
    assert read_latency(report) == {"best": 88, "worst": None}
