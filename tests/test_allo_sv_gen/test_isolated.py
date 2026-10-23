"""The kernel build in its own Allo interpreter, its cache and its provenance
(cadl-allo-integration.md 5.A).

Builds use the manual GEMM RTL (--rtl), so no Vitis run is needed; the
interface report check is exercised on a report written from a known port list.
"""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cadl_frontend.allo_kernel.manifest import check_interface_report
from test_banks import banked_instance

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "examples/allo_manual"
GEMM = ROOT / "examples/allo/gemm.cadl"
FP_OPS = ROOT / "examples/allo/accept/fp_ops.cadl"

pytest.importorskip("allo")  # the worker needs it; the parent below does not


def _build(*args, **kwargs):
    from cadl_frontend.allo_sv_gen.build import build
    return build(*args, allo_python=sys.executable, **kwargs)


def test_parent_never_imports_allo(tmp_path):
    script = f"""
import sys
sys.modules["allo"] = None  # any import of Allo here fails
sys.path.insert(0, {str(ROOT)!r})
from cadl_frontend.allo_sv_gen.build import build
result = build({str(GEMM)!r}, {str(tmp_path / "out")!r}, rtl={str(MANUAL / "rtl")!r},
               allo_python={sys.executable!r})
print("built", result.main_sv)
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "built" in run.stdout and (tmp_path / "out/main.sv").exists()


def test_cache_reuses_until_an_input_changes(tmp_path):
    first = _build(GEMM, tmp_path / "out", rtl=MANUAL / "rtl")
    assert not first.cached
    main = (tmp_path / "out/main.sv").read_text()
    second = _build(GEMM, tmp_path / "out", rtl=MANUAL / "rtl")
    assert second.cached and (tmp_path / "out/main.sv").read_text() == main
    assert not _build(GEMM, tmp_path / "out", rtl=MANUAL / "rtl", cache=False).cached

    # A change to the embedded Allo code invalidates the build.
    changed = tmp_path / "gemm.cadl"
    changed.write_text(GEMM.read_text().replace("s = allo.customize", "# edited\ns = allo.customize"))
    assert not _build(changed, tmp_path / "out", rtl=MANUAL / "rtl").cached
    stage = json.loads((tmp_path / "out/hls/stage.json").read_text())
    assert set(stage["inputs"]) >= {"code", "generator", "hardfloat", "allo", "python", "rtl"}


def test_worker_errors_keep_their_type(tmp_path):
    bad = tmp_path / "fp_ops.cadl"
    bad.write_text(FP_OPS.read_text().replace('s.pipeline("i")\n', ""))
    with pytest.raises(NotImplementedError, match="outside a pipelined loop"):
        _build(bad, tmp_path / "out", csynth=False)
    assert "Traceback" in json.loads((tmp_path / "out/hls/error.json").read_text())["traceback"]


def test_provenance_and_both_hls_sources(tmp_path):
    _build(GEMM, tmp_path / "out", rtl=MANUAL / "rtl")
    hls = tmp_path / "out/hls"
    assert "hf_add_f32(" not in (hls / "kernel.orig.cpp").read_text()
    assert "hf_add_f32(" in (hls / "kernel.cpp").read_text()
    provenance = json.loads((hls / "kernel.json").read_text())["provenance"]
    assert set(provenance["versions"]) == {"python", "allo", "vitis_hls", "generator", "hardfloat"}
    assert set(provenance["hashes"]) == {"kernel.orig.cpp", "kernel.cpp", "blackbox", "rtl"}
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert len(manifest["main_sv_sha256"]) == 64


def _report(instance, path: Path, edit=None) -> Path:
    """An interface summary as Vitis writes it, for the ports of `instance`."""
    root = ET.Element("profile")
    summary = ET.SubElement(root, "InterfaceSummary")
    objects = {p.name: ("ap_ctrl_hs", instance.top) for p in instance.ports}
    for mem in instance.memories:
        obj = mem.param if mem.bank is None else f"{mem.param}_{mem.bank}"
        objects |= {port.name: ("ap_memory", obj) for port in mem.signals.values()}
    for port in instance.ports:
        row = ET.SubElement(summary, "RtlPorts")
        protocol, obj = objects[port.name]
        fields = {"name": port.name, "Object": obj, "IOProtocol": protocol,
                  "Dir": "in" if port.direction == "input" else "out", "Bits": str(port.bits)}
        if edit:
            fields = edit(fields)
        for tag, text in fields.items():
            ET.SubElement(row, tag).text = text
    ET.ElementTree(root).write(path)
    return path


def test_interface_report_cross_check(tmp_path):
    instance = banked_instance()
    check_interface_report(_report(instance, tmp_path / "ok.xml"), instance)
    wider = lambda f: f | {"Bits": "4"} if f["name"] == "A_1_address0" else f  # noqa: E731
    with pytest.raises(ValueError, match=r"A_1_address0: report \('ap_memory', 'out', 4\)"):
        check_interface_report(_report(instance, tmp_path / "wide.xml", wider), instance)
    other = lambda f: f | {"Object": "A_0"} if f["name"] == "A_1_q0" else f  # noqa: E731
    with pytest.raises(ValueError, match="A_1_q0: report .* A_0, expected .* A_1"):
        check_interface_report(_report(instance, tmp_path / "obj.xml", other), instance)
