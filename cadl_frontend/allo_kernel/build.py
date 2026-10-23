"""Build an Allo schedule into a native-interface RTL module plus its manifest.

    <directory>/kernel.orig.cpp, kernel.cpp      HLS C++ before and after float lowering
    <directory>/run.tcl, blackbox/               rest of the HLS project (hls.py)
    <directory>/out.prj/.../syn/verilog, report  Vitis RTL and reports (csynth only)
    <directory>/rtl/, rtl_fixups.json            repaired Vitis Verilog and its record
    <directory>/kernel.json                      interface manifest (manifest.py), with
                                                 tool versions and content hashes
    <directory>/project.json                     the HLS project as data (project.py)

build_kernel runs in the process that executed the Allo code; isolated.py runs
it in another interpreter and reads the results back.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .hls import emit_project, run_csynth
from .interface import HlsInstance, classify, load_rtl_ports
from .manifest import check_interface_report, kernel_manifest, read_latency
from .project import HLSProject
from .stage import hash_files, versions


@dataclass
class KernelBuild:
    project: HLSProject
    rtl_dir: Path | None  # None when synthesis was skipped and no RTL was given
    interface: HlsInstance | None
    manifest: dict


def build_kernel(schedule, directory: Path, *, rtl: Path | None = None, csynth: bool = True,
                 vitis_hls: str = "vitis_hls") -> KernelBuild:
    """Emit the HLS project for `schedule`, then take RTL from `rtl` or from Vitis."""
    project = emit_project(schedule, directory)
    latency, report = None, None
    if rtl is not None:
        rtl_dir, provenance = Path(rtl).resolve(), "given"
    elif csynth:
        if shutil.which(vitis_hls) is None:
            raise FileNotFoundError(f"{vitis_hls} not on PATH; source the Vitis settings64.sh, "
                                    "pass --rtl, or use --no-csynth")
        rtl_dir, provenance = run_csynth(project, vitis_hls), "vitis_hls + rtl_fixups"
        report = project.directory / "out.prj/solution1/syn/report/csynth.xml"
        latency = read_latency(report)
    else:
        rtl_dir = None

    instance, source = None, None
    if rtl_dir is not None:
        instance = classify(project.top, load_rtl_ports(rtl_dir / f"{project.top}.v"),
                            project.params, project.returns)
        if report is not None:
            check_interface_report(report, instance)
        source = {"source": str(rtl_dir), "provenance": provenance}
    directory = project.directory
    hashes = {name: hash_files([directory / name]) for name in ("kernel.orig.cpp", "kernel.cpp")}
    hashes["blackbox"] = hash_files(sorted((directory / "blackbox").glob("*")))
    if rtl_dir is not None:
        hashes["rtl"] = hash_files(sorted(rtl_dir.glob("*.v")))
    manifest = kernel_manifest(project, instance, latency, source,
                               {"versions": versions(vitis_hls), "hashes": hashes})
    (directory / "kernel.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "project.json").write_text(json.dumps(project.to_json(), indent=2) + "\n")
    return KernelBuild(project, rtl_dir, instance, manifest)
