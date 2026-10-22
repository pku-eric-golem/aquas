"""Build an Allo schedule into a native-interface RTL module plus its manifest.

    <directory>/kernel.cpp, run.tcl, blackbox/   HLS project (hls.py)
    <directory>/rtl/                             repaired Vitis Verilog (csynth only)
    <directory>/kernel.json                      interface manifest (manifest.py)
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .hls import HLSProject, emit_project, run_csynth
from .interface import HlsInstance, classify, load_rtl_ports
from .manifest import kernel_manifest, read_latency


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
    latency = None
    if rtl is not None:
        rtl_dir, provenance = Path(rtl).resolve(), "given"
    elif csynth:
        if shutil.which(vitis_hls) is None:
            raise FileNotFoundError(f"{vitis_hls} not on PATH; source the Vitis settings64.sh, "
                                    "pass --rtl, or use --no-csynth")
        rtl_dir, provenance = run_csynth(project, vitis_hls), "vitis_hls + rtl_fixups"
        latency = read_latency(project.directory / "out.prj/solution1/syn/report/csynth.xml")
    else:
        rtl_dir = None

    instance, source = None, None
    if rtl_dir is not None:
        instance = classify(project.top, load_rtl_ports(rtl_dir / f"{project.top}.v"),
                            project.params, project.returns)
        source = {"source": str(rtl_dir), "provenance": provenance}
    manifest = kernel_manifest(project, instance, latency, source)
    (project.directory / "kernel.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return KernelBuild(project, rtl_dir, instance, manifest)
