"""Build a CADL allo block in a separate Allo interpreter, with a content-hashed cache.

The embedded Allo code is user Python and Allo carries its own MLIR bindings,
so it runs in a child process (worker.py) under `python`, the interpreter
with Allo installed; this process never imports Allo. The child leaves the
HLS project, RTL, kernel.json and project.json in the build directory, from
which the interface is classified again here.

The build directory's stage.json records the cache key: a hash of the
embedded code, the generator sources, the HardFloat IP contract, the Allo
version and interpreter, the vitis_hls release, given RTL and the options.
A later build with the same key reuses the results instead of running Allo
and Vitis again; any change runs them again.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import cadl_ast as ast
from .interface import HlsInstance, classify, load_rtl_ports
from .project import HLSProject
from .stage import ROOT, allo_version, generator_hash, hardfloat_hash, hash_files, sha256, vitis_hls_path

ERRORS = {e.__name__: e for e in (ValueError, NotImplementedError, FileNotFoundError, TypeError)}


@dataclass
class IsolatedBuild:
    project: HLSProject
    rtl_dir: Path | None
    interface: HlsInstance | None
    manifest: dict
    cached: bool  # results reused from an earlier build


def cache_inputs(kernel: ast.AlloKernel, *, rtl: Path | None, csynth: bool, vitis_hls: str,
                 python: str) -> dict:
    return {
        "kernel": kernel.name, "schedule": kernel.schedule, "code": sha256(kernel.source.text),
        "generator": generator_hash(), "hardfloat": hardfloat_hash(),
        "python": str(Path(python).resolve()), "allo": allo_version(python),
        "vitis_hls": vitis_hls_path(vitis_hls) if csynth and rtl is None else None,
        "rtl": hash_files(sorted(Path(rtl).glob("*.v"))) if rtl else None,
        "csynth": csynth,
    }


def load_build(directory: Path, cached: bool = False) -> IsolatedBuild:
    """The results a worker left in `directory`."""
    manifest = json.loads((directory / "kernel.json").read_text())
    project = HLSProject.from_json(json.loads((directory / "project.json").read_text()))
    rtl_dir, instance = None, None
    if manifest["rtl"] is not None:
        rtl_dir = Path(manifest["rtl"]["source"])
        instance = classify(project.top, load_rtl_ports(rtl_dir / f"{project.top}.v"),
                            project.params, project.returns)
    return IsolatedBuild(project, rtl_dir, instance, manifest, cached)


def _usable(directory: Path, key: str) -> bool:
    stage = directory / "stage.json"
    if not (stage.exists() and (directory / "kernel.json").exists()
            and (directory / "project.json").exists()):
        return False
    if json.loads(stage.read_text()).get("key") != key:
        return False
    rtl = json.loads((directory / "kernel.json").read_text())["rtl"]
    return rtl is None or Path(rtl["source"]).is_dir()


def build_kernel_isolated(cadl: Path, kernel: ast.AlloKernel, directory: Path, *,
                          rtl: Path | None = None, csynth: bool = True,
                          vitis_hls: str = "vitis_hls", python: str | None = None,
                          cache: bool = True) -> IsolatedBuild:
    python = python or os.environ.get("AQUAS_ALLO_PYTHON") or sys.executable
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    inputs = cache_inputs(kernel, rtl=rtl, csynth=csynth, vitis_hls=vitis_hls, python=python)
    key = sha256(json.dumps(inputs, sort_keys=True))
    if cache and _usable(directory, key):
        return load_build(directory, cached=True)

    (directory / "stage.json").unlink(missing_ok=True)
    request = {"cadl": str(Path(cadl).resolve()), "kernel": kernel.name,
               "directory": str(directory), "rtl": str(Path(rtl).resolve()) if rtl else None,
               "csynth": csynth, "vitis_hls": vitis_hls}
    (directory / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    run = subprocess.run([python, "-m", "cadl_frontend.allo_kernel.worker",
                          str(directory / "request.json")], env=env, capture_output=True, text=True)
    (directory / "worker.log").write_text(run.stdout + run.stderr)
    if run.returncode != 0:
        error = directory / "error.json"
        if not error.exists():
            raise RuntimeError(f"Allo worker ({python}) failed, see {directory / 'worker.log'}:\n"
                               + (run.stderr or run.stdout)[-2000:])
        failure = json.loads(error.read_text())
        raise ERRORS.get(failure["type"], RuntimeError)(failure["message"])
    (directory / "stage.json").write_text(json.dumps({"key": key, "inputs": inputs}, indent=2) + "\n")
    return load_build(directory)
