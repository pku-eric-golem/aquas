"""Content hashes and tool versions that identify a kernel build (no Allo import).

A build's provenance (kernel.json "provenance") records the versions and the
hashes of what it read and wrote. Its cache key (isolated.py) hashes every
input that can change the result: the embedded Allo code, this package's
sources (the generator), the HardFloat IP contract, the tools and options.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = Path(__file__).resolve().parent
# What the blackboxes are made of: pipeline, operation bank, pinned HardFloat sources.
HARDFLOAT = [ROOT / "hardware/hardfloat/HFNativePipeline.sv", ROOT / "hardware/hardfloat/HFCompute.sv",
             ROOT / "thirdparty/HardFloat/SOURCE.sha256"]


def sha256(*parts: bytes | str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        data = part.encode() if isinstance(part, str) else part
        digest.update(len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()


def hash_files(paths: list[Path], root: Path | None = None) -> str:
    """One hash over files, by name (relative to `root`) and content, in name order."""
    items = sorted((str(p.relative_to(root)) if root else p.name, p) for p in paths)
    return sha256(*(part for name, p in items for part in (name, p.read_bytes())))


def generator_hash() -> str:
    return hash_files(sorted(GENERATOR.glob("*.py")), GENERATOR)


def hardfloat_hash() -> str:
    return hash_files([p for p in HARDFLOAT if p.exists()], ROOT)


def allo_version(python: str) -> str:
    """The Allo distribution version in interpreter `python`, without importing Allo."""
    run = subprocess.run([python, "-c", "import importlib.metadata as m; print(m.version('allo'))"],
                         capture_output=True, text=True)
    return run.stdout.strip() if run.returncode == 0 else "unknown"


def vitis_hls_path(vitis_hls: str) -> str | None:
    """The resolved vitis_hls executable; its path names the release (e.g. 2024.1)."""
    found = shutil.which(vitis_hls)
    return str(Path(found).resolve()) if found else None


def versions(vitis_hls: str, python: str = sys.executable) -> dict:
    return {"python": sys.version.split()[0], "allo": allo_version(python),
            "vitis_hls": vitis_hls_path(vitis_hls), "generator": generator_hash(),
            "hardfloat": hardfloat_hash()}
