"""Run build_kernel for one CADL allo block in this interpreter (see isolated.py).

    python -m cadl_frontend.allo_kernel.worker request.json

The request names the .cadl file, the kernel and the build options. The
embedded Allo code runs here, with this interpreter's Allo and MLIR bindings.
On failure, error.json in the build directory carries the exception type and
message for the parent to raise again.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def run(request: dict) -> None:
    from ..allo import check_source
    from .build import build_kernel
    from .schedule import load_schedule

    cadl, directory = Path(request["cadl"]), Path(request["directory"])
    kernel = check_source(cadl.read_text(), str(cadl)).allo_kernels[request["kernel"]]
    rtl = request["rtl"] and Path(request["rtl"])
    build_kernel(load_schedule(kernel, directory, cadl), directory, rtl=rtl,
                 csynth=request["csynth"], vitis_hls=request["vitis_hls"])


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text())
    error = Path(request["directory"]) / "error.json"
    error.unlink(missing_ok=True)
    try:
        run(request)
    except Exception as e:  # noqa: BLE001 - every failure goes back to the parent
        error.write_text(json.dumps({"type": type(e).__name__, "message": str(e),
                                     "traceback": traceback.format_exc()}, indent=2) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
