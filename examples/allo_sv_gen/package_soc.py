"""Stage a Chipyard APSRocketConfig run of the GEMM ISAX built by tools/allo-gen.

    python examples/allo_sv_gen/package_soc.py [--rtl DIR] [--output build/gemm_soc]
    pixi run bash build/gemm_soc/run_soc.sh

The ISAX comes from examples/allo/gemm.cadl; the software, run_soc.sh and
checks are those of examples/allo_manual. Without --rtl, Vitis runs.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "examples/allo_manual"
sys.path.insert(0, str(ROOT))

from cadl_frontend.allo_sv_gen.build import build  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtl", type=Path, help="HLS Verilog to use instead of running Vitis")
    parser.add_argument("--output", type=Path, default=ROOT / "build/gemm_soc")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.parent.parent != ROOT:
        sys.exit("run_soc.sh finds the repository two levels above its directory")
    build(ROOT / "examples/allo/gemm.cadl", out, rtl=args.rtl)
    shutil.copy(MANUAL / "run_soc.sh", out)
    shutil.copytree(MANUAL / "software", out / "software", dirs_exist_ok=True)
    print(f"Staged {out}; run: pixi run bash {out / 'run_soc.sh'}")


if __name__ == "__main__":
    main()
