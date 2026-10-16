"""Build the allo_manual GEMM schedule through allo_sv_gen and run csynth."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples/allo_manual")]

from cadl_frontend.allo_sv_gen.hls import emit_project, run_csynth  # noqa: E402
from schedule import build_schedule  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "build/gemm_fp32")
    parser.add_argument("--no-csynth", dest="csynth", action="store_false")
    args = parser.parse_args()
    project = emit_project(build_schedule(), args.output)
    print("params:", [(p.name, p.kind) for p in project.params])
    print("float calls:", project.float_calls)
    if args.csynth:
        print("verilog:", run_csynth(project))


if __name__ == "__main__":
    main()
