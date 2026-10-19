"""Stage a Chipyard APSRocketConfig run of an ISAX built by tools/allo-gen.

    python examples/allo_sv_gen/package_soc.py [--example gemm|systolic] [--rtl DIR] [--output build/<name>]
    pixi run bash build/<name>/run_soc.sh

Each example names its .cadl and the RV32 programs run on Rocket, with the
line each must print; negative programs must also fail. The GEMM programs are
those of examples/allo_manual. Without --rtl, Vitis runs.
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANUAL = ROOT / "examples/allo_manual"
sys.path.insert(0, str(ROOT))

from cadl_frontend.allo_sv_gen.build import build  # noqa: E402

# name -> (.cadl, software directory, [(program, source, cflags, expected line, negative)])
EXAMPLES = {
    "gemm": (ROOT / "examples/allo/gemm.cadl", MANUAL / "software", [
        ("benchmark", "benchmark_fp32.c", "", "ALLO FP32 BENCH PASS trials=2 checked=2048", False),
        ("test_gemm", "test_gemm.c", "", "ALLO FP32 GEMM PASS tiles=32 elements=2048", False),
        ("test_gemm_fail", "test_gemm.c", "-DGEMM_FORCE_FAIL",
         "ALLO FP32 GEMM FAIL: batch=0 pass=0", True),
    ]),
    "systolic": (ROOT / "examples/allo/systolic.cadl", HERE / "software", [
        ("test_systolic", "test_systolic.c", "", "ALLO SYSTOLIC PASS batches=2 elements=128", False),
        ("test_systolic_fail", "test_systolic.c", "-DSYSTOLIC_FORCE_FAIL",
         "ALLO SYSTOLIC FAIL: batch=0 row=0 col=0", True),
    ]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", choices=EXAMPLES, default="gemm")
    parser.add_argument("--rtl", type=Path, help="HLS Verilog to use instead of running Vitis")
    parser.add_argument("--output", type=Path, help="default: build/<example>_soc")
    args = parser.parse_args()
    cadl, software, programs = EXAMPLES[args.example]
    out = (args.output or ROOT / f"build/{args.example}_soc").resolve()
    if out.parent.parent != ROOT:
        sys.exit("run_soc.sh finds the repository two levels above its directory")
    build(cadl, out, rtl=args.rtl)
    shutil.copy(HERE / "run_soc.sh", out)
    shutil.copytree(software, out / "software", dirs_exist_ok=True)
    (out / "programs.tsv").write_text("".join(
        f"{name}\t{source}\t{cflags or '-'}\t{expect}\t{int(negative)}\n"
        for name, source, cflags, expect, negative in programs))
    print(f"Staged {out}; run: pixi run bash {out / 'run_soc.sh'}")


if __name__ == "__main__":
    main()
