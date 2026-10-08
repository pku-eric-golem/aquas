"""Reproduce numerical, CE/reset, and three-enabled-edge blackbox checks."""
import os
import argparse
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BB = HERE / "blackbox"
sys.path.insert(0, str(ROOT / "tests/hardfloat"))
from ieee_oracle import IEEE


def run(args, log):
    with log.open("w") as stream:
        subprocess.run(list(map(str, args)), cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hls-model", action="store_true",
                        help="Run native RTL checks without requiring Vitis C++ headers")
    args = parser.parse_args()
    fp = IEEE(32)
    rng = random.Random(0xCE3101)
    edge = [0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x80800000,
            0x3f800000,0xbf800000,0x3f800001,0x7f7fffff,0xff7fffff,
            0x7f800000,0xff800000,0x7fc00000,0x7f800001]
    pairs = [(a,b) for a in edge for b in edge]
    pairs += [(rng.getrandbits(32),rng.getrandbits(32)) for _ in range(3000)]
    (BB / "vectors.txt").write_text("".join(
        f'{a:08x} {b:08x} {fp.operation("add",a,b)[0]:08x} {fp.operation("mul",a,b)[0]:08x}\n'
        for a,b in pairs))
    run(["pixi","run","verilator","--cc","--exe","--build","--no-timing",
         "--assert","-Wno-fatal","--top-module","test_pair",
         BB / "HardFloatNative.v", BB / "hf_add_f32.v", BB / "hf_mul_f32.v",
         BB / "test_pair.v", BB / "test_pair.cpp", "--Mdir", BB / "obj",
         "-CFLAGS","-O1","-j","4"], BB / "build.log")
    run([BB / "obj/Vtest_pair", BB / "vectors.txt"], BB / "test.log")
    print((BB / "test.log").read_text().strip())
    if args.skip_hls_model:
        return
    include = Path(os.getenv("VITIS_HLS_ROOT", "/opt/Xilinx/Vitis_HLS/2024.1")) / "include"
    run(["g++","-O2","-fno-fast-math","-ffp-contract=off","-Wno-unknown-pragmas",
         "-I" + str(include), HERE / "hls_native/kernel.cpp", BB / "hf_add_f32.cpp",
         BB / "hf_mul_f32.cpp", HERE / "test_hls_model.cpp", "-o", HERE / "hls-model-test"],
        HERE / "hls-model-build.log")
    run([HERE / "hls-model-test"], HERE / "hls-model-test.log")
    print((HERE / "hls-model-test.log").read_text().strip())


if __name__ == "__main__":
    main()
