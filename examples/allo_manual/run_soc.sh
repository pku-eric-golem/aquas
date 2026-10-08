#!/usr/bin/env bash
# Called by run.sh --soc, or from pixi after package_rtl.py in the work directory.
set -euo pipefail
ulimit -c 0
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(realpath "$OUT/../..")
CHIPYARD="$ROOT/thirdparty/chipyard"
JOBS=${JOBS:-12}
SIM_CXX=${SIM_CXX:-/usr/bin/clang++}
sim_extra_flags='-std=c++20 -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-c++11-narrowing'
export APS_CONFIG="$OUT/aps_config.yaml"
export JAVA_TOOL_OPTIONS="-Xmx16G -Xss8M -XX:ActiveProcessorCount=$JOBS"
export TMPDIR="$OUT/temp"
mkdir -p "$TMPDIR" "$OUT/soc" "$OUT/software"
printf '%s\n' "$SIM_CXX ${SIM_CXX_OPT:--O1} $sim_extra_flags" > "$OUT/soc/host-flags.new"
if ! cmp -s "$OUT/soc/host-flags.new" "$OUT/soc/host-flags.txt"; then
  mv "$OUT/soc/host-flags.new" "$OUT/soc/host-flags.txt"
else
  rm "$OUT/soc/host-flags.new"
fi
flags=(-std=c11 -O3 -Wall -Wextra -Werror -fno-common -fno-builtin-printf
       -fno-fast-math -ffp-contract=off -march=rv32imaf_zicsr_zifencei -mabi=ilp32f
       -mcmodel=medany -specs=htif_nano.specs -static -T htif.ld -I "$OUT/software")
riscv32-unknown-elf-gcc "${flags[@]}" "$OUT/software/benchmark_fp32.c" -o "$OUT/software/benchmark.riscv"
riscv32-unknown-elf-gcc "${flags[@]}" "$OUT/software/test_gemm.c" -o "$OUT/software/test_gemm.riscv"
riscv32-unknown-elf-gcc "${flags[@]}" -DGEMM_FORCE_FAIL "$OUT/software/test_gemm.c" -o "$OUT/software/test_gemm_fail.riscv"
riscv32-unknown-elf-objdump -d "$OUT/software/benchmark.riscv" > "$OUT/software/benchmark.asm"
if grep -Eq 'fmadd.s|__addsf3|__mulsf3' "$OUT/software/benchmark.asm"; then
  echo 'Unexpected contracted/emulated FP32 software arithmetic'; exit 1
fi
args=(-C "$CHIPYARD/sims/verilator" CONFIG=APSRocketConfig "-j$JOBS" "nproc=$JOBS"
      "gen_dir=$OUT/soc/generated-src" "output_dir=$OUT/soc/output"
      "sim=$OUT/soc/simulator" "sim_debug=$OUT/soc/simulator-debug"
      "EXTRA_SIM_REQS=$OUT/run_soc.sh $OUT/rtl.f $OUT/soc/host-flags.txt"
      "SIM_OPT_CXXFLAGS=${SIM_CXX_OPT:--O1}"
      "EXTRA_SIM_CXXFLAGS=$sim_extra_flags"
      "VERILATOR_OPT_FLAGS=-O3 --x-assign fast --x-initial fast --output-split 10000 --output-split-cfuncs 100 -Wno-fatal -Wno-ZERODLY"
      "CXX=$SIM_CXX" CC=/usr/bin/gcc AR=/usr/bin/ar "LINK=$SIM_CXX")
rtl_reqs=$(python -c 'import json,os; print(" ".join(json.load(open(os.environ["APS_CONFIG"]))["vsrc"]))')
args+=('EXTRA_GENERATOR_REQS=$(BOOTROM_TARGETS) '"$APS_CONFIG $rtl_reqs")
make "${args[@]}" default > "$OUT/soc/build.log" 2>&1
ALLO_SIM_CXX="$SIM_CXX" ALLO_SIM_OPT="${SIM_CXX_OPT:--O1}" python - <<'PY'
import hashlib,json,os
from pathlib import Path
p=Path(os.environ['APS_CONFIG']).parent
model=p/'soc/generated-src/chipyard.harness.TestHarness.APSRocketConfig/chipyard.harness.TestHarness.APSRocketConfig'
(p/'host-simulator.json').write_text(json.dumps({
    'generated_cpp_unchanged':True,'generated_cpp_files':len(list(model.glob('*.cpp'))),
    'optimization':os.environ['ALLO_SIM_CXX']+' '+os.environ['ALLO_SIM_OPT']+' C++20; generated model unchanged',
    'simulator_sha256':hashlib.sha256((p/'soc/simulator').read_bytes()).hexdigest(),
    'model_rtl_filelist':'aps_config.yaml'},indent=2)+'\n')
PY
make "${args[@]}" run-binary-fast "BINARY=$OUT/software/benchmark.riscv" LOADMEM=1 \
     TIMEOUT_CYCLES=5000000 > "$OUT/soc/benchmark.log" 2>&1
grep -q 'ALLO FP32 BENCH PASS trials=2 checked=2048' "$OUT/soc/benchmark.log"
make "${args[@]}" run-binary-fast "BINARY=$OUT/software/test_gemm.riscv" LOADMEM=1 \
     TIMEOUT_CYCLES=5000000 > "$OUT/soc/test.log" 2>&1
grep -q 'ALLO FP32 GEMM PASS tiles=32 elements=2048' "$OUT/soc/test.log"
if make "${args[@]}" run-binary-fast "BINARY=$OUT/software/test_gemm_fail.riscv" LOADMEM=1 \
        TIMEOUT_CYCLES=5000000 > "$OUT/soc/negative.log" 2>&1; then
  echo 'FAIL: corrupted golden passed'; exit 1
fi
grep -q 'ALLO FP32 GEMM FAIL: batch=0 pass=0' "$OUT/soc/negative.log"
grep -Eq '\*\*\* FAILED \*\*\* \((tohost|exit code) = *1\)' "$OUT/soc/negative.log"
echo 'ALLO FP32 full Rocket/C RTL simulation PASS, including corrupted-golden rejection'
