#!/usr/bin/env bash
# Run from pixi after package_soc.py: build the APSRocketConfig simulator for
# the staged ISAX, then run each program of programs.tsv (tab-separated: name,
# source, extra cflags or -, expected line, negative 0/1). A program passes if
# it prints its expected line; a negative one must also fail through tohost.
# The simulator build follows examples/allo_manual/run_soc.sh.
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
while IFS=$'\t' read -r name source cflags expect negative <&3; do
  [ "$cflags" = - ] && cflags=
  # shellcheck disable=SC2086  # cflags is a word list
  riscv32-unknown-elf-gcc "${flags[@]}" $cflags "$OUT/software/$source" -o "$OUT/software/$name.riscv"
  riscv32-unknown-elf-objdump -d "$OUT/software/$name.riscv" > "$OUT/software/$name.asm"
  # The reference arithmetic must be separately rounded hardware mul/add.
  if grep -Eq 'fmadd.s|__addsf3|__mulsf3' "$OUT/software/$name.asm"; then
    echo "Unexpected contracted/emulated FP32 software arithmetic in $name"; exit 1
  fi
done 3< "$OUT/programs.tsv"

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

while IFS=$'\t' read -r name source cflags expect negative <&3; do
  log="$OUT/soc/$name.log"
  if make "${args[@]}" run-binary-fast "BINARY=$OUT/software/$name.riscv" LOADMEM=1 \
          TIMEOUT_CYCLES=5000000 > "$log" 2>&1; then
    if [ "$negative" = 1 ]; then echo "FAIL: negative program $name passed"; exit 1; fi
  else
    if [ "$negative" != 1 ]; then echo "FAIL: $name failed, see $log"; exit 1; fi
    grep -Eq '\*\*\* FAILED \*\*\* \((tohost|exit code) = *1\)' "$log"
  fi
  grep -qF "$expect" "$log" || { echo "FAIL: $name did not print: $expect"; exit 1; }
  echo "$name: ok"
done 3< "$OUT/programs.tsv"
echo 'ALLO full Rocket/C RTL simulation PASS, including negative programs'
