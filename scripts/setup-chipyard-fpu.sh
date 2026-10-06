#!/usr/bin/env bash
# Minimal full-Chipyard RTL prerequisites, without pk/FPGA/Linux toolchains.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CHIPYARD=${APS_CHIPYARD:-$ROOT/thirdparty/chipyard}
JOBS=${JOBS:-16}
: "${RISCV:?Run this script through pixi run}"
if [[ "$CHIPYARD" == "$ROOT/thirdparty/chipyard" ]]; then
  git -C "$ROOT" submodule update --init --depth 1 thirdparty/chipyard
fi
# All generator projects are compiled by Chipyard's SBT assembly, even when
# only Rocket is elaborated. Do not recursively fetch unrelated FPGA/Linux IP.
mapfile -t generators < <(git -C "$CHIPYARD" config -f .gitmodules \
  --get-regexp '^submodule\..*\.path$' | awk '$2 ~ /^generators\// {print $2}')
git -C "$CHIPYARD" submodule update --init --depth 1 --jobs "$JOBS" \
  "${generators[@]}" sims/firesim fpga/fpga-shells tools/cde tools/fixedpoint \
  tools/dsptools tools/rocket-dsp-utils tools/firrtl2 tools/DRAMSim2 \
  toolchains/riscv-tools/riscv-isa-sim toolchains/libgloss
git -C "$CHIPYARD/generators/constellation" submodule update --init --depth 1 espresso

mkdir -p "$ROOT/build/chipyard-spike" "$ROOT/build/chipyard-libgloss"
(
  cd "$ROOT/build/chipyard-spike"
  "$CHIPYARD/toolchains/riscv-tools/riscv-isa-sim/configure" --prefix="$RISCV" \
    --with-boost=no --with-boost-asio=no --with-boost-regex=no
  make -j"$JOBS"
  make install
  make -j"$JOBS" libfesvr.a
  cp libfesvr.a "$RISCV/lib/"
)
# GCC 13 requires explicit zicsr for startup CSR instructions. Reuse Chipyard's
# own RV32 patch, but do not reapply it on subsequent setup runs.
libgloss="$CHIPYARD/toolchains/libgloss"
patch="$CHIPYARD/toolchains/aps_rv32_patch/1_libgloss.patch"
if ! git -C "$libgloss" apply --reverse --check "$patch" 2>/dev/null; then
  git -C "$libgloss" apply "$patch"
fi
(
  cd "$ROOT/build/chipyard-libgloss"
  "$libgloss/configure" --prefix="$RISCV/riscv32-unknown-elf" --host=riscv32-unknown-elf
  make -j"$JOBS"
  make install
)
cmake -S "$CHIPYARD/generators/constellation/espresso" \
  -B "$ROOT/build/chipyard-espresso" -DBUILD_DOC=OFF -DCMAKE_INSTALL_PREFIX="$RISCV"
cmake --build "$ROOT/build/chipyard-espresso" --parallel "$JOBS"
cmake --install "$ROOT/build/chipyard-espresso"
echo 'Chipyard prerequisites ready. Run: pixi run bf16-gemm-rocket'
