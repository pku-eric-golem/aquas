#!/usr/bin/env bash
# Reproduce Allo -> Vitis HLS -> native ports -> RoCC -> full Rocket C/RTL tests.
set -euo pipefail
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(realpath "$OUT/../..")
cd "$ROOT"
source "$OUT/env.sh"
export VITIS_HLS_ROOT=${VITIS_HLS_ROOT:-/opt/Xilinx/Vitis_HLS/2024.1}
cmake -S "$ROOT/thirdparty/allo/mlir" -B "$OUT/allo-build" -G Ninja \
      -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" -DPython3_EXECUTABLE="$(command -v python)" \
      -Dnanobind_DIR="$(python -m nanobind --cmake_dir)" -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=allo \
      > "$OUT/allo-configure.log" 2>&1
cmake --build "$OUT/allo-build" -j "${ALLO_BUILD_JOBS:-4}" > "$OUT/allo-build.log" 2>&1
python "$OUT/bootstrap.py"
python "$OUT/generate_vectors.py"
python "$OUT/verify_allo.py" > "$OUT/allo-validation.log" 2>&1
python "$OUT/generate.py" > "$OUT/generate.log" 2>&1
source "$VITIS_HLS_ROOT/settings64.sh"
for project in hls_axi hls_native; do
  (cd "$OUT/$project"; "$VITIS_HLS_ROOT/bin/vitis_hls" -f run.tcl) > "$OUT/${project/_/-}.log" 2>&1
done
python "$OUT/verify_blackbox.py"
python "$OUT/package_blackbox.py"
python "$OUT/package_rtl.py"
pixi run verilator --cc --exe --build --no-timing --assert -Wno-fatal --top-module main \
     -f "$OUT/rtl.f" "$OUT/test_wrapper.cpp" --Mdir "$OUT/obj-wrapper" \
     -CFLAGS "-O0 -I$OUT" -MAKEFLAGS 'CXX=/usr/bin/g++ LINK=/usr/bin/g++' -j "${JOBS:-12}" \
     > "$OUT/wrapper-build.log" 2>&1
"$OUT/obj-wrapper/Vmain" > "$OUT/wrapper-test.log" 2>&1
pixi run bash "$OUT/run_soc.sh" > "$OUT/soc-runner.log" 2>&1
python "$OUT/summarize.py"
cat "$OUT/wrapper-test.log"
grep -E 'ALLO_BENCH|ALLO FP32.*PASS' "$OUT/soc/benchmark.log" "$OUT/soc/test.log"
