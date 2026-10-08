#!/usr/bin/env bash
set -euo pipefail
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(realpath "$OUT/../..")
cd "$ROOT"
python3 "$OUT/verify_blackbox.py" --skip-hls-model > "$OUT/blackbox-validation.log" 2>&1
pixi run verilator --cc --exe --build --no-timing --assert -Wno-fatal --top-module main \
  -f "$OUT/rtl.f" "$OUT/test_wrapper.cpp" --Mdir "$OUT/obj-wrapper" \
  -CFLAGS "-O0 -I$OUT" -MAKEFLAGS 'CXX=/usr/bin/g++ LINK=/usr/bin/g++' -j "${JOBS:-12}" \
  > "$OUT/wrapper-build.log" 2>&1
"$OUT/obj-wrapper/Vmain" > "$OUT/wrapper-test.log" 2>&1
cat "$OUT/blackbox-validation.log" "$OUT/wrapper-test.log"
