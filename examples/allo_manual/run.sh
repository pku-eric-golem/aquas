#!/usr/bin/env bash
# Stage a reviewable archive into tmp/ before rebuilding anything.
set -euo pipefail
SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(realpath "$SOURCE/../..")
MODE=${1:---full}
case "$MODE" in
  --full|--rtl|--soc) ;;
  *) echo "Usage: bash examples/allo_manual/run.sh [--full|--rtl|--soc]" >&2; exit 2 ;;
esac
WORK_NAME=${ALLO_WORK_NAME:-allo_manual}
if [[ ! $WORK_NAME =~ ^[a-zA-Z0-9_-]+$ || $WORK_NAME == allo ]]; then
  echo 'ALLO_WORK_NAME must be a simple directory name other than allo.' >&2
  exit 2
fi
OUT="$ROOT/tmp/$WORK_NAME"
mkdir -p "$OUT"
if [[ $(realpath "$OUT") != "$OUT" ]]; then
  echo 'The work directory must not be a symlink.' >&2; exit 2
fi
python3 "$SOURCE/check_archive.py"
cp -a "$SOURCE/." "$OUT/"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
printf 'Reproduction mode %s, work directory %s\n' "$MODE" "$OUT"
if [[ $MODE == --full ]]; then
  exec bash "$OUT/pipeline.sh"
fi
python3 "$OUT/generate_vectors.py"
python3 "$OUT/build_blackbox.py" > "$OUT/blackbox-generation.log"
python3 "$OUT/package_blackbox.py"
python3 "$OUT/package_rtl.py"
if [[ $MODE == --rtl ]]; then
  exec bash "$OUT/validate_rtl.sh"
fi
exec pixi run bash "$OUT/run_soc.sh"
