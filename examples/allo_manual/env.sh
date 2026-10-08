#!/usr/bin/env bash
# Full Allo source build only; RTL/SoC reproduction does not activate conda.
ALLO_WORK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALLO_REPO=$(realpath "$ALLO_WORK/../..")
if [[ ${ALLO_SKIP_CONDA:-0} != 1 ]]; then
  source "${ALLO_CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${ALLO_CONDA_ENV:-allo}"
fi
export PYTHONPATH="$ALLO_WORK/python:$ALLO_WORK${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -z ${LLVM_BUILD_DIR:-} ]]; then
  for candidate in "$ALLO_REPO/thirdparty/allo/externals/llvm-project/build" "$HOME/repos/allo/externals/llvm-project/build"; do
    if [[ -f $candidate/lib/cmake/mlir/MLIRConfig.cmake ]]; then
      export LLVM_BUILD_DIR="$candidate"
      break
    fi
  done
fi
if [[ -z ${LLVM_BUILD_DIR:-} || ! -f $LLVM_BUILD_DIR/lib/cmake/mlir/MLIRConfig.cmake ]]; then
  echo 'Set LLVM_BUILD_DIR to an existing compatible Allo LLVM/MLIR build.' >&2
  return 1
fi
export TMPDIR="$ALLO_WORK/temp"
mkdir -p "$TMPDIR"
