#!/usr/bin/env bash
# Source from the repository root or a script under tmp/allo.
ALLO_WORK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALLO_REPO=$(realpath "$ALLO_WORK/../..")
source /home/ytsun/miniconda3/etc/profile.d/conda.sh
conda activate allo
export PYTHONPATH="$ALLO_WORK/python:$ALLO_WORK${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export LLVM_BUILD_DIR=/home/ytsun/repos/allo/externals/llvm-project/build
export TMPDIR="$ALLO_WORK/temp"
mkdir -p "$TMPDIR"
