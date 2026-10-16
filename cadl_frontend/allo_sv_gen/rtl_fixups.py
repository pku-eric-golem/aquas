"""Constrained repairs of Vitis HLS 2024.1 RTL around ap_ctrl_none HardFloat blackboxes.

Vitis 2024.1 (a) may connect a second ap_ready pin to a scalar-return
blackbox call, next to the auxiliary ap_ready output, and (b) makes the
loop pipelines that call the blackboxes wait for a completion handshake an
ap_ctrl_none call never produces, so they stall forever. These are the
repairs of examples/allo_manual/package_blackbox.py. Each rewrite is gated:
an unused pin is only dropped if nothing reads it, and a stall predicate is
only cleared if it consists of loop-valid terms, so a real handshake (FIFO,
ready/valid, variable latency) raises instead of being patched.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

INSTANCE = re.compile(r"\bhf_(?:add|mul)_f32(?:_rtl)?\s+\w+\s*\([\s\S]*?\);")
STALL = re.compile(r"\b(ap_block_pp0_stage0_(?:11001(?:_ignoreCallOp\d+)?|subdone))\s*=\s*([^;]+);")
LOOP_VALID = re.compile(
    r"\b(?:icmp_ln\d+_reg_\d+_pp0_iter\d+_reg|ap_enable_reg_pp0_iter\d+)\b|\d+'[db][01]|[()&|=\s]")
HANDSHAKES = ("_empty_n", "_full_n", "FIFO")


def _drop_unused_ready(source: str, file: str, record: list[dict]) -> str:
    def fix(match):
        instance = match.group(0)
        pins = re.findall(r"\.ap_ready\((\w+)\)", instance)
        if len(pins) < 2:
            return instance
        # The first pin is the declared auxiliary output (`<inst>_ap_ready_r`); the
        # second is the extra control pin, which must have no reader.
        aux, extra = pins if len(pins) == 2 else (None, None)
        if not (aux and aux.endswith("_ap_ready_r") and extra == aux[:-2]
                and len(re.findall(rf"\b{re.escape(extra)}\b", source)) == 2):
            raise ValueError(f"{file}: unexpected ap_ready pins {pins}")
        instance = re.sub(rf"\s*\.ap_ready\({re.escape(extra)}\)\s*,?", "", instance)
        record.append(dict(file=file, fix="drop unused duplicate ap_ready pin", signal=extra))
        return re.sub(r",\s*(?=\);)", "\n", instance)
    return INSTANCE.sub(fix, source)


def _clear_stalls(source: str, file: str, record: list[dict]) -> str:
    if not INSTANCE.search(source) or "_Pipeline_" not in file:
        return source
    if any(h in source for h in HANDSHAKES):
        raise ValueError(f"{file}: blackbox pipeline with a real handshake; not patching")

    def fix(match):
        signal, expression = match.groups()
        if LOOP_VALID.sub("", expression):
            raise ValueError(f"{file}: unexpected stall dependency in {signal}: {expression}")
        record.append(dict(file=file, fix="clear ap_ctrl_none completion stall", signal=signal,
                           original=expression.strip()))
        return f"{signal} = 1'b0;"
    fixed, count = STALL.subn(fix, source)
    if not count:
        raise ValueError(f"{file}: no stall predicate found in a blackbox pipeline")
    return fixed


def fix_rtl(source_dir: Path, out_dir: Path) -> list[dict]:
    """Copy Vitis Verilog from `source_dir` to `out_dir`, repairing it; return the record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.v"):
        old.unlink()
    record: list[dict] = []
    for path in sorted(source_dir.glob("*.v")):
        text = path.read_text()
        if "floating_point" in text:
            raise ValueError(f"{path.name}: Xilinx floating-point IP in the HLS output")
        fixed = _clear_stalls(_drop_unused_ready(text, path.name, record), path.name, record)
        if fixed == text:
            shutil.copyfile(path, out_dir / path.name)
        else:
            (out_dir / path.name).write_text(fixed)
    return record
