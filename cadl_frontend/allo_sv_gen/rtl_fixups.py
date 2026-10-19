"""Constrained repair of Vitis HLS 2024.1 RTL around ap_ctrl_none HardFloat blackboxes.

Vitis 2024.1 may connect an `ap_ready` pin to a scalar-return ap_ctrl_none
blackbox call although the blackbox has no such port (blackbox.py). The pin
drives a wire nothing reads; it is dropped only if the wire has no other use.

The archived manual flow (examples/allo_manual/package_blackbox.py) also had to
clear pipeline stalls: its blackboxes declared an extra `ap_ready` output, and
Vitis then stalls the calling pipeline on a completion that never comes. Our
blackboxes do not declare it, so that stall must not appear; a stall predicate
with a term made only of loop-valid signals raises instead of being patched.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

INSTANCE = re.compile(r"\bhf_(?:add|mul)_f32(?:_rtl)?\s+\w+\s*\([\s\S]*?\);")
STALL = re.compile(r"\b(ap_block_pp0_stage\d+\w*)\s*=\s*([^;]+);")
LOOP_SIGNAL = re.compile(r"\b(?:icmp_ln\d+_reg_\d+_pp0_iter\d+_reg|ap_enable_reg_pp0_iter\d+)\b")
CONSTANT = re.compile(r"\d+'[db][01]|[()&|=~\s]")


def _drop_dangling_ready(source: str, file: str, record: list[dict]) -> str:
    def fix(match):
        instance = match.group(0)
        pins = re.findall(r"\.ap_ready\((\w+)\)", instance)
        if not pins:
            return instance
        # Exactly one pin, whose wire appears only in its declaration and here.
        if len(pins) > 1 or len(re.findall(rf"\b{re.escape(pins[0])}\b", source)) != 2:
            raise ValueError(f"{file}: unexpected ap_ready pins {pins}")
        instance = re.sub(rf"\s*\.ap_ready\({re.escape(pins[0])}\)\s*,?", "", instance)
        record.append(dict(file=file, fix="drop dangling ap_ready pin", signal=pins[0]))
        return re.sub(r",\s*(?=\);)", "\n", instance)
    return INSTANCE.sub(fix, source)


def _terms(expression: str) -> list[str]:
    """Top-level `|` operands of a predicate, enclosing parentheses removed."""
    text = expression.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        for i, ch in enumerate(text):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(text) - 1:
                break
        else:
            text = text[1:-1].strip()
            continue
        break
    depth, current, terms = 0, "", []
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        if ch == "|" and depth == 0:
            terms.append(current)
            current = ""
        else:
            current += ch
    return terms + [current]


def _check_stalls(source: str, file: str) -> None:
    if not INSTANCE.search(source):
        return
    for signal, expression in STALL.findall(source):
        for term in _terms(expression):
            if LOOP_SIGNAL.search(term) and not CONSTANT.sub("", LOOP_SIGNAL.sub("", term)):
                raise ValueError(f"{file}: blackbox completion stall in {signal}: {expression}")


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
        _check_stalls(text, path.name)
        fixed = _drop_dangling_ready(text, path.name, record)
        if fixed == text:
            shutil.copyfile(path, out_dir / path.name)
        else:
            (out_dir / path.name).write_text(fixed)
    return record
