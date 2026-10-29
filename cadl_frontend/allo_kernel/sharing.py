"""Check that Vitis bound every HardFloat blackbox call to an instance of its own.

In a loop with II > 1 Vitis may bind several calls of one blackbox function to
a single instance. Its RTL then can give one call's in-flight operand hold
priority over another call's issue cycle, so that call computes on the other
one's operands: two float accumulations in one row-sum loop (II 3) shared an
adder, and the sum came out wrong while the sum of squares was right
(2026-10-29). blackbox.py gives every call site its own blackbox; this check
also catches calls that Vitis itself duplicates, e.g. when unrolling.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

# `ST_4 : Operation 57 [4/4] ... "%v11 = call i32 @hf_add_f32_s0_rtl, ...`: one line per state.
CALL = re.compile(r'"%(\w+) = call \S+ @(hf_\w+_rtl)\b')
# `|grp_hf_add_f32_s0_rtl_fu_114  |hf_add_f32_s0_rtl  |...`: the instance tables.
INSTANCE = re.compile(r"^\s*\|\s*(grp_hf_\w+?)\s*\|\s*(hf_\w+_rtl)\s*\|", re.M)


def shared_blackboxes(report: str) -> dict[str, tuple[set[str], set[str]]]:
    """Blackbox functions of one verbose.bind.rpt with more calls than instances."""
    calls, instances = defaultdict(set), defaultdict(set)
    for result, function in CALL.findall(report):
        calls[function].add(result)
    for instance, function in INSTANCE.findall(report):
        instances[function].add(instance)
    return {f: (calls[f], instances[f]) for f in calls if len(calls[f]) > len(instances[f])}


def check_unshared(db: Path) -> None:
    """Raise if any module of the Vitis project in `db` shares a blackbox instance."""
    reports = sorted(Path(db).glob("*.verbose.bind.rpt"))
    if not reports:
        raise RuntimeError(f"no Vitis binding reports (*.verbose.bind.rpt) in {db}")
    problems = []
    for path in reports:
        module = path.name.removesuffix(".verbose.bind.rpt")
        for function, (calls, instances) in sorted(shared_blackboxes(path.read_text()).items()):
            problems.append(f"{module}: calls {', '.join(sorted(calls))} of {function} "
                            f"share {len(instances)} instance(s)")
    if problems:
        raise RuntimeError("Vitis bound several HardFloat calls to one blackbox instance, "
                           "whose RTL computes wrong results: " + "; ".join(problems))
