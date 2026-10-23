"""The interface manifest of a built kernel: what a wrapper needs to drive it.

Per parameter and per return value (ret0, ret1, ... in Python return order,
with the RTL port name Allo gave it) it records the Allo element type and shape, and for arrays the flattening
order, the bank partition, the number of ap_memory ports (over all banks)
and their read latency. Fields
that only the RTL can tell (direction, port count) are None until RTL exists;
the compute latency comes from the Vitis report and is None without one.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from .hls import HLSProject
from .interface import HlsInstance, KernelParam

CONTROL = "ap_ctrl_hs"  # hls.py pins it on every top


def read_latency(report: Path) -> dict | None:
    """Best/worst-case cycles from start to done, per csynth.xml; None if unknown."""
    if not report.exists():
        return None
    summary = ET.parse(report).getroot().find("PerformanceEstimates/SummaryOfOverallLatency")
    cycles = {}
    for key, tag in (("best", "Best-caseLatency"), ("worst", "Worst-caseLatency")):
        text = summary.findtext(tag) if summary is not None else None
        cycles[key] = int(text) if text and text.isdigit() else None
    return cycles


def _direction(param: KernelParam, instance: HlsInstance | None) -> str | None:
    if instance is None:
        return None
    if param.kind == "ap_none":
        return "in"
    if param.kind == "ap_vld":
        return "out"
    roles = {role for m in instance.memories if m.param == param.name for role in m.signals}
    reads, writes = "q" in roles, "d" in roles
    return "inout" if reads and writes else "in" if reads else "out" if writes else None


def _param(param: KernelParam, instance: HlsInstance | None, memory_latency: int) -> dict:
    entry = {"name": param.name, "interface": param.kind,
             "direction": _direction(param, instance),
             "dtype": param.dtype, "elem_bits": param.bits, "shape": list(param.shape)}
    if param.kind == "ap_memory":
        ports = None if instance is None else sum(m.param == param.name for m in instance.memories)
        part = param.partition
        entry.update(elements=math.prod(param.shape), layout="row-major",
                     partition=part and {"type": part.kind, "dim": part.dim, "factor": part.factor},
                     banks=param.banks, ports=ports, read_latency=memory_latency)
    return entry


def kernel_manifest(project: HLSProject, instance: HlsInstance | None,
                    latency: dict | None, rtl: dict | None) -> dict:
    return {
        "top": project.top,
        "control": CONTROL,
        "latency": latency,
        "params": [_param(p, instance, project.memory_latency) for p in project.params],
        "returns": [{"name": f"ret{i}", "port": r.name}
                    | {k: v for k, v in _param(r, instance, project.memory_latency).items()
                       if k != "name"} for i, r in enumerate(project.returns)],
        "float_calls": project.float_calls,
        "rtl": rtl,
    }
