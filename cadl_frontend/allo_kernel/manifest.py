"""The interface manifest of a built kernel: what a wrapper needs to drive it.

Per parameter and per return value (ret0, ret1, ... in Python return order,
with the RTL port name Allo gave it) it records the Allo element type and shape, and for arrays the flattening
order, the bank partition, the number of ap_memory ports (over all banks)
and their read latency. Fields
that only the RTL can tell (direction, port count) are None until RTL exists;
the compute latency comes from the Vitis report and is None without one.

After Vitis runs, `check_interface_report` cross-checks the ports classified
from the Verilog against the report's interface summary (protocol, object,
direction, width, and address and data widths from the manifest), so a
misread interface fails here rather than in simulation.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from .interface import CONTROL as CONTROL_PORTS, HlsInstance, KernelParam
from .project import HLSProject

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


def check_interface_report(report: Path, instance: HlsInstance) -> None:
    """Raise ValueError where Vitis' interface summary disagrees with `instance`."""
    rows = {row.findtext("name"): row for row in ET.parse(report).getroot().iter("RtlPorts")}
    ports = {p.name: p for p in instance.ports}
    problems = [f"{name} only in {'the report' if name in rows else 'the RTL'}"
                for name in sorted(set(rows) ^ set(ports))]

    def expect(name: str, protocol: str, obj: str | None = None, bits: int | None = None):
        row, port = rows.get(name), ports.get(name)
        if row is None or port is None:
            return
        seen = (row.findtext("IOProtocol"), row.findtext("Dir"), int(row.findtext("Bits")))
        want = (protocol, "in" if port.direction == "input" else "out", bits or port.bits)
        if seen != want or (obj and row.findtext("Object") != obj):
            problems.append(f"{name}: report {seen} {row.findtext('Object')}, expected {want} {obj}")

    for name in ("ap_clk", "ap_rst", *CONTROL_PORTS):
        expect(name, "ap_ctrl_hs")
    params = {p.name: p for p in [*instance.params, *instance.returns]}
    for mem in instance.memories:
        param = params[mem.param]
        shape = param.partition.bank_shape(param.shape) if param.partition else param.shape
        widths = {"address": max(1, math.ceil(math.log2(math.prod(shape)))), "q": param.bits,
                  "d": param.bits, "ce": 1, "we": 1}
        obj = mem.param if mem.bank is None else f"{mem.param}_{mem.bank}"
        for role, port in mem.signals.items():
            expect(port.name, "ap_memory", obj, widths[role] if param.bits else None)
    for name in instance.scalars:
        expect(name, "ap_none", name)
    for name in instance.results:
        expect(name, "ap_vld", name)
        expect(HlsInstance.valid(name), "ap_vld", name)
    if problems:
        raise ValueError(f"{report}: interface report disagrees with the RTL: " + "; ".join(problems))


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
                    latency: dict | None, rtl: dict | None, provenance: dict | None = None) -> dict:
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
        "provenance": provenance,
    }
