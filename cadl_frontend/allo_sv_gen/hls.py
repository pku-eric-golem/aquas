"""Emit a native-interface Vitis HLS project from an Allo schedule.

Allo's `vivado_hls` target emits array arguments as C arrays and skips the
`vitis_hls` post-processing that adds m_axi pragmas, so no AXI is generated.
Interfaces are pinned with Tcl directives instead of editing the C++.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from allo._mlir.dialects import func as func_d
from allo._mlir.ir import (F32Type, FlatSymbolRefAttr, FunctionType, InsertionPoint,
                           MemRefType, StringAttr)

from .blackbox import OPERATORS, build_blackboxes
from .instantiate import KernelParam
from .rtl_fixups import fix_rtl

FLOAT_LOWERING = {"arith.addf": "hf_add_f32", "arith.mulf": "hf_mul_f32"}
FLOAT_ARITH = {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.remf",
               "arith.negf", "math.fma"}


@dataclass
class HLSProject:
    top: str
    directory: Path
    params: list[KernelParam]
    float_calls: dict[str, int]


def _walk(op):
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _walk(child.operation)


def name_ports(schedule) -> list[KernelParam]:
    """Keep the Python argument names on the HLS top function."""
    top = schedule.top_func
    names = [arg.name for arg in schedule.func_args[schedule.top_func_name]]
    types = list(top.type.inputs)
    assert len(names) == len(types), (names, types)
    with schedule.module.context:
        top.attributes["inputs"] = StringAttr.get(",".join(names))
    return [KernelParam(name, "ap_memory" if MemRefType.isinstance(ty) else "ap_none")
            for name, ty in zip(names, types)]


def lower_float_ops(schedule) -> dict[str, int]:
    """Replace f32 add/mul with calls to the HardFloat blackboxes."""
    module = schedule.module
    counts = {name: 0 for name in OPERATORS}
    with module.context, schedule.top_func.location:
        f32 = F32Type.get()
        targets = []
        for op in _walk(module.operation):
            name = op.operation.name
            if name in FLOAT_LOWERING and op.result.type == f32:
                targets.append(op)
            elif name in FLOAT_ARITH:
                raise NotImplementedError(f"no HardFloat blackbox for {op}")
        for op in targets:
            callee = FLOAT_LOWERING[op.operation.name]
            with InsertionPoint(op), op.location:
                call = func_d.CallOp([f32], FlatSymbolRefAttr.get(callee), list(op.operands))
            op.result.replace_all_uses_with(call.result)
            op.operation.erase()
            counts[callee] += 1
        with InsertionPoint.at_block_begin(module.body):
            for callee, count in counts.items():
                if count:
                    decl = func_d.FuncOp(callee, FunctionType.get([f32, f32], [f32]))
                    decl.attributes["sym_visibility"] = StringAttr.get("private")
    return {name: count for name, count in counts.items() if count}


def _tcl(top: str, ports: list[KernelParam], blackboxes: list[Path], part: str,
         clock_ns: float, memory_latency: int) -> str:
    lines = ["open_project -reset out.prj", f"set_top {top}", "add_files kernel.cpp"]
    lines += [f"add_files -blackbox {{{path}}}" for path in blackboxes]
    lines += ["open_solution -reset solution1 -flow_target vivado",
              f"set_part {{{part}}}", f"create_clock -period {clock_ns}",
              "config_compile -pipeline_loops 0",
              f"set_directive_interface -mode ap_ctrl_hs {top}"]
    for port in ports:
        latency = f" -latency {memory_latency}" if port.kind == "ap_memory" else ""
        lines.append(f"set_directive_interface -mode {port.kind}{latency} {top} {port.name}")
    lines += ["csynth_design", "exit"]
    return "\n".join(lines) + "\n"


def emit_project(schedule, directory: Path, *, part: str = "xc7z020clg400-1",
                 clock_ns: float = 10, memory_latency: int = 1) -> HLSProject:
    """Lower the schedule in place and write kernel.cpp, blackboxes and run.tcl."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ports = name_ports(schedule)
    calls = lower_float_ops(schedule)
    blackboxes = build_blackboxes(directory / "blackbox", list(calls))
    code = schedule.build(target="vivado_hls").hls_code
    if calls:
        code = '#include "blackbox/hardfloat_model.h"\n' + code
    (directory / "kernel.cpp").write_text(code)
    (directory / "run.tcl").write_text(
        _tcl(schedule.top_func_name, ports, blackboxes, part, clock_ns, memory_latency))
    return HLSProject(schedule.top_func_name, directory, ports, calls)


def run_csynth(project: HLSProject, vitis_hls: str = "vitis_hls") -> Path:
    """Run C synthesis and repair the RTL; return the directory of usable Verilog."""
    with open(project.directory / "csynth.log", "w") as log:
        subprocess.run([vitis_hls, "-f", "run.tcl"], cwd=project.directory,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    rtl = project.directory / "rtl"
    record = fix_rtl(project.directory / "out.prj/solution1/syn/verilog", rtl)
    (project.directory / "rtl_fixups.json").write_text(json.dumps(record, indent=2) + "\n")
    return rtl
