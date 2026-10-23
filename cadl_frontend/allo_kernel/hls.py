"""Emit a native-interface Vitis HLS project from an Allo schedule.

Allo's `vivado_hls` target emits array arguments as C arrays and skips the
`vitis_hls` post-processing that adds m_axi pragmas, so no AXI is generated.
Interfaces are pinned with Tcl directives instead of editing the C++.

The HLS C++ is kept twice: kernel.orig.cpp as Allo emits the schedule, and
kernel.cpp after the float operations became blackbox calls (what Vitis runs).

Allo turns return values into trailing output arguments (an array, or a
pointer for a scalar), in Python return order. Its Vivado HLS emitter ignores
the `outputs` naming attribute, so they keep Allo's value names (v1, ...),
which are read back from the emitted signature. Array returns use ap_memory
like array arguments, scalar returns ap_vld.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from allo._mlir.dialects import arith as arith_d, func as func_d
from allo._mlir.ir import (F32Type, FlatSymbolRefAttr, FloatAttr, FunctionType, InsertionPoint,
                           IntegerAttr, MemRefType, StringAttr)

from .blackbox import CMPF_PREDICATES, OPERATORS, build_blackboxes
from .interface import KernelParam, Partition
from .project import HLSProject
from .rtl_fixups import fix_rtl

# f32 operation -> blackbox. Conversions are to/from i32; cmpf picks one per predicate.
FLOAT_LOWERING = {"arith.addf": "hf_add_f32", "arith.subf": "hf_sub_f32",
                  "arith.mulf": "hf_mul_f32", "arith.negf": "hf_neg_f32",
                  "math.absf": "hf_abs_f32", "arith.sitofp": "hf_sitofp_f32",
                  "arith.uitofp": "hf_uitofp_f32", "arith.fptosi": "hf_fptosi_f32",
                  "arith.fptoui": "hf_fptoui_f32", "arith.cmpf": None}
# Float operations without a blackbox, rejected rather than left to Vitis IP.
FLOAT_ARITH = {"arith.divf", "arith.remf", "math.fma", "arith.maximumf", "arith.minimumf",
               "arith.maxnumf", "arith.minnumf", "arith.extf", "arith.truncf", "math.sqrt",
               "math.exp", "math.log", "math.powf", "math.tanh"}


def _walk(op, pipelined: bool = False):
    """Nested operations, each with whether a loop around it is pipelined."""
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                inside = pipelined or "pipeline_ii" in child.operation.attributes
                yield child, inside
                yield from _walk(child.operation, inside)


PARTITION_KINDS = {0: "complete", 1: "block", 2: "cyclic"}  # allo.customize.Partition


def _partition(name: str, shape: tuple[int, ...], entries: list) -> Partition:
    """The one-dimension partition of top argument `name` from Allo's record."""
    if len(entries) != 1:
        raise NotImplementedError(f"{name}: partitions of several dimensions {entries}")
    kind, dim, factor = entries[0]
    kind = PARTITION_KINDS[kind]
    if dim == 0 and len(shape) == 1:
        dim = 1  # all dimensions of a 1-D array
    if not 1 <= dim <= len(shape):
        raise NotImplementedError(f"{name}: partition of all dimensions of {list(shape)}")
    if kind == "complete":
        factor = shape[dim - 1]
    if factor < 1 or shape[dim - 1] % factor:
        raise NotImplementedError(f"{name}: {kind} factor {factor} does not divide "
                                  f"dimension {dim} of {list(shape)}")
    return Partition(kind, dim, factor)


def name_ports(schedule) -> list[KernelParam]:
    """Keep the Python argument names on the HLS top function.

    An array is one row-major memory, or one per bank if partitioned on one
    dimension by a factor that divides it (see interface.Partition).
    """
    top = schedule.top_func
    args = schedule.func_args[schedule.top_func_name]
    names = [arg.name for arg in args]
    types = list(top.type.inputs)
    assert len(names) == len(types), (names, types)
    with schedule.module.context:
        top.attributes["inputs"] = StringAttr.get(",".join(names))
    params = []
    for arg, ty in zip(args, types):
        memory = MemRefType.isinstance(ty)
        shape = tuple(MemRefType(ty).shape) if memory else ()
        entries = schedule.partitioned_arrays.get(f"{schedule.top_func_name}:{arg.name}")
        params.append(KernelParam(arg.name, "ap_memory" if memory else "ap_none",
                                  str(arg.dtype), arg.dtype.bits, shape,
                                  _partition(arg.name, shape, entries) if entries else None))
    return params


def _element(ty, signedness: str) -> tuple[str, int]:
    """Allo-style element type name and width of an MLIR type, e.g. ("ui8", 8)."""
    name = str(MemRefType(ty).element_type if MemRefType.isinstance(ty) else ty)
    match = re.fullmatch(r"(i|f|bf)(\d+)", name)
    if not match:
        raise NotImplementedError(f"return element type {name}")
    return ("u" if signedness == "u" else "") + name, int(match[2])


def _signature(code: str, top: str) -> list[str]:
    """Argument names of the top function in emitted HLS C++."""
    match = re.search(rf"^void {re.escape(top)}\(\n([^()]*?)\n\) {{", code, re.M)
    if not match:
        raise ValueError(f"no definition of {top} in the HLS code")
    return [re.match(r"[^\[]*?\*?(\w+)(?:\[|$)", arg.strip())[1]
            for arg in match[1].split(",\n")]


def find_returns(schedule, code: str, inputs: int) -> list[KernelParam]:
    """The output arguments Allo emitted for the return values, in return order."""
    top = schedule.top_func
    results = list(top.type.results)
    names = _signature(code, schedule.top_func_name)
    if len(names) != inputs + len(results):
        raise ValueError(f"{schedule.top_func_name}: {len(names)} HLS arguments for "
                         f"{inputs} inputs and {len(results)} return values")
    signs = StringAttr(top.attributes["otypes"]).value if "otypes" in top.attributes else ""
    returns = []
    for i, (name, ty) in enumerate(zip(names[inputs:], results)):
        dtype, bits = _element(ty, signs[i] if i < len(signs) else "")
        memory = MemRefType.isinstance(ty)
        returns.append(KernelParam(name, "ap_memory" if memory else "ap_vld", dtype, bits,
                                   tuple(MemRefType(ty).shape) if memory else ()))
    return returns


def _blackbox(op) -> str:
    """The blackbox for float operation `op`, which must be on f32 (and i32)."""
    name = op.operation.name
    callee = FLOAT_LOWERING[name]
    types = [str(v.type) for v in op.operands] + [str(r.type) for r in op.results]
    if name == "arith.cmpf":
        predicate = CMPF_PREDICATES[IntegerAttr(op.attributes["predicate"]).value]
        callee = f"hf_cmp_{predicate}_f32"
        if callee not in OPERATORS:
            raise NotImplementedError(f"cmpf {predicate} at {op.location}")
    want = [("i32" if ty.startswith(("int", "uint")) else "i1" if ty == "bool" else "f32")
            for ty in (*OPERATORS[callee].args, OPERATORS[callee].ret)]
    if types != want:
        raise NotImplementedError(f"{name} on {types} at {op.location}; HardFloat "
                                  f"blackboxes take {want}")
    return callee


def _constant_int(value) -> int | None:
    """The integer of an arith.constant i32 operand, else None."""
    owner = value.owner
    if getattr(owner, "name", None) != "arith.constant" or str(value.type) != "i32":
        return None
    return IntegerAttr(owner.attributes["value"]).value


def _fold_conversions(module) -> None:
    """Replace sitofp/uitofp of an integer constant by the f32 constant (e.g. `v: float32 = 0`).

    Python's float() of a 32-bit integer rounds to nearest even once, as the
    blackbox would; the constant needs no hardware and no pipelined loop.
    """
    for op, _ in list(_walk(module.operation)):
        name = op.operation.name
        if name in ("arith.sitofp", "arith.uitofp") and str(op.results[0].type) == "f32":
            value = _constant_int(op.operands[0])
            if value is None:
                continue
            value &= 0xFFFFFFFF
            if name == "arith.sitofp" and value >> 31:
                value -= 1 << 32
            with InsertionPoint(op), op.location:
                const = arith_d.ConstantOp(F32Type.get(), FloatAttr.get(F32Type.get(), float(value)))
            op.results[0].replace_all_uses_with(const.result)
            op.operation.erase()


def lower_float_ops(schedule) -> dict[str, int]:
    """Replace f32 operations with calls to the HardFloat blackboxes (blackbox.OPERATORS).

    Vitis accepts the ap_ctrl_none blackboxes only in pipelined regions
    (HLS 200-1918), so float arithmetic outside a pipelined loop is rejected;
    integer constants converted to f32 are folded first.
    """
    module = schedule.module
    counts = {name: 0 for name in OPERATORS}
    signatures = {}
    with module.context, schedule.top_func.location:
        _fold_conversions(module)
        targets = []
        for op, pipelined in _walk(module.operation):
            name = op.operation.name
            if name in FLOAT_LOWERING:
                callee = _blackbox(op)
                if not pipelined:
                    raise NotImplementedError(
                        f"{name} at {op.location} is outside a pipelined loop; the HardFloat "
                        "blackboxes need one (Vitis HLS 200-1918), e.g. s.pipeline(<loop>)")
                targets.append((op, callee))
            elif name in FLOAT_ARITH:
                raise NotImplementedError(f"no HardFloat blackbox for {op}")
        for op, callee in targets:
            result = op.results[0].type
            with InsertionPoint(op), op.location:
                call = func_d.CallOp([result], FlatSymbolRefAttr.get(callee), list(op.operands))
            signatures[callee] = FunctionType.get([v.type for v in op.operands], [result])
            op.results[0].replace_all_uses_with(call.result)
            op.operation.erase()
            counts[callee] += 1
        with InsertionPoint.at_block_begin(module.body):
            for callee, signature in signatures.items():
                decl = func_d.FuncOp(callee, signature)
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
    """Lower the schedule in place; write kernel.orig.cpp, kernel.cpp, blackboxes, run.tcl."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ports = name_ports(schedule)
    # Allo emits from a copy of the module, so this leaves the schedule as it is.
    (directory / "kernel.orig.cpp").write_text(schedule.build(target="vivado_hls").hls_code)
    calls = lower_float_ops(schedule)
    blackboxes = build_blackboxes(directory / "blackbox", list(calls))
    code = schedule.build(target="vivado_hls").hls_code
    returns = find_returns(schedule, code, len(ports))
    if calls:
        code = '#include "blackbox/hardfloat_model.h"\n' + code
    (directory / "kernel.cpp").write_text(code)
    (directory / "run.tcl").write_text(
        _tcl(schedule.top_func_name, ports + returns, blackboxes, part, clock_ns, memory_latency))
    return HLSProject(schedule.top_func_name, directory, ports, calls, memory_latency, returns)


def run_csynth(project: HLSProject, vitis_hls: str = "vitis_hls") -> Path:
    """Run C synthesis and repair the RTL; return the directory of usable Verilog."""
    with open(project.directory / "csynth.log", "w") as log:
        subprocess.run([vitis_hls, "-f", "run.tcl"], cwd=project.directory,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    rtl = project.directory / "rtl"
    record = fix_rtl(project.directory / "out.prj/solution1/syn/verilog", rtl)
    (project.directory / "rtl_fixups.json").write_text(json.dumps(record, indent=2) + "\n")
    return rtl
