"""Native HardFloat FP32 stages packaged as Vitis HLS RTL blackboxes.

Allo's HLS emitter turns a scalar `func.call` result into a pointer argument,
`hf_add_f32(a, b, &r)`. Blackboxing that shape directly makes Vitis see a
loop-carried dependence on `r` (II 1 -> 2), so the header provides inline
pointer-form adapters around the scalar-return blackbox `hf_<op>_f32_rtl`.

The blackbox has no output besides its return value. The archived manual flow
also declared a constant `bool &ap_ready` output; Vitis then stalls the calling
pipeline waiting for that output forever, which needed a stall-predicate patch.
Without it no stall appears (GEMM and the Allo library systolic array).
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class FloatOp:
    """A HardFloat operation packaged as blackbox `<name>_rtl`."""
    op: int  # HFNativePipeline OP code
    latency: int  # cycles, from hardware/hardfloat/ip_catalog.yaml
    args: tuple[str, ...]  # C argument types
    ret: str  # C return type
    expr: str  # C model
    pred: int = 0  # HFCompute comparison predicate
    rm: int = 0  # HardFloat rounding mode: 0 nearest even, 1 toward zero

    @property
    def result_bits(self) -> int:
        return 1 if self.ret == "bool" else 32


FF, F = ("float", "float"), ("float",)
OPERATORS = {
    "hf_add_f32": FloatOp(0, 3, FF, "float", "a + b"),
    "hf_sub_f32": FloatOp(1, 3, FF, "float", "a - b"),
    "hf_mul_f32": FloatOp(2, 3, FF, "float", "a * b"),
    "hf_sitofp_f32": FloatOp(7, 2, ("int32_t",), "float", "(float)a"),
    "hf_uitofp_f32": FloatOp(8, 2, ("uint32_t",), "float", "(float)a"),
    # C conversions truncate; out of range, HardFloat saturates as RISC-V fcvt does.
    "hf_fptosi_f32": FloatOp(9, 1, F, "int32_t", "(int32_t)a", rm=1),
    "hf_fptoui_f32": FloatOp(10, 1, F, "uint32_t", "(uint32_t)a", rm=1),
    "hf_neg_f32": FloatOp(13, 1, F, "float", "-a"),
    "hf_abs_f32": FloatOp(14, 1, F, "float", "__builtin_fabsf(a)"),
}
# arith.cmpf predicates (MLIR CmpFPredicate order) -> HFCompute PRED and the C model.
COMPARISONS = {
    "oeq": (1, "a == b"), "ogt": (2, "a > b"), "oge": (3, "a >= b"), "olt": (4, "a < b"),
    "ole": (5, "a <= b"), "one": (6, "a < b || a > b"), "ord": (7, "a == a && b == b"),
    "ueq": (9, "!(a < b || a > b)"), "ugt": (10, "!(a <= b)"), "uge": (11, "!(a < b)"),
    "ult": (12, "!(a >= b)"), "ule": (13, "!(a > b)"), "une": (14, "a != b"),
    "uno": (8, "a != a || b != b"),
}
CMPF_PREDICATES = ["false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord",
                   "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true"]
OPERATORS |= {f"hf_cmp_{name}_f32": FloatOp(6, 1, FF, "bool", expr, pred=pred)
              for name, (pred, expr) in COMPARISONS.items()}
LATENCY = 3  # add, sub and mul

HEADER = """#ifndef ALLO_HARDFLOAT_MODEL_H
#define ALLO_HARDFLOAT_MODEL_H
#include <cstdint>
#include <cstring>

// The native RISC-V specialization returns canonical NaNs. C models are used
// only by C simulation; the blackbox RTL implements the arithmetic.
static inline bool hf_canonical(bool value) { return value; }
static inline int32_t hf_canonical(int32_t value) { return value; }
static inline uint32_t hf_canonical(uint32_t value) { return value; }
static inline float hf_canonical(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        bits = 0x7fc00000U;
        std::memcpy(&value, &bits, sizeof(bits));
    }
    return value;
}
PROTOTYPES
#endif
"""


def _packager():
    spec = importlib.util.spec_from_file_location(
        "hardfloat_packager", ROOT / "hardware/hardfloat/build-hardfloat-ip.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    package.verify_sources()
    return package


def _native_bundle(package) -> str:
    names = ["HardFloat_primitives.v", "RISCV/HardFloat_specialize.v", "HardFloat_rawFN.v",
             "isSigNaNRecFN.v", "fNToRecFN.v", "recFNToFN.v", "addRecFN.v", "mulRecFN.v",
             "mulAddRecFN.v", "compareRecFN.v", "iNToRecFN.v", "recFNToIN.v", "recFNToRecFN.v"]
    bundle = "// Pinned HardFloat Release 1, native RISC-V FP32 pipeline.\n"
    bundle += "\n".join(package.expand_includes(package.SOURCE / name) for name in names)
    # The single-cycle operations of HFNativePipeline (compare, convert, sign).
    bundle += "\n" + (ROOT / "hardware/hardfloat/HFCompute.sv").read_text()
    source = (ROOT / "hardware/hardfloat/HFNativePipeline.sv").read_text()
    assert source.count("always @(posedge clock)") == 5
    # The only datapath adaptation: a clock enable on the existing registers.
    native = source.replace("module HFNativePipeline #", "module HFNativePipelineCE #")
    native = native.replace("input wire clock,", "input wire clock, clock_enable,")
    native = native.replace("always @(posedge clock)", "always @(posedge clock) if(clock_enable)")
    return bundle + "\n" + native


def _params(op: FloatOp) -> str:
    return ", ".join(f"{ty} {arg}" for ty, arg in zip(op.args, "ab"))


def _adapter(name: str, op: FloatOp) -> str:
    args = ", ".join("ab"[:len(op.args)])
    return f"""extern "C" {op.ret} {name}_rtl({_params(op)});
static inline void {name}({_params(op)}, {op.ret} *result) {{
#pragma HLS inline
    *result = {name}_rtl({args});
}}"""


def _rtl(name: str, op: FloatOp) -> str:
    # Vitis instantiates the blackbox under its C function name.
    inputs = ", ".join("ab"[:len(op.args)])
    b = "b" if len(op.args) == 2 else "32'b0"
    bits = op.result_bits
    return f"""module {name}_rtl(
 input wire ap_clk, ap_rst, ap_ce,
 input wire [31:0] {inputs},
 output wire [{bits - 1}:0] result
);
 // Vitis 2024.1 may still wire an unread ap_ready pin to some instances;
 // rtl_fixups.py drops it.
 wire [36:0] payload;
 HFNativePipelineCE #(.WIDTH(32), .OP({op.op}), .PRED({op.pred}), .LATENCY({op.latency})) arithmetic(
  .clock(ap_clk), .clock_enable(ap_ce && !ap_rst),
  .a(a), .b({b}), .c(32'b0), .rm(3'd{op.rm}), .payload(payload));
 assign result = ap_rst ? {bits}'b0 : payload[{bits - 1}:0];
endmodule
"""


def _model(name: str, op: FloatOp) -> str:
    return f"""#include "hardfloat_model.h"
extern "C" {op.ret} {name}_rtl({_params(op)}) {{
    #pragma HLS interface ap_ctrl_none port=return
    volatile {op.ret} value = {op.expr};
    return hf_canonical(value);
}}
"""


def _config(name: str, op: FloatOp, out: Path, with_bundle: bool) -> dict:
    return {
        "c_function_name": f"{name}_rtl",
        "rtl_top_module_name": f"{name}_rtl",
        "c_files": [{"c_file": str(out / f"{name}.cpp"),
                     "cflag": "-fno-fast-math -ffp-contract=off"}],
        "rtl_files": ([str(out / "HardFloatNative.v")] if with_bundle else [])
                     + [str(out / f"{name}.v")],
        "c_parameters": [{"c_name": arg, "c_port_direction": "in",
                          "rtl_ports": {"data_read_in": arg}} for arg in "ab"[:len(op.args)]],
        "c_return": {"c_port_direction": "out", "rtl_ports": {"data_write_out": "result"}},
        "rtl_common_signal": {
            "module_clock": "ap_clk", "module_reset": "ap_rst", "module_clock_enable": "ap_ce",
            **{f"ap_ctrl_chain_protocol_{x}": "" for x in
               ("idle", "start", "ready", "done", "continue")},
        },
        "rtl_performance": {"latency": str(op.latency), "II": "1"},
        # Placeholders, not ASIC characterization.
        "rtl_resource_usage": {x: "0" for x in ("FF", "LUT", "BRAM", "URAM", "DSP")},
    }


def build_blackboxes(out: Path, names: list[str]) -> list[Path]:
    """Write models, RTL and Vitis blackbox JSON for `names`; return the JSON paths."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "HardFloatNative.v").write_text(_native_bundle(_packager()))
    adapters = "\n".join(_adapter(name, op) for name, op in OPERATORS.items())
    (out / "hardfloat_model.h").write_text(HEADER.replace("PROTOTYPES", adapters))
    configs = []
    for index, name in enumerate(names):
        op = OPERATORS[name]
        (out / f"{name}.v").write_text(_rtl(name, op))
        (out / f"{name}.cpp").write_text(_model(name, op))
        # HardFloatNative.v may be listed by only one blackbox, or Vitis
        # sees its modules defined twice.
        config = _config(name, op, out, with_bundle=index == 0)
        path = out / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        configs.append(path)
    return configs
