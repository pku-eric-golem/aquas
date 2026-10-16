"""Native HardFloat FP32 stages packaged as Vitis HLS RTL blackboxes.

Allo's HLS emitter turns a scalar `func.call` result into a pointer argument,
`hf_add_f32(a, b, &r)`. Blackboxing that shape directly makes Vitis see a
loop-carried dependence on `r` (II 1 -> 2), so the header provides inline
pointer-form adapters around the scalar-return blackbox `hf_<op>_f32_rtl`.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Function name -> HFNativePipeline OP code and the C expression for the model.
OPERATORS = {
    "hf_add_f32": (0, "a + b"),
    "hf_mul_f32": (2, "a * b"),
}
LATENCY = 3

HEADER = """#ifndef ALLO_HARDFLOAT_MODEL_H
#define ALLO_HARDFLOAT_MODEL_H
#include <cstdint>
#include <cstring>

// The native RISC-V specialization returns canonical NaNs. C models are used
// only by C simulation; the blackbox RTL implements the arithmetic.
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
             "isSigNaNRecFN.v", "fNToRecFN.v", "recFNToFN.v", "mulAddRecFN.v"]
    bundle = "// Pinned HardFloat Release 1, native RISC-V FP32 pipeline.\n"
    bundle += "\n".join(package.expand_includes(package.SOURCE / name) for name in names)
    source = (ROOT / "hardware/hardfloat/HFNativePipeline.sv").read_text()
    assert source.count("always @(posedge clock)") == 5
    # The only datapath adaptation: a clock enable on the existing registers.
    native = source.replace("module HFNativePipeline #", "module HFNativePipelineCE #")
    native = native.replace("input wire clock,", "input wire clock, clock_enable,")
    native = native.replace("always @(posedge clock)", "always @(posedge clock) if(clock_enable)")
    return bundle + "\n" + native


def _adapter(name: str) -> str:
    return f"""extern "C" float {name}_rtl(float a, float b, bool &ap_ready);
static inline void {name}(float a, float b, float *result) {{
#pragma HLS inline
    bool ready;
    *result = {name}_rtl(a, b, ready);
}}"""


def _rtl(name: str, op: int) -> str:
    return f"""module {name}(
 input wire ap_clk, ap_rst, ap_ce,
 input wire [31:0] a, b,
 output wire [31:0] result,
 output wire ap_ready
);
 wire [36:0] payload;
 HFNativePipelineCE #(.WIDTH(32), .OP({op}), .LATENCY({LATENCY})) arithmetic(
  .clock(ap_clk), .clock_enable(ap_ce && !ap_rst),
  .a(a), .b(b), .c(32'b0), .rm(3'b000), .payload(payload));
 assign result = ap_rst ? 32'b0 : payload[31:0];
 // Vitis 2024.1 wires a scalar-return blackbox call only with an extra
 // output; this constant is that auxiliary port, not a control signal.
 assign ap_ready = 1'b1;
endmodule
"""


def _model(name: str, expr: str) -> str:
    return f"""#include "hardfloat_model.h"
extern "C" float {name}_rtl(float a, float b, bool &ap_ready) {{
    #pragma HLS interface ap_ctrl_none port=return
    ap_ready = true;
    volatile float value = {expr};
    return hf_canonical(value);
}}
"""


def _config(name: str, out: Path, with_bundle: bool) -> dict:
    return {
        "c_function_name": f"{name}_rtl",
        "rtl_top_module_name": name,
        "c_files": [{"c_file": str(out / f"{name}.cpp"),
                     "cflag": "-fno-fast-math -ffp-contract=off"}],
        "rtl_files": ([str(out / "HardFloatNative.v")] if with_bundle else [])
                     + [str(out / f"{name}.v")],
        "c_parameters": [
            {"c_name": "a", "c_port_direction": "in", "rtl_ports": {"data_read_in": "a"}},
            {"c_name": "b", "c_port_direction": "in", "rtl_ports": {"data_read_in": "b"}},
            {"c_name": "ap_ready", "c_port_direction": "out",
             "rtl_ports": {"data_write_out": "ap_ready"}},
        ],
        "c_return": {"c_port_direction": "out", "rtl_ports": {"data_write_out": "result"}},
        "rtl_common_signal": {
            "module_clock": "ap_clk", "module_reset": "ap_rst", "module_clock_enable": "ap_ce",
            **{f"ap_ctrl_chain_protocol_{x}": "" for x in
               ("idle", "start", "ready", "done", "continue")},
        },
        "rtl_performance": {"latency": str(LATENCY), "II": "1"},
        # Placeholders, not ASIC characterization.
        "rtl_resource_usage": {x: "0" for x in ("FF", "LUT", "BRAM", "URAM", "DSP")},
    }


def build_blackboxes(out: Path, names: list[str]) -> list[Path]:
    """Write models, RTL and Vitis blackbox JSON for `names`; return the JSON paths."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "HardFloatNative.v").write_text(_native_bundle(_packager()))
    adapters = "\n".join(_adapter(name) for name in OPERATORS)
    (out / "hardfloat_model.h").write_text(HEADER.replace("PROTOTYPES", adapters))
    configs = []
    for index, name in enumerate(names):
        op, expr = OPERATORS[name]
        (out / f"{name}.v").write_text(_rtl(name, op))
        (out / f"{name}.cpp").write_text(_model(name, expr))
        # HardFloatNative.v may be listed by only one blackbox, or Vitis
        # sees its modules defined twice.
        config = _config(name, out, with_bundle=index == 0)
        path = out / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        configs.append(path)
    return configs
