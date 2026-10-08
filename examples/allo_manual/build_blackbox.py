"""Register the repository's native HardFloat stages as Vitis RTL blackboxes.

The only datapath adaptation is a clock enable on existing registers. No
registers are added, removed, moved, or inserted to match a Xilinx FP latency.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "blackbox"


def build():
    OUT.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("hardfloat_packager", ROOT / "hardware/hardfloat/build-hardfloat-ip.py")
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    package.verify_sources()
    names = ["HardFloat_primitives.v", "RISCV/HardFloat_specialize.v", "HardFloat_rawFN.v",
             "isSigNaNRecFN.v", "fNToRecFN.v", "recFNToFN.v", "mulAddRecFN.v"]
    bundle = "// Pinned HardFloat Release 1, native RISC-V FP32 pipeline.\n"
    bundle += "\n".join(package.expand_includes(package.SOURCE / name) for name in names)
    source = (ROOT / "hardware/hardfloat/HFNativePipeline.sv").read_text()
    assert source.count("always @(posedge clock)") == 5
    native = source.replace("module HFNativePipeline #", "module HFNativePipelineCE #")
    native = native.replace("input wire clock,", "input wire clock, clock_enable,")
    native = native.replace("always @(posedge clock)", "always @(posedge clock) if(clock_enable)")
    bundle += "\n" + native
    (OUT / "HardFloatNative.v").write_text(bundle)
    contracts = []
    for op, name in ((0, "hf_add_f32"), (2, "hf_mul_f32")):
        timing = package.native_timing(32, op)
        assert timing["latency_upper_bound"] == 3 and timing["no_stall_II"] == 1
        rtl = f'''// Same native stage boundaries, with HLS CE and reset adaptation.
module {name}(
 input wire ap_clk, ap_rst, ap_ce,
 input wire [31:0] a, b,
 output wire [31:0] result,
 output wire ap_ready
);
 wire [36:0] payload;
 HFNativePipelineCE #(.WIDTH(32), .OP({op}), .LATENCY(3)) arithmetic(
  .clock(ap_clk), .clock_enable(ap_ce && !ap_rst),
  .a(a), .b(b), .c(32'b0), .rm(3'b000), .payload(payload));
 // HLS tracks result validity. Three enabled edges refill native stages.
 assign result = ap_rst ? 32'b0 : payload[31:0];
 // Vitis 2024.1 connects this unused pin for a scalar-return call.
 // Register it as an auxiliary scalar output, not a chain control signal.
 assign ap_ready = 1'b1;
endmodule
'''
        (OUT / (name + ".v")).write_text(rtl)
        config = {
            "c_function_name": name, "rtl_top_module_name": name,
            "c_files": [{"c_file": str(OUT / (name + ".cpp")), "cflag": "-fno-fast-math -ffp-contract=off"}],
            "rtl_files": ([str(OUT / "HardFloatNative.v")] if op == 0 else []) + [str(OUT / (name + ".v"))],
            "c_parameters": [{"c_name": x, "c_port_direction": "in", "rtl_ports": {"data_read_in": x}} for x in ("a", "b")]
                            + [{"c_name": "ap_ready", "c_port_direction": "out", "rtl_ports": {"data_write_out": "ap_ready"}}],
            "c_return": {"c_port_direction": "out", "rtl_ports": {"data_write_out": "result"}},
            "rtl_common_signal": {"module_clock": "ap_clk", "module_reset": "ap_rst", "module_clock_enable": "ap_ce",
                                  **{f"ap_ctrl_chain_protocol_{x}": "" for x in ("idle", "start", "ready", "done", "continue")}},
            "rtl_performance": {"latency": "3", "II": "1"},
            # These placeholders are not ASIC characterization or area estimates.
            "rtl_resource_usage": {x: "0" for x in ("FF", "LUT", "BRAM", "URAM", "DSP")},
        }
        (OUT / (name + ".json")).write_text(json.dumps(config, indent=2) + "\n")
        contracts.append({"function": name, "operation": op, "latency": 3, "II": 1})
    metadata = {"mode": "Vitis RTL blackbox, native HardFloat", "contracts": contracts,
                "source": "hardware/hardfloat/HFNativePipeline.sv",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "datapath_adaptation": "CE guards on existing register boundaries; no added pipeline stages",
                "protocol": "ap_ctrl_none, wire inputs/return, global CE",
                "auxiliary_output": "constant bool ap_ready registered as a C output argument to match Vitis 2024.1 scalar-return call wiring",
                "reset": "mask result and freeze arithmetic; HLS validity pipeline handles flushing",
                "rounding": "RNE", "nan": "RISC-V canonical", "tininess": "after rounding",
                "no_xilinx_fp_ip": True, "post_hls_leaf_replacement": False,
                "area_timing": "blackbox resource entries are placeholders; ASIC STA required"}
    (OUT / "contracts.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
