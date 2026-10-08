"""Generate a Vitis-HLS AXI project and a copy with native memory interfaces."""
import json
import argparse
import re
from pathlib import Path
import allo
from gemm import gemm_fp32
from schedule import build_schedule, PRIMITIVES
from build_blackbox import build as build_blackbox

HERE = Path(__file__).resolve().parent


def main():
    global HERE
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--partition-acc", action="store_true")
    parser.add_argument("--no-partition-acc", dest="partition_acc", action="store_false")
    parser.set_defaults(partition_acc=True)
    args = parser.parse_args()
    build_blackbox()
    HERE = args.output.resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule(partition_acc=args.partition_acc)
    (HERE / "gemm.mlir").write_text(str(schedule.module))
    project = HERE / "hls_axi"
    module = schedule.build(target="vitis_hls", mode="csyn", project=str(project),
                            wrap_io=False,
                            configs={"device": "pynqz2", "frequency": 100})
    code = module.hls_code
    # Lower only floating-point SSA binary operations to registered RTL calls.
    # Loop scheduling and array transformations have already been applied by Allo.
    calls = {"+": "hf_add_f32", "*": "hf_mul_f32"}
    counts = {"hf_add_f32": 0, "hf_mul_f32": 0}
    def lower(match):
        prefix, a, op, b = match.groups()
        name = calls[op]
        counts[name] += 1
        value = re.search(r"float\s+(\w+)", prefix).group(1)
        # Vitis 2024.1 needs this mapped auxiliary output for scalar returns.
        return f"bool hf_ready_{value};\n        {prefix}{name}({a}, {b}, hf_ready_{value});"
    code = re.sub(r"(float\s+\w+\s*=\s*)(\w+)\s*([+*])\s*(\w+)\s*;", lower, code)
    assert counts == {"hf_add_f32": 2, "hf_mul_f32": 1}, counts
    blackbox = Path(__file__).resolve().parent / "blackbox"
    code = '#include "' + str(blackbox / "hardfloat_model.h") + '"\n' + code
    (HERE / "kernel_axi.cpp").write_text(code)
    (project / "kernel.cpp").write_text(code)
    tcl = project / "run.tcl"
    text = tcl.read_text()
    text = text.replace("add_files kernel.cpp", "add_files kernel.cpp\n" + "\n".join(
        f"add_files -blackbox {{{blackbox / (name + '.json')}}}" for name in counts))
    tcl.write_text(text)
    # Preserve every computational statement; replace only interface pragmas.
    native = re.sub(r"^\s*#pragma HLS interface m_axi[^\n]*\n", "", code, flags=re.M)
    top_start = native.index("void gemm_fp32(")
    head, top = native[:top_start], native[top_start:]
    signature = top[:top.index(") {")]
    names = re.findall(r"(?:float|uint8_t)\s+\*?(\w+)", signature)
    assert len(names) == 5, signature
    for old, new in zip(names, ("A", "BT", "C", "use_c", "D")):
        top = re.sub(rf"\b{old}\b", new, top)
    for port in ("A", "BT", "C", "D"):
        top = top.replace(f"float *{port}", f"float {port}[64]")
    native = head + top
    marker = ") {"
    at = native.index(marker, native.index("void gemm_fp32(")) + len(marker)
    pragmas = "\n  #pragma HLS interface ap_ctrl_hs port=return\n"
    for port in ("A", "BT", "C", "D"):
        pragmas += f"  #pragma HLS interface ap_memory port={port} latency=1\n"
    pragmas += "  #pragma HLS interface ap_none port=use_c\n"
    native = native[:at] + pragmas + native[at:]
    project = HERE / "hls_native"
    project.mkdir(exist_ok=True)
    (project / "kernel.cpp").write_text(native)
    (project / "run.tcl").write_text("""open_project -reset out.prj
set_top gemm_fp32
add_files kernel.cpp
BLACKBOX_FILES
open_solution -reset solution1 -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10
config_compile -pipeline_loops 0
csynth_design
exit
""".replace("BLACKBOX_FILES", "\n".join(
        f"add_files -blackbox {{{blackbox / (name + '.json')}}}" for name in counts)))
    (HERE / "generation.json").write_text(json.dumps({
        "allo_source": str(Path(allo.__file__).resolve()),
        "target": "vitis_hls", "precision": "FP32, separate mul/add",
        "interface": "ap_memory + ap_ctrl_hs; AXI removed before HLS synthesis",
        "schedule": PRIMITIVES + (['partition(s.acc, dim=0)'] if args.partition_acc else []),
        "fp_implementation": "native HardFloat RTL blackboxes, latency=3, II=1",
        "fp_lowering_counts": counts,
    }, indent=2) + "\n")
    print(f"Generated {project / 'kernel.cpp'}")


if __name__ == "__main__":
    main()
