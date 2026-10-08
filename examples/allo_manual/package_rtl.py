"""Generate APS-compatible main and connect the actual native HLS port list."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RTL = HERE / "rtl"


def main():
    scala = (ROOT / "thirdparty/chipyard/generators/aps/src/main/scala/ApsWrapper.scala").read_text()
    fields = scala[scala.index("private def fields:"):scala.index(") ++ csrNames")]
    ports = re.findall(r'"(\w+)" -> (Input|Output)\((Clock|Bool|UInt)\((?:(\d+)\.W)?\)\)', fields)
    assert len(ports) > 60
    declarations = []
    for name, direction, kind, bits in ports:
        width = f"[{int(bits)-1}:0] " if bits else ""
        declarations.append(f"  {'input' if direction=='Input' else 'output'} wire {width}{name}")
    body = (HERE / "wrapper_body.sv").read_text()
    for name, direction, _, _ in ports:
        if direction == "Output" and not re.search(rf"assign\s+{name}\b", body):
            body += f"\nassign {name} = '0;"
    core = (RTL / "gemm_fp32.v").read_text()
    core_ports = re.findall(r"^(input|output)\s+(?:wire\s+)?(?:\[(\d+):0\]\s+)?(\w+);", core, re.M)
    assert core_ports, core[:3000]
    connections = []
    memory_code = []
    for direction, high, name in core_ports:
        bits = int(high)+1 if high else 1
        if name == "ap_clk": signal = "clk"
        elif name == "ap_rst": signal = "rst"
        elif name == "ap_start": signal = "(!rst && state == START)"
        elif name == "ap_done": signal = "hls_done"
        elif name == "use_c": signal = "use_c"
        elif name in {"ap_idle", "ap_ready"}: signal = ""
        else:
            match = re.fullmatch(r"(A|BT|C|D)_(address|ce|we|q|d)([01])", name)
            assert match, f"Unexpected native HLS port: {name}"
            array, role, index = match.groups()
            signal = f"hls_{name}"
            body += f"\n{'reg' if direction=='input' else 'wire'} [{bits-1}:0] {signal};"
            if role == "q":
                mem, tile = {"A": ("mem_a", "tile_a"), "BT": ("mem_b", "tile_b"),
                             "C": ("mem_c", "tile_c")}[array]
                memory_code.append(f"  if (!rst && hls_{array}_ce{index})\n"
                                   f"    {signal} <= {mem}[({{16'b0, {tile}}} << 6) + hls_{array}_address{index}];")
            elif role == "d":
                assert array == "D" and direction == "output"
                memory_code.append(f"  if (!rst && state == RUN && hls_D_ce{index} && hls_D_we{index})\n"
                                   f"    mem_c[({{16'b0, tile_d}} << 6) + hls_D_address{index}] <= {signal};")
        connections.append(f"  .{name}({signal})")
    body += "\nwire hls_done;\n"
    body += "\ngemm_fp32 accelerator(\n" + ",\n".join(connections) + "\n);\n"
    body += "\nalways @(posedge clk) begin\n" + "\n".join(memory_code) + "\nend\n"
    (HERE / "main.sv").write_text("// Handwritten RoCC adapter around Allo/Vitis-generated RTL.\nmodule main(\n"
                                  + ",\n".join(declarations) + "\n);\n" + body + "\nendmodule\n")
    files = [HERE / "main.sv"] + sorted(RTL.glob("*.v")) + sorted(RTL.glob("*.sv"))
    assert not any("axi" in p.name.lower() for p in files), files
    (HERE / "rtl.f").write_text("\n".join(map(str, files))+"\n")
    config = dict(backend="rocc", arch="rv32", core_count=1,
                  top_module="aps_rocc_wrapper", name="aps_rocc_module",
                  vsrc=list(map(str, files)), maxBurstBytes=128, nXacts=2)
    (HERE / "aps_config.yaml").write_text(json.dumps(config, indent=2)+"\n")
    print(f"Packaged {len(files)} RTL files; native ports: {[p[2] for p in core_ports]}")


if __name__ == "__main__":
    main()
