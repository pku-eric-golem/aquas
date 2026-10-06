#!/usr/bin/env python3
"""Package pinned HardFloat into a self-contained SV bundle and FIRRTL extern.

Named key=value arguments match the CMT2 ModuleLibrary build convention.
Run in the desired output directory; --resource-output overlays the existing
integer resource database with wrapper timing (not a physical characterization).
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thirdparty/HardFloat/source"
CATALOG = ROOT / "hardware/hardfloat/ip_catalog.yaml"


def verify_sources():
    vendor = SOURCE.parent
    for line in (vendor / "SOURCE.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"HardFloat source hash mismatch: {name}")


def expand_includes(path):
    def replace(match):
        name = match.group(1)
        candidates = [SOURCE / "RISCV" / name, SOURCE / name]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise ValueError(f"Cannot resolve {name} in {path}")
        return expand_includes(found)

    text = re.sub(r'`include\s+"([^"]+)"', replace, path.read_text())
    # Release 1 redundantly redeclares an ANSI output net. Normalize for SV,
    # retaining the unmodified, hash-verified upstream sources.
    if path.name == 'divSqrtRecFN_small.v':
        text = text.replace('    wire sqrtOpOut;\n', '')
    # Release 1 uses ceil(log2(p)), insufficient when p is a power of two:
    # BF16 p=8 saturates alignment at 7, inside the guard/round window.
    # Reserve enough range for p+3; unchanged for binary32 p=24 (5 bits).
    if path.name == 'addRecFN.v':
        text = text.replace('localparam alignDistWidth = clog2(sigWidth);',
                            'localparam alignDistWidth = clog2(sigWidth + 3);')
    return text


def atomic_write(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def native_timing(width, operation):
    catalog = yaml.safe_load(CATALOG.read_text())
    spec = list(catalog['operations'].values())[operation]
    variable = 'sig_latency' in spec
    latency = width - 8 + spec['sig_latency'] if variable else spec['latency']
    return dict(variable_latency=variable, latency_upper_bound=latency,
                no_stall_II=latency if variable else 1,
                outstanding_capacity=1 if variable else latency + catalog['response_slack'])


def transaction_timing(width, operation, latency, capacity):
    timing = native_timing(width, operation)
    if latency != timing['latency_upper_bound']:
        raise ValueError(f"Stale HardFloat latency {latency}; native contract requires {timing['latency_upper_bound']}")
    if timing['variable_latency'] and capacity != 1:
        raise ValueError("Iterative HardFloat capacity must be 1")
    timing['outstanding_capacity'] = capacity
    return timing


def main():
    catalog = yaml.safe_load(CATALOG.read_text())
    params = {"op": 0, "latency": None, "capacity": None, "width": 0, "pred": 0}
    resource_output = None
    cpp_output = None
    for arg in sys.argv[1:]:
        if arg.startswith("--cpp-output="):
            cpp_output = Path(arg.split("=", 1)[1])
        elif arg.startswith("--resource-output="):
            resource_output = Path(arg.split("=", 1)[1])
        else:
            key, value = arg.split("=", 1)
            if key not in params:
                raise ValueError(f"Unknown parameter: {key}")
            params[key] = int(value)
    if params['op'] not in range(19) or params['width'] not in (0,16,32):
        raise ValueError("Invalid operation or precision")
    timing = native_timing(params['width'] or 32, params['op'])
    if params['latency'] is None: params['latency'] = timing['latency_upper_bound']
    if params['capacity'] is None: params['capacity'] = timing['outstanding_capacity']
    op, latency, capacity = (params[k] for k in ("op", "latency", "capacity"))
    universal = params["width"] != 0
    if op not in range(19 if universal else 3) or not 1 <= capacity <= 256:
        raise ValueError("Invalid operation or capacity")
    transaction_timing(params['width'] or 32, op, latency, capacity)
    if cpp_output:
        cpp_output.parent.mkdir(parents=True, exist_ok=True)
        rows = ',\n'.join('  {' + ','.join(str(native_timing(w, k)['latency_upper_bound']) for k in range(19)) + '}' for w in (16,32))
        cpp_output.write_text(f'''// Generated from hardware/hardfloat/ip_catalog.yaml; do not edit.
#ifndef APS_HARDFLOATCONFIG_H
#define APS_HARDFLOATCONFIG_H
namespace mlir::hardfloat {{
constexpr unsigned latencies[2][19] = {{
{rows}
}};
constexpr unsigned latency(unsigned op, unsigned width) {{ return latencies[width==16 ? 0 : 1][op]; }}
constexpr unsigned capacity(unsigned op, unsigned width) {{ return op==3 || op==4 ? 1 : latency(op,width)+{catalog['response_slack']}; }}
}} // namespace mlir::hardfloat
#endif
''')
        return
    verify_sources()

    # A shared absolute bundle path makes BlackBoxPathAnno deduplication work
    # across add/sub/mul externs. No external include paths remain in the bundle.
    bundle = ROOT / "build/hardfloat/rtl/HardFloatIP.sv"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    dependencies = [
        "HardFloat_primitives.v",
        "RISCV/HardFloat_specialize.v",
        "HardFloat_rawFN.v",
        "isSigNaNRecFN.v",
        "fNToRecFN.v",
        "recFNToFN.v",
        "addRecFN.v",
        "mulRecFN.v", "mulAddRecFN.v", "compareRecFN.v", "iNToRecFN.v",
        "recFNToIN.v", "recFNToRecFN.v", "divSqrtRecFN_small.v",
    ]
    contents = "// Generated from pinned HardFloat Release 1; see vendor license.\n"
    expressions=[]
    for kind, spec in enumerate(catalog['operations'].values()):
        value=f"((W)-8+{spec['sig_latency']})" if 'sig_latency' in spec else str(spec['latency'])
        expressions.append(f"(O)=={kind} ? {value} : ")
    contents += "`define HF_NATIVE_LATENCY(W,O) (" + ''.join(expressions) + "0)\n"
    contents += f"`define HF_NATIVE_CAPACITY(W,O) (((O)==3 || (O)==4) ? 1 : (`HF_NATIVE_LATENCY(W,O)+{catalog['response_slack']}))\n"
    contents += "\n".join(expand_includes(SOURCE / name) for name in dependencies)
    for name in ("HF32AddSub", "HF32Mul", "HF32Binary", "HFCompute", "HFNativePipeline", "HFUnit"):
        contents += "\n" + (ROOT / f"hardware/hardfloat/{name}.sv").read_text()
    # Avoid unnecessary rewrites of the shared bundle (parallel build readers).
    if not bundle.exists() or bundle.read_text() != contents:
        temp = bundle.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(contents)
        temp.replace(bundle)

    if universal:
        emit_universal(params, catalog, bundle)
        return
    name = f"HardFloat32_op{op}_l{latency}_q{capacity}"
    # Give each specialization a unique Verilog defname. Native firtool's OM
    # class generation otherwise collides for parameterized extern defnames.
    specialization = bundle.parent / f"{name}.sv"
    specialization.write_text(
        f"""module {name} (
    input wire clock, reset, issue_enable,
    output wire issue_ready,
    input wire [31:0] operand0, operand1,
    input wire collect_enable,
    output wire collect_ready,
    output wire [31:0] collect_data,
    output wire [4:0] collect_flags
);
    HF32Binary #(.OP({op}), .LATENCY({latency}), .CAPACITY({capacity})) impl (.*);
endmodule
"""
    )
    mlir = f"""module {{
  firrtl.circuit "{name}" {{
    firrtl.extmodule @{name}(
      in clock: !firrtl.clock,
      in reset: !firrtl.uint<1>,
      in issue_enable: !firrtl.uint<1>,
      out issue_ready: !firrtl.uint<1>,
      in operand0: !firrtl.uint<32>,
      in operand1: !firrtl.uint<32>,
      in collect_enable: !firrtl.uint<1>,
      out collect_ready: !firrtl.uint<1>,
      out collect_data: !firrtl.uint<32>,
      out collect_flags: !firrtl.uint<5>
    ) attributes {{defname = "{name}", annotations = [{{class = "firrtl.transforms.BlackBoxPathAnno", path = {json.dumps(str(bundle))}}}, {{class = "firrtl.transforms.BlackBoxPathAnno", path = {json.dumps(str(specialization))}}}]}}
  }}
}}
"""
    Path(f"{name}.mlir").write_text(mlir)
    Path(f"{name}.f").write_text(str(bundle) + "\n" + str(specialization) + "\n")
    metadata = dict(catalog, **params, module=name, rtl=str(bundle))
    Path(f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if resource_output:
        resources = json.loads((ROOT / "examples/resource_ihp130.json").read_text())
        for kind, spec in enumerate(catalog['operations'].values()):
            resource_name = spec['resource']
            for width, suffix in ((16, '_bf16'), (32, '_fp32')):
                t = native_timing(width, kind)
                resources[resource_name + suffix] = dict(delay=[0]*7, latency=[t['latency_upper_bound']]*7,
                    II=t['no_stall_II'], constr=0, amount=-1)
            resources[resource_name] = dict(resources[resource_name + '_fp32'])
            resources[resource_name]['latency'] = list(resources[resource_name]['latency'])
            resources[resource_name]['latency'][4] = native_timing(16, kind)['latency_upper_bound']
        resources['fp'] = dict(resources['addf'])
        resource_output.parent.mkdir(parents=True, exist_ok=True)
        resource_output.write_text(json.dumps(resources, indent=2) + "\n")
    print(f"{name}.mlir")


def emit_universal(params, catalog, bundle):
    w,op,pred,l,q = (params[k] for k in ('width','op','pred','latency','capacity'))
    if w not in (16,32) or pred not in range(16):
        raise ValueError('Require width=16 (BF16)/32 (FP32), pred=0..15')
    name = f'HardFloat_w{w}_o{op}_p{pred}_l{l}_q{q}'
    ports = [('in','clock','!firrtl.clock'),('in','reset','!firrtl.uint<1>'),
             ('in','issue_enable','!firrtl.uint<1>'),('out','issue_ready','!firrtl.uint<1>'),
             ('in','operand0','!firrtl.uint<32>'),('in','operand1','!firrtl.uint<32>'),
             ('in','operand2','!firrtl.uint<32>'),('in','rounding_mode','!firrtl.uint<3>'),
             ('in','collect_enable','!firrtl.uint<1>'),('out','collect_ready','!firrtl.uint<1>'),
             ('out','collect_data','!firrtl.uint<32>'),('out','collect_flags','!firrtl.uint<5>')]
    impl = bundle.parent / (name+'.sv')
    atomic_write(impl, f'''module {name}(
 input wire clock,reset,issue_enable, output wire issue_ready,
 input wire [31:0] operand0,operand1,operand2, input wire [2:0] rounding_mode,
 input wire collect_enable, output wire collect_ready,
 output wire [31:0] collect_data, output wire [4:0] collect_flags
);
 HFUnit #(.WIDTH({w}),.OP({op}),.PRED({pred}),.LATENCY({l}),.CAPACITY({q})) impl(.*);
endmodule
''')
    port_text=',\n'.join(f'      {d} {n}: {t}' for d,n,t in ports)
    annotations=', '.join('{class = "firrtl.transforms.BlackBoxPathAnno", path = '+json.dumps(str(p))+'}' for p in (bundle,impl))
    atomic_write(Path(name+'.mlir'), f'''module {{
 firrtl.circuit "{name}" {{
  firrtl.extmodule @{name}(\n{port_text}\n  ) attributes {{defname = "{name}", annotations = [{annotations}]}}
 }}
}}
''')
    atomic_write(Path(name+'.f'), str(bundle)+'\n'+str(impl)+'\n')
    timing = transaction_timing(w, op, l, q)
    metadata = dict(catalog, **params)
    metadata.update(II=timing['no_stall_II'], timing=timing)
    metadata['latency_parameter_meaning'] = 'transaction_upper_bound' if op in (3, 4) else 'transaction'
    atomic_write(Path(name+'.json'), json.dumps(metadata,indent=2)+'\n')
    print(name+'.mlir')


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
