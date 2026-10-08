"""Package native HardFloat HLS RTL with constrained Vitis 2024.1 control fixes.

Fixed latency/II and nonblocking ap_memory make results available at the
scheduled stages. Vitis's extracted-loop control incorrectly waits for a
nonexistent ap_ctrl_none completion handshake. Never apply these fixes to
variable-latency, FIFO, or other handshaking operators.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "hls_native/out.prj/solution1/syn/verilog"
OUT = HERE / "rtl"


def main():
    OUT.mkdir(exist_ok=True)
    assert not list(SOURCE.glob("*_ip.tcl")), "Unexpected vendor arithmetic IP"
    files = list(SOURCE.glob("*.v"))
    assert files
    text = "\n".join(p.read_text() for p in files)
    assert "hf_add_f32" in text and "hf_mul_f32" in text
    assert "floating_point" not in text
    for file in OUT.iterdir():
        if file.suffix in {".v", ".sv"}:
            file.unlink()
    hashes = {}
    original_hashes = {}
    packaged_hashes = {}
    fixes = []
    for file in files:
        source = file.read_text()
        original_hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        def fix(match):
            instance = match.group(0)
            ready = re.findall(r"\.ap_ready\((\w+)\)", instance)
            if len(ready) < 2:
                return instance
            assert len(ready) == 2 and ready[0].endswith("_ap_ready_r")
            assert ready[1].endswith("_ap_ready")
            # The extra signal has precisely one declaration and one pin
            # connection; no consumer or control logic can observe it.
            assert len(re.findall(rf"\b{re.escape(ready[1])}\b", source)) == 2
            instance = re.sub(rf"\s*\.ap_ready\({re.escape(ready[1])}\)\s*,?", "", instance)
            instance = re.sub(r",\s*(?=\);)", "\n", instance)
            fixes.append(dict(file=file.name, unused_signal=ready[1],
                              reason="Vitis 2024.1 emits duplicate ap_ready; extra wire has no consumers",
                              arithmetic_or_schedule_changed=False))
            return instance
        packaged = re.sub(r"\bhf_(?:add|mul)_f32\s+\w+\s*\([\s\S]*?\);", fix, source)
        if file.name in {"gemm_fp32_gemm_fp32_Pipeline_l_S_k_0_k_l_j.v",
                          "gemm_fp32_gemm_fp32_Pipeline_l_outputs_u_l_v.v"}:
            assert "hf_add_f32" in source
            assert "FIFO" not in source and "_empty_n" not in source and "_full_n" not in source
            def fixed_ready(match):
                signal, expression = match.groups()
                # Only the observed bare loop-valid/exit predicates qualify.
                # A real ready/valid/empty/full dependency must fail this gate.
                remaining = re.sub(
                    r"\b(?:icmp_ln\d+_reg_\d+_pp0_iter\d+_reg|ap_enable_reg_pp0_iter\d+)\b|1'[db][01]|[()&|=\s]",
                    "", expression)
                assert not remaining, f"Unexpected stall dependency: {expression}"
                fixes.append(dict(file=file.name, signal=signal,
                                  original_expression=expression, replacement="1'b0",
                                  reason="Fixed-latency II=1 ap_ctrl_none wire result has no dynamic done handshake",
                                  arithmetic_or_schedule_changed=False))
                return f"{signal} = 1'b0;"
            packaged, count = re.subn(
                r"\b(ap_block_pp0_stage0_(?:11001(?:_ignoreCallOp\d+)?|subdone))\s*=\s*([^;]+);",
                fixed_ready, packaged)
            assert count == (4 if "l_S_k_0" in file.name else 3), count
        (OUT / file.name).write_text(packaged)
        packaged_hashes[file.name] = hashlib.sha256((OUT/file.name).read_bytes()).hexdigest()
        if packaged == source:
            hashes[file.name] = original_hashes[file.name]
    external = {}
    for name in ("hf_add_f32", "hf_mul_f32"):
        config = json.loads((HERE / "blackbox" / (name + ".json")).read_text())
        assert config["rtl_performance"] == {"latency": "3", "II": "1"}
        for path in config["rtl_files"]:
            file = Path(path)
            expected = hashlib.sha256(file.read_bytes()).hexdigest()
            if (OUT / file.name).exists():
                assert hashlib.sha256((OUT / file.name).read_bytes()).hexdigest() == expected
            else:
                shutil.copyfile(file, OUT / file.name)
            external[file.name] = expected
    metadata = dict(json.loads((HERE / "blackbox/contracts.json").read_text()),
                    unmodified_hls_verilog_sha256=hashes,
                    original_hls_verilog_sha256=original_hashes,
                    packaged_verilog_sha256=packaged_hashes,
                    rtl_interface_fixes=fixes,
                    external_rtl_sha256=external)
    (HERE / "fp-blackbox.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Packaged {len(files)} HLS files, {len(external)} native files, {len(fixes)} constrained control repairs; no arithmetic replacement")


if __name__ == "__main__":
    main()
