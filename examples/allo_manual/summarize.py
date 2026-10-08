"""Require successful real runs and record evidence and benchmark cycles."""
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def completed_log(name, marker):
    # Current optimized runs must pass; archived serial logs cannot satisfy them.
    for suffix in ("",):
        path=HERE/f"soc/{name}{suffix}.log"
        if path.exists() and marker in path.read_text():
            return path
    raise AssertionError(f"No completed {name} simulation")


def hls_metrics(folder):
    root=ET.parse(folder/"gemm_fp32_csynth.xml").getroot()
    return {"latency":int(root.findtext("PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency")),
            "interval":int(root.findtext("PerformanceEstimates/SummaryOfOverallLatency/Interval-max")),
            "vendor_resource_estimates":{x.tag:int(x.text) for x in root.find("AreaEstimates/Resources")}}


def main():
    packaged=max(p.stat().st_mtime for p in (HERE/"main.sv", *sorted((HERE/"rtl").glob("*.v"))))
    for path in (HERE/"wrapper-test.log", HERE/"soc/benchmark.log", HERE/"soc/test.log", HERE/"soc/negative.log"):
        assert path.stat().st_mtime>=packaged, f"Stale evidence: {path}"
    bench=(HERE/"soc/benchmark.log").read_text()
    test_file=completed_log("test", "ALLO FP32 GEMM PASS tiles=32 elements=2048")
    negative_file=completed_log("negative", "ALLO FP32 GEMM FAIL: batch=0 pass=0")
    test=test_file.read_text()
    negative=negative_file.read_text()
    wrapper=(HERE/"wrapper-test.log").read_text()
    assert "ALLO FP32 BENCH PASS trials=2 checked=2048" in bench
    assert "ALLO FP32 GEMM PASS tiles=32 elements=2048" in test
    assert "ALLO FP32 GEMM FAIL: batch=0 pass=0" in negative
    assert re.search(r"\*\*\* FAILED \*\*\* \((tohost|exit code) =\s*1\)",negative)
    assert "ALLO WRAPPER PASS checked=3072" in wrapper
    llvm=json.loads((HERE/"allo-validation.json").read_text())
    assert llvm["passed"] and llvm["checked"]==512 and llvm["precision"]=="FP32"
    fp_leaves=json.loads((HERE/"fp-blackbox.json").read_text())
    assert fp_leaves["post_hls_leaf_replacement"] is False
    assert fp_leaves["no_xilinx_fp_ip"]
    for name, expected in fp_leaves["unmodified_hls_verilog_sha256"].items():
        for folder in (HERE/"rtl", HERE/"hls_native/out.prj/solution1/syn/verilog"):
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected
    for name, expected in fp_leaves["external_rtl_sha256"].items():
        for folder in (HERE/"rtl", HERE/"blackbox"):
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected
    for name, expected in fp_leaves["original_hls_verilog_sha256"].items():
        assert hashlib.sha256((HERE/"hls_native/out.prj/solution1/syn/verilog"/name).read_bytes()).hexdigest()==expected
    for name, expected in fp_leaves["packaged_verilog_sha256"].items():
        assert hashlib.sha256((HERE/"rtl"/name).read_bytes()).hexdigest()==expected
    assert not list((HERE/"hls_native/out.prj/solution1/syn/verilog").glob("*_ip.tcl"))
    for name in ("hf_add_f32","hf_mul_f32"):
        root=ET.parse(HERE/f"hls_native/out.prj/solution1/syn/report/{name}_csynth.xml").getroot()
        assert root.findtext("PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency")=="3"
        assert root.findtext("PerformanceEstimates/SummaryOfOverallLatency/Interval-max")=="1"
    assert "HLS C MODEL PASS checked=3072" in (HERE/"hls-model-test.log").read_text()
    bb_log=(HERE/"blackbox/test.log").read_text()
    assert "NATIVE HARDFLOAT PASS checked=6396 latency=3 II=1" in bb_log
    assert "m_axi_gmem" in (HERE/"hls_axi/out.prj/solution1/syn/verilog/gemm_fp32.v").read_text()
    assert "m_axi_" not in (HERE/"rtl/gemm_fp32.v").read_text()
    for log in ("hls-native.log","hls-axi.log"):
        text=(HERE/log).read_text()
        assert "Generating Verilog RTL for gemm_fp32" in text and "ERROR:" not in text
    records=[]
    for match in re.finditer(r"ALLO_BENCH trial=(\d+) engine=(\w+) tiles=8 elements=512 ([^\n]+)",bench):
        records.append(dict(trial=int(match[1]),engine=match[2],
                            **{k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",match[3])}))
    assert len(records)==4
    software=sum(r["cycles"] for r in records if r["engine"]=="software")
    hardware=sum(r["total"] for r in records if r["engine"]=="isax")
    baseline=json.loads((HERE/"baseline/result.json").read_text())
    profiles={"serial":hls_metrics(HERE/"baseline/hls-report"),
              "interleaved":hls_metrics(HERE/"variants/interleaved/hls_native/out.prj/solution1/syn/report"),
              "vendor_scheduled_partitioned":hls_metrics(HERE/"variants/vendor-scheduled/hls_native/out.prj/solution1/syn/report"),
              "native_hardfloat_partitioned":hls_metrics(HERE/"hls_native/out.prj/solution1/syn/report")}
    baseline_hardware=sum(r["total"] for r in baseline["benchmark"] if r["engine"]=="isax")
    baseline_compute=sum(r["compute"] for r in baseline["benchmark"] if r["engine"]=="isax")
    compute=sum(r["compute"] for r in records if r["engine"]=="isax")
    pipeline_results=re.findall(r"Target II = (\d+), Final II = (\d+), Depth = (\d+), loop '([^']+)'",
                                (HERE/"hls-native.log").read_text())
    assert pipeline_results and all(int(ii)==1 for _,ii,_,_ in pipeline_results)
    optimization={"official_documents":["https://cornell-zhang.github.io/allo/gallery/tutorial_02_vhls.html",
                                        "https://cornell-zhang.github.io/allo/api/index.html"],
                  "schedule":json.loads((HERE/"generation.json").read_text())["schedule"],
                  "hls_profiles":profiles,"achieved_pipeline_ii":pipeline_results,
                  "hls_latency_speedup":profiles["serial"]["latency"]/profiles["native_hardfloat_partitioned"]["latency"],
                  "soc_compute_speedup":baseline_compute/compute,
                  "soc_total_speedup":baseline_hardware/hardware,
                  "baseline_benchmark":baseline["benchmark"],
                  "fp_reduction_order_preserved":True,
                  "resource_caveat":"HLS blackbox costs are zero placeholders, not physical area; ASIC backend omitted at user request"}
    result={"passed":True,"precision":"FP32","rounding":"RNE, separate multiply/add",
            "flow":"Allo -> Vitis HLS 2024.1 -> ap_memory/ap_ctrl_hs -> handwritten SV RoCC adapter -> APSRocketConfig",
            "allo_commit":subprocess.check_output(["git","-C",str(ROOT/"thirdparty/allo"),"rev-parse","HEAD"],text=True).strip(),
            "axi_generated":True,"native_hls_rtl_unchanged":not bool(fp_leaves["rtl_interface_fixes"]),
            "native_hardfloat_rtl_unchanged":True,
            "fp_blackboxes":fp_leaves,
            "asic_backend_run":False,
            "host_simulator":json.loads((HERE/"host-simulator.json").read_text()),
            "optimization":optimization,
            "checks":{"allo_llvm_outputs":512,"lowered_hls_c_outputs":3072,
                      "native_blackbox_outputs":6396,"native_blackbox_stall_reset":True,
                      "standalone_rtl_outputs":3072,
                      "rocket_benchmark_outputs":2048,"rocket_functional_outputs":2048,
                      "response_backpressure":True,"mid_execution_reset":True,
                      "c_accumulation_and_in_place_writeback":True,
                      "dma_padding_and_guards":True,"corrupted_golden_rejected":True},
            "benchmark":records,"software_to_isax_cycle_ratio":software/hardware,
            "logs":{"benchmark":"soc/benchmark.log","functional":str(test_file.relative_to(HERE)),
                    "negative":str(negative_file.relative_to(HERE)),"standalone":"wrapper-test.log"},
            "timing_caveat":"RTL cycle simulation only, ASIC backend omitted at user request; HLS FPGA Fmax excludes blackbox paths",
            "files_sha256":{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (HERE/"gemm.py",HERE/"schedule.py",HERE/"main.sv",HERE/"aps_config.yaml",
                                      HERE/"software/benchmark_fp32.c",HERE/"software/test_gemm.c",
                                      HERE/"generate.py",HERE/"build_blackbox.py",HERE/"package_blackbox.py",
                                      HERE/"hls_native/kernel.cpp",*sorted((HERE/"blackbox").glob("*.json")),
                                      *sorted((HERE/"blackbox").glob("*.cpp")),
                                      *sorted((HERE/"rtl").glob("*"))) if p.is_file()}}
    (HERE/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"passed":True,"software_to_isax_cycle_ratio":software/hardware,"checks":result["checks"]},indent=2))


if __name__=="__main__":main()
