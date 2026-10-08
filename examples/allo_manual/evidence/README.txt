Allo / Vitis HLS FP32 GEMM, direct RTL ISAX integration
====================================================

Run from the repository root:
  bash tmp/allo/run.sh

To repeat only full Rocket/C simulation using the packaged RTL:
  pixi run bash tmp/allo/run_soc.sh
  python tmp/allo/summarize.py

SIM_CXX selects the host compiler (default /usr/bin/clang++ with C++20);
SIM_CXX_OPT selects its optimization (default -O1). Use -O0 for a faster
initial build but a substantially slower simulation. This changes
only host C++ optimization, not generated RTL or accelerator cycle counts.

All task-owned source, builds, generated RTL, ELF files and logs are in tmp/allo.
The checked-in Allo Python source is linked into python/allo. Its C++/MLIR
backend is built in allo-build against the already-installed Allo LLVM tree.
env.sh selects the existing conda allo environment. No thirdparty source is
modified. Existing Chipyard Java jars and DRAMSim library are reused read-only.

gemm.py defines an 8x8x8 FP32 tile kernel: D = A * transpose(BT) + optional C.
Each multiply/add rounds separately; the software benchmark disables FMA
contraction. schedule.py applies the official Allo scalar-vector/interleaved
GEMM schedule, plus local accumulation partitioning and output pipelining:
  s.reorder("k", "j")
  s.buffer_at(s.acc, axis="i")
  s.pipeline("j", initiation_interval=1)
  s.pipeline("v", initiation_interval=1)
  s.partition(s.acc, dim=0)
The algorithm uses an explicit local accumulation tensor to expose the perfect
reduction loop nest expected by buffer_at. Each output still accumulates in
ascending k order; there is no reassociation, fast math, or reduction tree.
External memory ports and the RoCC/DMA ABI are unchanged.

Official references:
  https://cornell-zhang.github.io/allo/gallery/tutorial_02_vhls.html
  https://cornell-zhang.github.io/allo/api/index.html

The previous vendor-scheduled profiles are preserved as comparisons:
  baseline: serial latency 6289
  variants/interleaved: reorder + buffering + pipeline latency 990
  variants/vendor-scheduled: partitioned latency 909, full prior SoC evidence
Current native HardFloat blackbox results are in result.json and HLS reports.
All requested loop pipeline IIs must achieve II=1.

generate.py first applies the Allo schedule, then lowers the three generated
FP32 SSA arithmetic sites to hf_add_f32/hf_mul_f32 calls. It asserts the exact
call counts and preserves loop structure and accumulation order. This is an
explicit C++ lowering step under tmp/allo, not a modification to Allo's backend.
It creates hls_axi and a native-port variant hls_native with the same computation.
Both use Vitis HLS 2024.1, xc7z020-clg400-1, nominal 10 ns HLS clock. Integration
uses one-cycle ap_memory + ap_ctrl_hs, with no top-level AXI.

Native HardFloat scheduling, RTL inclusion and Vitis compatibility:
  build_blackbox.py registers hf_add_f32/hf_mul_f32 with add_files -blackbox.
  Their JSON latency=3 and II=1 come from the current native IP catalog.
  HardFloatNative.v reuses pinned HardFloat sources and HFNativePipeline.sv;
  only CE guards are added to existing stage registers. No arithmetic stage
  is inserted, removed, or retimed to match vendor floating-point IP.
  The C++ implementations are simulation-only reference models. Generated
  hardware uses actual native HardFloat pipelines, RNE, canonical RISC-V NaNs,
  gradual underflow, and separate multiply/add operations.
  Vitis exports the external RTL automatically in syn/verilog. The packager
  copies the external HardFloat RTL unchanged and verifies hashes. Vitis
  2024.1 additionally emits a duplicate ap_ready connection in the output
  loop. The packager removes only the extra connection after proving its
  signal has no consumers. Original/exported and packaged hashes plus the
  exact repairs are recorded in fp-blackbox.json. Vitis also generates
  completion waits for ap_ctrl_none calls in extracted pipelines, which
  deadlock at the first valid call. For these two fixed-latency II=1 native
  operators with fixed-latency ap_memory, the packager sets those waits to
  false. It accepts only the observed loop-valid/exit expressions, validates
  operator contracts and rejects actual ready/valid/FIFO stall dependencies.
  Arithmetic, native stage boundaries and scheduling are unchanged. This
  workaround is kernel/version-specific, not a general HLS RTL transform.
  The exported AXI project is retained for inspection; full RTL simulation
  uses the repaired native-port project. There is no post-HLS arithmetic
  replacement or encrypted Xilinx FP leaf in this route.
  Blackboxes use ap_ctrl_none with wire input/return ports. CE freezes every
  existing arithmetic stage. Reset masks output; HLS resets validity/control,
  and three enabled edges refill the native stages.
  Vitis 2024.1 emits an unlisted ap_ready pin for one scalar-return call. To
  keep JSON and RTL consistent, the C signature explicitly includes a bool
  output argument named ap_ready, mapped to a constant-true auxiliary output.
  It is not a result-valid or chain handshake signal; callers ignore it.
  A minimal probe confirmed this mapping preserves II=1. The full output
  loop also needs the proven unused duplicate connection repair above.
  ap_ctrl_chain was also tested: Vitis 2024.1 rejects that protocol inside the pipeline region.


ASIC scope:
  The requested acceptance is full SoC RTL simulation. ASIC mapping, gate
  simulation, physical implementation and STA are not part of the run.
  HLS FPGA resource/Fmax estimates exclude actual HardFloat costs (blackbox
  resource entries are zero placeholders). Vitis does not import Liberty or
  auto-retime native HardFloat; changing stage boundaries requires updating
  JSON and rerunning scheduling, synthesis and RTL simulations.

Official references:
  https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/Adding-RTL-Blackbox-Functions
  https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/JSON-File-for-RTL-Blackbox

package_rtl.py generates APS-compatible main.sv from wrapper_body.sv and the
actual HLS port list. It reuses the current APS MainBlackBox port convention
and generates aps_config.yaml. The direct path bypasses CADL, TOR, APS-to-CMT2
and CMT2-to-FIRRTL. The existing Chipyard SoC integration still uses Chisel/
FIRRTL, as usual.

The wrapper handles one command at a time and holds the response until accepted.
Every supported instruction requires xd=1 and returns zero after completion.
It snapshots rd, operand values and tile offsets at command acceptance.

Instruction ABI (matching the existing BF16 example's logical tile ABI):
  custom0 (0x0b), funct7=0: compute
    rs1 = (tileA << 16) | tileB | (useC << 15)
    rs2 = (tileC << 16) | tileD
  custom3 (0x7b), funct7=0/1/2: load A / load BT / store D
    rs1 = CPU memory pointer
    rs2 = (scratchpad element offset << 16) | DRAM row pitch in bytes

Scratchpads contain 131072 FP32 elements each:
  A byte base 0x000000; BT base 0x080000; C/D base 0x100000.
Each DMA instruction transfers eight tiles in an 8-row by 64-column layout.
Each 256-byte FP32 row is split into two 128-byte bursts, because the existing
DMA transfer counter and configuration support 128-byte transactions. The
scratchpad stride is 32 bytes within a tile row and eight rows between tiles.

Tests:
  verify_allo.py: actual Allo LLVM JIT against independent FP32 mul/add goldens.
  verify_blackbox.py: native latency=3/II=1, CE stall and reset checks, 6396
    independent integer-oracle results; lowered HLS C model checks 3072 outputs.
  test_wrapper.cpp: actual HLS+HardFloat+adapter RTL, 3072 outputs including
    zeros, cancellation, rounding boundaries, subnormals, overflow, NaN/Inf,
    random bit patterns, C accumulation, in-place writes, response stalls,
    destination-register preservation and mid-execution reset.
  software/benchmark_fp32.c: native RV32F software and accelerator in one ELF,
    two trials with reversed order, separate DMA/compute/store cycle counters,
    2048 checked outputs; based on the existing BF16 benchmark layout/ABI.
  software/test_gemm.c: adapted existing C test, 32 tiles / 2048 outputs,
    optional C, separate and in-place writes, high offsets, padded DRAM rows,
    and memory guards. A deliberately corrupted golden must exit with code 1.

Evidence:
  result.json                 Required checks and measured cycle counts
  wrapper-test.log            Standalone RTL result
  soc/benchmark.log           Full Rocket benchmark result
  soc/test.log                Full Rocket functional/DMA result
  soc/negative.log            Deliberately incorrect golden rejection
  baseline/soc/*              Archived serial baseline evidence
  host-simulator.json         Unchanged generated C++ model and compiler details
  hls-native.log, hls-axi.log  Vitis generation logs
  fp-blackbox.json            Native contracts, control repairs and RTL hashes
  blackbox/test.log           Enabled-edge latency, CE/reset and arithmetic checks
  hls-model-test.log          Lowered HLS C model validation

Measured current and comparison cycles are recorded in result.json.
Current native HardFloat blackbox HLS latency: 900 cycles, all pipelines II=1.
Current Rocket ISAX: compute 6977 / 6951, total 9138 / 9062 including DMA.
Mean software/ISAX ratio including DMA: 1.77835x.
Full SoC functional test: 32 tiles / 2048 outputs PASS; benchmark checks 2048
outputs PASS; corrupted golden exits with code 1 as expected.
Standalone RTL: 3072 outputs PASS. Native operator latency/CE/reset: 6396 PASS.
Previous vendor-scheduled ISAX: compute 7049 / 7023, total 9210 / 9134.
Serial ISAX: compute 50489 / 50463, total 52650 / 52574.
Software reference: 16553 / 15813 cycles.
The optimization records retain Allo primitives, official references, HLS
profiles, achieved IIs, and measured compute/DMA-inclusive speedups.

bf16-experiment contains the initial BF16 feasibility work, before the request
was changed to FP32. It is not used by this FP32 flow.
