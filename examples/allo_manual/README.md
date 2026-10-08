# Allo → Vitis HLS → HardFloat → RoCC：手动接入示例

本目录归档了 `tmp/allo/` 实验中已经通过现有 Rocket SoC RTL 仿真的 FP32 GEMM，包括源码、生成前后的 HLS C++、修改前后的 RTL、Chipyard 入口、软件、复现脚本和原始验证记录。这是一条独立于 CADL synthesis 的路径；验收范围到 **SoC RTL 仿真**，没有运行 ASIC 后端。

## 实现路径

```text
gemm.py + schedule.py
  │ Allo schedule → Vitis HLS 后端
  ▼
kernel_allo.cpp                    原始 Allo 输出，FP32 +/*，顶层 AXI
  │ generate.py：3 个 FP32 运算点改为 HardFloat blackbox 调用
  ▼
kernel_axi.cpp / hls_axi/kernel.cpp 仍有顶层 AXI，算术已绑定 HardFloat
  │ 仅转换顶层接口及参数名，计算语句保持一致
  ▼
hls_native/kernel.cpp              ap_memory(latency=1) + ap_ctrl_hs
  │ Vitis HLS 2024.1，blackbox latency=3 / II=1
  ▼
hls_native/.../syn/verilog/         HLS 原始 RTL，自动包含 HardFloat
  │ package_blackbox.py：受约束的 Vitis 控制修补
  ▼
rtl/                               接入使用的 RTL，无 AXI、无 Xilinx FP IP
  │ wrapper_body.sv + package_rtl.py
  ▼
main.sv + aps_config.yaml           RoCC / DMA / scratchpad 包装器
  │ 现有 Chipyard APSRocketConfig
  ▼
RV32F C 程序 → 真实 Rocket + 加速器 RTL 仿真
```

此处“剥掉 AXI”是在 **HLS C++ 接口层替换 pragma 并重新综合**，没有从 AXI RTL 中直接删掉总线状态机。`hls_axi/` 保留 AXI 版本供对照；通过验证的接入采用 `hls_native/` 经修补后的版本。

加速器生成不经过 CADL、TOR、CMT2 或其 FIRRTL 降低流程。SoC 自身仍使用 Chipyard 正常的 Chisel/FIRRTL 生成过程。手写 SV 接口沿用现有 `MainBlackBox` 端口约定，通过 APS 的外部 RTL 配置接入。

### 计算与 Allo 优化

固定 8×8×8 tile，`BT[j,k] = B[k,j]`，计算 `D = A × BTᵀ + (use_c ? C : 0)`。FP32 乘法、加法分别执行 RNE 舍入，保留递增的 `k` 累加顺序；不是融合 FMA。

`gemm.py` 显式声明 `acc[8,8]`，以暴露适合 `buffer_at` 的归约循环。`schedule.py` 按 [Allo 官方 Vitis GEMM 教程](https://cornell-zhang.github.io/allo/gallery/tutorial_02_vhls.html) 和 [API 文档](https://cornell-zhang.github.io/allo/api/index.html) 应用：

```python
s.reorder("k", "j")
s.buffer_at(s.acc, axis="i")
s.pipeline("j", initiation_interval=1)
s.pipeline("v", initiation_interval=1)
s.partition(s.acc, dim=0)
```

这些变换把不同输出的累加交错执行，局部缓冲和完全分区支持流水线；不进行归约重结合。当前 HLS 报告总延迟 900 cycles，初始化、MAC、回写、输出流水线均达到 II=1。

### HardFloat 如何进入调度和 RTL

`build_blackbox.py` 读取仓库 `hardware/hardfloat/HFNativePipeline.sv` 和校验过的 HardFloat Release 1 源码，生成 `blackbox/HardFloatNative.v`。只给原有寄存器边界增加 CE，未添加、移动或删除算术流水级。加法与乘法均为 **3 个有效时钟沿延迟、II=1**。

`hf_add_f32.json` / `hf_mul_f32.json` 用 Vitis 的 `add_files -blackbox` 注册 C 函数、RTL 文件、端口和延迟。Vitis 因而在调度时采用原生 HardFloat 合约，并在输出 RTL 时自动包含实际 HardFloat 模块。C++ 模型只用于 C 仿真，硬件使用外部 RTL；本路径没有在 HLS 后替换 Xilinx 浮点叶子。

算术固定 RNE、RISC-V canonical NaN、支持渐进下溢；CE 同时冻结各原有流水级。复位时输出被屏蔽、算术冻结，HLS 清空控制/有效标记，再以三个有效时钟沿重新填充流水线。

官方接口资料：[添加 RTL blackbox](https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/Adding-RTL-Blackbox-Functions)、[JSON 格式](https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/JSON-File-for-RTL-Blackbox)。

## 文件说明

| 文件/目录 | 用途 |
| --- | --- |
| `gemm.py`, `schedule.py` | Allo 算法与优化 schedule |
| `gemm.mlir` | 优化后的 Allo MLIR 快照 |
| `kernel_allo.cpp` | 原始 Allo 后端输出；与原实验的算术替换逆变换逐字一致 |
| `kernel_axi.cpp`, `hls_axi/kernel.cpp` | FP32 运算改为 HardFloat 调用后的 AXI HLS C++ |
| `hls_native/kernel.cpp` | 去 AXI、使用原生内存接口的 HLS C++ |
| `hls_axi/run.tcl`, `hls_native/run.tcl` | 两个 HLS 工程的综合入口 |
| `hls_{axi,native}/out.prj/solution1/syn/verilog/` | Vitis 原始生成 RTL，包括 AXI 对照与原生接口版本 |
| `hls_{axi,native}/out.prj/solution1/syn/report/` | 原始综合报告、延迟、II 和 blackbox 合约报告 |
| `blackbox/` | 原生 HardFloat RTL、Vitis JSON 描述、C 模型及算子测试 |
| `rtl/` | 修补后的实际接入 RTL；与原 SoC 通过记录一致 |
| `fp-blackbox.json` | 3-cycle / II=1 合约、每处控制修补、修改前后 RTL 哈希 |
| `wrapper_body.sv` | 手写 RoCC 命令状态机、DMA 与 scratchpad 主体 |
| `main.sv` | 给 Chipyard 的完整 `main` 模块，包含实际 HLS 端口连接 |
| `aps_config.yaml.in`, `rtl.f.in` | 配置/文件表模板；`@WORK@` 由工作目录路径取代，实际文件由 `package_rtl.py` 生成 |
| `software/benchmark_fp32.c` | 基于现有 BF16 示例布局/ABI 改写的 FP32 软件/加速器对照基准 |
| `software/test_gemm.c` | 全 SoC 功能、可选 C、原地回写、DMA 步长/边界及负例测试 |
| `software/vectors.h`, `software/bench_vectors.h` | 固定种子的位精确测试向量 |
| `generate_vectors.py` | 使用仓库独立整数 IEEE oracle 重新生成向量 |
| `test_wrapper.cpp`, `test_hls_model.cpp` | 独立 RTL 包装器测试及改造后 HLS C 模型测试 |
| `bootstrap.py`, `env.sh` | 在工作目录构建 Allo Python/MLIR overlay、选择现有 LLVM 和 Python 环境 |
| `generate.py`, `build_blackbox.py` | 生成各阶段 HLS C++/工程及 HardFloat blackbox |
| `package_blackbox.py`, `package_rtl.py` | 受约束的 RTL 修补和 Chipyard 包装/配置生成 |
| `run.sh`, `pipeline.sh`, `validate_rtl.sh`, `run_soc.sh` | 归档入口、完整流水线、快速 RTL 检查、全 SoC 仿真 |
| `verify_allo.py`, `verify_blackbox.py`, `summarize.py` | Allo LLVM、算子/C 模型检查，以及完整新运行结果汇总 |
| `toolchain.json` | Allo/Chipyard revision、HLS 版本及综合参数 |
| `evidence/` | 原 `tmp/allo/` 已通过运行的日志、结果和必要原始输入；`reproduction/` 保存归档脚本完整重放的通过记录 |
| `baseline/`, `variants/` | 历史优化对照的报告/结果/日志，只保留比较所需的小文件 |
| `artifact-manifest.json`, `artifact-origins.json`, `check_archive.py` | 当前归档哈希、原文件来源/哈希、完整性与原验证输入检查 |

所有硬件 RTL、`main.sv`、算法/schedule、C 软件与原验证版本保持逐字一致。复现脚本和 HLS C++ 的 include 路径经过可迁移调整；改动前版本保存在 `evidence/original-inputs/`，清单中标注 `adapted`。`evidence/` 的日志和原始 JSON 保留历史绝对路径以维持原记录，不作为新运行的文件表。没有复制大型构建缓存、ELF、SoC 生成目录和模拟器二进制。

## 复现方式

以下命令均从仓库根目录执行。入口先检查归档哈希，再复制到 `tmp/allo_manual/` 构建，不在 `examples/` 中生成编译缓存，不依赖原 `tmp/allo/`。重跑会更新该工作目录中的源码、生成文件和日志；可用 `ALLO_WORK_NAME=allo_manual_run2` 选择另一个 `tmp/` 子目录。

### 已生成 RTL 的快速检查

需要仓库 Pixi 的 Verilator、Python 3、G++，以及现有 `thirdparty/HardFloat` 和 Chipyard APS 源码。无需运行 Allo、Vitis，也无需 Vitis C++ 头文件。

```bash
python3 examples/allo_manual/check_archive.py
bash examples/allo_manual/run.sh --rtl
```

这会重新生成 HardFloat、从归档的 HLS 原始 RTL 重做修补和 `main.sv`，运行 6396 个算子结果检查与 3072 个包装器 RTL 输出检查。结果在 `tmp/allo_manual/blackbox-validation.log`、`wrapper-test.log`。**这一模式不包含 Rocket 仿真。**

### 已生成 RTL 接入完整 SoC

需要本仓库已经可用的 Chipyard/Pixi 仿真环境，包括 RV32 bare-metal GCC/HTIF、Java/SBT、Verilator、DRAMSim，以及默认 `/usr/bin/clang++`。本示例沿用现有 Chipyard 依赖，不重新安装整个环境。

```bash
bash examples/allo_manual/run.sh --soc
```

该模式重新配置包装器，编译软件并生成/构建真实 `APSRocketConfig` 模拟器，依次运行 benchmark、功能测试、故意破坏 golden 的负例。负例必须退出失败，脚本才整体成功。日志位于 `tmp/allo_manual/soc/{build,benchmark,test,negative}.log`。

宿主编译器可以用 `SIM_CXX`、`SIM_CXX_OPT` 选择，默认 Clang/C++20/`-O1`；只影响模型编译和仿真速度，不改变 RTL 或被测周期数。`JOBS` 默认 12。生成的 SoC、模拟器、软件 ELF 均在工作目录中。

### 从 Allo 源码完整重做

还需要与 `toolchain.json` 中 revision 对应的 Allo 子模块、兼容的现成 LLVM/MLIR build、带 Allo Python 依赖的环境，以及 **Vitis HLS 2024.1**。本次使用 LLVM `22.0.0git`（commit `6b09f739c4d085dc39eb9ff220c786bc3aa8c7fb`）、Python 3.12.13、NumPy 2.4.4、nanobind 2.12.0。本机 HLS 安装为 `/opt/Xilinx/Vitis_HLS/2024.1`；并非 `/opt/Xilinx/Vitis/` 中的 Vitis 统一入口。

```bash
# 默认使用 $HOME/miniconda3 下的 allo 环境；其他安装可设置：
export ALLO_CONDA_ROOT="$HOME/miniconda3"
export ALLO_CONDA_ENV=allo
# 指向你的兼容 LLVM build；本机原实验使用此位置：
export LLVM_BUILD_DIR="$HOME/repos/allo/externals/llvm-project/build"
export VITIS_HLS_ROOT=/opt/Xilinx/Vitis_HLS/2024.1
bash examples/allo_manual/run.sh --full
```

已有激活的其他 Python 环境可设置 `ALLO_SKIP_CONDA=1`；该环境仍须包含 Allo 所需的 Python 依赖和 nanobind。没有显式设置 `LLVM_BUILD_DIR` 时，`env.sh` 搜索子模块内部 LLVM build 和上述本机路径。脚本不会下载/构建 LLVM 或安装 Xilinx 工具。

完整模式依次构建 Allo MLIR bindings、建立指向仓库 Allo 源码的 overlay、运行 LLVM 检查、生成 C++/blackbox、综合 AXI/native 两个工程、检查算子/HLS C 模型、修补/包装 RTL、独立 RTL 仿真、全 Rocket 仿真，最后生成工作目录的 `result.json`。`ALLO_BUILD_JOBS` 默认 4。

若单独重跑已生成的 HLS 工程，必须先刷新描述文件为实际工作目录的绝对路径：

```bash
python3 tmp/allo_manual/build_blackbox.py
export VITIS_HLS_ROOT=/opt/Xilinx/Vitis_HLS/2024.1
source "$VITIS_HLS_ROOT/settings64.sh"
(cd tmp/allo_manual/hls_native && "$VITIS_HLS_ROOT/bin/vitis_hls" -f run.tcl)
python3 tmp/allo_manual/package_blackbox.py
python3 tmp/allo_manual/package_rtl.py
```

归档内 blackbox JSON 使用相对文件名便于审阅；`build_blackbox.py` 会生成 HLS 实际使用的绝对文件名。综合参数为 `xc7z020clg400-1`、名义 10 ns，这是 HLS 调度设置，不是 ASIC 时序证明。

### 直接供 Chipyard 使用

先运行任一复现模式生成实际路径配置。之后可使用：

```bash
pixi run bash tmp/allo_manual/run_soc.sh
```

`run_soc.sh` 设置 `APS_CONFIG=tmp/allo_manual/aps_config.yaml` 的绝对路径，并传给现有 Chipyard Makefile。`vsrc` 包含 `main.sv` 和全部 `rtl/` 文件。不要直接把带 `@WORK@` 的模板或原实验 `evidence/aps_config.yaml` 当作运行配置。

## RoCC 与 DMA ABI

软件遵循现有 BF16 示例的逻辑 tile ABI，但数据改为 FP32：

| 指令 | 参数 | 行为 |
| --- | --- | --- |
| custom0 `0x0b`, funct7=0 | `rs1=(tileA<<16)\|tileBT\|(useC<<15)`；`rs2=(tileC<<16)\|tileD` | 计算一个 8×8 tile |
| custom3 `0x7b`, funct7=0/1/2 | `rs1=CPU 内存指针`；`rs2=(scratchpad元素偏移<<16)\|DRAM行步长字节数` | 加载 A / 加载 BT / 存储 D |

支持的指令要求 `xd=1`，完成后返回 0。包装器接受命令时锁存操作数、目标寄存器编号和 tile 偏移，单条命令在途，响应遇到反压时保持稳定。

三个 scratchpad 各有 131072 个 FP32 元素：A 字节基址 `0x000000`，BT `0x080000`，C/D `0x100000`。一次 DMA 指令按 8 行 × 64 列的布局传输 8 个 tile。每行 256 字节被拆成两个 128 字节 burst，以适配现有 DMA 计数器；配置 `maxBurstBytes=128, nXacts=2`。tile 内行跨度为 32 字节，tile 间隔为 8 行。

## 重要问题及解决办法

1. **直接使用 FPGA 浮点运算不能表达原生 HardFloat 调度。** 在 HLS 前把 2 个加法、1 个乘法 SSA 运算点显式改为 blackbox 调用，用 JSON 指定原生 3-cycle/II=1 合约。转换校验精确调用数量，保留计算顺序。这是本示例的 C++ 转换，不是修改 Allo 后端或自动识别所有浮点运算的通用 pass。
2. **共享 RTL 重复注册。** 两个 JSON 同时列出 `HardFloatNative.v` 会导致 Vitis 重复处理定义。公共文件仅列在 add 描述中，mul 描述只列自己的 wrapper；HLS 将所有 blackbox RTL 合并导出。
3. **Vitis 2024.1 对 scalar-return blackbox 的额外 `ap_ready` 接线。** C 函数增加 `bool &ap_ready` 输出，映射到常真辅助 RTL 输出，使 JSON 和实际端口一致。它不是结果有效信号，也不是 chain 握手。完整输出循环仍生成重复的同名端口连接；packager 确认多余信号仅出现于声明和接线、没有消费者后移除该连接。
4. **提取后的流水循环等待不存在的完成握手，导致死锁。** `ap_ctrl_none` 算子的结果已经按固定延迟到达，Vitis 却生成 completion stall。packager 仅针对本 kernel 的两个确切模块，将 7 个已观察的 stall 表达式设为 false；只接受循环 valid/exit 寄存器及常量组合，遇到实际 ready/valid/FIFO 依赖就拒绝。连同重复端口，总计 **8 处修补、涉及 2 个文件**，详见 `fp-blackbox.json`。算术和流水寄存器没有被替换。
5. **其他协议并未解决这一版本的问题。** 实测 `ap_ctrl_chain` 被 Vitis 拒绝用于 pipeline region；改为 void + 标量引用输出引入伪依赖使 II=2；FRP 尝试回退到 STP。这里采用的固定延迟控制修补是明确的工程 workaround，不能认为 HLS 原生生成的控制已经完全正确。
6. **FP32 DMA 行长度超过原 DMA 计数能力。** 8 个 tile 的一行是 256 字节，改为两个 128 字节 burst，保留原有 SoC DMA 接口；测试覆盖 padded DRAM pitch、高 scratchpad 偏移和 guard 区。
7. **宿主编译与被测硬件性能需要分开。** SoC 模型默认采用 Clang/C++20，并以宿主 flags 文件追踪 Make 依赖；不修改生成的 C++ 模型。软件用 RV32F，关闭 fast math/FMA contraction，并检查反汇编中不存在融合/软件模拟浮点调用。

## 验证记录与性能

原实验完成记录在 `evidence/result.json`，对应日志保存在 `evidence/`。本次已经从此归档执行 `run.sh --rtl` 和 `run.sh --full`，完整重放的通过记录在 `evidence/reproduction/result.json` 及同目录日志中。重新生成的原始/修补后 RTL、`main.sv`、原始 Allo C++、MLIR、测试向量和三个软件 ELF 与原实验逐字一致，benchmark 周期数也一致；改造后的 HLS C++ 与归档中调整为相对 include 的版本逐字一致。归档清单记录当前文件与原路径的哈希；`check_archive.py` 同时确认两次通过版本的输入仍被完整保存。之后新的执行结果写到工作目录，与这些归档记录分开。

| 检查层 | 已通过内容 |
| --- | --- |
| Allo LLVM JIT | 512 个输出，与独立整数 IEEE oracle 比较 |
| 改造后的 HLS C 模型 | 3072 个输出 |
| 原生 HardFloat RTL | 6396 个结果，3-cycle/II=1，3256 次 CE stall、29 次 reset |
| HLS + HardFloat + RoCC 包装器 RTL | 3072 个输出；17 cycles 响应反压、目标寄存器保持、运行中 reset |
| 完整 Rocket benchmark | 两轮软件/硬件对照，共检查 2048 个输出 |
| 完整 Rocket 功能测试 | 32 tiles / 2048 个输出，可选 C、独立/原地写回、DMA padding/guard |
| 完整 Rocket 负例 | 故意破坏 golden，被正确拒绝，tohost/退出码为 1 |

原实验测得：

| trial | 软件 cycles | ISAX load | compute | store | ISAX total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 16553 | 1555 | 6977 | 606 | 9138 |
| 1 | 15813 | 1505 | 6951 | 606 | 9062 |

包含 DMA 的两轮平均加速比为 **1.77835×**。历史 HLS 对照为 serial 6289、interleaved 990、partitioned/vendor-scheduled 909 cycles；当前 native HardFloat 为 900 cycles。历史实现曾按 vendor 浮点延迟调度后替换算术叶子，当前实现直接按 native blackbox 延迟调度，因此 909→900 不能单独归因于 Allo schedule。比较文件供分析，不是当前复现的构建输入。

## 局限性

- 固定 FP32、8×8×8、特定数组布局和单条 RoCC 命令在途；没有 DMA/计算重叠、多核或多任务并发支持。
- 算术绑定和 RTL 修补针对本 kernel、Allo revision、Vitis HLS 2024.1 及 3-cycle/II=1 固定延迟合约。换 kernel、版本、存储器时延或可反压算子需重新检查/验证，不能直接复用这 7 个 stall 消除规则。
- 固定 RNE；未把 HardFloat 异常标记接入 CPU `fflags`，没有动态舍入模式支持。软件与硬件比较采用相同的逐次舍入语义。
- scratchpad 使用直接 RTL 数组描述，是否推断或替换为合适 SRAM 宏是后续 ASIC 集成工作；指令参数/越界行为没有做完整保护，调用者须遵循合法 ABI。
- HLS 指定 FPGA part 和时钟，blackbox FPGA 资源成本填写 0 占位。报告中的 LUT/FF/DSP/Fmax 不包含真实 HardFloat 成本，不能用作 ASIC 面积/频率结论。
- Vitis 不读取本项目 Liberty 来自动选择 HardFloat 流水深度或重定时。当前延迟来自现有原生 IP 合约；更换流水边界后必须更新合约并重跑 HLS 和 RTL 验证。
- 已证明该 `APSRocketConfig` 中的功能与周期行为；没有运行 ASIC 综合、门级仿真、物理实现或 STA，也没有证明任意 SoC 配置下的 DMA/缓存一致性。
- 完整复现依赖外部 LLVM/MLIR、Python 环境、Xilinx 安装和现有 Chipyard 工具链；归档提供具体依赖和版本，不是自包含的工具安装包。
