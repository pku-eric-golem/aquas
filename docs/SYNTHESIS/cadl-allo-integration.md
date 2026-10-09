# CADL 内嵌 Allo：语法与独立 RTL 接入路线

CADL 负责指令接口和数据搬运，Allo + Vitis 负责计算核及其调度。用户在 CADL 中定义 rs1/rs2/rd 的含义、CPU 内存访问、DMA、SPM 分配与读写，再把 SPM 视图绑定给内嵌的 Allo 设计。接口适用于不同名称、参数数量、数组形状和返回类型的计算核，没有 GEMM 专用语法。

**本次交付到前端为止**：语法、AST、结构检查、旧后端保护、编辑器支持、完整 GEMM 源码示例和本文。独立 wrapper 后端尚未实现，新的 `.cadl` 示例还不能生成 RTL 或执行 SoC 仿真。[手动接入归档](../../examples/allo_manual/README.md) 是后续实现的对照基线，其已有仿真结果不能算作新后端的通过结果。

## 1. 路径与职责

```text
CADL source
  ├─ rtype / register / static / load-store / DMA / invoke
  │    └─ 接口与绑定检查 → 控制 FSM + SPM + APS-Itfc SystemVerilog
  └─ 内嵌 Python 算法和 schedule
       └─ 独立 Allo 进程 → Vitis C++ → 原生端口与算术 IP 绑定 → Vitis RTL
                             │                                 │
                             └─ 接口 manifest ─────────────────┘
                                                │
                         main.sv + RTL 文件表 + aps_config.yaml
                                                │
                            现有 Chipyard APSRocketConfig / RoCC
                                                │
                                     C 软件 + SoC RTL 仿真
```

上图是**待实现的后端**。加速器生成不经过 Aquas 的 APS MLIR → HLS passes → TOR/SDC 调度 → CMT2 → FIRRTL 流程。复用的是 APS 外部接口约定、SoC 接入和验证设施；SoC 本身仍使用 Chipyard 的正常生成流程。

wrapper 仍需要控制状态机来等待内存、DMA 和计算核握手，但不对 Allo 运算进行第二次 HLS 调度。CADL 的语句顺序决定搬运和调用顺序；计算循环、流水和资源选择由 Allo schedule 与 Vitis 决定。

## 2. 新语法

### 声明算法与 schedule

```cadl
static samples: [i32; 256];
static output: [i32; 256];

allo add_bias with s """
import allo
from allo.ir.types import int32

def add_bias(X: int32[16], bias: int32) -> int32[16]:
    Y: int32[16]
    for i in range(16):
        Y[i] = X[i] + bias
    return Y

s = allo.customize(add_bias)
s.pipeline("i", initiation_interval=1)
""";

#[opcode(7'b0001011)]
#[funct7(0)]
rtype run(rs1: u5, rs2: u5, rd: u5) {
    let base: u32 = _irf[rs1];
    let bias: i32 = _irf[rs2];
    invoke add_bias(X = samples[base +: 16], bias = bias)
        -> output[0 +: 16];
    _irf[rd] = 0;
}
```

- `allo <name> with <schedule-name> <Python block>;` 是顶层声明。`name` 选择块内同名的顶层 Python `def`；`with s` 指定该 Python 模块导出的 schedule 对象。
- 每块是独立 Python 模块，可以包含 import、常量、辅助函数和完整 schedule。CADL 变量不会隐式注入 Python；运行时数据只通过 `invoke` 绑定传递。后续构建器必须检查导出的 schedule 确实对应所选 top。
- Python 块使用 `"""..."""` 或 `'''...'''`。相同三引号首次出现即结束 CADL 块，不提供转义结束符；内部文档字符串应使用另一种引号。CADL 不在块内解释 `//`、`/* */`、花括号或关键字。
- AST 保存含分隔符的原文、完整 source span，另提供去公共缩进的 Python 文本和位置映射。解析与 `check` 只读取 Python AST，不 import 或执行块内代码。

### 参数、返回值与 SPM 视图

```cadl
// 写入型数组参数仍是普通命名绑定；方向由 Allo 编译后的接口确定。
invoke transform(src = samples[0 +: 16], dst = output[0 +: 16]);

// 标量返回声明一个新的 CADL 局部值；不会自动写 rd。
invoke reduce(X = samples[0 +: 16]) -> sum: i32;
_irf[rd] = sum;

// 数组返回直接指定目标 SPM 范围。
invoke transform_return(X = samples[0 +: 16]) -> output[16 +: 16];

// 多返回值按 Python 返回顺序绑定，可混合数组和标量。
invoke analyze(X = samples[0 +: 16]) -> (output[0 +: 16], status: u32);
```

以上调用形式分别示意不同签名，所调用的核需要另行声明。所有 Python 形参都必须显式绑定，包括有 Python 默认值的形参；绑定顺序任意，不允许重复、遗漏或额外参数。不支持 `*args`、`**kwargs`。CADL 参数名是 Allo 形参的名字，不是 Vitis 改写后的端口名。

`spm[base +: count]` 表示连续的逻辑元素视图：

- `spm` 必须是未被局部同名变量遮蔽的全局 `static` 数组。用户声明容量和分区属性，显式管理内容；`invoke` 不隐式分配、复制、DMA 或写回 CPU 内存。
- `base`、`count` 的单位是**元素**，`count` 必须是显式正整数常量表达式。目前折叠整数字面量、加减乘、位运算和常量移位，不解析命名常量，也不折叠除法/取模。`base` 可以是运行时值。
- 多维固定形状按 row-major 展平，例如 Allo `float32[4, 8]` 对应 32 个连续元素；形状来自 Allo 签名，不在 CADL 重复定义。当前检查只检查 SPM 容量，**尚不核对 count 与 Allo shape/type**。
- Allo 的元素类型/位宽必须与 SPM 一致，不能隐式改变浮点格式。标量返回用 `name: type` 声明新局部值，数组返回只能绑定已有 SPM 范围；新局部值在本次调用的全部参数与结果地址求值之后才进入作用域。
- 两个形参或返回值可以指向同一 SPM。前端不保证这种别名安全；未来后端必须检查访问方向、重叠、bank 与端口冲突。GEMM 的 C/D 原地使用不能推导为任意核都支持原地返回。

### 顺序与接口语义

`invoke` 的硬件语义是阻塞调用：先完成前面的访问，求值并锁存标量、SPM 基址及返回目标地址；触发核；等待核结束并排空最终写入；锁存标量返回；再执行后续 CADL 语句。当前拒绝在 `spawn` 内使用 `invoke`，没有异步启动/等待语法。

rs1/rs2/rd 仍使用 `_irf` 显式读写。CPU 内存访问继续使用 `_mem[address]`；DMA 使用已有范围拷贝及 stride 指令。`_mem` 地址、`stride_x` 是字节，SPM 索引和拷贝长度是元素，`stride_y` 是无量纲布局参数。后端需要明确实现这些访问的请求、完成和反压语义，不能把发出请求视为完成。

## 3. 当前实现与验证

| 文件 | 职责 |
| --- | --- |
| `cadl_frontend/grammar.lark` | `allo`、不透明 Python 块、命名 `invoke` 与返回绑定语法 |
| `cadl_frontend/cadl_ast.py` | `AlloKernel`、`PythonSource`、`SourceSpan`、绑定与返回节点、`InvokeStmt`、`Proc.allo_kernels` |
| `cadl_frontend/parser.py` | 构造 AST、保存 span、重复声明与语法错误定位 |
| `cadl_frontend/allo.py` | 不执行 Python 的结构检查；禁止新语法误入旧后端 |
| `tools/aps-frontend` | 新 `check` 子命令；延迟加载旧编译器以分离 CIRCT 依赖 |
| `cadl_frontend/to_{mlir,c}/` | Python API 同样拒绝 Allo，不只在 CLI 拦截 |
| `examples/allo/gemm.cadl` | 完整 FP32 GEMM 与显式 load/compute/store 接口 |
| `tests/test_parser/test_allo.py` | 通用核、错误绑定、作用域、源码位置、不执行 Python、旧后端保护回归 |
| `editors/cadl/` | Tree-sitter、TextMate、Zed queries 与生成 parser；Python 块作为不透明区域 |

从仓库根目录运行：

```bash
pixi run python tools/aps-frontend check examples/allo/gemm.cadl
pixi run python tools/aps-frontend encoding examples/allo/gemm.cadl
pixi run python -m pytest tests/test_parser tests/test_mlir tests/test_transpile_to_c.py tests/hardfloat/test_frontend.py -q
cd editors/cadl
npm test
```

`check` 仅依赖 Python 前端及 Lark，不依赖 Allo、Vitis 或 CIRCT Python bindings。它检查 CADL 与 Python 语法、重复核声明、顶层函数与 schedule 名称存在、命名参数完整性、调用值可见性、SPM 范围形式/常量边界、返回局部变量冲突，以及能从 Python 注解确定的返回数量。schedule 名存在不代表对象类型正确；类型别名、表达式和返回签名仍需 Allo 实际编译确认。`check` 也不是已有 CADL 的完整类型检查器。

`encoding` 继续提取 rtype 编码，不执行 Python。`mlir` 和 `cadl2c` 在发现 Allo 声明（即便未使用）或嵌套调用时明确失败，使用 `-o` 时不会覆盖已有输出文件。不能通过旧命令得到一个静默漏掉计算核的结果。

## 4. GEMM 示例与手动版本的对应关系

[`examples/allo/gemm.cadl`](../../examples/allo/gemm.cadl) 内嵌手动版本的 `gemm_fp32` 算法和 schedule。固定 8×8 tile，FP32 分别乘、加，计算 `D = A × BTᵀ + (use_c ? C : 0)`。`reorder`、`buffer_at`、`pipeline` 和 `partition` 都在 Python schedule 内。

| CADL rtype | opcode / funct7 | 软件约定 |
| --- | --- | --- |
| `gemm8x8_fp32` | `0x0b / 0` | rs1 编码 A/BT tile 与 use_c；rs2 编码 C/D tile |
| `loadmat_a_8tile` | `0x7b / 0` | rs1 为 DRAM 字节地址；rs2 为 SPM 元素偏移和行跨度 |
| `loadmat_b_8tile` | `0x7b / 1` | 同上，目标为转置 B 的 SPM |
| `storemat_c_8tile` | `0x7b / 2` | 同上，从 C/D SPM 写回 |

三个 `[f32; 131072]` SPM 与手动版本的逻辑容量相同。每轮搬运八个相邻 tile 的一行，拆成两次 32 元素/128 字节的 burst，再执行八行；第二段 DRAM 加 128 字节、SPM 加 256 元素。`invoke` 中的 tile 地址乘 64，不能把 DMA 的元素偏移当作 tile 编号。

示例保留手动版本的 A tile 掩码、C/D 共用 SPM 及指令编码，作为后续与 [`software/benchmark_fp32.c`](../../examples/allo_manual/software/benchmark_fp32.c) 和 [`software/test_gemm.c`](../../examples/allo_manual/software/test_gemm.c) 对照的 ABI 基线。当前示例通过结构检查，尚未证明新生成 RTL 与该 ABI 一致。

## 5. 后续实现步骤与验收

### A. 独立构建入口与接口 manifest

新增独立 backend/CLI 入口，不把 Allo 声明 lowering 成 APS MLIR 或普通 CADL 函数调用。对 Python 块生成真实 `.py` 文件与 CADL source map，保留可供 `inspect` 读取的源码；在独立进程导入模块，取出指定 schedule，核对 top 后调用 Allo Vitis 后端。这里才会执行用户 Python，应使用用户已配置的 Allo 环境，并隔离其 MLIR bindings 与 Aquas/CIRCT bindings。

从 Allo IR、HLS 签名和综合接口报告生成并交叉核验 manifest，至少包含：逻辑参数/返回编号、输入/输出/读写方向、元素类型与形状、flatten 顺序、标量位宽、端口/银行映射、地址单位、读时延、写使能、控制/复位协议。每个阶段保留源码、报告、工具版本、schedule 和 IP 合约哈希；变化必须使缓存失效。

先用向量变换、reduction、二维数组和混合返回验证 manifest，再接 GEMM。不能从 `A/BT/C/D`、64 元素或生成 RTL 的特定信号名推导通用接口。

### B. 原生内存端口与算术合约

在 HLS C++ 接口层把顶层 AXI pragma 转为原生数组端口及 `ap_ctrl_hs`，重新综合，保留修改前后的产物。不要直接删 AXI RTL 状态机。第一版可约束为固定数组、固定读时延、单时钟；核内部局部数组继续由 Allo/Vitis 管理。

用户的 SPM 分区属性和 Allo 顶层数组的 partition/reshape 必须一起检查。内部 `buffer_at` 不等于替用户重新分配外部 SPM。`ap_memory` 通常没有逐次访问的 ready：不能在端口不够时插入任意仲裁并让 HLS 继续运行。第一版应拒绝不能证明合法的同时访问，或采用已经验证的全核停顿协议；不能只在 wrapper 上拉低一个未被核理解的 ready。

浮点绑定应在 HLS 调度前转换成 HardFloat blackbox 调用，并按 manifest 提供真实 latency、II、CE、reset 与端口合约。使用结构化的 Allo IR/C++ 转换与算子库，保留逐次舍入语义和对照模型。其他整数/存储算子也要检查最终 RTL 是否含 FPGA 专用原语。

手动基线仅覆盖 **3 个 FP32 运算点、2 个生成模块、8 处 RTL 修补**。其中 Vitis 2024.1 的 blackbox completion/重复端口问题尚不是通用解决方案。必须先寻找并验证稳定的 blackbox 控制协议，或实现有前置条件、结构校验和失败诊断的适配器；遇到未知结构应失败。不能把归档中的模块名或 stall 替换脚本推广成“支持任意 Allo 核”。

### C. CADL glue → FSM/SystemVerilog

直接将受支持的 wrapper 语句转成状态与握手：命令译码、局部/持久寄存器、标量整数地址运算、if、顺序循环、SPM 单元素读写、CPU load/store、DMA 和阻塞 invoke。每个等待操作有显式请求/等待/完成状态；普通组合地址表达式不进入 TOR/SDC。

首版一个命令在途：锁存命令和 rd，完成上一访问后发起下一访问；调用期间核独占已绑定 SPM，DMA 不与其并发。核的最终写入提交后才能发送指令完成，响应反压期间保持 rd/data 稳定。复位必须取消在途状态并使核、DMA 和 wrapper 回到一致状态。

生成器按 `SPM base + kernel logical address` 映射元素地址，再根据分区映射到物理 bank。动态基址、地址加法溢出、越界、别名以及端口占用必须在实现阶段校验；不能把当前 `check` 的通过当作运行时安全证明。动态越界的错误响应 ABI 需要在该阶段定义并加入软件负例，防止出现部分 DMA 写入后仍返回成功。

首版明确拒绝尚未实现的 wrapper 结构和调度指令，例如 CADL `pipeline/unroll`、并发 spawn、流式通道和任意跨 flow 调用；报错不能忽略它们或回退到 Aquas HLS。后续拓展仍走接口状态机路径。

### D. APS 打包与功能验收

遵循现有外部 RTL 的 `main` 模块/端口约定，生成 `main.sv`、核 RTL 文件表、`aps_config.yaml` 和指令编码。沿用现有 Chipyard 配置、软件构建和 RTL 仿真设施，产物写入独立构建目录，保留 Allo 源码、原始/改造 HLS、原始/改造 RTL 与版本信息。

验收顺序：

1. 通用接口：至少覆盖向量输入/输出、标量 reduction、固定二维数组、混合返回、一个核多次调用；不得依赖 GEMM 参数名。
2. 计算核：Allo LLVM/HLS C 模型与 RTL 对比；HardFloat 覆盖舍入、特殊值、CE 停顿与复位。
3. wrapper：覆盖 SPM 基址非零、DMA stride、边界/错误路径、C/D 别名、命令和响应反压、运行中复位。
4. 完整 SoC：复用手动版本 FP32 软件并对比 golden，保留故意破坏 golden 的失败用例，确认实际运行的是新生成 RTL。

验收到 SoC RTL 功能仿真即可，本阶段不要求运行 ASIC 后端。Vitis 的 FPGA part、时钟和资源报告不构成 ASIC PPA 结论；HardFloat 延迟合约也需要来自实际 IP，而非由 Vitis 自动读取 Liberty 推导。

## 6. 当前局限

- 支持通用固定签名的语法，不代表任意 Python/Allo 程序都可硬件化；Allo 自身限制仍然适用。
- 没有独立构建命令、RTL 生成器、运行时 SPM 检查、自动软件模型或新路径 SoC 通过记录。
- `check` 不执行类型表达式，不解析 Allo 真实读写效果、返回类型别名、bank、端口时序或物理冲突。
- 首版路线以固定形状、同步调用和单命令在途为目标；动态形状、stream/channel、异步多核、多命令并发、动态舍入模式与 CPU 浮点异常标记均需另行设计。
- 手动归档及其修补保持独立；任何新 kernel/schedule/工具版本都必须重新验证控制与数值行为。
