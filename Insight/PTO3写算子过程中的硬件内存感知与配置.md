# PTO3 写算子过程中的硬件内存感知与配置

> 本文基于 `repo/pto` 当前开发者文档、C++ 头文件和 Python DSL 实现整理，重点回答：在 PTO3 编写和降低算子的过程中，哪些环节需要感知硬件内存，涉及哪些参数，以及这些参数最终如何影响 Tile、搬运、分块、复用和运行时执行。

## 1. 总览

PTO3 的内存管理不是在某一个算子 API 中一次性完成，而是沿着编译流水线逐步确定：

```text
Tensor / Tile 语义
    ↓
target_memory、shape、layout、valid_shape
    ↓
InferTileMemorySpace       推导 Tile 所在内存空间
    ↓
AutoTileMatmulL0           根据 L0 容量选择 m/n/k
    ↓
ResolveBackendOpLayouts    修复后端要求的布局
    ↓
ExpandMixedKernel          划分 AIC/AIV 和跨核边界
    ↓
LowerPipelineLoops         生成多 stage / ping-pong 缓冲
    ↓
InitMemRef                 创建片上缓冲区描述
    ↓
MemoryReuse                按生命周期和别名规则复用缓冲
    ↓
AllocateMemoryAddr         分配实际片上地址
    ↓
PTO codegen / PTOAS         生成硬件 Tile 指令
```

因此，“内存配置”同时包括：位置、容量、布局、地址、生命周期、并发度和运行时 workspace。

## 2. 硬件内存层级

PyPTO 的 `MemorySpace` 主要包含以下空间：

| PyPTO 空间 | 硬件含义 | 典型用途 |
|---|---|---|
| `DDR` | 片外全局内存 / GM | 输入、输出 Tensor |
| `Vec` | UB / 向量统一缓冲区 | Vector 计算、普通 Tile load |
| `Mat` | L1 / CBUF | Cube 操作数 staging、矩阵中间结果 |
| `Left` | L0A | Matmul 左操作数 |
| `Right` | L0B | Matmul 右操作数 |
| `Acc` | L0C | Matmul 累加器 |
| `Bias` | Bias 缓冲区 | Matmul bias |
| `ScalarLocal` | 标量寄存器或 C 栈 | `ArrayType` |

两个基本规则是：

1. `TensorType` 默认并固定表示 `DDR` 数据；
2. `TileType.memory_space` 是 Tile 片上放置位置的唯一来源。

参考：[IR 类型与 MemorySpace](../repo/pto/docs/zh-cn/dev/ir/02-types.md)、[memory_space.h](../repo/pto/include/pypto/ir/memory_space.h)。

## 3. 算子编写阶段的主要场景

### 3.1 GM 到片上内存的加载

```python
x_vec = pl.tile.load(
    x, offsets=[0, 0], shapes=[64, 128], target_memory=pl.Mem.Vec
)

x_mat = pl.tile.load(
    x, offsets=[0, 0], shapes=[64, 128], target_memory=pl.Mem.Mat
)
```

相关参数：

| 参数 | 含义 |
|---|---|
| `offsets` | 源 Tensor 坐标系中的起始偏移 |
| `shapes` | 要加载的物理区域大小 |
| `valid_shapes` | 实际有效区域大小，可小于 `shapes` |
| `target_memory` | 目标片上空间，目前 `tile.load` 主要支持 `Vec`、`Mat` |

`tile.load` 不能直接把数据加载到 `Left`、`Right` 或 `Acc`。典型路径是：

```text
DDR → Vec/Mat → Left/Right/Acc
```

### 3.2 片上空间之间的数据搬运

```python
a_left = pl.tile.move(a_mat, target_memory=pl.Mem.Left)
b_right = pl.tile.move(b_mat, target_memory=pl.Mem.Right)
```

`tile.move` 既改变 memory space，也可能改变布局：

```python
fixed = pl.tile.move(
    x,
    target_memory=pl.Mem.Vec,
    blayout=pl.TileLayout.row_major,
    slayout=pl.TileLayout.none_box,
)
```

`InferTileMemorySpace` 会读取算子注册表中的输入约束。例如 matmul 输入需要 `Left`/`Right`，如果生产者只得到 `Vec`，编译器会自动插入对应的 `tile.move`。

### 3.3 零拷贝视图与真实搬运的区别

以下操作通常只改变视图或元数据，不复制数据：

- `tile.slice`
- `tile.reshape`
- `tile.transpose_view`
- `tile.set_validshape`

但如果硬件指令无法直接消费该视图，`CanonicalizeTileSlice` 会将其转换为 `tile.extract`，产生真实的数据搬运。例如 Mat 上喂给 matmul 的切片通常会变成：

```python
lhs_l0a = pl.tile.extract(lhs_mat, row, col, [m, k], target_memory=pl.Mem.Left)
rhs_l0b = pl.tile.extract(rhs_mat, row, col, [k, n], target_memory=pl.Mem.Right)
```

## 4. 内存布局与有效区域

### 4.1 TileView 参数

Tile 侧需要关注：

| 参数 | 作用 |
|---|---|
| `blayout` | block layout，如 `row_major`、`col_major` |
| `slayout` | scatter / box layout，如 `none_box` |
| `fractal` | 分形尺寸参数 |
| `stride` | 各维元素步长 |
| `valid_shape` | 有效数据区域 |
| `start_offset` | 视图在底层 buffer 中的起始偏移 |
| `pad` | 越过有效区域时的填充值模式 |

PTO codegen 会根据这些字段生成 `pto.alloc_tile` 的 `loc`、`rows`、`cols`、`blayout`、`slayout`、`fractal` 和 `pad` 属性。

### 4.2 TensorView 参数

Tensor 侧主要使用：

- `shape`
- `stride`
- `layout`：`ND`、`DN`、`NZ`
- `valid_shape`
- `pad`

`MaterializeTensorStrides` 会把隐式 stride 展开为显式 stride。`NZ` 是分形布局，主要用于 Tile，不适用于普通 Tensor 的 flat stride 表达。

### 4.3 `valid_shape` 与边界 Tile

```python
tile = pl.tile.load(
    x,
    offsets=[0, 0],
    shapes=[128, 128],
    valid_shapes=[100, 128],
)
```

此时物理缓冲区按 `[128, 128]` 规划，但只有 `[100, 128]` 是有效数据。该机制用于：

- 非整除矩阵的尾块；
- 动态 shape；
- padding；
- 分块后的边界区域。

## 5. Matmul 的内存感知

Matmul 是硬件内存参数最密集的场景。其典型映射为：

```text
A: [m, k] → Left / L0A
B: [k, n] → Right / L0B
C: [m, n] → Acc / L0C
```

### 5.1 L0 容量约束

`AutoTileMatmulL0` 根据后端参数选择 `(m, n, k)`。基本约束近似为：

```text
m × k × bytes_a ≤ L0A capacity
k × n × bytes_b ≤ L0B capacity
m × n × bytes_c ≤ L0C capacity
```

后端提供的关键参数包括：

| 参数 | 用途 |
|---|---|
| `GetL0aCapacityBytes()` | L0A / Left 容量 |
| `GetL0bCapacityBytes()` | L0B / Right 容量 |
| `GetL0cCapacityBytes()` | L0C / Acc 容量 |
| `GetMatCapacityBytes()` | Mat/L1 scratch 容量 |
| `GetL0FractalAlignment()` | M/N/K 的分形对齐粒度 |
| `GetMinL0TileDim()` | L0 单维最小合法尺寸 |
| `GetL0CostModel()` | L1↔L0 带宽、MAD 和 FIXPIPE 回写代价 |

当前文档中的典型容量：

| 后端 | L0A | L0B | L0C | Mat |
|---|---:|---:|---:|---:|
| Ascend910B | 64 KiB | 64 KiB | 128 KiB | 512 KiB |
| Ascend950 | 64 KiB | 64 KiB | 256 KiB | 512 KiB |

### 5.2 K 切分和 M/N 切分

- K 维超出 L0A/L0B 时，拆成 K-loop，在一个 `Acc` Tile 中累加；
- 输出 `[M, N]` 超出 L0C 时，进一步切分 M/N；
- 子块可以直接 `Acc → DDR`，也可以 `Acc → Mat scratch`。

因此，一个大矩阵乘的实际片上峰值不只由单个 Tile shape 决定，还取决于：

- K-loop 的 tile 数量；
- M/N 网格大小；
- 是否双缓冲；
- 是否保留 Mat scratch；
- 是否存在并发的输入面板和累加器。

### 5.3 Mat scratch

链式 matmul 可能把中间结果留在 Mat/L1：

```python
scratch = pl.tile.create(
    [128, 128], dtype=pl.FP32, target_memory=pl.Mem.Mat
)
```

只有在以下条件同时满足时才适合使用：

1. 中间结果的所有消费者都是矩阵乘操作数；
2. `[M, N]` scratch 不超过 `GetMatCapacityBytes()`；
3. Mat 的布局符合后续矩阵指令要求。

## 6. 流水线和双缓冲

```python
for ko in pl.pipeline(0, K, tile_k, stage=2):
    ...
```

关键参数是 `stage`，它决定循环体复制几份，以便预取下一阶段数据并与当前计算重叠。

影响内存峰值的因素包括：

- `stage` 或 `pipeline_stages`；
- 每个 stage 产生的 load Tile 数量；
- 每个 Tile 的 shape 和 dtype；
- 是否启用 L0C double buffer；
- 是否存在嵌套 pipeline。

需要区分两类数据：

- load 得到的输入 Tile：通常必须按 stage 保留独立 buffer；
- 串行化的 Acc 累加器：通常可以复用，除非显式启用真正的双累加器。

`LowerPipelineLoops` 会为克隆出的 Tile 标记 `pipeline_membership`，`MemoryReuse` 再据此禁止不安全的跨 stage 复用或根据容量降低有效并发深度。

参考：[LowerPipelineLoops](../repo/pto/docs/zh-cn/dev/passes/25-lower_pipeline_loops.md)、[MemoryReuse](../repo/pto/docs/zh-cn/dev/passes/30-memory_reuse.md)。

## 7. Buffer 生命周期、复用与原地安全

`MemoryReuse` 的复用条件主要包括：

- 生命周期不重叠；
- memory space 相同；
- 物理 buffer 大小足够；
- 不违反算子的 no-alias 约束；
- 不破坏 pipeline stage 的并发关系。

典型不可原地复用的算子包括：

| 算子 | 原因 |
|---|---|
| `tile.recip`、`tile.rsqrt` | 计算过程中需要同时读取输入和 scratch |
| `tile.row_sum/max/min` | 输出时仍要读取完整输入行 |
| `tile.transpose` | 边读边写会破坏尚未读取的数据 |
| `tile.sel` | 输出不能覆盖 mask 或 tmp |
| `tile.row/col_expand_*` | 广播向量会被后续行/列重复读取 |
| 升精度 `tile.cast` | 更宽的输出写入可能覆盖尚未读取的输入 |

当前实现允许不同 shape、dtype 或 TileView 的 Tile 共享同一个物理 MemRef，但仍必须满足上述别名和容量约束。

## 8. MemRef 和地址分配

### 8.1 InitMemRef

`InitMemRef` 根据 Tile 的 `memory_space` 创建 MemRef，并为非 DDR buffer 生成：

```text
tile.alloc(memory_space, addr=-1, size, id)
```

其中：

| 字段 | 含义 |
|---|---|
| `memory_space` | 片上空间 |
| `addr` | 起始地址，初始化阶段为 `-1` |
| `size` | 分配大小，单位为字节 |
| `id` | MemRef 标识 |

Tile view、`tile.store` 输出、`matmul_acc` 累加器和循环携带变量可能共享同一个 MemRef。

### 8.2 AllocateMemoryAddr

`AllocateMemoryAddr` 在 `MemoryReuse` 之后运行，负责：

- 按 memory space 分组；
- 跳过 DDR；
- 解析 `reserve_buffer(base=AUTO)`；
- 默认按 32 字节对齐；
- 默认按 MemRef ID 排序；
- 为 view 共享的物理 slot 分配统一基址；
- 回填 Tile/Tensor 类型和 `tile.alloc` 中的地址。

默认地址计算近似为：

```text
next_addr = align32(current_addr + size)
```

后端可以通过 `MemoryAllocatorPolicy` 自定义：

- `ShouldAllocate(space)`；
- `AlignAddress(addr, space)`；
- `OrderMemRefs(refs)`。

## 9. 跨核通信与同步 workspace

### 9.1 GM pipe buffer

在 Ascend910B 上，AIC/AIV 之间的 `tpush/tpop` 通过 GM 中转，需要额外的 `__gm_pipe_buffer`。

`InjectGMPipeBuffer` 会：

1. 给 AIC/AIV 函数增加 workspace 参数；
2. 沿调用图向上传播；
3. 在 orchestration 层注入占位 Tensor；
4. 由 codegen 根据 pipe 数量、方向和槽位计算实际 footprint。

Ascend950 的直连跨核通道通常不需要该 GM workspace。

### 9.2 `system.syncall`

`syncall` 有 hard 和 soft 两种模式：

| 模式 | 机制 | 关键参数 |
|---|---|---|
| `hard` | FFTS 硬屏障 | `core_type`、满占用、`sync_start=True` |
| `soft` | 共享 GM workspace 轮询 | `gm_workspace`、`used_cores`、UB/L1 scratch |

soft 模式的 GM workspace 需要约：

```text
used_cores × 8 × sizeof(INT32)
```

其中：

- `aiv_only`：GM workspace + Vec scratch；
- `aic_only`：GM workspace + 扁平 Mat scratch；
- `mix`：GM workspace + Vec scratch + Mat scratch。

Mat scratch 必须使用 `flat_layout=True`，不能使用普通 boxed NZ 布局，否则计数槽位置会错位。

## 10. 访存粒度和性能提示

算子设计不仅要考虑 Tile 能否放入内存，还要考虑访存粒度：

```text
innermost_dim_bytes = shape[-1] × sizeof(dtype)
```

后端相关参数包括：

| 参数 | Ascend910B | Ascend950 | 作用 |
|---|---:|---:|---|
| GM access granularity | 512 B | 128 B | GM 最小有效访问粒度 |
| L2 cache line | 512 B | 512 B | L2 cache line 大小 |
| 推荐最内维大小 | 512 B | 128 B | `tile.load/store` 的建议下限 |

如果 Tile 的最内维过窄，诊断 Pass 会产生 `PH001` 性能提示。通常应优先增大最内维，或调整数据类型和分块方向。

参考：[性能诊断](../repo/pto/docs/zh-cn/dev/passes/92-diagnostics.md)、[BackendHandler](../repo/pto/include/pypto/backend/common/backend_handler.h)。

## 11. 运行时 Ring 资源

除了算子数据 buffer，PTO3 运行时还需要 task ring 资源：

| `RunConfig` 参数 | 作用 | 约束 |
|---|---|---|
| `ring_task_window` | 在途 task slot 数量 | 2 的幂，至少 4 |
| `ring_heap` | 每个 ring 的 task 输出堆大小 | 2 的幂，至少 1024 字节 |
| `ring_dep_pool` | 依赖边池容量 | `[4, INT32_MAX]` |

这些参数不直接决定 Tile 的片上地址，但会影响任务并发、依赖图规模和运行时内存压力。

## 12. 写算子时的检查清单

### 内存位置

- 该数据应该位于 `Vec`、`Mat`、`Left`、`Right` 还是 `Acc`？
- 是否需要显式填写 `target_memory`？
- 是否会触发编译器自动插入 `tile.move`？

### 尺寸和容量

- `shape × sizeof(dtype)` 是否超过目标空间容量？
- Matmul 的 `(m, n, k)` 是否满足 L0A/L0B/L0C 约束？
- 是否需要 M/N/K 切分？
- 是否启用了双缓冲或多 stage？

### 布局和边界

- `layout`、`blayout`、`slayout` 是否满足目标 PTO 指令？
- 是否需要 NZ/fractal 布局？
- 是否正确设置 `valid_shape` 和 `pad`？
- `offsets`、`stride`、`start_offset` 是否处在正确坐标系？

### 生命周期和复用

- 两个 Tile 的生命周期是否重叠？
- 是否存在 no-alias 算子？
- 是否会错误地合并 pipeline 的不同 stage？
- view 是否安全地共享父 buffer？

### 通信和运行时

- 是否需要 `__gm_pipe_buffer`？
- `syncall` 应使用 hard 还是 soft？
- soft sync 的 `gm_workspace` 和 scratch 是否足够？
- `used_cores`、`sync_start` 和 `core_type` 是否匹配？
- ring 资源是否足以支撑任务并发？

## 13. 结论

PTO3 中最重要的内存设计顺序是：

```text
先确定数据流经过哪些内存层级
→ 再确定每一级 Tile 的 shape、dtype 和 layout
→ 再根据硬件容量选择分块和 pipeline stage
→ 最后由生命周期分析决定复用和实际地址
```

也就是说，`target_memory` 只是入口参数；真正的硬件内存感知还包括后端容量、分形对齐、访存粒度、布局约束、跨核 workspace、buffer 生命周期以及运行时 ring 资源。

