# PyPTO `memory_map.py` Tile 异常判断原理

## 1. 文档目的

`memory_map.py` 用于读取 `passes_dump/NN_after_<Pass>.py`，将每个片上内存 Tile 转换为“地址 × 生命周期”二维矩形，并检查内存复用、内存视图、地址冲突及容量越界。

本文说明：数据从哪里来、生命周期如何计算、什么情况属于正常复用、什么情况会被判定为异常，以及页面中的视觉标记如何解读。

## 2. 整体流程

```text
Pass dump Python 文件
        │
        ▼
Python AST 解析
        │
        ├─ 识别 AIC / AIV / InCore 计算函数
        ├─ 提取 Tile 类型、MemRef、shape、dtype 和 producer
        └─ 统计每个 SSA 名称的首次定义行与最后使用行
        │
        ▼
Tile → Box
        │
        ├─ 合并同一 slot 的连续 SSA alias
        ├─ 保留生命周期不相交的复用 Box
        └─ 检查地址与生命周期的二维重叠
        │
        ▼
Memory Map
        ├─ 横轴：内存地址 offset
        ├─ 纵轴：源码行号 / 静态生命周期
        └─ Vec、Mat、Left、Right、Acc 分空间展示
```

## 3. Tile 数据模型

工具识别带 `pl.Tile` 类型注解且拥有片上 `MemRef` 的值，例如：

```python
t__tile: pl.Tile[
    [16, 1024],
    pl.FP32,
    pl.MemRef(mem_vec_5, pl.const(0, pl.INT64), 65536),
    pl.Mem.Vec,
] = pl.tile.load(...)
```

解析后得到：

| 字段 | 含义 |
|---|---|
| `name` | Tile 的 SSA 名称 |
| `space` | Vec、Mat、Left、Right、Acc 等内存空间 |
| `base` | 底层分配对象，如 `mem_vec_5` |
| `offset` | 在内存空间中的起始地址 |
| `size` | 占用字节数 |
| `shape` / `dtype` | 数据形状与类型 |
| `op` | 产生 Tile 的操作 |
| `start` | 首次定义或出现的源码行 |
| `end` | 最后一次读取该名称的源码行 |

解析入口是 `parse_dump()`；单个 Tile 由 `_tile_from_annotation()` 构造。

## 4. 生命周期如何计算

这里的生命周期是静态源码生命周期，不是硬件 cycle：

```text
Tile lifetime = [首次定义行, 最后一次读取行]
```

处理步骤：

1. 函数参数中带 Tile 注解的值，从参数所在行开始存活。
2. `AnnAssign` 形式的 Tile 赋值，从赋值语句所在行开始存活。
3. 遍历函数中全部 `ast.Name` 读取节点，将 `end` 延长到最后一次使用。
4. 同名 SSA 在不同分支中重复绑定时，生命周期覆盖所有分支中的最早定义和最晚使用。

页面纵轴直接使用 dump 的源码行号，因此点击代码行时，可以找到该行上所有满足以下条件的 Tile：

```text
tile.start <= current_line <= tile.end
```

## 5. Alias 合并与正常内存复用

### 5.1 相同 slot

以下四项相同的 Tile 被认为使用同一物理 slot：

```text
(space, base, offset, size)
```

### 5.2 连续或重叠的 SSA alias

若同一 slot 的两个 SSA 生命周期相接或重叠：

```text
next.start <= current.end
```

工具会将其合并成一个 Box，并把其他名称记录在 `aliases` 中。常见来源包括 phi、yield 和循环携带值。

### 5.3 正常内存复用

如果两个 Tile 地址相同，但生命周期不相交：

```text
A.end < B.start
或
B.end < A.start
```

它们会保留为两个独立 Box。这表示同一片物理内存在不同时间被重新使用，是正常且期望的内存复用。

## 6. 异常判断核心

两个 Box 只有在“地址”和“生命周期”同时重叠时才需要进一步检查。

### 6.1 地址重叠

Tile 的地址区间采用半开区间：

```text
[offset, offset + size)
```

两个地址区间重叠的条件为：

```text
A.offset < B.offset + B.size
并且
B.offset < A.offset + A.size
```

### 6.2 生命周期重叠

```text
A.start <= B.end
并且
B.start <= A.end
```

### 6.3 同 base：合法 View

地址与生命周期同时重叠，但两者 `base` 相同：

```text
A.base == B.base
```

这通常是 `tile.slice`、`transpose_view` 或父 Tile 的局部视图。工具将较窄的一方标记为 `view = true`，页面使用虚线框展示，不计入异常。

### 6.4 不同 base：异常冲突

地址与生命周期同时重叠，且两者来自不同底层分配：

```text
A.base != B.base
```

这意味着两个独立分配在同一时间占用了相同字节，工具将两边标记为 `conflict = true`。页面使用红框展示，并计入“冲突”和“异常”统计。

## 7. 容量越界

工具从 PyPTO Backend/SoC 读取各片上内存空间容量，并计算函数的高水位：

```text
high_water_mark = max(tile.offset + tile.size)
```

单个 Tile 的越界判断为：

```text
tile.offset + tile.size > space.limit
```

当前 V2 页面使用 Ascend 910B `a2a3` 容量：

| 空间 | 安全容量 |
|---|---:|
| Vec | 184 KB |
| Mat | 512 KB |
| Left | 64 KB |
| Right | 64 KB |
| Acc | 128 KB |

越界 Tile 计入“越界”和“异常”统计。高水位达到容量的 95% 以上属于容量压力提示，但不等同于已经发生冲突。

## 8. 未完成地址分配

在 `AllocateMemoryAddr` 之前，多个独立 base 可能全部位于 offset 0。若同一内存空间存在两个及以上不同 base 都停在 offset 0，工具会认为地址尚未真正分配，并要求使用 `AllocateMemoryAddr` 或之后的 pass dump。

该检查用于避免把“尚未分配地址”误显示成大量内存冲突。

## 9. 页面视觉含义

| 页面表现 | 含义 | 是否异常 |
|---|---|---|
| 普通实线 Tile | 独立 Tile live range | 否 |
| 同地址、不同纵向位置 | 生命周期不相交的正常复用 | 否 |
| `+N` | 多个连续 SSA alias 已合并 | 否 |
| 虚线框 | 同一 base 上的合法 View | 否 |
| 红框 | 不同 base 在地址与生命周期上同时重叠 | 是 |
| 越界统计 | Tile 结束地址超过硬件空间容量 | 是 |
| 黄色源码行标记 | 当前点击的源码行 |
| 多个增强 Tile | 当前源码行上同时存活的 Tile |

## 10. 当前 Demo 数据结果

对 `32_after_AllocateMemoryAddr.py` 的解析结果：

- 38 个 AIC/AIV/InCore 计算函数
- 697 个合并后的 Tile Box
- 108 个合法 View
- 0 个跨 base 冲突
- 0 个容量越界

因此，按本工具的静态判断规则，当前 dump 没有明确的 Tile 内存异常。

## 11. 判断边界与注意事项

1. 纵轴是源码行号，不是设备真实执行 cycle。
2. 循环、分支和异步流水会被静态生命周期保守近似。
3. 同 base 重叠默认解释为 View；若上游错误地让不相关 Tile 共用 base，需要结合 IR 语义进一步确认。
4. 静态检查无法替代运行时事件、同步依赖和实际硬件 trace 分析。
5. 当怀疑异步流水导致同时存活时，应结合 msprof、Runtime Trace 或事件依赖检查。

## 12. 快速判定表

```text
地址重叠？
  ├─ 否 → 正常
  └─ 是
      └─ 生命周期重叠？
          ├─ 否 → 正常内存复用
          └─ 是
              └─ base 相同？
                  ├─ 是 → 合法 View
                  └─ 否 → 异常冲突

另外检查：offset + size 是否超过对应空间容量。
```

