# PyPTO User Skills 深度洞察报告

> **报告性质：源码洞察与报告模板**  
> 本报告聚焦 `pypto-user` 插件的五个 skills：`setup-and-run`、`generate-ir-trace`、`incore-profiling`、`dependency-redundancy`、`critical-path-analysis`。文中涉及具体命令、输入格式和安全约束，均来自仓库源码；没有执行真实设备、CANN 或 PyPTO 案例。

## 1. 执行摘要

`pypto-user` 的核心价值不是替用户“跑一个命令”，而是建立从环境、运行、编译过程到性能解释的证据链：

```text
环境可用性
  → 端到端运行有效性
  → IR lowering 过程可追溯
  → kernel 指令/pipe 行为可验证
  → 依赖图可约简性可证明
  → makespan 的因果归因
```

五个 skills 共同形成三个层次：

| 层次 | Skills | 解决的问题 |
|---|---|---|
| 运行准入 | `setup-and-run` | 环境是否正确、案例是否真的运行、失败属于环境还是模型 |
| 编译与执行观察 | `generate-ir-trace`、`incore-profiling` | IR 如何变化、kernel 是否执行了预期工作 |
| 调度与性能解释 | `dependency-redundancy`、`critical-path-analysis` | 哪些依赖可删除、总耗时究竟由什么限制 |

最重要的设计判断是：**退出码、文件生成、工具成功返回，都不能单独证明分析结论成立。必须验证输入身份、来源、完整性、运行阶段和结果语义。**

## 2. 研究范围与证据边界

### 2.1 直接证据

- [pypto-user plugin manifest](../repo/pypto-skills/plugins/pypto-user/.codex-plugin/plugin.json)
- [setup-and-run/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/setup-and-run/SKILL.md)
- [generate-ir-trace/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/generate-ir-trace/SKILL.md)
- [incore-profiling/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/incore-profiling/SKILL.md)
- [dependency-redundancy/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/dependency-redundancy/SKILL.md)
- [critical-path-analysis/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/critical-path-analysis/SKILL.md)
- [DFX capture contract](../repo/pypto-skills/plugins/pypto-user/lib/dfx/capture.md)
- profiling helpers：`incore_profile.py`、`gen_profiling_case.py`、`profile_extern_cce.py`

### 2.2 分析边界

以下是对 skill 设计的推断，不是一次真实运行结果：

- 没有据此断言某个模型、kernel 或设备存在性能瓶颈。
- 没有把示例数字当作真实 profiling 数据。
- 没有验证当前主机是否具备 CANN、msprof、camodel 或真实设备。
- 没有执行 skill 文档中要求的模型运行和 trace 生成。

## 3. 用户侧产品模型

### 3.1 用户真正要解决的不是“怎么运行”，而是“我能不能相信结果”

PyPTO 用户通常经历以下疑问：

1. 环境到底装对了吗？
2. 这个案例是真运行，还是只编译了？
3. 当前 HTML trace 来自哪一次源码和哪一个 dump？
4. profiling 为空，是 kernel 很快还是采集失败？
5. 删除依赖会不会破坏 tensor 生命周期？
6. makespan 高，是计算慢、数据没到、核心排队，还是调度出现空洞？

`pypto-user` 的五个 skills 正好对应这些疑问，形成由“事实确认”逐步走向“性能解释”的路径。

### 3.2 统一工作范式

每个 skill 都隐含相同的生命周期：

```text
Resolve 输入
  → Validate 来源和完整性
  → Run 最小必要操作
  → Validate 产物语义
  → Interpret 结果
  → Report 证据、限制和下一步
```

这使它们更像“诊断协议”，而不是普通帮助文档。

## 4. 五个 Skills 的角色分析

### 4.1 `setup-and-run`：运行准入和故障归因层

来源：[setup-and-run/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/setup-and-run/SKILL.md)

它采用四阶段阶梯：

```text
Stage 0 目标确认：模拟器/设备、模型、卡数、已有环境
Stage 1 环境：framework、torch、harness、assembler、Tile-ISA pin
Stage 2 Smoke：编译、输入、golden、运行、验证
Stage 3 模型阶梯：operator → 单层 → 多层 forward
Stage 4 结果解读：确认 pass 信号、交接给后续调试流程
```

关键洞察：

- **exit code 不等于 validated run。** simulator 可能走 compile-only 路径，必须检查输出阶段。
- **版本能运行不等于版本匹配。** assembler、framework、runtime、Tile-ISA 必须和 checkout pin 对齐。
- **失败先归因环境。** 在 pin 未对齐前，不应把 tile-op contract 或 runtime 错误归因于模型。
- **模型阶梯是诊断工具。** 每一级通过后，下一层失败才具备更强的定位价值。
- **运行具有资源伦理。** 不探测共享主机的“空闲设备”，只使用用户已分配的设备。

产品含义：这个 skill 把“启动模型”重新定义为一个有门禁的诊断流程，而不是一条命令。

### 4.2 `generate-ir-trace`：编译过程的来源证明层

来源：[generate-ir-trace/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/generate-ir-trace/SKILL.md)

它支持两条路径：

- 已有 `passes_dump`：转换为自包含 HTML。
- 没有 dump：运行指定案例、在新的空目录中捕获 dump，再转换报告。

它的核心不是 HTML 生成，而是 provenance 控制：

- 只使用用户指定的准确 dump，不按修改时间猜最近文件。
- 要求 dump 来自当前 worktree，而不是已安装包或历史构建。
- 运行前建立空的输出边界，运行后要求恰好得到一个新 dump。
- 报告需要验证 doctype、资源、trace data 和 JavaScript 的自包含性。
- 若目标路径已有报告，不应静默覆盖。

产品含义：IR trace 被定义为调试证据，必须能够回答“这份报告来自哪份源码、哪次运行、哪个输入”。

### 4.3 `incore-profiling`：防止“成功采集、错误解释”

来源：[incore-profiling/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/incore-profiling/SKILL.md)

该 skill 通过 bundled helpers 完成：函数发现、standalone case 生成、构建、`msprof op simulator` 采集、manifest 记录和 trace 清理。

其最有价值的部分是采集后的语义校验：

- matmul 应有非零 CUBE cycles。
- vector work 应有非零 VECTOR cycles。
- mixed kernel 应出现预期的多个 pipe 类别。
- instruction CSV 不能只有 scalar 或 synchronization 工作。
- worker 与 injection library 不应来自不匹配的 CANN 版本。
- near-empty 或 wrong-pipe trace 不能被解释成“kernel 很快”。

此外，它明确区分 synthetic single-core simulator 结果与真实设备结论，并要求报告真实设备测量仍需验证的部分。

产品含义：profiling 的完成标准从“文件导出成功”升级为“目标 workload 已被证明执行”。

### 4.4 `dependency-redundancy`：从图论约简走向数据流证明

来源：[dependency-redundancy/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/dependency-redundancy/SKILL.md)

它审计的是：某条显式依赖是否已经被更长路径隐含，从而具备删除候选资格。但它没有把普通 transitive reduction 直接当成安全删除证明，而是区分：

| 模式 | 含义 | 证据强度 |
|---|---|---|
| `reduced` | 普通图结构约简 | 只证明可达性层面的冗余 |
| `reduced_dataflow` | 结合 tensor dataflow 的约简 | 可进一步验证 creator edge 的数据流条件 |
| `omitted*` | 输出被省略的边 | 需要结合输入和模式解释，不能单独下结论 |

`creator` edge 被视为受保护边，因为它可能承担 tensor 生命周期；只有 dataflow 模式证明完整字节流转和 reuse generation 条件时，才可能成为可删除边。

产品含义：这个 skill 把“删除依赖”从一个图算法结果，提升为一个需要解释来源、深度、tensor metadata 的工程决策。

### 4.5 `critical-path-analysis`：把耗时变成因果解释

来源：[critical-path-analysis/SKILL.md](../repo/pypto-skills/plugins/pypto-user/skills/critical-path-analysis/SKILL.md)

该 skill 依赖 DFX capture 的 rank directory。完整 timing 分析需要同目录中的：

- `chip_swimlane_records.json` 或 legacy `l2_perf_records.json`
- `deps.json`
- `name_map*.json`
- 可选的 `merged_swimlane*.json`

它试图把 makespan 拆为：

- dependency-limited critical path
- as-executed path
- compute
- data-wait
- core-wait
- front-gap stall

因此它回答的不是“哪个 kernel 最大”，而是“总耗时由哪一种约束形成”。这支持更准确的归因：

| 观察 | 可能含义 | 不应直接推出 |
|---|---|---|
| static critical path 高 | 依赖图形成较高延迟下限 | 增加 core 一定无效 |
| data-wait 高 | 上游数据生产或传输影响明显 | consumer kernel 本身一定慢 |
| core-wait 高 | 资源竞争或调度排队 | 删除任意依赖都会加速 |
| front-gap 高 | 调度前端存在空洞 | 仅优化算术即可解决 |

产品含义：它把性能报告从指标罗列推进到“瓶颈类别 + 证据 + 下一步实验”。

## 5. 五个 Skills 的关系与使用顺序

推荐的端到端使用路径不是固定地执行全部 skills，而是按问题逐层深入：

```text
首次运行失败
  → setup-and-run

需要理解编译过程
  → generate-ir-trace

需要理解单个 kernel
  → incore-profiling

怀疑调度依赖过多
  → dependency-redundancy

需要解释整段运行的 makespan
  → critical-path-analysis
```

其中存在两条重要的证据汇合：

1. `dependency-redundancy` 给出可删除依赖的结构/数据流证据。
2. `critical-path-analysis` 判断这些依赖是否位于实际耗时路径上。

因此，“发现冗余边”不等于“优先删除冗余边”；只有当冗余边与 critical path、data-wait 或 fan-in 证据相交时，才值得进入优化实验。

## 6. 设计优势

### 6.1 对误报的防御强

五个 skills 共同防止以下常见错误：

- 把编译成功当作运行成功。
- 把 exit code 0 当作数值验证通过。
- 把错误版本的工具链问题当作模型 bug。
- 把历史 dump 当作当前源码产物。
- 把空 trace 当作快速 kernel。
- 把图结构冗余当作 tensor 生命周期安全。
- 把静态关键路径占比当作已经实现的性能收益。

### 6.2 适合 Agent 执行

文档对 Agent 特别友好，因为它明确了：

- 输入如何解析。
- 什么时候必须停止。
- 哪些信息可以推断，哪些必须询问。
- 产物应当保存在哪里。
- 报告必须包含哪些字段。
- 哪些结论需要真实设备重新确认。

### 6.3 把 repository-specific 信息留在消费者仓库

`setup-and-run` 不硬编码模型入口、flag、平台支持或案例清单，而是要求从消费者仓库的文档和 case inventory 获取。这是实现跨 `pypto` / `pypto-lib` 复用的关键。

## 7. 主要风险与改进机会

### 7.1 “五个 skill”之间还缺少机器可读的编排协议

当前关系主要写在 Markdown 中。建议为每个 skill 增加统一元数据：

```yaml
inputs: [build_output, passes_dump, rank_directory]
outputs: [html_report, cleaned_trace, markdown_report]
evidence_required: [provenance, freshness, semantic_validation]
hardware_required: false
next_skills: [critical-path-analysis]
```

这样 Agent 可以自动判断下一步，而不是依赖文本理解。

### 7.2 报告字段可以进一步统一

当前各 skill 的 Reporting 段落各自定义字段。建议统一包含：

| 字段 | 目的 |
|---|---|
| Input identity | 说明分析的确切输入 |
| Provenance | 说明来源、worktree、capture round |
| Environment | framework、runtime、assembler、SoC、CANN |
| Validation | 说明通过了什么检查 |
| Finding | 只写证据支持的现象 |
| Interpretation | 明确区分推断与事实 |
| Limitation | 说明 simulator、synthetic、warm-up 等限制 |
| Next experiment | 给出能证伪当前判断的下一步 |

### 7.3 profiling helper 需要更多契约测试

`incore-profiling` 携带了较大的 Python helpers，但现有仓库测试主要验证 skill 结构、静态合同和 shell 行为。建议补充 fixture-based 测试：

- 最小 `.pto` 输入。
- 缺少 `.pto` sibling 的失败场景。
- 动态 scalar 超过 allocation bound 的失败场景。
- near-empty、wrong-pipe、mixed-pipe trace。
- 不同 CANN worker/injection library 的版本冲突。

### 7.4 `setup-and-run` 的安全性可能带来较高交互成本

它在不确定时倾向停止，这是正确的安全选择，但初学者可能不知道下一步该提供什么。建议增加“缺失信息诊断卡”，一次列出：

- 需要的目标选择。
- 需要的设备分配信息。
- 需要的版本文件。
- 需要的最小案例。
- 当前已通过的 gate。

### 7.5 需要明确“分析证据”与“优化决策”的分界

依赖约简和关键路径分析都能产生强证据，但不能直接替代实验。建议报告中固定增加：

```text
事实：工具观察到了什么
推断：这说明什么
待验证：下一次运行需要确认什么
```

## 8. 推荐的用户体验闭环

对 PyPTO Studio 或类似工具，可以把这些 skills 产品化成四个面板：

### 环境面板

展示 framework、runtime、assembler、Tile-ISA、CANN 和 device allocation 的 gate 状态，并把版本 pin 差异直接标红。

### 运行面板

展示当前运行处于 compile、input、golden、runtime、validation 哪个阶段，避免用户只看到绿色 exit code。

### 证据面板

统一管理 passes dump、IR HTML、profiling manifest、cleaned trace、deps.json 和 capture provenance。

### 解释面板

把依赖冗余、critical path、data-wait、core-wait 和 front-gap 连接起来，显示“当前建议来自哪条证据”，并把优化建议标记为“已验证”或“待实验”。

## 9. 最终判断

`pypto-user` 已经具备一个相当清晰的性能工程方法论：

1. 用 `setup-and-run` 证明环境和运行链路。
2. 用 `generate-ir-trace` 证明编译过程和源码来源。
3. 用 `incore-profiling` 证明 kernel 确实执行了预期工作。
4. 用 `dependency-redundancy` 证明哪些依赖具备删除候选资格。
5. 用 `critical-path-analysis` 判断哪些约束真正影响 makespan。

它最大的产品机会，是把这些分散的 Markdown workflow 进一步升级为一个统一的“运行证据与性能解释系统”。核心不是增加更多分析命令，而是让用户能沿着同一条 provenance 链回答：

> 我分析的是什么？它是否真的运行了？结果是否可信？结论能否被下一次实验验证？

## 10. 交付检查清单

- [ ] 当前 framework、runtime、assembler、Tile-ISA pin 已记录
- [ ] smoke test 明确区分 compile-only 与 validated run
- [ ] 案例入口、参数、平台和设备集合已记录
- [ ] IR dump 与当前 worktree 的来源关系已验证
- [ ] HTML trace 为自包含文件并通过完整性检查
- [ ] profiling manifest、原始数据和清理后 trace 已保留
- [ ] CUBE/VECTOR/Scalar/Sync 等 pipe 证据已检查
- [ ] `reduced` 与 `reduced_dataflow` 的差异已解释
- [ ] critical path 与 as-executed path 已区分
- [ ] 所有结论都标注了事实、推断和限制
- [ ] 优化建议对应明确的下一次验证实验
