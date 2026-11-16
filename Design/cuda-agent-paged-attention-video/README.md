# CUDA Agent 生成 PagedAttention 算子：全流程还原视频

还原一名普通开发者使用 [BytedTsinghua-SIA/CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent) 生成 PagedAttention decode 算子的完整过程。视频约 3 分钟，1920×1080，30 fps。

- 成片：`cuda-agent-paged-attention.mp4`
- 可交互预览：用浏览器打开 `index.html`。空格键播放或暂停，←/→ 键跳转 5 秒，点击进度条可以跳到任意位置。

## 分镜（12 步）

| 步骤 | 画面与工具 | 内容 |
| --- | --- | --- |
| 01 准备环境 | Windows Terminal + SSH | 登录 H100 开发机，运行 `nvidia-smi`，检查 nvcc 和 torch 版本 |
| 02 获取 CUDA-Agent | Windows Terminal | 运行 `git clone`、`ls agent_workdir`、`cat utils/compile.sh` |
| 03 定义算子任务 | VS Code Remote-SSH | 把 `model.py` 替换为 PagedAttention 的 PyTorch 参考实现 |
| 04 下达任务 | VS Code 侧边 Agent 面板 | 引用 `model.py` 和 `SKILL.md`，描述验收目标 |
| 05 理解约束 | Agent 面板 + 集成终端 | 读取 SKILL.md 和 model.py，列出 kernels/、utils/ |
| 06 实现 v1 | 编辑器流式写入 | 删除 axpby 示例，写入 `kernels/paged_attention.cu`、`_binding.cpp`，修改 `model_new.py` |
| 07 编译与修复 | `TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh` | `__shfl_xor` 未定义导致编译失败，改用 `__shfl_xor_sync` 后编译通过 |
| 08 正确性校验 | `sudo python3 -m utils.verification` | 31/32 个头不匹配，定位到 GQA 的 kv_head 映射错误，修复后 5/5 PASS |
| 09 性能分析 | `utils.profiling` + Nsight Compute | v1 比 torch.compile 快 1.13×；ncu 显示 DRAM 吞吐只有 17% |
| 10 优化 v2 | 编辑器 + 终端 | 采用 Split-K 分区、GQA 共享 K/V 和 16B 向量化访存后，快 3.57× |
| 11 清理与复测 | 终端 | 按 SKILL.md 删除中间文件，然后依次重跑编译、校验和性能测试 |
| 12 审阅交付 | VS Code 源代码管理 | 开发者审阅改动；片尾展示结果卡 |

## 事实与示意的边界

**取自仓库（2026-11-14 读取）：**

- `agent_workdir` 的目录结构，包括 `SKILL.md`、`model.py`、`model_new.py`、`binding.cpp`、`binding_registry.h`、`kernels/`、`utils/`。
- 三条命令：`TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh`、`sudo python3 -m utils.verification`、`sudo python3 -m utils.profiling`。
- 日志格式：`Compiling N files: …`、`Compile success: cuda_extension.so`、`[TIME] Compilation took Xs`、`[PASS] check i/5`、`[PASS] verify success`，以及 `Torch Baseline: …us, Torch Compile: …us, CUDA Extension: …us`。
- SKILL.md 中的规则：只能在 `kernels/` 写原生 CUDA；`model_new.py` 不得使用 torch 算子；校验容差为 atol = rtol = 1e-2；性能至少比 torch.compile 快 5%（目标是 0.8×）；结束前清理中间版本。
- 来源：仓库 README、`agent_workdir/SKILL.md`、`utils/compile.sh`、`utils/compile.py`、`utils/verification.py`、`utils/profiling.py`，以及项目主页 cuda-agent.github.io。

**示意内容（非真实运行）：**

- 仓库没有公开 Agent 的启动脚手架（项目主页只说明是 “ReAct-style workflow with coding tools”）。因此视频用一个通用的 VS Code 侧边 Agent 面板来表示，工具名 bash、read、write、edit 为示意。
- PagedAttention 的 `model.py`、`kernels/*.cu` 和 `_binding.cpp` 是为本视频编写的示意实现，未在 GPU 上编译或运行。
- 所有耗时、性能数字、ncu 指标和 traceback 行号都是示意值。数量级按 H100 HBM3 带宽估算，约 335 MB 的 K/V 读取对应约 100 µs 的带宽下限。
- 视频中显示的 Agent 耗时（约 17 分钟）是模拟时钟，长时间运行的步骤带有“快进”标记。

## 重新渲染

依赖：Node.js、本机 Chrome、`playwright-core`、`ffmpeg-static`。

```bash
npm install --prefix <tools-dir> playwright-core@1.49.1 ffmpeg-static@5.2.0
```

```bash
NODE_PATH=<tools-dir>/node_modules node record.cjs --fps 30
```

```bash
NODE_PATH=<tools-dir>/node_modules node record.cjs --stills 12,60,128 --still-dir ./stills
```

最后一条命令只导出指定秒数的截图，用于检查画面。

## 文件

- `index.html`：舞台布局与样式，包括桌面、Windows Terminal、VS Code、Agent 面板和片头片尾。
- `engine.js`：确定性时间轴引擎。`render(t)` 只依赖时间 t，便于逐帧截图。
- `timeline.js`：分镜脚本。
- `sources.js`：视频中出现的源码与示例文件。
- `record.cjs`：逐帧截图并通过 ffmpeg 合成 H.264 MP4。
