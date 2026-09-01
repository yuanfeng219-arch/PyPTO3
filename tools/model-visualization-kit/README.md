# Model Visualization & Operator Development Kit

一个面向模型可视化、算子理解和算子开发辅助的标准 MCP 工程。使用官方 MCP SDK，同时提供本地 STDIO 和远端 Streamable HTTP 入口。

## 当前能力

- `search_visual_assets`：按场景、框架和关键词检索可视化资产。
- `render_model_graph`：读取一个简单的模型 JSON，生成 HTML 模型拓扑图和结构化摘要。
- `render_swimlane`：读取 Chrome Trace/DFX `traceEvents`，生成可筛选、可点击的 HTML 泳道时间线。
- `render_python_model_graph`：静态读取 PyTorch/Python 模型源码，提取 class、`self` 子模块和 `forward` 调用，生成整网结构 HTML；不会执行源码。

标准 HTTP 入口支持 Streamable HTTP MCP：`POST /mcp`；生成文件可通过受保护的 `GET /artifacts/<filename>` 访问。云端模型源码应使用 `source_content` 和 `filename`，不要传调用者电脑上的本地路径。
- `inspect_operator`：根据算子名、输入 shape 和 dtype，输出算子契约、风险提示和推荐可视化。
- `generate_operator_scaffold`：生成一个最小的算子开发目录草稿，包含实现、测试和可视化元数据。

## 快速启动

在仓库根目录执行：

```powershell
node .\tools\model-visualization-kit\standard_server.mjs
```

该进程会等待 STDIO JSON-RPC 请求，不会向 stdout 打印日志；诊断信息写入 stderr。

## 接入 Codex

### 方式一：命令行添加

```powershell
codex mcp add model_visualization_kit -- node .\tools\model-visualization-kit\standard_server.mjs
codex mcp list
```

之后在 Codex 中输入：

```text
请使用 model_visualization_kit 分析 tools/model-visualization-kit/examples/resnet50.json，
生成模型拓扑图，并指出可能需要优先检查的算子。
```

### 方式二：项目配置

如果把本目录作为独立项目打开，可以复制 `codex.config.toml` 的内容到该项目的 `.codex/config.toml`。Codex 的桌面应用、CLI 和 IDE 扩展会共享同一主机上的 MCP 配置。

## 示例输入

```powershell
node .\tools\model-visualization-kit\demo_call.mjs
```

该命令会通过 MCP JSON-RPC 调用 `tools/list` 和 `tools/call`，验证服务发现与模型图渲染。

## 目录结构

```text
model-visualization-kit/
├── assets/manifest.json                 # 可检索资产索引
├── examples/resnet50.json               # 示例模型图输入
├── skills/model-visualization/SKILL.md  # Agent 调用规范
├── mcp_server.mjs                        # MCP STDIO Server（默认）
├── standard_server.mjs                    # 官方 SDK STDIO Server
├── standard_http_server.mjs               # 官方 SDK Streamable HTTP Server
├── package.json / package-lock.json       # SDK 依赖锁定
├── tests/                                 # 协议与输入安全冒烟测试
├── python_model_parser.mjs                # Python 模型源码静态解析与渲染
├── remote_server.mjs                     # 云端 HTTP MCP 入口
├── Dockerfile                             # 云端容器镜像
├── mcp_server.py                         # Python 标准库实现备选
├── demo_call.mjs                         # 本地协议冒烟测试
└── codex.config.toml                     # Codex 项目配置模板
```

调用示例：

```text
请使用 model_visualization_kit 的 render_python_model_graph 分析
Data/DeepSeek-V4-Flash-Official/model.py，生成整网结构可视化。
```

## 后续扩展

建议优先接入仓库中已有的真实数据解析器和页面渲染器，再逐步增加 ONNX、PyTorch、PTO/IR、DFX 和性能 trace 适配器。MCP 工具应返回结构化 JSON 与产物路径，避免把大型 SVG、日志或 trace 直接塞进对话上下文。

## 云端启动

```bash
docker build -t model-visualization-kit .
docker run -d --name model-visualization-kit \
  -p 8787:8787 \
  -e MCP_AUTH_TOKEN='请替换为长随机字符串' \
  model-visualization-kit
```

健康检查：`GET /healthz`。标准 MCP 地址：`POST /mcp`。模型源码云端调用使用 `source_content` + `filename`，生成的 HTML 可通过受保护的 `/artifacts/<filename>` 获取。生产环境仍应在前置代理配置 HTTPS 域名、限流和更严格的 CORS。
