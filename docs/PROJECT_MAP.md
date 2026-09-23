# PyPTO3 Project Map

面向在本仓库定位和修改页面的 coding agent。PyPTO3 是 PTO / PyPTO 的产品研究与浏览器原型工作仓库，不是单一应用：大部分页面是无需打包的静态 HTML，少数目录是独立的 React/Node 子应用。

## 先从这里开始

- `AGENTS.md`：本仓库协作约定；修改 `repo/pto/` 前还必须阅读该目录自己的 `AGENTS.md`。
- `launch.html`：Design Lab 的首页、分类/搜索逻辑，以及已发布 Demo 的路由注册表 `demos`。新增希望在首页出现的页面，要在此注册，而不只是新增 HTML。
- `.github/workflows/pages.yml`：GitHub Pages 静态发布清单。它将 `launch.html`、`Design/`、`inference/`、`Insight/` 和指定 `Data/` 子集复制到站点；不要假定仓库中的所有文件都会发布。
- `README.md`：仓库定位与本地静态预览方式。

## 目录职责

| 路径 | 职责 |
| --- | --- |
| `Design/` | 主要交互原型。通常每个子目录是一个独立页面，入口为 `index.html`，同目录的 `app.js` / `js/` / `styles.css` 或 `styles/` 负责交互和样式。|
| `inference/` | 推理、Serving 和 KV Cache 观察原型；它有自己的样式约定，见 `AGENTS.md` 的 `inference/` 规则。|
| `Insight/`、`Competitive_Analysis/`、`Product_Planning/`、`userResearch/` | 白皮书、竞品分析、规划材料和用户研究，不是应用运行时代码。|
| `Data/` | 模型源码快照、配置与 DFX/trace/pass 产物；是多个原型的数据来源，原则上把它作为输入归档而非页面实现位置。|
| `github_issues/` | PTO 相关仓库的 Issue 原始页和整理结果。|
| `repo/pto/` | PyPTO 源码镜像，仅用于研究和对照；不是 Design Lab 的页面代码。|
| `repo/pypto-skills/` | PyPTO 相关 Codex skill 的镜像。|
| `vendor/pto-design-system/` | 随仓库保存的 PTO 设计系统；提供 tokens、patterns 和基础样式。把它视作上游依赖，页面优先引用/复用，避免在此目录修改。|
| `tools/model-visualization-kit/` | 独立的 Node MCP 可视化服务，不属于静态 Design Lab。|
| `docs/` | 设计文档与本导航图。|

`components/` 当前没有可复用的受版本控制组件源码；跨页面没有统一 React component library。复用优先级为：`vendor/pto-design-system/` 的 pattern/token → 相邻 Demo 的既有实现 → 当前页面局部模块。

## 应用入口与页面

### 发布入口

- `/` 或 `/launch.html` → `launch.html`：设计 Demo 启动台；`demos` 数组是已发布页面的权威清单。
- GitHub Pages 由 `.github/workflows/pages.yml` 部署。根仓库没有统一 `package.json`、router 或 SPA shell。

### 主要静态页面（按功能分组）

| 场景 | 入口路径 | 主要实现/数据位置 |
| --- | --- | --- |
| 调优闭环与运行回放 | `Design/operator-tuning-console/index.html`；`Design/operator-run-overview/index.html`；`Design/operator-debug-tuning-lab/index.html`；`Design/pypto-studio-v2/index.html` | `operator-tuning-console/{app.js,data.js,build-data.cjs}`；各页面本地 `app.js`/`js/`/CSS；DFX 输入位于 `Data/DeepseekV4/` 和 `Data/_jit_decode_fwd_layers_20260625_184941/`。 |
| Kernel/算子开发 | `Design/Toolkit Studio/index.html`、`index_v2.html`、`index_v2多卡.html`、`index_v2-slz.html`；`Design/operator-coding-agent/index.html`；`Design/operator-object-workbench/index.html` | `Toolkit Studio/js/` 与 `styles/` 是该工作台的局部模块；其余页面以同目录 `app.js`、`styles.css` 为主。 |
| 编译、Pass 与控制流 | `Design/compile-failure-diagnosis/index.html`；`Design/pass-transform-explorer/index.html`；`Design/pass-atlas/index.html`；`Design/pass-decision-studio/index.html`；`Design/llvm-flow/llvmcfg-standalone.html` | `pass-transform-explorer/{build.mjs,lib/}` 从 `Data/**/passes_dump` 生成页面数据；`pass-atlas/`、`pass-decision-studio/` 为静态 JS 页面；LLVM 见下方独立子应用。 |
| 内存与性能 | `Design/gdr-memory-optimizer/index.html`；`Design/memory-viz-demo/index.html`；`Design/memory-inspector/Memory_V1.html`、`Memory_V2.html` | 每页的 `js/` 和 `styles/`；`Design/assets/` 存放部分内存图及页面预览资源。 |
| 模型与推理 | `Design/model-inference-studio/index.html`；`Design/architecture/deepseek_v4_flash_csa/workbench.html`；`inference/serving-observability-ide.html`；`inference/kv-cache-explorer/index.html` | 架构工作台的 `mapping-registry.js`、`source-manifest.js`、`model_architecture*.json`；推理共享读取器为 `inference/dfx-real-data.js`。 |
| 图谱、重放及独立实验 | `Design/decode-layer-graph/index.html`；`Design/flash-attention-sync-replay/index.html`；`Design/deepseek-gap-investigator/index.html` | Decode Graph 使用 `js/{graph.js,graph-data.js}`；Gap Investigator 使用 `project-manifest.json`、`data.js`、`investigation/`、`render/`。后者目前未在 `launch.html` 注册。 |
| Insight 页面 | `Insight/*/index.html`、`Insight/nvidia_dynamo_ux_product_analysis.html` | 自包含 HTML/CSS；启动台只注册其中部分白皮书。 |

## 独立可构建子应用

- `Design/operator-performance-control-room/`：Vite + React 19。入口 `src/main.jsx`，主界面 `src/App.jsx`，样式 `src/styles.css`，部署兜底 Worker 为 `worker/index.js`，测试为 `tests/sites-worker.test.mjs`。它不是根 `launch.html` 的当前注册路由。
- `Design/llvm-flow/`：发布给 Design Lab 的静态控制流页面是 `llvmcfg-standalone.html`；其来源前端在 `llvm-flow-frontend/`（CRACO/React）。前端入口 `src/index.tsx`，页面选择由 `src/App.tsx` 和 `src/components/pages/` 管理，状态在 `src/redux/`，静态证据在 `src/data/`、`src/exData/`。顶层 `package.json` 仅服务静态工作台的本地启动和契约检查。
- `tools/model-visualization-kit/`：独立 MCP server；入口 `standard_server.mjs`（stdio）和 `standard_http_server.mjs`（HTTP），测试在 `tests/`。不要把它当作网页路由。

## 核心能力与数据流

1. **启动台路由流**：`launch.html` 的 `demos` 配置 → 链接到 `Design/` / `inference/` / `Insight/` 静态入口 → 页面加载局部 JS/CSS 与数据。
2. **DFX 可视化流**：`Data/**/dfx_outputs/` 的 `deps.json`、`merged_swimlane_*.json`、`name_map*.json`、`distributed_meta.json` → `inference/dfx-real-data.js` 或各 Demo 局部 adapter → 图、泳道、依赖/性能诊断 UI。数据读取依赖 HTTP `fetch`，本地必须经 HTTP server 打开，不能直接用 `file://`。
3. **模型/源码映射流**：`Data/DeepSeek-V4-Flash-Official/` 与 `Data/DeepseekV4/deepseek_v4_flash_dspark/` → `Design/architecture/deepseek_v4_flash_csa/{mapping-registry.js,source-manifest.js,workbench.js}` → 架构、源码和执行产物之间的交叉导航。
4. **Pass 分析流**：`Data/**/passes_dump/` → `Design/pass-transform-explorer/build.mjs` 与 `lib/` 分析脚本 → 页面使用的数据文件/`app.js`。带有“generated/auto-generated”标记的结果不要手改，应回到构建脚本或输入数据。
5. **发布流**：推送 `main` / `insight` 的受监控路径 → Pages workflow 复制静态文件与有限数据集 → GitHub Pages。对需要新数据的页面，同时检查 workflow 是否复制了对应数据路径。

## 样式、组件与配置

- **设计 token / 基础样式**：`vendor/pto-design-system/tokens/{foundation.css,semantic.css,components.css}`；`launch.html` 已直接引用它们。可复用工作台 pattern 位于 `vendor/pto-design-system/patterns/`，尤其是 `ide-frame/`、`workbench-shell/` 等。
- **页面样式**：没有仓库级全局 app CSS。静态页面通常在本目录 `styles.css`、`styles/` 或内嵌 `<style>`；改页面先找同目录入口中加载的样式。
- **页面专属数据**：常见命名为同目录 `data.js`、`js/*-data.js`、`*.json`；例如 `Design/operator-tuning-console/data.js`、`Design/pass-atlas/data.js`、`Design/decode-layer-graph/js/graph-data.js`。
- **静态资产**：跨页面预览图集中在 `Design/assets/launch-previews/`，其他页面图片通常与入口同目录。
- **发布配置**：`.github/workflows/pages.yml`；Vite/React 子应用的配置在各自目录的 `vite.config.mjs`、`package.json`，不能套用到整个仓库。

## 常用开发、构建与验证

从仓库根目录预览静态页面：

```bash
cd /Users/songchenfei/Documents/pypto项目/PyPTO3
python3 -m http.server 8000
# 浏览 http://localhost:8000/launch.html
```

独立项目的命令应在相应目录执行（依赖已安装时）：

```bash
# Operator Performance Control Room
cd Design/operator-performance-control-room
npm run dev
npm run build
npm run test:sites

# LLVM 静态工作台：启动与 UI 契约检查
cd ../llvm-flow
npm run start
npm run check

# LLVM React 源码前端
cd llvm-flow-frontend
npm start
npm run build
npm test

# Model Visualization Kit MCP server
cd ../../../tools/model-visualization-kit
npm run start:stdio
npm test
```

根仓库没有统一 build/test 命令。对纯静态页面，最低验证是通过 HTTP server 打开对应入口并检查浏览器控制台与关键交互；对新增发布页面，再检查 `launch.html` 注册和 `.github/workflows/pages.yml` 的复制范围。

## 不要误入的区域

检索和改动页面时默认忽略 `**/node_modules/`、构建产物（`dist/`、`build/`）、自动生成文件及 `vendor/` / `repo/` 的第三方或镜像代码。它们只有在需要追溯依赖、重新生成数据，或用户明确要求时才进入修改范围。
