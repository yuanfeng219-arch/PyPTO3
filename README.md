# PyPTO3

PyPTO3 是一个围绕 PTO / PyPTO 生态进行产品设计、用户体验分析和可视化原型探索的工作仓库。仓库同时收录了参考源码、GitHub Issue 归档与 PTO 设计系统。

## 仓库目录结构

```text
PyPTO3/
├── .obsidian/                     # 根目录 Obsidian 工作区配置
├── Design/
│   ├── memory-viz-demo/          # 内存可视化交互原型
│   └── Toolkit Studio/           # 可信 Kernel Agent 工作台
├── Insight/                      # 产品、仓库与用户体验洞察报告
├── github_issues/                 # PTO 生态各仓库的 Issue 数据归档
│   ├── pto/
│   ├── PTOAS/
│   ├── pto-isa/
│   ├── pypto-lib/
│   ├── pypto-serving/
│   ├── simpler/
│   └── pypto_top_level_documents/
├── repo/                          # 用于研究和对照的源码镜像
│   └── pto/                    # PyPTO 核心源码、文档、示例与测试
├── tools/                         # 仓库维护工具（包含 GitHub CLI）
├── vendor/
│   └── pto-design-system/      # PTO 设计系统、组件模式与设计令牌
├── launch.html                    # Design Demo 统一启动台
└── README.md
```

## 启动设计 Demo

**在线访问：[PTO3 Design Lab](https://yuanfeng219-arch.github.io/PyPTO3/)**

Launch 页面由 GitHub Pages 自动发布，可在线浏览并启动 Toolkit Studio 与其他设计 Demo。

如需本地运行，在仓库根目录启动任意静态 HTTP 服务器，然后访问 `launch.html`：

```bash
python -m http.server 8000
```

打开 `http://localhost:8000/launch.html` 即可浏览、搜索并启动 `Design/` 下的交互 Demo。

## 主要内容

- `.obsidian/`：保存以仓库根目录为 Vault 的 Obsidian 工作区配置。
- `Design/`：存放可直接在浏览器中运行的设计原型。
- `Insight/`：汇总 PyPTO 及关联仓库的体验分析与产品洞察。
- `github_issues/`：按项目保存 Issue 的 CSV、JSON、Markdown 及原始分页数据。
- `repo/pto/`：保存 PTO/PyPTO 代码镜像，便于产品分析、设计对照和原型验证。
- `vendor/pto-design-system/`：提供 PTO 产品的视觉基础、交互模式、页面示例和主题工具。

> 上述目录树以当前已纳入 Git 版本控制的内容为准，不包含本地忽略文件和空目录。
