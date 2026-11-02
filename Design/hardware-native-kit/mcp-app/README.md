# Hardware-Native MCP App

提供 `show_hardware_native` 工具，把现有 Kit 作为 MCP Apps 界面资源交给支持该协议的 Agent 客户端渲染。服务通过 STDIO 工作，不启动网络端口，也不读取用户工程文件。

## 安装与连接

需要 Node.js 20 或以上。在本目录运行：

```powershell
npm ci --legacy-peer-deps
```

在支持 **MCP Apps 和本地 STDIO MCP server** 的客户端中，添加以下服务（配置格式按客户端调整；请将路径换成本机绝对路径）：

```json
{
  "mcpServers": {
    "hardware-native-kit": {
      "command": "node",
      "args": ["D:/project/PyPTO3/Design/hardware-native-kit/mcp-app/server.mjs"]
    }
  }
}
```

重启或重新连接客户端，在对话中输入：

> 使用 hardware-native-kit 展示 fa_fused 的硬件落点，将 SEQ_TILE 设为 256，FA_STAGE 设为 2。

Agent 调用 `show_hardware_native` 后，支持 MCP Apps 的客户端读取关联的 `ui://hardware-native-kit/workbench.html` 并显示交互界面。界面资源打包了本地样式和脚本，不依赖网络。源码行、逻辑节点、硅片区域和缓冲区支持联动；面板中可调整参数和切换 kernel。

## 工具参数

- `kernel`：`fa_fused`（默认）或 `matmul_64`。
- `knobs`：可选数字对象。`fa_fused` 支持 `SEQ_TILE`（64、128、256）、`FA_STAGE`（1–4）、`NUM_KV_HEADS`（4、8）、`HEAD_DIM`（64、128）；`matmul_64` 支持 `TILE`（32、64、128、256）。无效组合返回工具错误。

调用示例：`{"kernel":"fa_fused","knobs":{"SEQ_TILE":256,"FA_STAGE":2}}`。

UI 中后续的本地切换和调参不会自动同步回 Agent 上下文。估算结果来自 manifest，不表示编译通过或真实设备结果。客户端若只支持 MCP 工具、不支持 MCP Apps，只能看到工具文本结果，不会显示嵌入式界面。此服务不会自动配置到当前 Codex 或其他客户端。

## 验证

```powershell
npm test
```

测试通过 MCP 内存传输检查工具发现、UI 资源读取和参数校验；目标客户端的实际嵌入显示仍需在该客户端验收。

## 导入 CSA qk_pv

`import_csa_qk_pv` 接收本机 `source_path` 和 `dump_path`，分别指向含 `name_hint="qk_pv"` 的 Python 源码与 `36_after_AllocateMemoryAddr.py` 输出。可用仓库样例：

```text
source_path: D:/project/PyPTO3/Data/DeepseekV4/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py
dump_path: D:/project/PyPTO3/Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/36_after_AllocateMemoryAddr.py
```

工具只读取文件，不执行源码。编译输出中的地址与大小作为证据显示；源码节点与输出按名称匹配，节点间依次连接用于阅读，均为推断，不代表已验证的数据依赖。尚不检查两文件是否来自同一次编译；未识别地址不会纳入高水位。平台容量和保留区沿用示例口径，不能当成编译器判定或上板验证。当前仅支持 CSA qk_pv 样例格式。
