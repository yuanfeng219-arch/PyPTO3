import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { registerAppTool, registerAppResource, RESOURCE_MIME_TYPE } from '@modelcontextprotocol/ext-apps/server';
import { z } from 'zod';
import { importCsaQkPv, toCsaManifest } from './csa-parser.mjs';

const appDir = path.dirname(fileURLToPath(import.meta.url));
const kitDir = path.resolve(appDir, '..');
const resourceUri = 'ui://hardware-native-kit/workbench.html';
const options = {
  fa_fused: { SEQ_TILE: [64, 128, 256], FA_STAGE: [1, 2, 3, 4], NUM_KV_HEADS: [4, 8], HEAD_DIM: [64, 128] },
  matmul_64: { TILE: [32, 64, 128, 256] },
};

async function htmlResource() {
  const files = ['kit/platform.js', 'kit/core.js', 'kit/view.js', 'kit/kit.js', 'kernels/fa_fused.js', 'kernels/matmul_64.js'];
  const [css, ...scripts] = await Promise.all(['styles/kit.css', ...files].map((file) => fs.readFile(path.join(kitDir, file), 'utf8')));
  const bundle = await build({ entryPoints: [path.join(appDir, 'view.js')], bundle: true, write: false, format: 'iife', platform: 'browser', target: 'es2020' });
  const js = [...scripts, bundle.outputFiles[0].text].join('\n;\n').replace(/<\/script/gi, '<\\/script');
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Hardware-Native Kit</title><style>html,body{margin:0;background:#111}#connection-status{color:#ababab;font:13px sans-serif;padding:16px}${css}</style></head><body><div id="connection-status" role="status">正在连接 Agent…</div><div id="kit"></div><script>${js}</script></body></html>`;
}

export function createServer() {
  const server = new McpServer({ name: 'hardware-native-kit', version: '1.0.0' });
  registerAppTool(server, 'show_hardware_native', {
    title: '展示算子的硬件落点',
    description: '在支持 MCP Apps 的客户端展示源码、算子逻辑和硅片落点的交互工作台。当前支持 fa_fused 和 matmul_64；容量为编码期估算。',
    inputSchema: { kernel: z.enum(['fa_fused', 'matmul_64']).default('fa_fused'), knobs: z.record(z.number()).optional().describe('可选参数：fa_fused 支持 SEQ_TILE、FA_STAGE、NUM_KV_HEADS、HEAD_DIM；matmul_64 支持 TILE。') },
    annotations: { readOnlyHint: true, destructiveHint: false, openWorldHint: false },
    _meta: { ui: { resourceUri } },
  }, async ({ kernel, knobs = {} }) => {
    for (const [key, value] of Object.entries(knobs)) {
      if (!options[kernel][key] || !options[kernel][key].includes(value)) {
        return { isError: true, content: [{ type: 'text', text: `不支持 ${kernel} 的参数 ${key}=${value}。可用值：${JSON.stringify(options[kernel])}` }] };
      }
    }
    return { content: [{ type: 'text', text: `已准备 ${kernel} 工作台。容量为编码期估算，未经编译或上板验证。` }], structuredContent: { kernel, knobs } };
  });
  registerAppTool(server, 'import_csa_qk_pv', {
    title: '导入 CSA qk_pv 源码和编译输出',
    description: '读取用户指定的 CSA qk_pv Python 源码及 AllocateMemoryAddr 编译输出，显示三视图。仅支持此样例格式；地址来自编译输出，源码映射和节点顺序为推断。',
    inputSchema: { source_path: z.string().min(1), dump_path: z.string().min(1) },
    annotations: { readOnlyHint:true, destructiveHint:false, openWorldHint:false },
    _meta: { ui: { resourceUri } },
  }, async ({source_path, dump_path}) => {
    try {
      const parsed = await importCsaQkPv(source_path, dump_path);
      const manifest = toCsaManifest(parsed, source_path, dump_path);
      if (!manifest.nodes.length) throw new Error('未识别 CSA qk_pv 操作节点');
      return { content:[{type:'text',text:'已导入 CSA qk_pv：'+parsed.source.length+' 行源码，'+parsed.allocations.length+' 个编译分配。地址为编译证据，源码匹配与顺序为推断，未验证文件是否来自同次编译。'}], structuredContent:{kernel:'csa_qk_pv',knobs:{},manifest} };
    } catch (error) { return {isError:true,content:[{type:'text',text:'导入失败：'+error.message}]}; }
  });
  registerAppResource(server, 'Hardware-Native 工作台', resourceUri, { description: '源码、算子逻辑与硅片落点交互界面' }, async () => ({
    contents: [{ uri: resourceUri, mimeType: RESOURCE_MIME_TYPE, text: await htmlResource(), _meta: { ui: { prefersBorder: false } } }],
  }));
  return server;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await createServer().connect(new StdioServerTransport());
}
