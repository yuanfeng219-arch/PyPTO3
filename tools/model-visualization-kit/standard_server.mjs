import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const localServer = path.join(ROOT, 'mcp_server.mjs');
const textResult = (data) => ({ content: [{ type: 'text', text: JSON.stringify(data, null, 2) }], structuredContent: data });

function invokeLegacy(name, args) {
  let normalized = { ...args }; let temporary = null;
  if (name === 'render_python_model_graph' && typeof args.source_content === 'string') {
    const inputDir = path.join(ROOT, 'artifacts', '.runtime-inputs'); fs.mkdirSync(inputDir, { recursive: true });
    temporary = path.join(inputDir, `${crypto.randomUUID()}-${String(args.filename || 'model.py').replace(/[^A-Za-z0-9_.-]/g, '_')}`);
    fs.writeFileSync(temporary, args.source_content, 'utf8'); normalized = { ...args, source_path: temporary };
  }
  if (process.env.MCP_REMOTE_MODE === '1' && normalized.source_path) {
    const sourcePath = path.resolve(String(normalized.source_path)); if (!sourcePath.startsWith(`${ROOT}${path.sep}`)) throw new Error('remote source_path must be inside the MCP workspace; use source_content for client files');
    normalized.source_path = sourcePath;
  }
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [localServer], { cwd: ROOT, stdio: ['pipe', 'pipe', 'pipe'] });
    let output = ''; let errors = ''; let settled = false;
    const finish = (fn, value) => { if (!settled) { settled = true; fn(value); } };
    child.stdout.on('data', (chunk) => { output += chunk; }); child.stderr.on('data', (chunk) => { errors += chunk; });
    child.on('error', (error) => finish(reject, error));
    child.on('close', (code) => { try { if (code !== 0) throw new Error(errors || `legacy server exited with ${code}`); const result = output.split(/\r?\n/).filter(Boolean).map(JSON.parse).find((item) => item.id === 1); if (!result) throw new Error('legacy server returned no response'); if (result.error) throw new Error(result.error.message); const data = result.result.structuredContent; const base = process.env.MCP_ARTIFACT_BASE_URL; if (base && data?.artifacts) data.artifacts = data.artifacts.map((artifact) => ({ ...artifact, url: `${base.replace(/\/$/, '')}/artifacts/${encodeURIComponent(path.basename(artifact.path))}` })); finish(resolve, data); } catch (error) { finish(reject, error); } finally { if (temporary) fs.rmSync(temporary, { force: true }); } });
    child.stdin.end(`${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name, arguments: normalized } })}\n`);
  });
}

export function createServer() {
  const server = new McpServer({ name: 'model-visualization-kit', version: '0.2.0' });
  const register = (name, description, shape) => server.registerTool(name, { description, inputSchema: shape }, async (args) => textResult(await invokeLegacy(name, args)));
  register('search_visual_assets', 'Search visualization assets by keyword, category, or framework.', { query: z.string().optional(), category: z.string().optional(), framework: z.string().optional() });
  register('render_model_graph', 'Render a model JSON into an HTML graph and return a structural summary.', { model_path: z.string(), output_name: z.string().optional() });
  register('render_swimlane', 'Render a Chrome Trace or DFX traceEvents file into an interactive HTML swimlane timeline.', { trace_path: z.string(), output_name: z.string().optional(), max_events: z.number().optional() });
  register('render_python_model_graph', 'Statically inspect a Python/PyTorch model source file and render a hierarchical whole-network architecture graph. Does not execute the source.', { source_path: z.string().optional(), source_content: z.string().optional(), filename: z.string().optional(), output_name: z.string().optional() });
  register('inspect_operator', 'Inspect an operator contract and provide static visualization and validation guidance.', { operator: z.string(), framework: z.string().optional(), input_shapes: z.array(z.unknown()).optional(), dtype: z.string().optional() });
  register('generate_operator_scaffold', 'Generate a reviewable operator development scaffold with tests and visualization metadata.', { operator_name: z.string(), framework: z.string().optional(), output_dir: z.string().optional() });
  return server;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const server = createServer();
  await server.connect(new StdioServerTransport());
}
