import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
function run(requests) { return new Promise((resolve, reject) => { const child = spawn(process.execPath, [path.join(ROOT, '..', 'standard_server.mjs')], { cwd: path.join(ROOT, '..'), stdio: ['pipe', 'pipe', 'pipe'] }); let out = ''; let err = ''; child.stdout.on('data', (chunk) => { out += chunk; }); child.stderr.on('data', (chunk) => { err += chunk; }); child.on('error', reject); child.on('close', (code) => code === 0 ? resolve(out.trim().split(/\r?\n/).map(JSON.parse)) : reject(new Error(err))); child.stdin.end(requests.map((item) => JSON.stringify(item)).join('\n') + '\n'); }); }

test('standard MCP advertises tools with JSON Schema', async () => {
  const [initialize, list] = await run([{ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', capabilities: {}, clientInfo: { name: 'test', version: '0.1' } } }, { jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} }]);
  assert.equal(initialize.result.serverInfo.name, 'model-visualization-kit'); assert.equal(list.result.tools.length, 6); assert.ok(list.result.tools.some((tool) => tool.name === 'render_python_model_graph')); assert.equal(list.result.tools.find((tool) => tool.name === 'render_python_model_graph').inputSchema.type, 'object');
});

test('model graph accepts source_content without executing Python', async () => {
  const source = 'class Tiny:\n    def __init__(self):\n        self.layer = Linear(4, 4)\n    def forward(self, x):\n        return self.layer(x)\n';
  const [result] = await run([{ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'render_python_model_graph', arguments: { source_content: source, filename: 'tiny.py', output_name: 'test_tiny' } } }]);
  assert.equal(result.result.structuredContent.summary.execution, false); assert.equal(result.result.structuredContent.summary.class_count, 1); assert.ok(result.result.structuredContent.artifacts[0].path.endsWith('test_tiny.html'));
});
