import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const request = (id, method, params = {}) => JSON.stringify({ jsonrpc: '2.0', id, method, params });
const input = [
  request(1, 'initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'demo', version: '0.1' } }),
  JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }),
  request(2, 'tools/list'),
  request(3, 'tools/call', { name: 'render_model_graph', arguments: { model_path: path.join(root, 'examples', 'resnet50.json') } }),
].join('\n') + '\n';
const child = spawnSync(process.execPath, [path.join(root, 'mcp_server.mjs')], { input, encoding: 'utf8' });
if (child.status !== 0) throw new Error(child.stderr || `server exited with ${child.status}`);
const responses = child.stdout.trim().split(/\r?\n/).map(JSON.parse);
if (responses.length !== 3 || responses[1].result.tools.length !== 6) throw new Error('MCP discovery assertion failed');
const artifact = responses[2].result.structuredContent.artifacts[0].path;
console.log(JSON.stringify({ ok: true, tool_count: responses[1].result.tools.length, artifact }, null, 2));
