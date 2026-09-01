import http from 'node:http';
import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8787);
const TOKEN = process.env.MCP_AUTH_TOKEN || '';
const MAX_BODY = 15 * 1024 * 1024;

function authorized(request) {
  if (!TOKEN) return true;
  return request.headers.authorization === `Bearer ${TOKEN}`;
}

function runLocalMcp(payload) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [path.join(ROOT, 'mcp_server.mjs')], { cwd: ROOT, stdio: ['pipe', 'pipe', 'pipe'] });
    let output = ''; let error = ''; let settled = false;
    const finish = (fn, value) => { if (!settled) { settled = true; fn(value); } };
    child.stdout.on('data', (chunk) => { output += chunk; });
    child.stderr.on('data', (chunk) => { error += chunk; });
    child.on('error', (err) => finish(reject, err));
    child.on('close', (code) => {
      if (code !== 0) return finish(reject, new Error(error || `local MCP exited with ${code}`));
      const response = output.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line)).find((item) => item.id === payload.id);
      if (!response) return finish(reject, new Error('local MCP returned no response'));
      finish(resolve, response);
    });
    child.stdin.end(`${JSON.stringify(payload)}\n`);
  });
}

const server = http.createServer(async (request, response) => {
  if (request.method === 'GET' && request.url === '/healthz') {
    response.writeHead(200, { 'content-type': 'application/json' }); response.end(JSON.stringify({ ok: true, service: 'model-visualization-kit' })); return;
  }
  if (request.method !== 'POST' || request.url !== '/mcp') { response.writeHead(404); response.end('Not found'); return; }
  if (!authorized(request)) { response.writeHead(401, { 'content-type': 'application/json' }); response.end(JSON.stringify({ error: 'unauthorized' })); return; }
  let body = ''; request.setEncoding('utf8');
  request.on('data', (chunk) => { body += chunk; if (body.length > MAX_BODY) request.destroy(); });
  request.on('end', async () => {
    try {
      const payload = JSON.parse(body); const result = await runLocalMcp(payload);
      response.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' }); response.end(JSON.stringify(result));
    } catch (error) {
      response.writeHead(400, { 'content-type': 'application/json' }); response.end(JSON.stringify({ jsonrpc: '2.0', id: null, error: { code: -32000, message: error.message } }));
    }
  });
});

server.listen(PORT, '0.0.0.0', () => console.error(`remote MCP listening on 0.0.0.0:${PORT}`));
process.on('SIGTERM', () => server.close(() => process.exit(0)));
