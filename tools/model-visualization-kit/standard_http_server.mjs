import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { createServer } from './standard_server.mjs';

const PORT = Number(process.env.PORT || 8787);
const TOKEN = process.env.MCP_AUTH_TOKEN || '';
const MAX_BODY = Number(process.env.MCP_MAX_BODY_BYTES || 20 * 1024 * 1024);
const ROOT = path.dirname(fileURLToPath(import.meta.url));

function allow(request) { return !TOKEN || request.headers.authorization === `Bearer ${TOKEN}`; }
function headers() { return { 'content-type': 'application/json', 'cache-control': 'no-store', 'access-control-allow-origin': process.env.MCP_CORS_ORIGIN || 'null', 'access-control-allow-headers': 'Authorization, Content-Type, MCP-Protocol-Version, MCP-Session-Id', 'access-control-allow-methods': 'POST, GET, DELETE, OPTIONS' }; }
function readBody(request) { return new Promise((resolve, reject) => { let body = ''; request.setEncoding('utf8'); request.on('data', (chunk) => { body += chunk; if (Buffer.byteLength(body) > MAX_BODY) reject(new Error('request body exceeds limit')); }); request.on('end', () => { try { resolve(body ? JSON.parse(body) : undefined); } catch (error) { reject(new Error(`invalid JSON: ${error.message}`)); } }); request.on('error', reject); }); }

const server = http.createServer(async (request, response) => {
  if (request.method === 'OPTIONS') { response.writeHead(204, headers()); response.end(); return; }
  if (request.method === 'GET' && request.url === '/healthz') { response.writeHead(200, { ...headers(), 'content-type': 'application/json' }); response.end(JSON.stringify({ ok: true, service: 'model-visualization-kit', transport: 'streamable-http', version: '0.2.0' })); return; }
  if (request.method === 'GET' && request.url.startsWith('/artifacts/')) { if (!allow(request)) { response.writeHead(401, { ...headers(), 'www-authenticate': 'Bearer' }); response.end(JSON.stringify({ error: 'unauthorized' })); return; } const filename = path.basename(decodeURIComponent(request.url.slice('/artifacts/'.length))); const artifact = path.join(ROOT, 'artifacts', filename); if (!fs.existsSync(artifact) || !fs.statSync(artifact).isFile()) { response.writeHead(404, headers()); response.end(JSON.stringify({ error: 'artifact_not_found' })); return; } const type = filename.endsWith('.html') ? 'text/html; charset=utf-8' : filename.endsWith('.json') ? 'application/json' : 'application/octet-stream'; response.writeHead(200, { ...headers(), 'content-type': type }); fs.createReadStream(artifact).pipe(response); return; }
  if (request.url !== '/mcp') { response.writeHead(404, headers()); response.end(JSON.stringify({ error: 'not_found' })); return; }
  if (!allow(request)) { response.writeHead(401, { ...headers(), 'www-authenticate': 'Bearer' }); response.end(JSON.stringify({ error: 'unauthorized' })); return; }
  if (request.method !== 'POST') { response.writeHead(405, { ...headers(), allow: 'POST, OPTIONS' }); response.end(JSON.stringify({ jsonrpc: '2.0', id: null, error: { code: -32000, message: 'Use POST for stateless Streamable HTTP MCP.' } })); return; }
  try {
    const body = await readBody(request); const mcp = createServer(); const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    await mcp.connect(transport); await transport.handleRequest(request, response, body);
    response.on('close', () => { transport.close(); mcp.close(); });
  } catch (error) {
    if (!response.headersSent) { response.writeHead(error.message.startsWith('request body') ? 413 : 400, headers()); response.end(JSON.stringify({ jsonrpc: '2.0', id: null, error: { code: -32000, message: error.message } })); }
  }
});

server.listen(PORT, '0.0.0.0', () => console.error(`standard Streamable HTTP MCP listening on 0.0.0.0:${PORT}`));
process.on('SIGTERM', () => server.close(() => process.exit(0)));
