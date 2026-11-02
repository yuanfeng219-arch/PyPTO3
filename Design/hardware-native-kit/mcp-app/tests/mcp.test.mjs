import test from 'node:test';
import assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { createServer } from '../server.mjs';

test('MCP tool links to a self-contained UI and validates configurations', async () => {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const server = createServer();
  const client = new Client({ name: 'test-host', version: '1.0.0' });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const tools = await client.listTools();
    const tool = tools.tools.find((item) => item.name === 'show_hardware_native');
    assert.equal(tool._meta.ui.resourceUri, 'ui://hardware-native-kit/workbench.html');
    const resource = await client.readResource({ uri: tool._meta.ui.resourceUri });
    assert.equal(resource.contents[0].mimeType, 'text/html;profile=mcp-app');
    assert.match(resource.contents[0].text, /HNKit\.mount/);
    assert.doesNotMatch(resource.contents[0].text, /<script[^>]+src=/);
    const result = await client.callTool({ name: tool.name, arguments: { kernel: 'fa_fused', knobs: { SEQ_TILE: 256, FA_STAGE: 2 } } });
    assert.deepEqual(result.structuredContent, { kernel: 'fa_fused', knobs: { SEQ_TILE: 256, FA_STAGE: 2 } });
    const other = await client.callTool({ name: tool.name, arguments: { kernel: 'matmul_64', knobs: { TILE: 32 } } });
    assert.equal(other.structuredContent.kernel, 'matmul_64');
    const invalid = await client.callTool({ name: tool.name, arguments: { kernel: 'matmul_64', knobs: { FA_STAGE: 2 } } });
    assert.equal(invalid.isError, true);
  } finally { await client.close(); await server.close(); }
});
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

test('starts as a local STDIO MCP server', async () => {
  const serverPath = fileURLToPath(new URL('../server.mjs', import.meta.url));
  const transport = new StdioClientTransport({ command: process.execPath, args: [serverPath], cwd: path.dirname(serverPath), stderr: 'pipe' });
  const client = new Client({ name: 'stdio-test-host', version: '1.0.0' });
  try {
    await client.connect(transport);
    const result = await client.callTool({ name: 'show_hardware_native', arguments: {} });
    assert.equal(result.structuredContent.kernel, 'fa_fused');
  } finally { await client.close(); }
});
