import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { createServer } from '../server.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../../..');
test('MCP imports CSA files and returns a renderable evidence manifest', async()=>{
 const [a,b]=InMemoryTransport.createLinkedPair(), server=createServer(), client=new Client({name:'test',version:'1'});
 await Promise.all([server.connect(b),client.connect(a)]);
 try {
  const list=await client.listTools(); assert.ok(list.tools.find(t=>t.name==='import_csa_qk_pv')._meta.ui.resourceUri);
  const r=await client.callTool({name:'import_csa_qk_pv',arguments:{source_path:path.join(root,'Data/DeepseekV4/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py'),dump_path:path.join(root,'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/36_after_AllocateMemoryAddr.py')}});
  assert.equal(r.isError,undefined);assert.equal(r.structuredContent.manifest.entry,'qk_pv');assert.ok(r.structuredContent.manifest.nodes.length>4);
  assert.equal(r.structuredContent.manifest.compiledAllocations.find(x=>x.id==='mem_vec_27' && x.worker==='aiv').address,49920);
  const bad=await client.callTool({name:'import_csa_qk_pv',arguments:{source_path:'nonexistent',dump_path:'nonexistent'}});assert.equal(bad.isError,true);
 } finally {await client.close();await server.close();}
});
