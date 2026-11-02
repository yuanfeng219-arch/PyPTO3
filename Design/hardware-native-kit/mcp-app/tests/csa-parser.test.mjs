import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { importCsaQkPv, parseCsaQkPv } from '../csa-parser.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
test('extracts CSA qk_pv source and compiler addresses without double-counting aliases', async () => {
  const result=await importCsaQkPv(path.join(root,'Data/DeepseekV4/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py'),path.join(root,'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/36_after_AllocateMemoryAddr.py'));
  assert.equal(result.source[0].no,208);
  const work=result.allocations.find(x=>x.id==='mem_vec_27' && x.worker==='aiv');
  assert.equal(work.address,49920); assert.equal(work.size,32768); assert.ok(work.refs.length>1);
  assert.equal(result.allocations.filter(x=>x.id==='mem_vec_27' && x.worker==='aiv').length,1);
});
test('rejects unrelated files',()=>assert.throws(()=>parseCsaQkPv('x = 1',''),/qk_pv/));
