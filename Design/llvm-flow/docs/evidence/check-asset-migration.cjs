/* One-time migration proof. Later intentional asset edits can change these hashes. */
const path = require('path');
const vm = require('vm');
const assert = require('assert/strict');
const crypto = require('crypto');
const root = path.resolve(__dirname, '../..');
const {resources} = require('./load-page.cjs')(path.join(root, 'llvmcfg-standalone.html'));
const assets = new Map(resources.map(r=>[path.relative(path.join(root,'assets/llvmcfg'),r.file),r.content]));
const window = {};
vm.runInNewContext(assets.get('data/deepseek-evidence.js'), {window});
for (const record of require('../../assets/llvmcfg/extraction-manifest.cjs')) {
  let content = record.parts.map(p=>assets.get(p)).join('');
  for (const key of ['DATA','CFG_PACKED']) {
    content = content.replace('window.LLVMCfgEvidence.'+key, ()=>JSON.stringify(window.LLVMCfgEvidence[key]));
  }
  assert.equal(crypto.createHash('sha256').update(content).digest('hex'), record.sha256, record.parts.join(', '));
}
console.log('PASS: all 12 original inline blocks are byte-identical after reassembly, including both evidence payloads.');
