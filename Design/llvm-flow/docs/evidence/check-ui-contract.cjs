/* Read-only regression gate. Run from any cwd; no browser or compiler execution. */
const path = require('path');
const { spawnSync } = require('child_process');
const root = path.resolve(__dirname, '../..');
for (const args of [
  ['node', 'docs/evidence/test-stage2-revised.cjs'],
  ['python3', '-B', 'docs/evidence/test-stage2-cfg.py'],
  ['git', 'diff', '--check', '--', 'llvmcfg-standalone.html', 'AGENTS.md', 'docs/ui-contract.md', 'docs/evidence/check-ui-contract.cjs', 'docs/evidence/test-stage2-revised.cjs'],
]) {
  const result = spawnSync('rtk', args, { cwd: root, stdio: 'inherit' });
  if (result.error) { console.error(result.error.message); process.exit(1); }
  if (result.status !== 0) process.exit(result.status || 1);
}
console.log('PASS: static regression gate. Visual/interactive acceptance is NOT established.');
