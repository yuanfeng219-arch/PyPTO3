/* Resolve the actual local entry resources for static checks; never fetch or execute them. */
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
module.exports = function loadPage(entry) {
  const source = fs.readFileSync(entry, 'utf8');
  const base = path.dirname(entry);
  const resources = [];
  const html = source.replace(/<link\b([^>]*rel="stylesheet"[^>]*)\/>|<script\b([^>]*)>([\s\S]*?)<\/script>/g, (tag, link, script, body) => {
    const attrs = link || script;
    const ref = attrs.match(link ? /href="([^"]+)"/ : /\bsrc="([^"]+)"/);
    assert.ok(ref, 'Entry scripts and styles must be external');
    assert.ok(ref[1].startsWith('./assets/llvmcfg/'), 'Resources must stay within the canonical asset directory');
    assert.ok(!/\b(async|defer|type="module")\b/.test(attrs), 'Preserve synchronous script initialization order');
    const file = path.resolve(base, ref[1]);
    assert.ok(file.startsWith(path.join(base, 'assets/llvmcfg') + path.sep));
    const content = fs.readFileSync(file, 'utf8');
    resources.push({file, content, type:link ? 'style' : 'script'});
    return link ? `<style${attrs}>${content}</style>` : `<script${attrs.replace(/\s+src="[^"]+"/, '')}>${content}</script>`;
  });
  assert.ok(!/<style\b/.test(source), 'No inline stylesheets in entry');
  assert.equal(resources.filter(r=>r.type==='style').length, 8);
  assert.deepEqual(resources.filter(r=>r.type==='script').map(r=>path.relative(base,r.file)), [
    'assets/llvmcfg/vendor/pto-design-system/patterns/workbench-shell/pattern.js',
    'assets/llvmcfg/vendor/pto-design-system/patterns/ide-frame/pattern.js',
    'assets/llvmcfg/runtime/llvmcfg.js',
    'assets/llvmcfg/scripts/code-enhancements.js',
    'assets/llvmcfg/data/deepseek-evidence.js',
    'assets/llvmcfg/scripts/review.js',
  ]);
  return {html, resources};
};
