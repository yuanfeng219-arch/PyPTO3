// Build a single self-contained demo.html.
//
//   node build-demo.mjs [--runs a,b] [--out demo.html]
//
// Everything the tool needs - design tokens, stylesheet, parser bundle, app
// code, the pass index, the pass docs and every IR snapshot - is gzipped,
// base64'd and embedded in one file. No build step, no server, no repo: open
// it and it works.
//
// Snapshots stay individually compressed and are inflated only when a view
// asks for one, so opening the file costs the index (~110 KB compressed), not
// the 47 MB of IR behind it.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../..');
const DS = path.join(REPO, 'vendor/pto-design-system');

const args = process.argv.slice(2);
const outName = (() => {
  const i = args.indexOf('--out');
  return i >= 0 ? args[i + 1] : 'demo.html';
})();
const onlyRuns = (() => {
  const i = args.indexOf('--runs');
  return i >= 0 ? new Set(args[i + 1].split(',')) : null;
})();

const read = (p) => fs.readFileSync(path.isAbsolute(p) ? p : path.join(HERE, p), 'utf8');
const pack = (text) => zlib.gzipSync(Buffer.from(text, 'utf8'), { level: 9 }).toString('base64');

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

for (const required of ['data/index.js', 'data/docs.js', 'lib/bundle.js']) {
  if (!fs.existsSync(path.join(HERE, required))) {
    console.error(`! ${required} is missing — run \`node build.mjs\` first.`);
    process.exit(1);
  }
}

// The design system has no url() or font references, so concatenating the
// token files and stripping their @imports produces a self-contained sheet.
const css = [
  'tokens/foundation.css',
  'tokens/semantic.css',
  'tokens/components.css',
  'css/style.css',
  'patterns/ide-frame/pattern.css',
]
  .map((f) => read(path.join(DS, f)))
  .concat([read('styles.css')])
  .join('\n')
  .replace(/^\s*@import[^;]*;\s*$/gm, '');

const bundleSrc = read('lib/bundle.js');
const appSrc = read('app.js');

// `data/index.js` and `data/docs.js` are `window.X = {...};` assignments; the
// demo wants the JSON on its own.
const jsonFrom = (file, global) => {
  const src = read(file);
  const start = src.indexOf('=', src.indexOf(global)) + 1;
  return src.slice(start).trim().replace(/;\s*$/, '');
};

const indexJson = jsonFrom('data/index.js', 'window.PTX_INDEX');
const docsJson = jsonFrom('data/docs.js', 'window.PTX_DOCS');
const index = JSON.parse(indexJson);

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

const snapshots = {};
let rawBytes = 0;
const runs = index.runs.filter((r) => !onlyRuns || onlyRuns.has(r.id));

for (const run of runs) {
  const dir = path.join(REPO, run.dir);
  for (const p of run.passes) {
    const text = fs.readFileSync(path.join(dir, p.file), 'utf8');
    rawBytes += Buffer.byteLength(text);
    snapshots[`${run.id}:${p.idx}`] = pack(text);
  }
  process.stdout.write(`  ${run.id}: ${run.passes.length} snapshots embedded\n`);
}

const keptIndex = onlyRuns ? JSON.stringify({ ...index, runs }) : indexJson;

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

// Reuse the served page's markup, minus everything that points outside the file.
const body = read('index.html')
  .replace(/^[\s\S]*?<body([^>]*)>/, '<body$1>')
  .replace(/<\/body>[\s\S]*$/, '</body>')
  .replace(/<script[^>]*><\/script>\s*/g, '')
  .replace(/<a class="btn btn-ghost btn-sm" href="\.\.\/\.\.\/launch\.html">PTO<\/a>/,
    '<span class="ptx-brand">PTO</span>')
  .replace(/<\/body>$/, '');

const packed = {
  css: pack(css),
  bundle: pack(bundleSrc),
  app: pack(appSrc),
  index: pack(keptIndex),
  docs: pack(docsJson),
  snapshots,
};

const bootstrap = `
(function () {
  'use strict';
  var P = window.PTX_PACKED;
  var status = document.getElementById('ptx-boot-status');

  function fail(msg) {
    document.getElementById('ptx-boot').innerHTML =
      '<div class="ptx-boot__card"><h1>无法启动</h1><p>' + msg + '</p>' +
      '<p style="font-size:12px;color:#9ca3af">如果问题持续，可以把这个文件放到任意静态服务器下打开' +
      '（例如 <code>npx http-server . -p 4178</code>），或改用仓库里的完整版 ' +
      '<code>Design/pass-transform-explorer/index.html</code>。</p></div>';
  }

  if (typeof DecompressionStream === 'undefined') {
    fail('这个浏览器不支持 DecompressionStream，无法解压内嵌数据。' +
         '请使用 Chrome 80+、Edge 80+、Firefox 113+ 或 Safari 16.4+ 打开。');
    return;
  }

  function inflate(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
  }

  // Indirect eval runs the source in global scope, exactly as a <script> tag
  // would, without needing a blob: URL - which a page opened from file:// is
  // not guaranteed to be allowed to load a script from.
  var globalEval = eval;
  function runScript(source) { globalEval(source); }

  function step(label) { if (status) status.textContent = label; }

  step('解压样式…');
  inflate(P.css).then(function (cssText) {
    var style = document.createElement('style');
    style.textContent = cssText;
    document.head.appendChild(style);
    step('解压 Pass 索引…');
    return inflate(P.index);
  }).then(function (indexJson) {
    window.PTX_INDEX = JSON.parse(indexJson);
    step('解压 Pass 文档…');
    return inflate(P.docs);
  }).then(function (docsJson) {
    window.PTX_DOCS = JSON.parse(docsJson);
    window.PTX_EMBEDDED = P.snapshots;
    window.PTX_STANDALONE = true;
    step('加载解析器…');
    return inflate(P.bundle);
  }).then(runScript).then(function () {
    step('启动界面…');
    return inflate(P.app);
  }).then(runScript).then(function () {
    var boot = document.getElementById('ptx-boot');
    if (boot) boot.remove();
    // The packed payload is the biggest thing on the heap; the snapshot store
    // is still referenced through PTX_EMBEDDED, the rest can go.
    delete P.css; delete P.bundle; delete P.app; delete P.index; delete P.docs;
  }).catch(function (err) {
    fail((err && err.message) || String(err));
  });
})();
`;

const html = `<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pass Transform Explorer · PyPTO（独立 demo）</title>
<meta name="description" content="单文件 demo：按 Pass 逐站检查 PyPTO 编译流水线的代码 Diff、结构图变化与实测优化证据。">
<style>
  /* Only what the boot screen needs; the real stylesheet is unpacked below. */
  html, body { height: 100%; margin: 0; }
  body { background: #fbfbfb; color: #111827;
         font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
  #ptx-boot { position: fixed; inset: 0; display: grid; place-items: center; z-index: 999;
              background: #fbfbfb; }
  .ptx-boot__card { max-width: 560px; padding: 0 28px; text-align: center; }
  .ptx-boot__card h1 { font-size: 20px; font-weight: 650; margin: 0 0 10px; }
  .ptx-boot__card p { font-size: 13.5px; line-height: 1.7; color: #4b5563; margin: 0 0 6px; }
  .ptx-boot__bar { width: 200px; height: 3px; margin: 18px auto 0; border-radius: 2px;
                   background: #e5e7eb; overflow: hidden; }
  .ptx-boot__bar i { display: block; width: 40%; height: 100%; background: #2563eb;
                     animation: ptx-slide 1.1s ease-in-out infinite; }
  @keyframes ptx-slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(350%); } }
  .ptx-brand { font-size: 12px; font-weight: 700; letter-spacing: .06em; opacity: .7; padding: 0 8px; }
  .is-inert { cursor: help; text-decoration: none; opacity: .75; }
</style>
</head>
${body}
<div id="ptx-boot">
  <div class="ptx-boot__card">
    <h1>Pass Transform Explorer</h1>
    <p>正在解压内嵌的 ${runs.reduce((n, r) => n + r.passes.length, 0)} 份 IR 快照索引…</p>
    <p id="ptx-boot-status" style="font-size:12px;color:#9ca3af">准备中…</p>
    <div class="ptx-boot__bar"><i></i></div>
  </div>
</div>
<script id="ptx-packed">window.PTX_PACKED=${JSON.stringify(packed)};</script>
<script>${bootstrap}</script>
</body>
</html>
`;

const outPath = path.join(HERE, outName);
fs.writeFileSync(outPath, html, 'utf8');

const mb = (n) => (n / 1048576).toFixed(2) + ' MB';
console.log(`\n${outName}`);
console.log(`  embedded IR      ${mb(rawBytes)} raw -> ${mb(Object.values(snapshots).reduce((s, v) => s + v.length, 0))} base64`);
console.log(`  file size        ${mb(fs.statSync(outPath).size)}`);
console.log(`  runs             ${runs.map((r) => r.id).join(', ')}`);
