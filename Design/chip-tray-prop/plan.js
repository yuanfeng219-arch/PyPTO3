#!/usr/bin/env node
// 芯片模型托盘道具 · SVG 平面图
// 几何全部取自 geometry.js，与 STL / 3MF 同源，改尺寸不会两边走样。
// 输出 A3 横版（420 x 297 mm），用浏览器打开或直接打印。

'use strict';
const fs = require('fs');
const path = require('path');
const G = require('./geometry');

/* ---------- 绘图基元 ---------- */

const n = (v) => Math.round(v * 1000) / 1000;
const esc = (s) => String(s).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));

const rect = (x, y, w, h, cls, fill) =>
  `<rect x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" class="${cls}"${fill ? ` fill="${fill}"` : ''}/>`;
const line = (x1, y1, x2, y2, cls) =>
  `<line x1="${n(x1)}" y1="${n(y1)}" x2="${n(x2)}" y2="${n(y2)}" class="${cls}"/>`;
const txt = (x, y, s, cls = 't', anchor = 'start') =>
  `<text x="${n(x)}" y="${n(y)}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;

// 水平尺寸线。ext 为引出线起点的 y（省略则不画引出线）
function dimH(x1, x2, y, label, ext) {
  let s = '';
  if (ext !== undefined) {
    s += line(x1, ext, x1, y + (ext < y ? 1.2 : -1.2), 'ext');
    s += line(x2, ext, x2, y + (ext < y ? 1.2 : -1.2), 'ext');
  }
  s += `<line x1="${n(x1)}" y1="${n(y)}" x2="${n(x2)}" y2="${n(y)}" class="dim" marker-start="url(#ar)" marker-end="url(#ar)"/>`;
  s += txt((x1 + x2) / 2, y - 1.1, label, 'td', 'middle');
  return s;
}

// 垂直尺寸线。标注文字横排放在尺寸线左侧
function dimV(y1, y2, x, label, ext) {
  let s = '';
  if (ext !== undefined) {
    s += line(ext, y1, x + (ext < x ? 1.2 : -1.2), y1, 'ext');
    s += line(ext, y2, x + (ext < x ? 1.2 : -1.2), y2, 'ext');
  }
  s += `<line x1="${n(x)}" y1="${n(y1)}" x2="${n(x)}" y2="${n(y2)}" class="dim" marker-start="url(#ar)" marker-end="url(#ar)"/>`;
  s += txt(x - 1.3, (y1 + y2) / 2 + 0.9, label, 'td', 'end');
  return s;
}

/* ---------- 视图 ---------- */

// 俯视图：直接遍历高度场的 ops 画矩形，保证和实体模型一致
function planView(ox, oy, k, ops, outline, dimLabel) {
  const X = (v) => ox + v * k, Y = (v) => oy + v * k;
  let s = rect(ox, oy, outline.w * k, outline.d * k, 'out', '#ffffff');
  for (const o of ops) {
    if (!o.v) continue;                       // 深度 0 的只是网格辅助线，不是实体特征
    const cls = o.v < 2 ? 'fea-thin' : 'fea';
    const fill = o.t === 'r' ? '#ffffff' : (o.v < 2 ? '#dcdcdc' : '#eeeeee');
    s += rect(X(o.x), Y(o.y), o.w * k, o.d * k, cls, fill);
  }
  if (dimLabel) {
    s += dimH(ox, ox + outline.w * k, oy - 6, dimLabel(outline.w), oy);
    s += dimV(oy, oy + outline.d * k, ox - 6, dimLabel(outline.d), ox);
  }
  return s;
}

// 剖面：在给定 y 处切高度场，合并同高度的相邻段
function sectionView(ox, oyBase, k, boxes, atY) {
  const seg = boxes.filter((b) => b[1] < atY && b[4] > atY)
    .map((b) => ({ x0: b[0], x1: b[3], z: b[5] }))
    .sort((a, b) => a.x0 - b.x0);
  const merged = [];
  for (const s of seg) {
    const last = merged[merged.length - 1];
    if (last && Math.abs(last.z - s.z) < 1e-9 && Math.abs(last.x1 - s.x0) < 1e-9) last.x1 = s.x1;
    else merged.push({ ...s });
  }
  let out = '';
  for (const m of merged) {
    out += rect(ox + m.x0 * k, oyBase - m.z * k, (m.x1 - m.x0) * k, m.z * k, 'fea', 'url(#hatch)');
  }
  out += line(ox - 3, oyBase, ox + G.TRAY.W * k + 3, oyBase, 'ctr');
  return out;
}

/* ---------- 图纸 ---------- */

function sheet(scale) {
  const S = scale;
  const L = (v) => (S === 1 ? String(v) : String(Math.round(v * S * 10) / 10));
  const k = 0.4;                                   // 俯视图比例 1:2.5
  const trayBoxes = G.field({ x0: 0, y0: 0, x1: G.TRAY.W, y1: G.TRAY.D }, 0, G.TRAY.H, G.trayOps());
  let s = '';

  /* 图框 */
  s += rect(0, 0, 420, 297, 'none', '#ffffff');
  s += rect(10, 10, 400, 277, 'frame');

  /* 视图一 · 托盘俯视 */
  const t = { x: 26, y: 32 };
  s += txt(t.x, t.y - 12, '视图一  托盘俯视  1:2.5', 'th');
  s += planView(t.x, t.y, k, G.trayOps(), { w: G.TRAY.W, d: G.TRAY.D }, L);
  // 剖切符号 A-A
  const atY = 210;
  s += line(t.x - 4, t.y + atY * k, t.x + G.TRAY.W * k + 4, t.y + atY * k, 'ctr');
  s += txt(t.x - 5, t.y + atY * k + 1, 'A', 'td', 'end');
  s += txt(t.x + G.TRAY.W * k + 5, t.y + atY * k + 1, 'A', 'td');
  // 关键定位尺寸
  s += dimH(t.x + 15 * k, t.x + 95 * k, t.y + G.TRAY.D * k + 7, L(80), t.y + G.TRAY.D * k);
  s += dimH(t.x + 115 * k, t.x + 305 * k, t.y + G.TRAY.D * k + 7, L(190), t.y + G.TRAY.D * k);
  s += dimH(t.x + 325 * k, t.x + 405 * k, t.y + G.TRAY.D * k + 7, L(80), t.y + G.TRAY.D * k);
  s += dimV(t.y + 33 * k, t.y + 73 * k, t.x + G.TRAY.W * k + 11, L(40), t.x + G.TRAY.W * k);
  s += dimV(t.y + 193 * k, t.y + 267 * k, t.x + G.TRAY.W * k + 11, L(74), t.x + G.TRAY.W * k);

  /* 剖面 A-A */
  const secBase = 193;
  s += txt(t.x, 167, '剖面 A-A  1:2.5', 'th');
  s += sectionView(t.x, secBase, k, trayBoxes, atY);
  // 标高引线。真实高度挤在一起，所以折一下，把文字均匀排开
  const ex = t.x + G.TRAY.W * k;
  [[7, 'HBM 槽底'], [25, '托盘顶面'], [31, '底板'], [43, '核顶面'], [53, '凸台顶']]
    .forEach(([z, lab], i) => {
      const ly = secBase - z * k, ty = secBase - 4 - i * 4.2;
      s += line(ex + 2, ly, ex + 5, ly, 'ext');
      s += line(ex + 5, ly, ex + 8, ty, 'ext');
      s += txt(ex + 9, ty + 0.9, `${L(z)}  ${lab}`, 'ts');
    });

  /* 视图二 · 盖子俯视（外观面） */
  const d = { x: 232, y: 32 };
  s += txt(d.x, d.y - 12, '视图二  盖子外观面俯视  1:2.5', 'th');
  s += rect(d.x, d.y, G.TRAY.W * k, G.TRAY.D * k, 'out', '#ffffff');
  const w = G.LID.wall;
  s += rect(d.x + w * k, d.y + w * k, (G.TRAY.W - 2 * w) * k, (G.TRAY.D - 2 * w) * k, 'hid');
  // 外观面是整片平的，灰色是下挖 0.6 的分隔槽，白色是贴片
  const F = G.LID_FIELD, gw = G.LID.gw;
  s += rect(d.x + (F.x0 - gw) * k, d.y + (F.y0 - gw) * k,
    (F.x1 - F.x0 + 2 * gw) * k, (F.y1 - F.y0 + 2 * gw) * k, 'fea', '#cfcfcf');
  for (const [u0, v0, u1, v1] of G.PADS) {
    s += rect(d.x + u0 * k, d.y + v0 * k, (u1 - u0) * k, (v1 - v0) * k, 'fea', '#ffffff');
  }
  s += rect(d.x + 160 * k, d.y + 130 * k, 100 * k, 36 * k, 'hid');
  s += txt(d.x + 210 * k, d.y + 150 * k + 0.9, '标识位', 'ts', 'middle');
  s += dimH(d.x, d.x + G.TRAY.W * k, d.y - 6, L(G.TRAY.W), d.y);
  s += dimV(d.y, d.y + G.TRAY.D * k, d.x - 6, L(G.TRAY.D), d.x);
  s += dimH(d.x + 46 * k, d.x + 372 * k, d.y + G.TRAY.D * k + 7, L(326), d.y + G.TRAY.D * k);
  s += dimV(d.y + 96 * k, d.y + 200 * k, d.x + G.TRAY.W * k + 7, L(104), d.x + G.TRAY.W * k);
  s += txt(d.x, 166, `外观面整片平齐，灰色为下挖 ${L(G.LID.gd)} 的分隔槽；只有白色贴片是银色，边框同本体色`, 'ts');

  /* 视图三 · 放大核详图 1:1 */
  const c = { x: 256, y: 188 };
  s += txt(232, 172, '视图三  放大核详图  1:1（两个核同款）', 'th');
  s += planView(c.x, c.y, 1, G.coreOps(0, 0), { w: G.CORE.w, d: G.CORE.d }, null);
  // 横向：先标各槽宽度，再标总宽
  const hx = (v) => c.x + v;
  s += dimH(hx(4), hx(24), c.y - 5, L(20), c.y);
  s += dimH(hx(28), hx(44), c.y - 5, L(16), c.y);
  s += dimH(hx(48), hx(68), c.y - 5, L(20), c.y);
  s += dimH(hx(72), hx(83), c.y - 5, L(11), c.y);
  for (const [a, z] of [[0, 4], [24, 28], [44, 48], [68, 72], [83, 86]]) {
    s += dimH(hx(a), hx(z), c.y - 5, L(z - a), c.y);
  }
  s += dimH(hx(0), hx(86), c.y - 11, L(86), c.y - 6);
  // 纵向
  const vy = (v) => c.y + v;
  s += dimV(vy(10), vy(57), c.x - 5, L(47), c.x);
  s += dimV(vy(10), vy(30), c.x - 13, L(20), c.x - 6);
  s += dimV(vy(0), vy(74), c.x - 21, L(74), c.x - 14);
  // 槽里直接写深度，避开两条搬运槽所在的高度
  for (const [x, y, lab] of [
    [14, 36, `深 ${L(10)}`], [36, 14, `深 ${L(4)}`], [36, 40, `深 ${L(4)}`],
    [58, 34, `凸 ${L(10)}`], [77.5, 32, `深 ${L(4)}`],
  ]) s += txt(c.x + x, c.y + y, lab, 'ts', 'middle');

  /* 视图四 · 数据块 */
  const b = { x: 26, y: 205 };
  const bs = G.BIGN * G.PITCH - 1;
  s += txt(b.x, 200, '视图四  数据块  1:2.5', 'th');
  s += rect(b.x, b.y, bs * k, bs * k, 'out', '#ffffff');
  for (let i = 1; i < G.BIGN; i++) {
    s += rect(b.x + (i * G.PITCH - 1) * k, b.y, 1 * k, bs * k, 'fea-thin', '#dcdcdc');
    s += rect(b.x, b.y + (i * G.PITCH - 1) * k, bs * k, 1 * k, 'fea-thin', '#dcdcdc');
  }
  s += dimH(b.x, b.x + bs * k, b.y + bs * k + 5, L(bs), b.y + bs * k);
  s += txt(b.x, b.y + bs * k + 11, `大颗粒 ${L(bs)}×${L(bs)}×${L(2 * G.PITCH - 1)}`, 'ts');
  s += txt(b.x, b.y + bs * k + 15, `由 ${G.BIGN}×${G.BIGN}×2 个 Tile 拼成`, 'ts');
  const tx = b.x + bs * k + 16;
  s += rect(tx, b.y, G.TILE * k, G.TILE * k, 'out', '#ffffff');
  s += dimH(tx, tx + G.TILE * k, b.y + G.TILE * k + 5, L(G.TILE), b.y + G.TILE * k);
  s += txt(tx, b.y + bs * k + 11, `小颗粒 Tile`, 'ts');
  s += txt(tx, b.y + bs * k + 15, `${L(G.TILE)} 立方`, 'ts');

  /* 深度表 */
  const tb = { x: 100, y: 204, w: 128, rh: 4.2 };
  const rows = [
    ['特征', '平面尺寸', '深度 / 高度'],
    ['HBM 槽', `${L(80)} × ${L(246)}`, `深 ${L(18)}`],
    ['芯片底板', `${L(190)} × ${L(246)}`, `凸 ${L(6)}`],
    ['普通核格', `${L(40)} × ${L(34)}`, `凸 ${L(12)}`],
    ['核格凹槽', `${L(32)} × ${L(16)}`, `深 ${L(10)}`],
    ['放大核', `${L(86)} × ${L(74)}`, `凸 ${L(12)}`],
    ['放大核 左大槽', `${L(20)} × ${L(47)}`, `深 ${L(10)}`],
    ['放大核 中上 / 中下', `${L(16)} × ${L(20)}`, `深 ${L(4)}`],
    ['放大核 右长槽', `${L(11)} × ${L(32)}`, `深 ${L(4)}`],
    ['白色凸台', `${L(20)} × ${L(46)}`, `凸 ${L(10)}`],
    ['搬运槽', `宽 ${L(2.5)}`, `深 ${L(1.5)}`],
    ['PCB 色皮', '顶面', `${L(0.6)}`],
    ['盖子', `${L(420)} × ${L(300)}`, `高 ${L(G.LID.H)}，壁 ${L(G.LID.wall)}`],
    ['盖子外观面分隔槽', `宽 ${L(G.LID.gw)} 起`, `深 0.6`],
  ];
  s += txt(tb.x, 200, '特征尺寸表', 'th');
  rows.forEach((r, i) => {
    const y = tb.y + i * tb.rh;
    s += rect(tb.x, y, tb.w, tb.rh, 'tbl');
    s += txt(tb.x + 1.5, y + tb.rh - 1.5, r[0], i === 0 ? 'tb' : 'ts');
    s += txt(tb.x + 62, y + tb.rh - 1.5, r[1], i === 0 ? 'tb' : 'ts');
    s += txt(tb.x + 97, y + tb.rh - 1.5, r[2], i === 0 ? 'tb' : 'ts');
  });

  /* 注释 */
  const nb = { x: 26, y: 252 };
  s += txt(nb.x, nb.y, '注', 'th');
  [
    '1  所有槽自顶面加工，无倒扣，不需支撑。',
    '2  槽深三档：深 10 放料，深 4 中转，刻槽 1.5 为通道。',
    '3  凸起 10 的白色凸台为计算单元。',
    '4  两个放大核几何完全相同，区别只在刻字。',
    '5  刻字、徽标与金色引脚不在本图范围。',
    '6  数据块为独立配件，不与托盘连接。',
    '7  示意尺寸，不按真实芯片比例。',
  ].forEach((line2, i) => { s += txt(nb.x, nb.y + 4.5 + i * 3.6, line2, 'ts'); });

  /* 标题栏 */
  const ti = { x: 250, y: 266, w: 155, h: 20 };
  s += rect(ti.x, ti.y, ti.w, ti.h, 'tbl');
  s += line(ti.x, ti.y + 7, ti.x + ti.w, ti.y + 7, 'tbl-l');
  s += line(ti.x, ti.y + 13.5, ti.x + ti.w, ti.y + 13.5, 'tbl-l');
  s += line(ti.x + 100, ti.y + 7, ti.x + 100, ti.y + ti.h, 'tbl-l');
  s += txt(ti.x + 2, ti.y + 5, 'PyPTO Lab Tour · 芯片模型托盘道具', 'tt');
  s += txt(ti.x + 2, ti.y + 11.8, `图号 CTP-01`, 'ts');
  s += txt(ti.x + 102, ti.y + 11.8, `比例 见各视图`, 'ts');
  s += txt(ti.x + 2, ti.y + 18.3, `单位 mm${S === 1 ? '' : `（已按 ${S} 折算）`}`, 'ts');
  s += txt(ti.x + 102, ti.y + 18.3, `A3 420 × 297`, 'ts');

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" viewBox="0 0 420 297">
<title>芯片模型托盘道具 平面图</title>
<defs>
<marker id="ar" markerWidth="4" markerHeight="4" refX="3.4" refY="2" orient="auto-start-reverse"><path d="M0,0.7 L3.6,2 L0,3.3 z" fill="#111"/></marker>
<pattern id="hatch" width="2" height="2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="2" stroke="#999" stroke-width="0.25"/></pattern>
</defs>
<style>
rect,line{fill:none}
.frame{stroke:#111;stroke-width:0.7}
.out{stroke:#111;stroke-width:0.5}
.fea{stroke:#111;stroke-width:0.3}
.fea-thin{stroke:#555;stroke-width:0.18}
.hid{stroke:#777;stroke-width:0.22;stroke-dasharray:2 1.2}
.ctr{stroke:#c0392b;stroke-width:0.25;stroke-dasharray:6 1.5 1 1.5}
.dim,.ext{stroke:#111;stroke-width:0.18}
.tbl{stroke:#111;stroke-width:0.25}
.tbl-l{stroke:#111;stroke-width:0.25}
text{font-family:"Source Han Sans SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#111}
.t{font-size:2.6px}.th{font-size:3.4px;font-weight:500}.tt{font-size:4px;font-weight:500}
.ts{font-size:2.4px}.td{font-size:2.4px}.tb{font-size:2.4px;font-weight:500}
</style>
${s}
</svg>
`;
}

function main() {
  const args = process.argv.slice(2);
  const get = (f, dflt) => { const i = args.indexOf(f); return i >= 0 && args[i + 1] ? args[i + 1] : dflt; };
  const scale = parseFloat(get('--scale', '1'));
  const out = path.resolve(get('--out', path.join(__dirname, `plan${scale === 1 ? '' : `-${scale}`}.svg`)));
  fs.mkdirSync(path.dirname(out), { recursive: true });
  const svg = sheet(scale);
  fs.writeFileSync(out, svg, 'utf8');
  console.log(`已写出 ${out}`);
  console.log(`A3 横版 420 x 297 mm，标注按 ${scale === 1 ? '1:1 设计尺寸' : `${scale} 折算后的实际尺寸`}`);
  console.log(`${(svg.length / 1024).toFixed(0)} KB`);
}

main();
