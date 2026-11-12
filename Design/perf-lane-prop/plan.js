#!/usr/bin/env node
// 性能泳道板道具 · A3 平面图生成器
// 尺寸全部来自 geometry.js，与 STL / 3MF 同一份口径。
// 输出 plan.svg：A3 横（420 x 297 mm），浏览器可直接打开，打印选 A3 横、100%。

'use strict';
const fs = require('fs');
const path = require('path');
const G = require('./geometry');

const n = (v) => (Math.round(v * 1000) / 1000).toString();

/* ---------- 画图原语 ---------- */
const O = [];
const rect = (x, y, w, h, o = {}) => O.push(
  `<rect x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" fill="${o.fill || 'none'}" `
  + `stroke="${o.s || '#222'}" stroke-width="${n(o.w || 0.25)}"`
  + `${o.dash ? ` stroke-dasharray="${o.dash}"` : ''}/>`);
const line = (x1, y1, x2, y2, o = {}) => O.push(
  `<line x1="${n(x1)}" y1="${n(y1)}" x2="${n(x2)}" y2="${n(y2)}" stroke="${o.s || '#222'}" `
  + `stroke-width="${n(o.w || 0.25)}"${o.dash ? ` stroke-dasharray="${o.dash}"` : ''}/>`);
const txt = (x, y, s, o = {}) => O.push(
  `<text x="${n(x)}" y="${n(y)}" font-family="Noto Sans SC, Microsoft YaHei, sans-serif" `
  + `font-size="${n(o.fs || 2.6)}" fill="${o.fill || '#222'}" text-anchor="${o.a || 'start'}"`
  + `${o.wt ? ` font-weight="${o.wt}"` : ''}>${s}</text>`);

// 尺寸线。ext 为引出线长度，正数向上/向左引。
function dimH(x1, x2, y, label, ext = 2) {
  line(x1, y - ext, x1, y + 1, { s: '#888', w: 0.18 });
  line(x2, y - ext, x2, y + 1, { s: '#888', w: 0.18 });
  O.push(`<line x1="${n(x1)}" y1="${n(y)}" x2="${n(x2)}" y2="${n(y)}" stroke="#888" stroke-width="0.18" `
    + `marker-start="url(#a)" marker-end="url(#a)"/>`);
  txt((x1 + x2) / 2, y - 0.9, label, { a: 'middle', fs: 2.3, fill: '#555' });
}
function dimV(y1, y2, x, label, ext = 2) {
  line(x - ext, y1, x + 1, y1, { s: '#888', w: 0.18 });
  line(x - ext, y2, x + 1, y2, { s: '#888', w: 0.18 });
  O.push(`<line x1="${n(x)}" y1="${n(y1)}" x2="${n(x)}" y2="${n(y2)}" stroke="#888" stroke-width="0.18" `
    + `marker-start="url(#a)" marker-end="url(#a)"/>`);
  O.push(`<text x="${n(x - 0.9)}" y="${n((y1 + y2) / 2)}" font-family="Noto Sans SC, Microsoft YaHei, sans-serif" `
    + `font-size="2.3" fill="#555" text-anchor="middle" transform="rotate(-90 ${n(x - 0.9)} ${n((y1 + y2) / 2)})">${label}</text>`);
}
const title = (x, y, s, sub) => {
  txt(x, y, s, { fs: 3.4, wt: 600 });
  if (sub) txt(x, y + 4, sub, { fs: 2.3, fill: '#777' });
};

/* ---------- 视图一：泳道板俯视 ---------- */
function boardView(ox, oy, k, S) {
  const { BOARD, LANE, TRACK, UNIT, TICKS, AXIS, HEADER, laneYs, AIC_N } = G;
  rect(ox, oy, BOARD.W * k, BOARD.D * k, { w: 0.4 });

  // 时间轴刻度
  for (let i = 0; i <= TICKS; i++) {
    const x = ox + (TRACK.x + i * UNIT) * k;
    const len = (i % 5 === 0 ? G.TICK.major : G.TICK.minor) * k;
    line(x, oy + (AXIS.top + AXIS.h) * k - len, x, oy + (AXIS.top + AXIS.h) * k, { s: '#666', w: 0.22 });
    line(x, oy + AXIS.bottom * k, x, oy + AXIS.bottom * k + len, { s: '#666', w: 0.22 });
    if (i % 5 === 0) {
      txt(x, oy + AXIS.top * k - 1.2, String(i), { a: 'middle', fs: 2.2, fill: '#666' });
      txt(x, oy + (AXIS.bottom + G.TICK.major) * k + 3, String(i), { a: 'middle', fs: 2.2, fill: '#666' });
    }
  }
  txt(ox + 3, oy + (AXIS.top + AXIS.h) * k, '时间 →', { fs: 2.6, fill: '#666', wt: 500 });
  txt(ox + 3, oy + (AXIS.top + AXIS.h) * k + 3.4, '一格 = ' + UNIT + ' mm', { fs: 2.2, fill: '#888' });

  // 泳道与标签坑
  laneYs.forEach((y, i) => {
    const aic = i < AIC_N;
    rect(ox + TRACK.x * k, oy + y * k, TRACK.len * k, LANE.w * k,
      { w: 0.3, fill: aic ? '#f2f2f2' : 'none' });
    rect(ox + HEADER.x * k, oy + (y + (LANE.w - 16) / 2) * k, HEADER.w * k, 16 * k,
      { w: 0.22, dash: '1.2 0.8' });
    txt(ox + (HEADER.x + HEADER.w / 2) * k, oy + (y + LANE.w / 2 + 1.6) * k,
      (aic ? 'AIC ' + i : 'AIV ' + (i - AIC_N)), { a: 'middle', fs: 2.4, wt: 500 });
  });
  txt(ox + (BOARD.W + 44) * k, oy + (laneYs[0] + LANE.w) * k, 'Cube', { fs: 2.4, fill: '#777' });
  txt(ox + (BOARD.W + 44) * k, oy + (laneYs[AIC_N + 1] + LANE.w) * k, 'Vector', { fs: 2.4, fill: '#777' });

  // 尺寸
  dimH(ox, ox + BOARD.W * k, oy - 14, `${BOARD.W} (${S(BOARD.W)})`, 3);
  dimH(ox + TRACK.x * k, ox + (TRACK.x + TRACK.len) * k, oy - 7,
    `泳道长 ${TRACK.len} = ${TICKS} 格 (${S(TRACK.len)})`, 3);
  dimV(oy, oy + BOARD.D * k, ox - 10, `${BOARD.D} (${S(BOARD.D)})`, 3);
  dimV(oy + laneYs[0] * k, oy + (laneYs[0] + LANE.w) * k, ox + (BOARD.W + 16) * k, `槽宽 ${LANE.w}`, 2.5);
  dimV(oy + laneYs[0] * k, oy + laneYs[1] * k, ox + (BOARD.W + 26) * k, `间距 ${LANE.pitch}`, 2.5);
  const gy0 = laneYs[AIC_N - 1] + LANE.w, gy1 = laneYs[AIC_N];
  dimV(oy + gy0 * k, oy + gy1 * k, ox + (BOARD.W + 36) * k, `组间 ${gy1 - gy0}`, 2.5);

  // 剖切符号。剖面 A-A 看的是跨泳道方向的断面，所以剖切线是竖的。
  const cx = TRACK.x + 16 * UNIT;
  line(ox + cx * k, oy - 4, ox + cx * k, oy + BOARD.D * k + 2, { s: '#c33', w: 0.3, dash: '4 1.5 1 1.5' });
  txt(ox + cx * k, oy - 5, 'A', { a: 'middle', fs: 3, fill: '#c33', wt: 600 });
  txt(ox + cx * k + 1.6, oy + BOARD.D * k + 1, 'A', { fs: 3, fill: '#c33', wt: 600 });
}

/* ---------- 视图二：剖面 A-A ---------- */
// 过第 3 条道（第一条 AIV）横切，把底板厚、槽深、道间筋一次讲清。
function sectionView(ox, oy, k, S) {
  const { BOARD, LANE, laneYs, AIC_N } = G;
  // 刻意跨过 AIC 与 AIV 的分组带，一张断面上同时看到白色的 AIC 道底和深色的 AIV 道底
  const y0 = laneYs[AIC_N - 1] - 4, y1 = laneYs[AIC_N] + LANE.w + 4;
  const base = oy + BOARD.H * k;
  // 画成「底板整条 + 道间凸起的筋」，剖面一眼看出薄板靠筋撑住
  rect(ox, base - BOARD.floor * k, (y1 - y0) * k, BOARD.floor * k, { fill: '#e7e7e7', w: 0.3 });
  let prev = y0;
  for (const y of laneYs) {
    if (y + LANE.w < y0 || y > y1) continue;
    if (y > prev) rect(ox + (prev - y0) * k, base - BOARD.H * k, (y - prev) * k, (BOARD.H - BOARD.floor) * k,
      { fill: '#e7e7e7', w: 0.3 });
    // 槽底的白色色皮
    if (y < laneYs[AIC_N]) rect(ox + (y - y0) * k, base - BOARD.floor * k, LANE.w * k, G.SKIN * k,
      { fill: '#fff', w: 0.25 });
    prev = y + LANE.w;
  }
  if (prev < y1) rect(ox + (prev - y0) * k, base - BOARD.H * k, (y1 - prev) * k, (BOARD.H - BOARD.floor) * k,
    { fill: '#e7e7e7', w: 0.3 });

  // 一条 Task 落在 AIV0 的槽里
  const lane = laneYs[AIC_N], bw = G.LANE.w - 0.7;
  rect(ox + (lane - y0 + 0.35) * k, base - (BOARD.floor + G.BAR.h) * k, bw * k, G.BAR.h * k,
    { fill: '#cfe0f5', w: 0.3 });
  const barTop = base - (BOARD.floor + G.BAR.h) * k;
  const b1 = G.barBoxes(G.BAR_LENS[0], 1), fit = (b1.ps - G.TILE) / 2;
  const px = ox + (lane - y0 + 0.35 + (bw - b1.ps) / 2) * k;
  rect(px, barTop, b1.ps * k, G.POCKET.depth * k, { fill: '#fff', w: 0.25 });           // tile 坑
  rect(px + fit * k, barTop - (G.TILE - G.POCKET.depth) * k, G.TILE * k, G.TILE * k,
    { fill: '#9ec5f0', w: 0.3 });                                                        // 坐在坑里的 tile
  txt(px + b1.ps * k / 2, barTop - (G.TILE - G.POCKET.depth) * k - 2,
    'tile', { a: 'middle', fs: 2.4, fill: '#36c', wt: 500 });
  txt(ox + (lane - y0 + 2) * k, base - (BOARD.floor + G.BAR.h / 2) * k + 1,
    'Task 条', { fs: 2.4, fill: '#36c', wt: 500 });
  txt(ox + (laneYs[AIC_N - 1] - y0 + LANE.w / 2) * k, base - BOARD.floor * k - 1.6,
    'AIC 道底（白）', { a: 'middle', fs: 2.1, fill: '#777' });

  dimV(base - BOARD.floor * k, base, ox - 7, `底板 ${BOARD.floor} (${S(BOARD.floor)})`, 2.5);
  dimV(base - BOARD.H * k, base - BOARD.floor * k, ox - 18, `槽深 ${LANE.depth}`, 2.5);
  dimV(base - (BOARD.floor + G.BAR.h) * k, base - BOARD.floor * k, ox + (y1 - y0 + 10) * k, `条高 ${G.BAR.h}`, 2.5);
  const rib0 = laneYs[AIC_N] + LANE.w, rib1 = laneYs[AIC_N + 1];
  dimH(ox + (rib0 - y0) * k, ox + (rib1 - y0) * k, base + 7, `道间筋 ${rib1 - rib0}`, 2.5);
  txt(ox, base + 13, `板总厚 ${BOARD.H} mm（${S(BOARD.H)}）；筋就是道间那 ${rib1 - rib0} mm，薄板靠它撑住`,
    { fs: 2.3, fill: '#777' });
}

/* ---------- 视图三：五种 Task 条 ---------- */
function barsView(ox, oy, k, pitch, S) {
  const { BAR_LENS, BAR, UNIT, BAR_EACH, POCKET } = G;
  let y = oy;
  for (const len of BAR_LENS) {
    const b = G.barBoxes(len, 1);
    rect(ox, y, len * k, BAR.h * k, { fill: '#cfe0f5', w: 0.3 });
    const gx = (len - b.n * POCKET.pitch) / 2;
    for (let i = 0; i < b.n; i++) {
      const cx = gx + (i + 0.5) * POCKET.pitch;
      rect(ox + (cx - b.ps / 2) * k, y, b.ps * k, POCKET.depth * k, { fill: '#fff', w: 0.25 });
    }
    txt(ox + len * k + 2.5, y + BAR.h * k - 2.4,
      `${len / UNIT} 格 · ${b.n} 个 tile 坑`, { fs: 2.5, wt: 500 });
    txt(ox + len * k + 2.5, y + BAR.h * k + 1.4,
      `${len} mm (${S(len)}) · x${BAR_EACH}`, { fs: 2.2, fill: '#888' });
    y += pitch;
  }
  dimV(oy, oy + BAR.h * k, ox - 5, `高 ${BAR.h}`, 2);
  dimV(oy, oy + POCKET.depth * k, ox - 11, `坑深 ${POCKET.depth}`, 2);
}

/* ---------- 视图四：最优摆法 ---------- */
// 18+7 三道、12+9+4 三道 → 每道 25 格，6 道整齐齐平。
const SOLUTION = [[18, 7], [18, 7], [18, 7], [12, 9, 4], [12, 9, 4], [12, 9, 4]];
function solutionView(ox, oy, k) {
  const { TICKS, UNIT, LANE, AIC_N } = G;
  const pitch = LANE.w * k + 2.2;
  const span = Math.max(...SOLUTION.map((l) => l.reduce((a, b) => a + b, 0)));
  SOLUTION.forEach((lane, i) => {
    const y = oy + i * pitch;
    rect(ox, y, TICKS * UNIT * k, LANE.w * k, { w: 0.25, fill: i < AIC_N ? '#f2f2f2' : 'none' });
    txt(ox - 2, y + LANE.w * k - 1.2, i < AIC_N ? 'AIC' : 'AIV', { a: 'end', fs: 2.1, fill: '#777' });
    let x = 0;
    for (const u of lane) {
      rect(ox + x * UNIT * k, y, u * UNIT * k, LANE.w * k, { fill: '#cfe0f5', w: 0.25 });
      txt(ox + (x + u / 2) * UNIT * k, y + LANE.w * k - 1.4, String(u), { a: 'middle', fs: 2.1 });
      x += u;
    }
  });
  const h = SOLUTION.length * pitch;
  line(ox + span * UNIT * k, oy - 2.5, ox + span * UNIT * k, oy + h, { s: '#c33', w: 0.35 });
  txt(ox + span * UNIT * k + 1.2, oy - 3, `${span} 格`, { fs: 2.4, fill: '#c33', wt: 600 });
  line(ox + TICKS * UNIT * k, oy - 2.5, ox + TICKS * UNIT * k, oy + h, { s: '#999', w: 0.25, dash: '1.5 1' });
  txt(ox + TICKS * UNIT * k, oy - 3, `道长 ${TICKS}`, { a: 'end', fs: 2.2, fill: '#888' });
  txt(ox, oy + h + 3.6, `穷举最优就是 ${span} 格：18+7 三道、12+9+4 三道，六道齐平。`, { fs: 2.3, fill: '#555' });
  txt(ox, oy + h + 7, `右边空出的 ${TICKS - span} 格（${((TICKS - span) / TICKS * 100).toFixed(0)}%）就是调度省下来的时间。`,
    { fs: 2.3, fill: '#555' });
}

/* ---------- 特征尺寸表 ---------- */
function table(ox, oy, w, S) {
  const rows = [
    ['板外廓', `${G.BOARD.W} x ${G.BOARD.D} x ${G.BOARD.H}`, `${S(G.BOARD.W)} x ${S(G.BOARD.D)} x ${S(G.BOARD.H)}`],
    ['泳道槽', `${G.TRACK.len} x ${G.LANE.w} 深 ${G.LANE.depth}`, `${S(G.TRACK.len)} x ${S(G.LANE.w)} 深 ${S(G.LANE.depth)}`],
    ['道数', `${G.AIC_N} AIC + ${G.AIV_N} AIV`, '真实芯片 24 + 48，同为 1:2'],
    ['道间距 / 筋', `${G.LANE.pitch} / ${G.LANE.pitch - G.LANE.w}`, `${S(G.LANE.pitch)} / ${S(G.LANE.pitch - G.LANE.w)}`],
    ['底板厚', `${G.BOARD.floor}`, `${S(G.BOARD.floor)}`],
    ['一格时间', `${G.UNIT}`, `${S(G.UNIT)}`],
    ['Task 条', `${G.BAR_LENS.map((l) => l / G.UNIT).join(' / ')} 格，各 ${G.BAR_EACH} 条`, `共 ${G.BAR_LENS.length * G.BAR_EACH} 条、${G.BAR_LENS.reduce((a, b) => a + b, 0) * G.BAR_EACH / G.UNIT} 格`],
    ['条截面', `宽 ${G.LANE.w} - 2x间隙，高 ${G.BAR.h}`, `高 ${S(G.BAR.h)}`],
    ['tile 坑', `每 ${G.POCKET.pitch / G.UNIT} 格一个，深 ${G.POCKET.depth}`,
      `${G.BAR_LENS.map((l) => G.pocketCount(l)).join('/')} 个，15 条共 ${G.BAR_LENS.reduce((a, l) => a + G.pocketCount(l), 0) * G.BAR_EACH} 个`],
    ['tile', `${G.TILE} 立方`, `${S(G.TILE)} 立方，与芯片托盘同件`],
    ['配合间隙', `每边 ${G.FIT} mm`, '不随比例缩放，打出来恒为 ' + G.FIT],
  ];
  const rh = 4.4, c1 = 24, c2 = 58;
  txt(ox, oy - 2, '特征尺寸', { fs: 3, wt: 600 });
  rect(ox, oy, w, rh * (rows.length + 1), { w: 0.3 });
  txt(ox + 1.5, oy + 3, '项', { fs: 2.3, wt: 600 });
  txt(ox + c1, oy + 3, '1:1 设计值 mm', { fs: 2.3, wt: 600 });
  txt(ox + c1 + c2, oy + 3, '实际打印值 / 说明', { fs: 2.3, wt: 600 });
  line(ox, oy + rh, ox + w, oy + rh, { w: 0.3 });
  rows.forEach((r, i) => {
    const y = oy + rh * (i + 2) - 1.4;
    txt(ox + 1.5, y, r[0], { fs: 2.3 });
    txt(ox + c1, y, r[1], { fs: 2.3, fill: '#444' });
    txt(ox + c1 + c2, y, r[2], { fs: 2.3, fill: '#777' });
    if (i < rows.length - 1) line(ox, oy + rh * (i + 2), ox + w, oy + rh * (i + 2), { s: '#ddd', w: 0.15 });
  });
  line(ox + c1 - 2, oy, ox + c1 - 2, oy + rh * (rows.length + 1), { s: '#ddd', w: 0.15 });
  line(ox + c1 + c2 - 2, oy, ox + c1 + c2 - 2, oy + rh * (rows.length + 1), { s: '#ddd', w: 0.15 });
}

/* ---------- 注 + 标题栏 ---------- */
function notes(ox, oy, lines) {
  txt(ox, oy, '注', { fs: 3, wt: 600 });
  lines.forEach((s, i) => txt(ox, oy + 4.2 + i * 3.1, s, { fs: 2.1, fill: '#555' }));
}
function titleBlock(ox, oy, w, h, scale) {
  rect(ox, oy, w, h, { w: 0.4 });
  line(ox, oy + 7, ox + w, oy + 7, { w: 0.3 });
  line(ox + w * 0.62, oy, ox + w * 0.62, oy + h, { w: 0.3 });
  txt(ox + 2, oy + 5, '算子性能泳道板 · Lab Tour 实物道具', { fs: 3.2, wt: 600 });
  txt(ox + w * 0.62 + 2, oy + 5, 'PyPTO 3.0', { fs: 3, wt: 600 });
  const L = [
    ['比例', scale === 1 ? '1:1' : `${scale}（标注括号内为实际打印值）`],
    ['单位', 'mm，第一象限投影'],
    ['件数', `板 1 + Task 条 ${G.BAR_LENS.length * G.BAR_EACH}`],
    ['几何来源', 'Design/perf-lane-prop/geometry.js'],
  ];
  L.forEach(([k, v], i) => {
    txt(ox + 2, oy + 11.8 + i * 3.6, k, { fs: 2.2, fill: '#888' });
    txt(ox + 17, oy + 11.8 + i * 3.6, v, { fs: 2.2 });
  });
  txt(ox + w * 0.62 + 2, oy + 11.5, '数据口径', { fs: 2.2, fill: '#888' });
  txt(ox + w * 0.62 + 2, oy + 14.9, 'perf-lane-kit/data/example.json', { fs: 2.1 });
  txt(ox + w * 0.62 + 2, oy + 18.3, 'rank0：72 核 = 24 AIC + 48 AIV', { fs: 2.1 });
}

/* ---------- 组版 ---------- */
function main() {
  const args = process.argv.slice(2);
  const get = (f, d) => { const i = args.indexOf(f); return i >= 0 && args[i + 1] ? args[i + 1] : d; };
  const scale = parseFloat(get('--scale', '0.6'));
  const out = path.resolve(get('--out', path.join(__dirname, 'plan.svg')));
  const S = (v) => (v * scale).toFixed(1);

  title(22, 16, '算子性能泳道板', `1:1 设计尺寸 ${G.BOARD.W} x ${G.BOARD.D} x ${G.BOARD.H} mm；括号内为 ${scale} 比例的实际打印值`);

  txt(22, 24, '视图一  泳道板俯视', { fs: 3, wt: 600 });
  boardView(22, 46, 0.46, S);

  txt(278, 24, '视图二  剖面 A-A', { fs: 3, wt: 600 });
  sectionView(292, 56, 1.1, S);

  table(278, 94, 126, S);

  txt(278, 160, '视图四  编译器会排成什么样', { fs: 3, wt: 600 });
  solutionView(292, 170, 0.3);

  txt(22, 192, '视图三  Task 条（五种长度，各 3 条）', { fs: 3, wt: 600 });
  barsView(28, 202, 0.45, 12.5, S);

  notes(22, 267, [
    '1. 板与 Task 条分两次打印。板平贴热床、泳道朝上，零支撑，首层 100% 实心。',
    `2. 0.6 比例下板为 ${(G.BOARD.W * 0.6).toFixed(0)} x ${(G.BOARD.D * 0.6).toFixed(0)}，与芯片托盘同宽；256 床面每边只剩 2 mm，不要开 brim，`,
    '   用光面板 + 65 度热床压住翘边。全板唯一的分色面是 AIC 道底的白皮，打到该层停机换料即可，不需要擦料塔。',
    `3. 板面（z=${G.BOARD.H}）顶面约 191 cm²，顶层层数设到 6 层以上，否则大面会起波纹；槽底（z=${G.BOARD.floor}）只有 9 层，切片会整片打实。`,
    `4. 配合间隙恒为每边 ${G.FIT} mm，不随比例缩放；条略松于槽、tile 略松于坑，便于反复取放。`,
    '5. tile 就是芯片托盘那套小方块，必须与 Task 条按同一比例打印，否则放不进坑里。',
  ]);

  titleBlock(252, 265, 153, 23, scale);

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" viewBox="0 0 420 297">
<defs><marker id="a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="4" markerHeight="4"
 orient="auto-start-reverse"><path d="M 0 1 L 8 4 L 0 7 z" fill="#888"/></marker></defs>
<rect width="420" height="297" fill="#fff"/>
<rect x="8" y="8" width="404" height="281" fill="none" stroke="#222" stroke-width="0.5"/>
${O.join('\n')}
</svg>`;
  fs.writeFileSync(out, svg);
  console.log(`已写出 ${out}  （A3 横，打印请选 A3、100%，比例标注 ${scale}）`);
}

main();
