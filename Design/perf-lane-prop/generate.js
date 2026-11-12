#!/usr/bin/env node
// 性能泳道板道具 · STL / 3MF 生成器
// 无第三方依赖。用法见同目录 README.md。
//
// 几何来自 geometry.js，plan.js 用的是同一份尺寸。
// 3MF 写入复用隔壁 chip-tray-prop 的 lib3mf.js —— 那是个与模型无关的通用写入器，
// 两个道具共用一份，改一处两边都受益。

'use strict';
const fs = require('fs');
const path = require('path');
const { build3mf } = require('../chip-tray-prop/lib3mf');
const {
  BOARD, LANE, TRACK, UNIT, BAR_LENS, BAR_EACH, TILE, POCKET, pocketCount,
  boardBoxes, colorize, barBoxes,
} = require('./geometry');

/* ---------- STL ---------- */

function boxTris(x0, y0, z0, x1, y1, z1) {
  if (!(x1 > x0 && y1 > y0 && z1 > z0)) return [];
  const v = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const faces = [
    [[0, 2, 1], [0, 0, -1]], [[0, 3, 2], [0, 0, -1]],
    [[4, 5, 6], [0, 0, 1]], [[4, 6, 7], [0, 0, 1]],
    [[0, 1, 5], [0, -1, 0]], [[0, 5, 4], [0, -1, 0]],
    [[1, 2, 6], [1, 0, 0]], [[1, 6, 5], [1, 0, 0]],
    [[2, 3, 7], [0, 1, 0]], [[2, 7, 6], [0, 1, 0]],
    [[3, 0, 4], [-1, 0, 0]], [[3, 4, 7], [-1, 0, 0]],
  ];
  return faces.map(([idx, n]) => ({ n, v: idx.map((k) => v[k]) }));
}

function writeStl(file, tris, name) {
  const buf = Buffer.alloc(84 + tris.length * 50);
  buf.write(`${name} - PyPTO lab tour prop`.slice(0, 79), 0, 'ascii');
  buf.writeUInt32LE(tris.length, 80);
  let o = 84;
  for (const t of tris) {
    buf.writeFloatLE(t.n[0], o); buf.writeFloatLE(t.n[1], o + 4); buf.writeFloatLE(t.n[2], o + 8);
    o += 12;
    for (const p of t.v) {
      buf.writeFloatLE(p[0], o); buf.writeFloatLE(p[1], o + 4); buf.writeFloatLE(p[2], o + 8);
      o += 12;
    }
    buf.writeUInt16LE(0, o); o += 2;
  }
  fs.writeFileSync(file, buf);
  return tris.length;
}

/* ---------- 工具 ---------- */

function clip(boxes, x0, y0, x1, y1) {
  const out = [];
  for (const [bx0, by0, bz0, bx1, by1, bz1] of boxes) {
    const cx0 = Math.max(bx0, x0), cx1 = Math.min(bx1, x1);
    const cy0 = Math.max(by0, y0), cy1 = Math.min(by1, y1);
    if (cx1 - cx0 > 1e-9 && cy1 - cy0 > 1e-9) out.push([cx0 - x0, cy0 - y0, bz0, cx1 - x0, cy1 - y0, bz1]);
  }
  return out;
}

const move = (boxes, dx, dy) =>
  boxes.map(([x0, y0, z0, x1, y1, z1]) => [x0 + dx, y0 + dy, z0, x1 + dx, y1 + dy, z1]);

function emit(outDir, name, boxes, scale, color) {
  if (!boxes.length) return null;
  const tris = [];
  for (const b of boxes) {
    tris.push(...boxTris(b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale, b[4] * scale, b[5] * scale));
  }
  writeStl(path.join(outDir, `${name}.stl`), tris, name);
  let vol = 0, maxX = 0, maxY = 0, maxZ = 0;
  for (const b of boxes) {
    vol += (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]) * scale ** 3;
    maxX = Math.max(maxX, b[3] * scale); maxY = Math.max(maxY, b[4] * scale); maxZ = Math.max(maxZ, b[5] * scale);
  }
  return { name, color, tris: tris.length, vol, size: [maxX, maxY, maxZ] };
}

function emit3mf(outDir, name, groups, scale) {
  const parts = [];
  for (const g of groups) {
    if (!g.boxes.length) continue;
    const tris = [];
    for (const b of g.boxes) {
      tris.push(...boxTris(b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale, b[4] * scale, b[5] * scale));
    }
    parts.push({ name: g.label, tris, extruder: g.extruder });
  }
  if (!parts.length) return null;
  fs.writeFileSync(path.join(outDir, `${name}.3mf`), build3mf(name, parts));
  return { name: `${name}.3mf`, parts: parts.map((p) => `${p.name}=${p.extruder}号`).join(' ') };
}

/* ---------- 15 条 Task 在盘面上的排布 ---------- */
// 按长度从大到小，一行行往右摆，摆不下就换行。行距 = 条宽 + gap。
// 以打印后的实际毫米判断，所以换比例也不用改。
function layout(bars, scale, limit, gap) {
  const placed = [];
  let rowY = 0, rowX = 0, rowH = 0;
  for (const b of bars) {
    const w = b.len * scale, h = b.w * scale;
    if (rowX > 0 && rowX + w > limit) { rowY += rowH + gap; rowX = 0; rowH = 0; }
    placed.push({ ...b, dx: rowX / scale, dy: rowY / scale });
    rowX += w + gap;
    rowH = Math.max(rowH, h);
  }
  return { placed, w: Math.max(...placed.map((p) => p.dx * scale + p.len * scale)), d: rowY + rowH };
}

function main() {
  const args = process.argv.slice(2);
  const get = (flag, dflt) => {
    const i = args.indexOf(flag);
    return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
  };
  const scale = parseFloat(get('--scale', '0.6'));
  const outDir = path.resolve(get('--out', path.join(__dirname, 'stl')));
  const bed = parseFloat(get('--bed', '256'));
  // 每边留出的空地。板要与托盘同宽，0.6 下 252 在 256 的床上每边只剩 2 —— 这是
  // 刻意付的代价，不开 brim。板上唯一的分色是一道 Z 向色皮，换料靠「打到某层停一下」，
  // 不用擦料塔，所以 2 mm 够用。想让生成器替你把板切开，把这个值调大即可。
  const margin = parseFloat(get('--margin', '2'));
  if (!(scale > 0.05 && scale <= 2)) { console.error('--scale 需在 0.05 与 2 之间'); process.exit(1); }
  const lim = bed - 2 * margin;
  const fits = (w, d) => w * scale <= lim && d * scale <= lim;
  let split = get('--split', 'auto');
  // 分割缝刻意落在不碍事的地方：X 缝在第 16 格的刻度上，Y 缝落在 AIC 组与 AIV 组之间的空带里。
  const SX = TRACK.x + 16 * UNIT, SY = 115;
  if (split === 'auto') split = fits(BOARD.W, BOARD.D) ? '1'
    : fits(SX, BOARD.D) && fits(BOARD.W - SX, BOARD.D) ? '2' : '4';
  if (!['1', '2', '4'].includes(split)) { console.error('--split 只接受 auto / 1 / 2 / 4'); process.exit(1); }
  fs.mkdirSync(outDir, { recursive: true });

  const board = colorize(boardBoxes());
  const quads = split === '1' ? [['', 0, 0, BOARD.W, BOARD.D]]
    : split === '2' ? [['-a', 0, 0, SX, BOARD.D], ['-b', SX, 0, BOARD.W, BOARD.D]]
    : [['-a', 0, 0, SX, SY], ['-b', SX, 0, BOARD.W, SY],
       ['-c', 0, SY, SX, BOARD.D], ['-d', SX, SY, BOARD.W, BOARD.D]];

  const report = [];
  const add = (...a) => { const r = emit(...a); if (r) report.push(r); };
  for (const [q, x0, y0, x1, y1] of quads) {
    add(outDir, `board${q}-1-body`, clip(board.body, x0, y0, x1, y1), scale, '深灰');
    add(outDir, `board${q}-2-aic`, clip(board.aic, x0, y0, x1, y1), scale, '白色');
  }

  // 每种长度一个打印件，自己印 3 份；另外给一个 15 条排好的整盘。
  const bars = [];
  for (const len of BAR_LENS) {
    const b = barBoxes(len, scale);
    const u = len / UNIT;
    add(outDir, `bar-${String(u).padStart(2, '0')}u`, b.body, scale, '蓝色');
    for (let i = 0; i < BAR_EACH; i++) bars.push({ len, u, ...b });
  }

  const made = [];
  for (const [q, x0, y0, x1, y1] of quads) {
    const r = emit3mf(outDir, `board${q}`, [
      { label: '本体 深灰', extruder: 1, boxes: clip(board.body, x0, y0, x1, y1) },
      { label: 'AIC 道底 白色', extruder: 2, boxes: clip(board.aic, x0, y0, x1, y1) },
    ], scale);
    if (r) made.push(r);
  }
  for (const len of BAR_LENS) {
    const b = barBoxes(len, scale);
    const u = len / UNIT;
    const r = emit3mf(outDir, `bar-${String(u).padStart(2, '0')}u`, [
      { label: `Task ${u} 格 蓝色`, extruder: 1, boxes: b.body },
    ], scale);
    if (r) made.push(r);
  }
  const lay = layout(bars, scale, lim, 6);
  const allBody = [];
  for (const p of lay.placed) allBody.push(...move(p.body, p.dx, p.dy));
  const rAll = emit3mf(outDir, 'bars-all', [
    { label: 'Task 条 蓝色', extruder: 1, boxes: allBody },
  ], scale);
  if (rAll) made.push(rAll);

  // tile 整盘。数量 = 15 条 Task 上的坑总数，刚好填满。
  // 形状与芯片托盘的 tile 完全一致（TILE 就是从那边 require 过来的），
  // 已经打过托盘的话那 40 个直接拿来用，这里只是补齐的方便件。
  const tileN = BAR_LENS.reduce((a, l) => a + pocketCount(l), 0) * BAR_EACH;
  const step = TILE + 4 / scale;                       // 件间距打印后恒为 4 mm
  const cols = Math.max(1, Math.min(Math.floor(lim / (step * scale)), Math.ceil(Math.sqrt(tileN))));
  const tiles = [];
  for (let i = 0; i < tileN; i++) {
    const cx = (i % cols) * step, cy = Math.floor(i / cols) * step;
    tiles.push([cx, cy, 0, cx + TILE, cy + TILE, TILE]);
  }
  add(outDir, 'tile', [[0, 0, 0, TILE, TILE, TILE]], scale, '蓝色');
  const rT = emit3mf(outDir, 'tiles-all', [{ label: `tile x${tileN} 蓝色`, extruder: 1, boxes: tiles }], scale);
  if (rT) made.push(rT);
  const tileW = Math.min(cols, tileN) * step * scale - 4;
  const tileD = Math.ceil(tileN / cols) * step * scale - 4;

  /* ---------- 报表 ---------- */
  const pad = (s, n) => String(s).padEnd(n);
  const num = (v, n, d = 1) => v.toFixed(d).padStart(n);
  console.log(`\n比例 ${scale}  输出 ${outDir}\n`);
  console.log(`${pad('零件', 22)}${pad('颜色', 7)}${pad('三角面', 9)}${pad('外廓 mm', 20)}${pad('体积 cm³', 10)}`);
  console.log('-'.repeat(68));
  const byColor = new Map();
  let full = 0;
  for (const r of report) {
    const sz = r.size.map((v) => v.toFixed(0)).join(' x ');
    console.log(`${pad(r.name, 22)}${pad(r.color, 7)}${pad(r.tris, 9)}${pad(sz, 20)}${num(r.vol / 1000, 8)}`);
    const v = r.vol * (r.name.startsWith('bar-') ? BAR_EACH : r.name === 'tile' ? tileN : 1);
    byColor.set(r.color, (byColor.get(r.color) || 0) + v);
    full += v;
  }
  console.log('-'.repeat(68));
  console.log(`整套各色用料估算（每种 Task 条 x${BAR_EACH} 共 ${BAR_LENS.length * BAR_EACH} 条，tile x${tileN}；2 层壁 + 10% 填充）：`);
  for (const [c, v] of [...byColor].sort((a, b) => b[1] - a[1])) {
    console.log(`  ${pad(c, 6)}${num(v / 1000, 8)} cm³   约 ${(v / 1000 * 1.24 * 0.35).toFixed(0)} g`);
  }
  console.log(`  ${pad('合计', 6)}${num(full / 1000, 8)} cm³   约 ${(full / 1000 * 1.24 * 0.35).toFixed(0)} g\n`);
  console.log('组装好的 3MF（打开即可切片，耗材号已指定）：');
  for (const m of made) console.log(`  ${m.name.padEnd(18)}${m.parts}`);
  console.log('');
  console.log(`bars-all 整盘占 ${(lay.w).toFixed(0)} x ${(lay.d).toFixed(0)} mm，${bars.length} 条一次打完`);
  console.log(`tiles-all 整盘占 ${tileW.toFixed(0)} x ${tileD.toFixed(0)} mm，${tileN} 个 tile 一次打完`);
  console.log(`Task 条上的 tile 坑：${BAR_LENS.map((l) => pocketCount(l)).join(' / ')} 个（每 ${POCKET.pitch / UNIT} 格一个），15 条共 ${tileN} 个
`);

  const over = report.filter((r) => r.size.slice(0, 2).some((v) => v > lim));
  if (over.length) console.log(`注意：${over.map((r) => r.name).join(', ')} 在 ${bed} mm 幅面上每边留不出 ${margin} mm，请加 --split 2 或 4\n`);
  else console.log(`所有零件在 ${bed} x ${bed} 床面上每边都留得出 ${margin} mm 空地（brim / 擦料塔用）\n`);

  // 摆法提示：最优跨度 = max(最长条, 总长 / 道数) 的下界，靠人工配好的解报出
  const lanes = 6, total = BAR_LENS.reduce((s, l) => s + l, 0) * BAR_EACH / UNIT;
  console.log(`15 条合计 ${total} 格；6 条道、每道 ${TRACK.len / UNIT} 格。`);
  console.log(`理论下界 max(${Math.max(...BAR_LENS) / UNIT}, ceil(${total}/${lanes})) = ${Math.max(Math.max(...BAR_LENS) / UNIT, Math.ceil(total / lanes))} 格，已知可达 25 格。\n`);
}

main();
