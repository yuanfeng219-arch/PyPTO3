#!/usr/bin/env node
// 芯片模型托盘道具 · STL 生成器
// 无第三方依赖。用法见同目录 README.md。
//
// 所有造型都是「自顶向下的棱柱」：一块矩形板，从顶面挖若干矩形坑、垫若干矩形台。
// 因此可以用高度场求解：把所有特征边界投影成网格，逐格算出该格的顶面高度，
// 每格输出一个长方体。相邻同高度的格子沿 X 合并，降低三角面数量。
// 输出的若干长方体互相不重叠，整体是闭合实体，切片软件可直接读。

'use strict';
const fs = require('fs');
const path = require('path');
const { build3mf } = require('./lib3mf');

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

/* ---------- 几何定义 ---------- */
// 尺寸与造型全部来自 geometry.js，plan.js 用的是同一份。
const {
  TRAY, CELL, LID, TILE,
  field, trayOps, colorize, lidBoxes, bigBlockBoxes,
} = require('./geometry');

/* ---------- 切块与输出 ---------- */

function clip(boxes, x0, y0, x1, y1) {
  const out = [];
  for (const [bx0, by0, bz0, bx1, by1, bz1] of boxes) {
    const cx0 = Math.max(bx0, x0), cx1 = Math.min(bx1, x1);
    const cy0 = Math.max(by0, y0), cy1 = Math.min(by1, y1);
    if (cx1 - cx0 > 1e-9 && cy1 - cy0 > 1e-9) out.push([cx0 - x0, cy0 - y0, bz0, cx1 - x0, cy1 - y0, bz1]);
  }
  return out;
}

function emit(outDir, name, boxes, scale, color) {
  if (!boxes.length) return null;
  const tris = [];
  for (const b of boxes) {
    tris.push(...boxTris(b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale, b[4] * scale, b[5] * scale));
  }
  const file = path.join(outDir, `${name}.stl`);
  writeStl(file, tris, name);
  let vol = 0, maxX = 0, maxY = 0, maxZ = 0;
  for (const b of boxes) {
    vol += (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]) * scale ** 3;
    maxX = Math.max(maxX, b[3] * scale); maxY = Math.max(maxY, b[4] * scale); maxZ = Math.max(maxZ, b[5] * scale);
  }
  return { name, color, tris: tris.length, vol, size: [maxX, maxY, maxZ] };
}

// 一个打印件 = 一个 3MF，内含若干已指定耗材的零件
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

function main() {
  const args = process.argv.slice(2);
  const get = (flag, dflt) => {
    const i = args.indexOf(flag);
    return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
  };
  const scale = parseFloat(get('--scale', '1'));
  const outDir = path.resolve(get('--out', path.join(__dirname, 'stl')));
  const bed = parseFloat(get('--bed', '256'));
  // 每边要留出的空地。brim 要 8，AMS 的擦料塔还要更多。
  // 以前这里只留 4，结果 0.6 比例的 252 件在 256 的床上每边只剩 2 —— brim 和擦料塔都放不下。
  const margin = parseFloat(get('--margin', '20'));
  const faceUp = args.includes('--lid-up');   // 盖子外观面朝上导出，需要开支撑
  if (!(scale > 0.05 && scale <= 2)) { console.error('--scale 需在 0.05 与 2 之间'); process.exit(1); }
  const fits = (w, d) => w * scale + 2 * margin <= bed && d * scale + 2 * margin <= bed;
  let split = get('--split', 'auto');
  if (split === 'auto') split = fits(TRAY.W, TRAY.D) ? '1' : fits(TRAY.W / 2, TRAY.D) ? '2' : '4';
  if (!['1', '2', '4'].includes(split)) { console.error('--split 只接受 auto / 1 / 2 / 4'); process.exit(1); }
  fs.mkdirSync(outDir, { recursive: true });

  const tray = field({ x0: 0, y0: 0, x1: TRAY.W, y1: TRAY.D }, 0, TRAY.H, trayOps());
  const lid = lidBoxes(scale, faceUp);
  const quads = split === '1' ? [['', 0, 0, TRAY.W, TRAY.D]]
    : split === '2' ? [['-a', 0, 0, 210, 300], ['-b', 210, 0, 420, 300]]
    : [['-a', 0, 0, 210, 150], ['-b', 210, 0, 420, 150],
       ['-c', 0, 150, 210, 300], ['-d', 210, 150, 420, 300]];

  const trayC = colorize(tray);
  const report = [];
  const add = (...a) => { const r = emit(...a); if (r) report.push(r); };
  for (const [q, x0, y0, x1, y1] of quads) {
    add(outDir, `tray${q}-1-body`, clip(trayC.body, x0, y0, x1, y1), scale, '深灰');
    add(outDir, `tray${q}-2-pcb`, clip(trayC.pcb, x0, y0, x1, y1), scale, '绿色');
    add(outDir, `tray${q}-3-block`, clip(trayC.block, x0, y0, x1, y1), scale, '白色');
  }
  for (const [q, x0, y0, x1, y1] of quads) {
    add(outDir, `lid${q}-1-body`, clip(lid.body, x0, y0, x1, y1), scale, '近黑');
    add(outDir, `lid${q}-2-pads`, clip(lid.pads, x0, y0, x1, y1), scale, '银色');
  }
  add(outDir, 'tile', [[0, 0, 0, TILE, TILE, TILE]], scale, '蓝色');
  add(outDir, 'bigblock', bigBlockBoxes(), scale, '蓝色');

  // 组装好的 3MF：打开即是拼好的对象，耗材号已指定
  const made = [];
  for (const [q, x0, y0, x1, y1] of quads) {
    const r = emit3mf(outDir, `tray${q}`, [
      { label: '本体 深灰', extruder: 1, boxes: clip(trayC.body, x0, y0, x1, y1) },
      { label: 'PCB 绿色', extruder: 2, boxes: clip(trayC.pcb, x0, y0, x1, y1) },
      { label: '凸台 白色', extruder: 3, boxes: clip(trayC.block, x0, y0, x1, y1) },
    ], scale);
    if (r) made.push(r);
  }
  for (const [q, x0, y0, x1, y1] of quads) {
    const r = emit3mf(outDir, `lid${q}`, [
      { label: '本体 近黑', extruder: 1, boxes: clip(lid.body, x0, y0, x1, y1) },
      { label: '贴片 银色', extruder: 2, boxes: clip(lid.pads, x0, y0, x1, y1) },
    ], scale);
    if (r) made.push(r);
  }
  emit3mf(outDir, 'tile', [{ label: 'Tile 蓝色', extruder: 1, boxes: [[0, 0, 0, TILE, TILE, TILE]] }], scale);
  emit3mf(outDir, 'bigblock', [{ label: '大块 蓝色', extruder: 1, boxes: bigBlockBoxes() }], scale);

  const pad = (s, n) => String(s).padEnd(n);
  const num = (v, n, d = 1) => v.toFixed(d).padStart(n);
  console.log(`\n比例 ${scale}  输出 ${outDir}\n`);
  console.log(`${pad('零件', 20)}${pad('颜色', 7)}${pad('三角面', 9)}${pad('外廓 mm', 20)}${pad('体积 cm³', 10)}`);
  console.log('-'.repeat(66));
  const sets = { 'tile': 40, 'bigblock': 3 };
  const byColor = new Map();
  let full = 0;
  for (const r of report) {
    const sz = r.size.map((v) => v.toFixed(0)).join(' x ');
    console.log(`${pad(r.name, 20)}${pad(r.color, 7)}${pad(r.tris, 9)}${pad(sz, 20)}${num(r.vol / 1000, 8)}`);
    const v = r.vol * (sets[r.name] || 1);
    byColor.set(r.color, (byColor.get(r.color) || 0) + v);
    full += v;
  }
  console.log('-'.repeat(66));
  console.log('整套各色用料估算（Tile x40、大块 x3，其余各 1；2 层壁 + 10% 填充）：');
  for (const [c, v] of [...byColor].sort((a, b) => b[1] - a[1])) {
    console.log(`  ${pad(c, 6)}${num(v / 1000, 8)} cm³   约 ${(v / 1000 * 1.24 * 0.35).toFixed(0)} g`);
  }
  console.log(`  ${pad('合计', 6)}${num(full / 1000, 8)} cm³   约 ${(full / 1000 * 1.24 * 0.35).toFixed(0)} g\n`);
  console.log('组装好的 3MF（打开即可切片，耗材号已指定）：');
  for (const m of made) console.log(`  ${m.name.padEnd(18)}${m.parts}`);
  console.log('');

  const lim = bed - 2 * margin;
  const over = report.filter((r) => r.size.slice(0, 2).some((v) => v > lim));
  if (over.length) console.log(`注意：${over.map((r) => r.name).join(', ')} 在 ${bed} mm 幅面上每边留不出 ${margin} mm，请加 --split 2 或 4\n`);
  else console.log(`所有零件在 ${bed} x ${bed} 床面上每边都留得出 ${margin} mm 空地（brim / 擦料塔用）\n`);
}

main();
