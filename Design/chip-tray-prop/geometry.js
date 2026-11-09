'use strict';
// 道具几何定义（单位 mm，1:1 尺寸）。
// generate.js（STL / 3MF）与 plan.js（SVG 平面图）共用本文件，保证两边尺寸永远一致。
//
// 所有造型都是「自顶向下的棱柱」：一块矩形板，从顶面挖若干矩形坑、垫若干矩形台。
// 因此可以用高度场求解：把所有特征边界投影成网格，逐格算出该格的顶面高度，
// 每格输出一个长方体。相邻同高度的格子沿 X 合并，降低三角面数量。
// 输出的若干长方体互相不重叠，整体是闭合实体。


// op: {t:'r'|'p', x, y, w, d, v}  r 抬高 v，p 下挖 v
const R = (x, y, w, d, v) => ({ t: 'r', x, y, w, d, v });
const P = (x, y, w, d, v) => ({ t: 'p', x, y, w, d, v });

function field(rect, baseZ, topZ, ops) {
  const xs = new Set([rect.x0, rect.x1]);
  const ys = new Set([rect.y0, rect.y1]);
  for (const o of ops) { xs.add(o.x); xs.add(o.x + o.w); ys.add(o.y); ys.add(o.y + o.d); }
  const X = [...xs].filter((v) => v >= rect.x0 && v <= rect.x1).sort((a, b) => a - b);
  const Y = [...ys].filter((v) => v >= rect.y0 && v <= rect.y1).sort((a, b) => a - b);
  const boxes = [];
  for (let j = 0; j < Y.length - 1; j++) {
    const cy = (Y[j] + Y[j + 1]) / 2;
    let run = null;
    for (let i = 0; i < X.length - 1; i++) {
      const cx = (X[i] + X[i + 1]) / 2;
      let h = topZ;
      for (const o of ops) {
        if (cx > o.x && cx < o.x + o.w && cy > o.y && cy < o.y + o.d) {
          h += o.t === 'r' ? o.v : -o.v;
        }
      }
      if (run && Math.abs(run.h - h) < 1e-9) { run.x1 = X[i + 1]; }
      else { if (run) boxes.push(run); run = { x0: X[i], x1: X[i + 1], y0: Y[j], y1: Y[j + 1], h }; }
    }
    if (run) boxes.push(run);
  }
  return boxes.filter((b) => b.h > baseZ + 1e-9)
    .map((b) => [b.x0, b.y0, baseZ, b.x1, b.y1, b.h]);
}

/* ---------- 零件定义（单位 mm，1:1 尺寸） ---------- */

const TRAY = { W: 420, D: 300, H: 25 };
const BIN = { x: 80, y: 246, depth: 18, left: 15, right: 325, top: 27 };
const DIE = { x: 115, y: 27, w: 190, d: 246, rise: 6 };      // 芯片底板，顶面 31
const CELL = { w: 40, d: 34, rise: 12, px: 4, py: 14, pw: 32, pd: 16, pdep: 10 };
const CORE = { w: 86, d: 74, rise: 12 };
const GROOVE = 1.5;        // 搬运通道刻槽深度
const GW = 2.5;            // 搬运通道宽度
// gd / gw 是外观面分隔槽的深宽。槽深只决定「桥在多高处搭」，不影响首层覆盖，
// 所以可以放心做深，让贴片真的读成一块一块，而不是一张平板上的发丝线。
const LID = {
  // top 薄到切片会整片打实心（0.6 比例 1.8 mm = 9 层，默认底 3 + 顶 4 已经盖满），
  // 刚度全部交给加强筋。厚面板中间那几层稀疏填充就是内腔天花板塌陷的根源。
  H: 36, wall: 10, top: 3, skin: 0.6, gw: 3, gd: 2,
  // 面板内侧的加强筋。420 x 300 的薄板翻过来打，最怕的就是冷却收缩把边翘起来。
  // 筋深 10，筋底落在托盘坐标 46（托盘顶 25 + 21）；托盘上最高的白色凸台是 53，
  // 所以 xs / ys 都避开凸台所在的 x 169-189 / 261-281 和 y 203-249。
  rib: { w: 3, depth: 10, xs: [50, 100, 150, 205, 250, 300, 350], ys: [60, 120, 180, 258] },
};
const LID_FIELD = { x0: 46, y0: 44, x1: 372, y1: 255 };   // 贴片区范围，与 PADS 对应
const TILE = 15;
const BIGN = 4;            // 大块由 4x4x2 个 Tile 拼成
const PITCH = 16;

const cellXs = [0, 1, 2, 3].map((c) => 121 + 46 * c);
const cellYs = [0, 1, 2, 3].map((r) => 33 + 40 * r);
const coreXs = [121, 213];
const coreY = 193;

function coreOps(bx, by) {
  return [
    R(bx, by, CORE.w, CORE.d, CORE.rise),                 // 核体，顶面 43
    P(bx + 4, by + 10, 20, 47, 10),                       // 左侧深槽 L1 / UB
    P(bx + 28, by + 10, 16, 20, 4),                       // 中上浅槽
    P(bx + 28, by + 36, 16, 20, 4),                       // 中下浅槽
    R(bx + 48, by + 10, 20, 46, 10),                      // 白色凸台，顶面 53
    P(bx + 72, by + 16, 11, 32, 4),                       // 右侧长槽
    // 贯通的两条搬运通道
    P(bx + 24, by + 18, 4, GW, GROOVE), P(bx + 24, by + 44, 4, GW, GROOVE),
    P(bx + 44, by + 18, 4, GW, GROOVE), P(bx + 44, by + 44, 4, GW, GROOVE),
    P(bx + 48, by + 18, 20, GW, GROOVE), P(bx + 48, by + 44, 20, GW, GROOVE),
    P(bx + 68, by + 18, 4, GW, GROOVE), P(bx + 68, by + 44, 4, GW, GROOVE),
  ];
}

const PCB = { x: 8, y: 8, w: 404, d: 284, skin: 0.6 };  // 顶面绿色区域
const BLOCK_Z = 43;                                      // 此高度以上只剩白色凸台

function trayOps() {
  const ops = [
    P(PCB.x, PCB.y, PCB.w, PCB.d, 0),   // 深度 0：只为让网格在绿色区边界上分格
    P(BIN.left, BIN.top, BIN.x, BIN.y, BIN.depth),
    P(BIN.right, BIN.top, BIN.x, BIN.y, BIN.depth),
    R(DIE.x, DIE.y, DIE.w, DIE.d, DIE.rise),
  ];
  for (const cx of cellXs) for (const cy of cellYs) {
    ops.push(R(cx, cy, CELL.w, CELL.d, CELL.rise));
    ops.push(P(cx + CELL.px, cy + CELL.py, CELL.pw, CELL.pd, CELL.pdep));
  }
  for (const bx of coreXs) ops.push(...coreOps(bx, coreY));
  // 两个放大核之间的连接槽
  ops.push(P(207, coreY + 18, 6, GW, GROOVE), P(207, coreY + 44, 6, GW, GROOVE));
  return ops;
}

// 盖子外观面的银色贴片（lid 局部坐标 u 沿 420，v 沿 300）
// 贴片之间统一留 3 的缝。缝心沿用原布局：u 151.5 / 266.5，v 92.5 / 202.5，
// 贴片区外廓仍是 46,44 - 372,255，所以整体比例没变，只是缝细了。
const PADS = [
  [46, 94, 372, 201],                                                 // 中带
  [46, 44, 150, 91], [153, 44, 265, 91], [268, 44, 372, 91],          // 上排
  [46, 204, 150, 255], [153, 204, 265, 255], [268, 204, 372, 255],    // 下排
  [46, 91, 96, 94], [322, 201, 372, 204],                             // 两处台阶
];

// 贴片厚度不随比例缩放：它是给换色用的表面细节，固定 0.6 mm 才好打印。
// 把托盘实体按颜色拆成三组：深灰本体、绿色顶面皮、白色凸台。
// 拆分面都落在整齐的高度上，没有 AMS 时也能用「打印中暂停换料」做出白色凸台。
function colorize(boxes) {
  const body = [], pcb = [], block = [];
  for (const [x0, y0, z0, x1, y1, z1] of boxes) {
    if (z1 > BLOCK_Z + 1e-9) {
      if (BLOCK_Z > z0 + 1e-9) body.push([x0, y0, z0, x1, y1, BLOCK_Z]);
      block.push([x0, y0, BLOCK_Z, x1, y1, z1]);
    } else if (Math.abs(z1 - TRAY.H) < 1e-9
      && (x0 + x1) / 2 > PCB.x && (x0 + x1) / 2 < PCB.x + PCB.w
      && (y0 + y1) / 2 > PCB.y && (y0 + y1) / 2 < PCB.y + PCB.d) {
      body.push([x0, y0, z0, x1, y1, z1 - PCB.skin]);
      pcb.push([x0, y0, z1 - PCB.skin, x1, y1, z1]);
    } else {
      body.push([x0, y0, z0, x1, y1, z1]);
    }
  }
  return { body, pcb, block };
}

// 盖子必须翻过来打：外观面贴板，空腔朝上。
// 所以外观面必须是一整片平面——任何凸起都会让它成为首层，
// 剩下的部分就得在半空起步。贴片分区因此做成「刻槽」而不是「凸台」：
// 贴片顶面与边框同高，只有缝是凹的，首层照样铺满 420 x 300。
// 槽深 gd 只决定桥的高度，跨距仍是槽宽，所以做深不影响打印。
// 最外 0.6 单独分一色，没有 AMS 时打到 0.6 mm 换料即可。
function lidBoxes(scale, faceUp) {
  const { H, wall, top, gw, gd } = LID;
  const sk = LID.skin / scale;         // 不随比例缩放，打出来始终 0.6 mm
  const F = LID_FIELD;

  const ops = [P(F.x0 - gw, F.y0 - gw, F.x1 - F.x0 + 2 * gw, F.y1 - F.y0 + 2 * gw, gd)];
  for (const [u0, v0, u1, v1] of PADS) ops.push(R(u0, v0, u1 - u0, v1 - v0, gd));
  const plate = field({ x0: 0, y0: 0, x1: TRAY.W, y1: TRAY.D }, H - top, H, ops);

  // 只有贴片顶上那 0.6 分到银色，外圈边框和槽底都归本体。
  // field 的网格是按贴片边界切的，每格要么整格在贴片内、要么整格在外，中心点判断足够。
  const inPad = (cx, cy) => PADS.some(([u0, v0, u1, v1]) => cx > u0 && cx < u1 && cy > v0 && cy < v1);
  const body = [], pads = [];
  for (const [x0, y0, z0, x1, y1, z1] of plate) {
    if (z1 > H - sk + 1e-9) {
      body.push([x0, y0, z0, x1, y1, H - sk]);
      const top0 = [x0, y0, H - sk, x1, y1, z1];
      (inPad((x0 + x1) / 2, (y0 + y1) / 2) ? pads : body).push(top0);
    } else body.push([x0, y0, z0, x1, y1, z1]);
  }
  body.push(
    [0, 0, 0, TRAY.W, wall, H - top],
    [0, TRAY.D - wall, 0, TRAY.W, TRAY.D, H - top],
    [0, wall, 0, wall, TRAY.D - wall, H - top],
    [TRAY.W - wall, wall, 0, TRAY.W, TRAY.D - wall, H - top],
  );

  // 加强筋。纵筋整条通到两侧墙，横筋在纵筋处断开，保证所有盒子互不重叠。
  const { w: rw, depth: rd, xs, ys } = LID.rib;
  const z0 = H - top - rd, z1 = H - top;
  const x0 = wall, x1 = TRAY.W - wall, y0 = wall, y1 = TRAY.D - wall;
  for (const x of xs) body.push([x, y0, z0, x + rw, y1, z1]);
  const cut = [x0];
  for (const x of xs) cut.push(x, x + rw);
  cut.push(x1);
  for (const y of ys) {
    for (let i = 0; i < cut.length; i += 2) {
      if (cut[i + 1] > cut[i]) body.push([cut[i], y, z0, cut[i + 1], y + rw, z1]);
    }
  }
  // 默认导出成「外观面贴床」的姿态（外观面落在 z = 0），切片里不用再旋转。
  // faceUp = true 时保持外观面朝上：外观面变成顶面，可以熨烫，但整片面板悬在
  // 墙围成的空腔上方，必须开支撑，支撑疤痕留在盖子内侧（看不见）。
  if (faceUp) return { body, pads };
  const flip = (bs) => bs.map(([x0, y0, z0, x1, y1, z1]) => [x0, y0, H - z1, x1, y1, H - z0]);
  return { body: flip(body), pads: flip(pads) };
}

function bigBlockBoxes() {
  const S = BIGN * PITCH - 1;          // 63
  const H = 2 * PITCH - 1;             // 31
  const ops = [];
  for (let k = 1; k < BIGN; k++) ops.push(P(k * PITCH - 1, 0, 1, S, 0.8));
  for (let k = 1; k < BIGN; k++) ops.push(P(0, k * PITCH - 1, S, 1, 0.8));
  return field({ x0: 0, y0: 0, x1: S, y1: S }, 0, H, ops);
}


module.exports = {
  R, P, field,
  TRAY, BIN, DIE, CELL, CORE, GROOVE, GW, LID, TILE, BIGN, PITCH,
  PCB, BLOCK_Z, PADS, LID_FIELD, cellXs, cellYs, coreXs, coreY,
  coreOps, trayOps, colorize, lidBoxes, bigBlockBoxes,
};
