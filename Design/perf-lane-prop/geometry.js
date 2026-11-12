'use strict';
// 性能泳道板道具的几何定义（单位 mm，1:1 尺寸）。
// generate.js（STL / 3MF）与 plan.js（SVG 平面图）共用本文件。
//
// 造型全部是「自顶向下的棱柱」：一块矩形板，从顶面挖矩形槽、垫矩形台。
// 用高度场求解：把所有特征边界投影成网格，逐格算出顶面高度，每格输出一个长方体，
// 相邻同高度的格子沿 X 合并。输出的长方体互不重叠，整体是闭合实体。
//
// 数据口径来自 Design/perf-lane-kit/data/example.json 的 rank0：
//   72 核 = 24 AIC + 48 AIV（1:2）；窗口 4823.6 us；86 个 Task、4038 个 Block
//   AIC 的 Setup 占核上占用 2.2%，AIV 占 18.0%（单核最高 35.0%）
//   28.3% 的时间里同时只有 <=2 个 Block 在跑
// 道具不复刻 72 道，取 2 AIC + 4 AIV 保住 1:2 的比例，让 15 条 Task 排起来有难度。

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

/* ---------- 板 ---------- */

// 外廓与芯片托盘同为 420 x 300，0.6 比例下两件都是 252 x 180，摆在一起是一套。
// 板与 Task 条是两次打印，不用挤同一张盘，所以板可以按与托盘一致的尺寸做满。
// 代价：252 在 256 的床上每边只剩 2 mm，不能开 brim。板上唯一的分色面是 AIC 道底
// 那 0.6 的色皮，落在整齐高度上，靠「打到某层换料」就行，不需要 AMS 和擦料塔，
// 所以不开 brim 也只有翘边一个风险，用光面板加 65 度热床压住即可。
// H 给到 14（0.6 下 8.4 mm）：252 x 180 的板要被人反复拿起来，底板只有 3
// 是故意的 —— 槽底那 242 cm² 的顶面压在 9 层之上，不留稀疏填充，不会塌。
// 刚度全部交给道间那 5 宽、11 高的筋，和注塑件的薄壁加筋是一个路数。
const BOARD = { W: 420, D: 300, H: 14, floor: 3 };
const HEADER = { x: 8, w: 42, pocket: 1.2 };     // 左侧贴标签的浅坑
const LANE = { w: 32, depth: 11, pitch: 37 };    // 槽宽 32、槽深 11、道间距 37（道间留 5 的筋）
const TRACK = { x: 56, len: 350 };               // 泳道可用长度 = 35 格
const UNIT = 10;                                 // 一格时间 = 10 mm
const TICKS = TRACK.len / UNIT;                  // 34
const TICK = { w: 1.5, depth: 1.5, minor: 6, major: 12 };
const AXIS = { top: 12, bottom: 274, h: 14 };    // 上下两条时间轴带
const SKIN = 0.6;                                // AIC 道底的白色色皮

const AIC_N = 2;
const AIV_N = 4;
const LANE_Y0 = 36;                              // 第一条 AIC 道的上沿
const GROUP_GAP = 20;                            // AIC 组与 AIV 组之间多留的空

// 每条道的上沿 y。前 AIC_N 条是 AIC，其余是 AIV。
const laneYs = [];
for (let i = 0; i < AIC_N; i++) laneYs.push(LANE_Y0 + i * LANE.pitch);
const aivY0 = LANE_Y0 + AIC_N * LANE.pitch + GROUP_GAP - (LANE.pitch - LANE.w);
for (let i = 0; i < AIV_N; i++) laneYs.push(aivY0 + i * LANE.pitch);

const isAIC = (i) => i < AIC_N;

function boardOps() {
  const ops = [];
  for (let i = 0; i < laneYs.length; i++) {
    const y = laneYs[i];
    ops.push(P(TRACK.x, y, TRACK.len, LANE.w, LANE.depth));                 // 泳道槽
    ops.push(P(HEADER.x, y + (LANE.w - 20) / 2, HEADER.w, 20, HEADER.pocket)); // 标签坑
  }
  // 上下两条时间轴的刻度。每格一道短刻，每 5 格一道长刻。
  for (let k = 0; k <= TICKS; k++) {
    const x = TRACK.x + k * UNIT - TICK.w / 2;
    const len = k % 5 === 0 ? TICK.major : TICK.minor;
    ops.push(P(x, AXIS.top + AXIS.h - len, TICK.w, len, TICK.depth));
    ops.push(P(x, AXIS.bottom, TICK.w, len, TICK.depth));
  }
  return ops;
}

// 把板拆成颜色组：本体深灰、AIC 道底的白色色皮。
// 拆分面落在 floor - SKIN 这个整齐高度上，没有 AMS 时也能靠「打到某层换料」做出来。
function colorize(boxes) {
  const body = [], aic = [];
  const aicLanes = laneYs.slice(0, AIC_N);
  const inAIC = (cx, cy) => cx > TRACK.x && cx < TRACK.x + TRACK.len
    && aicLanes.some((y) => cy > y && cy < y + LANE.w);
  for (const [x0, y0, z0, x1, y1, z1] of boxes) {
    const onFloor = Math.abs(z1 - BOARD.floor) < 1e-9;
    if (onFloor && inAIC((x0 + x1) / 2, (y0 + y1) / 2)) {
      body.push([x0, y0, z0, x1, y1, z1 - SKIN]);
      aic.push([x0, y0, z1 - SKIN, x1, y1, z1]);
    } else body.push([x0, y0, z0, x1, y1, z1]);
  }
  return { body, aic };
}

function boardBoxes() {
  return field({ x0: 0, y0: 0, x1: BOARD.W, y1: BOARD.D }, 0, BOARD.H, boardOps());
}

/* ---------- Task 条 ---------- */

// 5 种长度各 3 条，共 15 条，合计 150 格。
// 长度是倒推出来的：6 条道刚好能排成每道 25 格的完美解 —— 18+7 三道、12+9+4 三道。
// 道长给到 35 格，最优解排完右边空出 10 格，占道长 28.6%，刻意贴近真实数据里
// 「28.3% 的时间同时只有 <=2 个 Block 在跑」那个数 —— 那段空白就是讲解要指的东西。
// 随手摆通常落在 27 到 30 格，谁也不会摆到放不下，不至于卡住参观者。
const BAR_LENS = [18, 12, 9, 7, 4].map((u) => u * UNIT);   // 180 120 90 70 40
const BAR_EACH = 3;
// 条高 20、槽深 11 → 露出 9（0.6 下 5.4 mm），捏得住。条上没有任何凸起。
const BAR = { h: 20 };
const FIT = 0.35;                           // 每边的配合间隙，固定 0.35 mm，不随比例缩放

// tile 就是芯片托盘那套 15 的小方块。直接 require 过来，两件的口径不可能走偏。
// 代价：这个文件依赖隔壁目录；挪动目录会立刻报错，而不是悄悄对不上。
const { TILE } = require('../chip-tray-prop/geometry');

// Task 条顶面的 tile 坑：一个 tile = 一块要算的数据。
// 坑心每 3 格（30 mm）一个，整组沿条居中，所以条越长坑越多：6 / 4 / 3 / 2 / 1，共 48 个。
// 为什么不是每 2 格一个：坑边长 16.17，每 2 格排就只剩 1.9 的端壁（0.6 下 1.15 mm），
// 参观者往里按 tile 时就是从那儿裂。每 3 格排，端壁 6.9、坑间筋 13.8，踏实得多。
// 坑深 6（0.6 下 3.6），tile 高 15（0.6 下 9），露出 9 —— 捏得出来，又不会一碰就倒。
const POCKET = { pitch: 3 * UNIT, depth: 6 };
const pocketCount = (len) => Math.floor(len / POCKET.pitch);

// 条宽比槽窄 2 x FIT，坑边长比 tile 大 2 x FIT。FIT 除以 scale，保证缩放后打出来恒为 0.35。
// 所以 tile 和 Task 条必须按同一比例打印。
function barBoxes(len, scale) {
  const fit = FIT / scale;
  const w = LANE.w - 2 * fit;
  const ps = TILE + 2 * fit;                      // 坑边长
  const n = pocketCount(len);
  const x0 = (len - n * POCKET.pitch) / 2;        // 整组居中后的起点
  const ops = [];
  for (let i = 0; i < n; i++) {
    const cx = x0 + (i + 0.5) * POCKET.pitch;
    ops.push(P(cx - ps / 2, (w - ps) / 2, ps, ps, POCKET.depth));
  }
  return { body: field({ x0: 0, y0: 0, x1: len, y1: w }, 0, BAR.h, ops), w, n, ps };
}

module.exports = {
  R, P, field,
  BOARD, HEADER, LANE, TRACK, UNIT, TICKS, TICK, AXIS, SKIN,
  AIC_N, AIV_N, LANE_Y0, GROUP_GAP, laneYs, isAIC,
  BAR_LENS, BAR_EACH, BAR, FIT, TILE, POCKET, pocketCount,
  boardOps, boardBoxes, colorize, barBoxes,
};
