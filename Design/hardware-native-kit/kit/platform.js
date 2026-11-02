/* ============================================================================
 * Hardware-Native Kit · 平台描述
 * ----------------------------------------------------------------------------
 * 一块芯片一份。kernel manifest 不写容量数字，也不写硅片坐标 —— 都从这里取。
 *
 * 两部分：
 *   spaces  片上空间的容量口径（名义容量、顶部保留、归谁）
 *   chip    硅片区域的盒子树；坐标由 kit/core.js 的 layoutBoxes() 自动算
 *
 * 容量数字随平台与 CANN 版本变化，需按目标配置核对；此处为示例口径。
 * ========================================================================== */
(function (root) {
  'use strict';

  const KB = 1024;

  // ── 片上空间 ──────────────────────────────────────────────────────────────
  // reserved > 0 的空间才会出现「分配器放行、运行时读写垃圾」那一档。
  // Vec 的 8 KB 取自 pto-isa#170（a2a3，ReserveBufferOp / pipe init）。
  const A2A3_SPACES = {
    mat: { label: 'L1 · Mat', nominal: 512 * KB, reserved: 0, modeled: false },
    left: { label: 'L0A · Left', nominal: 64 * KB, reserved: 0 },
    right: { label: 'L0B · Right', nominal: 64 * KB, reserved: 0 },
    acc: { label: 'L0C · Acc', nominal: 256 * KB, reserved: 0 },
    vec: { label: 'Vec · UB', nominal: 192 * KB, reserved: 8 * KB }
  };

  // ── 硅片区域拓扑 ──────────────────────────────────────────────────────────
  // dir 排布方向；w / h 固定尺寸，缺省则按 flex 分剩余空间。
  // kind: frame 虚框 · mem 存储块 · unit 计算单元 · chan 交接通道
  //       cores SPMD 核阵列 · bar 地址条 · slots 由 manifest 的 tensors 填充
  const A2A3_CHIP = {
    id: 'chip', dir: 'col', pad: 0, gap: 12,
    children: [
      {
        id: 'gm', label: 'GM · HBM', kind: 'frame', h: 84, dir: 'row', gap: 8,
        children: [{ id: 'gm-slots', kind: 'slots', space: 'gm', dir: 'row', gap: 8, pad: 0 }]
      },
      { id: 'cores', label: null, kind: 'cores', h: 32 },
      {
        id: 'block', label: 'BLOCK 0 · 其余 block 用同一份片上分配',
        kind: 'frame', dir: 'row', gap: 10,
        children: [
          {
            id: 'aic', label: 'AIC · CUBE', kind: 'frame', dir: 'col', gap: 8, flex: 5,
            children: [
              { id: 'mat', label: 'L1 · Mat', kind: 'mem', space: 'mat', h: 42 },
              {
                id: 'l0row', dir: 'row', pad: 0, gap: 8, h: 68,
                children: [
                  { id: 'l0a', label: 'L0A · Left', kind: 'mem', space: 'left' },
                  { id: 'l0b', label: 'L0B · Right', kind: 'mem', space: 'right' }
                ]
              },
              { id: 'cube', label: 'CUBE', kind: 'unit', h: 40 },
              { id: 'l0c', label: 'L0C · Acc', kind: 'mem', space: 'acc', h: 54 }
            ]
          },
          {
            id: 'chan', dir: 'col', pad: 0, gap: 10, w: 62,
            children: [
              { id: 'ch-c2v', label: 'C2V', kind: 'chan', h: 36 },
              { id: 'ch-v2c', label: 'V2C', kind: 'chan', h: 36 }
            ]
          },
          {
            id: 'aiv', label: 'AIV · VECTOR', kind: 'frame', dir: 'col', gap: 8, flex: 5,
            children: [{ id: 'ub', label: 'Vec · UB', kind: 'bar', space: 'vec' }]
          }
        ]
      }
    ]
  };

  const PLATFORMS = {
    a2a3: {
      id: 'a2a3',
      label: 'Ascend 950B · a2a3',
      note: 'Vec 保留区取自 pto-isa#170；其余容量沿用示例口径，需按目标核对。',
      cores: 24,
      cubeMFractal: 16,
      spaces: A2A3_SPACES,
      chip: A2A3_CHIP,
      // 空间 → 画布区域，用于「某个 buffer 落在哪个盒子里」
      spaceRegion: { mat: 'mat', left: 'l0a', right: 'l0b', acc: 'l0c', vec: 'ub' }
    }
  };

  root.HNKitPlatforms = PLATFORMS;
})(typeof window !== 'undefined' ? window : this);
