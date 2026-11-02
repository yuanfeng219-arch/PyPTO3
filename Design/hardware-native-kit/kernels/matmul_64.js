/* ============================================================================
 * kernel manifest · matmul_64
 * ----------------------------------------------------------------------------
 * 来源：repo/pto/examples/kernels/03_matmul.py，L32–L41
 *
 * 这份 manifest 里没有一个行号，也没有一个像素坐标：
 *   定位靠 assigns / call / scope（赋值目标、调用、作用域）
 *   容量靠 shape × dtype × 所在 frame 的 residency
 *   布局靠 after（依赖）+ phase（语义分段）
 *
 * {{TILE}} 是参数代入位；原文件在这些位置是字面量 64。
 * ========================================================================== */
(function (root) {
  'use strict';

  root.HNKitKernels = root.HNKitKernels || {};
  root.HNKitKernels.matmul_64 = {
    id: 'matmul_64',
    entry: 'matmul_64',
    file: 'examples/kernels/03_matmul.py',
    platform: 'a2a3',
    note: 'L1·Mat 本例不建模容量；源码未写 dtype，这里按 BF16 入 / FP32 出估算。'
      + '原文件 tile 固定 64，TILE 旋钮是「换个尺寸会怎样」的推演。',

    consts: { TILE: 64 },
    knobs: [{ id: 'TILE', label: 'TILE（M=N=K）', options: [32, 64, 128, 256] }],
    derive: function (c) { return { M: c.TILE, N: c.TILE, K: c.TILE }; },

    source: [
      { no: 32, t: '@pl.jit' },
      { no: 33, t: 'def matmul_64(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):' },
      { no: 34, t: '    with pl.at(level=pl.Level.CORE_GROUP):' },
      { no: 35, t: '        tile_a_l1 = pl.load(a, [0, 0], [{{M}}, {{K}}], target_memory=pl.MemorySpace.Mat)' },
      { no: 36, t: '        tile_b_l1 = pl.load(b, [0, 0], [{{K}}, {{N}}], target_memory=pl.MemorySpace.Mat)' },
      { no: 37, t: '        tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)' },
      { no: 38, t: '        tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)' },
      { no: 39, t: '        tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)' },
      { no: 40, t: '        pl.store(tile_c_l0c, [0, 0], c)' },
      { no: 41, t: '    return c' }
    ],

    tensors: [
      { id: 'a', label: 'a · 左矩阵', note: '[{{M}}, {{K}}]' },
      { id: 'b', label: 'b · 右矩阵' },
      { id: 'c', label: 'c · 输出' }
    ],

    frames: [
      { id: 'f-cg', scope: 'pl.at', label: 'PL.AT · CORE_GROUP', residency: 1, regions: ['block'] }
    ],

    phases: [
      { id: 'ph-up', label: '① 搬运上片', nodes: ['load-a', 'load-b', 'mv-a', 'mv-b'] },
      { id: 'ph-mm', label: '② CUBE 计算', nodes: ['mm'] },
      { id: 'ph-st', label: '③ 写回', nodes: ['st'] }
    ],

    nodes: [
      { id: 'load-a', assigns: 'tile_a_l1', label: 'a → L1', frame: 'f-cg', regions: ['gm:a'] },
      { id: 'load-b', assigns: 'tile_b_l1', label: 'b → L1', frame: 'f-cg', regions: ['gm:b'] },
      { id: 'mv-a', assigns: 'tile_a_l0a', label: 'L1 → L0A', frame: 'f-cg', after: ['load-a'] },
      { id: 'mv-b', assigns: 'tile_b_l0b', label: 'L1 → L0B', frame: 'f-cg', after: ['load-b'] },
      {
        id: 'mm', assigns: 'tile_c_l0c', label: 'A · B', frame: 'f-cg',
        after: ['mv-a', 'mv-b'], regions: ['cube']
      },
      { id: 'st', call: 'pl.store', label: '写回 C', frame: 'f-cg', after: ['mm'], regions: ['gm:c'] }
    ],

    buffers: [
      { assigns: 'tile_a_l1', space: 'mat', shape: ['M', 'K'], dtype: 'bf16', frame: 'f-cg' },
      { assigns: 'tile_b_l1', space: 'mat', shape: ['K', 'N'], dtype: 'bf16', frame: 'f-cg' },
      { assigns: 'tile_a_l0a', space: 'left', shape: ['M', 'K'], dtype: 'bf16', frame: 'f-cg' },
      { assigns: 'tile_b_l0b', space: 'right', shape: ['K', 'N'], dtype: 'bf16', frame: 'f-cg' },
      { assigns: 'tile_c_l0c', space: 'acc', shape: ['M', 'N'], dtype: 'fp32', frame: 'f-cg' }
    ]
  };
})(typeof window !== 'undefined' ? window : this);
