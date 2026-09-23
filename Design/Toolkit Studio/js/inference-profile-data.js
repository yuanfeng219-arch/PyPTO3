/**
 * 推理性能分析 · 模拟采集数据
 *
 * 数据口径（全部为演示用模拟值，但内部自洽）：
 *   Qwen3-14B · hidden 5120 · 40 layers · Q40/KV8 · head_dim 128 · FFN 17408 · vocab 152064
 *   BF16 权重 · batch 16 · 平均 seq 1614 · page 128 token
 *   Ascend 950B · HBM 64 GB / 3.6 TB/s peak · Cube BF16 800 TFLOPS peak
 *
 * 关键恒等式（改数时请一起改）：
 *   每层 360.0 us x 40 = 14.400 ms + 边界 0.605 ms + Host 空隙 0.195 ms = TPOT 15.200 ms
 *   权重 27.99 GB + KV 4.23 GB + 激活 0.42 GB = 32.64 GB / step
 *   32.64 GB / 15.2 ms = 2.147 TB/s 达成 = 59.6% 峰值；理论下界 32.64 / 3.6 = 9.07 ms
 */
(function registerInferenceProfileData() {
  'use strict';

  const PEAK_BW = 3.6;       // TB/s
  const PEAK_FLOPS = 800;    // TFLOPS · BF16 Cube
  const TPOT = 15.2;         // ms · p50

  // 每请求 seq_len（16 路），合计 25,824 -> 平均 1,614
  const seqLens = [1893, 2048, 1367, 1544, 2048, 1211, 1832, 1207, 2048, 743, 1655, 2048, 1389, 1920, 866, 2005];

  /** 确定性伪随机，保证每次刷新图形一致 */
  function jitter(seed, n, base, spread) {
    const out = [];
    let s = seed;
    for (let i = 0; i < n; i += 1) {
      s = (s * 1103515245 + 12345) % 2147483648;
      out.push(base * (1 + ((s / 2147483648) - 0.5) * spread));
    }
    return out;
  }

  /**
   * 逐层曲线：首层 cache 冷、尾层要喂给输出边界，两端偏高。
   * 归一化到均值恰好等于 baseUs——时间线按逐层值铺 40 层，均值不准会让整步总时长偏离 TPOT。
   */
  function perLayer(seed, baseUs, spread, headBoost, tailBoost) {
    const values = jitter(seed, 40, baseUs, spread);
    values[0] *= headBoost;
    values[1] *= 1 + (headBoost - 1) * 0.35;
    values[39] *= tailBoost;
    const scale = (baseUs * 40) / values.reduce((a, v) => a + v, 0);
    const rounded = values.map((v) => Number((v * scale).toFixed(2)));
    // 2 位小数的舍入残差集中补到中间一层，保证 sum === baseUs * 40
    const residual = Number((baseUs * 40 - rounded.reduce((a, v) => a + v, 0)).toFixed(2));
    rounded[20] = Number((rounded[20] + residual).toFixed(2));
    return rounded;
  }

  const ops = [
    {
      id: 'gate-up-proj', name: 'gate_proj · up_proj', scope: 'Scope 3 · MLP', group: 'mlp',
      calls: 40, totalMs: 5.244, perLayerUs: 131.1, share: 34.5,
      units: { cube: 5.1, vector: 3.2, mte2: 89.4, mte3: 1.1, sync: 1.2 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 94,
      bytesIn: 14.264, bytesOut: 0.084, reuse: 1.0,
      gflop: 228.0, achievedTflops: 43.5, achievedBw: 2.72, ai: 16.0,
      cores: 64, imbalance: 0.06,
      perLayer: perLayer(7, 131.1, 0.05, 1.09, 1.02),
      source: 'decode_layer.py:612-688',
      static: [
        ['UB 峰值占用', '58%（编译期预算）', '63%', '+5 pt', 'warn'],
        ['Split-K tile 数', '5', '5', '一致', 'ok'],
        ['达成带宽', '2.66 TB/s（估）', '2.72 TB/s', '+2.3%', 'ok'],
      ],
      note: '权重搬运完全主导：89.4% 时间花在 MTE2。已贴近内存屋顶，除非改精度或做权重复用，否则无进一步空间。',
    },
    {
      id: 'down-proj', name: 'down_proj', scope: 'Scope 3 · MLP', group: 'mlp',
      calls: 40, totalMs: 2.640, perLayerUs: 66.0, share: 17.4,
      units: { cube: 5.0, vector: 2.8, mte2: 88.7, mte3: 2.1, sync: 1.4 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 93,
      bytesIn: 7.132, bytesOut: 0.131, reuse: 1.0,
      gflop: 114.0, achievedTflops: 43.2, achievedBw: 2.70, ai: 16.0,
      cores: 64, imbalance: 0.07,
      perLayer: perLayer(11, 66.0, 0.05, 1.08, 1.02),
      source: 'decode_layer.py:742-801',
      static: [
        ['UB 峰值占用', '54%（编译期预算）', '55%', '+1 pt', 'ok'],
        ['Split-K/N 拆分', '17 × 5', '17 × 5', '一致', 'ok'],
        ['达成带宽', '2.64 TB/s（估）', '2.70 TB/s', '+2.3%', 'ok'],
      ],
      note: '与 gate/up 同形态，FP32 atomic 累加的写回量略高，MTE3 占比 2.1%。',
    },
    {
      id: 'fa-fused', name: 'fa_fused', scope: 'Scope 2 · Attention', group: 'attn',
      calls: 65920, totalMs: 2.276, perLayerUs: 56.9, share: 15.0,
      units: { cube: 12.8, vector: 21.4, mte2: 58.1, mte3: 3.2, sync: 4.5 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 52,
      bytesIn: 4.240, bytesOut: 0.211, reuse: 3.2,
      gflop: 21.2, achievedTflops: 9.3, achievedBw: 1.86, ai: 5.0,
      cores: 64, imbalance: 0.18,
      perLayer: perLayer(23, 56.9, 0.09, 1.14, 1.05),
      source: 'decode_layer.py:301-398',
      static: [
        ['真实 KV block 数', '256（MCB 静态上界）', '206', '−19.5%', 'ok'],
        ['UB 峰值占用', '58%（编译期预算）', '63%', '+5 pt', 'warn'],
        ['单 work item 时延', '1.76 μs（估）', '2.21 μs', '+25.6%', 'warn'],
        ['达成带宽', '2.60 TB/s（估）', '1.86 TB/s', '−28.5%', 'bad'],
      ],
      note: 'paged K/V 的非连续访存使 MTE2 只跑到 51.7% 峰值，而同批投影算子在 70–76%。这是当前最大的可回收项。',
    },
    {
      id: 'qkv-proj', name: 'q_proj · k_proj · v_proj', scope: 'Scope 1 · 投影', group: 'proj',
      calls: 40, totalMs: 1.124, perLayerUs: 28.1, share: 7.4,
      units: { cube: 4.9, vector: 3.1, mte2: 87.2, mte3: 2.6, sync: 2.2 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 92,
      bytesIn: 2.936, bytesOut: 0.094, reuse: 1.0,
      gflop: 47.0, achievedTflops: 41.8, achievedBw: 2.61, ai: 16.0,
      cores: 64, imbalance: 0.09,
      perLayer: perLayer(31, 28.1, 0.06, 1.10, 1.01),
      source: 'decode_layer.py:429-505',
      static: [
        ['并行拆分', 'Q 10×5 · K/V 2×5', '一致', '一致', 'ok'],
        ['UB 峰值占用', '49%（编译期预算）', '51%', '+2 pt', 'ok'],
        ['达成带宽', '2.58 TB/s（估）', '2.61 TB/s', '+1.2%', 'ok'],
      ],
      note: '读取上一层 dcr_xgamma 预生成的 BF16 normed_in，省掉一次层内 cast。',
    },
    {
      id: 'out-proj', name: 'out_proj', scope: 'Scope 3 · 投影', group: 'proj',
      calls: 40, totalMs: 0.824, perLayerUs: 20.6, share: 5.4,
      units: { cube: 4.8, vector: 3.0, mte2: 86.5, mte3: 3.1, sync: 2.6 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 91,
      bytesIn: 2.096, bytesOut: 0.131, reuse: 1.0,
      gflop: 33.6, achievedTflops: 40.8, achievedBw: 2.54, ai: 16.0,
      cores: 64, imbalance: 0.10,
      perLayer: perLayer(41, 20.6, 0.06, 1.09, 1.01),
      source: 'decode_layer.py:518-577',
      static: [
        ['并行拆分', '10 × 5 split-N/K', '一致', '一致', 'ok'],
        ['UB 峰值占用', '46%（编译期预算）', '47%', '+1 pt', 'ok'],
        ['达成带宽', '2.55 TB/s（估）', '2.54 TB/s', '−0.4%', 'ok'],
      ],
      note: '规模最小的投影，固定开销占比略高，达成带宽因此比 gate/up 低约 7%。',
    },
    {
      id: 'rms-lm-head', name: 'rms_lm_head', scope: '输出边界', group: 'boundary',
      calls: 1, totalMs: 0.598, perLayerUs: null, share: 3.9,
      units: { cube: 6.2, vector: 8.4, mte2: 84.9, mte3: 0.3, sync: 0.2 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 91,
      bytesIn: 1.556, bytesOut: 0.005, reuse: 1.0,
      gflop: 24.9, achievedTflops: 41.6, achievedBw: 2.60, ai: 16.0,
      cores: 64, imbalance: 0.05,
      perLayer: null,
      source: 'decode_layer.py:1204-1266',
      static: [
        ['词表投影规模', '5120 × 152064', '一致', '一致', 'ok'],
        ['达成带宽', '2.58 TB/s（估）', '2.60 TB/s', '+0.8%', 'ok'],
      ],
      note: '整个 step 只执行一次，但 1.56 GB 的词表权重让它单独占了 3.9%。',
    },
    {
      id: 'silu', name: 'silu', scope: 'Scope 3 · MLP', group: 'mlp',
      calls: 40, totalMs: 0.352, perLayerUs: 8.8, share: 2.3,
      units: { cube: 0, vector: 62.3, mte2: 24.1, mte3: 11.4, sync: 2.2 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 78,
      bytesIn: 0.136, bytesOut: 0.045, reuse: 1.4,
      gflop: 0.13, achievedTflops: 0.37, achievedBw: 0.51, ai: 0.96,
      cores: 64, imbalance: 0.12,
      perLayer: perLayer(53, 8.8, 0.07, 1.06, 1.01),
      source: 'decode_layer.py:704-736',
      static: [
        ['延迟应用 post_inv_rms', '合并进 SwiGLU', '一致', '一致', 'ok'],
        ['UB 峰值占用', '38%（编译期预算）', '39%', '+1 pt', 'ok'],
      ],
      note: 'Vector-bound 的纯 elementwise 段，已把 post-RMS 缩放折叠进来，省掉一次全量读写。',
    },
    {
      id: 'online-softmax', name: 'online_softmax', scope: 'Scope 2 · Attention', group: 'attn',
      calls: 5120, totalMs: 0.344, perLayerUs: 8.6, share: 2.3,
      units: { cube: 0, vector: 71.5, mte2: 18.2, mte3: 6.1, sync: 4.2 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 64,
      bytesIn: 0.104, bytesOut: 0.021, reuse: 2.1,
      gflop: 0.50, achievedTflops: 1.45, achievedBw: 0.36, ai: 4.8,
      cores: 64, imbalance: 0.31,
      perLayer: perLayer(61, 8.6, 0.11, 1.12, 1.03),
      source: 'decode_layer.py:404-460',
      static: [
        ['work items', 'BATCH × NUM_KV_HEADS = 128', '128', '一致', 'ok'],
        ['负载不均衡度 CV', '0.10（假设均匀）', '0.31', '+0.21', 'bad'],
        ['UB 峰值占用', '31%（编译期预算）', '34%', '+3 pt', 'warn'],
      ],
      note: '每个 work item 要归并的块数与该请求 seq_len 成正比，ragged 输入直接变成 0.31 的负载倾斜。',
    },
    {
      id: 'residual-cast', name: 'residual_rms_cast', scope: 'Scope 3 · Norm', group: 'norm',
      calls: 40, totalMs: 0.272, perLayerUs: 6.8, share: 1.8,
      units: { cube: 0, vector: 51.4, mte2: 33.2, mte3: 12.1, sync: 3.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 72,
      bytesIn: 0.072, bytesOut: 0.031, reuse: 1.2,
      gflop: 0.05, achievedTflops: 0.18, achievedBw: 0.38, ai: 0.48,
      cores: 64, imbalance: 0.09,
      perLayer: perLayer(67, 6.8, 0.06, 1.05, 1.01),
      source: 'decode_layer.py:589-606',
      static: [['UB 峰值占用', '29%（编译期预算）', '30%', '+1 pt', 'ok']],
      note: '一次读取同时产出 FP32 残差与 BF16 的 MLP 输入，避免二次扫描。',
    },
    {
      id: 'dcr-xgamma', name: 'dcr_xgamma', scope: 'Scope 3 · Carry', group: 'norm',
      calls: 200, totalMs: 0.264, perLayerUs: 6.6, share: 1.7,
      units: { cube: 0, vector: 48.2, mte2: 39.4, mte3: 9.8, sync: 2.6 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 74,
      bytesIn: 0.061, bytesOut: 0.038, reuse: 1.1,
      gflop: 0.04, achievedTflops: 0.15, achievedBw: 0.37, ai: 0.40,
      cores: 64, imbalance: 0.04,
      perLayer: perLayer(71, 6.6, 0.05, 1.04, 1.16),
      source: 'decode_layer.py:812-869',
      static: [
        ['SPMD 路数', '5', '5', '一致', 'ok'],
        ['跨层输出', 'out FP32 + normed BF16', '一致', '一致', 'ok'],
      ],
      note: '层尾单次读取产出两份跨层输出，L39 偏高是因为要额外喂给输出边界。',
    },
    {
      id: 'rope-qkv', name: 'rope_qkv', scope: 'Scope 2 · Attention', group: 'attn',
      calls: 40, totalMs: 0.256, perLayerUs: 6.4, share: 1.7,
      units: { cube: 0, vector: 54.8, mte2: 31.0, mte3: 11.5, sync: 2.7 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 69,
      bytesIn: 0.124, bytesOut: 0.106, reuse: 1.0,
      gflop: 0.05, achievedTflops: 0.20, achievedBw: 0.90, ai: 0.40,
      cores: 64, imbalance: 0.14,
      perLayer: perLayer(79, 6.4, 0.07, 1.06, 1.01),
      source: 'decode_layer.py:212-288',
      static: [
        ['Q pad', '5 real → 16 tile rows', '一致', '一致', 'ok'],
        ['paged 写入对齐', 'page 边界对齐', '未对齐 2 处', '偏差', 'warn'],
      ],
      note: 'MTE3 占比 11.5%，是全表最高：当前 token 的 K/V 要散写进 paged cache。',
    },
    {
      id: 'qk-norm', name: 'qk_norm', scope: 'Scope 1 · Norm', group: 'norm',
      calls: 320, totalMs: 0.232, perLayerUs: 5.8, share: 1.5,
      units: { cube: 0, vector: 68.4, mte2: 22.7, mte3: 6.2, sync: 2.7 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 70,
      bytesIn: 0.048, bytesOut: 0.024, reuse: 1.3,
      gflop: 0.03, achievedTflops: 0.13, achievedBw: 0.31, ai: 0.42,
      cores: 64, imbalance: 0.08,
      perLayer: perLayer(83, 5.8, 0.06, 1.05, 1.01),
      source: 'decode_layer.py:508-556',
      static: [['task 数', '8（每 KV head 一路）', '8', '一致', 'ok']],
      note: 'gamma 与 reciprocal 合并在同一趟里做，避免二次读 q_proj / k_proj。',
    },
    {
      id: 'rms-recip', name: 'rms_recip', scope: 'Scope 1 · Norm', group: 'norm',
      calls: 40, totalMs: 0.168, perLayerUs: 4.2, share: 1.1,
      units: { cube: 0, vector: 74.2, mte2: 21.1, mte3: 1.4, sync: 3.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 66,
      bytesIn: 0.026, bytesOut: 0.001, reuse: 1.0,
      gflop: 0.01, achievedTflops: 0.06, achievedBw: 0.16, ai: 0.38,
      cores: 64, imbalance: 0.05,
      perLayer: perLayer(89, 4.2, 0.06, 1.05, 1.01),
      source: 'decode_layer.py:118-164',
      static: [['pipeline stage', '4', '4', '一致', 'ok']],
      note: '只算倒数标量，与 QKV 投影重叠执行，实际暴露在关键路径上的不足 1 μs。',
    },
    {
      id: 'post-rms-reduce', name: 'post_rms_reduce', scope: 'Scope 3 · Norm', group: 'norm',
      calls: 40, totalMs: 0.136, perLayerUs: 3.4, share: 0.9,
      units: { cube: 0, vector: 72.8, mte2: 22.4, mte3: 1.5, sync: 3.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 67,
      bytesIn: 0.021, bytesOut: 0.001, reuse: 1.0,
      gflop: 0.01, achievedTflops: 0.05, achievedBw: 0.15, ai: 0.36,
      cores: 64, imbalance: 0.05,
      perLayer: perLayer(97, 3.4, 0.06, 1.04, 1.01),
      source: 'decode_layer.py:171-204',
      static: [['与 residual_cast 并行', '是', '是', '一致', 'ok']],
      note: '与 residual_rms_cast 并行，reciprocal 延迟到 silu 里才应用。',
    },
    {
      id: 'fa-work-build', name: 'fa_work_build', scope: 'Scope 2 · Attention', group: 'attn',
      calls: 40, totalMs: 0.088, perLayerUs: 2.2, share: 0.6,
      units: { cube: 0, vector: 58.1, mte2: 12.4, mte3: 24.2, sync: 5.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 61,
      bytesIn: 0.001, bytesOut: 0.002, reuse: 1.0,
      gflop: 0.00, achievedTflops: 0.01, achievedBw: 0.03, ai: 0.20,
      cores: 64, imbalance: 0.03,
      perLayer: perLayer(101, 2.2, 0.05, 1.03, 1.01),
      source: 'decode_layer.py:88-112',
      static: [
        ['work table 稠密率', '—（静态无法知道）', '80.5%', '206 / 256', 'ok'],
        ['去除的空块', '—', '50', '省 19.5% 迭代', 'ok'],
      ],
      note: '花 2.2 μs 把 ragged 请求压紧成无空洞工作表，为 fa_fused 省掉 19.5% 的空块迭代。',
    },
    {
      id: 'copy-hidden', name: 'copy_hidden', scope: '输入边界', group: 'boundary',
      calls: 1, totalMs: 0.003, perLayerUs: null, share: 0.02,
      units: { cube: 0, vector: 12.4, mte2: 61.2, mte3: 24.1, sync: 2.3 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 84,
      bytesIn: 0.00016, bytesOut: 0.00033, reuse: 1.0,
      gflop: 0.00, achievedTflops: 0.00, achievedBw: 0.16, ai: 0.00,
      cores: 64, imbalance: 0.02,
      perLayer: null,
      source: 'decode_layer.py:1155',
      static: [['精度转换点', '1（入口唯一）', '1', '一致', 'ok']],
      note: '整个 step 唯一的入口精度边界，BF16 → FP32 只做一次。',
    },
    {
      id: 'cast-lmhead', name: 'cast_lmhead_in', scope: '输出边界', group: 'boundary',
      calls: 1, totalMs: 0.002, perLayerUs: null, share: 0.01,
      units: { cube: 0, vector: 14.1, mte2: 59.8, mte3: 24.0, sync: 2.1 },
      bound: 'mte2', boundLabel: 'MTE2', efficiency: 83,
      bytesIn: 0.00033, bytesOut: 0.00016, reuse: 1.0,
      gflop: 0.00, achievedTflops: 0.00, achievedBw: 0.25, ai: 0.00,
      cores: 64, imbalance: 0.02,
      perLayer: null,
      source: 'decode_layer.py:1189',
      static: [['精度转换点', '1（出口唯一）', '1', '一致', 'ok']],
      note: '40 层循环之后唯一的 FP32 → BF16 转换。',
    },
    {
      id: 'x-gamma0', name: 'x_gamma0', scope: '输入边界', group: 'boundary',
      calls: 1, totalMs: 0.002, perLayerUs: null, share: 0.01,
      units: { cube: 0, vector: 56.2, mte2: 30.1, mte3: 11.4, sync: 2.3 },
      bound: 'vector', boundLabel: 'Vector', efficiency: 70,
      bytesIn: 0.00033, bytesOut: 0.00016, reuse: 1.0,
      gflop: 0.00, achievedTflops: 0.00, achievedBw: 0.25, ai: 0.00,
      cores: 64, imbalance: 0.02,
      perLayer: null,
      source: 'decode_layer.py:1168',
      static: [['仅 layer 0', '是', '是', '一致', 'ok']],
      note: '只为 layer 0 补一份预缩放输入，其余各层由上一层的 dcr_xgamma 直接产出。',
    },
    {
      id: '__idle__', name: '同步与 Host 空隙', scope: '未归属', group: 'idle',
      calls: null, totalMs: 0.375, perLayerUs: 4.5, share: 2.5,
      units: { cube: 0, vector: 0, mte2: 0, mte3: 0, sync: 100 },
      bound: 'idle', boundLabel: 'Idle', efficiency: null,
      bytesIn: 0, bytesOut: 0, reuse: null,
      gflop: 0, achievedTflops: 0, achievedBw: 0, ai: null,
      cores: null, imbalance: null,
      perLayer: null,
      source: '—',
      static: [],
      note: '层内 barrier 0.180 ms（每层 4.5 μs，Scope 间 3 处）+ Host dispatch 0.195 ms。时间线页签按阈值实时检出暴露空隙。',
    },
  ];

  const groups = [
    { id: 'mlp', label: 'MLP', detail: 'gate/up · silu · down', ms: 8.236, share: 54.2 },
    { id: 'attn', label: 'Attention', detail: 'work_build · rope · fa_fused · softmax', ms: 2.964, share: 19.5 },
    { id: 'proj', label: 'QKV / Out 投影', detail: 'q/k/v_proj · out_proj', ms: 1.948, share: 12.8 },
    { id: 'norm', label: 'Norm / Carry', detail: 'rms · qk_norm · residual · dcr_xgamma', ms: 1.072, share: 7.1 },
    { id: 'boundary', label: 'LM Head 与边界', detail: 'copy · cast · rms_lm_head', ms: 0.605, share: 4.0 },
    { id: 'idle', label: '同步与空隙', detail: 'barrier · host dispatch', ms: 0.375, share: 2.5 },
  ];

  // ITL 直方图：512 steps，14.4 -> 18.8 ms，右偏长尾
  const itlBins = [
    [14.4, 6], [14.8, 41], [15.2, 168], [15.6, 131], [16.0, 74],
    [16.4, 41], [16.8, 22], [17.2, 13], [17.6, 8], [18.0, 5], [18.4, 2], [18.8, 1],
  ];

  /* ---------------- 访存与缓存（P4） ---------------- */

  const KV_BYTES_PER_TOKEN_LAYER = 8 * 128 * 2 * 2;                 // kv_heads × dim × (K,V) × BF16
  const KV_BYTES_PER_TOKEN = KV_BYTES_PER_TOKEN_LAYER * 40;         // 160 KiB
  const PAGE_TOKENS = 128;
  const PAGE_BYTES = KV_BYTES_PER_TOKEN * PAGE_TOKENS;              // 20.97 MB
  const PAGES_TOTAL = 320;

  const kvPerRequest = seqLens.map((seq, i) => ({
    req: `req-${String(i).padStart(2, '0')}`,
    seq,
    pages: Math.ceil(seq / PAGE_TOKENS),
  }));
  const pagesUsed = kvPerRequest.reduce((a, r) => a + r.pages, 0);   // 206
  const tokensLive = seqLens.reduce((a, s) => a + s, 0);             // 25,824
  const tokensAllocated = pagesUsed * PAGE_TOKENS;                   // 26,368
  const MCB = Math.ceil(Math.max(...seqLens) / PAGE_TOKENS);         // 16
  const blocksPadded = MCB * seqLens.length;                         // 256

  const memory = {
    hbm: {
      capacity: 64,
      items: [
        ['weights', '权重 · BF16', 27.99, '40 层 26.43 GB + LM Head 1.56 GB'],
        ['kv', 'KV Cache 页池', PAGES_TOTAL * PAGE_BYTES / 1e9, `${PAGES_TOTAL} 页 × ${(PAGE_BYTES / 1e6).toFixed(2)} MB`],
        ['workspace', 'Workspace / 累加器', 0.28, 'FP32 atomic 累加 + fa 分块中间量'],
        ['act', '激活与跨层 carry', 0.02, 'cur FP32 + normed BF16 + 层内 tile'],
      ],
    },
    onchip: [
      ['L0A / L0B', 71, null, 'matmul 操作数 tile · Cube 直读'],
      ['L0C', 52, null, 'FP32 累加器'],
      ['L1', 64, null, '权重 tile 预取缓冲'],
      ['UB', 63, 58, '峰值来自 gate_up_proj，超编译期预算 5 pt'],
    ],
    kv: {
      pageTokens: PAGE_TOKENS,
      pageBytesMb: PAGE_BYTES / 1e6,
      pagesTotal: PAGES_TOTAL,
      pagesUsed,
      tokensLive,
      tokensAllocated,
      bytesAllocated: pagesUsed * PAGE_BYTES / 1e9,
      bytesPool: PAGES_TOTAL * PAGE_BYTES / 1e9,
      fragmentation: (tokensAllocated - tokensLive) / tokensAllocated * 100,
      utilization: pagesUsed / PAGES_TOTAL * 100,
      hitRate: 98.2,
      preempt: 0,
      swap: 0,
      mcb: MCB,
      blocksPadded,
      blocksReal: pagesUsed,
      density: pagesUsed / blocksPadded * 100,
      perRequest: kvPerRequest,
    },
    precision: [
      ['入口 BF16 → FP32', 'copy_hidden', 1, 'ok'],
      ['层内转换', '—', 0, 'ok'],
      ['出口 FP32 → BF16', 'cast_lmhead_in', 1, 'ok'],
      ['跨层 carry', 'out FP32 0.33 MB/层 · normed BF16 0.16 MB/层', null, 'ok'],
    ],
  };

  /* ---------------- 批处理与调度（P5） ---------------- */

  // batch 扫描：权重流量恒定，KV 与激活随 batch 线性增长，达成带宽随 batch 略升
  const SWEEP_BW = { 1: 2.05, 4: 2.10, 8: 2.12, 16: 2.147, 32: 2.18, 64: 2.21 };
  const SWEEP_MTE2 = { 1: 78, 4: 76, 8: 73, 16: 70.4, 32: 66, 64: 61 };
  const sweep = [1, 4, 8, 16, 32, 64].map((batch) => {
    const kv = 4.231 / 16 * batch;
    const act = 0.42 / 16 * batch;
    const traffic = 27.99 + kv + act;
    const bw = SWEEP_BW[batch];
    // 用展示精度的 tpot 反算 tps，避免与 summary.tps 差 1
    const tpot = Number((traffic / bw).toFixed(2));
    return {
      batch,
      traffic: Number(traffic.toFixed(2)),
      bw,
      tpot,
      tps: Math.round(batch / (tpot / 1000)),
      mte2: SWEEP_MTE2[batch],
      perToken: Number((traffic / batch).toFixed(2)),
      current: batch === 16,
    };
  });

  const batchOverTime = jitter(1337, 64, 14.7, 0.26).map((v) => Math.max(12, Math.min(16, Math.round(v))));
  const batchAvg = batchOverTime.reduce((a, v) => a + v, 0) / batchOverTime.length;

  // 连续批处理：16 个槽位在窗口内被复用，共 26 个请求
  const WINDOW_MS = 512 * 15.2;
  const requestLanes = (() => {
    const lanes = [];
    let s = 20260803;
    const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
    let seq = 0;
    for (let slot = 0; slot < 16; slot += 1) {
      const items = [];
      let t = slot < 12 ? 0 : rnd() * 400;
      while (t < WINDOW_MS) {
        const wait = slot < 12 && t === 0 ? 0 : 6 + rnd() * 34;
        const prefill = 120 + rnd() * 140;
        const decode = 1400 + rnd() * 4200;
        const end = Math.min(t + wait + prefill + decode, WINDOW_MS);
        items.push({
          id: `r-${String(seq).padStart(3, '0')}`,
          t0: t, wait, prefill,
          decode: Math.max(end - t - wait - prefill, 0),
          done: end < WINDOW_MS,
        });
        seq += 1;
        t = end + 2 + rnd() * 10;
      }
      lanes.push({ slot, items });
    }
    return lanes;
  })();

  const serving = {
    windowMs: WINDOW_MS,
    steps: 512,
    queue: { running: 16, waiting: 3, waitP50: 22, waitP99: 61, preempt: 0, recompute: 0, chunkedPrefill: 7 },
    split: { prefill: 14.2, decode: 85.8 },
    batchOverTime,
    batchAvg,
    sweep,
    lanes: requestLanes,
    totalRequests: requestLanes.reduce((a, l) => a + l.items.length, 0),
  };

  const profiles = {
    'run-0803-a': {
      id: 'run-0803-a',
      title: 'Decode Fused · 全链路 Profiling',
      token: 'ptok://qwen3-14b/decode-fused@run-0803-a',
      meta: {
        model: 'Qwen3-14B', params: '14.8 B', dtype: 'BF16',
        batch: 16, layers: 40, seqAvg: 1614, page: 128,
        device: 'Ascend 950B', hbm: 64, peakBw: PEAK_BW, peakFlops: PEAK_FLOPS,
        env: 'env:8da1bf09', envMatch: true,
        steps: 512, capturedAt: '2026-08-03 14:32:08', duration: '7.8 s',
        collector: 'PyPTO DFX · msprof v9.0',
      },
      summary: {
        tpot: { p50: 15.2, p90: 16.4, p99: 18.1 }, tpotDelta: 3.4,
        tps: 1053, tpsDelta: -3.3,
        ttft: 184, ttftDelta: 1.2,
        kvUsed: memory.kv.bytesAllocated, kvPool: memory.kv.bytesPool, kvPct: memory.kv.utilization,
        preempt: 0, batchAvg,
        sol: [
          { id: 'cube', label: 'Cube (AIC)', pct: 3.7, detail: '29.4 / 800 TFLOPS' },
          { id: 'vector', label: 'Vector (AIV)', pct: 17.9, detail: 'elementwise + 归约' },
          { id: 'mte2', label: 'MTE2 · HBM → 片上', pct: 70.4, detail: '32.64 GB / step · 2.15 TB/s' },
          { id: 'mte3', label: 'MTE3 · 片上 → HBM', pct: 5.1, detail: '0.79 GB / step' },
        ],
        bound: 'memory',
        lowerBoundMs: 9.07, efficiency: 59.7,
        traffic: { weights: 27.99, kv: 4.23, act: 0.42, total: 32.64 },
        flops: { total: 447.5, achieved: 29.4, ai: 13.7, ridge: 222.2 },
      },
      groups,
      ops,
      itlBins,
      seqLens,
      memory,
      serving,
      baseline: { id: 'b8160fd', token: 'ptok://qwen3-14b/decode-fused@b8160fd', tpot: 14.7, tps: 1088, label: '可信基线 · 08/01' },
    },
  };

  /* ================= 多机多卡：2 节点 × 8 卡 · TP=8 · PP=2 ================= */
  /*
   * 全部由单卡 profile 推导：
   *   TP=8 把每层工作切成 8 份 → 每 rank 每层 = 355.5 / 8 = 44.44 μs
   *   PP=2 把 40 层切成两段 → 每 rank 承担 20 层，batch 32 拆成 2 个 microbatch × 16
   *   每 rank 层内 device time = 355.5/8 × 20 × 2 = 单卡 14.22 ms / 8 = 1.7775 ms
   *
   * 关键恒等式：
   *   1.7775(层) + 0.150(边界) + 0.100(barrier) + 0.3776(AllReduce) + 0.0371(P2P) + 1.1092(气泡) = 3.5514 ms
   */
  const DIST = (() => {
    const TP = 8; const PP = 2; const NODES = 2; const CARDS = 8;
    const MB_COUNT = 2; const MB_BATCH = 16; const BATCH = MB_COUNT * MB_BATCH;
    const STAGE_LAYERS = 40 / PP;
    const HCCS = 400;  // GB/s · 节点内
    const ROCE = 25;   // GB/s · 跨节点

    const perLayerUs = 355.5 / TP;
    const compUs = perLayerUs * STAGE_LAYERS;
    const arPayload = MB_BATCH * 5120 * 2;                       // [16,5120] BF16
    const arBus = 2 * (TP - 1) / TP * arPayload;                 // ring AllReduce 每卡总线流量
    const arUs = arBus / (HCCS * 1e9) * 1e6 + 4;                 // + 固定延迟
    const arPerMb = arUs * 2 * STAGE_LAYERS;                     // 每层 attn / mlp 各一次
    const p2pUs = arPayload / (ROCE * 1e9) * 1e6 + 12;
    const barPerMb = 2.5 * STAGE_LAYERS;
    const edgeIn = 5 / TP;
    const edgeOut = 600 / TP;
    const s0Mb = compUs + arPerMb + barPerMb + edgeIn;
    const s1Mb = compUs + arPerMb + barPerMb + edgeOut;

    // 1F1B 流水：stage1 必须等 stage0 的 microbatch 传过来
    const schedule = [];
    let t0 = 0; let t1 = 0;
    for (let i = 0; i < MB_COUNT; i += 1) {
      schedule.push({ stage: 0, mb: i, t0, dur: s0Mb, kind: 'compute' });
      const arrive = t0 + s0Mb + p2pUs;
      schedule.push({ stage: 0, mb: i, t0: t0 + s0Mb, dur: p2pUs, kind: 'p2p' });
      const start = Math.max(t1, arrive);
      if (start > t1) schedule.push({ stage: 1, mb: i, t0: t1, dur: start - t1, kind: 'bubble' });
      schedule.push({ stage: 1, mb: i, t0: start, dur: s1Mb, kind: 'compute' });
      t1 = start + s1Mb;
      t0 += s0Mb;
    }
    const tpotUs = t1;
    if (t0 < tpotUs) schedule.push({ stage: 0, mb: null, t0, dur: tpotUs - t0, kind: 'bubble' });

    const cardTotal = (NODES * CARDS) * tpotUs;
    const busy0 = CARDS * MB_COUNT * s0Mb;
    const busy1 = CARDS * MB_COUNT * s1Mb;
    const arTotal = (NODES * CARDS) * MB_COUNT * arPerMb;
    const p2pTotal = CARDS * MB_COUNT * p2pUs;

    // 每 rank 负载：TP 组内本应均衡，ragged seq 与链路差异造成小幅倾斜；rank 11 为 straggler
    let seed = 90210;
    const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const ranks = [];
    for (let r = 0; r < NODES * CARDS; r += 1) {
      const stage = Math.floor(r / CARDS);
      const base = stage === 0 ? s0Mb : s1Mb;
      const straggler = r === 11;
      const scale = 1 + (rnd() - 0.5) * 0.05 + (straggler ? 0.082 : 0);
      const busy = base * MB_COUNT * scale;
      const comm = arPerMb * MB_COUNT * scale;
      ranks.push({
        rank: r, node: Math.floor(r / CARDS), stage, straggler,
        busyUs: busy, computeUs: busy - comm, commUs: comm,
        idleUs: Math.max(tpotUs - busy, 0),
        utilization: busy / tpotUs * 100,
      });
    }

    const weightsStage0 = STAGE_LAYERS * 660.7 / TP / 1000;
    const weightsStage1 = weightsStage0 + 1.556 / TP;
    const kvPerTokenCard = 1 * 128 * 2 * 2 * STAGE_LAYERS;       // 1 kv head × 20 层
    const kvPagesPerReq = Math.ceil(1614 / 128);
    const kvPages = BATCH * kvPagesPerReq;
    const kvPageMb = 128 * kvPerTokenCard / 1e6;
    const kvPool = 640 * kvPageMb / 1000;
    const trafficPerRank = weightsStage1 * MB_COUNT + BATCH * 1614 * kvPerTokenCard / 1e9 + 0.025;
    const computeMs = (1.7775 + 0.150);

    // 扩展性对照：同为 batch 32
    const tp8Only = (() => {
      const w = 27.99 / TP;
      const kv = BATCH * 1614 * (1 * 128 * 2 * 2 * 40) / 1e9;
      const ar = (2 * (BATCH * 5120 * 2) * (TP - 1) / TP / (HCCS * 1e9) * 1e6 + 4) * 2 * 40;
      return ((w + kv + 0.025) / 2.55 * 1000 + ar + 2.5 * 40) / 1000;
    })();
    const singleCard = 17.11;
    const mk = (label, cards, tpot, note, current) => ({
      label, cards, tpot: Number(tpot.toFixed(3)),
      tps: Math.round(BATCH / (tpot / 1000)),
      speedup: Number((singleCard / tpot).toFixed(2)),
      efficiency: Number((singleCard / tpot / cards * 100).toFixed(1)),
      note, current: !!current,
    });

    return {
      topology: { nodes: NODES, cardsPerNode: CARDS, world: NODES * CARDS, tp: TP, pp: PP, dp: 1, batch: BATCH, microbatches: MB_COUNT, mbBatch: MB_BATCH, stageLayers: STAGE_LAYERS },
      tpotMs: tpotUs / 1000,
      tps: Math.round(BATCH / (tpotUs / 1e6)),
      stages: [
        { id: 0, node: 0, layers: '0 – 19', perMbUs: s0Mb, utilization: MB_COUNT * s0Mb / tpotUs * 100, weights: weightsStage0, role: '输入边界 + 前 20 层' },
        { id: 1, node: 1, layers: '20 – 39', perMbUs: s1Mb, utilization: MB_COUNT * s1Mb / tpotUs * 100, weights: weightsStage1, role: '后 20 层 + LM Head' },
      ],
      schedule,
      cardTime: {
        totalMs: cardTotal / 1000,
        computePct: (busy0 + busy1 - arTotal) / cardTotal * 100,
        commPct: arTotal / cardTotal * 100,
        p2pPct: p2pTotal / cardTotal * 100,
        bubblePct: (cardTotal - busy0 - busy1) / cardTotal * 100,
      },
      collectives: [
        {
          op: 'AllReduce', algo: 'Ring', scope: '节点内 TP 组 · 8 rank', link: 'HCCS',
          calls: 2 * STAGE_LAYERS * MB_COUNT, payloadKb: arPayload / 1024, busKb: arBus / 1024,
          usPerCall: arUs, totalMs: arPerMb * MB_COUNT / 1000,
          achievedGbs: arBus / (arUs * 1e-6) / 1e9, peakGbs: HCCS,
          trigger: '每层 attention out_proj 与 MLP down_proj 之后各一次',
        },
        {
          op: 'Send / Recv', algo: 'P2P', scope: '跨节点 PP 边界', link: 'RoCE',
          calls: MB_COUNT, payloadKb: arPayload / 1024, busKb: arPayload / 1024,
          usPerCall: p2pUs, totalMs: p2pUs * MB_COUNT / 1000,
          achievedGbs: arPayload / (p2pUs * 1e-6) / 1e9, peakGbs: ROCE,
          trigger: 'stage 0 的 hidden_states 传给 stage 1',
        },
      ],
      links: [
        { id: 'hccs', label: '节点内 HCCS', peak: HCCS, achieved: arBus / (arUs * 1e-6) / 1e9, scope: `${CARDS} 卡全互联`, carries: 'TP AllReduce' },
        { id: 'roce', label: '跨节点 RoCE', peak: ROCE, achieved: arPayload / (p2pUs * 1e-6) / 1e9, scope: `${NODES} 节点`, carries: 'PP P2P' },
      ],
      ranks,
      memory: {
        weightsStage0, weightsStage1,
        kvPool, kvUsed: kvPages * kvPageMb / 1000, kvPages, kvPagesTotal: 640, kvPageMb,
        workspace: 0.09, act: 0.01,
        perCard: weightsStage1 + kvPool + 0.09 + 0.01,
        capacity: 64,
      },
      traffic: { perRank: trafficPerRank, achievedBw: trafficPerRank / computeMs, weightReReads: MB_COUNT },
      scaling: [
        mk('单卡', 1, singleCard, '权重 27.99 GB 全在一张卡上'),
        mk('TP=8 · 单节点', TP, tp8Only, '权重 3.50 GB/卡，AllReduce 全走节点内 HCCS'),
        mk('TP=8 + PP=2 · 双节点', NODES * CARDS, tpotUs / 1000, '权重 1.85 GB/卡，但引入流水线气泡', true),
      ],
      perRankOps: {
        layerMs: 355.5 / TP * STAGE_LAYERS * MB_COUNT / 1000,
        edgeMs: edgeOut * MB_COUNT / 1000,
        barrierMs: barPerMb * MB_COUNT / 1000,
        allreduceMs: arPerMb * MB_COUNT / 1000,
        p2pMs: p2pUs * MB_COUNT / 1000,
        bubbleMs: tpotUs / 1000 - (355.5 / TP * STAGE_LAYERS * MB_COUNT + edgeOut * MB_COUNT + barPerMb * MB_COUNT + arPerMb * MB_COUNT + p2pUs * MB_COUNT) / 1000,
      },
    };
  })();

  /* 由单卡 ops 推导每 rank 的算子表：TP 切 8 份，PP 只承担一半层但跑 2 个 microbatch */
  function distributedOps(base) {
    const TP = DIST.topology.tp;
    const stage1Edge = ['cast-lmhead', 'rms-lm-head'];
    const stage0Edge = ['copy-hidden', 'x-gamma0'];
    const out = base.filter((o) => o.group !== 'idle' && !stage0Edge.includes(o.id)).map((o) => {
      const boundary = stage1Edge.includes(o.id);
      const div = boundary ? TP / DIST.topology.microbatches : TP;  // 边界算子每 microbatch 各跑一次
      const totalMs = o.totalMs / div;
      return {
        ...o,
        totalMs: Number(totalMs.toFixed(4)),
        share: 0,
        perLayerUs: o.perLayerUs === null ? null : Number((o.perLayerUs / TP).toFixed(3)),
        perLayer: o.perLayer ? o.perLayer.slice(0, DIST.topology.stageLayers).map((v) => Number((v / TP).toFixed(3))) : null,
        bytesIn: o.bytesIn / div,
        bytesOut: o.bytesOut / div,
        gflop: Number((o.gflop / div).toFixed(3)),
        achievedTflops: Number((o.gflop / div / (totalMs || 1)).toFixed(2)),
        scope: `${o.scope} · TP 1/${TP}`,
        note: `${o.note}\nTP=${TP} 后每 rank 只做 1/${TP} 的工作，权重分片 ${(o.bytesIn / div * 1000).toFixed(0)} MB。`,
      };
    });

    const mk = (id, name, scope, group, calls, totalMs, bound, note, units) => ({
      id, name, scope, group, calls, totalMs: Number(totalMs.toFixed(4)), share: 0,
      perLayerUs: null, perLayer: null, units, bound, boundLabel: bound === 'comm' ? 'Comm' : 'Idle',
      efficiency: null, bytesIn: 0, bytesOut: 0, reuse: null,
      gflop: 0, achievedTflops: 0, achievedBw: 0, ai: null,
      cores: null, imbalance: null, source: '—', static: [], note,
    });
    const commUnits = { cube: 0, vector: 0, mte2: 0, mte3: 0, sync: 100 };
    out.push(mk('tp-allreduce', 'TP AllReduce', '节点内集合通信', 'comm',
      DIST.collectives[0].calls, DIST.perRankOps.allreduceMs, 'comm', commUnits
      && `每层 attention 与 MLP 之后各一次 Ring AllReduce，${DIST.collectives[0].calls} 次/step，单次 ${DIST.collectives[0].usPerCall.toFixed(2)} μs。走节点内 HCCS，未与计算重叠。`, commUnits));
    out.push(mk('pp-p2p', 'PP Send / Recv', '跨节点点对点', 'comm',
      DIST.collectives[1].calls, DIST.perRankOps.p2pMs, 'comm',
      `stage 边界传 hidden_states，${DIST.collectives[1].calls} 次/step，单次 ${DIST.collectives[1].usPerCall.toFixed(2)} μs。载荷小，跨节点带宽不是瓶颈。`, commUnits));
    out.push(mk('__barrier__', '层内 barrier', '未归属', 'idle',
      null, DIST.perRankOps.barrierMs, 'idle', '每层 2.5 μs 同步开销。', commUnits));
    out.push(mk('__bubble__', '流水线气泡', '未归属', 'idle',
      null, DIST.perRankOps.bubbleMs, 'idle',
      `PP=${DIST.topology.pp} 只有 ${DIST.topology.microbatches} 个 microbatch，气泡比 = (p−1)/(m+p−1) = ${((DIST.topology.pp - 1) / (DIST.topology.microbatches + DIST.topology.pp - 1) * 100).toFixed(0)}%。stage 1 开头空等 stage 0 的首个 microbatch，stage 0 结尾空等。`, commUnits));

    const total = out.reduce((a, o) => a + o.totalMs, 0);
    out.forEach((o) => { o.share = Number((o.totalMs / total * 100).toFixed(2)); });
    return out.sort((a, b) => b.totalMs - a.totalMs);
  }

  const distOps = distributedOps(ops);
  const distTotal = distOps.reduce((a, o) => a + o.totalMs, 0);
  const distGroups = [
    { id: 'mlp', label: 'MLP', detail: 'gate/up · silu · down（TP 1/8）' },
    { id: 'attn', label: 'Attention', detail: 'work_build · rope · fa_fused · softmax' },
    { id: 'proj', label: 'QKV / Out 投影', detail: 'q/k/v_proj · out_proj' },
    { id: 'norm', label: 'Norm / Carry', detail: 'rms · qk_norm · residual · dcr_xgamma' },
    { id: 'boundary', label: 'LM Head 与边界', detail: 'cast · rms_lm_head（仅 stage 1）' },
    { id: 'comm', label: '集合通信', detail: 'TP AllReduce · PP Send/Recv' },
    { id: 'idle', label: '气泡与同步', detail: '流水线气泡 · 层内 barrier' },
  ].map((g) => {
    const ms = distOps.filter((o) => o.group === g.id).reduce((a, o) => a + o.totalMs, 0);
    return { ...g, ms: Number(ms.toFixed(4)), share: Number((ms / distTotal * 100).toFixed(2)) };
  }).filter((g) => g.ms > 0);

  profiles['run-0812-mn'] = {
    id: 'run-0812-mn',
    title: 'Decode Fused · 2 节点 × 8 卡 · TP8/PP2',
    token: 'ptok://qwen3-14b/decode-fused-tp8pp2@run-0812-mn',
    meta: {
      model: 'Qwen3-14B', params: '14.8 B', dtype: 'BF16',
      batch: DIST.topology.batch, layers: 40, seqAvg: 1614, page: 128,
      device: `Ascend 950B × ${DIST.topology.world}`, hbm: 64, peakBw: PEAK_BW, peakFlops: PEAK_FLOPS,
      env: 'env:8da1bf09', envMatch: true,
      steps: 512, capturedAt: '2026-08-12 10:07:41', duration: '2.1 s',
      collector: 'PyPTO DFX · msprof v9.0 · 16 rank 汇聚',
      distributed: true,
    },
    summary: {
      tpot: { p50: Number(DIST.tpotMs.toFixed(2)), p90: Number((DIST.tpotMs * 1.09).toFixed(2)), p99: Number((DIST.tpotMs * 1.21).toFixed(2)) },
      tpotDelta: -76.6,
      tps: DIST.tps, tpsDelta: 755.7,
      ttft: 61, ttftDelta: -66.8,
      kvUsed: DIST.memory.kvUsed, kvPool: DIST.memory.kvPool, kvPct: DIST.memory.kvPages / DIST.memory.kvPagesTotal * 100,
      preempt: 0, batchAvg: DIST.topology.batch * 0.92,
      sol: [
        { id: 'cube', label: 'Cube (AIC)', pct: 2.1, detail: '每 rank 只做 1/8 的矩阵计算' },
        { id: 'vector', label: 'Vector (AIV)', pct: 9.8, detail: 'elementwise + 归约' },
        { id: 'mte2', label: 'MTE2 · HBM → 片上', pct: 38.6, detail: `${DIST.traffic.perRank.toFixed(2)} GB/rank · ${DIST.traffic.achievedBw.toFixed(2)} TB/s` },
        { id: 'mte3', label: 'MTE3 · 片上 → HBM', pct: 3.4, detail: '累加器回写' },
      ],
      bound: 'bubble',
      lowerBoundMs: Number((DIST.traffic.perRank / PEAK_BW).toFixed(2)),
      efficiency: Number((DIST.traffic.perRank / PEAK_BW / DIST.tpotMs * 100).toFixed(1)),
      traffic: {
        weights: Number((DIST.memory.weightsStage1 * DIST.topology.microbatches).toFixed(3)),
        kv: Number((DIST.topology.batch * 1614 * 1 * 128 * 2 * 2 * DIST.topology.stageLayers / 1e9).toFixed(3)),
        act: 0.025,
        total: Number(DIST.traffic.perRank.toFixed(3)),
      },
      flops: { total: Number((447.5 * 2 / 8).toFixed(1)), achieved: Number((447.5 * 2 / 8 / DIST.tpotMs).toFixed(1)), ai: Number((447.5 * 2 / 8 / DIST.traffic.perRank).toFixed(1)), ridge: 222.2 },
    },
    groups: distGroups,
    ops: distOps,
    itlBins: itlBins.map(([ms, n]) => [Number((ms / 15.2 * DIST.tpotMs).toFixed(2)), n]),
    seqLens,
    memory: {
      hbm: {
        capacity: 64,
        items: [
          ['weights', '权重分片 · BF16', Number(DIST.memory.weightsStage1.toFixed(3)), `20 层 / 8 分片 + LM Head 1/8（stage 1 卡）`],
          ['kv', 'KV Cache 页池', Number(DIST.memory.kvPool.toFixed(3)), `${DIST.memory.kvPagesTotal} 页 × ${DIST.memory.kvPageMb.toFixed(2)} MB · 1 个 KV head`],
          ['workspace', 'Workspace / 累加器', DIST.memory.workspace, '含 AllReduce 收发缓冲'],
          ['act', '激活与跨层 carry', DIST.memory.act, '每 rank 只持有分片'],
        ],
      },
      onchip: [
        ['L0A / L0B', 44, null, 'tile 变小，Cube 利用率随之下降'],
        ['L0C', 31, null, 'FP32 累加器'],
        ['L1', 38, null, '权重 tile 预取'],
        ['UB', 41, 58, '远低于预算 — 并行度过高导致每 rank 工作量太小'],
      ],
      kv: {
        pageTokens: 128, pageBytesMb: DIST.memory.kvPageMb,
        pagesTotal: DIST.memory.kvPagesTotal, pagesUsed: DIST.memory.kvPages,
        tokensLive: DIST.topology.batch * 1614,
        tokensAllocated: DIST.memory.kvPages * 128,
        bytesAllocated: DIST.memory.kvUsed, bytesPool: DIST.memory.kvPool,
        fragmentation: (DIST.memory.kvPages * 128 - DIST.topology.batch * 1614) / (DIST.memory.kvPages * 128) * 100,
        utilization: DIST.memory.kvPages / DIST.memory.kvPagesTotal * 100,
        hitRate: 98.4, preempt: 0, swap: 0,
        mcb: 16, blocksPadded: 16 * DIST.topology.batch, blocksReal: DIST.memory.kvPages,
        density: DIST.memory.kvPages / (16 * DIST.topology.batch) * 100,
        perRequest: Array.from({ length: DIST.topology.batch }, (_, i) => {
          const seq = seqLens[i % seqLens.length];
          return { req: `req-${String(i).padStart(2, '0')}`, seq, pages: Math.ceil(seq / 128) };
        }),
      },
      precision: memory.precision,
    },
    serving: {
      ...serving,
      queue: { ...serving.queue, running: DIST.topology.batch, waiting: 1 },
      sweep: serving.sweep.map((s) => ({ ...s, current: s.batch === DIST.topology.batch })),
    },
    dist: DIST,
    baseline: { id: 'run-0803-a', token: profiles['run-0803-a'].token, tpot: 15.2, tps: 1053, label: '单卡基线 · 08/03' },
  };

  window.PtoInferenceProfile = {
    profiles,
    current: 'run-0803-a',
    constants: { PEAK_BW, PEAK_FLOPS, TPOT },
    list: () => Object.values(profiles).map((p) => ({ id: p.id, title: p.title, device: p.meta.device, batch: p.meta.batch, tpot: p.summary.tpot.p50, tps: p.summary.tps, distributed: !!p.meta.distributed })),
    get: (id) => profiles[id || 'run-0803-a'],
  };
})();
