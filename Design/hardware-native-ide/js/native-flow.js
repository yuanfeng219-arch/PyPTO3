(function () {
  'use strict';
  const host = document.getElementById('nativeFlow');
  const workbench = document.querySelector('[data-ide-frame]');
  const state = { seq: 256, page: 256, plan: 'current', focus: 'right', committed: false };
  const nodes = {
    cache: { name: 'KV Cache', eyebrow: 'GM · 全局内存', title: '逻辑块从哪里来', text: 'block_table 将逻辑块映射到物理页。此示例要求 SEQ_TILE 与 runtime.page_size 相等。', code: 'cache_row = layer_cache_base +\n  (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE\nk_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]' },
    right: { name: 'Right buffer', eyebrow: 'L0B · 片上存储', title: '数据能否放得下', text: '按两份完整 BF16 块同时驻留估算。真实分配、缓冲数量与驻留关系需要编译产物确认。', code: 'k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]\nraw_scores = pl.matmul(q_padded, k_tile,\n                       b_trans=True, out_dtype=pl.FP32)' },
    cube: { name: 'Cube', eyebrow: 'AIC · 矩阵计算', title: '分块如何影响计算', text: 'QK 按序列维分段时需要保留各段分数，并在完整分数上执行后续 Softmax。这里的循环是方案示意。', code: 'raw_scores = pl.matmul(q_padded, k_tile,\n                       b_trans=True, out_dtype=pl.FP32)\nscores_scaled = pl.mul(raw_scores, ATTN_SCALE)' },
    vector: { name: 'Vector', eyebrow: 'AIV · 向量计算', title: '下游语义需要保持', text: '有效区域、填充和 Softmax 必须保持原语义；分块后的正确性不能由容量估算推断。', code: 'scores_valid = pl.set_validshape(scores_scaled,\n                                  Q_HEAD_BATCH, valid_len)\nscores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)\ncur_mi = pl.row_max(scores)' }
  };
  function values() {
    const seq = state.plan === 'small' ? 128 : state.seq;
    const chunk = state.plan === 'chunk' ? 128 : seq;
    return { seq: seq, page: state.page, chunk: chunk, kib: chunk * 128 * 4 / 1024, match: seq === state.page };
  }
  function buffer(v) {
    const total = v.kib, max = Math.max(128, total), p = 100 * total / max, boundary = 100 * 64 / max;
    return '<div class="nf-buffer"><div class="nf-buffer-top"><span>RIGHT / L0B</span><b>' + total + ' <small>/ 64 KiB</small></b></div>'
      + '<div class="nf-track"><div class="nf-capacity" style="width:' + boundary + '%"></div><div class="nf-fill" style="width:' + p + '%"><span>0</span><span>1</span></div><i style="left:' + boundary + '%"></i></div>'
      + '<div class="nf-scale"><span>0</span><span>↑ 64 KiB 容量边界</span></div><p>' + v.chunk + ' × 128 × 2 B × 2 份 = ' + total + ' KiB <em>· 双缓冲假设</em></p></div>';
  }
  function render() {
    const v = values(), n = nodes[state.focus], over = v.kib > 64;
    const planLabel = state.plan === 'current' ? '当前配置' : state.plan === 'small' ? '方案 A · 缩小逻辑块' : '方案 B · 内部分块';
    const node = function (id, icon, summary, status) { return '<button class="nf-node ' + (state.focus === id ? 'selected' : '') + '" data-nf-focus="' + id + '" aria-pressed="' + (state.focus === id) + '"><span class="nf-node-top"><i>' + icon + '</i><small>' + nodes[id].eyebrow + '</small></span><b>' + nodes[id].name + '</b><span>' + summary + '</span>' + (status ? '<em>' + status + '</em>' : '') + '</button>'; };
    host.innerHTML = '<header class="nf-header"><div><b>HARDWARE-NATIVE <span>LENS</span></b><i></i><span>decode_layer.py / fa_fused</span></div><div><span class="nf-target">● 910B 容量参考 · 示例模型</span><button data-nf-action="workbench">打开完整 Lens 工作台 ↗</button></div></header>'
      + '<main class="nf-main"><div class="nf-heading"><div><div class="nf-eyebrow">CODE → DATA FLOW → HARDWARE → DECISION</div><h1>让代码落到硬件上</h1><p>沿着一次 QK 计算，看数据如何经过片上空间，以及代码选择带来的约束。</p></div><button class="nf-reset" data-nf-action="reset">↺ 重置场景</button></div>'
      + '<div class="nf-context"><div><small>工作负载</small><b>Qwen3-14B · decode</b></div><div><small>目标与依据</small><b>910B 参考容量 · 64 KiB</b></div><div><small>当前视角</small><b>' + planLabel + '</b></div><span>示意估算 · 未编译 · 未上板</span></div>'
      + '<div class="nf-layout"><section class="nf-code nf-panel"><div class="nf-panel-head"><div><small>01 / SOURCE</small><h2>代码选择</h2></div><span>decode_layer.py</span></div><p class="nf-explain">参数决定逻辑块的形状。点击数据路径中的节点，查看对应代码。</p>'
      + '<div class="nf-param"><div><span>SEQ_TILE</span><strong>' + v.seq + '</strong><small>KV 逻辑块长度</small></div><div><span>page_size</span><strong>' + v.page + '</strong><small>运行时物理页</small></div></div>'
      + '<div class="nf-code-label"><span>代码落点 · ' + n.name + '</span><span>源码节选</span></div><pre class="nf-source">' + n.code.replace(/&/g,'&amp;').replace(/</g,'&lt;') + '</pre>'
      + '<div class="nf-linkage"><span>当前形状</span><b>K tile [' + v.seq + ', 128] · BF16</b><small>' + (state.plan === 'chunk' ? '探索：单次处理 ' + v.chunk + ' 行，不改变逻辑块。' : '一个逻辑块作为一次完整块估算。') + '</small></div></section>'
      + '<section class="nf-flow nf-panel"><div class="nf-panel-head"><div><small>02 / EXECUTION PATH</small><h2>数据如何经过硬件</h2></div><span>选择节点查看关系</span></div>'
      + '<div class="nf-node-grid">' + node('cache','01','逻辑块 ' + v.seq + ' · 物理页 ' + v.page, v.match ? '分页契约匹配' : '分页契约不匹配')
      + node('right','02','BF16 · 两份驻留',over ? '估算超出 ' + (v.kib - 64) + ' KiB' : '估算 ' + v.kib + ' / 64 KiB')
      + node('cube','03','Q × Kᵀ · 矩阵乘',state.plan === 'chunk' ? '示意分 ' + Math.ceil(v.seq / v.chunk) + ' 段' : '按完整块计算')
      + node('vector','04','mask · softmax','语义待验证') + '</div>'
      + '<div class="nf-detail"><div class="nf-detail-head"><div><small>' + n.eyebrow + '</small><h3>' + n.title + '</h3></div><span>当前 · ' + n.name + '</span></div>'
      + (state.focus === 'right' ? buffer(v) : '<div class="nf-focus-explain"><b>' + n.name + '</b><span>' + (state.focus === 'cache' ? '逻辑块 ' + v.seq + ' / 物理页 ' + v.page + ' · ' + (v.match ? '契约匹配' : '契约不匹配') : state.focus === 'cube' ? 'K [' + v.chunk + ', 128] → QK 分数' : '分数 → 有效区域 → Softmax') + '</span></div>')
      + '<p class="nf-detail-text">' + n.text + '</p></div><div class="nf-provenance">示意模型：不是实际指令、编译后分配或设备采样。 <a href="https://github.com/hw-native-sys/pypto/issues/1595" target="_blank" rel="noopener noreferrer">容量参考 ↗</a></div></section>'
      + '<aside class="nf-decision nf-panel"><div class="nf-panel-head"><div><small>03 / DECISION</small><h2>比较开发决策</h2></div></div><div class="nf-verdict ' + (over ? 'bad' : 'good') + '"><small>当前选择 · ' + planLabel + '</small><b>' + (over ? '完整块超出参考容量' : '容量估算未超限') + '</b><span>' + v.kib + ' / 64 KiB · ' + (v.match ? '分页契约匹配' : '分页契约不匹配') + '</span></div>'
      + '<p class="nf-explain">选择一个方案，代码、路径和硬件占用会一起变化。</p><div class="nf-plans">'
      + '<button data-nf-plan="current" aria-pressed="' + (state.plan === 'current') + '"><i>—</i><span><b>当前配置</b><small>完整块 ' + state.seq + ' · 双缓冲 128 KiB</small></span></button>'
      + '<button data-nf-plan="small" aria-pressed="' + (state.plan === 'small') + '"><i>A</i><span><b>缩小逻辑块</b><small>SEQ_TILE 128 · 需处理分页契约</small></span></button>'
      + '<button data-nf-plan="chunk" aria-pressed="' + (state.plan === 'chunk') + '"><i>B</i><span><b>保持逻辑块，内部再分块</b><small>256 拆为 2 × 128 · 改变计算循环</small></span></button></div>'
      + '<div class="nf-consequence"><small>选择后的影响</small><p>' + (state.plan === 'small' ? '容量估算降至 64 KiB；但 page_size 仍为 256，当前分页契约不匹配。需要确认运行时是否可同步调整。' : state.plan === 'chunk' ? '单次占用估算降至 64 KiB，分页契约不变；需要实现分段计算并核对分数合并与 Softmax 语义。' : '逻辑块与分页契约匹配，但双缓冲估算超出 Right 参考容量 64 KiB。') + '</p></div>'
      + '<div class="nf-validation"><small>接下来需要验证</small><div><span>○ 编译分配</span><span>○ 数值正确性</span><span>○ 设备性能</span></div><p>均未执行 · 页面不修改源码或运行时；与完整工作台配置独立。</p></div></aside></div></main>';
  }
  host.addEventListener('click', function (event) {
    const target = event.target.closest('[data-nf-focus], [data-nf-plan], [data-nf-action]');
    if (!target) return;
    if (target.dataset.nfFocus) state.focus = target.dataset.nfFocus;
    if (target.dataset.nfPlan) { state.plan = target.dataset.nfPlan; state.focus = 'right'; }
    if (target.dataset.nfAction === 'reset') { state.plan = 'current'; state.focus = 'right'; }
    if (target.dataset.nfAction === 'workbench') { host.hidden = true; workbench.inert = false; document.getElementById('openNativeFlow').focus(); return; }
    render();
    const attr = target.dataset.nfFocus ? 'data-nf-focus' : target.dataset.nfPlan ? 'data-nf-plan' : 'data-nf-action';
    const replacement = host.querySelector('[' + attr + '="' + target.getAttribute(attr) + '"]');
    if (replacement) replacement.focus({ preventScroll: true });
  });
  document.getElementById('openNativeFlow').addEventListener('click', function () {
    workbench.inert = true;
    host.hidden = false;
    host.querySelector('[data-nf-action="workbench"]').focus();
  });
  render();
})();
