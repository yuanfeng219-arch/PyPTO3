/*
 * Correctness / Numerical Accuracy profiles.
 *
 * This is the only mock-data source for the Run diagnosis experience. The
 * renderer consumes this schema; it does not own a second copy of any result,
 * tensor, runtime, or diagnosis state. Production adapters can replace this
 * registry with DFX / pass-validation artifacts without changing the view.
 */
(function () {
  'use strict';

  const runtimeDataError106 = {
    id: 'numerical-runtime-data-error-106',
    runId: 'run_106',
    type: 'numerical',
    source: 'mock',
    result: { verdict: 'fail', output: 'out', maxAbs: 0.382, maxRel: 0.117 },
    reference: { source: 'PyTorch Golden', actual: 'Args Dump · Run #106', fixed: true, status: 'valid' },
    tolerance: { rtol: '5e-2', atol: '5e-2', status: 'valid' },
    expectedDifference: { status: 'none', reason: '未声明允许的数值偏差' },
    compiler: {
      structuralVerification: { status: 'pass' },
      numericalValidation: {
        status: 'pass', passed: 42, total: 42, firstDivergentPass: null,
        summary: '所有 Pass 正常完成，且每个 Pass 后 selected output 与 Golden 匹配'
      }
    },
    repeatability: { runs: 3, stable: false, summary: '3 次运行不稳定' },
    firstDivergence: {
      kind: 'tensor', id: 'attention_out', outputId: 'out',
      trail: [
        { kind: 'tensor', id: 'out', label: '输出不一致' },
        { kind: 'op', id: 'attention', label: 'Attention' },
        { kind: 'tensor', id: 'attention_out', label: 'attention_out' },
        { kind: 'task', id: '182', label: 'Task #182' }
      ]
    },
    semanticGraph: {
      ops: [
        { id: 'q_proj', name: 'Q Projection', be: 'AIC', src: 'decode_layer.py:701', ins: 1, outs: 1 },
        { id: 'k_proj', name: 'K Projection', be: 'AIC', src: 'decode_layer.py:703', ins: 1, outs: 1 },
        { id: 'v_proj', name: 'V Projection', be: 'AIC', src: 'decode_layer.py:705', ins: 1, outs: 1 },
        { id: 'rope_q', name: 'RoPE(Q)', be: 'AIV', src: 'decode_layer.py:712', ins: 1, outs: 1 },
        { id: 'rope_k', name: 'RoPE(K)', be: 'AIV', src: 'decode_layer.py:713', ins: 1, outs: 1 },
        { id: 'score', name: 'Attention Score', be: 'AIC', src: 'decode_layer.py:718', ins: 2, outs: 1 },
        { id: 'softmax', name: 'Softmax', be: 'AIV', src: 'decode_layer.py:721', ins: 1, outs: 1 },
        { id: 'attention', name: 'Attention', be: 'AIC', src: 'decode_layer.py:728', ins: 2, outs: 1 },
        { id: 'out_proj', name: 'Output Projection', be: 'AIC', src: 'decode_layer.py:735', ins: 1, outs: 1 },
        { id: 'residual', name: 'Residual Add', be: 'AIV', src: 'decode_layer.py:741', ins: 2, outs: 1 }
      ],
      edges: [
        { from: 'hidden_states', to: 'q_proj' }, { from: 'hidden_states', to: 'k_proj' }, { from: 'hidden_states', to: 'v_proj' },
        { from: 'q_proj', via: 'q', to: 'rope_q' }, { from: 'k_proj', via: 'k', to: 'rope_k' },
        { from: 'rope_q', via: 'q_rotated', to: 'score' }, { from: 'rope_k', via: 'k_rotated', to: 'score' },
        { from: 'v_proj', via: 'v', to: 'attention', lane: 'right' }, { from: 'score', via: 'attn_score', to: 'softmax' },
        { from: 'softmax', via: 'softmax_p', to: 'attention' }, { from: 'attention', via: 'attention_out', to: 'out_proj' },
        { from: 'out_proj', via: 'projected_out', to: 'residual' }, { from: 'hidden_states', to: 'residual', lane: 'left' },
        { from: 'residual', via: 'out', to: null }
      ]
    },
    tensors: [
      { id: 'hidden_states', name: 'hidden_states', tid: 'T12', shape: '[16, 40, 5120]', dtype: 'BF16', state: 'match', ref: true, act: true, maxAbs: 0.0009, maxRel: 0.0004 },
      { id: 'q', name: 'q', tid: 'T31', shape: '[16, 40, 5120]', dtype: 'BF16', state: 'match', ref: true, act: true, maxAbs: 0.0011, maxRel: 0.0006 },
      { id: 'k', name: 'k', tid: 'T34', shape: '[16, 40, 5120]', dtype: 'BF16', state: 'match', ref: true, act: true, maxAbs: 0.0013, maxRel: 0.0007 },
      { id: 'v', name: 'v', tid: 'T36', shape: '[16, 40, 5120]', dtype: 'BF16', state: 'match', ref: true, act: true, maxAbs: 0.0014, maxRel: 0.0007 },
      { id: 'q_rotated', name: 'q_rotated', tid: 'T32', shape: '[16, 40, 5120]', dtype: 'BF16', state: 'match', ref: true, act: true, maxAbs: 0.0016, maxRel: 0.0009 },
      { id: 'k_rotated', name: 'k_rotated', tid: 'T35', shape: '[16, 40, 5120]', dtype: 'BF16', state: 'match', ref: true, act: true, maxAbs: 0.0017, maxRel: 0.0009 },
      { id: 'attn_score', name: 'attn_score', tid: 'T38', shape: '[16, 40, 40]', dtype: 'FP32', state: 'unchecked', ref: false, act: true },
      { id: 'softmax_p', name: 'softmax_p', tid: 'T39', shape: '[16, 40, 40]', dtype: 'FP32', state: 'unchecked', ref: false, act: true },
      { id: 'attention_out', name: 'attention_out', tid: 'T37', shape: '[16, 40, 128]', dtype: 'BF16', state: 'first', ref: true, act: true, maxAbs: 0.214, maxRel: 0.083 },
      { id: 'projected_out', name: 'projected_out', tid: 'T40', shape: '[16, 40, 5120]', dtype: 'BF16', state: 'unchecked', ref: false, act: true },
      { id: 'out', name: 'out', tid: 'T41', shape: '[16, 40, 5120]', dtype: 'BF16', state: 'propagated', ref: true, act: true, maxAbs: 0.382, maxRel: 0.117 }
    ],
    runtime: {
      expansion: { afterTensor: 'attention_out', entryTask: '178', exitTask: '196' },
      tasks: [
        { id: '178', role: 'upstream', label: '上游任务', core: 'AICore 3', reads: [], writes: ['q_rotated', 'k_rotated', 'v'], note: '准备 q / k / v' },
        { id: '182', role: 'producer', label: '生产者', core: 'AICore 7', reads: ['q_rotated', 'k_rotated', 'v'], writes: ['attention_out'], shared: { buffer: 'B2', op: 'read' }, semantic: 'Attention', note: '读共享 buffer B2 · 产生 attention_out' },
        { id: '196', role: 'consumer', label: '消费者', core: 'AICore 2', reads: ['attention_out'], writes: [], semantic: 'Output Projection', note: '消费 attention_out' },
        { id: '197', role: 'suspicious', label: '可疑写入', core: 'AICore 5', reads: [], writes: [], shared: { buffer: 'B2', op: 'write' }, expectsAfter: '182', note: '写共享 buffer B2 · 缺少与 Task #182 的排序依赖' }
      ],
      runtimeChain: [
        { from: '178', to: '182', via: null, label: 'q_rotated · k_rotated · v' },
        { from: '182', to: '196', via: 'attention_out', label: 'attention_out' }
      ],
      timeline: {
        base: 100, span: 300, tickStep: 50, buffer: 'B2',
        rows: [{ task: '178', s: 104, e: 146 }, { task: '182', s: 158, e: 286 }, { task: '197', s: 238, e: 344 }, { task: '196', s: 322, e: 381 }],
        overlap: { from: 238, to: 286, a: '182', b: '197' }
      }
    },
    diagnosis: {
      category: 'runtime_data_error', confidence: 'high', label: '可能原因',
      summary: '运行时排序缺失或有误',
      route: 'runtime',
      routeLabel: '定位运行时数据 / 执行排序',
      rationale: '结构校验与逐 Pass 数值校验均通过，但设备结果不匹配',
      evidence: ['上游 tensor 全部匹配', 'attention_out 是首个分歧点', '3 次重复运行结果不一致', 'Task #182 与 #197 时间线重叠', '两者之间缺少排序依赖边']
    }
  };

  /* Keep this profile attached to the existing Compilation fixture instead of
     copying per-pass values here. The fixture remains the single source for
     the numerical window, its mismatch metrics, and the existing IR diff. */
  const compilerSemanticError109 = {
    id: 'numerical-compiler-semantic-error-109',
    runId: 'run_109',
    type: 'numerical',
    source: 'fixture',
    fixture: 'compiler_semantic_error',
    result: { verdict: 'fail', output: 'out', maxAbs: 0.028, maxRel: 0.014 },
    reference: { source: 'PyTorch Golden', actual: 'Host IR execution · per-pass validation', fixed: true, status: 'valid' },
    tolerance: { rtol: '5e-2', atol: '5e-2', status: 'valid' },
    expectedDifference: { status: 'none', reason: '未声明允许的数值偏差' },
    compiler: {
      structuralVerification: { status: 'pass' },
      numericalValidation: window.PTO_COMPILATION?.numericalFixtures?.compiler_semantic_error || {
        status: 'not_collected', tolerance: null, passes: [], firstDivergentPass: null
      }
    },
    firstDivergence: { kind: 'pass', id: 'ExpandMixedKernel', trail: [] },
    semanticGraph: { ops: [], edges: [] },
    tensors: [],
    diagnosis: {
      category: 'compiler_semantic_error', confidence: 'high', label: '诊断结论',
      summary: '编译语义变换引入数值偏差',
      route: 'compilation',
      routeLabel: '定位编译语义变换',
      rationale: '结构校验通过，但 Host IR execution 在 ExpandMixedKernel 后首次偏离 Golden',
      evidence: ['PyTorch Golden reference 有效', 'Tolerance 有效，且未声明允许差异', '结构校验: PASS', 'ExpandMixedKernel: FIRST DIVERGENCE', '后续 Pass 持续 MISMATCH']
    }
  };

  const profiles = Object.freeze({ run_106: runtimeDataError106, run_109: compilerSemanticError109, default: runtimeDataError106 });
  window.PTO_CORRECTNESS_DIAGNOSTICS = Object.freeze({
    schemaVersion: 1,
    profiles,
    get(runId) { return profiles[runId] || profiles.default; }
  });
}());
