/*
 * Run investigation graph adapter.
 *
 * This module deliberately derives its evidence from the existing correctness
 * profiles and compilation fixture. It contains no second numerical fixture:
 * missing source material stays missing in the returned investigation.
 */
(function () {
  'use strict';

  const TABS = new Set(['correctness', 'execution', 'compilation']);
  const KINDS = new Set(['tensor', 'task', 'pass', 'buffer']);
  const STATES = new Set(['normal', 'abnormal', 'unresolved']);
  const EDGE_TYPES = new Set(['locate', 'promote', 'continue', 'checked', 'unresolved', 'support', 'diagnose']);

  const asArray = value => Array.isArray(value) ? value : [];
  const present = value => value !== undefined && value !== null && value !== '';
  const text = value => present(value) ? String(value) : '未采集';
  const action = (runId, tab, kind, id, extra) => Object.assign({ runId, tab, kind, id }, extra || {});

  function evidence(id, title, note, columns, rows, source, scope, missing, next) {
    return { id, title, note, columns, rows, source, scope, missing: missing ? '此范围证据不完整，不能视为通过' : '', action: next || null };
  }

  function node(id, kind, state, title, signal, summary, evidenceRefs) {
    return { id, kind, statusLabel: signal.includes('未采集') ? '未采集' : signal === '未执行' ? '未执行' : state === 'normal' ? '已检查且匹配' : null, state: STATES.has(state) ? state : 'unresolved', title, signal, summary, evidenceRefs };
  }

  function edge(from, to, type, label) {
    return { id: from + '-' + to, source: from, target: to, relation: EDGE_TYPES.has(type) ? type : 'unresolved', label };
  }

  function profile106() {
    return window.PTO_CORRECTNESS_DIAGNOSTICS?.profiles?.run_106 || null;
  }

  function fixture109() {
    return window.PTO_COMPILATION?.numericalFixtures?.compiler_semantic_error || null;
  }

  function build106(profile) {
    const result = profile?.result;
    const first = profile?.firstDivergence;
    const tensors = asArray(profile?.tensors);
    const runtime = profile?.runtime;
    const tasks = asArray(runtime?.tasks);
    const output = tensors.find(t => t.id === result?.output || t.id === 'out');
    const attention = tensors.find(t => t.id === 'attention_out');
    const score = tensors.find(t => t.id === 'attn_score');
    const softmax = tensors.find(t => t.id === 'softmax_p');
    const observedInputs = ['q', 'k', 'v', 'q_rotated', 'k_rotated'].map(id => tensors.find(t => t.id === id));
    const producer = tasks.find(t => t.id === '182');
    const overlap = runtime?.timeline?.overlap;
    const repeatability = profile?.repeatability;
    const compiler = profile?.compiler;
    const validation = compiler?.numericalValidation;
    const hasOutput = !!result && !!output && output.ref && output.act && present(result.maxAbs);
    const hasFirst = !!first && first.id === 'attention_out' && !!attention && attention.ref && attention.act && present(attention.maxAbs);
    const inputsKnown = observedInputs.length === 5 && observedInputs.every(t => t?.state === 'match' && t.ref && t.act);
    const hasProducer = !!producer && producer.semantic === 'Attention';
    const hasOverlap = !!overlap && overlap.a === '182' && overlap.b === '197' && runtime?.timeline?.buffer === 'B2';
    const hasRepeatability = !!repeatability && Number.isFinite(repeatability.runs) && typeof repeatability.stable === 'boolean';
    const passesKnown = validation?.status === 'pass' && Number.isFinite(validation.passed) && Number.isFinite(validation.total);
    const structuralKnown = compiler?.structuralVerification?.status === 'pass';
    const refs = {};

    refs.output = evidence('output', '输出比较', hasOutput ? '设备输出与 Golden 的比较已记录。' : '缺少输出比较，不能确认输出偏差。',
      ['输出', '最大绝对误差', '最大相对误差', '结论'], hasOutput ? [[text(result.output), result.maxAbs, result.maxRel, text(result.verdict)]] : [],
      'Run #106 正确性与执行采集记录', '输出 out', !hasOutput,
      action('run_106', 'correctness', 'tensor', 'out'));
    refs['first-divergence'] = evidence('first-divergence', '已采集检查点中的首个分歧', hasFirst ? '采样链将首个已记录的分歧定位到 attention_out。' : '采样链或 attention_out 比较缺失。',
      ['Tensor', '最大绝对误差', '最大相对误差', '参考'], hasFirst ? [[attention.id, attention.maxAbs, attention.maxRel, attention.ref ? '已提供' : '缺失']] : [],
      'Run #106 正确性与执行采集记录', '已采样 tensor', !hasFirst,
      action('run_106', 'correctness', 'tensor', 'attention_out'));
    refs['unchecked-intermediates'] = evidence('unchecked-intermediates', '未比较的中间值', 'attn_score 与 softmax_p 没有参考值；输入匹配不等于计算链已证明正确。',
      ['Tensor', '实际值', '参考值', '状态'], [[text(score?.id), score?.act ? '已提供' : '缺失', score?.ref ? '已提供' : '缺失', '未解决'], [text(softmax?.id), softmax?.act ? '已提供' : '缺失', softmax?.ref ? '已提供' : '缺失', '未解决']],
      'Run #106 正确性与执行采集记录', 'Attention 中间值', true,
      action('run_106', 'correctness', 'tensor', 'attn_score'));
    refs['observed-inputs'] = evidence('observed-inputs', '已观察到的上游输入', inputsKnown ? '当前采样中这些输入均有参考比较并匹配。' : '上游输入比较不完整，不能将其视为已匹配。',
      ['Tensor', '状态', '参考'], inputsKnown ? observedInputs.map(t => [t.id, t.state, '已提供']) : [],
      'Run #106 正确性与执行采集记录', 'q / k / v / q_rotated / k_rotated', !inputsKnown,
      action('run_106', 'correctness', 'tensor', 'q'));
    refs['producer-task'] = evidence('producer-task', 'Attention 生产任务', hasProducer ? 'attention_out 的生产任务为 Task #182（Attention）。' : 'Task #182 与 Attention 的映射缺失。',
      ['任务', '语义', '读取', '写入'], hasProducer ? [[producer.id, producer.semantic, producer.reads.join(' · '), producer.writes.join(' · ')]] : [],
      'Run #106 正确性与执行采集记录', 'Task #182', !hasProducer,
      action('run_106', 'execution', 'task', '182'));
    refs['buffer-overlap'] = evidence('buffer-overlap', '共享缓冲区重叠', hasOverlap ? 'Task #182 与 #197 在 B2 上发生时间重叠；排序关系仍需验证。' : 'B2 重叠或任务标识未采集。',
      ['Buffer', '任务 A', '任务 B', '重叠区间'], hasOverlap ? [[runtime.timeline.buffer, overlap.a, overlap.b, overlap.from + '–' + overlap.to]] : [],
      'Run #106 正确性与执行采集记录', 'B2 与任务时间线', !hasOverlap,
      action('run_106', 'execution', 'buffer', 'B2', { taskIds: ['182', '197'], range: overlap ? { from: overlap.from, to: overlap.to } : null }));
    refs.repeatability = evidence('repeatability', '重复运行', hasRepeatability ? (repeatability.stable ? '重复运行结果稳定。' : '重复运行结果不稳定，作为排序调查的支持线索。') : '重复运行记录缺失。',
      ['运行次数', '稳定', '说明'], hasRepeatability ? [[repeatability.runs, repeatability.stable ? '是' : '否', text(repeatability.summary)]] : [],
      'Run #106 正确性与执行采集记录', 'Run #106', !hasRepeatability,
      action('run_106', 'execution', 'task', '182'));
    refs['compiler-checks'] = evidence('compiler-checks', '编译阶段检查', structuralKnown && passesKnown ? '结构检查通过；已记录的 ' + validation.passed + ' 个 selected-output Pass 检查通过。' : '编译检查证据不完整，不能据此排除编译问题。',
      ['检查', '状态', '范围'], [structuralKnown ? ['Structural verification', 'pass', '已记录'] : ['Structural verification', '未采集', '未知'], passesKnown ? ['Selected-output Pass validation', validation.passed + '/' + validation.total + ' pass', '已记录的 Pass 输出'] : ['Selected-output Pass validation', '未采集', '未知']],
      'Run #106 正确性与执行采集记录', '结构与 selected output', !(structuralKnown && passesKnown),
      action('run_106', 'compilation', 'pass', null));

    const runtimeEvidenceReady = hasOutput && hasFirst && hasProducer && hasOverlap && tasks.some(t => t.id === '197' && t.expectsAfter === '182');
    const nodes = [
      node('output-out', 'finding', hasOutput ? 'abnormal' : 'unresolved', '输出 out 不一致', hasOutput ? 'max abs ' + result.maxAbs : '输出比较未采集', hasOutput ? '输出偏差已观察到。' : '无法确认输出状态。', ['output']),
      node('device-result', 'finding', hasOutput ? 'abnormal' : 'unresolved', '设备结果偏差', hasOutput ? '已记录输出比较' : '未采集', '沿已采集检查点定位。', ['output']),
      node('intermediate-gap', 'reasoning', 'unresolved', '中间计算证据缺口', '未采集 reference', 'attn_score / softmax_p 缺少参考值，尚不能排除中间计算。', ['unchecked-intermediates']),
      node('repeatability', 'reasoning', 'unresolved', '重复运行线索', hasRepeatability ? repeatability.runs + ' 次运行' : '未采集', hasRepeatability && !repeatability.stable ? '结果不稳定，支持进一步检查排序。' : '尚无不稳定性支持证据。', ['repeatability']),
      node('input-observed', 'entity', inputsKnown ? 'normal' : 'unresolved', '已观察到的上游输入', inputsKnown ? '已采集输入匹配' : '上游输入比较未完整采集', '这只覆盖已采样输入，不证明全部输入或整个计算链。', ['observed-inputs', 'unchecked-intermediates']),
      node('first-attention-out', 'finding', hasFirst ? 'abnormal' : 'unresolved', 'attention_out 是已采集检查点中的首个分歧', hasFirst ? 'max abs ' + attention.maxAbs : '首个分歧未采集', hasFirst ? '分歧在 Attention 输出处出现。' : '无法确认首个分歧。', ['first-divergence']),
      node('attention-182', 'entity', hasProducer ? 'abnormal' : 'unresolved', 'Attention / Task #182', hasProducer ? '产生 attention_out' : 'Task 映射未采集', hasProducer ? '继续检查该生产任务的执行排序。' : '缺少生产任务证据。', ['producer-task']),
      node('b2-overlap', 'finding', hasOverlap ? 'abnormal' : 'unresolved', 'B2 重叠访问', hasOverlap ? 'Task #182 与 #197 重叠' : '时间线未采集', hasOverlap ? '重叠是待验证的运行时风险信号。' : '无法判断共享缓冲区关系。', ['buffer-overlap']),
      node('war-hypothesis', 'hypothesis', runtimeEvidenceReady ? 'unresolved' : 'unresolved', '缺少 WAR 排序依赖', runtimeEvidenceReady ? '待验证：#182 → #197' : '证据不足，待验证', '时间线支持此假设，但尚未证明其为根因。', ['buffer-overlap', 'producer-task', 'repeatability']),
      node('compiler-checks', 'reasoning', structuralKnown && passesKnown ? 'normal' : 'unresolved', '编译检查未发现已记录分歧', passesKnown ? validation.passed + '/' + validation.total + ' selected output 通过' : '检查未完整采集', '该证据范围有限，不能证明所有输入或计算正确。', ['compiler-checks']),
      node('runtime-judgment', 'diagnosis', runtimeEvidenceReady ? 'unresolved' : 'unresolved', 'Runtime 判断待验证', runtimeEvidenceReady ? '优先验证 B2 的 WAR 排序' : '关键证据缺失，不能归因', '运行时数据流是假设方向，尚不能形成确认结论。', ['buffer-overlap', 'repeatability', 'unchecked-intermediates'])
    ];
    const edges = [
      edge('output-out', 'device-result', 'locate', '设备比较'),
      edge('device-result', 'first-attention-out', 'locate', '沿检查点定位'),
      edge('device-result', 'input-observed', 'checked', '检查输入'),
      edge('device-result', 'compiler-checks', 'checked', '检查编译'),
      edge('first-attention-out', 'intermediate-gap', 'unresolved', '参考缺口'),
      edge('attention-182', 'repeatability', 'support', '重复运行'),
      edge('repeatability', 'war-hypothesis', 'support', '支持线索'),

      edge('first-attention-out', 'attention-182', 'continue', '生产任务'),
      edge('attention-182', 'b2-overlap', 'locate', '共享 B2'),
      edge('b2-overlap', 'war-hypothesis', 'promote', '待验证的排序风险'),
      edge('war-hypothesis', 'runtime-judgment', 'diagnose', '运行时方向'),
      edge('compiler-checks', 'runtime-judgment', 'support', '已记录编译检查'),
      edge('first-attention-out', 'runtime-judgment', 'unresolved', '中间值无参考')
    ];
    return { nodes, edges, evidenceById: refs, summary: hasOutput && hasFirst ? '输出偏差已定位到 attention_out 的采样链，下一步验证 Task #182 与 #197 的 B2 排序。' : hasOutput ? '输出偏差已观察到，但已采集检查点中的首个分歧尚未完整定位。' : 'Run #106 的输出比较未完整采集。', judgment: runtimeEvidenceReady ? '优先检查 Runtime 排序，根因待验证。' : '关键证据不足，暂不能确定调查方向。', nextStep: '检查 Task #182 与 #197 的依赖与 B2 访问顺序；补齐依赖后固定输入复验，并补齐中间值参考比较。', action: action('run_106', 'execution', 'buffer', 'B2', { taskIds: ['182', '197'], range: overlap ? { from: overlap.from, to: overlap.to } : null }) };
  }

  function build109(fixture) {
    const structural = window.PTO_CORRECTNESS_DIAGNOSTICS?.profiles?.run_109?.compiler?.structuralVerification?.status;
    const passes = asArray(fixture?.passes);
    const firstName = fixture?.firstDivergentPass;
    const firstIndex = passes.findIndex(p => p.name === firstName);
    const first = firstIndex >= 0 ? passes[firstIndex] : null;
    const adjacent = firstIndex > 0 ? passes[firstIndex - 1] : null;
    const sourceReady = fixture?.status === 'fail' && first && fixture?.tolerance && present(fixture.tolerance.rtol) && present(fixture.tolerance.atol);
    const refs = {};
    refs['compiler-numerical-validation'] = evidence('compiler-numerical-validation', '编译数值校验', sourceReady ? 'Host IR 的逐 Pass 比较首先在 ExpandMixedKernel 记录为 mismatch。' : '编译数值校验证据不完整。',
      ['Pass', '状态', '最大绝对误差', '最大相对误差'], sourceReady ? [[first.name, first.status, first.maxAbs, first.maxRel]] : [],
      'Run #109 逐 Pass 数值校验记录', 'Host IR per-pass validation', !sourceReady,
      action('run_109', 'compilation', 'pass', firstName || null));
    refs['compiler-tolerance'] = evidence('compiler-tolerance', '比较容差', sourceReady ? '使用本次编译校验记录的容差。' : '容差未采集。',
      ['rtol', 'atol'], sourceReady ? [[fixture.tolerance.rtol, fixture.tolerance.atol]] : [],
      'Run #109 逐 Pass 数值校验记录', 'Host IR validation', !sourceReady,
      action('run_109', 'compilation', 'pass', firstName || null));
    refs['adjacent-pass'] = evidence('adjacent-pass', '相邻 Pass', adjacent ? '首个分歧前的相邻 Pass 已记录。IR 变换细节仍需精确比对。' : '缺少首个分歧前的相邻 Pass。',
      ['Pass', '状态'], adjacent ? [[adjacent.name, adjacent.status], [text(first?.name), text(first?.status)]] : [],
      'Run #109 逐 Pass 数值校验记录', 'ExpandMixedKernel 相邻 Pass', !adjacent,
      action('run_109', 'compilation', 'pass', firstName || null));
    refs['device-not-evaluated'] = evidence('device-not-evaluated', '设备执行', '该 Run 未进入设备执行；不能从本调查归因运行时或设备。',
      ['设备执行'], [['未执行']], 'Run #109 执行阶段记录', '设备阶段', false, null);
    refs.structure = evidence('structure', '结构校验', structural === 'pass' ? '结构校验通过；不代表数值语义等价。' : '结构校验记录缺失。', ['检查', '状态'], [['结构', structural || '未采集']], 'Run #109 编译结构校验', '已检查 IR 结构', structural !== 'pass', action('run_109', 'compilation', 'pass', firstName || null));
    const nodes = [
      node('structure', 'reasoning', structural === 'pass' ? 'normal' : 'unresolved', '结构校验', structural === 'pass' ? '通过' : '未采集', '结构有效不代表数值等价。', ['structure']),
      node('compiler-validation', 'finding', sourceReady ? 'abnormal' : 'unresolved', 'Host IR 数值校验出现分歧', sourceReady ? text(first?.status) : '校验未采集', '依据本次 Host IR 校验记录。', ['compiler-numerical-validation', 'compiler-tolerance']),
      node('expand-mixed-kernel', 'finding', sourceReady ? 'abnormal' : 'unresolved', 'ExpandMixedKernel 是首个记录分歧', sourceReady ? text(firstName) : '首个 Pass 未采集', '首个分歧定位不等于已知具体变换。', ['compiler-numerical-validation']),
      node('adjacent-pass-ir', 'reasoning', adjacent ? 'normal' : 'unresolved', '相邻 Pass IR 对比', adjacent ? text(adjacent.name) + ' → ' + text(firstName) : '相邻 Pass 未采集', '需要对这一区间的精确变换做 IR diff。', ['adjacent-pass']),
      node('device-scope', 'entity', 'unresolved', '设备阶段未执行', '未执行', '没有设备证据，不能形成运行时诊断。', ['device-not-evaluated']),
      node('compiler-diagnosis', 'diagnosis', sourceReady && adjacent ? 'unresolved' : 'unresolved', 'Compiler 阶段诊断待精确变换验证', sourceReady ? '首个分歧后继续比对 IR' : '数值证据不完整', '当前仅定位到编译阶段和 Pass 边界。', ['compiler-numerical-validation', 'adjacent-pass', 'device-not-evaluated'])
    ];
    const edges = [
      edge('compiler-validation', 'structure', 'checked', '结构检查'),
      edge('compiler-validation', 'device-scope', 'unresolved', '未执行'),
      edge('compiler-validation', 'expand-mixed-kernel', 'locate', '首个记录分歧'),
      edge('expand-mixed-kernel', 'adjacent-pass-ir', 'continue', '检查相邻 IR'),
      edge('adjacent-pass-ir', 'compiler-diagnosis', 'diagnose', '精确变换待验证'),
      edge('device-scope', 'compiler-diagnosis', 'unresolved', '设备未评估')
    ];
    return { nodes, edges, evidenceById: refs, summary: sourceReady ? 'Host IR 数值校验首先在 ExpandMixedKernel 出现分歧；应比较相邻 Pass 的精确 IR 变换。' : 'Run #109 的编译校验证据不完整。', judgment: sourceReady ? '偏差定位到 Compiler 阶段，具体错误变换待检查 IR Diff。' : '编译校验证据不足，暂不能确认首个分歧 Pass。', nextStep: '对 ExpandMixedKernel 与其相邻 Pass 的 IR 做精确 diff，并保留 Host IR 数值比较上下文。', action: action('run_109', 'compilation', 'pass', firstName || null) };
  }

  function build(runId) {
    if (runId === 'run_106') return profile106() ? build106(profile106()) : null;
    if (runId === 'run_109') return window.PTO_CORRECTNESS_DIAGNOSTICS?.profiles?.run_109 ? build109(fixture109()) : null;
    return null;
  }

  function resolveAction(runId, candidate) {
    if (!candidate || candidate.runId !== runId || !TABS.has(candidate.tab) || !KINDS.has(candidate.kind)) return null;
    const allowedTarget = (candidate.kind === 'tensor' && candidate.tab === 'correctness') ||
      (candidate.kind === 'task' && candidate.tab === 'execution') ||
      (candidate.kind === 'buffer' && candidate.tab === 'execution') ||
      (candidate.kind === 'pass' && candidate.tab === 'compilation');
    if (!allowedTarget) return null;
    const investigation = build(runId);
    if (!investigation) return null;
    const profile = runId === 'run_106' ? profile106() : null;
    const fixture = runId === 'run_109' ? fixture109() : null;
    const id = candidate.id == null ? null : String(candidate.id);
    const hasEntity = runId === 'run_106' && (
      (candidate.kind === 'tensor' && asArray(profile?.tensors).some(t => t.id === id)) ||
      (candidate.kind === 'task' && asArray(profile?.runtime?.tasks).some(t => t.id === id)) ||
      (candidate.kind === 'buffer' && profile?.runtime?.timeline?.buffer === id) ||
      (candidate.kind === 'pass' && id === null)
    ) || runId === 'run_109' && candidate.kind === 'pass' && asArray(fixture?.passes).some(p => p.name === id);
    if (candidate.id !== null && candidate.id !== undefined && !hasEntity) return null;
    if (!hasEntity || (candidate.kind === 'pass' && candidate.tab !== 'compilation')) return null;
    const normalized = { runId, tab: candidate.tab, kind: candidate.kind, id };
    if (candidate.taskIds) { if (!Array.isArray(candidate.taskIds) || !candidate.taskIds.every(id => asArray(profile?.runtime?.tasks).some(t => t.id === String(id)))) return null; normalized.taskIds = candidate.taskIds.map(String); }
    if (candidate.range) { const {from,to} = candidate.range; if (!Number.isFinite(from) || !Number.isFinite(to) || from >= to) return null; normalized.range = {from,to}; }
    return normalized;
  }

  window.PTO_RUN_INVESTIGATION = Object.freeze({ build, resolveAction });
}());
