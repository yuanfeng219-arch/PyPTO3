/* ---- 源码扫描器 ------------------------------------------------------------
   给没有人工建模的文件现算一份 OperatorProfile 的原料：函数层级、阶段划分、
   以及编码期可静态判定的风险与语义护栏。

   **这是逐行文本扫描，不是 AST。** 浏览器里没有 Python 前端，所以这里只做
   在词法层面可靠的判断，且每条结论都写明它看到的是什么。真实产品里这一步
   应当由编译器前端产出——那时同样的 profile 字段可以直接换源。
   因此扫描出的风险一律标注「来自源码扫描」，不冒充编译事实。

   规范：算子属性面板统一规范 §七 风险 Lens · §十一 未建模文件 */
(function () {
  'use strict';

  const DECORATORS = [
    [/@pl\.jit\.incore\b/, 'incore', 'InCore kernel'],
    [/@pl\.jit\.inline\b/, 'inline', 'inline 子函数'],
    [/@pl\.jit\.host\b/, 'host', 'HOST orchestrator'],
    [/@pl\.jit\b/, 'orchestration', 'Orchestration 入口'],
    [/@pl\.function\s*\(\s*type\s*=\s*pl\.FunctionType\.InCore/, 'incore', 'InCore kernel'],
    [/@pl\.function\s*\(\s*type\s*=\s*pl\.FunctionType\.Orchestration/, 'orchestration', 'Orchestration 入口'],
    [/@pl\.inline\b/, 'inline', 'inline 子函数'],
    [/@pl\.function\b/, 'opaque', 'Opaque 函数'],
  ];

  const KIND_LABEL = {
    incore: 'incore', inline: 'inline', orchestration: 'orchestration',
    host: 'host', opaque: 'opaque', megakernel: 'megakernel', module: '模块',
  };

  function classifyDecorator(text) {
    for (const [re, kind, label] of DECORATORS) if (re.test(text)) return { kind, label };
    return null;
  }

  // 每个被装饰的函数就是一个天然的阶段——未建模文件由此直接获得源码联动，
  // 不需要任何人工标注。
  function buildScopes(lines) {
    const found = [];
    for (let i = 0; i < lines.length; i++) {
      const m = /^\s*def\s+([A-Za-z_][\w]*)\s*\(/.exec(lines[i]);
      if (!m) continue;
      // 往上找连续的装饰器行
      let deco = null;
      for (let j = i - 1; j >= 0 && j >= i - 6; j--) {
        const t = lines[j].trim();
        if (!t || t.startsWith('#')) continue;
        if (!t.startsWith('@')) break;
        deco = classifyDecorator(t) || deco;
      }
      found.push({ name: m[1], line: i + 1, kind: deco ? deco.kind : null, decoLabel: deco ? deco.label : '普通函数' });
    }
    const scopes = [];
    if (found.length && found[0].line > 1) {
      scopes.push({ id: 'module', label: '模块头', lines: [1, found[0].line - 1], detail: '导入、模块级常量与动态维声明' });
    }
    found.forEach((fn, idx) => {
      const end = idx + 1 < found.length ? found[idx + 1].line - 1 : lines.length;
      scopes.push({
        id: 'fn_' + idx + '_' + fn.name,
        label: fn.name,
        lines: [fn.line, end],
        detail: `${fn.decoLabel} · 第 ${fn.line}–${end} 行`,
      });
    });
    if (!scopes.length) scopes.push({ id: 'module', label: '全文', lines: [1, Math.max(1, lines.length)], detail: '未识别到函数定义' });
    return { scopes, functions: found };
  }

  const SCAN_NOTE = '本条来自<b>逐行源码扫描</b>（非 AST），只在词法层面成立；编译后应以 IR 为准。';

  /* ---- 检查项。宁可漏报，不可误报——每条都写清它看到的是什么。 ---- */

  function checkOutNeverWritten(lines, risks) {
    const outParams = [];
    lines.forEach((line, i) => {
      const m = /^\s*([A-Za-z_][\w]*)\s*:\s*pl\.Out\[/.exec(line);
      if (m) outParams.push({ name: m[1], line: i + 1 });
    });
    const body = lines.join('\n');
    for (const p of outParams) {
      // 保守规则：只有当这个名字在全文<b>只出现过一次</b>（就是它自己的声明）时才报。
      //
      // 更聪明的写法（找 pl.store / pl.assemble / 赋值）会误报：SSA 的 loop-carry
      // 会把参数改名——`pij_buf` 进 init_values 变成 `pij_buf_iter`，store 写的是
      // 后者，再 yield_ 成 `pij_buf_carry`。而 init_values + yield_ 正是 PyPTO 的
      // 惯用写法，对惯用代码误报比不报更糟。文本扫描证明不了"没被写"，
      // 只能证明"这个名字再没出现过"。
      const occurrences = (body.match(new RegExp(`\\b${p.name}\\b`, 'g')) || []).length;
      if (occurrences > 1) continue;
      risks.push({
        level: 'block', cls: '契约', lines: [p.line, p.line],
        title: `pl.Out 参数 ${p.name} 没有被写过`,
        why: `第 ${p.line} 行把 <code>${p.name}</code> 声明为 <code>pl.Out[...]</code>，而这个名字在<b>全文只出现这一次</b>——函数体里再没碰过它。`,
        impact: '输出缓冲区内容未定义，而编译和运行都不会报错——这正是 pypto-lib #294「多个 pl.Out 只有第一个写回」的形态。',
        fix: `为 <code>${p.name}</code> 补上写回，或者它本就不该是 <code>Out</code>——改成 <code>In</code> 让契约说实话。`,
        verify: `编译后检查该参数的 ArgDirection；运行时把缓冲区预填哨兵值，确认它被覆盖。<br><small>${SCAN_NOTE}</small>`,
      });
    }
  }

  function checkTensorLayout(lines, risks) {
    lines.forEach((line, i) => {
      // 注意不能用 [^\]]*：pl.Tensor[[K, N], ...] 里有内层 ]，会提前截断。
      if (/pl\.Tensor\[[^\n]*pl\.DN\b/.test(line)) {
        risks.push({
          level: 'warn', cls: '类型布局', lines: [i + 1, i + 1],
          title: '使用了已弃用的 pl.Tensor[..., pl.DN]',
          why: `第 ${i + 1} 行在 Tensor annotation 上写了 layout 标记 <code>pl.DN</code>。RFC #1300 已弃用这种写法，解析期会触发 DeprecationWarning。`,
          impact: 'layout-only 简写迫使读者同时持有 IR 逻辑视图与 runtime 行优先两套坐标系，正是该 RFC 想消除的歧义来源。',
          fix: '去掉 layout 标记、直接写 runtime shape；matmul 的 Bᵀ 场景改用 <code>pl.matmul(..., b_trans=True)</code>，或 load 后用 <code>pl.tile.transpose_view(...)</code>。',
          verify: `解析该文件，确认 DeprecationWarning 消失。<br><small>${SCAN_NOTE}</small>`,
        });
      }
      if (/pl\.Tensor\[[^\n]*pl\.NZ\b/.test(line)) {
        risks.push({
          level: 'block', cls: '类型布局', lines: [i + 1, i + 1],
          title: 'pl.NZ 用作 TensorType annotation',
          why: `第 ${i + 1} 行把 <code>pl.NZ</code> 写进了 <code>pl.Tensor[...]</code>。NZ 是硬件 tile layout，<b>只允许出现在 <code>pl.Tile[...]</code> 上</b>。`,
          impact: '解析期即报错，编译无法进行。',
          fix: '把该 annotation 改成 <code>pl.Tile[..., pl.NZ]</code>；若它确实是 DDR 上的 Tensor，则去掉 NZ。',
          verify: `重新解析该文件。<br><small>${SCAN_NOTE}</small>`,
        });
      }
    });
  }

  function checkManualScopeWithoutDeps(lines, risks) {
    const manualLines = [];
    lines.forEach((line, i) => { if (/pl\.manual_scope\s*\(|pl\.scope\s*\(\s*mode\s*=\s*pl\.ScopeMode\.MANUAL/.test(line)) manualLines.push(i + 1); });
    if (!manualLines.length) return;
    const body = lines.join('\n');
    if (/\bdeps\s*=/.test(body)) return;
    risks.push({
      level: 'block', cls: '依赖', lines: [manualLines[0], manualLines[0]],
      title: 'manual_scope 内没有任何显式依赖边',
      why: `第 ${manualLines[0]} 行进入了 MANUAL scope——区域内 runtime <b>不做自动依赖跟踪</b>——但全文没有出现任何 <code>deps=</code>。`,
      impact: '区域内所有任务之间没有声明顺序，调度可以任意重排；表现为偶发的错值或竞态，而不是编译错误。',
      fix: '用 <code>pl.submit(..., deps=[tid])</code> 或 <code>pl.at(..., deps=[...]) as tid</code> 把需要的排序边显式声明出来；不需要接管顺序的话就不要进 manual_scope。',
      verify: `编译后读依赖图，确认区域内任务的 fanin 不为空。<br><small>${SCAN_NOTE}</small>`,
    });
  }

  // 语义护栏：带外承诺。删了照样编译通过，然后静默错值——这是读代码读不出来的信息。
  const GUARD_PATTERNS = [
    [/pl\.no_dep\s*\(/, 'pl.no_dep(arg)',
      '该调用点的这个参数槽位不进入自动依赖跟踪；调用方在带外承诺此处不存在 RaW / WaW / WaR 冲突。',
      '所有本应与该槽位形成依赖的下游 task，其正确性都建立在这个承诺上。',
      '退回自动依赖跟踪：多出一条保守边（性能下降），或在写偏移数据相关时反而排不出正确顺序。'],
    [/manual_dep\s*=\s*True/, 'pl.create_tensor(..., manual_dep=True)',
      '该 tensor <b>整个生命周期</b>跳过 OverlapMap 的查找与写入，不受 scope 影响。',
      '任何读写该 tensor 的 task，顺序完全由显式边决定。',
      '该 tensor 重新进入自动跟踪，可能与显式边产生重复或冲突的排序。'],
    [/no_dep_args\s*=/, 'pl.at(..., no_dep_args=[...])',
      '列出的 tensor 在合成 kernel call 上被标为 <code>NoDep</code>，等价于在调用点写 <code>pl.no_dep(...)</code>。',
      '该 pl.at 块的下游消费者。',
      '这些槽位回到自动跟踪，outliner 推断出的 In/Out/InOut 方向重新生效。'],
  ];

  function collectGuards(lines) {
    const guards = [];
    lines.forEach((line, i) => {
      for (const [re, api, promise, dependents, ifRemoved] of GUARD_PATTERNS) {
        if (!re.test(line)) continue;
        guards.push({
          lines: [i + 1, i + 1], api,
          promise: `${promise}<br><small>源码原文：<code>${line.trim().slice(0, 90)}</code></small>`,
          dependents, ifRemoved,
        });
      }
    });
    return guards;
  }

  function scan(source, path) {
    const lines = String(source || '').split('\n');
    const { scopes, functions } = buildScopes(lines);
    const risks = [];
    checkOutNeverWritten(lines, risks);
    checkTensorLayout(lines, risks);
    checkManualScopeWithoutDeps(lines, risks);
    const guards = collectGuards(lines);

    const kinds = new Set(functions.map((f) => f.kind).filter(Boolean));
    let kind = 'module';
    if (kinds.size > 1 && kinds.has('orchestration')) kind = 'megakernel';
    else if (kinds.size) kind = [...kinds][0];

    const plCalls = (String(source || '').match(/\bpl\.[a-z_]+\s*\(/g) || []).length;
    const decorated = functions.filter((f) => f.kind).length;

    return {
      lineCount: lines.length,
      functions, scopes, risks, guards,
      kind, kindLabel: KIND_LABEL[kind] || kind,
      plCalls, decorated,
      // 跑过的检查项数，用于"静态检查通过 · n 项"
      checkCount: 3 + GUARD_PATTERNS.length,
      summary: decorated
        ? `${decorated} 个带 PyPTO 装饰器的函数 · ${functions.length} 个函数定义 · ${plCalls} 处 pl.* 调用 · ${lines.length} 行。`
        : `未识别到 PyPTO 装饰器；${functions.length} 个函数定义 · ${plCalls} 处 pl.* 调用 · ${lines.length} 行。`,
      path,
    };
  }

  window.PtoOperatorScan = { scan };
})();
