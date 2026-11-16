(() => {
  // 每条亮点：[设计, 它做了什么、解决什么问题]。只讲设计与问题，不写具体交互手法。
  const highlights = {
    'agent.html': [
      ['生成过程透明可核对', 'Agent 生成算子时同步给出上下文、切分方案、编译与精度结果，解决 AI 生成代码“只有结果、无从判断”的问题。'],
      ['生成后衔接硬件验证', '编译通过、精度一致不等于硬件上成立；生成完成即引导进入 Operator Lens，检查任务分配与存储落点。']
    ],
    'lens.html': [
      ['从整层走进 Kernel', '在同一画布中从整层计算图下钻到 Kernel，结合源码理解循环、并行和分支结构。'],
      ['计算与存储对应查看', '展开存储落点，计算图与硬件分配上下对照；悬停对象时，同步高亮相关区域和源码。'],
      ['上板前查看任务分配', '沿 Work Item 的生成、排序和分配过程查看各 Lane 的工作量，辅助发现负载差异。有效与空块使用示例输入。']
    ],
    'workbench.html': [
      ['源码、计算图与硬件对齐', '同一对象在源码、计算图与芯片落点中一一对应，解决代码与硬件执行之间无从对照的问题。'],
      ['分工与落点上板前可见', '任务如何分给各核、数据落在哪块存储，编译后即可查看与评估，避免分工不均、核心空闲到上板才暴露。']
    ],
    'compile.html': [
      ['编译过程可视化回看', '将每个 Pass 的变换以图形与代码对照呈现，解决结果异常却无报错时，不清楚编译器做了什么的问题。'],
      ['问题定位到具体 Pass', '对照变换前后与诊断信息，把问题收敛到具体 Pass 与源码位置，减少逐层排查。']
    ],
    'loading.html': [
      ['编译上板过程透明', '编译、下发与执行的阶段和日志实时呈现，上板过程不再是黑盒。'],
      ['从原始数据到分析结论', '先呈现原始泳道，再提炼出关键路径进入分析，原始数据与分析结论之间可追溯。']
    ],
    'result.html': [
      ['先给结论，再给证据', '首屏直接指出最值得优化的任务及其对关键路径的影响，解决性能数据量大、不知从何看起的问题。'],
      ['按瓶颈类型给出优化方向', '自动识别负载不均、Setup 占比高等瓶颈特征，并按收益排序优化候选，解决“知道哪里慢，却不知先改什么”的问题。']
    ],
    'all-passes.html': [
      ['完整编译链回看', '按编译顺序查看各 Pass 的中间状态，把单步变换放回整条编译链中理解。'],
      ['相邻阶段对照', '对照相邻阶段的差异，确认每一步变换的作用与影响范围。']
    ]
  };

  const page = location.pathname.split('/').pop();
  const embeddedLens = page === 'workbench.html' && new URLSearchParams(location.search).get('embed') === 'lens';
  if (embeddedLens || !highlights[page]) return;

  const stylesheet = document.createElement('link');
  stylesheet.rel = 'stylesheet';
  stylesheet.href = new URL('demo-highlights.css', document.currentScript.src);
  document.head.append(stylesheet);

  const panel = document.createElement('details');
  panel.className = 'demo-highlights';
  panel.open = page !== 'lens.html';
  panel.dataset.page = page.replace('.html', '');
  const summary = document.createElement('summary');
  summary.innerHTML = '<span class="demo-highlights-mark" aria-hidden="true">✦</span><span class="demo-highlights-label">体验设计亮点<small>DESIGN NOTES</small></span><span class="demo-highlights-count">' + String(highlights[page].length).padStart(2, '0') + '</span><svg class="demo-highlights-chevron" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg>';

  const list = document.createElement('dl');
  for (const [title, description] of highlights[page]) {
    const heading = document.createElement('dt');
    heading.textContent = title;
    const text = document.createElement('dd');
    text.textContent = description;
    list.append(heading, text);
  }
  panel.append(summary, list);
  document.body.append(panel);
})();
