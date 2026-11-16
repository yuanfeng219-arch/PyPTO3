(() => {
  // 每条说明：[标题，正文或标签列表，可选图示]。
  const highlights = {
    'agent.html': [
      ['开发方式在变化', '开发者开始借助 Agent 生成和修改算子，从逐行编写转向描述需求、引导生成。'],
      ['理解与审阅成为新挑战', '代码生成更快，但面对数百上千行代码，开发者仍需看懂逻辑、判断实现是否合理。']
    ],
    'lens.html': [
      ['计算图', '从整层下钻到 Kernel，对照源码看循环和分支。', 'compute'],
      ['分工图', '五步看清数据怎么切、各核怎么分工、跑几轮。', 'assignment'],
      ['硅片落点图', '从源码追到芯片，看到数据存在哪、由谁计算。', 'placement']
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
      ['PyPTO Toolkit 泳道图', '在同一条时间轴上查看各核的任务、耗时、并行和空闲时段。新版把泳道图从 IDE 带进 Agent 工作台。'],
      ['落地局点', ['盘古', 'MY', '车BU']]
    ],
    'result.html': [
      ['关键路径', '高亮影响总耗时的任务依赖链。', 'critical-path'],
      ['信息减负', '淡化任务颜色，突出低并行度，细节按需展开。', 'declutter'],
      ['信息聚合', '把任务、核内数据和关键 Pass 串起来。', 'evidence']
    ],
    'all-passes.html': [
      ['完整编译链回看', '按编译顺序查看各 Pass 的中间状态，把单步变换放回整条编译链中理解。'],
      ['相邻阶段对照', '对照相邻阶段的差异，确认每一步变换的作用与影响范围。']
    ]
  };

  const page = location.pathname.split('/').pop();
  const embed = new URLSearchParams(location.search).get('embed');
  const embeddedLens = page === 'workbench.html' && embed === 'lens';
  const embeddedResult = page === 'result.html' && embed === 'agent' && window.parent !== window;
  if (embeddedLens || embeddedResult || !highlights[page]) return;

  if (!document.querySelector('link[data-demo-highlights-style]')) {
    const stylesheet = document.createElement('link');
    stylesheet.rel = 'stylesheet';
    stylesheet.href = new URL('demo-highlights.css', document.currentScript.src);
    document.head.append(stylesheet);
  }

  function createPanel(page) {
    const panel = document.createElement('details');
    panel.className = 'demo-highlights';
    panel.open = true;
    panel.dataset.page = page.replace('.html', '');
    const panelTitles = {
      'agent.html': '开发者的变化',
      'lens.html': '算子理解三张图',
      'loading.html': '泳道图与落地局点',
      'result.html': '泳道排查三步走'
    };
    const takeaways = {
      'lens.html': '算什么、怎么分、落在哪。',
      'result.html': '抓主线、看重点、追证据。'
    };
    const summary = document.createElement('summary');
    const title = panelTitles[page] ?? '体验设计亮点';
    const count = String(highlights[page].length).padStart(2, '0');
    summary.innerHTML = `
      <span class="demo-highlights-mark" aria-hidden="true">✦</span>
      <span class="demo-highlights-label">${title}<small>DESIGN NOTES</small></span>
      <span class="demo-highlights-count">${count}</span>
      <svg class="demo-highlights-chevron" width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg>`;

    const list = document.createElement('dl');
    for (const [title, description, diagram] of highlights[page]) {
      const heading = document.createElement('dt');
      heading.textContent = title;
      const text = document.createElement('dd');
      if (Array.isArray(description)) {
        const labels = document.createElement('ul');
        labels.className = 'demo-note-labels';
        for (const label of description) {
          const item = document.createElement('li');
          item.textContent = label;
          labels.append(item);
        }
        text.append(labels);
      } else {
        text.textContent = description;
      }
      if (diagram) {
        const row = document.createElement('div');
        row.className = 'highlight-row';
        const mark = document.createElement('span');
        mark.className = 'highlight-diagram ' + diagram;
        mark.setAttribute('aria-hidden', 'true');
        mark.innerHTML = '<i></i><i></i><i></i>';
        heading.prepend(mark);
        row.append(heading, text);
        list.append(row);
      } else {
        list.append(heading, text);
      }
    }
    if (page === 'agent.html') {
      const visual = document.createElement('div');
      visual.className = 'agent-change';
      visual.innerHTML = `
        <section class="agent-change-section">
          <p class="agent-change-label">开发方式</p>
          <div class="agent-change-flow">
            <div class="agent-change-step"><small>过去</small><strong>手写代码</strong></div>
            <span class="agent-change-arrow" aria-hidden="true">→</span>
            <div class="agent-change-step is-after"><small>Agent 时代</small><strong>描述需求，引导生成</strong></div>
          </div>
        </section>
        <section class="agent-change-section">
          <p class="agent-change-label">新的挑战</p>
          <div class="agent-challenge-line"><span>AI 写得快</span><span class="agent-change-arrow" aria-hidden="true">→</span><strong>人要看懂、审阅</strong></div>
          <p class="agent-challenge-note">几百上千行代码，如何快速理解和判断？</p>
        </section>`;
      panel.append(summary, visual);
    } else if (takeaways[page]) {
      const content = document.createElement('div');
      content.className = 'highlight-content';
      const takeaway = document.createElement('p');
      takeaway.className = 'highlight-takeaway';
      takeaway.textContent = takeaways[page];
      content.append(takeaway, list);
      panel.append(summary, content);
    } else {
      panel.append(summary, list);
    }
    return panel;
  }

  document.body.append(createPanel(page));
  if (page === 'loading.html') {
    // 卡片随 Agent 工作台挂载，定位使用外层视口。
    document.getElementById('analysisShell').content.firstElementChild.append(createPanel('result.html'));
  }
})();
