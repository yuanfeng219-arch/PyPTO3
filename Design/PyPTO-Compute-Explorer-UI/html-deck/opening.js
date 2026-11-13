(function (global) {
  'use strict';

  const axes = [
    ['计算','Graph','做什么、依赖什么'],
    ['数据','Assignment','处理哪些数据、如何划分与分配'],
    ['时间','Pipeline','什么时候做、如何协同'],
  ];
  const scenes = [
    {
      title:'一份冲印订单 · 128 张照片',
      labels:['工序依赖','4 批 × 32 张','各工作位的交接'],
      details:['每批照片沿同一组工序推进。','照片按批交给各工序的工作位。','同一工作位依次处理各批，不同工作位可以同时处理不同批次。'],
      caption:'知道工序之后，还要知道照片怎么分批、各批什么时候推进。',
      stages:['选片','修图','冲印','交付'],
    },
    {
      title:'C = (A+B) ⊙ A · 逐元素计算',
      labels:['运算依赖','A、B 各 128 元素 · 对齐切分','Tile 的逻辑推进'],
      details:['A 同时输入 Add 和 Multiply。Add 的输出也是 Multiply 的输入。','A、B 按相同范围切分。每组 Tile 处理 32 个位置，写回 C 的对应范围。','同一 Tile 先加后乘。资源和调度条件允许时，不同 Tile 可以交错。'],
      caption:'从一份订单到一个算子，仍然要看工作依赖、数据范围和推进顺序。',
      stages:['Add','Multiply'],
    },
    {
      title:'接手真实算子 · 在运行前还原一次计算',
      labels:['通过阅读分析源码还原计算过程','推导数据范围和归属','手工推演执行过程'],
      evidence:[
        null,
        ['Triton-Viz/Excel手工Tiling','观察局部 Tile 访问区域，辅助理解分块','triton-viz.png','Triton-Viz 张量视图中高亮的数据访问区域'],
        ['手排流水','推演执行顺序及哪些步骤可以重叠','manual-pipeline-scheduling.png','手排流水图，包含任务创建、任务执行和流水排布'],
      ],
      caption:'这些信息可以分别获取，但开发者仍需要手工还原并串联，才能理解同一次计算。',
    },
  ];
  const judgments = [
    {
      label:'开发者承担的认知工作',
      names:['还原','关联','推演'],
      details:['从代码中提取并解释计算规则。','在三个视角中找到同一次计算的对应对象。','输入与参数变化后，重新推导计算组织。'],
      foot:'每次换证据、换视角或换输入，都可能需要重新建立这些关系。',
    },
    {
      label:'希望开发者能够做到',
      names:['降低理解门槛','建立认知连续性','支持主动推演'],
      details:['直接观察计算结构和执行过程。','沿着同一对象切换视角或深入细节。','带入输入、比较不同参数下的计算组织。'],
      foot:'按问题拆开理解，再借助同一对象把三种观察接起来。',
    },
    {
      label:'Compute Explorer 的设计策略',
      names:['算法机制可感知','共享计算上下文与关联交互','主动计算推演'],
      details:['把规则呈现为空间分配、计算结构和动态流水。','切换视角或下钻时，沿用同一 Task 与输入配置。','更改输入参数，比较分块、逻辑轮次和流水怎样变化。'],
      foot:'接下来从任务分配开始，依次看算法表达、跨视角追踪和参数推演。',
    },
  ];
  const batchColors = ['#83a5ee','#6ad9a2','#d8a767','#b6a1e8'];
  const steps = {
    code:['代码来源','认知还原','执行规则','硬件资源','结论'],
    scenes:['01 冲印店','02 数学计算','03 工具现状'],
    judgment:['认知负担','设计目标','设计策略'],
  };

  function graph(stages) {
    const photo = stages.length === 4;
    const node = (x,y,label) => `<g><rect x="${x-47}" y="${y-17}" width="94" height="34" rx="5"/><text x="${x}" y="${y}">${label}</text></g>`;
    const nodes = photo
      ? stages.map((name,i) => node(215,26+i*59,name)).join('')
      : [[134,26,'A'],[295,26,'B'],[215,85,'Add'],[215,144,'Multiply'],[215,203,'C']].map(args => node(...args)).join('');
    const links = photo
      ? [43,102,161].map(y => `M215 ${y}v25 m-4 -5l4 5 4 -5`).join(' ')
      : 'M134 43v12h81v13m-4 -5l4 5 4 -5 M295 43v12h-80 M134 43v101h34m-5 -4l5 4 -5 4 M215 102v25m-4 -5l4 5 4 -5 M215 161v25m-4 -5l4 5 4 -5';
    const label = photo ? stages.join(' → ') : 'A 和 B 输入 Add，Add 的结果与 A 输入 Multiply，输出 C';
    return `<svg class="opening-graph" viewBox="0 0 430 230" role="img" aria-label="${label}"><path d="${links}"/>${nodes}</svg>`;
  }

  function tiles(photo) {
    const cells = '<i></i>'.repeat(32);
    return `<div class="opening-tiles" aria-label="${photo ? '4 批照片，每批 32 张' : '4 个 Tile，每块 32 个元素'}">${batchColors.map((color,i) =>
      `<div class="opening-tile" data-reveal style="--batch:${color}"><b>${photo ? '批次 ' + (i+1) : 'Tile ' + (i+1) + ' · [' + i*32 + ', ' + (i+1)*32 + ')'}</b><div class="opening-cells" aria-hidden="true">${cells}</div></div>`
    ).join('')}</div>`;
  }

  function timeline(stages) {
    return `<div class="opening-timeline" aria-label="四组工作依序交接并交错推进">${stages.map((name,lane) =>
      `<div class="opening-lane"><span>${name}</span><div>${batchColors.map((color,i) =>
        `<b data-reveal data-delay="${(i+lane)*100}" style="--slot:${i+lane+1};--batch:${color}">${stages.length === 4 ? '批' : 'T'}${i+1}</b>`
      ).join('')}</div></div>`
    ).join('')}<div class="opening-time">逻辑顺序 →</div></div>`;
  }

  function renderScene(stage, index) {
    const scene = scenes[index];
    const figures = scene.stages
      ? [graph(scene.stages), tiles(index === 0), timeline(scene.stages)]
      : scene.evidence.map(item => {
        if (!item) return '';
        const [name,detail,asset,alt] = item;
        return `<div class="opening-evidence" data-reveal><strong>${name}</strong><small>${detail}</small><img src="assets/tool-references/${asset}" alt="${alt}" draggable="false"></div>`;
      });
    stage.innerHTML = `<h2 data-reveal>${scene.title}</h2><div class="opening-columns">${figures.map((figure,i) =>
      `<div data-axis="${i}"><h3 data-reveal>${scene.labels[i]}</h3>${scene.evidence ? figure : `<div class="opening-figure" data-reveal>${figure}</div>`}${scene.details ? `<p>${scene.details[i]}</p>` : ''}</div>`
    ).join('')}</div>`;
    return scene.caption;
  }

  function mount(slide,{onChange}) {
    const mode = slide.dataset.opening;
    const last = steps[mode].length - 1;
    const axesRow = slide.querySelector('[data-opening-axes]');
    const flowElements = [...slide.querySelectorAll('[data-flow-step]')];
    const stage = slide.querySelector('[data-opening-stage]');
    const caption = slide.querySelector('.opening-caption');
    const overlay = slide.querySelector('.opening-overlay');
    const controls = slide.querySelector('[data-opening-controls]');
    let beat = 0;
    let closed = false;
    let paused = false;
    let animations = [];

    if (axesRow) axesRow.innerHTML = axes.map(([name,view,question],i) =>
      `<div data-axis="${i}"><b>${name}</b><small>/ ${view}</small><p>${question}</p></div>`
    ).join('');

    function button(label, action) {
      const element = document.createElement('button');
      element.type = 'button';
      element.textContent = label;
      element.onclick = action;
      controls.append(element);
      return element;
    }
    const previous = button('上一段', () => set(beat-1));
    const tabs = steps[mode].map((label,index) => button(label, () => set(index,{replay:true})));
    const next = button('下一段', () => advance(1));
    const pauseButton = button('暂停动画', () => {
      if (paused) animations.forEach(animation => animation.play());
      else animations.filter(animation => animation.playState === 'running').forEach(animation => animation.pause());
      paused = !paused;
      updateControls();
    });
    const replay = button('重放', () => set(beat,{replay:true}));
    const toggleOverlay = overlay && button('关闭判断', () => {
      if (closed) set(beat);
      else close();
    });

    function updateControls() {
      tabs.forEach((tab,index) => {
        if (!closed && index === beat) tab.setAttribute('aria-current','step');
        else tab.removeAttribute('aria-current');
      });
      previous.disabled = closed || beat === 0;
      next.disabled = !closed && beat === last;
      pauseButton.disabled = closed || !animations.some(animation => ['running','paused'].includes(animation.playState));
      if (pauseButton.textContent !== (paused ? '继续动画' : '暂停动画')) pauseButton.textContent = paused ? '继续动画' : '暂停动画';
      replay.disabled = closed;
      if (toggleOverlay) toggleOverlay.textContent = closed ? '打开判断' : '关闭判断';
    }

    function cancel() {
      animations.forEach(animation => animation.cancel());
      animations = [];
      paused = false;
    }

    function paint() {
      slide.dataset.beat = beat;
      if (mode === 'code') {
        flowElements.forEach(element =>
          element.classList.toggle('is-pending',Number(element.dataset.flowStep) > beat)
        );
        return flowElements.filter(element => Number(element.dataset.flowStep) === beat);
      }
      if (mode === 'scenes') {
        caption.textContent = renderScene(stage,beat);
        return [...stage.querySelectorAll('[data-reveal]')];
      }
      if (!stage.childElementCount) caption.textContent = renderScene(stage,2);
      slide.dataset.overlay = closed ? 'closed' : 'open';
      overlay.hidden = closed;
      if (closed) return [];
      const judgment = judgments[beat];
      overlay.querySelector('.opening-stage-label').textContent = judgment.label;
      const grid = overlay.querySelector('.opening-judgment-grid');
      grid.innerHTML = judgment.names.map((name,i) => `<div data-axis="${i}"><small>${beat === 0 ? '' : judgments[0].names[i]}</small><h2>${name}</h2><p>${judgment.details[i]}</p></div>`).join('');
      overlay.querySelector('.opening-judgment-foot').textContent = judgment.foot;
      return [overlay.querySelector('.opening-stage-label'),...grid.children,overlay.querySelector('.opening-judgment-foot')];
    }

    function render(animate) {
      cancel();
      const elements = paint();
      if (animate && document.body.dataset.deckRole !== 'preview' && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
        animations = elements.map((element,index) => element.animate(
          [{opacity:.08,transform:'translateY(8px)'},{opacity:1,transform:'translateY(0)'}],
          {duration:270,delay:Number(element.dataset.delay ?? Math.min(index,5)*38),easing:'ease-out',fill:'backwards'}
        ));
        Promise.allSettled(animations.map(animation => animation.finished)).then(updateControls);
      }
      updateControls();
    }

    function set(index,{context,quiet=false,animate=true,replay=false} = {}) {
      if (!Number.isInteger(index) || index < 0 || index > last) return false;
      const nextClosed = !!overlay && !!context?.closed;
      if (index === beat && closed === nextClosed && !replay) return true;
      beat = index;
      closed = nextClosed;
      render(animate);
      if (!quiet) onChange();
      return true;
    }

    function advance(direction) {
      return closed ? direction > 0 && set(beat) : set(beat+direction);
    }

    function close() {
      return !!overlay && !closed && set(beat,{context:{closed:true}});
    }

    return {
      get beat() { return closed ? -1 : beat; },
      get context() { return overlay ? {closed,resumeBeat:beat} : null; },
      last,
      enter(index,context) {
        beat = context?.closed ? context.resumeBeat : index;
        closed = !!overlay && !!context?.closed;
        render(true);
      },
      set(index,options) {
        if (overlay && index === -1 && options?.context?.closed) return set(options.context.resumeBeat,options);
        return set(index,options);
      },
      advance,
      close,
      pause() { cancel(); updateControls(); },
      leave() { cancel(); },
    };
  }

  const notes = {
    'opening-context':'从 Ascend C 的单个 Kernel 观察范围，走到 PyPTO 3.0 的融合算子。右上用 Paged Attention 说明计算结构复杂化：前后阶段、循环和数据组织要一起理解。右下沿用硬件结构示意，强调开发者要跟踪的执行范围和约束扩大了。\n\n随着融合程度和执行组织复杂度提高，开发者需要理解整个计算如何组织、分配并执行。这对接手一份代码的人意味着什么？',
    'opening-code':'开发者常常从已有代码开始。代码可以来自 AI、已有算子资产、历史模板、开源仓库或示例。接手后，需要在脑中建立一遍这份代码可能怎样执行。先问它要做什么、依赖什么，再问工作作用于哪些数据、怎样切分，最后问各步何时推进、怎样协同。把这些规则放到有限的计算、存储和搬运资源里，才能理解执行如何组织。\n\n下一页用一份冲印订单，把这三个问题讲具体。',
    'opening-scenes':'先讲 128 张照片。工序有固定先后，分批后工作位能同时处理不同批次。观察同色批次如何沿着工序移动。切到逐元素算子，沿用相同形状表达 4 组 Tile 和可能的交错。逻辑推进图说明依赖允许的顺序。\n\n回到真实算子的运行前理解：通过阅读分析源码还原计算过程，用 Excel 推导数据范围和归属，借 Triton-Viz 的局部访问可视化理解类似的分块问题。再依据依赖和同步规则手排流水，判断等待与重叠。开发者需要把这些局部线索串联到同一次计算。',
    'opening-judgment':'先讲开发者承担的三项工作，再按同一位置对应三项目标，最后用后续演示落地。空间分配、计算图与动态流水展示算法机制。跨视角切换和逐层深入展示怎样追踪同一个 Task。参数演示展示如何带入输入，观察工作组织和逻辑轮次变化。\n\n三个视角按问题降低局部理解成本。共享对象与输入配置支持视角之间的定位和追踪，是连续探索的基础。下一页先看 Assignment 怎样让任务组织与归属可观察。',
  };

  global.PtoDeckOpening = {mount,notes};
})(window);
