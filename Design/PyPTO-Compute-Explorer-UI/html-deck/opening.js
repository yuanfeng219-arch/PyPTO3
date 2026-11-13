(function (global) {
  'use strict';

  const axes = [
    ['计算','Graph','做什么、依赖什么'],
    ['数据','Assignment','处理哪些数据、如何划分与分配'],
    ['执行时间','Pipeline','什么时候做、如何协同'],
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
      labels:['运算与依赖','分块范围与归属','执行顺序与重叠'],
      evidenceDetails:['对照源码、IR 和计算图，梳理运算依赖','用 Excel 推导分块范围，用 Triton-Viz 查看局部 Tile 访问','依据依赖和同步规则，手工推演等待与重叠'],
      caption:'这些信息可以分别获取，但开发者仍需要手工还原并串联，才能理解同一次计算。',
    },
  ];
  const toolEvidence = [
    [['源码 / IR / 计算图','compute-ir.png','源码、IR 与计算图中的运算及其依赖']],
    [
      ['Excel 手工 Tiling','excel-tiling.svg','表格中的数据范围和任务分组'],
      ['Triton-Viz','triton-viz.png','Triton-Viz 张量视图中高亮的数据访问区域'],
    ],
    [['手排流水','manual-pipeline-scheduling.png','手排流水图，包含任务创建、任务执行和流水排布']],
  ];
  const judgmentStrategies = [
    {
      pain:'需要人工还原', expectation:'容易理解', title:'机制可感知',
      detail:'让抽象计算规则变得直观，降低人工还原成本。',
    },
    {
      pain:'对象难以关联', expectation:'连续追踪', title:'共享上下文',
      detail:'让同一计算对象跨视角连续可追踪。',
    },
    {
      pain:'变化难以推演', expectation:'主动探索', title:'交互式推演',
      detail:'让开发者主动观察输入和参数变化对计算组织的影响。',
    },
  ];
  const batchColors = ['#83a5ee','#6ad9a2','#d8a767','#b6a1e8'];
  const steps = {
    code:['已有代码','执行目标','认知缺口'],
    scenes:['01 冲印店','02 数学计算','03 运行前证据'],
    judgment:['设计判断','策略展开'],
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

  function evidenceFigures(items) {
    return items.map(([label,asset,alt]) =>
      `<figure><figcaption>${label}</figcaption><img src="assets/tool-references/${asset}" alt="${alt}" draggable="false"></figure>`
    ).join('');
  }

  function renderScene(stage, index) {
    const scene = scenes[index];
    const figures = scene.stages
      ? [graph(scene.stages), tiles(index === 0), timeline(scene.stages)].map(figure =>
        `<div class="opening-figure" data-reveal>${figure}</div>`)
      : toolEvidence.map((items,i) => `
        <div class="opening-evidence" data-reveal>
          <small>${scene.evidenceDetails[i]}</small>
          <div class="opening-evidence-figures">${evidenceFigures(items)}</div>
        </div>`);
    stage.innerHTML = `<h2 data-reveal>${scene.title}</h2><div class="opening-columns">${figures.map((figure,i) =>
      `<div data-axis="${i}"><h3 data-reveal>${scene.labels[i]}</h3>${figure}${scene.details ? `<p>${scene.details[i]}</p>` : ''}</div>`
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
    const controls = slide.querySelector('[data-opening-controls]');
    let beat = 0;
    let paused = false;
    let animations = [];

    const strategies = slide.querySelector('.opening-judgment-strategies');
    const challenges = slide.querySelector('.opening-judgment-challenges');
    if (strategies) {
      challenges.textContent = judgmentStrategies.map(({pain}) => pain).join('、');
      strategies.innerHTML = judgmentStrategies.map(({title,detail},index) => `
        <li data-delay="${index*125}">
          <span aria-hidden="true">0${index+1} /</span><h3>${title}</h3><p>${detail}</p>
        </li>
      `).join('');
    }

    if (axesRow) axesRow.innerHTML = axes.map(([name,view,question],i) =>
      `<div data-axis="${i}"><b data-opening-axis-title>${name}</b><small>/ ${view}</small><p>${question}</p></div>`
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
    button('重放', () => set(beat,{replay:true}));
    function updateControls() {
      tabs.forEach((tab,index) => {
        if (index === beat) tab.setAttribute('aria-current','step');
        else tab.removeAttribute('aria-current');
      });
      previous.disabled = beat === 0;
      next.disabled = beat === last;
      pauseButton.disabled = !animations.some(animation => ['running','paused'].includes(animation.playState));
      if (pauseButton.textContent !== (paused ? '继续动画' : '暂停动画')) pauseButton.textContent = paused ? '继续动画' : '暂停动画';
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
        return [...stage.querySelectorAll('[data-reveal]'), ...(beat === last ? [caption] : [])];
      }
      strategies.hidden = beat === 0;
      return beat === 0 ? [] : [...strategies.children];
    }

    function render(animate) {
      cancel();
      const elements = paint();
      if (animate && document.body.dataset.deckRole !== 'preview' && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
        animations = elements.map((element,index) => element.animate(
          [{opacity:.08,translate:'0 8px'},{opacity:1,translate:'0 0'}],
          {duration:270,delay:Number(element.dataset.delay ?? Math.min(index,5)*38),easing:'ease-out',fill:'backwards'}
        ));
        if (challenges && beat === 0) {
          animations.push(challenges.animate(
            [{opacity:0},{opacity:1,offset:.15},{opacity:1,offset:.65},{opacity:0}],
            {duration:1000,easing:'ease-in-out'}
          ));
        }
        Promise.allSettled(animations.map(animation => animation.finished)).then(updateControls);
      }
      updateControls();
    }

    function set(index,{quiet=false,animate=true,replay=false} = {}) {
      if (!Number.isInteger(index) || index < 0 || index > last) return false;
      if (index === beat && !replay) return true;
      beat = index;
      render(animate);
      if (!quiet) onChange();
      return true;
    }

    function advance(direction) { return set(beat+direction); }

    return {
      get beat() { return beat; },
      last,
      enter(index) {
        beat = index;
        render(true);
      },
      set,
      advance,
      pause() { cancel(); updateControls(); },
      leave() { cancel(); },
    };
  }

  let axisFlights = [];

  function prepareAxisTransition(from,to) {
    axisFlights.forEach(animation => animation.cancel());
    axisFlights = [];
    if (from.dataset.opening !== 'code' || from.dataset.beat !== '2' ||
        to.dataset.opening !== 'scenes' || document.body.dataset.deckRole === 'preview' ||
        matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const sourceLabels = [...from.querySelectorAll('[data-opening-axis-title]')].map(label => ({
      text:label.textContent,
      rect:label.getBoundingClientRect(),
      color:getComputedStyle(label).color,
    }));
    return () => {
      to.querySelectorAll('[data-opening-axis-title]').forEach((label,index) => {
        const source = sourceLabels[index];
        const target = label.getBoundingClientRect();
        if (!source.rect.width || !target.width) return;
        const floating = document.createElement('b');
        floating.className = 'opening-axis-in-flight';
        floating.textContent = source.text;
        floating.setAttribute('aria-hidden','true');
        Object.assign(floating.style,{
          left:target.left+'px',top:target.top+'px',color:source.color,
          fontFamily:getComputedStyle(label).fontFamily,fontSize:target.height+'px',
          lineHeight:'1',fontWeight:getComputedStyle(label).fontWeight,
        });
        document.body.append(floating);
        const size = floating.getBoundingClientRect();
        const duration = 520;
        const flight = floating.animate([
          {transform:`translate(${source.rect.left-target.left}px,${source.rect.top-target.top}px) scale(${source.rect.width/size.width})`},
          {transform:`scale(${target.width/size.width})`},
        ],{duration,easing:'cubic-bezier(.2,.75,.25,1)'});
        const reveal = label.animate([{opacity:0},{opacity:0}],{duration});
        axisFlights.push(flight,reveal);
        const remove = () => floating.remove();
        flight.finished.then(remove,remove);
      });
    };
  }

  const notes = {
    'opening-context':'从 Ascend C 的单个 Kernel 观察范围，走到 PyPTO 3.0 的融合算子。右上用 Paged Attention 说明计算结构复杂化：前后阶段、循环和数据组织要一起理解。右下沿用硬件结构示意，强调开发者要跟踪的执行范围和约束扩大了。\n\n随着融合程度和执行组织复杂度提高，开发者需要理解整个计算如何组织、分配并执行。这对接手一份代码的人意味着什么？',
    'opening-code':'开发者可能从 AI、示例、团队资产或框架迁移中拿到一份已有代码。先让观众认出输入，再揭示执行目标。右侧参考昇腾 910 的计算芯粒、I/O、HBM 与片上互连关系，按演示需要简化。单元数量与局部状态用于说明资源约束。\n\n执行如何组织仍需要开发者还原：计算做了什么，工作如何拆分与映射，各项工作何时推进、如何协同。下一页用冲印店和逐元素运算讲清这三个问题。',
    'opening-scenes':'先讲 128 张照片。工序有固定先后，分批后工作位能同时处理不同批次。观察同色批次如何沿着工序移动。切到逐元素算子，沿用相同形状表达 4 组 Tile 和可能的交错。逻辑推进图说明依赖允许的顺序。\n\n回到真实算子的运行前理解：通过阅读分析源码还原计算过程，用 Excel 推导数据范围和归属，借 Triton-Viz 的局部访问可视化理解类似的分块问题。再依据依赖和同步规则手排流水，判断等待与重叠。开发者需要把这些局部线索串联到同一次计算。',
    'opening-judgment':[
      '上页展示了理解同一次计算时需要人工串联的信息。这里提出一套连续的计算探索体验，让开发者能看懂规则、追踪对象，并观察条件变化带来的影响。',
      judgmentStrategies.map(({pain,expectation,title}) => `${pain} → ${expectation} → ${title}`).join('\n'),
      '这三个策略可以在不同视角配合使用。下一页从任务分配入手，看开发者如何直接观察任务的生成与归属。',
    ].join('\n\n'),

  };

  global.PtoDeckOpening = {mount,notes,prepareAxisTransition};
})(window);
