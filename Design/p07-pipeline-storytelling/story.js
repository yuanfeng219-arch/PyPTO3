(() => {
  'use strict';

  const sceneMarkup = `
      <header class="mast"><span>PyPTO Compute Explorer</span></header>
      <div class="overview-copy"><h1>执行流水的动态演绎</h1><p>一次读取要同时满足数据、同步和资源约束。开发者原本要从不同代码段还原这些关系。</p></div>
      <div class="timeline viz2-intra" aria-label="Task 完整逻辑流水"></div>
      <section class="closeup" aria-label="当前关系特写" hidden>
        <figure><div class="cut viz2-intra"></div><figcaption data-cut-label></figcaption></figure>
        <div class="annotation">
          <p class="locator" data-locator></p>
          <h2 data-heading></h2>
          <p data-explanation></p>
          <div class="evidence"><b data-event aria-live="polite"></b><small data-source></small></div>
        </div>
      </section>
      <p class="map-note" data-map-note></p>
      <p class="takeaway" data-takeaway hidden>从手工排布，走到可连续观察的执行协同。</p>
      <p class="boundary">Logical Tick 表示源码循环位置。WAIT 演绎的是就绪条件，实际停顿和耗时需要运行记录。</p>
      <div class="source-renderer" hidden></div>
      <p data-error hidden role="alert">无法加载原 Pipeline。请确认本目录与 PyPTO-Compute-Explorer-UI 保持在同一级。</p>
`;

  function mount(story,transport,{onChange = () => {}} = {}) {
    story.innerHTML = sceneMarkup;
    const model = window.PtoPagedAttentionIntro.model;
    const graph = window.PTO_PYPA_SEMANTIC_GRAPH;
    const value = name => graph.constants[name].value;
    const anatomy = window.PtoVisualization2Anatomy.createModel({
      seqLens:model.defaultSeqLens, blocks:value('ATTN_SPMD_BLOCKS'), numHeads:value('NUM_HEADS'),
      kvHeads:value('NUM_KV_HEADS'), headDim:value('HEAD_DIM'), rowTile:value('ROW_TILE'),
      pageTokens:value('BLOCK_SIZE'), pagesPerStack:value('STACK_PAGES'), transferSlots:value('TRANSFER_SLOTS')
    });
    const pipelineAPI = window.PtoVisualization2PipelineIntra;
    const pipeline = pipelineAPI.createModel({anatomy,preLaunch:value('PRE_LAUNCH')});
    const task = pipeline.tasks[48];
    const evidence = window.PtoPagedAttentionStackDependencyModel.build(model.pipelineFor(task.seqLen),graph.synchronizations);
    const names = {qk:'QK',softmax:'Softmax',pv:'PV',accumulate:'Accumulate'};
    const scenario = window.PtoComputeScenario.create(task);
    const source = story.querySelector('.source-renderer');
    source.id = 'story-native-pipeline';
    const native = pipelineAPI.mount(source,{
      model:pipeline, scenario:{get:()=>scenario,limits:()=>window.PtoComputeScenario.limits(scenario),task:()=>null},
      onCamera() {}
    });
    native.update(task.id,null,null,{active:true});
    const original = source.querySelector('.viz2-intra-svg');
    const timeline = story.querySelector('.timeline');
    const map = original.cloneNode(true);
    namespaceMarkers(map,'overview');
    // The Deck bundles the Pipeline renderer without the workbench glyph adapter.
    if (!window.PtoAIInfraPresentation?.operationGlyph) {
      map.querySelectorAll('[data-intra-kind="stage"]').forEach(node=>{
        const rect = node.querySelector('[data-intra-operation]');
        const [stack,stage] = rect.dataset.intraOperation.split(':');
        const label = `S${stack} · ${names[stage]}`;
        const geometry = ['x','y','width','height'].map(key=>Number(rect.getAttribute(key)));
        node.querySelectorAll(':scope > text').forEach(text=>text.remove());
        node.insertAdjacentHTML('beforeend',operationGlyph(`p07-${stack}-${stage}`,label,...geometry,task.stacks[Number(stack)].slot));
      });
    }
    const fullViewBox = map.getAttribute('viewBox');
    const [, , nativeWidth] = fullViewBox.split(' ').map(Number);
    // Keep every Tick, stage and output in the lower overview. Trim the
    // native axis padding and the redundant Stack shortcut strip.
    const firstTick = map.querySelector('.viz2-intra-tick-hit');
    const top = Number(firstTick.getAttribute('y'));
    const height = Number(firstTick.getAttribute('height'));
    const overviewCrop = `0 ${top} ${nativeWidth} ${height+4}`;
    timeline.append(map);
    native.destroy();

    const closeup = story.querySelector('.closeup');
    const cut = story.querySelector('.cut');
    const write = (name,text) => {story.querySelector(`[data-${name}]`).textContent = text;};
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const defaultLink = task.dependencies.find(link=>link.stack===0 && link.buffer==='probability_transfer');
    const operation = (stack,stage) => task.operations.find(op=>op.stack===stack && op.stage===stage);
    const stageNode = op => map.querySelector(`[data-intra-operation="${op.stack}:${op.stage}"]`).parentElement;
    const syncEvidence = link => evidence.edges.find(edge=>edge.kind==='sync' && edge.from===`s${link.stack}:${link.producer}` &&
      edge.to===`s${link.stack}:${link.consumer}`);
    const resourceEdge = link => task.reuse.find(item=>item.buffer===link.buffer && (item.from===link.stack || item.to===link.stack));
    const play = transport.querySelector('[data-action="play"]');
    let scene = 0, link = defaultLink, clock = null, position = 0, lastFrame = null, phase = -1;
    let activePath;
    const example = task.ticks.find(tick=>tick.produce!=null && tick.consume!=null);
    write('map-note',`T${example.index}：Stack ${example.produce} 生产，Stack ${example.consume} 消费 · ${task.stacks.length} 个 Stack 轮用 ${task.transferSlots} 个 Slot`);

    function namespaceMarkers(svg,prefix) {
      svg.querySelectorAll('marker[id]').forEach(marker=>{
        const old = marker.id;
        marker.id = prefix+'-'+old;
        svg.querySelectorAll(`[marker-end="url(#${old})"]`).forEach(path=>path.setAttribute('marker-end',`url(#${marker.id})`));
      });
    }

    function pathFor() {
      if (scene === 3) {
        const edge = resourceEdge(link);
        return map.querySelector(`[data-intra-reuse="${edge.buffer}:${edge.from}:${edge.to}"]`);
      }
      return map.querySelector(`[data-intra-dependency="${link.stack}:${link.buffer}"]`);
    }

    function endpoints() {
      if (scene === 3) {
        const edge = resourceEdge(link);
        return [operation(edge.from,edge.consumer),operation(edge.to,edge.producer)];
      }
      return [operation(link.stack,link.producer),operation(link.stack,link.consumer)];
    }

    function annotateMap() {
      map.querySelectorAll('.story-focus').forEach(node=>node.classList.remove('story-focus','is-related'));
      map.querySelectorAll('[aria-pressed="true"]').forEach(node=>node.setAttribute('aria-pressed','false'));
      if (scene === 0) return;
      const stages = endpoints();
      if (scene === 4) {
        const reuse = resourceEdge(link);
        stages.push(operation(reuse.from,reuse.consumer),operation(reuse.to,reuse.producer));
        map.querySelector(`[data-intra-reuse="${reuse.buffer}:${reuse.from}:${reuse.to}"]`).classList.add('story-focus','is-related');
      }
      stages.forEach(op=>stageNode(op).classList.add('story-focus','is-related'));
      const path = pathFor();
      story.style.setProperty('--event-slot', getComputedStyle(path).getPropertyValue('--slot-color'));
      path.classList.add('story-focus','is-related');
      const event = map.querySelector(`[data-intra-event="${link.event}"][data-intra-index="${link.stack}"]`);
      if (scene !== 3) event.classList.add('story-focus','is-related');
      if (path.hasAttribute('aria-pressed')) path.setAttribute('aria-pressed','true');
      else event.setAttribute('aria-pressed','true');
    }

    function renderCut() {
      closeup.hidden = scene === 0 || scene === 4;
      if (closeup.hidden) {
        activePath = null;
        return;
      }
      const [from,to] = endpoints();
      const fromNode = stageNode(from), toNode = stageNode(to);
      const path = pathFor();
      const event = scene === 3 ? null :
        map.querySelector(`[data-intra-event="${link.event}"][data-intra-index="${link.stack}"]`);
      const nodes = [fromNode,toNode,path,...(event?[event]:[])];
      const boxes = nodes.map(node=>node.getBBox());
      const x = Math.min(...boxes.map(box=>box.x)) - 18;
      const y = Math.min(...boxes.map(box=>box.y)) - 16;
      const width = Math.max(...boxes.map(box=>box.x+box.width)) - x + 18;
      const height = Math.max(...boxes.map(box=>box.y+box.height)) - y + 16;
      const svg = map.cloneNode(false);
      svg.setAttribute('viewBox',`${x} ${y} ${width} ${height}`);
      svg.removeAttribute('aria-label');
      svg.setAttribute('aria-hidden','true');
      const fragment = document.createElementNS(svg.namespaceURI,'g');
      svg.append(fragment);
      fragment.append(map.querySelector('defs').cloneNode(true));
      const clone = node => {
        const copy = node.cloneNode(true);
        for (const attribute of ['tabindex','role','aria-pressed']) copy.removeAttribute(attribute);
        const slot = node.style.getPropertyValue('--slot-color') || node.parentElement.style.getPropertyValue('--slot-color');
        if (slot) copy.style.setProperty('--slot-color',slot);
        fragment.append(copy);
        return copy;
      };
      const base = clone(path);
      activePath = base.cloneNode(true);
      activePath.setAttribute('class','story-signal');
      activePath.removeAttribute('marker-end');
      activePath.setAttribute('pathLength','100');
      fragment.append(activePath);
      if (event) clone(event);
      const producer = clone(fromNode), consumer = clone(toNode);
      producer.classList.remove('is-related');
      consumer.classList.remove('is-related');
      producer.dataset.endpoint = 'from';
      consumer.dataset.endpoint = 'to';
      namespaceMarkers(svg,'detail');
      cut.replaceChildren(svg);
    }

    function ranges(list) {
      return list.map(({start,end})=>start===end?`L${start}`:`L${start}–${end}`).join('、');
    }

    function describe() {
      if (closeup.hidden) return;
      const [from,to] = endpoints();
      const where = op => `T${op.tick} · Stack ${op.stack} · ${names[op.stage]}`;
      write('locator',`${where(from)}  →  ${where(to)}`);
      write('cut-label',`${link.buffer} / Slot ${from.slot}`);
      if (scene === 3) {
        const edge = evidence.edges.find(item=>item.kind==='reuse' && item.buffer===link.buffer &&
          item.from===`s${from.stack}:${from.stage}` && item.to===`s${to.stack}:${to.stage}`);
        write('heading','同一地址交给下一轮');
        write('explanation',`${names[from.stage]} 在 T${from.tick} 读取 Stack ${from.stack} 的数据。前一次读取完成后，${names[to.stage]} 才能在 T${to.tick} 把 Stack ${to.stack} 的结果写入同一个 Slot。`);
        write('event','前次读取 → 后次覆盖');
        write('source',`静态覆盖约束 · ${edge.versions.join(' → ')}`);
        return;
      }
      const sync = syncEvidence(link);
      if (scene === 1) {
        write('heading',`${names[to.stage]} 在等什么？`);
        write('explanation',`读取之前，${names[to.stage]} 要通过 WAIT。${names[from.stage]} 先写入 ${link.buffer}，再 SET 就绪事件。`);
        write('event',`WAIT · ${sync.label}`);
      } else {
        write('heading','就绪事件接通数据交接');
        write('explanation',`${names[from.stage]} 在 T${from.tick} 写入 ${link.buffer} 并 SET。${names[to.stage]} 在 T${to.tick} 通过 WAIT 后读取同一批数据。`);
        write('event',`SET → WAIT · ${sync.label}`);
      }
      write('source',`源码 SET ${ranges(sync.synchronization.producer.source.ranges)} / WAIT ${ranges(sync.consumerSource.ranges)}`);
    }

    // The path arrives exactly when the consumer or next writer lights up.
    const handoff = {start:220,arrival:960,end:1300};
    const reuse = {start:300,arrival:920,end:1200};
    const timing = () => scene === 2 ? handoff : reuse;
    const duration = () => timing().end;
    const playable = () => scene === 2 || scene === 3;
    function drawProgress() {
      const {start,arrival} = timing();
      const nextPhase = scene === 2 ? (position < start ? 0 : position < arrival ? 1 : 2) :
        scene === 3 && position >= arrival ? 1 : 0;
      if (nextPhase !== phase) {
        phase = nextPhase;
        story.dataset.phase = String(phase);
        if (!closeup.hidden) {
          cut.querySelector('[data-endpoint="from"]').classList.toggle('is-related',scene === 3 ? phase === 0 : scene === 2 && phase === 1);
          cut.querySelector('[data-endpoint="to"]').classList.toggle('is-related',scene === 3 ? phase === 1 : scene === 2 && phase === 2);
        }
      }
      if (activePath) {
        const progress = playable() ? Math.max(0,Math.min(1,(position-start)/(arrival-start))) : 0;
        activePath.style.strokeDashoffset = String(100*(1-progress));
        activePath.style.opacity = progress > 0 ? '1' : '0';
      }
    }

    function stop() {
      if (clock !== null) cancelAnimationFrame(clock);
      clock = null;
      lastFrame = null;
      play.textContent = position >= duration() ? '重播' :
        media.matches && playable() ? '下一状态' : position > 0 ? '继续' : '播放';
    }

    function start() {
      if (!playable() || clock !== null) return;
      if (media.matches) {
        position = position >= duration() ? 0 : scene === 3 || position >= handoff.start ? duration() : handoff.start;
        drawProgress();
        stop();
        return;
      }
      if (position >= duration()) position = 0;
      play.textContent = '暂停';
      function frame(now) {
        position = Math.min(duration(),position + (lastFrame === null ? 0 : now-lastFrame));
        lastFrame = now;
        drawProgress();
        if (position < duration()) clock = requestAnimationFrame(frame);
        else stop();
      }
      clock = requestAnimationFrame(frame);
    }

    function show(next,{autoplay = true} = {}) {
      stop();
      scene = Math.max(0,Math.min(4,next));
      position = 0;
      phase = -1;
      story.dataset.scene = String(scene);
      map.setAttribute('viewBox',scene > 0 && scene < 4 ? overviewCrop : fullViewBox);
      story.querySelector('[data-takeaway]').hidden = scene !== 4;
      story.querySelector('[data-map-note]').hidden = scene !== 0;
      annotateMap();
      renderCut();
      describe();
      drawProgress();
      transport.querySelectorAll('[data-scene]').forEach(button=>{
        button.setAttribute('aria-pressed',String(Number(button.dataset.scene)===scene));
      });
      transport.querySelector('[data-action="prev"]').disabled = scene===0;
      transport.querySelector('[data-action="next"]').disabled = scene===4;
      play.disabled = !playable();
      play.textContent = media.matches && playable() ? '下一状态' : '播放';
      if (autoplay && playable() && !media.matches) start();
    }

    function pick(node) {
      const {intraKind:kind,intraIndex:index,intraStage:stage,intraBuffer:buffer,intraFrom:previous} = node.dataset;
      const stack = Number(kind === 'slot' ? previous : index);
      const candidates = task.dependencies.filter(edge=>edge.stack===stack);
      const selected = kind === 'stage' ? candidates.find(edge=>edge.consumer===stage) ?? candidates.find(edge=>edge.producer===stage)
        : candidates.find(edge=>edge.buffer===buffer);
      if (!selected) return;
      link = selected;
      show(kind === 'slot' ? 3 : 1);
      onChange();
    }

    map.querySelectorAll('[data-intra-kind]').forEach(node=>{
      if (!['stage','event','slot'].includes(node.dataset.intraKind)) {
        node.removeAttribute('role');
        node.removeAttribute('tabindex');
        node.removeAttribute('aria-pressed');
      }
    });
    map.addEventListener('click',event=>{
      const node = event.target.closest('[data-intra-kind]');
      if (node) pick(node);
    });
    map.addEventListener('keydown',event=>{
      const node = event.target.closest('[data-intra-kind][role="button"]');
      if (node && ['Enter',' '].includes(event.key)) {
        event.preventDefault();
        pick(node);
      }
    });
    function context() {
      return {stack:link.stack,buffer:link.buffer,position,playing:clock !== null};
    }

    function set(next,{context:remote,quiet = false} = {}) {
      if (!Number.isInteger(next) || next < 0 || next > 4) return false;
      if (remote) {
        link = task.dependencies.find(edge=>edge.stack===remote.stack && edge.buffer===remote.buffer) ?? defaultLink;
      }
      show(next,{autoplay:!remote});
      if (remote && playable()) {
        position = Math.max(0,Math.min(duration(),Number(remote.position) || 0));
        drawProgress();
        if (remote.playing && position < duration()) start();
        else stop();
      }
      if (!quiet) onChange();
      return true;
    }

    transport.addEventListener('click',event=>{
      const button = event.target.closest('button');
      if (!button) return;
      if (button.hasAttribute('data-scene')) {set(Number(button.dataset.scene)); return;}
      switch (button.dataset.action) {
        case 'prev': set(scene-1); break;
        case 'next': set(scene+1); break;
        case 'play':
          if (clock === null) start();
          else stop();
          onChange();
          break;
        case 'reset': link = defaultLink; set(0); break;
        case 'fullscreen': {
          const presentation = transport.closest('#presentation');
          if (document.fullscreenElement) document.exitFullscreen();
          else presentation?.requestFullscreen?.().catch(()=>{});
          break;
        }
      }
    });
    document.addEventListener('visibilitychange',()=>{if (document.hidden && clock !== null) stop();});
    return {
      last:4,
      get beat(){return scene;},
      get context(){return context();},
      set,
      enter(index = 0,remote) {
        link = defaultLink;
        set(index,{context:remote,quiet:true});
      },
      leave:stop,
      pause:stop
    };
  }

  function operationGlyph(id,label,x,y,w,h,slot) {
    const key = /QK|PV/.test(label) ? 'matmul' : /Softmax/.test(label) ? 'softmax' : 'reduce';
    const icon = window.PtoOpIcons.find(entry=>entry.id===key);
    const node = window.VisualSystem.node('op',id,label,0,0)
      .replace('width="260"',`width="${w}"`).replace('height="48"','height="34"')
      .replace('rx="24"','rx="17"').replace('x="130"',`x="${w/2+8}"`).replace('y="24"','y="17"')
      .replace('font-size="18"','style="font-size:12px"');
    const badge = icon ? `<g class="ai-operation-category"><circle cx="16" cy="17" r="9" fill="#222"/><path transform="translate(10 11) scale(.5)" d="${icon.path}" fill="none" stroke="#eee" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></g>` : '';
    return `<g class="ai-operation-glyph" data-op-category="${key}" pointer-events="none" transform="translate(${x} ${y})">${node}${badge}<circle cx="8" cy="48" r="3" fill="var(--slot-color)"/><text class="ai-operation-slot" x="${w/2}" y="48" text-anchor="middle">Slot ${slot}</text></g>`;
  }

  window.PtoPipelineStory = {mount};
  const presentation = document.querySelector('#presentation:has([data-p07-story])');
  if (!presentation) return;
  const story = presentation.querySelector('[data-p07-story]');
  const transport = presentation.querySelector('.transport');
  try {
    const controller = mount(story,transport);
    controller.enter();
    document.addEventListener('keydown',event=>{
      if (event.defaultPrevented || event.altKey || event.metaKey || event.ctrlKey) return;
      if (['ArrowLeft','ArrowRight'].includes(event.key)) {
        event.preventDefault();
        controller.set(controller.beat+(event.key==='ArrowRight'?1:-1));
      }
    });
    if (!presentation.requestFullscreen) transport.querySelector('[data-action="fullscreen"]').hidden = true;
  } catch (error) {
    story.querySelector('[data-error]').hidden = false;
    transport.inert = true;
    console.error(error);
  }
})();
