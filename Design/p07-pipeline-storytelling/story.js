(() => {
  'use strict';

  const modes = [
    ['all', '全部', ''],
    ['data', '等数据', '追踪数据从生产到消费'],
    ['sync', '等同步', '追踪信号从发出到等待'],
    ['reuse', '等资源', '追踪资源从占用到释放'],
  ];
  const sceneMarkup = `
    <div class="waiting-toolbar">
      <div data-p07-controls role="group" aria-label="等待类型">
        ${modes.map(([key,label]) => `<button type="button" data-wait-mode="${key}">${label}</button>`).join('')}
        <span class="waiting-unavailable" aria-label="当前演示未覆盖">
          <button type="button" disabled title="当前演示未覆盖">等前序状态</button>
          <button type="button" disabled title="当前演示未覆盖">等任务完成</button>
        </span>
      </div>
      <p data-wait-hint aria-live="polite"></p>
    </div>
    <div class="timeline viz2-intra" aria-label="Task 完整逻辑流水"></div>
    <div class="waiting-footer">
      <div class="waiting-legend" aria-label="关系图例">
        <span><i class="waiting-key data"></i>数据交接</span>
        <span><i class="waiting-key sync"></i>READY · SET → WAIT</span>
        <span><i class="waiting-key reuse"></i>Slot 复用</span>
      </div>
      <p data-wait-detail aria-live="polite"></p>
    </div>
    <div class="source-renderer" hidden></div>
    <p data-error hidden role="alert">无法加载原 Pipeline。</p>`;

  function namespaceMarkers(svg,prefix) {
    svg.querySelectorAll('marker[id]').forEach(marker=>{
      const old = marker.id;
      marker.id = prefix+'-'+old;
      svg.querySelectorAll(`[marker-end="url(#${old})"]`).forEach(path=>path.setAttribute('marker-end',`url(#${marker.id})`));
    });
  }

  // The filter consumes typed edges and SVG element references, independent of
  // how the underlying schedule or route geometry was produced.
  function bindFocus(map, relationships, detail, interactiveEnabled, onChange) {
    const items = new Set(map.querySelectorAll('[data-intra-kind="stage"],[data-intra-kind="output"],.viz2-intra-output-link'));
    const interactive = new Map();
    for (const relation of relationships) {
      [...relation.elements, ...relation.secondary].forEach(element => items.add(element));
      interactive.set(relation.anchor, relation);
    }
    let mode, selected;
    const describe = relation => { detail.textContent = relation?.description ?? ''; };
    function render(nextMode, id) {
      mode = nextMode;
      selected = relationships.find(item=>item.id===id && item.kind===mode);
      const focused = selected ? [selected] : relationships.filter(item=>item.kind===mode);
      const primary = new Set(focused.flatMap(item=>item.elements));
      const secondary = new Set(focused.flatMap(item=>item.secondary));
      for (const element of items) {
        element.classList.toggle('wait-primary',primary.has(element));
        element.classList.toggle('wait-muted',mode!=='all' && !primary.has(element) && !secondary.has(element));
        element.classList.toggle('wait-secondary',mode!=='all' && !primary.has(element) && secondary.has(element));
      }
      for (const [element, relation] of interactive) {
        const enabled = interactiveEnabled && relation.kind===mode;
        element.setAttribute('tabindex',enabled ? '0' : '-1');
        element.setAttribute('aria-disabled',String(!enabled));
        element.classList.toggle('wait-target',enabled);
        element.setAttribute('aria-pressed',String(relation===selected));
      }
      describe(selected);
    }
    function target(event) {
      const relation = interactive.get(event.target.closest('[data-wait-relation]'));
      return interactiveEnabled && relation?.kind===mode ? relation : undefined;
    }
    function select(event) {
      const relation = target(event);
      if (!relation) return;
      render(mode,relation===selected ? undefined : relation.id);
      onChange();
    }
    map.addEventListener('click',select);
    map.addEventListener('keydown',event=>{
      if (!target(event) || !['Enter',' '].includes(event.key)) return;
      event.preventDefault();
      select(event);
    });
    map.addEventListener('pointerover',event=>{const relation=target(event); if (relation) describe(relation);});
    map.addEventListener('pointerleave',()=>describe(selected));
    map.addEventListener('focusin',event=>{const relation=target(event); if (relation) describe(relation);});
    map.addEventListener('focusout',()=>describe(selected));
    return {render,get selectedId(){return selected?.id;}};
  }

  function pipelineRelationships(map,task,evidence) {
    const nodes = new Map(evidence.nodes.map(node=>[node.id,node]));
    const edgeById = new Map(evidence.edges.map(edge=>[edge.id,edge]));
    const stages = [ ...map.querySelectorAll('[data-intra-operation]') ];
    const stageElements = new Map(stages.map(rect=>[`s${rect.dataset.intraOperation}`,rect.parentElement]));
    const where = node => `T${node.tick} · S${node.stack} ${node.stage === 'accumulate' ? 'Accumulate' : {qk:'QK',softmax:'Softmax',pv:'PV'}[node.stage]}`;
    const paths = new Map([...map.querySelectorAll('[data-intra-dependency]')].map(path=>[path.dataset.intraDependency,path]));
    const events = new Map([...map.querySelectorAll('[data-intra-event]')].map(event=>[`${event.dataset.intraIndex}:${event.dataset.intraEvent}`,event]));
    const reusePaths = new Map([...map.querySelectorAll('[data-intra-reuse]')].map(path=>[path.dataset.intraReuse,path]));

    function project(edge, anchor, marks, secondary, description) {
      if (!edge || !anchor || marks.some(mark=>!mark)) return [];
      const endpoints = [stageElements.get(edge.from),stageElements.get(edge.to)];
      if (!endpoints.every(Boolean)) return [];
      anchor.dataset.waitRelation = edge.id;
      anchor.setAttribute('role','button');
      anchor.setAttribute('aria-label',description(nodes.get(edge.from),nodes.get(edge.to)));
      return [{id:edge.id,kind:edge.kind,anchor,elements:[...endpoints,...marks],secondary,
        description:anchor.getAttribute('aria-label')}];
    }

    const handoffs = task.dependencies.flatMap(link=>{
      const path = paths.get(`${link.stack}:${link.buffer}`);
      const data = edgeById.get(`data:s${link.stack}:${link.buffer}`);
      const event = events.get(`${link.stack}:${link.event}`);
      const sync = evidence.edges.find(edge=>edge.kind==='sync' && edge.from===data?.from &&
        edge.to===data?.to && edge.synchronization.buffer===link.buffer &&
        `${edge.synchronization.name}_EVENT`===link.event);
      const relationships = project(data,path,[path],event ? [event] : [],
        (from,to)=>`${where(from)} → ${where(to)} · 等待 ${link.buffer} 写入后读取`);
      if (path && event && sync) {
        // Reuse the handoff geometry as a visual alignment. The dotted overlay
        // belongs to the source-backed READY pair, while the solid line stays data.
        const route = path.cloneNode(false);
        route.removeAttribute('data-intra-dependency');
        route.removeAttribute('marker-end');
        route.setAttribute('class','waiting-sync-route');
        route.setAttribute('aria-hidden','true');
        event.before(route);
        relationships.push(...project(sync,event,[route,event],[path],
          (from,to)=>`${where(from)} SET → ${where(to)} WAIT · ${sync.label} 就绪后继续`));
      }
      return relationships;
    });
    const reuse = task.reuse.flatMap(link=>{
      const path = reusePaths.get(`${link.buffer}:${link.from}:${link.to}`);
      return project(edgeById.get(`reuse:${link.buffer}:s${link.from}:s${link.to}`),path,[path],[],
        (from,to)=>`${where(from)} 读完 → ${where(to)} 覆盖 · ${link.buffer} / Slot ${link.slot}`);
    });
    return [...handoffs,...reuse];
  }

  function mount(story,{interactive = true,onChange = () => {}} = {}) {
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
    story.querySelector('.timeline').append(map);
    native.destroy();

    map.querySelectorAll('[data-intra-kind]').forEach(node=>{
      node.removeAttribute('role');
      node.removeAttribute('tabindex');
      node.removeAttribute('aria-pressed');
    });
    const relationships = pipelineRelationships(map,task,evidence);
    const supported = new Set(['all',...relationships.map(item=>item.kind)]);
    const focus = bindFocus(map,relationships,story.querySelector('[data-wait-detail]'),interactive,onChange);
    story.querySelector('[data-p07-controls]').inert = !interactive;
    const buttons = [...story.querySelectorAll('[data-wait-mode]')];
    let beat = 0, mode = 'all';
    function show(next,selectedId) {
      mode = supported.has(next) ? next : 'all';
      story.dataset.waitMode = mode;
      const [, , hint] = modes.find(([key])=>key===mode);
      story.querySelector('[data-wait-hint]').textContent = hint;
      buttons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.waitMode===mode)));
      focus.render(mode,selectedId);
    }

    buttons.forEach(button=>{
      const key = button.dataset.waitMode;
      button.disabled = !supported.has(key);
      if (button.disabled) button.title = '暂无可展示关系';
      button.addEventListener('click',()=>{
        show(key);
        onChange();
      });
    });
    function set(next,{context,quiet = false} = {}) {
      if (!Number.isInteger(next) || next < 0 || next >= modes.length) return false;
      beat = next;
      show(context?.mode ?? modes[beat][0],context?.relationId);
      if (!quiet) onChange();
      return true;
    }
    set(0,{quiet:true});
    return {
      get beat(){return beat;},
      get context(){return {mode,relationId:focus.selectedId};},
      last:modes.length-1,
      resetOnEnter:true,
      set,
      enter(step = 0,context){set(step,{context,quiet:true});},
      leave(){set(0,{quiet:true});},
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
  const story = document.querySelector('#presentation[data-p07-preview] [data-p07-story]');
  if (!story) return;
  try {
    const controller = mount(story);
    document.addEventListener('keydown',event=>{
      if (event.defaultPrevented || event.altKey || event.metaKey || event.ctrlKey) return;
      if (['ArrowLeft','ArrowRight'].includes(event.key) &&
          controller.set(controller.beat+(event.key==='ArrowRight'?1:-1))) event.preventDefault();
    });
  } catch (error) {
    story.querySelector('[data-error]').hidden = false;
    console.error(error);
  }
})();
