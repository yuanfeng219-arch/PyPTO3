/* QK Work Assignment for decode_sparse_attn_csa.py.
   Work items -> qk_plan reorder -> qk_order -> SPMD block assignment.
   Pure data functions first; one renderer that owns only its own host element. */
(function (global) {
  'use strict';

  const TOKENS = 5;
  const SOURCE_TOKENS = 256;
  const TOKENS_PER_REQUEST = 8;
  const SPARSE_BLOCKS = 5;
  const ITEM_COUNT = TOKENS * SPARSE_BLOCKS;
  const WINDOW_SB = 0;                       // SB0 is the sliding-window block, always live
  const NUM_QK_CORES = 24;                   // decode_sparse_attn_csa.py L67
  const WIN = 128, IDX_TOPK = 512, ATTN_K_TILE = 128, COMPRESS_RATIO = 4;
  const POSITIONS = [7, 1023, 2047, 8192];

  // 旧版固定 mask（1 = valid），仅作 buildWorkItems 的数组入参兼容保留。
  const DEFAULT_MASK = Object.freeze([
    Object.freeze([1, 1, 0, 0, 0]),
    Object.freeze([1, 0, 0, 0, 0]),
    Object.freeze([1, 1, 1, 0, 0]),
    Object.freeze([1, 0, 1, 0, 0]),
    Object.freeze([1, 1, 0, 0, 0])
  ]);
  const DEFAULT_SELECTION = 11;              // Token 2 / Sparse Block 1

  const MODES = Object.freeze({
    source: Object.freeze({ tokens: SOURCE_TOKENS, blocks: NUM_QK_CORES })
  });

  const SUB_FILE = 'decode_sparse_attn_csa.py';
  const REPO_BLOB = 'https://github.com/yuanfeng219-arch/PyPTO3/blob/main/Data/DeepseekV4/deepseek_v4_flash_dspark/';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* ------------------------------------------------------------------ data */

  function syntheticTopk(position) {
    const upper = Math.max(0, Math.floor((position + 1) / COMPRESS_RATIO));
    const count = Math.min(IDX_TOPK, upper);
    const values = Array(IDX_TOPK).fill(-1);
    for (let i = 0; i < count; i++) values[i] = (i * 37) % upper;
    return values;
  }

  function buildWorkItems(input) {
    const legacyMask = Array.isArray(input) ? input : null;
    const count = legacyMask ? legacyMask.length : (MODES[input.mode] || MODES.source).tokens;
    const items = [];
    for (let token = 0; token < count; token++) {
      const requestIndex = Math.floor(token / TOKENS_PER_REQUEST);
      const basePosition = input.workload === 'mixed' && input.mode === 'source' ? POSITIONS[requestIndex % POSITIONS.length] : input.position;
      const positionId = legacyMask ? null : basePosition + token % TOKENS_PER_REQUEST;
      const indices = legacyMask ? null : syntheticTopk(positionId);
      const upper = positionId == null ? null : Math.floor((positionId + 1) / COMPRESS_RATIO);
      for (let sparseBlock = 0; sparseBlock < SPARSE_BLOCKS; sparseBlock++) {
        const slotStart = (sparseBlock - 1) * ATTN_K_TILE;
        const validKvCount = sparseBlock === WINDOW_SB ? Math.min(WIN, (positionId ?? WIN - 1) + 1)
          : legacyMask ? (legacyMask[token][sparseBlock] ? ATTN_K_TILE : 0)
          : indices.slice(slotStart, slotStart + ATTN_K_TILE).filter(index => index >= 0 && index < upper).length;
        const override = !legacyMask && input.overrides?.get(token * SPARSE_BLOCKS + sparseBlock);
        const valid = legacyMask ? !!legacyMask[token][sparseBlock] : override == null ? sparseBlock === WINDOW_SB || validKvCount > 0 : override;
        items.push({
          id: token * SPARSE_BLOCKS + sparseBlock,   // workItemId = token * SPARSE_BLOCKS + sb
          token,
          requestIndex,
          positionId,
          rank: legacyMask ? null : input.rank,
          scenarioId: legacyMask ? 'legacy' : input.mode + '-' + (input.workload === 'mixed' ? 'mixed' : 'position-' + input.position),
          sparseBlock,
          valid,
          validKvCount,
          validityReason: legacyMask ? '教学 mask' : override == null ? sparseBlock === WINDOW_SB ? 'SB0：滑动窗口固定有效' : validKvCount ? 'Synthetic idx_topk 在本块含 ' + validKvCount + ' 个满足范围的索引' : '本块没有满足范围的 Synthetic idx_topk 索引' : 'Advanced Simulation 手动覆盖',
          originalPosition: token * SPARSE_BLOCKS + sparseBlock
        });
      }
    }
    return items;
  }

  function deriveScenario(input) {
    const items = buildWorkItems(input), queue = buildQueue(items), byId = new Map(items.map(item => [item.id, item]));
    const blockCount = MODES[input.mode]?.blocks ?? NUM_QK_CORES;
    const blocks = assignToBlocks(queue, blockCount);
    blocks.forEach(block => { block.validCount = block.entries.filter(entry => byId.get(entry.id).valid).length; block.invalidCount = block.entries.length - block.validCount; });
    return { items, queue, blocks, blockCount, byId, validCount: items.filter(item => item.valid).length, input };
  }

  // qk_plan: pass 1 writes valid tiles in token-major order, pass 2 appends the rest.
  // Invalid items are never dropped, only pushed to the tail.
  function buildQueue(items) {
    const valid = items.filter(item => item.valid).map(item => item.id);
    const invalid = items.filter(item => !item.valid).map(item => item.id);
    return [...valid, ...invalid];
  }

  // qk_flat = qk_core + qk_it * blockCount  =>  block = pos % N, iteration = floor(pos / N).
  function assignToBlocks(queue, blockCount) {
    const blocks = Array.from({ length: blockCount }, (_, index) => ({ index, entries: [] }));
    queue.forEach((id, queuePosition) => {
      const block = queuePosition % blockCount;
      blocks[block].entries.push({ id, queuePosition, iteration: Math.floor(queuePosition / blockCount) });
    });
    return blocks;
  }

  function getWorkItemDetails(id, queue, blockCount) {
    const queuePosition = queue.indexOf(id);
    return {
      id,
      token: Math.floor(id / SPARSE_BLOCKS),
      sparseBlock: id % SPARSE_BLOCKS,
      originalPosition: id,
      queuePosition,
      block: queuePosition % blockCount,
      iteration: Math.floor(queuePosition / blockCount),
      blockCount,
      qkFlat: (queuePosition % blockCount) + Math.floor(queuePosition / blockCount) * blockCount
    };
  }

  function getBlockDetails(index, items, queue, blockCount) {
    const byId = new Map(items.map(item => [item.id, item]));
    const entries = [];
    for (let queuePosition = index; queuePosition < queue.length; queuePosition += blockCount) {
      entries.push({
        id: queue[queuePosition],
        queuePosition,
        iteration: Math.floor(queuePosition / blockCount),
        valid: !!byId.get(queue[queuePosition])?.valid
      });
    }
    return {
      index,
      blockCount,
      entries,
      // qk_lane_iters = (qk_items - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES  (L213)
      laneIterations: Math.max(0, Math.floor((queue.length - index + blockCount - 1) / blockCount)),
      validCount: entries.filter(entry => entry.valid).length,
      invalidCount: entries.filter(entry => !entry.valid).length
    };
  }

  /* --------------------------------------------------------------- excerpts */

  const excerptCache = new Map();
  function excerpt(file, start, end) {
    const key = file + ':' + start + ':' + end;
    if (excerptCache.has(key)) return excerptCache.get(key);
    const text = global.PTO_CSA_SOURCE?.dependencies?.[file];
    const out = text
      ? text.split('\n').slice(start - 1, end).map((line, i) => (start + i) + '  ' + line).join('\n')
      : '';
    excerptCache.set(key, out);
    return out;
  }

  /* ----------------------------------------------------- Canvas-first view */
  const PHASES = ['generation', 'ordering', 'assignment'];
  const PHASE_LABELS = {generation:'Work Generation',ordering:'Work Ordering',assignment:'SPMD Lane Assignment'};
  function create(host, options = {}) {
    const onSource = options.onSource || (() => {}), onSelection = options.onSelection || (() => {}), onContext = options.onContext || (() => {});
    const state = {mode:options.mode || 'source', workload:'mixed', position:8192, rank:options.rank ?? 0,
      requestIndex:0, selection:options.selection || null, inspectorOpen:false, phase:'assignment', playing:false, loop:false,
      scale:1, panX:0, panY:0, manualCamera:false, advanced:false, overrides:new Map()};
    if(options.selection?.type==='request'||options.selection?.type==='query')state.requestIndex=options.selection.requestIndex;
    else if(options.selection?.type==='item')state.requestIndex=Math.floor(options.selection.id/(SPARSE_BLOCKS*TOKENS_PER_REQUEST));
    const reduced = !!global.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    let timer=null, drag=null, morphCameraTimer=null;
    const scenarioId=()=>state.mode+'-'+(state.workload==='mixed'?'mixed':'position-'+state.position);
    const derive=()=>deriveScenario({mode:state.mode,workload:state.workload,position:state.position,rank:state.rank,overrides:state.overrides});
    const itemDetails=(d,id)=>getWorkItemDetails(id,d.queue,d.blockCount);
    function stop(){if(timer)global.clearInterval(timer);timer=null;state.playing=false;const b=host.querySelector('[data-qk-play]');if(b)b.textContent='播放';}
    function start(){stop();state.playing=true;const b=host.querySelector('[data-qk-play]');if(b)b.textContent='暂停';timer=global.setInterval(()=>{const next=PHASES.indexOf(state.phase)+1;if(next>=PHASES.length){if(state.loop)updatePhase('generation');else stop();}else updatePhase(PHASES[next]);},1800);}
    function viewportPositions(d,phase){
      const positions=new Map(),source=state.mode==='source';
      if(phase==='generation'){
        const cols=source?8:1, groupW=source?164:1100, groupH=source?116:610;
        d.items.forEach(item=>{const group=source?item.requestIndex:0,baseX=42+(group%cols)*groupW,baseY=58+Math.floor(group/cols)*groupH;
          const row=source?item.token%8:item.token,stepX=source?17:172,stepY=source?11:95;
          positions.set(item.id,{x:baseX+(source?65:90)+item.sparseBlock*stepX,y:baseY+(source?22:12)+row*stepY,w:source?13:120,h:source?8:65});});
      }else if(phase==='ordering'){
        const cols=source?64:5,cellW=source?19:218,cellH=source?24:87,offsetX=source?98:140,offsetY=source?90:92;
        d.queue.forEach((id,p)=>positions.set(id,{x:offsetX+p%cols*cellW,y:offsetY+Math.floor(p/cols)*cellH,w:source?15:170,h:source?16:66}));
      }else{
        const cols=source?12:4,laneW=source?110:320,laneH=source?240:540;
        d.blocks.forEach(block=>block.entries.forEach(entry=>{const x=36+(block.index%cols)*laneW,y=52+Math.floor(block.index/cols)*laneH;
          const cellCol=source?entry.iteration%9:entry.iteration%3,cellRow=source?Math.floor(entry.iteration/9):Math.floor(entry.iteration/3);
          positions.set(entry.id,{x:x+7+cellCol*(source?10:92),y:y+(source?52:40)+cellRow*(source?28:92),w:source?8:70,h:source?20:70});}));
      }
      return positions;
    }
    function backdrop(d){
      const source=state.mode==='source',phase=state.phase;
      if(phase==='generation'){
        const cols=source?8:1,groupW=source?164:1100,groupH=source?116:610;
        return Array.from({length:source?32:1},(_,group)=>{const x=42+(group%cols)*groupW,y=58+Math.floor(group/cols)*groupH;
          const queryLabels=source?Array.from({length:8},(_,offset)=>'<text class="csa-qk-query-label" data-qk-query="'+(group*8+offset)+'" role="button" tabindex="-1" x="'+(x+7)+'" y="'+(y+29+offset*11)+'">Q'+(group*8+offset)+'</text>').join(''):Array.from({length:5},(_,token)=>'<text class="csa-qk-query-label" data-qk-query="'+token+'" role="button" tabindex="-1" x="'+(x+12)+'" y="'+(y+49+token*95)+'">Q'+token+'</text>').join('');
          return '<g class="csa-qk-canvas-group" data-qk-request-group="'+group+'"><rect x="'+x+'" y="'+y+'" width="'+(groupW-12)+'" height="'+(groupH-5)+'" rx="5" vector-effect="non-scaling-stroke"/><text x="'+(x+8)+'" y="'+(y+14)+'">Request '+group+'</text>'+queryLabels+'</g>';}).join('');
      }
      if(phase==='ordering'){
        const boundary=d.validCount,cols=source?64:5,cellH=source?24:87;
        const selectedLane=state.selection?.type==='block'?d.blocks[state.selection.block]:null;
        const positions=selectedLane?viewportPositions(d,'ordering'):null;
        const guide=selectedLane?'<polyline class="csa-qk-lane-guide" points="'+selectedLane.entries.slice(0,6).map(entry=>{const p=positions.get(entry.id);return (p.x+p.w/2)+','+(p.y+p.h/2);}).join(' ')+'"/>':'';
        return '<rect class="csa-qk-order-valid" x="86" y="75" width="1230" height="'+(Math.ceil(boundary/cols)*cellH+12)+'" rx="5"/>'+
          '<text class="csa-qk-canvas-axis" x="98" y="64">Pass 1 · valid '+boundary+' 项　　Pass 2 · invalid '+(d.items.length-boundary)+' 项（保留在队列尾部）</text>'+guide;
      }
      const cols=source?12:4,laneW=source?110:320,laneH=source?240:540;
      return d.blocks.map(block=>{const x=36+(block.index%cols)*laneW,y=52+Math.floor(block.index/cols)*laneH;
        return '<g class="csa-qk-canvas-lane" data-qk-lane="'+block.index+'" role="button" tabindex="0" aria-label="Lane '+block.index+' · '+block.entries.length+' Work Items"><rect x="'+x+'" y="'+y+'" width="'+(laneW-8)+'" height="'+(laneH-8)+'" rx="5" vector-effect="non-scaling-stroke"/><text x="'+(x+10)+'" y="'+(y+21)+'">Lane '+String(block.index).padStart(2,'0')+'</text><text class="csa-qk-canvas-count" x="'+(x+10)+'" y="'+(y+38)+'">'+block.entries.length+' 项 · '+block.validCount+' / '+block.invalidCount+'</text></g>';}).join('');
    }
    function selectionIds(d,selection=state.selection){
      if(!selection)return new Set();
      if(selection.type==='request')return new Set(d.items.filter(item=>item.requestIndex===selection.requestIndex).map(item=>item.id));
      if(selection.type==='query')return new Set(d.items.filter(item=>item.token===selection.tokenIndex).map(item=>item.id));
      if(selection.type==='block')return new Set(d.blocks[selection.block]?.entries.map(entry=>entry.id)||[]);
      return new Set([selection.id]);
    }
    function itemMarkup(d,positions){const relatedIds=selectionIds(d);return d.items.map(item=>{const p=positions.get(item.id),selected=state.selection?.type==='item'&&state.selection.id===item.id;
      const related=!selected&&relatedIds.has(item.id);
      const info=itemDetails(d,item.id);
      return '<g class="csa-qk-canvas-item '+(item.valid?'is-valid':'is-invalid')+(selected?' is-selected':'')+(related?' is-related':'')+(state.selection&&!related&&!selected?' is-dimmed':'')+'" data-qk-item="'+item.id+'" tabindex="'+(selected||(!state.selection&&item.id===0)?'0':'-1')+'" role="button" aria-label="Work Item '+item.id+' · Q'+item.token+' SB'+item.sparseBlock+' · qk_order['+info.queuePosition+'] · Lane '+info.block+' / it '+info.iteration+' · '+(item.valid?'valid':'invalid')+'" style="transform:translate('+p.x+'px,'+p.y+'px)"><rect width="'+p.w+'" height="'+p.h+'" rx="2" vector-effect="non-scaling-stroke"/></g>';}).join('');}
    function inspector(d){if(!state.selection||!state.inspectorOpen)return '';
      const selection=state.selection,block=selection.type==='block'?getBlockDetails(selection.block,d.items,d.queue,d.blockCount):null;
      let title='',facts=[],formula='';
      if(selection.type==='request'){
        const items=d.items.filter(item=>item.requestIndex===selection.requestIndex);
        title='Request '+selection.requestIndex;facts=[['Rank / Scenario',state.rank+' / '+scenarioId()],['Queries','Q'+items[0].token+'–Q'+items.at(-1).token],['Work Items',items.length],['valid / invalid',items.filter(item=>item.valid).length+' / '+items.filter(item=>!item.valid).length]];
        formula='Request '+selection.requestIndex+' → 8 Queries × 5 Sparse Blocks = 40 Work Items';
      }else if(selection.type==='query'){
        const items=d.items.filter(item=>item.token===selection.tokenIndex);
        title='Query '+selection.tokenIndex;facts=[['Rank / Scenario',state.rank+' / '+scenarioId()],['Request',selection.requestIndex],['Position',items[0].positionId],['Work Items',items.map(item=>item.id).join(', ')],['valid / invalid',items.filter(item=>item.valid).length+' / '+items.filter(item=>!item.valid).length]];
        formula='Q'+selection.tokenIndex+' → SB0–SB4 → Work Items '+items[0].id+'–'+items.at(-1).id;
      }else if(block){title='SPMD Lane '+block.index;facts=[['Rank / Scenario',state.rank+' / '+scenarioId()],['Work Items',block.entries.length],['valid / invalid',block.validCount+' / '+block.invalidCount],['Queue Position',block.entries[0]?.queuePosition+' → '+block.entries.at(-1)?.queuePosition]];
        formula='queuePosition = laneId + iteration × '+d.blockCount+'\n'+block.entries.map(e=>'it '+e.iteration+' → qk_order['+e.queuePosition+'] = '+e.id).join('\n');
      }else{const item=d.byId.get(selection.id);if(!item)return '';const info=itemDetails(d,item.id);title='Work Item '+item.id;
        facts=[['Rank / Scenario',item.rank+' / '+item.scenarioId],['Request / Query',item.requestIndex+' / Q'+item.token],['Position',item.positionId],['Sparse Block','SB'+item.sparseBlock],['KV 来源',item.sparseBlock===0?'Window / Original KV':'Selected Compressed KV'],['有效索引',item.validKvCount],['Validity',item.valid?'valid':'invalid'],['qk_order 位置',info.queuePosition],['Lane / Iteration',info.block+' / '+info.iteration]];
        formula='workItemId = '+item.token+' × 5 + '+item.sparseBlock+' = '+item.id+'\nqueuePosition = '+info.queuePosition+'\nlane = '+info.queuePosition+' % '+d.blockCount+' = '+info.block+'\niteration = floor('+info.queuePosition+' / '+d.blockCount+') = '+info.iteration;
      }
      const path=selection.type==='item'?(d.byId.get(selection.id).valid?'KV Gather → QK Matmul → Scale / Bias / Softmax → PV Matmul → Save m/l/o':'invalid 分支 → 写入中性 m/l/o；不进行 QK / PV 计算'):'';
      return '<aside class="csa-qk-drawer" aria-label="当前计算对象详情"><div class="csa-qk-drawer-head"><h2>'+esc(title)+'</h2><button type="button" data-qk-close aria-label="关闭详情">×</button></div><dl>'+facts.map(([k,v])=>'<dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd>').join('')+'</dl>'+
        '<details><summary>计算规则与来源</summary><pre>'+esc(formula)+'</pre>'+(selection.type==='item'?'<p>'+esc(d.byId.get(selection.id).validityReason)+'</p><p>'+esc(path)+'</p>':'')+'<p>模型配置 TP=2、H=64；Workload 输入 DECODE_BATCH=64、DECODE_SEQ=8、Position；算子常量 WIN=128、IDX_TOPK=512、ATTN_K_TILE=128、QK_M_TILE=32、NUM_QK_CORES=24。64 / 2 × 8 = 256 Query；(128 + 512) / 128 = 5 SB。T 随 Runtime 输入变化。</p><p>Synthetic idx_topk：每个 Query 最多 512 槽，前 min(512, floor((position+1)/4)) 槽使用确定性范围内索引，其余填 -1；真实索引由 Indexer 产生，-1 不表示 KV 值 0。</p></details>'+
        '<details><summary>qk_plan / qk_pv · Source Evidence</summary>'+[[134,196,'有效性与 qk_order'],[208,224,'Dispatch 与 Lane 领取'],[224,314,'valid / invalid 路径']].map(([a,z,label])=>'<p><button type="button" data-qk-source="'+SUB_FILE+':'+a+':'+z+'">'+label+' · '+SUB_FILE+' : '+a+'–'+z+' ↗</button></p><pre>'+esc(excerpt(SUB_FILE,a,z))+'</pre>').join('')+'</details>'+
        '<details><summary>显式任务依赖</summary><p>qk_plan_tid → qk_pv ← cache_ready_dep。这里是源码中的静态依赖，不是 Runtime Timeline。</p><button type="button" data-qk-source="decode_csa.py:384:393">查看调用处 ↗</button></details>'+
        (selection.type==='item'?'<details><summary>Advanced Simulation · 手动覆盖</summary><p>仅用于观察规则变化，不代表 Indexer 真实输出。SB0 固定有效。</p><button type="button" data-qk-advanced>'+(state.advanced?'关闭手动覆盖':'启用手动覆盖')+'</button>'+(state.advanced&&d.byId.get(selection.id).sparseBlock!==0?'<button type="button" data-qk-toggle="'+selection.id+'">切换当前 Work Item 有效性</button><button type="button" data-qk-reset>清除覆盖</button>':'')+'</details>':'')+'</aside>';
    }
    function shell(d){const count=d.items.length,positions=viewportPositions(d,state.phase);
      const requests='<label>Request <select data-qk-request>'+Array.from({length:32},(_,i)=>'<option value="'+i+'"'+(i===state.requestIndex?' selected':'')+'>'+i+'</option>').join('')+'</select></label>';
      return '<div class="csa-qk-canvas-shell"><div class="csa-qk-scenario"><label>场景 <select data-qk-workload><option value="mixed"'+(state.workload==='mixed'?' selected':'')+'>Mixed Context · Synthetic</option><option value="uniform"'+(state.workload==='uniform'?' selected':'')+'>Uniform Context · Synthetic</option></select></label>'+
        (state.workload==='uniform'?'<label>Position <select data-qk-position>'+POSITIONS.map(p=>'<option value="'+p+'"'+(p===state.position?' selected':'')+'>'+p+'</option>').join('')+'</select></label>':'')+
        '<label>Rank <select data-qk-rank><option value="0"'+(state.rank===0?' selected':'')+'>0</option><option value="1"'+(state.rank===1?' selected':'')+'>1</option></select></label>'+requests+'<span class="csa-qk-scenario-count">'+(count/SPARSE_BLOCKS)+' Queries · '+count+' Work Items · '+d.blockCount+' Lanes</span></div>'+
        '<div class="csa-qk-canvas-head"><div><h2 data-qk-title>'+PHASE_LABELS[state.phase]+'</h2><p data-qk-subtitle>'+subtitle(d)+'</p></div><div class="csa-qk-canvas-controls"><button type="button" class="btn btn-ghost btn-sm" data-qk-play>播放</button><button type="button" class="btn btn-ghost btn-sm" data-qk-replay>重播</button><div class="segmented-control segmented-control-muted">'+PHASES.map(p=>'<button type="button" class="btn btn-sm'+(state.phase===p?' is-selected':'')+'" data-qk-phase="'+p+'" aria-pressed="'+(state.phase===p)+'">'+({generation:'Generation',ordering:'Ordering',assignment:'Assignment'}[p])+'</button>').join('')+'</div><label><input type="checkbox" data-qk-loop'+(state.loop?' checked':'')+'> 循环</label></div></div>'+
        '<div class="csa-qk-canvas-viewport" tabindex="0" role="group" aria-label="QK/PV Work Assignment 画布"><svg viewBox="0 0 1400 790" preserveAspectRatio="xMidYMid meet"><g data-qk-scene><g data-qk-backdrop>'+backdrop(d)+'</g><g data-qk-items>'+itemMarkup(d,positions)+'</g></g></svg><div class="csa-qk-tooltip" data-qk-tooltip hidden></div></div>'+
        '<footer class="csa-qk-canvas-footer"><span data-qk-summary>'+summary(d)+'</span><span class="csa-qk-legend"><i class="is-valid"></i>valid <i class="is-invalid"></i>invalid</span><span>静态 Work Assignment · 动效仅表示结构变换</span><div class="csa-qk-zoom"><button type="button" class="btn btn-ghost btn-sm" data-qk-inspector hidden>对象详情</button><button type="button" class="btn btn-ghost btn-sm" data-qk-back hidden>返回全览</button><button type="button" class="btn btn-ghost btn-sm" data-qk-zoom="0.8" aria-label="缩小">−</button><output data-qk-scale>'+Math.round(state.scale*100)+'%</output><button type="button" class="btn btn-ghost btn-sm" data-qk-zoom="1.25" aria-label="放大">＋</button><button type="button" class="btn btn-ghost btn-sm" data-qk-fit>适应</button></div></footer>'+inspector(d)+'</div>';
    }
    function subtitle(d){if(state.phase==='generation')return (d.items.length/SPARSE_BLOCKS)+' Queries × 5 Sparse Blocks → '+d.items.length+' Work Items';if(state.phase==='ordering')return d.validCount+' valid 优先 · '+(d.items.length-d.validCount)+' invalid 追加';return d.items.length+' 个 Work Items → '+d.blockCount+' 条逻辑 Lane · 固定步长领取';}
    function summary(d){
      const s=state.selection;
      if(s?.type==='request')return 'Request '+s.requestIndex+' · 8 Queries / 40 Work Items · 三阶段持续追踪';
      if(s?.type==='query')return 'Q'+s.tokenIndex+' · Request '+s.requestIndex+' · 5 Sparse Blocks / Work Items';
      if(s?.type==='item'){const item=d.byId.get(s.id),info=itemDetails(d,item.id);return 'Work Item '+item.id+' · Q'+item.token+' / SB'+item.sparseBlock+' → qk_order['+info.queuePosition+'] → Lane '+info.block+' / it '+info.iteration;}
      if(s?.type==='block'){const b=d.blocks[s.block];return 'Lane '+b.index+' · qk_order['+b.index+'] → ['+(b.index+d.blockCount)+'] → … · '+b.entries.length+' 项';}
      return d.validCount+' valid / '+(d.items.length-d.validCount)+' invalid · 点击 Work Item 或 Lane 查看';
    }
    function sceneBounds(){const source=state.mode==='source';if(source)return state.phase==='generation'?{x:32,y:48,w:1330,h:486}:state.phase==='ordering'?{x:80,y:50,w:1250,h:530}:{x:26,y:42,w:1335,h:490};return state.phase==='generation'?{x:32,y:40,w:1130,h:635}:state.phase==='ordering'?{x:110,y:55,w:1120,h:490}:{x:26,y:42,w:1300,h:560};}
    function fitBox(){const viewport=host.querySelector('.csa-qk-canvas-viewport'),ratio=viewport?.clientWidth&&viewport?.clientHeight?viewport.clientWidth/viewport.clientHeight:1400/790,b=sceneBounds();let h=Math.max(b.h*1.06,b.w*1.06/ratio);return {x:b.x+b.w/2,y:b.y+b.h/2,w:h*ratio,h};}
    function camera(){const svg=host.querySelector('.csa-qk-canvas-viewport svg');if(!svg)return;const fit=fitBox(),w=fit.w/state.scale,h=fit.h/state.scale,x=fit.x-state.panX-w/2,y=fit.y-state.panY-h/2;svg.setAttribute('viewBox',x+' '+y+' '+w+' '+h);const out=host.querySelector('[data-qk-scale]');if(out)out.textContent=Math.round(state.scale*100)+'%';}
    function focusCamera(d){const s=state.selection;if(!s)return;const positions=viewportPositions(d,state.phase),ids=selectionIds(d),rects=[...ids].map(id=>positions.get(id)).filter(Boolean);if(!rects.length)return;
      const minX=Math.min(...rects.map(p=>p.x)),maxX=Math.max(...rects.map(p=>p.x+p.w)),minY=Math.min(...rects.map(p=>p.y)),maxY=Math.max(...rects.map(p=>p.y+p.h));
      const fit=fitBox(),bboxW=maxX-minX,bboxH=maxY-minY,maxZoom=s.type==='item'?2.8:s.type==='block'?1.8:s.type==='query'?1.7:1.45;
      if(!state.manualCamera)state.scale=Math.max(1,Math.min(maxZoom,fit.w/(bboxW+160),fit.h/(bboxH+110)));
      const viewW=fit.w/state.scale,targetX=(minX+maxX)/2,targetY=(minY+maxY)/2;
      state.panX=fit.x-(targetX+(state.inspectorOpen ? .18*viewW : 0));state.panY=fit.y-targetY;camera();
    }
    function refreshSelection(d,locate=true){const s=state.selection,ids=selectionIds(d),has=!!s;
      if(state.phase==='ordering'){const backdropNode=host.querySelector('[data-qk-backdrop]');if(backdropNode)backdropNode.innerHTML=backdrop(d);}
      host.querySelectorAll('[data-qk-item]').forEach(node=>{const id=Number(node.dataset.qkItem),selected=s?.type==='item'&&id===s.id,related=!selected&&ids.has(id);node.classList.toggle('is-selected',selected);node.classList.toggle('is-related',related);node.classList.toggle('is-dimmed',has&&!selected&&!related);node.setAttribute('tabindex',selected||(!has&&id===0)?'0':'-1');node.setAttribute('aria-pressed',String(selected));});
      host.querySelectorAll('[data-qk-lane]').forEach(node=>{const lane=Number(node.dataset.qkLane),selected=s?.type==='block'&&s.block===lane,contains=s?.type==='item'&&itemDetails(d,s.id).block===lane;node.classList.toggle('is-selected',selected);node.classList.toggle('is-containing',contains);node.classList.toggle('is-dimmed',has&&s?.type==='block'&&!selected);});
      const request=s?.type==='request'?s.requestIndex:s?.type==='query'?s.requestIndex:s?.type==='item'?d.byId.get(s.id)?.requestIndex:null;
      host.querySelectorAll('[data-qk-request-group]').forEach(node=>{const selected=request===Number(node.dataset.qkRequestGroup);node.classList.toggle('is-focused',selected);node.classList.toggle('is-dimmed',request!=null&&!selected);});
      const relatedQueries=new Set([...ids].map(id=>Math.floor(id/SPARSE_BLOCKS)));
      host.querySelectorAll('[data-qk-query]').forEach(node=>{const token=Number(node.dataset.qkQuery),active=s?.type==='query'&&token===s.tokenIndex||s?.type==='item'&&d.byId.get(s.id)?.token===token;node.classList.toggle('is-focused',active);node.classList.toggle('is-related',!active&&relatedQueries.has(token));node.setAttribute('tabindex',active||token===state.requestIndex*TOKENS_PER_REQUEST?'0':'-1');});
      const summaryNode=host.querySelector('[data-qk-summary]');if(summaryNode)summaryNode.textContent=summary(d);
      const requestSelect=host.querySelector('[data-qk-request]');if(requestSelect)requestSelect.value=String(state.requestIndex);
      host.querySelector('.csa-qk-drawer')?.remove();if(state.inspectorOpen&&s)host.querySelector('.csa-qk-canvas-shell')?.insertAdjacentHTML('beforeend',inspector(d));
      const back=host.querySelector('[data-qk-back]');if(back)back.hidden=!s&&state.scale===1;
      const inspectorButton=host.querySelector('[data-qk-inspector]');if(inspectorButton)inspectorButton.hidden=!s||state.inspectorOpen;
      if(locate)focusCamera(d);else camera();
    }
    function render(){const d=derive();host.innerHTML=shell(d);refreshSelection(d,!!state.selection);}
    function updatePhase(phase){if(!PHASES.includes(phase))return;if(morphCameraTimer){global.clearTimeout(morphCameraTimer);morphCameraTimer=null;}const oldPhase=state.phase;state.phase=phase;if(!state.selection){state.scale=1;state.panX=state.panY=0;state.manualCamera=false;}const d=derive(),positions=viewportPositions(d,phase);
      const bg=host.querySelector('[data-qk-backdrop]');if(bg)bg.innerHTML=backdrop(d);
      host.querySelectorAll('[data-qk-item]').forEach(node=>{const id=Number(node.dataset.qkItem),p=positions.get(id);if(!p)return;node.style.transform='translate('+p.x+'px,'+p.y+'px)';const rect=node.querySelector('rect');rect.setAttribute('width',p.w);rect.setAttribute('height',p.h);const label=node.querySelector('[data-qk-item-label]');if(label)label.textContent=phase==='ordering'?id+' · q'+d.queue.indexOf(id):String(id);});
      const title=host.querySelector('[data-qk-title]');if(title)title.textContent=PHASE_LABELS[phase];const sub=host.querySelector('[data-qk-subtitle]');if(sub)sub.textContent=subtitle(d);
      host.querySelectorAll('[data-qk-phase]').forEach(b=>{const active=b.dataset.qkPhase===phase;b.classList.toggle('is-selected',active);b.setAttribute('aria-pressed',String(active));});
      if(state.selection&&!reduced&&oldPhase!==phase){
        const saved={scale:state.scale,manualCamera:state.manualCamera},fit=fitBox(),old=viewportPositions(d,oldPhase),ids=selectionIds(d),points=[...ids].flatMap(id=>[old.get(id),positions.get(id)]).filter(Boolean);
        const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x+p.w)),minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y+p.h));
        state.scale=Math.min(saved.scale,fit.w/(maxX-minX+120),fit.h/(maxY-minY+100),1);state.panX=fit.x-(minX+maxX)/2;state.panY=fit.y-(minY+maxY)/2;
        refreshSelection(d,false);
        const identity=JSON.stringify(state.selection);
        morphCameraTimer=global.setTimeout(()=>{morphCameraTimer=null;if(state.phase!==phase||JSON.stringify(state.selection)!==identity)return;state.scale=saved.scale;state.manualCamera=saved.manualCamera;focusCamera(derive());},reduced?0:530);
      }else refreshSelection(d,!!state.selection);
    }
    function select(selection){stop();if(morphCameraTimer){global.clearTimeout(morphCameraTimer);morphCameraTimer=null;}state.selection=selection;state.inspectorOpen=false;const d=derive();
      if(selection?.type==='item'){const item=d.byId.get(selection.id);if(!item)return;const info=itemDetails(d,item.id);state.requestIndex=item.requestIndex;
        onSelection({type:'item',id:item.id,rank:state.rank,scenarioId:scenarioId(),requestIndex:item.requestIndex,tokenIndex:item.token,positionId:item.positionId,sparseBlockIndex:item.sparseBlock,valid:item.valid,validKvCount:item.validKvCount,queuePosition:info.queuePosition,laneId:info.block,laneIteration:info.iteration});
      }else if(selection?.type==='block')onSelection({type:'block',block:selection.block,rank:state.rank,scenarioId:scenarioId(),laneId:selection.block});
      else if(selection?.type==='request'){state.requestIndex=selection.requestIndex;onSelection({type:'request',requestIndex:selection.requestIndex,rank:state.rank,scenarioId:scenarioId()});}
      else if(selection?.type==='query'){state.requestIndex=selection.requestIndex;onSelection({type:'query',requestIndex:selection.requestIndex,tokenIndex:selection.tokenIndex,rank:state.rank,scenarioId:scenarioId()});}
      else onContext({rank:state.rank,scenarioId:scenarioId()});
      refreshSelection(d);
    }
    function changeContext(){stop();state.selection=null;state.inspectorOpen=false;state.requestIndex=0;state.overrides.clear();state.scale=1;state.panX=state.panY=0;state.manualCamera=false;onContext({rank:state.rank,scenarioId:scenarioId()});render();}
    // 外部（Pipeline 的 Rank 下拉）切 Rank：走同一条 changeContext，保证 Scenario / Focus / 选择一起重算
    function setRank(rank){if(rank==null||rank===state.rank)return;state.rank=rank;changeContext();}
    host.addEventListener('click',event=>{const b=event.target.closest('button,[data-qk-item],[data-qk-lane],[data-qk-query]');if(!b){if(event.target.closest('.csa-qk-canvas-viewport')&&state.selection)select(null);return;}
      if(b.dataset.qkItem!=null){select({type:'item',id:Number(b.dataset.qkItem)});return;}
      if(b.dataset.qkQuery!=null){const tokenIndex=Number(b.dataset.qkQuery);select({type:'query',tokenIndex,requestIndex:Math.floor(tokenIndex/TOKENS_PER_REQUEST)});return;}
      if(b.dataset.qkLane!=null){select({type:'block',block:Number(b.dataset.qkLane)});return;}
      if(b.dataset.qkPhase){stop();updatePhase(b.dataset.qkPhase);return;}
      if(b.hasAttribute('data-qk-play')){state.playing?stop():start();return;}
      if(b.hasAttribute('data-qk-replay')){stop();updatePhase('generation');start();return;}
      if(b.hasAttribute('data-qk-close')){state.inspectorOpen=false;refreshSelection(derive());return;}
      if(b.hasAttribute('data-qk-inspector')){state.inspectorOpen=true;refreshSelection(derive());return;}
      if(b.hasAttribute('data-qk-advanced')){state.advanced=!state.advanced;if(!state.advanced)state.overrides.clear();select(state.selection);return;}
      if(b.hasAttribute('data-qk-toggle')){const id=Number(b.dataset.qkToggle);if(state.advanced&&id%SPARSE_BLOCKS!==0){state.overrides.set(id,!derive().byId.get(id).valid);select(state.selection);}return;}
      if(b.hasAttribute('data-qk-reset')){state.overrides.clear();select(state.selection);return;}
      if(b.dataset.qkZoom){state.manualCamera=true;state.scale=Math.max(.7,Math.min(4,state.scale*Number(b.dataset.qkZoom)));camera();return;}
      if(b.hasAttribute('data-qk-fit')){state.manualCamera=false;state.scale=1;state.panX=state.panY=0;state.inspectorOpen=false;refreshSelection(derive(),false);return;}
      if(b.hasAttribute('data-qk-back')){state.manualCamera=false;state.scale=1;state.panX=state.panY=0;state.inspectorOpen=false;refreshSelection(derive(),false);return;}
      if(b.dataset.qkSource){const [file,a,z]=b.dataset.qkSource.split(':');stop();onSource(file,Number(a),Number(z));}
    });
    host.addEventListener('change',event=>{const t=event.target;
      if(t.matches('[data-qk-workload]')){state.workload=t.value;changeContext();}
      if(t.matches('[data-qk-position]')){state.position=Number(t.value);changeContext();}
      if(t.matches('[data-qk-rank]')){state.rank=Number(t.value);changeContext();}
      if(t.matches('[data-qk-request]'))select({type:'request',requestIndex:Number(t.value)});
      if(t.matches('[data-qk-loop]'))state.loop=t.checked;
    });
    host.addEventListener('keydown',event=>{if(event.key==='Escape'&&state.selection){select(null);}else if(event.key==='Enter'&&event.target.dataset.qkItem!=null)select({type:'item',id:Number(event.target.dataset.qkItem)});else if(event.key==='Enter'&&event.target.dataset.qkQuery!=null){const tokenIndex=Number(event.target.dataset.qkQuery);select({type:'query',tokenIndex,requestIndex:Math.floor(tokenIndex/TOKENS_PER_REQUEST)});}else if(event.key==='Enter'&&event.target.dataset.qkLane!=null)select({type:'block',block:Number(event.target.dataset.qkLane)});else if(['ArrowRight','ArrowLeft','ArrowDown','ArrowUp'].includes(event.key)&&event.target.dataset.qkQuery!=null){event.preventDefault();const tokenIndex=Math.max(0,Math.min(derive().items.length/SPARSE_BLOCKS-1,Number(event.target.dataset.qkQuery)+(event.key==='ArrowRight'||event.key==='ArrowDown'?1:-1)));select({type:'query',tokenIndex,requestIndex:Math.floor(tokenIndex/TOKENS_PER_REQUEST)});host.querySelector('[data-qk-query="'+tokenIndex+'"]').focus();}else if(['ArrowRight','ArrowLeft','ArrowDown','ArrowUp'].includes(event.key)&&event.target.dataset.qkItem!=null){event.preventDefault();const step=event.key==='ArrowRight'?1:event.key==='ArrowLeft'?-1:event.key==='ArrowDown'?5:-5;const id=Math.max(0,Math.min(derive().items.length-1,Number(event.target.dataset.qkItem)+step));select({type:'item',id});host.querySelector('[data-qk-item="'+id+'"]').focus();}});
    host.addEventListener('wheel',event=>{if(!event.target.closest('.csa-qk-canvas-viewport'))return;event.preventDefault();stop();state.manualCamera=true;state.scale=Math.max(.7,Math.min(4,state.scale*Math.exp(-event.deltaY*.001)));camera();},{passive:false});
    host.addEventListener('pointerover',event=>{const item=event.target.closest('[data-qk-item]'),lane=event.target.closest('[data-qk-lane]');if(state.playing&&(item||lane))stop();const tooltip=host.querySelector('[data-qk-tooltip]');if(!tooltip||(!item&&!lane))return;const d=derive();if(item){const data=d.byId.get(Number(item.dataset.qkItem)),info=itemDetails(d,data.id);tooltip.textContent='Work Item '+data.id+' · Q'+data.token+' / SB'+data.sparseBlock+' · qk_order['+info.queuePosition+'] · Lane '+info.block+' / it '+info.iteration+' · '+(data.valid?'valid':'invalid');}else{const b=d.blocks[Number(lane.dataset.qkLane)];tooltip.textContent='Lane '+b.index+' · qk_order['+b.index+'], ['+(b.index+d.blockCount)+'], … · '+b.entries.length+' Work Items';}tooltip.hidden=false;});
    host.addEventListener('pointerout',event=>{if(event.target.closest('[data-qk-item],[data-qk-lane]')&&!event.relatedTarget?.closest?.('[data-qk-item],[data-qk-lane]')){const tooltip=host.querySelector('[data-qk-tooltip]');if(tooltip)tooltip.hidden=true;}});
    host.addEventListener('pointerdown',event=>{if(event.button!==0||!event.target.closest('.csa-qk-canvas-viewport')||event.target.closest('[data-qk-item],[data-qk-lane]'))return;drag={x:event.clientX,y:event.clientY,panX:state.panX,panY:state.panY};});
    host.addEventListener('pointermove',event=>{const viewport=host.querySelector('.csa-qk-canvas-viewport'),tooltip=host.querySelector('[data-qk-tooltip]');if(tooltip&&!tooltip.hidden&&viewport){const r=viewport.getBoundingClientRect();tooltip.style.left=Math.min(r.width-220,event.clientX-r.left+12)+'px';tooltip.style.top=Math.max(8,event.clientY-r.top-32)+'px';}if(!drag||!viewport)return;const r=viewport.getBoundingClientRect(),fit=fitBox();state.manualCamera=true;state.panX=drag.panX+(event.clientX-drag.x)*(fit.w/state.scale)/r.width;state.panY=drag.panY+(event.clientY-drag.y)*(fit.h/state.scale)/r.height;camera();});
    host.addEventListener('pointerup',()=>{drag=null;});
    render();
    if(global.ResizeObserver)new global.ResizeObserver(()=>{if(!host.hidden){if(state.selection&&!state.manualCamera)focusCamera(derive());else camera();}}).observe(host);
    return {render,pause:()=>{stop();if(morphCameraTimer){global.clearTimeout(morphCameraTimer);morphCameraTimer=null;}},setSelection:selection=>{if(JSON.stringify(selection)===JSON.stringify(state.selection))return;state.selection=selection?{...selection}:null;state.inspectorOpen=false;if(selection?.type==='request'||selection?.type==='query')state.requestIndex=selection.requestIndex;else if(selection?.type==='item')state.requestIndex=Math.floor(selection.id/(SPARSE_BLOCKS*TOKENS_PER_REQUEST));refreshSelection(derive());},getState:()=>({mode:state.mode,workload:state.workload,position:state.position,rank:state.rank,requestIndex:state.requestIndex,phase:state.phase,selection:state.selection?{...state.selection}:null,inspectorOpen:state.inspectorOpen}),getScenario:derive,setRank};
  }

  global.PtoCsaQkWork = Object.freeze({
    create,
    buildWorkItems,
    buildQueue,
    assignToBlocks,
    getWorkItemDetails,
    getBlockDetails,
    DEFAULT_MASK,
    DEFAULT_SELECTION,
    syntheticTopk,
    deriveScenario,
    MODES,
    constants: { TOKENS, SOURCE_TOKENS, TOKENS_PER_REQUEST, SPARSE_BLOCKS, ITEM_COUNT, NUM_QK_CORES, WINDOW_SB, WIN, IDX_TOPK, ATTN_K_TILE, COMPRESS_RATIO, POSITIONS }
  });
})(typeof window !== 'undefined' ? window : globalThis);
