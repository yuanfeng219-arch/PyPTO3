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

  function traceRequestStarts(rank) {
    const r = global.PTO_CSA_QK_RUNTIME_TRACE?.ranks;
    const rd = r ? (r[String(rank ?? 0)] || r['0'] || null) : null;
    const starts = rd?.chain?.workProfile?.requestStarts;
    return Array.isArray(starts) && starts.length >= 32 ? starts : null;
  }

  function buildWorkItems(input) {
    const legacyMask = Array.isArray(input) ? input : null;
    const count = legacyMask ? legacyMask.length : (MODES[input.mode] || MODES.source).tokens;
    const traceStarts = !legacyMask && input.workload === 'trace' ? traceRequestStarts(input.rank) : null;
    const items = [];
    for (let token = 0; token < count; token++) {
      const requestIndex = Math.floor(token / TOKENS_PER_REQUEST);
      const basePosition = traceStarts
        ? traceStarts[requestIndex % traceStarts.length]
        : (input.workload === 'mixed' && input.mode === 'source' ? POSITIONS[requestIndex % POSITIONS.length] : input.position);
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
          scenarioId: legacyMask ? 'legacy' : input.mode + '-' + (traceStarts ? 'trace-r' + (input.rank ?? 0) : input.workload === 'mixed' ? 'mixed' : 'position-' + input.position),
          sparseBlock,
          valid,
          validKvCount,
          validityReason: legacyMask ? '教学 mask' : override == null ? sparseBlock === WINDOW_SB ? 'SB0：滑动窗口固定有效' : validKvCount ? (traceStarts ? 'Runtime Trace 输入（pos=' + positionId + '）在本块含 ' + validKvCount + ' 个有效压缩槽位' : 'Synthetic idx_topk 在本块含 ' + validKvCount + ' 个满足范围的索引') : (traceStarts ? 'Runtime Trace 输入（pos=' + positionId + '）在本块无有效压缩槽位' : '本块没有满足范围的 Synthetic idx_topk 索引') : 'Advanced Simulation 手动覆盖',
          originalPosition: token * SPARSE_BLOCKS + sparseBlock
        });
      }
    }
    return items;
  }

  function deriveScenario(input) {
    const items = buildWorkItems(input), queue = buildQueue(items, input.orderPolicy || 'source'), byId = new Map(items.map(item => [item.id, item]));
    const queuePos = new Map(queue.map((id, pos) => [id, pos]));
    const blockCount = MODES[input.mode]?.blocks ?? NUM_QK_CORES;
    const blocks = assignToBlocks(queue, blockCount);
    blocks.forEach(block => {
      let validCount = 0, sb0Count = 0, cmpCount = 0;
      for (const entry of block.entries) {
        const it = byId.get(entry.id);
        if (it?.valid) {
          validCount++;
          if (it.sparseBlock === WINDOW_SB) sb0Count++;
          else cmpCount++;
        }
      }
      block.validCount = validCount;
      block.invalidCount = block.entries.length - validCount;
      block.sb0Count = sb0Count;
      block.cmpCount = cmpCount;
    });
    return { items, queue, queuePos, blocks, blockCount, byId, validCount: items.filter(item => item.valid).length, input };
  }

  // qk_plan: pass 1 writes valid tiles, pass 2 appends invalid tiles to the tail.
  // orderPolicy='source' keeps (token, sb) row-major order; 'balanced' writes CMP (SB1..4) before SB0 (bit-exact).
  function buildQueue(items, orderPolicy = 'source') {
    const validItems = items.filter(item => item.valid);
    const valid = orderPolicy === 'balanced'
      ? [...validItems.filter(item => item.sparseBlock !== WINDOW_SB), ...validItems.filter(item => item.sparseBlock === WINDOW_SB)].map(item => item.id)
      : validItems.map(item => item.id);
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
    const onSelectAct = options.onSelectAct || (() => {}), onOpenPipeline = options.onOpenPipeline || (() => {}), onOpenGraph = options.onOpenGraph || (() => {});
    const hasRuntimeTrace = () => !!global.PTO_CSA_QK_RUNTIME_TRACE?.ranks;
    const SPEEDS = [0.25, 0.5, 1, 2];
    const state = {mode:options.mode || 'source', workload:hasRuntimeTrace() ? 'trace' : 'mixed', position:8192, rank:options.rank ?? 0,
      requestIndex:0, selection:options.selection || null, inspectorOpen:false, phase:'assignment', layout:'assignment', playing:false, loop:false, speed:1,
      scale:1, panX:0, panY:0, manualCamera:false, advanced:false, overrides:new Map(),
      colorBy:hasRuntimeTrace() ? 'kv-type' : 'validity', // 'validity' | 'kv-type'
      orderPolicy:'source', // 'source' | 'balanced'
      activeAct:options.activeAct || null};
    if(options.selection?.type==='request'||options.selection?.type==='query')state.requestIndex=options.selection.requestIndex;
    else if(options.selection?.type==='item')state.requestIndex=Math.floor(options.selection.id/(SPARSE_BLOCKS*TOKENS_PER_REQUEST));
    const visualUI = !!global.document?.body.classList.contains('csa-visual');
    const reduced = !!global.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const BASE_CELL = 12;
    let timer=null, drag=null, morphFrame=0, morphSerial=0, trailTimer=null, bridgeTimer=null, cameraFrame=0;
    let itemNodes=[], livePos=[], transitSpotlight=null;
    const cancelMorph=()=>{morphSerial++;if(morphFrame){cancelAnimationFrame(morphFrame);morphFrame=0;}};
    function applyTransitSpotlight(d,nextSpotlight){
      transitSpotlight=nextSpotlight||null;
      const active=state.selection?null:transitSpotlight;
      const count=d.items.length;
      if(itemNodes.length===count){
        for(let i=0;i<count;i++){
          const node=itemNodes[i];
          if(!node)continue;
          let dim=false;
          if(active){
            if(active.type==='request')dim=d.items[i].requestIndex!==active.index;
            else if(active.type==='lane')dim=(qPosOf(d,i)%d.blockCount)!==active.index;
          }
          node.classList.toggle('is-transit-dim',dim);
        }
      }
      host.querySelectorAll('[data-qk-order-col]').forEach(el=>el.classList.toggle('is-transit-dim',!!(active&&active.type==='lane'&&Number(el.dataset.qkOrderCol)!==active.index)));
      host.querySelectorAll('[data-qk-lane]').forEach(el=>el.classList.toggle('is-transit-dim',!!(active&&active.type==='lane'&&Number(el.dataset.qkLane)!==active.index)));
      host.querySelectorAll('[data-qk-request-group]').forEach(el=>el.classList.toggle('is-transit-dim',!!(active&&active.type==='request'&&Number(el.dataset.qkRequestGroup)!==active.index)));
    }
    const clearBridge=()=>{if(bridgeTimer){global.clearTimeout(bridgeTimer);bridgeTimer=null;}cancelMorph();if(transitSpotlight)applyTransitSpotlight(derive(),null);};
    const clamp01=v=>v<=0?0:v>=1?1:v;
    const smoothstep=t=>t*t*(3-2*t);
    const fmtTransform=(x,y,s)=>Math.abs(s-1)<0.002?'translate('+x.toFixed(1)+' '+y.toFixed(1)+')':'translate('+x.toFixed(1)+' '+y.toFixed(1)+') scale('+s.toFixed(3)+')';
    const cache=new Map();
    const scenarioId=()=>state.mode+'-'+(state.workload==='trace'&&traceRequestStarts(state.rank)?'trace-r'+state.rank:state.workload==='mixed'?'mixed':'position-'+state.position)+(state.orderPolicy==='balanced'?'-bal':'');
    const derive=()=>{const key=JSON.stringify([state.mode,state.workload,state.position,state.rank,state.orderPolicy,[...state.overrides]]);if(!cache.has(key)){cache.clear();cache.set(key,deriveScenario({mode:state.mode,workload:state.workload,position:state.position,rank:state.rank,orderPolicy:state.orderPolicy,overrides:state.overrides}));}return cache.get(key);};
    const qPosOf=(d,id)=>d.queuePos?.get(id)??d.queue.indexOf(id);
    const itemDetails=(d,id)=>{const queuePosition=qPosOf(d,id);return {id,token:Math.floor(id/SPARSE_BLOCKS),sparseBlock:id%SPARSE_BLOCKS,originalPosition:id,queuePosition,block:queuePosition%d.blockCount,iteration:Math.floor(queuePosition/d.blockCount),blockCount:d.blockCount,qkFlat:queuePosition};};
    const traceRoot=()=>global.PTO_CSA_QK_RUNTIME_TRACE||null;
    const traceRank=()=>{const r=traceRoot()?.ranks;return r?(r[String(state.rank)]||r['0']||null):null;};
    const physicalGroupOf=lane=>{const rd=traceRank();const linfo=rd?.chain?.workProfile?.lanes?.find(x=>x.lane===lane);return linfo?.physicalGroup??((lane%8)*3+Math.floor(lane/8));};
    const laneDurUs=(d,lane)=>{const rd=traceRank();const onTrace=hasRuntimeTrace()&&state.workload==='trace'&&rd;const b=d.blocks[lane];if(!b)return null;if(onTrace&&state.orderPolicy==='source'){const pg=physicalGroupOf(lane);const g=rd.groups?.find(x=>x.group===pg);if(g)return {us:g.aic.durUs,measured:true};}const fit=rd?.chain?.workProfile?.fittedCostUs;const sb0Us=fit?.sb0Us??7.9,cmpUs=fit?.cmpUs??40.0;return {us:b.sb0Count*sb0Us+b.cmpCount*cmpUs,measured:false};};
    const requestMeta=(d,reqIdx)=>{const items=d.items.filter(it=>it.requestIndex===reqIdx);const valid=items.filter(it=>it.valid);const invalid=items.filter(it=>!it.valid);const sb0=valid.filter(it=>it.sparseBlock===WINDOW_SB).length;const cmp=valid.length-sb0;const pos=items[0]?.positionId??null;const validSorted=valid.slice().sort((a,b)=>qPosOf(d,a.id)-qPosOf(d,b.id));const invalidSorted=invalid.slice().sort((a,b)=>qPosOf(d,a.id)-qPosOf(d,b.id));return {reqIdx,items,valid,invalid,sb0,cmp,pos,validSorted,invalidSorted};};
    // 默认选最能直观体现「1 绿 SB0 + 1 红 CMP」交替模式的 Request（如 Rank 0 的 Request 3、Rank 1 的 Request 1）
    const defaultTraceRequest=d=>{for(let r=0;r<32;r++){const m=requestMeta(d,r);if(m.sb0===8&&m.cmp===8)return r;}return state.requestIndex||0;};
    // 默认选一条重负载慢核 Lane（用于展示步长 +24 踩满 CMP 的领活路径）
    const defaultTraceLane=d=>{const slow=d.blocks.find(b=>b.cmpCount>=15);return slow?slow.index:0;};
    // 共享 1400 × 700 静态画布坐标系（5 种布局零 viewBox 跳变，所有矩阵严格正方形 cell × cell）：
    // - generation: 8×4 卡 (x: 44..1356, y: 34..666, cell=12)
    // - flatten / ordering: 40 列 × 32 行 (x: 330..1070, y: 58..650, cell=15)
    // - stride24: 24 列 × 54 行 (x: 563..837, y: 56..671, cell=9.5)
    // - assignment: 8×3 AI Core 卡 (x: 44..1356, y: 98..602, cell=12)
    const matrixLayout=(d,kind)=>{
      const source=state.mode==='source';
      if(!source){
        const cols=5,cell=66,step=86,offsetX=140,topY=92,rows=Math.ceil(d.queue.length/cols);
        return {cols,rows,cell,step,offsetX,topY,splitRow:rows-1,totalBottom:topY+rows*step+12};
      }
      if(kind==='stride24'){
        const cols=d.blockCount||NUM_QK_CORES, rows=Math.ceil(d.queue.length/cols);
        const cell=9.5, step=11.4, offsetX=564.15, topY=56;
        const splitRow=d.validCount>0?Math.floor((d.validCount-1)/cols):-1;
        return {cols,rows,cell,step,offsetX,topY,splitRow,totalBottom:topY+rows*step+10};
      }
      // flatten & ordering: 40 cols × 32 rows (identical grid so compaction animates in place)
      const cols=TOKENS_PER_REQUEST*SPARSE_BLOCKS; // 40
      const rows=Math.ceil(d.items.length/cols);   // 32
      const cell=15, step=18.5, offsetX=331.75, topY=58;
      const splitRow=d.validCount>0?Math.floor((d.validCount-1)/cols):-1;
      return {cols,rows,cell,step,offsetX,topY,splitRow,totalBottom:topY+rows*step+12};
    };
    const msFor=base=>Math.max(80,Math.round(base/state.speed));
    function stop(){if(timer)global.clearTimeout(timer);timer=null;clearBridge();state.playing=false;const b=host.querySelector('[data-qk-play]');if(b)b.textContent='播放';}
    function scheduleNextPlay(){
      if(!state.playing)return;
      if(timer)global.clearTimeout(timer);
      timer=global.setTimeout(()=>{
        timer=null;
        if(!state.playing)return;
        const next=PHASES.indexOf(state.phase)+1;
        if(next>=PHASES.length){
          if(state.loop)updatePhase('generation',{onComplete:scheduleNextPlay});
          else stop();
        }else{
          updatePhase(PHASES[next],{onComplete:scheduleNextPlay});
        }
      },msFor(800));
    }
    function start(){
      if(timer)global.clearTimeout(timer);
      timer=null; clearBridge();
      state.playing=true;
      const b=host.querySelector('[data-qk-play]');if(b)b.textContent='暂停';
      const next=PHASES.indexOf(state.phase)+1;
      if(next>=PHASES.length){
        updatePhase('generation',{onComplete:scheduleNextPlay});
      }else{
        updatePhase(PHASES[next],{onComplete:scheduleNextPlay});
      }
    }
    function viewportPositions(d,layout){
      const positions=new Map(),source=state.mode==='source';
      if(layout==='generation'){
        const cols=source?8:1, groupW=source?164:1100, groupH=source?(visualUI?158:124):610;
        const cell=source?(visualUI?12:11):65;
        d.items.forEach(item=>{const group=source?item.requestIndex:0,baseX=44+(group%cols)*groupW,baseY=34+Math.floor(group/cols)*groupH;
          const row=source?item.token%8:item.token,stepX=source?16:172,stepY=source?(visualUI?14.5:12):95;
          positions.set(item.id,{x:baseX+(source?66:90)+item.sparseBlock*stepX,y:baseY+(source?(visualUI?31:25):12)+row*stepY,w:cell,h:cell});});
      }else if(layout==='flatten'){
        const L=matrixLayout(d,'flatten');
        d.items.forEach(item=>{
          const r=item.requestIndex, c=(item.token%TOKENS_PER_REQUEST)*SPARSE_BLOCKS+item.sparseBlock;
          positions.set(item.id,{x:L.offsetX+c*L.step,y:L.topY+r*L.step,w:L.cell,h:L.cell});
        });
      }else if(layout==='ordering'){
        const L=matrixLayout(d,'ordering');
        d.queue.forEach((id,p)=>{const c=p%L.cols,r=Math.floor(p/L.cols);positions.set(id,{x:L.offsetX+c*L.step,y:L.topY+r*L.step,w:L.cell,h:L.cell});});
      }else if(layout==='stride24'){
        const L=matrixLayout(d,'stride24');
        d.queue.forEach((id,p)=>{const c=p%L.cols,r=Math.floor(p/L.cols);positions.set(id,{x:L.offsetX+c*L.step,y:L.topY+r*L.step,w:L.cell,h:L.cell});});
      }else{
        const cols=source?(visualUI?8:12):4,laneW=source?(visualUI?164:110):320,laneH=source?(visualUI?168:240):540;
        const cell=source?(visualUI?12:8):70;
        d.blocks.forEach(block=>block.entries.forEach(entry=>{const x=44+(block.index%cols)*laneW,y=98+Math.floor(block.index/cols)*laneH;
          const cellCol=source?entry.iteration%9:entry.iteration%3,cellRow=source?Math.floor(entry.iteration/9):Math.floor(entry.iteration/3);
          positions.set(entry.id,{x:x+(source&&visualUI?22:7)+cellCol*(source?(visualUI?14.5:10):92),y:y+(source?(visualUI?44:52):40)+cellRow*(source?(visualUI?14.5:28):92),w:cell,h:cell});}));
      }
      return positions;
    }
    function backdrop(d,layout=state.layout){
      const source=state.mode==='source';
      const showType=state.colorBy==='kv-type';
      const showPhys=hasRuntimeTrace()&&state.workload==='trace';
      if(layout==='generation'){
        const cols=source?8:1,groupW=source?164:1100,groupH=source?(visualUI?158:124):610;
        return Array.from({length:source?32:1},(_,group)=>{const x=44+(group%cols)*groupW,y=34+Math.floor(group/cols)*groupH;
          const firstItem=d.items[group*TOKENS_PER_REQUEST*SPARSE_BLOCKS];
          const posTag=source&&firstItem?.positionId!=null?'<text class="csa-qk-req-pos" x="'+(x+groupW-18)+'" y="'+(y+14)+'" text-anchor="end">pos '+firstItem.positionId+'</text>':'';
          const sbCols=source?Array.from({length:SPARSE_BLOCKS},(_,sb)=>'<text class="csa-qk-sb-col'+(sb===0?' is-sb0':'')+'" x="'+(x+66+sb*16+6)+'" y="'+(y+(visualUI?26:21))+'" text-anchor="middle">S'+sb+'</text>').join(''):'';
          const queryLabels=source?Array.from({length:8},(_,offset)=>'<text class="csa-qk-query-label" data-qk-query="'+(group*8+offset)+'" role="button" tabindex="-1" x="'+(x+8)+'" y="'+(y+(visualUI?40:33)+offset*(visualUI?14.5:12))+'">Q'+(group*8+offset)+'</text>').join(''):Array.from({length:5},(_,token)=>'<text class="csa-qk-query-label" data-qk-query="'+token+'" role="button" tabindex="-1" x="'+(x+12)+'" y="'+(y+49+token*95)+'">Q'+token+'</text>').join('');
          return '<g class="csa-qk-canvas-group" data-qk-request-group="'+group+'" role="button" tabindex="0" aria-label="点击追踪 Request '+group+(firstItem?.positionId!=null?' (pos='+firstItem.positionId+')':'')+'"><rect x="'+x+'" y="'+y+'" width="'+(groupW-10)+'" height="'+(groupH-6)+'" rx="5" vector-effect="non-scaling-stroke"/><text class="csa-qk-req-title" x="'+(x+8)+'" y="'+(y+14)+'">Request '+group+'</text>'+posTag+sbCols+queryLabels+'</g>';}).join('');
      }
      if(layout==='flatten'){
        const L=matrixLayout(d,'flatten'), gridW=L.cols*L.step-(L.step-L.cell), gridH=L.rows*L.step-(L.step-L.cell);
        const selReq=state.selection?.type==='request'||state.selection?.type==='query'?state.selection.requestIndex:null;
        const qHeaders=Array.from({length:TOKENS_PER_REQUEST},(_,qi)=>{
          const gx=L.offsetX+qi*SPARSE_BLOCKS*L.step, gw=SPARSE_BLOCKS*L.step-(L.step-L.cell);
          const sep=qi>0?'<line class="csa-qk-flat-sep" x1="'+(gx-1.8)+'" y1="'+(L.topY-4)+'" x2="'+(gx-1.8)+'" y2="'+(L.topY+gridH+4)+'"/>':'';
          return sep+'<rect class="csa-qk-ord-col-head" x="'+gx+'" y="34" width="'+gw+'" height="18" rx="3"/><text class="csa-qk-ord-col-id" x="'+(gx+gw/2)+'" y="46" text-anchor="middle">Q+'+qi+' (S0–S4)</text>';
        }).join('');
        const rowLabels=Array.from({length:L.rows},(_,r)=>{
          const firstItem=d.items[r*TOKENS_PER_REQUEST*SPARSE_BLOCKS];
          const cy=L.topY+r*L.step+L.cell/2+3;
          const isFoc=selReq===r;
          const hi=isFoc?'<rect class="csa-qk-flat-row-hi" x="'+(L.offsetX-4)+'" y="'+(L.topY+r*L.step-1.8)+'" width="'+(gridW+8)+'" height="'+(L.cell+3.6)+'" rx="3"/>':'';
          return hi+'<text class="csa-qk-ord-row-lbl'+(isFoc?' is-emph':' is-pass1')+'" x="'+(L.offsetX-10)+'" y="'+cy+'" text-anchor="end">R'+r+(firstItem?.positionId!=null?' · p'+firstItem.positionId:'')+'</text>';
        }).join('');
        return '<text class="csa-qk-canvas-axis" x="700" y="22" text-anchor="middle">展开矩阵（40 列 × 32 行）：每行 = 1 个 Request 的 8 Query × 5 SB（40 格），接下来无效灰块移至队尾、有效彩块向前补齐</text>'+
          qHeaders+rowLabels;
      }
      if(layout==='ordering'){
        const L=matrixLayout(d,'ordering'), boundary=d.validCount, invalidCount=d.items.length-boundary;
        const gridW=L.cols*L.step-(L.step-L.cell), gridH=L.rows*L.step-(L.step-L.cell);
        const orderHint=state.orderPolicy==='balanced'
          ?'优化入队：红块 CMP 连续在前，绿块 SB0 连续在中，灰块在尾'
          :'源码行主序：无效灰块移至下方队尾，有效彩块向前补齐';
        if(!source){
          return '<rect class="csa-qk-order-valid" x="128" y="75" width="1100" height="'+(Math.ceil(boundary/L.cols)*L.step+12)+'" rx="5"/>'+
            '<text class="csa-qk-canvas-axis" x="140" y="64">有效区 '+boundary+' 项　无效队尾 '+invalidCount+' 项</text>';
        }
        const colHeaders=Array.from({length:L.cols},(_,c)=>{
          const cx=L.offsetX+c*L.step;
          return '<text class="csa-qk-ord-col-tick'+(c%2===0?' is-even':'')+'" x="'+(cx+L.cell/2)+'" y="48" text-anchor="middle">+'+c+'</text>';
        }).join('');
        const rowLabels=Array.from({length:L.rows},(_,r)=>{
          const startIdx=r*L.cols, endIdx=Math.min(d.queue.length-1,startIdx+L.cols-1);
          const inValid=r<=L.splitRow;
          const cy=L.topY+r*L.step+L.cell/2+3;
          return '<text class="csa-qk-ord-row-lbl'+(inValid?' is-pass1':' is-pass2')+'" x="'+(L.offsetX-10)+'" y="'+cy+'" text-anchor="end">'+startIdx+'–'+endIdx+'</text>';
        }).join('');
        const validBottom=L.splitRow>=0?L.topY+(L.splitRow+1)*L.step-(L.step-L.cell):L.topY;
        const validBox=L.splitRow>=0
          ?'<rect class="csa-qk-order-valid" x="'+(L.offsetX-5)+'" y="'+(L.topY-4)+'" width="'+(gridW+10)+'" height="'+(validBottom-L.topY+8)+'" rx="5"/>'
          :'';
        let cutHtml='';
        if(boundary>0&&boundary<d.items.length){
          const bc=boundary%L.cols, br=Math.floor(boundary/L.cols);
          const cutX=L.offsetX+bc*L.step-(L.step-L.cell)/2;
          const cutY=L.topY+br*L.step;
          cutHtml='<line class="csa-qk-ord-cut" x1="'+cutX+'" y1="'+(cutY-2)+'" x2="'+cutX+'" y2="'+(cutY+L.cell+2)+'"/>';
        }
        const bx=L.offsetX+gridW+14;
        const p1Mid=L.splitRow>=0?(L.topY+validBottom)/2:L.topY;
        const p2Top=validBottom+6, p2Bottom=L.topY+gridH, p2Mid=(p2Top+p2Bottom)/2;
        const brackets=
          (L.splitRow>=0
            ?'<path class="csa-qk-ord-bracket is-pass1" d="M'+bx+','+L.topY+' h6 V'+validBottom+' h-6"/>'+
             '<text class="csa-qk-ord-bkt-title is-pass1" x="'+(bx+12)+'" y="'+(p1Mid-6)+'">有效区</text>'+
             '<text class="csa-qk-ord-bkt-sub" x="'+(bx+12)+'" y="'+(p1Mid+7)+'">0–'+(boundary-1)+' ('+boundary+' 项)</text>'+
             '<text class="csa-qk-ord-bkt-sub" x="'+(bx+12)+'" y="'+(p1Mid+19)+'">'+(state.orderPolicy==='balanced'?'先红 CMP 后绿 SB0':'绿/红交替排列')+'</text>'
            :'')+
          (invalidCount>0
            ?'<path class="csa-qk-ord-bracket is-pass2" d="M'+bx+','+p2Top+' h6 V'+p2Bottom+' h-6"/>'+
             '<text class="csa-qk-ord-bkt-title is-pass2" x="'+(bx+12)+'" y="'+(p2Mid-4)+'">无效队尾</text>'+
             '<text class="csa-qk-ord-bkt-sub" x="'+(bx+12)+'" y="'+(p2Mid+9)+'">'+boundary+'–'+(d.items.length-1)+' ('+invalidCount+' 项)</text>'
            :'');
        return '<text class="csa-qk-canvas-axis" x="700" y="22" text-anchor="middle">排序队列 qk_order[0..'+(d.items.length-1)+']（40 × 32 正方形矩阵） · '+orderHint+'</text>'+
          validBox+colHeaders+rowLabels+cutHtml+brackets;
      }
      if(layout==='stride24'){
        const L=matrixLayout(d,'stride24'), boundary=d.validCount, invalidCount=d.items.length-boundary;
        const gridW=L.cols*L.step-(L.step-L.cell), gridH=L.rows*L.step-(L.step-L.cell);
        const selLane=state.selection?.type==='block'?state.selection.block:null;
        const colHtml=d.blocks.map(block=>{
          const c=block.index, cx=L.offsetX+c*L.step, midX=cx+L.cell/2, isEven=c%2===0;
          const isSlow=showType&&state.orderPolicy==='source'&&block.cmpCount>=15;
          const isFast=showType&&state.orderPolicy==='source'&&block.cmpCount<15;
          const isSel=selLane===c, isDim=selLane!=null&&!isSel;
          const colCls='csa-qk-ord-col'+(isEven?' is-even':' is-odd')+(isSlow?' is-slow':isFast?' is-fast':'')+(isSel?' is-selected':'')+(isDim?' is-dimmed':'');
          const hy=isEven?39:49;
          return '<g class="'+colCls+'" data-qk-order-col="'+c+'" role="button" tabindex="0" aria-label="AI Core Lane '+c+' ('+(isEven?'偶':'奇')+') · '+block.sb0Count+' SB0 + '+block.cmpCount+' CMP">'+
            '<rect class="csa-qk-ord-col-strip" x="'+(cx-1)+'" y="'+(L.topY-3)+'" width="'+(L.cell+2)+'" height="'+(gridH+6)+'" rx="2"/>'+
            '<rect class="csa-qk-ord-col-head" x="'+(midX-7.5)+'" y="'+(hy-8)+'" width="15" height="10" rx="2"/>'+
            '<text class="csa-qk-stride-col-id" x="'+midX+'" y="'+hy+'" text-anchor="middle">'+String(c).padStart(2,'0')+'</text>'+
          '</g>';
        }).join('');
        const rowLabels=[];
        for(let r=0;r<L.rows;r++){
          if(r%3!==0&&r!==L.splitRow&&r!==L.rows-1)continue;
          const startIdx=r*L.cols, endIdx=Math.min(d.queue.length-1,startIdx+L.cols-1);
          const cy=L.topY+r*L.step+L.cell/2+2.5;
          rowLabels.push('<text class="csa-qk-ord-row-lbl'+(r<=L.splitRow?' is-pass1':' is-pass2')+'" x="'+(L.offsetX-8)+'" y="'+cy+'" text-anchor="end">'+startIdx+'–'+endIdx+'</text>');
        }
        const validBottom=L.splitRow>=0?L.topY+(L.splitRow+1)*L.step-(L.step-L.cell):L.topY;
        const validBox=L.splitRow>=0?'<rect class="csa-qk-order-valid" x="'+(L.offsetX-4)+'" y="'+(L.topY-3)+'" width="'+(gridW+8)+'" height="'+(validBottom-L.topY+6)+'" rx="4"/>':'';
        let cutHtml='';
        if(boundary>0&&boundary<d.items.length){
          const bc=boundary%L.cols, br=Math.floor(boundary/L.cols);
          const cutX=L.offsetX+bc*L.step-(L.step-L.cell)/2;
          const cutY=L.topY+br*L.step;
          cutHtml='<line class="csa-qk-ord-cut" x1="'+cutX+'" y1="'+(cutY-2)+'" x2="'+cutX+'" y2="'+(cutY+L.cell+2)+'"/>';
        }
        const bx=L.offsetX+gridW+12, p1Mid=(L.topY+validBottom)/2, p2Top=validBottom+6, p2Bottom=L.topY+gridH, p2Mid=(p2Top+p2Bottom)/2;
        const brackets=
          '<path class="csa-qk-ord-bracket is-pass1" d="M'+bx+','+L.topY+' h5 V'+validBottom+' h-5"/>'+
          '<text class="csa-qk-ord-bkt-title is-pass1" x="'+(bx+10)+'" y="'+(p1Mid-4)+'">有效区</text>'+
          '<text class="csa-qk-ord-bkt-sub" x="'+(bx+10)+'" y="'+(p1Mid+8)+'">'+boundary+' 项</text>'+
          '<path class="csa-qk-ord-bracket is-pass2" d="M'+bx+','+p2Top+' h5 V'+p2Bottom+' h-5"/>'+
          '<text class="csa-qk-ord-bkt-title is-pass2" x="'+(bx+10)+'" y="'+(p2Mid-4)+'">无效队尾</text>'+
          '<text class="csa-qk-ord-bkt-sub" x="'+(bx+10)+'" y="'+(p2Mid+8)+'">'+invalidCount+' 项</text>';
        return '<text class="csa-qk-canvas-axis" x="700" y="20" text-anchor="middle">按 24 核步长折行（24 列 × 54 行正方形矩阵）：每列 = 1 颗 AI Core（Lane 00–23，纵向步长 +24）</text>'+
          validBox+colHtml+rowLabels.join('')+cutHtml+brackets;
      }
      const cols=source?(visualUI?8:12):4,laneW=source?(visualUI?164:110):320,laneH=source?(visualUI?168:240):540;
      const maxWeight=Math.max(1,...d.blocks.map(b=>b.sb0Count+b.cmpCount*5));
      return d.blocks.map(block=>{const x=44+(block.index%cols)*laneW,y=98+Math.floor(block.index/cols)*laneH;
        const pg=physicalGroupOf(block.index);
        const tierCls=showType&&state.orderPolicy==='source'?(block.cmpCount>=15?' is-tier-slow':' is-tier-fast'):'';
        const countText=block.validCount+' valid';
        const physTag=showPhys?'<text class="csa-qk-canvas-phys" x="'+(x+laneW-14)+'" y="'+(y+18)+'" text-anchor="end">核组 '+String(pg).padStart(2,'0')+'</text>':'';
        const iterAxis=source&&visualUI?Array.from({length:6},(_,r)=>'<text class="csa-qk-lane-it" x="'+(x+18)+'" y="'+(y+53+r*14.5)+'" text-anchor="end">'+(r*9)+'</text>').join(''):'';
        let weightStrip='';
        if(source&&visualUI&&showType){
          const trackW=96, sbW=(block.sb0Count/maxWeight)*trackW, cmpW=(block.cmpCount*5/maxWeight)*trackW;
          const dur=laneDurUs(d,block.index);
          const durLbl=dur?(dur.measured?Math.round(dur.us)+' µs':'~'+Math.round(dur.us)+' µs'):'';
          weightStrip=
            '<rect class="csa-qk-lane-wtrack" x="'+(x+10)+'" y="'+(y+141)+'" width="'+trackW+'" height="4" rx="2"/>'+
            (sbW>0.4?'<rect class="csa-qk-lane-wsb0" x="'+(x+10)+'" y="'+(y+141)+'" width="'+sbW.toFixed(1)+'" height="4" rx="2"/>':'')+
            (cmpW>0.4?'<rect class="csa-qk-lane-wcmp" x="'+(x+10+sbW).toFixed(1)+'" y="'+(y+141)+'" width="'+cmpW.toFixed(1)+'" height="4" rx="2"/>':'')+
            '<text class="csa-qk-lane-wdur" x="'+(x+laneW-14)+'" y="'+(y+146)+'" text-anchor="end">'+durLbl+'</text>';
        }
        return '<g class="csa-qk-canvas-lane'+tierCls+'" data-qk-lane="'+block.index+'" role="button" tabindex="0" aria-label="Lane '+block.index+(showPhys?' 核组 '+String(pg).padStart(2,'0'):'')+' · '+block.validCount+' valid"><rect x="'+x+'" y="'+y+'" width="'+(laneW-8)+'" height="'+(laneH-12)+'" rx="5" vector-effect="non-scaling-stroke"/><text x="'+(x+10)+'" y="'+(y+18)+'">Lane '+String(block.index).padStart(2,'0')+'</text>'+physTag+'<text class="csa-qk-canvas-count" x="'+(x+10)+'" y="'+(y+34)+'">'+countText+'</text>'+iterAxis+weightStrip+'</g>';}).join('');
    }
    function overlay(d,positions,layout=state.layout){
      const source=state.mode==='source', s=state.selection;
      if(!source)return '';
      const out=[];
      const cellLabel=(p,txt,cls='')=>'<text class="csa-qk-cell-num '+cls+'" x="'+(p.x+p.w/2).toFixed(1)+'" y="'+(p.y+p.h/2+2.3).toFixed(1)+'" text-anchor="middle">'+txt+'</text>';
      if(layout==='generation'||layout==='flatten'){
        if(s?.type==='request'||s?.type==='query'){
          const reqIdx=s.requestIndex;
          const m=requestMeta(d,reqIdx);
          const subset=s.type==='query'?m.items.filter(it=>it.token===s.tokenIndex):m.items;
          const validOrdered=subset.filter(it=>it.valid).sort((a,b)=>qPosOf(d,a.id)-qPosOf(d,b.id));
          if(layout==='generation'&&validOrdered.length>1){
            const pts=validOrdered.map(it=>{const p=positions.get(it.id);return (p.x+p.w/2).toFixed(1)+','+(p.y+p.h/2).toFixed(1);}).join(' ');
            const p0=positions.get(validOrdered[0].id);
            out.push('<polyline class="csa-qk-scan-path" points="'+pts+'"/>');
            out.push('<circle class="csa-qk-scan-start" cx="'+(p0.x+p0.w/2).toFixed(1)+'" cy="'+(p0.y+p0.h/2).toFixed(1)+'" r="2.2"/>');
          }
          subset.forEach(it=>{
            const p=positions.get(it.id), qp=qPosOf(d,it.id);
            if(!p)return;
            out.push(cellLabel(p,String(qp),it.valid?'is-gen-valid':'is-gen-invalid'));
          });
        }else if(s?.type==='item'){
          const p=positions.get(s.id);
          if(p)out.push(cellLabel(p,String(qPosOf(d,s.id)),'is-gen-valid'));
        }
        return out.join('');
      }
      if(layout==='ordering'||layout==='stride24'){
        const L=matrixLayout(d,layout);
        if(!s){
          // Overview numbering: row 0 (0..cols-1) + col 0 (0, cols, 2*cols...) + boundary cell
          for(let c=0;c<L.cols&&c<d.queue.length;c++){
            const id=d.queue[c], p=positions.get(id), it=d.byId.get(id);
            if(p)out.push(cellLabel(p,String(c),it?.valid?'is-sq-valid':'is-sq-invalid'));
          }
          for(let r=1;r<=L.splitRow;r++){
            if(layout==='stride24'&&r%2!==0)continue;
            const idx=r*L.cols, id=d.queue[idx], p=positions.get(id), it=d.byId.get(id);
            if(p)out.push(cellLabel(p,String(idx),it?.valid?'is-sq-valid':'is-sq-invalid'));
          }
          if(d.validCount>0&&d.validCount<d.queue.length){
            const lastValidIdx=d.validCount-1, p=positions.get(d.queue[lastValidIdx]);
            if(p&&lastValidIdx>=L.cols&&lastValidIdx%L.cols!==0)out.push(cellLabel(p,String(lastValidIdx),'is-sq-valid'));
          }
        }else if(s.type==='request'||s.type==='query'){
          const items=s.type==='query'?d.items.filter(it=>it.token===s.tokenIndex):d.items.filter(it=>it.requestIndex===s.requestIndex);
          const validOrd=items.filter(it=>it.valid).sort((a,b)=>qPosOf(d,a.id)-qPosOf(d,b.id));
          const invOrd=items.filter(it=>!it.valid).sort((a,b)=>qPosOf(d,a.id)-qPosOf(d,b.id));
          for(let i=0;i<validOrd.length-1;i++){
            const qa=qPosOf(d,validOrd[i].id), qb=qPosOf(d,validOrd[i+1].id);
            if(qb===qa+1&&Math.floor(qa/L.cols)===Math.floor(qb/L.cols)){
              const pa=positions.get(validOrd[i].id), pb=positions.get(validOrd[i+1].id);
              if(pa&&pb)out.push('<line class="csa-qk-ord-seq-link" x1="'+(pa.x+pa.w)+'" y1="'+(pa.y+pa.h/2)+'" x2="'+pb.x+'" y2="'+(pb.y+pb.h/2)+'"/>');
            }
          }
          validOrd.forEach(it=>{const p=positions.get(it.id), qp=qPosOf(d,it.id);if(p)out.push(cellLabel(p,String(qp),'is-sq-emph'));});
          if(invOrd.length){
            [invOrd[0],invOrd.at(-1)].forEach(it=>{const p=positions.get(it.id), qp=qPosOf(d,it.id);if(p)out.push(cellLabel(p,String(qp),'is-sq-inv-emph'));});
          }
        }else if(s.type==='block'){
          const b=d.blocks[s.block];
          if(b&&b.entries.length){
            const validEntries=b.entries.filter(e=>d.byId.get(e.id)?.valid);
            if(layout==='stride24'){
              const colX=L.offsetX+b.index*L.step+L.cell/2;
              const firstP=positions.get(b.entries[0].id), lastValidP=validEntries.length?positions.get(validEntries.at(-1).id):firstP;
              if(firstP&&lastValidP)out.push('<line class="csa-qk-stride-rail" x1="'+colX+'" y1="'+(firstP.y+firstP.h/2)+'" x2="'+colX+'" y2="'+(lastValidP.y+lastValidP.h/2)+'"/>');
              for(let i=0;i<Math.min(3,validEntries.length-1);i++){
                const p1=positions.get(validEntries[i].id), p2=positions.get(validEntries[i+1].id);
                if(p1&&p2){
                  const midY=(p1.y+p1.h+p2.y)/2;
                  const badgeX=b.index>=L.cols-2?p1.x-16:p1.x+p1.w+14;
                  out.push('<text class="csa-qk-stride-step" x="'+badgeX+'" y="'+(midY+2.5)+'" text-anchor="middle">+24</text>');
                }
              }
            }
            validEntries.forEach(e=>{const p=positions.get(e.id);if(p)out.push(cellLabel(p,String(e.queuePosition),'is-sq-emph'));});
            const tailEntries=b.entries.filter(e=>!d.byId.get(e.id)?.valid);
            if(tailEntries.length){
              [tailEntries[0],tailEntries.at(-1)].forEach(e=>{const p=positions.get(e.id);if(p)out.push(cellLabel(p,String(e.queuePosition),'is-sq-inv-emph'));});
            }
          }
        }else if(s.type==='item'){
          const p=positions.get(s.id), info=itemDetails(d,s.id);
          if(p)out.push(cellLabel(p,String(info.queuePosition),'is-sq-emph'));
        }
        return out.join('');
      }
      // layout === 'assignment'
      if(s?.type==='block'){
        const b=d.blocks[s.block];
        if(b){
          const row0=b.entries.slice(0,Math.min(9,b.entries.length));
          if(row0.length>1){
            const pts=row0.map(e=>{const p=positions.get(e.id);return (p.x+p.w/2).toFixed(1)+','+(p.y+p.h/2).toFixed(1);}).join(' ');
            out.push('<polyline class="csa-qk-scan-path" points="'+pts+'"/>');
          }
          b.entries.forEach(e=>{
            const p=positions.get(e.id), it=d.byId.get(e.id);
            if(!p)return;
            if(it?.valid||e.iteration===b.validCount||e.iteration===b.entries.length-1){
              out.push(cellLabel(p,String(e.queuePosition),it?.valid?'is-assign-valid':'is-assign-invalid'));
            }
          });
        }
      }else if(s?.type==='request'||s?.type==='query'){
        const ids=selectionIds(d);
        ids.forEach(id=>{
          const it=d.byId.get(id), p=positions.get(id);
          if(p&&it?.valid)out.push(cellLabel(p,String(qPosOf(d,id)),'is-assign-valid'));
        });
      }else if(s?.type==='item'){
        const p=positions.get(s.id);
        if(p)out.push(cellLabel(p,String(qPosOf(d,s.id)),'is-assign-valid'));
      }
      return out.join('');
    }
    function renderMotionTrails(d,oldPhase,newPhase){
      const layer=host.querySelector('[data-qk-trails]');
      if(!layer)return;
      if(trailTimer){global.clearTimeout(trailTimer);trailTimer=null;}
      layer.innerHTML='';
      if(reduced||oldPhase===newPhase||!state.selection)return;
      const oldPos=viewportPositions(d,oldPhase), newPos=viewportPositions(d,newPhase);
      const ids=[...selectionIds(d)].filter(id=>d.byId.get(id)?.valid).slice(0,32);
      if(!ids.length)return;
      const paths=ids.map(id=>{
        const a=oldPos.get(id), b=newPos.get(id), it=d.byId.get(id);
        if(!a||!b)return '';
        const x1=a.x+a.w/2, y1=a.y+a.h/2, x2=b.x+b.w/2, y2=b.y+b.h/2;
        const mx=(x1+x2)/2, my=(y1+y2)/2-Math.min(48,Math.hypot(x2-x1,y2-y1)*0.12);
        const cls=it?.sparseBlock===WINDOW_SB?'is-sb0':'is-cmp';
        return '<path class="csa-qk-trail '+cls+'" d="M'+x1.toFixed(1)+','+y1.toFixed(1)+' Q'+mx.toFixed(1)+','+my.toFixed(1)+' '+x2.toFixed(1)+','+y2.toFixed(1)+'"/>';
      }).join('');
      layer.innerHTML=paths;
      layer.classList.remove('is-fading');
      trailTimer=global.setTimeout(()=>{layer.classList.add('is-fading');trailTimer=global.setTimeout(()=>{layer.innerHTML='';layer.classList.remove('is-fading');trailTimer=null;},450);},1250);
    }
    function buildTraceStripHtml(d){
      const s=state.selection, phase=state.phase, layout=state.layout;
      const isOverview=!s;
      const isReq=s?.type==='request'||s?.type==='query';
      const isLane=s?.type==='block';
      const isItem=s?.type==='item';
      const defReq=defaultTraceRequest(d), defLane=defaultTraceLane(d);
      const curReq=isReq?s.requestIndex:(isItem?d.byId.get(s.id)?.requestIndex??defReq:defReq);
      const curLane=isLane?s.block:(isItem?itemDetails(d,s.id).block:defLane);
      const showLaneMode=phase==='assignment'||layout==='stride24'||isLane;

      let copyHtml='';
      let ctaHtml='';

      if(layout==='flatten'){
        copyHtml='<span class="csa-qk-trace-msg"><b>过渡 1/2 · 横向展开为 40 × 32 矩阵</b>：32 个 Request 各占 1 行（每行 8 Query × 5 SB = 40 格），接下来把每行灰块移到下方队尾、有效彩块向前补齐</span>';
        ctaHtml='<button type="button" class="btn btn-sm csa-qk-trace-cta" data-qk-preview-layout="ordering">完成压实 → Ordering (40×32) →</button>';
      }else if(layout==='stride24'){
        if(isLane){
          const b=d.blocks[curLane], pg=physicalGroupOf(curLane), par=curLane%2===0?'偶':'奇';
          const sampleIdx=b.entries.slice(0,5).map(e=>'<b>#'+e.queuePosition+'</b>').join(', ');
          copyHtml='<span class="csa-qk-trace-msg"><b>24 × 54 步长折行 · 列 '+String(curLane).padStart(2,'0')+' · 核组 '+String(pg).padStart(2,'0')+'</b>：按步长 +24 依次领取 '+sampleIdx+'…（<b>'+b.validCount+' valid</b>）</span>';
        }else{
          copyHtml='<span class="csa-qk-trace-msg"><b>过渡 1/2 · 折行为 24 × 54 矩阵</b>：从 40 × 32 队列按 24 列重排，每列即 1 颗 AI Core 的步长 +24 队列，接下来每列移入对应 AI Core 格子</span>';
        }
        ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-preview-layout="ordering">← 回 40×32 排序队列</button><button type="button" class="btn btn-sm csa-qk-trace-cta" data-qk-preview-layout="assignment">每列移入 AI Core 格子 →</button>';
      }else if(isReq){
        const m=requestMeta(d,curReq);
        const first4=m.validSorted.slice(0,4).map(it=>{
          const qp=qPosOf(d,it.id), lane=qp%d.blockCount, par=lane%2===0?'偶':'奇';
          const dot=it.sparseBlock===WINDOW_SB?'<i class="csa-qk-dot is-sb0"></i>':'<i class="csa-qk-dot is-cmp"></i>';
          const tail=phase==='assignment'?' (L'+String(lane).padStart(2,'0')+' '+par+')':'';
          return '<span class="csa-qk-map-chip">'+dot+'Q'+it.token+'·S'+it.sparseBlock+' → <b>#'+qp+'</b>'+tail+'</span>';
        }).join('');
        const invRange=m.invalidSorted.length?' · 灰块 '+m.invalidSorted.length+' 项→队尾 #'+qPosOf(d,m.invalidSorted[0].id)+'–#'+qPosOf(d,m.invalidSorted.at(-1).id):'';
        copyHtml='<span class="csa-qk-trace-msg"><b>Request '+curReq+'</b>'+(m.pos!=null?' (pos='+m.pos+')':'')+' · '+m.valid.length+' 有效 ('+m.sb0+'绿+'+m.cmp+'红)：'+first4+invRange+'</span>';
        if(phase==='generation')ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-preview-layout="flatten">先看 40×32 展开</button><button type="button" class="btn btn-sm csa-qk-trace-cta" data-qk-guided-phase="ordering">保持 Request '+curReq+' 进 Ordering →</button>';
        else if(phase==='ordering')ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-guided-phase="generation">← 回 Generation</button><button type="button" class="btn btn-sm csa-qk-trace-cta" data-qk-guided-phase="assignment">进 Assignment (先折 24×54 再分核) →</button>';
        else ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-preview-layout="stride24">查看 24×54 折行</button><button type="button" class="btn btn-ghost btn-sm" data-qk-guided-phase="ordering">← 回 Ordering</button>';
      }else if(isLane){
        const b=d.blocks[curLane], pg=physicalGroupOf(curLane), par=curLane%2===0?'偶':'奇';
        const sampleIdx=b.entries.slice(0,5).map(e=>'<b>#'+e.queuePosition+'</b>').join(', ');
        const peerLane=curLane%2===0?(curLane+1)%d.blockCount:(curLane+d.blockCount-1)%d.blockCount;
        const peerPar=peerLane%2===0?'偶':'奇';
        copyHtml='<span class="csa-qk-trace-msg"><b>AI Core Lane '+String(curLane).padStart(2,'0')+'</b> · 核组 '+String(pg).padStart(2,'0')+'：<code>下标 = '+curLane+' + 轮次 × 24</code> → 领取 '+sampleIdx+'…（<b>'+b.validCount+' valid</b>）</span>';
        ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-focus-lane="'+peerLane+'">对比 '+peerPar+' Lane '+String(peerLane).padStart(2,'0')+'</button>';
        if(phase==='assignment')ctaHtml+='<button type="button" class="btn btn-ghost btn-sm" data-qk-preview-layout="stride24">← 查看 24×54 折行对应列</button>';
        else ctaHtml+='<button type="button" class="btn btn-sm csa-qk-trace-cta" data-qk-guided-phase="assignment">去 Assignment 看该核 →</button>';
      }else if(isItem){
        const it=d.byId.get(s.id), info=itemDetails(d,s.id), par=info.block%2===0?'偶':'奇';
        const laneText=phase==='assignment'?' → <b>Lane '+String(info.block).padStart(2,'0')+'</b> ('+par+' · 第 '+info.iteration+' 轮)':'';
        copyHtml='<span class="csa-qk-trace-msg"><b>Work Item '+it.id+'</b> · Request '+it.requestIndex+' / Q'+it.token+'·S'+it.sparseBlock+' → <b>qk_order['+info.queuePosition+']</b>'+laneText+'</span>';
        ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-focus-req="'+it.requestIndex+'">追踪 Request '+it.requestIndex+'</button>'+(phase==='assignment'?'<button type="button" class="btn btn-ghost btn-sm" data-qk-focus-lane="'+info.block+'">追踪 Lane '+String(info.block).padStart(2,'0')+'</button>':'');
      }else{
        if(phase==='generation'){
          copyHtml='<span class="csa-qk-trace-msg"><b>第 1 步 · 原始任务矩阵</b>：点击任意 Request 卡片固定单请求视角；进入 Ordering 时会先横向展开为 40×32 矩阵再压实</span>';
          ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-preview-layout="flatten">预览 40×32 展开</button><button type="button" class="btn btn-sm csa-qk-trace-cta" data-qk-focus-req="'+defReq+'">固定追踪 Request '+defReq+' →</button>';
        }else if(phase==='ordering'){
          copyHtml=state.orderPolicy==='balanced'
            ?'<span class="csa-qk-trace-msg"><b>第 2 步 · 40×32 排序队列 qk_order</b>：优化策略先连续写入全部红块 CMP，再连续写入全部绿块 SB0，队尾追加无效灰块</span>'
            :'<span class="csa-qk-trace-msg"><b>第 2 步 · 40×32 排序队列 qk_order</b>：无效灰块已移至下方队尾，有效彩块向前补齐；多数 Token 为 1 绿 + 1 红呈交替排列</span>';
          ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-focus-req="'+defReq+'">追踪 Request '+defReq+' 落点</button><button type="button" class="btn btn-sm csa-qk-trace-cta" data-qk-guided-phase="assignment">进 Assignment (24×54 折行 → 分核) →</button>';
        }else{
          copyHtml='<span class="csa-qk-trace-msg"><b>第 3 步 · 24 颗 AI Core 领活</b>：由 24×54 步长矩阵每列移入对应 AI Core（<code>下标 = Lane + 轮次 × 24</code>）；底部色条按代价加权</span>';
          ctaHtml='<button type="button" class="btn btn-ghost btn-sm" data-qk-preview-layout="stride24">查看 24×54 折行矩阵</button><button type="button" class="btn btn-sm csa-qk-trace-cta" data-qk-focus-lane="'+defLane+'">追踪慢核 Lane '+String(defLane).padStart(2,'0')+' →</button>';
        }
      }
      const detailToggle=s?'<button type="button" class="btn btn-ghost btn-sm'+(state.inspectorOpen?' is-selected':'')+'" data-qk-toggle-drawer>'+(state.inspectorOpen?'收起详情':'详情')+'</button>':'';
      return '<div class="csa-qk-trace-strip" data-qk-trace-strip role="region" aria-label="阶段追踪引导">'+
        '<div class="segmented-control segmented-control-muted csa-qk-trace-modes" role="group" aria-label="追踪视角">'+
          '<button type="button" class="btn btn-sm'+(isOverview?' is-selected':'')+'" data-qk-trace-mode="all" aria-pressed="'+isOverview+'">全景</button>'+
          '<button type="button" class="btn btn-sm'+(isReq?' is-selected':'')+'" data-qk-trace-mode="request" aria-pressed="'+isReq+'">追踪 Request</button>'+
          (showLaneMode?'<button type="button" class="btn btn-sm'+(isLane?' is-selected':'')+'" data-qk-trace-mode="lane" aria-pressed="'+isLane+'">追踪 AI Core</button>':'')+
        '</div>'+
        copyHtml+
        '<div class="csa-qk-trace-actions">'+ctaHtml+detailToggle+'</div>'+
      '</div>';
    }
    function selectionIds(d,selection=state.selection){
      if(!selection)return new Set();
      if(selection.type==='request')return new Set(d.items.filter(item=>item.requestIndex===selection.requestIndex).map(item=>item.id));
      if(selection.type==='query')return new Set(d.items.filter(item=>item.token===selection.tokenIndex).map(item=>item.id));
      if(selection.type==='block')return new Set(d.blocks[selection.block]?.entries.map(entry=>entry.id)||[]);
      return new Set([selection.id]);
    }
    function itemMarkup(d,positions){const relatedIds=selectionIds(d);return d.items.map(item=>{const p=positions.get(item.id),s=p.w/BASE_CELL,selected=state.selection?.type==='item'&&state.selection.id===item.id;
      const related=!selected&&relatedIds.has(item.id);
      const info=itemDetails(d,item.id);
      const typeCls=item.valid?(item.sparseBlock===WINDOW_SB?' is-sb0':' is-cmp'):'';
      return '<g class="csa-qk-canvas-item '+(item.valid?'is-valid':'is-invalid')+typeCls+(selected?' is-selected':'')+(related?' is-related':'')+(state.selection&&!related&&!selected?' is-dimmed':'')+'" data-qk-item="'+item.id+'" tabindex="'+(selected||(!state.selection&&item.id===0)?'0':'-1')+'" role="button" aria-label="Work Item '+item.id+' · Q'+item.token+' SB'+item.sparseBlock+' ('+(item.sparseBlock===WINDOW_SB?'SB0 滑窗':'CMP 压缩')+') · qk_order['+info.queuePosition+'] · Lane '+info.block+' / it '+info.iteration+' · '+(item.valid?'valid':'invalid')+'" transform="'+fmtTransform(p.x,p.y,s)+'"><rect width="'+BASE_CELL+'" height="'+BASE_CELL+'" rx="1.5" vector-effect="non-scaling-stroke"/></g>';}).join('');}
    function inspector(d){if(!state.selection||!state.inspectorOpen)return '';
      const selection=state.selection,block=selection.type==='block'?getBlockDetails(selection.block,d.items,d.queue,d.blockCount):null;
      let title='',facts=[],formula='',crossLinks='';
      const rd=traceRank();
      if(selection.type==='request'){
        const m=requestMeta(d,selection.requestIndex);
        const vRange=m.validSorted.length?'qk_order['+qPosOf(d,m.validSorted[0].id)+'..'+qPosOf(d,m.validSorted.at(-1).id)+']':'无';
        const iRange=m.invalidSorted.length?'qk_order['+qPosOf(d,m.invalidSorted[0].id)+'..'+qPosOf(d,m.invalidSorted.at(-1).id)+']':'无';
        title='Request '+selection.requestIndex;
        facts=[
          ['Rank / Scenario',state.rank+' / '+scenarioId()],
          ...(m.pos!=null?[['起始 Position','pos = '+m.pos+'（连续 8 Token）']]:[]),
          ['Queries','Q'+m.items[0].token+'–Q'+m.items.at(-1).token],
          ['Work Items',m.items.length+'（'+m.valid.length+' V / '+m.invalid.length+' I）'],
          ['有效块构成',m.sb0+' SB0 滑窗 + '+m.cmp+' CMP 压缩'],
          ['有效区落点',vRange],
          ['无效队尾落点',iRange]
        ];
        formula='Request '+selection.requestIndex+' → 8 Queries × 5 Sparse Blocks = 40 Work Items\n'+m.validSorted.slice(0,12).map(it=>{const qp=qPosOf(d,it.id);return 'Q'+it.token+'·SB'+it.sparseBlock+' → qk_order['+qp+'] → Lane '+String(qp%d.blockCount).padStart(2,'0')+' (it '+Math.floor(qp/d.blockCount)+')';}).join('\n')+(m.validSorted.length>12?'\n… 共 '+m.validSorted.length+' 个有效项':'');
      }else if(selection.type==='query'){
        const items=d.items.filter(item=>item.token===selection.tokenIndex);
        title='Query '+selection.tokenIndex;facts=[['Rank / Scenario',state.rank+' / '+scenarioId()],['Request',selection.requestIndex],['Position',items[0].positionId],['Work Items',items.map(item=>item.id).join(', ')],['valid / invalid',items.filter(item=>item.valid).length+' / '+items.filter(item=>!item.valid).length]];
        formula='Q'+selection.tokenIndex+' → SB0–SB4 → Work Items '+items[0].id+'–'+items.at(-1).id+'\n'+items.map(it=>{const qp=qPosOf(d,it.id);return 'SB'+it.sparseBlock+(it.valid?' (valid)':' (invalid)')+' → qk_order['+qp+'] → Lane '+(qp%d.blockCount)+' / it '+Math.floor(qp/d.blockCount);}).join('\n');
      }else if(block){title='SPMD Lane '+String(block.index).padStart(2,'0');
        const bMeta=d.blocks[block.index], pg=physicalGroupOf(block.index), g=rd?.groups?.find(x=>x.group===pg), wm=g?.workMix;
        const onTrace=hasRuntimeTrace()&&state.workload==='trace';
        facts=[
          ['Rank / 场景','Rank '+state.rank+' · '+(onTrace?'Runtime 实测输入':state.workload)],
          ['入队策略',state.orderPolicy==='balanced'?'优化：先 CMP 后 SB0（bit-exact）':'源码：行主序（易触发奇偶偏斜）'],
          ['Lane 奇偶',block.index%2===0?'偶数 Lane（只踩偶数 qk_order 下标）':'奇数 Lane（只踩奇数 qk_order 下标）'],
          ['分配总量',block.entries.length+' 项（'+block.validCount+' V / '+block.invalidCount+' I）'],
          ['有效块拆解',bMeta.sb0Count+' SB0 滑窗 (~8 µs) + '+bMeta.cmpCount+' CMP 压缩 (~39 µs)'],
          ...(onTrace&&g?[['物理核组映射','Group '+String(pg).padStart(2,'0')+'（3 线程步长落核）'],['关联实测时序','AIC '+g.aic.durUs.toFixed(1)+' µs · 队尾写零 +'+g.aiv0TailUs.toFixed(1)+' µs ('+(wm?wm.invalidZeroMiB.toFixed(2)+' MiB':'')+')']]:[]),
          ['Queue 步长','qk_order['+(block.entries[0]?.queuePosition??0)+'] → ['+(block.entries.at(-1)?.queuePosition??0)+'] (step 24)']
        ];
        formula='queuePosition = laneId + iteration × '+d.blockCount+'\n'+block.entries.map(e=>'it '+e.iteration+' → qk_order['+e.queuePosition+'] = '+e.id).join('\n');
        crossLinks='<div class="csa-qk-cross-actions">'+
          '<button type="button" class="btn btn-sm csa-rt-cta-primary" data-qk-open-pipeline="'+pg+'">在 Pipeline 查看 Group '+String(pg).padStart(2,'0')+' 实测时间轴 →</button>'+
          '<button type="button" class="btn btn-ghost btn-sm" data-qk-open-graph="sparse/attention/qk-pv/selected" data-qk-act-ctx="act2-qk-pv-parity">对照 SB0 / CMP 访存分支代价（Graph）→</button>'+
        '</div>';
      }else{const item=d.byId.get(selection.id);if(!item)return '';const info=itemDetails(d,item.id), pg=physicalGroupOf(info.block);title='Work Item '+item.id;
        facts=[['Rank / Scenario',item.rank+' / '+item.scenarioId],['Request / Query',item.requestIndex+' / Q'+item.token],['Position',item.positionId],['Sparse Block','SB'+item.sparseBlock],['KV 路径',item.sparseBlock===0?'SB0 · 滑窗连续 DMA (~8 µs)':'CMP · 128× 标量 Gather (~39 µs)'],['有效索引',item.validKvCount],['Validity',item.valid?'valid':'invalid（写零 128 KB）'],['qk_order 位置',info.queuePosition],['Lane / 物理核组','Lane '+info.block+' → 核组 '+String(pg).padStart(2,'0')+' (it '+info.iteration+')']];
        formula='workItemId = '+item.token+' × 5 + '+item.sparseBlock+' = '+item.id+'\nqueuePosition = '+info.queuePosition+'\nlane = '+info.queuePosition+' % '+d.blockCount+' = '+info.block+'\niteration = floor('+info.queuePosition+' / '+d.blockCount+') = '+info.iteration;
        crossLinks='<div class="csa-qk-cross-actions">'+
          '<button type="button" class="btn btn-sm csa-rt-cta-primary" data-qk-open-pipeline="'+pg+'">在 Pipeline 查看 Group '+String(pg).padStart(2,'0')+' 实测时间轴 →</button>'+
          '<button type="button" class="btn btn-ghost btn-sm" data-qk-open-graph="'+(item.valid?(item.sparseBlock===0?'sparse/attention/qk-pv/window':'sparse/attention/qk-pv/selected'):'sparse/attention/qk-pv/invalid-block')+'">在 Graph 查看对应执行路径 →</button>'+
        '</div>';
      }
      const path=selection.type==='item'?(d.byId.get(selection.id).valid?'KV Gather → QK Matmul → Scale / Bias / Softmax → PV Matmul → Save m/l/o':'invalid 分支 → 写入中性 m/l/o（每块写零 128 KB）；不进行 QK / PV 计算'):'';
      return '<aside class="csa-qk-drawer" aria-label="当前计算对象详情"><div class="csa-qk-drawer-head"><h2>'+esc(title)+'</h2><button type="button" data-qk-close aria-label="关闭详情">×</button></div><dl>'+facts.map(([k,v])=>'<dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd>').join('')+'</dl>'+crossLinks+
        '<details><summary>计算规则与来源</summary><pre>'+esc(formula)+'</pre>'+(selection.type==='item'?'<p>'+esc(d.byId.get(selection.id).validityReason)+'</p><p>'+esc(path)+'</p>':'')+'<p>模型配置 TP=2、H=64；Workload 输入 DECODE_BATCH=64、DECODE_SEQ=8；算子常量 WIN=128、IDX_TOPK=512、ATTN_K_TILE=128、QK_M_TILE=32、NUM_QK_CORES=24。256 Query × 5 SB = 1,280 Work Items。</p></details>'+
        '<details><summary>qk_plan / qk_pv · Source Evidence</summary>'+[[134,196,'有效性与 qk_order'],[208,224,'Dispatch 与 Lane 领取'],[224,314,'valid / invalid 路径']].map(([a,z,label])=>'<p><button type="button" data-qk-source="'+SUB_FILE+':'+a+':'+z+'">'+label+' · '+SUB_FILE+' : '+a+'–'+z+' ↗</button></p><pre>'+esc(excerpt(SUB_FILE,a,z))+'</pre>').join('')+'</details>'+
        '<details><summary>显式任务依赖</summary><p>qk_plan_tid → qk_pv ← cache_ready_dep。这里是源码中的静态依赖，执行时序请在 Pipeline 视图查看。</p><button type="button" data-qk-source="decode_csa.py:384:393">查看调用处 ↗</button></details>'+
        (selection.type==='item'?'<details><summary>Advanced Simulation · 手动覆盖</summary><p>仅用于观察规则变化，不代表 Indexer 真实输出。SB0 固定有效。</p><button type="button" data-qk-advanced>'+(state.advanced?'关闭手动覆盖':'启用手动覆盖')+'</button>'+(state.advanced&&d.byId.get(selection.id).sparseBlock!==0?'<button type="button" data-qk-toggle="'+selection.id+'">切换当前 Work Item 有效性</button><button type="button" data-qk-reset>清除覆盖</button>':'')+'</details>':'')+'</aside>';
    }
    function shell(d){const count=d.items.length,positions=viewportPositions(d,state.layout);
      const rd=traceRank(), wp=rd?.chain?.workProfile;
      const showType=state.colorBy==='kv-type';
      const colorSwitch='<div class="segmented-control segmented-control-muted csa-qp-mode-switch" role="group" aria-label="方块着色维度">'+
        '<button type="button" class="btn btn-sm'+(!showType?' is-selected':'')+'" data-qk-color="validity" aria-pressed="'+(!showType)+'">有效 / 无效</button>'+
        '<button type="button" class="btn btn-sm'+(showType?' is-selected':'')+'" data-qk-color="kv-type" aria-pressed="'+showType+'">块类型 SB0 / CMP</button>'+
      '</div>';
      const orderSwitch='<div class="segmented-control segmented-control-muted csa-qp-mode-switch" role="group" aria-label="qk_order 入队策略">'+
        '<button type="button" class="btn btn-sm'+(state.orderPolicy==='source'?' is-selected':'')+'" data-qk-order="source" aria-pressed="'+(state.orderPolicy==='source')+'">源码行主序</button>'+
        '<button type="button" class="btn btn-sm'+(state.orderPolicy==='balanced'?' is-selected':'')+'" data-qk-order="balanced" aria-pressed="'+(state.orderPolicy==='balanced')+'">优化: 先CMP后SB0</button>'+
      '</div>';
      const requests='<label>Request <select data-qk-request>'+Array.from({length:32},(_,i)=>{const first=d.items[i*TOKENS_PER_REQUEST*SPARSE_BLOCKS];const pStr=first?.positionId!=null?' · pos='+first.positionId:'';return '<option value="'+i+'"'+(i===state.requestIndex?' selected':'')+'>R'+i+pStr+'</option>';}).join('')+'</select></label>';
      const selLaneVal=state.selection?.type==='block'?String(state.selection.block):'';
      const lanesSelect='<label data-qk-lane-select-wrap'+(state.phase==='assignment'||state.layout==='stride24'||state.selection?.type==='block'?'':' hidden')+'>AI Core <select data-qk-lane-select><option value="">全部 (24)</option>'+d.blocks.map(b=>'<option value="'+b.index+'"'+(selLaneVal===String(b.index)?' selected':'')+'>L'+String(b.index).padStart(2,'0')+' ('+(b.index%2===0?'偶':'奇')+' · '+b.sb0Count+'S+'+b.cmpCount+'C)</option>').join('')+'</select></label>';
      const traceOption=hasRuntimeTrace()?'<option value="trace"'+(state.workload==='trace'?' selected':'')+'>Runtime Trace 输入 · Rank '+state.rank+(wp?' ('+wp.validItems+'V / '+wp.invalidItems+'I)':'')+'</option>':'';
      const scenarioBar='<div class="csa-qk-scenario">'+colorSwitch+orderSwitch+
        '<label>场景 <select data-qk-workload>'+traceOption+'<option value="mixed"'+(state.workload==='mixed'?' selected':'')+'>Mixed Context · Synthetic</option><option value="uniform"'+(state.workload==='uniform'?' selected':'')+'>Uniform Context · Synthetic</option></select></label>'+
        (state.workload==='uniform'?'<label>Position <select data-qk-position>'+POSITIONS.map(p=>'<option value="'+p+'"'+(p===state.position?' selected':'')+'>'+p+'</option>').join('')+'</select></label>':'')+
        '<label>Rank <select data-qk-rank><option value="0"'+(state.rank===0?' selected':'')+'>0</option><option value="1"'+(state.rank===1?' selected':'')+'>1</option></select></label>'+
        requests+lanesSelect+
        '<span class="csa-qk-scenario-count">'+(count/SPARSE_BLOCKS)+' Queries · '+d.validCount+' V / '+(count-d.validCount)+' I · '+d.blockCount+' Lanes'+(state.workload==='trace'?'（同步自 Runtime）':'')+'</span></div>';
      const legendHtml=showType
        ? '<span class="csa-qk-legend"><i class="is-sb0"></i>SB0 滑窗 <i class="is-cmp"></i>CMP 压缩 <i class="is-invalid"></i>invalid</span>'
        : '<span class="csa-qk-legend"><i class="is-valid"></i>valid <i class="is-invalid"></i>invalid</span>';
      const speedHtml='<div class="segmented-control segmented-control-muted csa-qk-speed" role="group" aria-label="动画播放速度">'+SPEEDS.map(sp=>'<button type="button" class="btn btn-sm'+(state.speed===sp?' is-selected':'')+'" data-qk-speed="'+sp+'" aria-pressed="'+(state.speed===sp)+'">'+sp+'×</button>').join('')+'</div>';
      return '<div class="csa-qk-canvas-shell'+(showType?' is-color-type':'')+'">'+scenarioBar+buildTraceStripHtml(d)+
        '<div class="csa-qk-canvas-head"><div><h2 data-qk-title>'+PHASE_LABELS[state.phase]+'</h2><p data-qk-subtitle>'+subtitle(d)+'</p></div><div class="csa-qk-canvas-controls"><button type="button" class="btn btn-ghost btn-sm" data-qk-play>播放</button><button type="button" class="btn btn-ghost btn-sm" data-qk-replay>重播</button><div class="segmented-control segmented-control-muted">'+PHASES.map(p=>'<button type="button" class="btn btn-sm'+(state.phase===p?' is-selected':'')+'" data-qk-phase="'+p+'" aria-pressed="'+(state.phase===p)+'">'+({generation:'Generation',ordering:'Ordering',assignment:'Assignment'}[p])+'</button>').join('')+'</div>'+speedHtml+'<label><input type="checkbox" data-qk-loop'+(state.loop?' checked':'')+'> 循环</label></div></div>'+
        '<div class="csa-qk-canvas-viewport" tabindex="0" role="group" aria-label="QK/PV Work Assignment 画布"><svg viewBox="0 0 1400 700" preserveAspectRatio="xMidYMid meet"><g data-qk-scene><g data-qk-backdrop-from></g><g data-qk-backdrop-to>'+backdrop(d,state.layout)+'</g><g data-qk-flex-frame></g><g data-qk-trails></g><g data-qk-items>'+itemMarkup(d,positions)+'</g><g data-qk-overlay>'+overlay(d,positions,state.layout)+'</g></g></svg><div class="csa-qk-tooltip" data-qk-tooltip hidden></div></div>'+
        '<footer class="csa-qk-canvas-footer"><span data-qk-summary>'+summary(d)+'</span>'+legendHtml+'<span>空间分配视图 · 执行时序请切至 Pipeline</span><div class="csa-qk-zoom"><button type="button" class="btn btn-ghost btn-sm" data-qk-zoom="0.8" aria-label="缩小">−</button><output data-qk-scale>'+Math.round(state.scale*100)+'%</output><button type="button" class="btn btn-ghost btn-sm" data-qk-zoom="1.25" aria-label="放大">＋</button><button type="button" class="btn btn-ghost btn-sm" data-qk-fit>适应</button></div></footer>'+inspector(d)+'</div>';
    }
    function subtitle(d){if(state.phase==='generation')return (d.items.length/SPARSE_BLOCKS)+' Queries × 5 Sparse Blocks → '+d.items.length+' Work Items';if(state.phase==='ordering')return d.validCount+' valid 优先 · '+(d.items.length-d.validCount)+' invalid 追加（40×32 队列）';return d.items.length+' 个 Work Items → '+d.blockCount+' 条逻辑 Lane · 固定步长领取';}
    function summary(d){
      const s=state.selection;
      if(s?.type==='request')return 'Request '+s.requestIndex+' · 8 Queries / 40 Work Items · 三阶段持续追踪';
      if(s?.type==='query')return 'Q'+s.tokenIndex+' · Request '+s.requestIndex+' · 5 Sparse Blocks / Work Items';
      if(s?.type==='item'){const item=d.byId.get(s.id),info=itemDetails(d,item.id);return 'Work Item '+item.id+' · Q'+item.token+' / SB'+item.sparseBlock+' → qk_order['+info.queuePosition+'] → Lane '+info.block+' / it '+info.iteration;}
      if(s?.type==='block'){const b=d.blocks[s.block];return 'Lane '+b.index+' · qk_order['+b.index+'] → ['+(b.index+d.blockCount)+'] → … · '+b.entries.length+' 项';}
      return d.validCount+' valid / '+(d.items.length-d.validCount)+' invalid · 点击 Request 或 Lane 追踪路径';
    }
    // 5 种布局共享同一静态 1400×700 坐标系，切换布局全程零 viewBox 抖动（对齐 Paged Attention Morph）
    function sceneBounds(){
      if(state.mode==='source')return {x:0,y:0,w:1400,h:700};
      return {x:26,y:40,w:1300,h:635};
    }
    function fitBox(){const viewport=host.querySelector('.csa-qk-canvas-viewport'),ratio=viewport?.clientWidth&&viewport?.clientHeight?viewport.clientWidth/viewport.clientHeight:1400/700,b=sceneBounds();let h=Math.max(b.h*1.02,b.w*1.02/ratio);return {x:b.x+b.w/2,y:b.y+b.h/2,w:h*ratio,h};}
    function camera(animate=false){const svg=host.querySelector('.csa-qk-canvas-viewport svg');if(!svg)return;const fit=fitBox(),w=fit.w/state.scale,h=fit.h/state.scale,x=fit.x-state.panX-w/2,y=fit.y-state.panY-h/2;const target=[x,y,w,h],from=(svg.getAttribute('viewBox')||'').split(/\s+/).map(Number);if(cameraFrame)cancelAnimationFrame(cameraFrame);if(visualUI&&animate&&!reduced&&from.length===4){const start=performance.now();const tick=now=>{const t=Math.min(1,(now-start)/420),e=smoothstep(t);svg.setAttribute('viewBox',target.map((v,i)=>from[i]+(v-from[i])*e).join(' '));if(t<1)cameraFrame=requestAnimationFrame(tick);else cameraFrame=0;};cameraFrame=requestAnimationFrame(tick);}else svg.setAttribute('viewBox',target.join(' '));const out=host.querySelector('[data-qk-scale]');if(out)out.textContent=Math.round(state.scale*100)+'%';}
    function focusCamera(d,animate=false){const s=state.selection;if(!s)return;const layout=state.layout;
      if((layout==='ordering'||layout==='flatten'||layout==='stride24')&&(s.type==='request'||s.type==='block')){
        if(!state.manualCamera){state.scale=1;state.panX=0;state.panY=0;}
        camera(animate);
        return;
      }
      const positions=viewportPositions(d,layout),ids=selectionIds(d),rects=[...ids].map(id=>positions.get(id)).filter(Boolean);if(!rects.length)return;
      const minX=Math.min(...rects.map(p=>p.x)),maxX=Math.max(...rects.map(p=>p.x+p.w)),minY=Math.min(...rects.map(p=>p.y)),maxY=Math.max(...rects.map(p=>p.y+p.h));
      const fit=fitBox(),bboxW=maxX-minX,bboxH=maxY-minY;
      const maxZoom=s.type==='item'?2.6:s.type==='block'?2.1:s.type==='query'?2.1:(layout==='generation'?2.2:1.45);
      if(!state.manualCamera)state.scale=Math.max(1,Math.min(maxZoom,fit.w/(bboxW+140),fit.h/(bboxH+100)));
      const viewW=fit.w/state.scale,targetX=(minX+maxX)/2,targetY=(minY+maxY)/2;
      state.panX=fit.x-(targetX+(state.inspectorOpen ? .18*viewW : 0));state.panY=fit.y-targetY;camera(animate);
    }
    function syncItemCache(d){
      itemNodes=Array(d.items.length);
      host.querySelectorAll('[data-qk-item]').forEach(node=>{const id=Number(node.dataset.qkItem);itemNodes[id]=node;});
      const pos=viewportPositions(d,state.layout);
      livePos=d.items.map(it=>{const p=pos.get(it.id);return p?{x:p.x,y:p.y,s:p.w/BASE_CELL}:{x:0,y:0,s:1};});
    }
    function refreshSelection(d,locate=true,opts={}){
      const s=state.selection,ids=selectionIds(d),has=!!s, layout=state.layout;
      const positions=viewportPositions(d,layout);
      if(!opts.skipBackdrop&&(layout==='ordering'||layout==='flatten'||layout==='stride24')){
        const toLayer=host.querySelector('[data-qk-backdrop-to]');
        if(toLayer)toLayer.innerHTML=backdrop(d,layout);
      }
      const overlayNode=host.querySelector('[data-qk-overlay]');if(overlayNode)overlayNode.innerHTML=overlay(d,positions,layout);
      const stripNode=host.querySelector('[data-qk-trace-strip]');if(stripNode)stripNode.outerHTML=buildTraceStripHtml(d);
      for(let id=0;id<itemNodes.length;id++){
        const node=itemNodes[id];
        if(!node)continue;
        const selected=s?.type==='item'&&id===s.id,related=!selected&&ids.has(id);
        node.classList.toggle('is-selected',selected);node.classList.toggle('is-related',related);node.classList.toggle('is-dimmed',has&&!selected&&!related);
        node.setAttribute('tabindex',selected||(!has&&id===0)?'0':'-1');node.setAttribute('aria-pressed',String(selected));
      }
      host.querySelectorAll('[data-qk-lane]').forEach(node=>{const lane=Number(node.dataset.qkLane),selected=s?.type==='block'&&s.block===lane,contains=s?.type==='item'&&itemDetails(d,s.id).block===lane;node.classList.toggle('is-selected',selected);node.classList.toggle('is-containing',contains);node.classList.toggle('is-dimmed',has&&s?.type==='block'&&!selected);});
      const request=s?.type==='request'?s.requestIndex:s?.type==='query'?s.requestIndex:s?.type==='item'?d.byId.get(s.id)?.requestIndex:null;
      host.querySelectorAll('[data-qk-request-group]').forEach(node=>{const selected=request===Number(node.dataset.qkRequestGroup);node.classList.toggle('is-focused',selected);node.classList.toggle('is-dimmed',request!=null&&!selected);});
      const relatedQueries=new Set([...ids].map(id=>Math.floor(id/SPARSE_BLOCKS)));
      host.querySelectorAll('[data-qk-query]').forEach(node=>{const token=Number(node.dataset.qkQuery),active=s?.type==='query'&&token===s.tokenIndex||s?.type==='item'&&d.byId.get(s.id)?.token===token;node.classList.toggle('is-focused',active);node.classList.toggle('is-related',!active&&relatedQueries.has(token));node.setAttribute('tabindex',active||token===state.requestIndex*TOKENS_PER_REQUEST?'0':'-1');});
      const summaryNode=host.querySelector('[data-qk-summary]');if(summaryNode)summaryNode.textContent=summary(d);
      const requestSelect=host.querySelector('[data-qk-request]');if(requestSelect)requestSelect.value=String(state.requestIndex);
      const laneSelect=host.querySelector('[data-qk-lane-select]');if(laneSelect)laneSelect.value=s?.type==='block'?String(s.block):'';
      const laneWrap=host.querySelector('[data-qk-lane-select-wrap]');if(laneWrap)laneWrap.hidden=!(state.phase==='assignment'||layout==='stride24'||s?.type==='block');
      host.querySelector('.csa-qk-drawer')?.remove();if(state.inspectorOpen&&s)host.querySelector('.csa-qk-canvas-shell')?.insertAdjacentHTML('beforeend',inspector(d));
      if(!opts.skipCamera){if(locate)focusCamera(d,!!opts.animateCamera);else camera(!!opts.animateCamera);}
    }
    function render(){const d=derive();host.innerHTML=shell(d);syncItemCache(d);refreshSelection(d,!!state.selection,{skipBackdrop:true});}
    // 单一 rAF 时钟 Morph 引擎（对齐 Paged Attention）：
    // - 0%–22% 旧壳平滑淡出，0%–100% 1,280 格按语义成组插值 SVG transform（零 width/height 重排、零 viewBox 抖动），78%–100% 新壳与编号浮现接住
    // - flex-narrow / flex-widen：模拟真实 CSS Flexbox 容器宽度收缩(40→24)/展开(24→40)，最右侧格子顺势换行向下流动
    function runMorph(d,targetLayout,opts={}){
      cancelMorph();
      const fromLayout=opts.fromLayout??state.layout;
      const mode=opts.mode||'generic'; // 'gen-flatten' | 'flatten-order' | 'flex-narrow' | 'flex-widen' | 'stride24-assign' | 'order-policy' | 'generic'
      const duration=msFor(opts.duration??850);
      const onDone=opts.onDone||null;
      state.layout=targetLayout;
      const targetPosMap=viewportPositions(d,targetLayout);
      const count=d.items.length;
      if(!itemNodes.length||itemNodes.length!==count)syncItemCache(d);

      const isFlex=mode==='flex-narrow'||mode==='flex-widen';
      const L40=matrixLayout(d,'ordering'), L24=matrixLayout(d,'stride24');
      const flexFrom=mode==='flex-narrow'?L40:L24;
      const flexTo=mode==='flex-narrow'?L24:L40;

      // Capture exact current (x, y, s) so even mid-flight interruption never snaps
      const sx=new Float32Array(count), sy=new Float32Array(count), ss=new Float32Array(count);
      const tx=new Float32Array(count), ty=new Float32Array(count), ts=new Float32Array(count);
      const delay=new Float32Array(count), span=new Float32Array(count), arcX=new Float32Array(count);
      const qIdx=new Int32Array(count);

      for(let i=0;i<count;i++){
        const cur=livePos[i]||{x:0,y:0,s:1};
        const dst=targetPosMap.get(i)||{x:cur.x,y:cur.y,w:BASE_CELL};
        sx[i]=cur.x; sy[i]=cur.y; ss[i]=cur.s;
        tx[i]=dst.x; ty[i]=dst.y; ts[i]=dst.w/BASE_CELL;
        const qp=qPosOf(d,i);
        qIdx[i]=qp;
        const it=d.items[i];
        if(mode==='gen-flatten'){
          // 每个 Request 40 格完全同步作为一整行展开/收起
          delay[i]=(it.requestIndex/32)*0.18; span[i]=0.82; arcX[i]=0;
        }else if(mode==='flatten-order'){
          // 两股清晰分流：无效灰块先行下沉（轻微右弧避让），有效彩块紧随向前补齐
          if(it.valid){ delay[i]=0.18; span[i]=0.82; arcX[i]=0; }
          else{ delay[i]=0.0; span[i]=0.74; arcX[i]=22; }
        }else if(mode==='stride24-assign'){
          // 24 列逐列整体飞入对应 AI Core（同列 54 格完全同步出发）
          const col=qp%d.blockCount;
          delay[i]=(col/d.blockCount)*0.28; span[i]=0.72; arcX[i]=0;
        }else if(mode==='order-policy'){
          // 切换源码/优化入队策略：CMP 红块与 SB0 绿块分成两股平滑换位
          if(!it.valid){ delay[i]=0; span[i]=0.6; arcX[i]=0; }
          else if(it.sparseBlock===WINDOW_SB){ delay[i]=0.14; span[i]=0.86; arcX[i]=14; }
          else{ delay[i]=0; span[i]=0.84; arcX[i]=-14; }
        }else{
          delay[i]=0; span[i]=1; arcX[i]=0;
        }
      }

      const fromLayer=host.querySelector('[data-qk-backdrop-from]');
      const toLayer=host.querySelector('[data-qk-backdrop-to]');
      const ovLayer=host.querySelector('[data-qk-overlay]');
      const flexLayer=host.querySelector('[data-qk-flex-frame]');

      const finish=()=>{
        for(let i=0;i<count;i++){
          livePos[i]={x:tx[i],y:ty[i],s:ts[i]};
          const node=itemNodes[i];
          if(node)node.setAttribute('transform',fmtTransform(tx[i],ty[i],ts[i]));
        }
        if(fromLayer){fromLayer.innerHTML='';fromLayer.style.opacity='0';}
        if(flexLayer){flexLayer.innerHTML='';flexLayer.style.opacity='0';}
        if(toLayer){toLayer.style.opacity='1';}
        if(ovLayer){ovLayer.style.opacity='1';}
        if(opts.clearSpotlightOnDone)applyTransitSpotlight(d,null);
        if(state.selection&&(targetLayout==='generation'||targetLayout==='assignment')){
          focusCamera(d,true);
        }
        if(visualUI)host.dispatchEvent(new CustomEvent('csa-view-change',{bubbles:true}));
        onDone?.();
      };

      // If reduced motion or non-DOM test environment, settle synchronously
      if(!visualUI||reduced||!toLayer){
        if(toLayer)toLayer.innerHTML=backdrop(d,targetLayout);
        refreshSelection(d,!!state.selection,{skipBackdrop:true});
        applyTransitSpotlight(d,null);
        finish();
        return;
      }

      // 1. Preserve old shell in fromLayer, place new shell in toLayer (initially transparent)
      if(fromLayer){
        fromLayer.innerHTML=toLayer.innerHTML;
        fromLayer.style.opacity='1';
      }
      toLayer.innerHTML=backdrop(d,targetLayout);
      toLayer.style.opacity='0';
      if(ovLayer)ovLayer.style.opacity='0';
      if(flexLayer){flexLayer.innerHTML='';flexLayer.style.opacity='0';}

      // Reset zoomed view smoothly to full 1400×700 shared space during transit so full trajectory is visible
      if(state.scale!==1||state.panX!==0||state.panY!==0){
        state.scale=1; state.panX=0; state.panY=0; state.manualCamera=false;
        camera(true);
      }

      refreshSelection(d,false,{skipBackdrop:true,skipCamera:true});
      applyTransitSpotlight(d,opts.spotlight||null);
      if(!isFlex)renderMotionTrails(d,fromLayout,targetLayout);
      else{const tl=host.querySelector('[data-qk-trails]');if(tl)tl.innerHTML='';}

      const token=++morphSerial;
      const startTime=performance.now();
      const tick=now=>{
        if(token!==morphSerial)return;
        const p=clamp01((now-startTime)/duration);
        if(fromLayer)fromLayer.style.opacity=String(clamp01(1-p/0.18));
        const inAlpha=clamp01((p-0.82)/0.18);
        toLayer.style.opacity=String(inAlpha);
        if(ovLayer)ovLayer.style.opacity=String(inAlpha);

        if(isFlex){
          const e=smoothstep(p);
          const cFloat=flexFrom.cols+(flexTo.cols-flexFrom.cols)*e;
          const cell=flexFrom.cell+(flexTo.cell-flexFrom.cell)*e;
          const step=flexFrom.step+(flexTo.step-flexFrom.step)*e;
          const topY=flexFrom.topY+(flexTo.topY-flexFrom.topY)*e;
          const s=cell/BASE_CELL;
          const gridW=cFloat*step-(step-cell);
          const leftX=700-gridW/2;
          const rowsNow=Math.ceil(count/cFloat);
          const cMinus1=cFloat-1;
          let maxY=topY;

          for(let i=0;i<count;i++){
            const node=itemNodes[i];
            if(!node)continue;
            const q=qIdx[i];
            const r=Math.floor(q/cFloat);
            const colRaw=q-r*cFloat;
            let x, y;
            if(colRaw<=cMinus1){
              x=leftX+colRaw*step;
              y=topY+r*step;
            }else{
              const w=smoothstep(colRaw-cMinus1);
              x=leftX+cMinus1*(1-w)*step;
              y=topY+(r+w)*step;
            }
            if(y>maxY)maxY=y;
            livePos[i].x=x; livePos[i].y=y; livePos[i].s=s;
            node.setAttribute('transform',fmtTransform(x,y,s));
          }

          // Live Flexbox container frame showing walls squeezing inward (40→24) or expanding outward (24→40)
          if(flexLayer){
            const gridH=(maxY-topY)+cell;
            const frameAlpha=Math.min(clamp01(p/0.1),clamp01((1-p)/0.12));
            flexLayer.style.opacity=String(frameAlpha);
            const fx=(leftX-7).toFixed(1), fy=(topY-6).toFixed(1), fw=(gridW+14).toFixed(1), fh=(gridH+12).toFixed(1);
            const arrowL=mode==='flex-narrow'?'▶':'◀', arrowR=mode==='flex-narrow'?'◀':'▶';
            const colsDisp=Math.round(cFloat);
            flexLayer.innerHTML=
              '<rect class="csa-qk-flex-box" x="'+fx+'" y="'+fy+'" width="'+fw+'" height="'+fh+'" rx="6"/>'+
              '<line class="csa-qk-flex-wall" x1="'+(leftX-7).toFixed(1)+'" y1="'+fy+'" x2="'+(leftX-7).toFixed(1)+'" y2="'+(topY+gridH+6).toFixed(1)+'"/>'+
              '<line class="csa-qk-flex-wall" x1="'+(leftX+gridW+7).toFixed(1)+'" y1="'+fy+'" x2="'+(leftX+gridW+7).toFixed(1)+'" y2="'+(topY+gridH+6).toFixed(1)+'"/>'+
              '<text class="csa-qk-flex-badge" x="700" y="'+(topY-15).toFixed(1)+'" text-anchor="middle">'+arrowL+' 容器宽度：'+colsDisp+' 列 × '+rowsNow+' 行 '+arrowR+'</text>';
          }
        }else{
          for(let i=0;i<count;i++){
            const node=itemNodes[i];
            if(!node)continue;
            const lp=smoothstep(clamp01((p-delay[i])/span[i]));
            const bx=arcX[i]?arcX[i]*Math.sin(lp*Math.PI):0;
            const x=sx[i]+(tx[i]-sx[i])*lp+bx;
            const y=sy[i]+(ty[i]-sy[i])*lp;
            const s=ss[i]+(ts[i]-ss[i])*lp;
            livePos[i].x=x; livePos[i].y=y; livePos[i].s=s;
            node.setAttribute('transform',fmtTransform(x,y,s));
          }
        }

        if(p<1){
          morphFrame=requestAnimationFrame(tick);
        }else{
          morphFrame=0;
          finish();
        }
      };
      morphFrame=requestAnimationFrame(tick);
    }
    function morphOrder(nextPolicy){
      if(nextPolicy===state.orderPolicy)return;
      clearBridge();
      state.orderPolicy=nextPolicy;
      if(state.layout==='flatten')state.layout=state.phase;
      const d=derive();
      host.querySelectorAll('[data-qk-order]').forEach(b=>{const active=b.dataset.qkOrder===nextPolicy;b.classList.toggle('is-selected',active);b.setAttribute('aria-pressed',String(active));});
      const laneSelect=host.querySelector('[data-qk-lane-select]');
      if(laneSelect){const curVal=laneSelect.value;laneSelect.innerHTML='<option value="">全部 (24)</option>'+d.blocks.map(b=>'<option value="'+b.index+'">L'+String(b.index).padStart(2,'0')+' ('+(b.index%2===0?'偶':'奇')+' · '+b.sb0Count+'S+'+b.cmpCount+'C)</option>').join('');laneSelect.value=curVal;}
      runMorph(d,state.layout,{fromLayout:state.layout,mode:'order-policy',duration:760});
    }
    function applyStoryAct(actId){
      clearBridge();
      state.activeAct=actId||null;
      if(actId&&hasRuntimeTrace()){
        state.workload='trace';
        state.overrides.clear();
        if(actId==='act2-qk-pv-parity'){state.phase='assignment';state.layout='assignment';state.colorBy='kv-type';state.orderPolicy='source';}
        else if(actId==='act3-invalid-zero-flow'){state.phase='ordering';state.layout='ordering';state.colorBy='validity';}
      }
    }
    function setStoryAct(actId){
      applyStoryAct(actId);
      render();
    }
    function updatePhase(phase,opts={}){
      if(!PHASES.includes(phase))return;
      clearBridge();
      const oldLayout=state.layout;
      state.phase=phase;
      const onComplete=opts.onComplete||null;
      const d=derive();
      const title=host.querySelector('[data-qk-title]');if(title)title.textContent=PHASE_LABELS[phase];
      const sub=host.querySelector('[data-qk-subtitle]');if(sub)sub.textContent=subtitle(d);
      host.querySelectorAll('[data-qk-phase]').forEach(b=>{const active=b.dataset.qkPhase===phase;b.classList.toggle('is-selected',active);b.setAttribute('aria-pressed',String(active));});

      const canAnimate=visualUI&&!reduced&&state.mode==='source';
      // 1. Generation <-> Ordering：正向全程聚焦 Request 0（其余 Request 降透明度，完成后恢复）
      if(canAnimate&&oldLayout==='generation'&&phase==='ordering'){
        const reqSpot={type:'request',index:0};
        runMorph(d,'flatten',{fromLayout:'generation',mode:'gen-flatten',duration:820,spotlight:reqSpot,onDone:()=>{
          bridgeTimer=global.setTimeout(()=>{bridgeTimer=null;if(state.phase==='ordering')runMorph(derive(),'ordering',{fromLayout:'flatten',mode:'flatten-order',duration:960,spotlight:reqSpot,clearSpotlightOnDone:true,onDone:onComplete});},msFor(420));
        }});
        return;
      }
      if(canAnimate&&oldLayout==='ordering'&&phase==='generation'){
        runMorph(d,'flatten',{fromLayout:'ordering',mode:'flatten-order',duration:840,onDone:()=>{
          bridgeTimer=global.setTimeout(()=>{bridgeTimer=null;if(state.phase==='generation')runMorph(derive(),'generation',{fromLayout:'flatten',mode:'gen-flatten',duration:800,onDone:onComplete});},msFor(360));
        }});
        return;
      }
      // 2. Ordering <-> Assignment：先 40→24 Flexbox 折行，变成 24 列后聚焦第 0 列（其余 23 列降透明度），移入 AI Core 完成后恢复
      if(canAnimate&&oldLayout==='ordering'&&phase==='assignment'){
        const colSpot={type:'lane',index:0};
        runMorph(d,'stride24',{fromLayout:'ordering',mode:'flex-narrow',duration:1450,onDone:()=>{
          applyTransitSpotlight(derive(),colSpot);
          bridgeTimer=global.setTimeout(()=>{bridgeTimer=null;if(state.phase==='assignment')runMorph(derive(),'assignment',{fromLayout:'stride24',mode:'stride24-assign',duration:960,spotlight:colSpot,clearSpotlightOnDone:true,onDone:onComplete});},msFor(640));
        }});
        return;
      }
      if(canAnimate&&oldLayout==='assignment'&&phase==='ordering'){
        runMorph(d,'stride24',{fromLayout:'assignment',mode:'stride24-assign',duration:860,onDone:()=>{
          bridgeTimer=global.setTimeout(()=>{bridgeTimer=null;if(state.phase==='ordering')runMorph(derive(),'ordering',{fromLayout:'stride24',mode:'flex-widen',duration:1350,onDone:onComplete});},msFor(420));
        }});
        return;
      }
      const mode=oldLayout==='flatten'&&phase==='ordering'?'flatten-order':oldLayout==='stride24'&&phase==='assignment'?'stride24-assign':oldLayout==='ordering'&&phase==='stride24'?'flex-narrow':oldLayout==='stride24'&&phase==='ordering'?'flex-widen':oldLayout==='generation'&&phase==='flatten'?'gen-flatten':'generic';
      const singleSpot=oldLayout==='flatten'&&phase==='ordering'?{type:'request',index:0}:oldLayout==='stride24'&&phase==='assignment'?{type:'lane',index:0}:null;
      runMorph(d,phase,{fromLayout:oldLayout,mode,duration:mode.startsWith('flex-')?1400:820,spotlight:singleSpot,clearSpotlightOnDone:!!singleSpot,onDone:onComplete});
    }
    function select(selection,opts={}){stop();const openDrawer=opts.openDrawer!==undefined?opts.openDrawer:!!selection;state.selection=selection;state.inspectorOpen=!!selection&&openDrawer;state.manualCamera=false;if(!selection){state.scale=1;state.panX=state.panY=0;}const d=derive();
      if(selection?.type==='item'){const item=d.byId.get(selection.id);if(!item)return;const info=itemDetails(d,item.id);state.requestIndex=item.requestIndex;
        onSelection({type:'item',id:item.id,rank:state.rank,scenarioId:scenarioId(),requestIndex:item.requestIndex,tokenIndex:item.token,positionId:item.positionId,sparseBlockIndex:item.sparseBlock,valid:item.valid,validKvCount:item.validKvCount,queuePosition:info.queuePosition,laneId:info.block,laneIteration:info.iteration});
      }else if(selection?.type==='block')onSelection({type:'block',block:selection.block,rank:state.rank,scenarioId:scenarioId(),laneId:selection.block});
      else if(selection?.type==='request'){state.requestIndex=selection.requestIndex;onSelection({type:'request',requestIndex:selection.requestIndex,rank:state.rank,scenarioId:scenarioId()});}
      else if(selection?.type==='query'){state.requestIndex=selection.requestIndex;onSelection({type:'query',requestIndex:selection.requestIndex,tokenIndex:selection.tokenIndex,rank:state.rank,scenarioId:scenarioId()});}
      else onContext({rank:state.rank,scenarioId:scenarioId()});
      refreshSelection(d,true,{animateCamera:true});
    }
    function changeContext(){stop();state.selection=null;state.inspectorOpen=false;state.requestIndex=0;state.overrides.clear();state.layout=state.phase;state.scale=1;state.panX=state.panY=0;state.manualCamera=false;onContext({rank:state.rank,scenarioId:scenarioId()});render();}
    // 外部（Pipeline 的 Rank 下拉）切 Rank：走同一条 changeContext，保证 Scenario / Focus / 选择一起重算
    function setRank(rank){if(rank==null||rank===state.rank)return;state.rank=rank;changeContext();}
    host.addEventListener('click',event=>{const b=event.target.closest('button,[data-qk-item],[data-qk-lane],[data-qk-query],[data-qk-order-col],[data-qk-request-group]');if(!b){if(event.target.closest('.csa-qk-canvas-viewport')&&state.selection)select(null);return;}
      if(b.dataset.qkColor){const next=b.dataset.qkColor;if(next!==state.colorBy){state.colorBy=next;render();}return;}
      if(b.dataset.qkOrder){stop();morphOrder(b.dataset.qkOrder);return;}
      if(b.dataset.qkOpenPipeline!=null){const raw=b.dataset.qkOpenPipeline;const lane=raw===''?null:Number(raw);onOpenPipeline({actId:state.activeAct,lane,rank:state.rank});return;}
      if(b.dataset.qkOpenGraph){onOpenGraph(b.dataset.qkOpenGraph,b.dataset.qkActCtx||state.activeAct);return;}
      if(b.dataset.qkSpeed){
        const next=Number(b.dataset.qkSpeed);
        if(Number.isFinite(next)&&next>0){
          state.speed=next;
          host.querySelectorAll('[data-qk-speed]').forEach(btn=>{const active=Number(btn.dataset.qkSpeed)===next;btn.classList.toggle('is-selected',active);btn.setAttribute('aria-pressed',String(active));});
        }
        return;
      }
      if(b.dataset.qkPreviewLayout){
        stop();
        const target=b.dataset.qkPreviewLayout, prev=state.layout;
        if(PHASES.includes(target)){
          state.phase=target;
          host.querySelectorAll('[data-qk-phase]').forEach(btn=>{const active=btn.dataset.qkPhase===target;btn.classList.toggle('is-selected',active);btn.setAttribute('aria-pressed',String(active));});
        }
        const mode=prev==='generation'&&target==='flatten'||prev==='flatten'&&target==='generation'?'gen-flatten':prev==='flatten'&&target==='ordering'||prev==='ordering'&&target==='flatten'?'flatten-order':prev==='ordering'&&target==='stride24'?'flex-narrow':prev==='stride24'&&target==='ordering'?'flex-widen':prev==='stride24'&&target==='assignment'||prev==='assignment'&&target==='stride24'?'stride24-assign':'generic';
        const previewSpot=(prev==='generation'&&target==='flatten')||(prev==='flatten'&&target==='ordering')?{type:'request',index:0}:(prev==='stride24'&&target==='assignment')?{type:'lane',index:0}:null;
        runMorph(derive(),target,{fromLayout:prev,mode,duration:mode.startsWith('flex-')?1450:860,spotlight:previewSpot,clearSpotlightOnDone:!!previewSpot});
        return;
      }
      if(b.dataset.qkTraceMode){
        stop();
        const mode=b.dataset.qkTraceMode, d=derive();
        if(mode==='all'){select(null);return;}
        if(mode==='request'){const r=state.selection?.type==='request'?state.selection.requestIndex:defaultTraceRequest(d);select({type:'request',requestIndex:r},{openDrawer:false});return;}
        if(mode==='lane'){const l=state.selection?.type==='block'?state.selection.block:defaultTraceLane(d);select({type:'block',block:l},{openDrawer:false});return;}
        return;
      }
      if(b.dataset.qkFocusReq!=null){stop();const r=Number(b.dataset.qkFocusReq);select({type:'request',requestIndex:r},{openDrawer:false});if(b.dataset.qkGoPhase)updatePhase(b.dataset.qkGoPhase);return;}
      if(b.dataset.qkFocusLane!=null){stop();const l=Number(b.dataset.qkFocusLane);select({type:'block',block:l},{openDrawer:false});if(b.dataset.qkGoPhase)updatePhase(b.dataset.qkGoPhase);return;}
      if(b.dataset.qkGuidedPhase){stop();updatePhase(b.dataset.qkGuidedPhase);return;}
      if(b.hasAttribute('data-qk-toggle-drawer')){if(state.selection){state.inspectorOpen=!state.inspectorOpen;refreshSelection(derive(),true,{animateCamera:true});}return;}
      if(b.dataset.qkItem!=null){select({type:'item',id:Number(b.dataset.qkItem)});return;}
      if(b.dataset.qkQuery!=null){const tokenIndex=Number(b.dataset.qkQuery);select({type:'query',tokenIndex,requestIndex:Math.floor(tokenIndex/TOKENS_PER_REQUEST)});return;}
      if(b.dataset.qkOrderCol!=null){const col=Number(b.dataset.qkOrderCol);if(state.selection?.type==='block'&&state.selection.block===col)select(null);else select({type:'block',block:col},{openDrawer:false});return;}
      if(b.dataset.qkRequestGroup!=null){const r=Number(b.dataset.qkRequestGroup);if(state.selection?.type==='request'&&state.selection.requestIndex===r)select(null);else select({type:'request',requestIndex:r},{openDrawer:false});return;}
      if(b.dataset.qkLane!=null){select({type:'block',block:Number(b.dataset.qkLane)});return;}
      if(b.dataset.qkPhase){stop();updatePhase(b.dataset.qkPhase);return;}
      if(b.hasAttribute('data-qk-play')){state.playing?stop():start();return;}
      if(b.hasAttribute('data-qk-replay')){stop();state.phase='generation';state.layout='generation';render();start();return;}
      if(b.hasAttribute('data-qk-close')){select(null);return;}
      if(b.hasAttribute('data-qk-advanced')){state.advanced=!state.advanced;if(!state.advanced)state.overrides.clear();select(state.selection);return;}
      if(b.hasAttribute('data-qk-toggle')){const id=Number(b.dataset.qkToggle);if(state.advanced&&id%SPARSE_BLOCKS!==0){state.overrides.set(id,!derive().byId.get(id).valid);select(state.selection);}return;}
      if(b.hasAttribute('data-qk-reset')){state.overrides.clear();select(state.selection);return;}
      if(b.dataset.qkZoom){state.manualCamera=true;state.scale=Math.max(.7,Math.min(4,state.scale*Number(b.dataset.qkZoom)));camera();return;}
      if(b.hasAttribute('data-qk-fit')){state.manualCamera=false;state.scale=1;state.panX=state.panY=0;refreshSelection(derive(),false,{animateCamera:true});return;}
      if(b.dataset.qkSource){const [file,a,z]=b.dataset.qkSource.split(':');stop();onSource(file,Number(a),Number(z));}
    });
    host.addEventListener('change',event=>{const t=event.target;
      if(t.matches('[data-qk-workload]')){state.workload=t.value;changeContext();}
      if(t.matches('[data-qk-position]')){state.position=Number(t.value);changeContext();}
      if(t.matches('[data-qk-rank]')){state.rank=Number(t.value);changeContext();}
      if(t.matches('[data-qk-request]'))select({type:'request',requestIndex:Number(t.value)},{openDrawer:false});
      if(t.matches('[data-qk-lane-select]')){if(t.value==='')select(null);else select({type:'block',block:Number(t.value)},{openDrawer:false});}
      if(t.matches('[data-qk-loop]'))state.loop=t.checked;
    });
    host.addEventListener('keydown',event=>{if(event.key==='Escape'&&state.selection){select(null);}else if(event.key==='Enter'&&event.target.dataset.qkItem!=null)select({type:'item',id:Number(event.target.dataset.qkItem)});else if(event.key==='Enter'&&event.target.dataset.qkQuery!=null){const tokenIndex=Number(event.target.dataset.qkQuery);select({type:'query',tokenIndex,requestIndex:Math.floor(tokenIndex/TOKENS_PER_REQUEST)});}else if(event.key==='Enter'&&event.target.dataset.qkOrderCol!=null){select({type:'block',block:Number(event.target.dataset.qkOrderCol)},{openDrawer:false});}else if(event.key==='Enter'&&event.target.dataset.qkRequestGroup!=null){select({type:'request',requestIndex:Number(event.target.dataset.qkRequestGroup)},{openDrawer:false});}else if(event.key==='Enter'&&event.target.dataset.qkLane!=null)select({type:'block',block:Number(event.target.dataset.qkLane)});else if(['ArrowRight','ArrowLeft','ArrowDown','ArrowUp'].includes(event.key)&&event.target.dataset.qkQuery!=null){event.preventDefault();const tokenIndex=Math.max(0,Math.min(derive().items.length/SPARSE_BLOCKS-1,Number(event.target.dataset.qkQuery)+(event.key==='ArrowRight'||event.key==='ArrowDown'?1:-1)));select({type:'query',tokenIndex,requestIndex:Math.floor(tokenIndex/TOKENS_PER_REQUEST)});host.querySelector('[data-qk-query="'+tokenIndex+'"]').focus();}else if(['ArrowRight','ArrowLeft','ArrowDown','ArrowUp'].includes(event.key)&&event.target.dataset.qkItem!=null){event.preventDefault();const step=event.key==='ArrowRight'?1:event.key==='ArrowLeft'?-1:event.key==='ArrowDown'?5:-5;const id=Math.max(0,Math.min(derive().items.length-1,Number(event.target.dataset.qkItem)+step));select({type:'item',id});host.querySelector('[data-qk-item="'+id+'"]').focus();}});
    host.addEventListener('wheel',event=>{if(!event.target.closest('.csa-qk-canvas-viewport'))return;event.preventDefault();stop();state.manualCamera=true;state.scale=Math.max(.7,Math.min(4,state.scale*Math.exp(-event.deltaY*.001)));camera();},{passive:false});
    host.addEventListener('pointerover',event=>{const item=event.target.closest('[data-qk-item]'),lane=event.target.closest('[data-qk-lane],[data-qk-order-col]');if(!visualUI&&state.playing&&(item||lane))stop();const tooltip=host.querySelector('[data-qk-tooltip]');if(!tooltip||(!item&&!lane))return;const d=derive();if(item){const data=d.byId.get(Number(item.dataset.qkItem)),info=itemDetails(d,data.id);const base='Work Item '+data.id+' · R'+data.requestIndex+' Q'+data.token+' / SB'+data.sparseBlock+(data.valid?(data.sparseBlock===0?' (SB0 滑窗 ~8µs)':' (CMP 压缩 ~40µs)'):' (invalid)');if(state.layout==='generation'||state.layout==='flatten')tooltip.textContent=base+' → 目标 qk_order['+info.queuePosition+']';else if(state.layout==='ordering')tooltip.textContent=base+' · 排序下标 qk_order['+info.queuePosition+']';else tooltip.textContent=base+' · qk_order['+info.queuePosition+'] · Lane '+info.block+' / 第 '+info.iteration+' 轮';}else{const lIdx=Number(lane.dataset.qkLane??lane.dataset.qkOrderCol);const b=d.blocks[lIdx],pg=physicalGroupOf(b.index);tooltip.textContent='Lane '+String(b.index).padStart(2,'0')+(hasRuntimeTrace()&&state.workload==='trace'?' · 核组 '+String(pg).padStart(2,'0'):'')+' · qk_order['+b.index+', '+(b.index+d.blockCount)+', …] · '+b.validCount+' valid';}tooltip.hidden=false;});
    host.addEventListener('pointerout',event=>{if(event.target.closest('[data-qk-item],[data-qk-lane],[data-qk-order-col]')&&!event.relatedTarget?.closest?.('[data-qk-item],[data-qk-lane],[data-qk-order-col]')){const tooltip=host.querySelector('[data-qk-tooltip]');if(tooltip)tooltip.hidden=true;}});
    host.addEventListener('pointerdown',event=>{if(event.button!==0||!event.target.closest('.csa-qk-canvas-viewport')||event.target.closest('[data-qk-item],[data-qk-lane],[data-qk-order-col],[data-qk-request-group]'))return;drag={x:event.clientX,y:event.clientY,panX:state.panX,panY:state.panY};});
    host.addEventListener('pointermove',event=>{const viewport=host.querySelector('.csa-qk-canvas-viewport'),tooltip=host.querySelector('[data-qk-tooltip]');if(tooltip&&!tooltip.hidden&&viewport){const r=viewport.getBoundingClientRect();tooltip.style.left=Math.min(r.width-240,event.clientX-r.left+12)+'px';tooltip.style.top=Math.max(8,event.clientY-r.top-32)+'px';}if(!drag||!viewport)return;const r=viewport.getBoundingClientRect(),fit=fitBox();state.manualCamera=true;state.panX=drag.panX+(event.clientX-drag.x)*(fit.w/state.scale)/r.width;state.panY=drag.panY+(event.clientY-drag.y)*(fit.h/state.scale)/r.height;camera();});
    host.addEventListener('pointerup',()=>{drag=null;});
    render();
    if(global.ResizeObserver)new global.ResizeObserver(()=>{if(!host.hidden){if(state.selection&&!state.manualCamera)focusCamera(derive());else camera();}}).observe(host);
    function syncRuntimeInput(){if(!hasRuntimeTrace()||state.workload==='trace')return derive();state.workload='trace';state.overrides.clear();onContext({rank:state.rank,scenarioId:scenarioId()});render();return derive();}
    return {render,pause:()=>{stop();if(trailTimer){global.clearTimeout(trailTimer);trailTimer=null;}if(cameraFrame)cancelAnimationFrame(cameraFrame);},setSelection:selection=>{if(JSON.stringify(selection)===JSON.stringify(state.selection))return;state.selection=selection?{...selection}:null;state.inspectorOpen=!!selection;if(selection?.type==='request'||selection?.type==='query')state.requestIndex=selection.requestIndex;else if(selection?.type==='item')state.requestIndex=Math.floor(selection.id/(SPARSE_BLOCKS*TOKENS_PER_REQUEST));refreshSelection(derive());},setStoryAct,syncRuntimeInput,getState:()=>({mode:state.mode,workload:state.workload,position:state.position,rank:state.rank,requestIndex:state.requestIndex,phase:state.phase,layout:state.layout,speed:state.speed,colorBy:state.colorBy,orderPolicy:state.orderPolicy,activeAct:state.activeAct,selection:state.selection?{...state.selection}:null,inspectorOpen:state.inspectorOpen}),getScenario:derive,setRank};
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
