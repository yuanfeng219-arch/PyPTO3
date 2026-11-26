/* Shared kernel-stage canvas. Domain models supply task, tile and operation semantics. */
(function (global) {
  'use strict';
  const phases = ['generation', 'ordering', 'assignment'];
  const phaseNames = {generation:'Work Generation', ordering:'Block Ordering', assignment:'Core Assignment'};
  const reduceMotion = global.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const escape = value => String(value).replace(/[&<>"']/g, char =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[char]);
  const button = (label, attr, value, selected = false) =>
    `<button type="button" class="btn btn-sm${selected ? ' is-selected' : ''}" ${attr}="${escape(value)}">${escape(label)}</button>`;

  function glyph(title, detail, role, unit, attributes) {
    return `<button type="button" class="viz2-instance-node csa-kvsp-node" data-role="${role}" ${attributes}>
      <span class="viz2-inst-head"><b>${escape(title)}</b>${unit ? `<em class="viz2-inst-unit" data-unit="${unit.toLowerCase()}">${unit}</em>` : ''}</span>
      <small>${escape(detail)}</small></button>`;
  }

  function operationScene(graph, selected, kernel, pipeline, expanded, identify, iterations, mIndex) {
    const marker=mIndex==null?'csa-kvsp-arrow':`csa-kvsp-arrow-m${mIndex}`;
    const {nodes,edges}=graph, byId=new Map(nodes.map(n=>[n.id,n]));
    const laneLayout=pipeline||kernel?.type==='projection'&&expanded;
    const laneOrder=graph.pipelineLanes
      ? ['Input','Load','Cube','Split 0 · Load','Split 0 · Cube','Split 0 · Write','Split 1 · Load','Split 1 · Cube','Split 1 · Write','Write']
      : ['Input','Control','MTE2','MTE1','Gather','Vector','M','FIX','Write'];
    const usedLanes=new Set(nodes.map(n=>n.lane));
    const lanes=[...laneOrder.filter(lane=>usedLanes.has(lane)),...[...usedLanes].filter(lane=>!laneOrder.includes(lane))];
    const windowColumns=new Map(nodes.filter(n=>n.op==='carry').map(n=>[n.step,n.rank]));
    const groups=new Map();
    const focused=new Set(nodes.filter(n=>selected && (selected.overview
      ? n.id===selected.overview
      : n.op===selected.op && (selected.split==null||n.split===selected.split) &&
        (!expanded||selected.step==null||n.step===selected.step))).map(n=>n.id));
    const adjacent=new Set(focused);
    for(const edge of edges) if(focused.has(edge.from)||focused.has(edge.to)) {
      adjacent.add(edge.from);adjacent.add(edge.to);
    }

    for(const node of nodes) {
      const projectionStep=kernel?.type==='projection'&&node.step!=null;
      node.column=node.layoutColumn??(projectionStep ? 1+node.step
        : kernel?.type==='projection' && expanded && node.role==='aggregate' ? iterations+1
        : pipeline && kernel?.type==='pool' && node.op==='window' ? windowColumns.get(node.step)
        : node.rank);
      const key=laneLayout?node.lane+':'+node.column:node.column;
      if(!groups.has(key))groups.set(key,[]);
      const slots=groups.get(key);node.slot=slots.length;slots.push(node);
    }
    const rowSize=65, nodeWidth=168, nodeHeight=44, colSize=216, margin=laneLayout?139:30;
    const rowByLane=new Map();
    let y=52;
    if(laneLayout) for(const lane of lanes) {
      const depth=Math.max(...[...groups].filter(([key])=>key.startsWith(lane+':')).map(([,slots])=>slots.length));
      const size=depth*rowSize+12;
      rowByLane.set(lane,{y,size}); y+=size;
    }
    for(const node of nodes) {
      node.x=margin+node.column*colSize;
      node.y=(laneLayout?rowByLane.get(node.lane).y+8:46)+(laneLayout?node.slot:node.layoutRow??node.slot)*rowSize;
    }
    const width=Math.max(...nodes.map(n=>n.x))+nodeWidth+40;
    const height=laneLayout?y+12:Math.max(...nodes.map(n=>n.y))+nodeHeight+28;
    const ports=new Map();
    for(const edge of edges) for(const [side,id] of [['from',edge.from],['to',edge.to]]) {
      const key=side+':'+id;
      if(!ports.has(key))ports.set(key,[]);
      ports.get(key).push(edge);
    }
    const portOffset=(side,id,edge)=>{
      const items=ports.get(side+':'+id);
      return 7+(nodeHeight-14)*(items.indexOf(edge)+1)/(items.length+1);
    };
    const occupiedRows=new Map();
    const links=edges.map(edge=>{
      const from=byId.get(edge.from), to=byId.get(edge.to);
      const a=from.x+nodeWidth, b=to.x;
      const sy=from.y+portOffset('from',from.id,edge), ty=to.y+portOffset('to',to.id,edge);
      const related=focused.has(from.id)||focused.has(to.id);
      const outgoingIndex=ports.get('from:'+from.id).indexOf(edge);
      const incomingIndex=ports.get('to:'+to.id).indexOf(edge);
      let path;
      if(from.column===to.column){
        const gutter=a+18+outgoingIndex*3+incomingIndex*6;
        path=`M${a} ${sy} C${gutter} ${sy} ${gutter} ${ty} ${a} ${ty}`;
      }else if(to.column-from.column>1){
        const row=from.y+nodeHeight;
        const occupied=occupiedRows.get(row)||[];
        const lane=Math.max(0,...occupied.filter(track=>track.to>a && track.from<b).map(track=>track.lane+1));
        occupied.push({from:a,to:b,lane});
        occupiedRows.set(row,occupied);
        const corridor=row+6+Math.min(lane*2,12);
        const exit=a+9+outgoingIndex*4, entry=b-9-incomingIndex*5;
        const turn=Math.sign(ty-corridor);
        const radius=Math.min(5,Math.abs(ty-corridor)/2);
        path=`M${a} ${sy} H${exit-5} Q${exit} ${sy} ${exit} ${sy+5} V${corridor-5}
          Q${exit} ${corridor} ${exit+5} ${corridor} H${entry-radius}
          Q${entry} ${corridor} ${entry} ${corridor+turn*radius} V${ty-turn*radius}
          Q${entry} ${ty} ${entry+radius} ${ty} H${b}`;
      }else{
        path=`M${a} ${sy} C${a+30} ${sy} ${b-30} ${ty} ${b} ${ty}`;
      }
      return `<path class="${pipeline?'csa-qp-timeline-data':''} ${related?'is-related':''}" data-kind="${edge.type}" d="${path}" style="marker-end:url(#${marker})"/>`;
    }).join('');
    return `<div class="csa-kvsp-graph csa-compressor-op-scene ${focused.size?'has-focus':''}" style="width:${width}px;height:${height}px">
      <svg class="${pipeline?'csa-qp-svg':''}" width="${width}" height="${height}" aria-hidden="true">
      <defs><marker id="${marker}" viewBox="0 0 8 8" markerWidth="6" markerHeight="6" markerUnits="userSpaceOnUse" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z"/></marker></defs>
      ${laneLayout?lanes.map(lane=>{
        const {y,size}=rowByLane.get(lane);
        return `<rect class="csa-compressor-region" x="10" y="${y}" width="${width-20}" height="${size-6}" rx="8"/><text class="csa-qp-row-label" x="24" y="${y+34}">${lane}</text>`;
      }).join(''):''}${links}</svg>
      ${nodes.map(node=>glyph(node.title,node.detail||node.condition||node.lane,node.role,node.unit||(node.role==='operation'?kernel?.coreType:''),
        `style="left:${node.x}px;top:${node.y}px;width:${nodeWidth}px;height:${nodeHeight}px" ${kernel?'data-kvsp-op':'data-kvsp-overview'}="${node.id}" data-object-id="${identify(node)}" ${mIndex==null?'':`data-kvsp-m="${mIndex}"`} title="${escape([node.title,node.detail,node.condition].filter(Boolean).join(' · '))}" data-related="${adjacent.has(node.id)}" aria-pressed="${focused.has(node.id)}"`)).join('')}</div>`;
  }

  // Preserve the viewport and logical SVG nodes when selection or phase changes.
  function syncElement(current,next) {
    if(current.isEqualNode(next))return;
    if(current.nodeName!==next.nodeName || current.nodeType!==next.nodeType ||
        current.nodeType===1 && current.getAttribute('data-owner')!==next.getAttribute('data-owner')) {
      current.replaceWith(next.cloneNode(true));return;
    }
    if(current.nodeType!==1){current.nodeValue=next.nodeValue;return;}
    const userExpanded = current.localName==='details';
    for(const attr of [...current.attributes]) {
      if(attr.name==='open'&&userExpanded)continue;
      if(!next.hasAttribute(attr.name))current.removeAttribute(attr.name);
    }
    for(const attr of next.attributes) {
      if(attr.name==='open'&&userExpanded)continue;
      if(current.getAttribute(attr.name)!==attr.value)current.setAttribute(attr.name,attr.value);
    }
    const existing=[...current.childNodes], incoming=[...next.childNodes];
    if(existing.length!==incoming.length)current.replaceChildren(...incoming.map(child=>child.cloneNode(true)));
    else for(let i=0;i<existing.length;i++)syncElement(existing[i],incoming[i]);
    if(current.localName==='select' && current.value!==next.value)current.value=next.value;
  }

  function mount(host,{onSource,onRank,onExit,onTab,model=global.PtoCsaCompressorModel,renderTiles,pipelineFactory}) {
    const {stage,measured:data,kernels:kernelByName,rows,kTile:inner,kCount:iterations,ticks,range,objectId,operationModel}=model;
    const title=model.title||'Compressor', layoutBranches=model.branches||{main:'Main',indexer:'Indexer'};
    const caseOptions=model.cases||[];
    let customPipelineHost=null, customPipelineCtrl=null, pendingPipelineState=null, pipelineSyncedKernel=null, kernelExplicitlyChosen=false;
    let tokens=caseOptions[0]?.tokens, tileIndex=0, selectedK=0, overviewId=null, allTiles=false;
    let rank=0,mode='assignment',scope='inter',scale='structure',block=0,kernel=model.defaultKernel||'kv_score_proj';
    const info=()=>kernelByName.get(kernel);
    const isProjection=name=>kernelByName.get(name).type==='projection';
    const localView=()=>({block,tokens,tileIndex});
    const tile=(index,name)=>model.blockShape(kernelByName.get(name),index,localView());
    const localGraph=()=>operationModel(info(),scale==='instances',localView());
    const defaultCount=name=>kernelByName.get(name).blockNum;
    const countFor=name=>model.blockCount?model.blockCount(kernelByName.get(name),tokens):defaultCount(name);
    const drawTiles=()=>renderTiles?.(info(),localView(),selectedK)||'';

    let sliceId = null, selected = null, blockSelected = false, inspectorOpen = false;
    let phase = 'generation', timer = null, loop = false, zoom = 1;
    let coreId = null, filter = 'all', branch = 'all', lastKey = null;
    let navigation = [], focusSelector = null, savedFocus = null;
    const cameras = new Map();
    const keyFor = () => JSON.stringify([rank,tokens,mode,scope,scope==='intra'?kernel:null,scope==='intra'?tileIndex:null,mode==='assignment'?phase:null,
      mode==='graph'||scope==='intra'?scale:null,mode==='graph'&&scale==='instances'&&allTiles]);
    const viewport=()=>host.querySelector('.csa-kvsp-main');
    const remember=()=>{
      const region=viewport();
      if(lastKey&&region)cameras.set(lastKey,{left:region.scrollLeft,top:region.scrollTop,zoom});
    };
    const updateZoom=()=>{
      host.querySelectorAll('.csa-compressor-sheet,.csa-compressor-zoom-content').forEach(el=>el.style.setProperty('--compressor-zoom',zoom));
      const output=host.querySelector('.csa-compressor-footer output');
      if(output)output.textContent=`${Math.round(zoom*100)}%`;
    };
    function frame(selector, smooth=true) {
      const item=viewport()?.querySelector(selector);
      if(!item||!item.getClientRects().length)return;
      const box=item.getBoundingClientRect(), view=viewport().getBoundingClientRect();
      if(box.left<view.left||box.right>view.right||box.top<view.top||box.bottom>view.bottom) {
        viewport().scrollBy({left:box.left+box.width/2-view.left-view.width/2,top:box.top+box.height/2-view.top-view.height/2,behavior:!smooth||reduceMotion?'instant':'smooth'});
      }
    }
    function focusObject(selector) { focusSelector=selector; }
    const jumpLabel = label => `在 ${label} 中查看`;
    function navActions(targetScope, marker) {
      const options={assignment:'Assignment',pipeline:'Pipeline',graph:'Graph'};
      return `<div class="csa-qp-nav-group">${Object.entries(options).filter(([id])=>id!==mode).map(([id,title])=>
        button(jumpLabel(title),'data-kvsp-nav',`${id}:${targetScope}:${marker}`)).join('')}</div>`;
    }
    function snapshot() {
      remember();
      return {mode,scope,scale,block,kernel,sliceId,selected,blockSelected,inspectorOpen,phase,coreId,filter,branch,tokens,tileIndex,selectedK,overviewId,allTiles,zoom,key:lastKey};
    }
    const stop = () => { if (timer != null) global.clearInterval(timer); timer = null; };
    function play() {
      stop();
      timer = global.setInterval(() => {
        const next = phases.indexOf(phase) + 1;
        if(next===phases.length&&!loop){stop();render();}
        else setPhase(phases[next%phases.length]);
      }, 1500);
    }
    function setKernel(name) {
      kernelExplicitlyChosen=true;
      pendingPipelineState=null;
      if(kernel===name)return;
      kernel=name; block=0; tileIndex=0; allTiles=false; selectedK=0; overviewId=null; selected=null; sliceId=null; coreId=null; blockSelected=false;
      if(branch!=='all'&&kernelByName.get(kernel).branch!==branch)branch='all';
    }
    const phaseButton = (name, attr, value, pressed) =>
      button(name, attr, value, pressed).replace('<button ', `<button aria-pressed="${pressed}" `);
    const current = () => data.ranks[rank];
    const blocks = () => countFor(kernel);

    function blockGrid(indices, label, name = kernel) {
      return `<div class="csa-kvsp-blocks" role="group" aria-label="${label}">${indices.map(index =>
        `<button type="button" class="btn btn-sm" data-kvsp-block="${index}" data-kvsp-block-kernel="${name}" aria-label="查看 Block ${index}" title="${escape(model.cases?tile(index,name).text:isProjection(name)?`M ${range(tile(index,name).row,rows)} · N ${range(tile(index,name).col,tile(index,name).n)}`:`逻辑 Block ${index}`)}">${index}</button>`).join('')}</div>`;
    }

    function assignment() {
      if (scope === 'intra') {
        if(renderTiles&&model.isMatmul(info()))return `<h3>${kernel} · Block ${block}</h3>${drawTiles()}${navActions('intra','block')}`;
        if (!isProjection(kernel)) return localAssignment();
        const { row, col, n } = tile(block, kernel);
        return `<h3>Block ${block} · Tile 与存储</h3>
          <p>M ${range(row, rows)} · N ${range(col, n)} · ${iterations} × K${inner}</p>
          <div class="csa-sv-table-wrap"><table class="csa-sv-table"><thead><tr><th>对象</th><th>逻辑 Tile</th><th>存储路径</th></tr></thead><tbody>
          <tr><td>X</td><td>BF16 · 16 × 512 / K Tile</td><td>GM → L1 (Mat) → L0A</td></tr>
          <tr><td>Wkv / Wgate</td><td>BF16 · ${n} × 512 / K Tile，各一份</td><td>GM → L1 (Mat) → L0B · 转置权重</td></tr>
          <tr><td>KV / Score 累加器</td><td>FP32 · 16 × ${n}，各一份</td><td>L0C → GM · 两个输出 Tile</td></tr>
          </tbody></table></div>${button('查看源码', 'data-kvsp-source', '110:136')}`;
      }
      return assignmentCanvas();
    }

    function assignmentCanvas() {
      const count = blocks(), projection = isProjection(kernel);
      const perRow=model.groupSize?.(info())??(projection?tile(0,kernel).nTiles:16);
      const groupCount=Math.ceil(count/perRow), sequenceColumns=32;
      const groupColumns=4, groupWidth=254, groupHeight=projection?79:69;
      const size=projection||model.groupSize?perRow:8;
      const generatedGroups=Math.ceil(count/size);
      const groupOrigin=i=>[22+i%groupColumns*groupWidth,58+Math.floor(i/groupColumns)*groupHeight];
      const records = current().kernels[kernel];
      const cores = new Map();
      for (const record of records) {
        if (!cores.has(record[1])) cores.set(record[1], []);
        cores.get(record[1]).push(record);
      }
      const byCore = [...cores].sort((a,b)=>a[0]-b[0]);
      const showMeasured = phase === 'assignment';
      const ordered = phase !== 'generation';
      const gridColumns = ordered ? sequenceColumns : perRow;
      const width = 1080, step = 27, rowStep = 23, left = 115, top = 68;
      const taskHeight=ordered?Math.ceil(count/gridColumns)*rowStep+top+14
        : 58+Math.ceil(generatedGroups/groupColumns)*groupHeight;
      const measuredTop = top;
      const height = showMeasured ? measuredTop+byCore.length*rowStep+18 : Math.max(310,taskHeight+22);
      const ownership = model.cases ? info().type==='qr'?`${count/2} N Groups × 2 Split-K`:
        info().type==='qmat'?`${count} N Groups × 1 Block`:info().type==='post'?`${count} Head Groups × 4 Heads`:
        info().type==='seed'?'1 Core Group · 遍历 M/N Tile':
        info().type==='qkplan'?`1 Core Group · 遍历 ${tile(0,kernel).tiles.length} Token Tiles + 2-Pass 重排`:
        info().type==='ropecs'?`1 Core Group · Swap Index + 遍历 ${tile(0,kernel).tiles.length} Token Tiles`:
        `${count} Token Tiles × 1 Block`
        : projection ? `${groupCount} M Tiles × ${perRow} N Tiles` : /pool|commit/.test(kernel) ? `${count} requests × 1 Block` : `${count} 个 16 行 Tiles × 1 Block`;
      const caption = phase === 'generation' ? `${ownership} → ${count} Work Items` :
        phase === 'ordering' ? `SPMD Block 0–${count-1} · ${info().type==='qr'?'block_idx = N Group × 2 + Split':info().type==='qkplan'||info().type==='ropecs'?'单 Core Group 串行派发 (pl.at)':'按 block_idx 编号'}` :
        `${count} 逻辑 Blocks · ${records.length} 实测切片 · ${cores.size} 个物理核${model.cases&&tokens!==caseOptions[0].tokens?'（存档为 256 行）':''}`;
      return `<div class="csa-qk-canvas-head csa-compressor-controls">
        <div><h2>${phaseNames[phase]}</h2><p>${caption}</p></div>
        <div class="csa-qk-canvas-controls">
          ${button(timer == null ? '播放' : '暂停','data-kvsp-play','')}
          ${button('重播','data-kvsp-replay','')}
          <div class="segmented-control segmented-control-muted" role="group" aria-label="任务分配阶段">
            ${phases.map((p,i)=>phaseButton(['Generation','Ordering','Assignment'][i],'data-kvsp-phase',p,p===phase)).join('')}
          </div>${phaseButton('循环','data-kvsp-loop','',loop)}
        </div></div>
        <div class="csa-compressor-sheet" style="--compressor-zoom:${zoom}">
          ${renderTiles&&model.isMatmul(info())?`<div ${showMeasured?'hidden':''}>${drawTiles()}</div>`:''}
          <svg class="csa-compressor-assignment" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="group" aria-label="${title} Work Assignment">
          <text x="22" y="26" class="csa-qk-canvas-axis">${escape(phaseNames[phase] + ' · ' + caption)}</text>
          <g data-kvsp-logical-backdrop display="${showMeasured?'none':'inline'}">
          ${ordered?ticks(Math.ceil(count/gridColumns)).map(i=>`<g class="csa-qk-canvas-group">
            <rect x="100" y="${top+i*rowStep-3}" width="${gridColumns*step+20}" height="22" rx="4"/>
            <text x="20" y="${top+i*rowStep+10}">B ${i*gridColumns}</text></g>`).join(''):
            ticks(generatedGroups).map(i=>{
              const [x,y]=groupOrigin(i), first=i*size;
              const active=blockSelected&&Math.floor(block/size)===i;
              const name=model.groupSize?` ${info().type==='qr'?'N Group '+i+' · Split 0 / 1':
                info().type==='qmat'?'N Groups '+first+'–'+Math.min(first+size-1,count-1):
                info().type==='post'?'Head Groups '+first+'–'+Math.min(first+size-1,count-1):
                info().type==='seed'?'QR FP32 初始化':
                info().type==='qkplan'?'Candidate Planning · 1 Core Group':
                info().type==='ropecs'?'Inverse RoPE · 1 Core Group':
                'Token Tiles '+first+'–'+Math.min(first+size-1,count-1)}`
                :projection?`M ${range(i*rows,rows)} × N Tiles`
                : `${model.requestScoped(kernelByName.get(kernel))?'Request':'M Tile'} ${first}–${Math.min(first+7,count-1)}`;
              return `<g class="csa-qk-canvas-group ${active?'is-focused':blockSelected?'is-dimmed':''}">
                <rect x="${x}" y="${y}" width="${groupWidth-12}" height="${groupHeight-8}" rx="5"/>
                <text x="${x+10}" y="${y+17}">${name}</text></g>`;
            }).join('')}</g>
          <g data-kvsp-logical-layer display="${showMeasured?'none':'inline'}" data-owner="${kernel}">
          ${ticks(count).map(i=>{
            const [baseX,baseY]=groupOrigin(Math.floor(i/size));
            const x=ordered?left+i%gridColumns*step:baseX+10+(i%size)%8*28;
            const y=ordered?top+Math.floor(i/gridColumns)*rowStep:baseY+27+Math.floor((i%size)/8)*19;
            const active=blockSelected&&block===i;
            const related=blockSelected&&Math.floor(block/size)===Math.floor(i/size);
            const dim=blockSelected&&!related;
            return `<g class="csa-qk-canvas-item is-valid ${active?'is-selected':related?'is-related':dim?'is-dimmed':''}" role="button" tabindex="0"
              data-kvsp-logical="${i}" data-object-id="${objectId(rank,kernelByName.get(kernel),i,null,localView())}" data-x="${x}" data-y="${y}" style="transform:translate(${x}px,${y}px)" aria-pressed="${active}" aria-label="逻辑 Block ${i}">
              <rect width="22" height="16" rx="3"/><title>Block ${i} · ${tile(i,kernel).text}</title></g>`;
          }).join('')}</g>
          <g data-kvsp-measured display="${showMeasured?'inline':'none'}">
          ${showMeasured ? `<text x="22" y="${measuredTop-22}" class="csa-qk-canvas-axis">按物理核分组 · 每个色块是一条实测切片${blockSelected?` · 逻辑 Block ${block} 保留在 Ordering`:''}</text>
            ${byCore.map(([core, entries],i)=>`<g class="csa-qk-canvas-lane ${coreId===core?'is-selected':coreId!=null?'is-dimmed':''}" role="button" tabindex="0" data-kvsp-core="${core}" aria-label="物理核 ${core}">
              <rect x="100" y="${measuredTop+i*rowStep-3}" width="916" height="22" rx="4"/>
              <text x="20" y="${measuredTop+i*rowStep+10}">${core<24?'AIC':'AIV'} ${core}</text></g>
              ${entries.map(([id,,time,dur],j)=>`<rect class="csa-compressor-slice" role="button" tabindex="0" data-kvsp-slice="${id}" data-unit="${core<24?'aic':'aiv'}"
                data-highlight="${coreId==null||coreId===core}" aria-pressed="${sliceId===id}" aria-label="物理切片 ${id}，Core ${core}" x="${left+j*Math.min(step,875/entries.length)}" y="${measuredTop+i*rowStep}" width="${Math.min(22,800/entries.length)}" height="16" rx="3"><title>#${id} · ${time} µs · ${dur} µs</title></rect>`).join('')}`).join('')}` : ''}
          </g></svg></div>`;
    }

    const classify = op => /load/.test(op) ? 'Load' : /store/.test(op) ? 'Store' :
      /extract|mov/.test(op) ? 'Tile Move / Extract' : /matmul/.test(op) ? 'Cube' :
      /flag|barrier/.test(op) ? 'Sync' : 'Vector / Scalar';

    function irTable(withSource) {
      return `<div class="csa-sv-table-wrap"><table class="csa-sv-table"><thead><tr><th>PTO IR 行</th><th>指令</th><th>类别</th>${withSource ? '<th>源码</th>' : ''}</tr></thead><tbody>
        ${data.ir[kernel].map(([op,line,py])=>`<tr><td>${line}</td><td>${op}</td><td>${classify(op)}</td>
          ${withSource ? `<td>${py ? button('L'+py, 'data-kvsp-source', py+':'+py) : '—'}</td>` : ''}</tr>`).join('')}</tbody></table></div>`;
    }

    function localAssignment() {
      const k=info(), graph=operationModel(k,false,localView());
      const inputs=graph.nodes.filter(n=>n.role==='activation');
      const outputs=graph.nodes.filter(n=>n.role==='aggregate');
      return `<h3>${kernel} · Block ${block} · ${k.coreType}</h3><p>${tile(block,kernel).text}</p>
        <div class="csa-compressor-flow">${[['输入',inputs],['输出及状态',outputs]].map(([label,nodes])=>`<section class="csa-sv-card">
        <h3>${label}</h3>${nodes.map(n=>`<dl class="csa-sv-dl"><dt>${n.title}</dt><dd>${n.detail}</dd></dl>`).join('')}</section>`).join('')}</div>
        ${button('查看切分与读写源码','data-kvsp-source',k.first+':'+k.last)}`;
    }

    function localOperations(pipeline) {
      const expanded=scale==='instances', info=kernelByName.get(kernel);
      const symbolic = ['pool','commit','write'].includes(info.type) || info.type==='norm' && info.branch==='main';
      return `<h3>${kernel} · Block ${block} · ${pipeline?'核内因果泳道':'计算依赖'}</h3>
        <p>${pipeline?'逻辑因果顺序 · 不含核内实测时间戳。':''}${symbolic?' 条件路径按源码展示，Token / Slot 的实际执行由输入决定。':''}</p>
        ${model.cases?`<p>${escape(tile(block,kernel).text)}</p>`:''}
        ${operationScene(localGraph(),selected,info,pipeline,expanded,
          node=>objectId(rank,info,node.block??block,node.id,localView()),iterations)}
        ${data.ir[kernel].length?`<details class="csa-sv-card"><summary>PTO IR 与源码位置</summary>${irTable(true)}</details>`:''}`;
    }

    function computation() {
      if(scope==='intra'){
        if(scale!=='instances'||!allTiles)return localOperations(false);
        const infoForBlock=info(), shape=tile(block,kernel);
        return shape.tiles.map((m,index)=>{
          const view={...localView(),tileIndex:index};
          const graph=operationModel(infoForBlock,true,view);
          return `<section class="csa-qp-tile-band">
            <h3>${escape(m.path)} · M ${range(m.start,m.valid)}${m.valid<m.size?` · ${m.valid}/${m.size} 行`:''}</h3>
            ${operationScene(graph,index===tileIndex?selected:null,infoForBlock,false,true,
              node=>objectId(rank,infoForBlock,node.block??block,node.id,view),iterations,index)}</section>`;
        }).join('');
      }
      const graph=model.overview(tokens);
      const instanceIndex=scale==='instances'?`<section class="csa-sv-card csa-qp-instance-index">
        <h3>逻辑 Block · ${stage.kernels.reduce((total,k)=>total+countFor(k.name),0)}</h3>
        ${stage.kernels.map(k=>`<details><summary>${escape(k.name)} · ${countFor(k.name)} Blocks</summary>
          ${blockGrid(ticks(countFor(k.name)),k.name,k.name)}</details>`).join('')}</section>`:'';
      return `${operationScene(graph,overviewId?{overview:overviewId}:null,null,false,false,
        node=>[stage.id,'rank'+rank,tokens,node.id].join('/'),iterations)}${instanceIndex}`;
    }

    function pipeline() {
      if (scope==='intra') return localOperations(true);
      const slices = Object.entries(current().kernels).flatMap(([name, entries]) => entries.map(e => [...e, name]));
      const start = Math.min(...slices.map(s => s[2])), end = Math.max(...slices.map(s => s[2] + s[3]));
      const visible=slices.filter(([,core,,, , name]) => (filter==='all'||filter===name||filter===kernelByName.get(name).branch) &&
        (coreId==null||coreId===core));
      const toX = time => 74 + (time - start) / (end - start) * 1000;
      return `<div class="csa-compressor-trace-head"><h3>实测执行泳道 · Rank ${rank}</h3><label>显示
        <select data-kvsp-filter><option value="all" ${filter==='all'?'selected':''}>全部 Kernel</option>
          ${Object.entries(layoutBranches).map(([id,name])=>`<option value="${id}" ${filter===id?'selected':''}>${name}</option>`).join('')}
          ${stage.kernels.map(k=>`<option ${filter===k.name?'selected':''}>${k.name}</option>`).join('')}</select></label>
        ${coreId==null?'':button('显示所有物理核','data-kvsp-all-cores','')}</div>
        <p>${model.cases?'存档使用 256 行本地 Q 与 512 行全组 KV。':''} 显示 ${visible.length} / ${slices.length} 条切片。横轴为合并记录时间（µs）。选择切片查看所属 Kernel、物理核和耗时。</p>
        <svg class="csa-kvsp-timeline" width="1100" height="914" viewBox="0 0 1100 914" role="group" aria-label="物理 AIC 与 AIV 泳道">
        ${ticks(6).map(i => { const x = 74 + 200 * i; return `<text x="${x}" y="16" text-anchor="${i===5?'end':'start'}">${(start+(end-start)*i/5).toFixed(1)}</text><path d="M${x} 24V902"/>`; }).join('')}
        ${ticks(72).map(i => `<g class="csa-compressor-core-label" role="button" tabindex="0" data-kvsp-core="${i}" aria-label="物理核 ${i}">
          <rect x="6" y="${29+i*12}" width="60" height="11" rx="2"/><text x="12" y="${39+i*12}">${i<24?'AIC':'AIV'} ${i}</text></g><path d="M72 ${42+i*12}H1074"/>`).join('')}
        ${visible.map(([id, core, time, duration, , name]) => `<rect role="button" tabindex="0" data-kvsp-slice="${id}" data-unit="${core<24?'aic':'aiv'}" data-object-id="${objectId(rank,kernelByName.get(name),null,'slice'+id)}" data-highlight="${sliceId==null||kernel===name}" aria-label="事件 ${id}，${name}，Core ${core}，${time} 微秒" aria-pressed="${sliceId === id}" x="${toX(time)}" y="${29+core*12}" width="${Math.max(1,toX(time+duration)-toX(time))}" height="10" rx="2"><title>#${id} · ${name} · Core ${core} · ${duration} µs</title></rect>`).join('')}
        </svg><small>${escape(current().source)} · ${title}</small>`;
    }

    function detail() {
      if (scope === 'inter' && sliceId != null && mode !== 'graph') {
        const [id, core, time, duration, elapsed] = current().kernels[kernel].find(s => s[0] === sliceId);
        return `<h3>${kernel} · 事件 #${id}</h3><dl class="csa-sv-dl"><dt>物理核</dt><dd>${core<24?'AIC':'AIV'} ${core}</dd><dt>起点</dt><dd>${time} µs</dd><dt>切片耗时</dt><dd>${duration} µs</dd><dt>Kernel 耗时</dt><dd>${elapsed} µs</dd>
          ${current().ids?.[kernel]?`<dt>Task / FuncId</dt><dd>${current().ids[kernel].join(' / ')}</dd>`:''}</dl><p>此记录没有逻辑 Block ID。可以查看同一 Kernel 的逻辑 Block 列表。</p>
          ${button('选择逻辑 Block','data-kvsp-logical-list','')}${navActions('inter','slice')}`;
      }
      if(scope==='inter' && coreId!=null){
        const shown=mode==='assignment'?[kernel]:[...kernelByName.keys()].filter(name=>filter==='all'||filter===name||filter===kernelByName.get(name).branch);
        const matches=shown.flatMap(name=>current().kernels[name].filter(record=>record[1]===coreId));
        return `<h3>${coreId<24?'AIC':'AIV'} ${coreId} · Rank ${rank}</h3><p>${matches.length} 条实测切片</p>
          ${navActions('inter','core')}`;
      }
      if (mode === 'assignment' && scope === 'inter' && blockSelected) {
        const {first:start,last:end}=kernelByName.get(kernel);
        const t=tile(block,kernel);
        const ownership=model.cases?t.text:isProjection(kernel)?`M ${range(t.row,rows)} · N ${range(t.col,t.n)}`:/pool|commit/.test(kernel)?`Request ${block}`:`M ${range(block*rows,rows)}`;
        return `<h3>${kernel} · Block ${block}</h3><dl class="csa-sv-dl"><dt>Rank</dt><dd>${rank}</dd><dt>逻辑归属</dt><dd>${ownership}</dd><dt>身份</dt><dd>${objectId(rank,kernelByName.get(kernel),block,null,localView())}</dd>
          ${t.kTile?`<dt>K Iteration ${selectedK}</dt><dd>${range(t.k+selectedK*t.kTile,t.kTile)}</dd>`:''}</dl>
          ${button('进入核内','data-kvsp-block',block)}${button('查看生成规则','data-kvsp-source',start+':'+end)}${navActions('intra','block')}`;
      }
      if(scope==='inter'&&mode==='graph'&&model.overview&&overviewId){
        const graph=model.overview(tokens), node=graph.nodes.find(i=>i.id===overviewId);
        const related=direction=>graph.edges.filter(edge=>edge[direction]===node.id).map(edge=>{
          const peer=graph.nodes.find(n=>n.id===edge[direction==='to'?'from':'to']);
          return button(peer.title,'data-kvsp-overview',peer.id);
        }).join('')||'阶段边界';
        return `<h3>${escape(node.title)}</h3><p>${escape(node.detail)}</p>
          <div class="csa-qp-relation"><b>输入</b>${related('to')}</div>
          <div class="csa-qp-relation"><b>输出</b>${related('from')}</div>
          ${button('查看源码 · L'+node.lines.join('–'),'data-kvsp-source',node.file+':'+node.lines.join(':'))}
          ${sliceId!=null&&node.kernelLink===kernel?'<p>实测切片只能定位到所属 Kernel。选择逻辑 Block 探索静态计算。</p>':''}
          ${node.kernelLink?button('展开 Kernel','data-kvsp-block',block)+navActions('inter','kernel')+
            (scale==='instances'?blockGrid(ticks(countFor(kernel)),kernel):button('查看任务切分','data-kvsp-logical-list','')):''}`;
      }
      if (scope !== 'intra') {
        const records = kernelByName.get(kernel);
        const peers=side=>model.stageEdges.filter(edge=>edge[side]===kernel).map(edge=>button(edge[side==='from'?'to':'from'],'data-kvsp-kernel',edge[side==='from'?'to':'from'])).join('')||'阶段边界';
        return `<h3>${kernel}</h3><p>${records.coreType} · ${countFor(kernel)} Blocks</p>
          <div class="csa-qp-relation"><b>输入任务</b>${peers('to')}</div><div class="csa-qp-relation"><b>下游任务</b>${peers('from')}</div>
          ${button('选择逻辑 Block', 'data-kvsp-logical-list', '')}${button('查看源码','data-kvsp-source',records.first+':'+records.last)}
          ${navActions('inter','kernel')}`;
      }
      const info=kernelByName.get(kernel), shape=tile(block,kernel);
      const inspectIteration=mode==='assignment'&&model.isMatmul?.(info)&&selected?.step!=null;
      const graph=inspectIteration?operationModel(info,true,localView()):localGraph();
      const node=graph.nodes.find(n=>n.id===selected?.id)||graph.nodes.find(n=>n.op===selected?.op &&
        (selected.split==null||n.split===selected.split)&&(!inspectIteration&&scale==='structure'||selected.step==null||n.step===selected.step));
      const neighbors=direction=>node ? graph.edges.filter(e=>e[direction]===node.id).map(e=>graph.nodes.find(n=>n.id===e[direction==='to'?'from':'to'])) : [];
      const nav=nodes=>nodes.length ? nodes.map(peer=>button(peer.title+(peer.step==null?'':` · ${peer.step}`),'data-kvsp-op',peer.id)).join('') : 'Block 边界';
      const facts=[['Rank',rank],['逻辑归属',shape.text],['对象 ID',objectId(rank,info,node?.block??block,node?.id,localView())]];
      if(node) facts.push(...(node.nameRange?[['K 输入区间',node.nameRange]]:[]),['执行类别',node.lane],['范围',node.detail||'当前 Block'],...(node.condition?[['条件',node.condition]]:[]));
      const lines=node?.lines||[info.first,info.last];
      return `<h3>${node?.title||kernel} · Block ${block}</h3><dl class="csa-sv-dl">
        ${facts.map(([key,value])=>`<dt>${key}</dt><dd>${escape(value)}</dd>`).join('')}</dl>
        ${node?`<div class="csa-qp-relation"><b>输入</b>${nav(neighbors('to'))}</div><div class="csa-qp-relation"><b>输出</b>${nav(neighbors('from'))}</div>`:''}
        ${node?.kernelLink?button('查看 '+node.kernelLink,'data-kvsp-kernel-link',node.kernelLink):''}
        ${button('查看源码 · L'+lines.join('–'),'data-kvsp-source',lines.join(':'))}${navActions('intra','op')}`;
    }

    function render() {
      const useCustomPipeline=mode==='pipeline'&&!!pipelineFactory;
      const existingShell=host.querySelector(':scope > .csa-sv-shell');
      if(useCustomPipeline){
        if(existingShell)existingShell.hidden=true;
        if(!customPipelineHost){
          customPipelineHost=document.createElement('div');
          customPipelineHost.className='csa-cmp-pipeline-host';
          host.append(customPipelineHost);
          customPipelineCtrl=pipelineFactory.create(customPipelineHost,{
            model,
            rank,
            kernel,
            filter: kernelExplicitlyChosen ? kernel : 'all',
            onSource,
            onRankChange:r=>{rank=r;onRank(r);},
            onKernelChange:(kName,blk)=>{
              if(kernelByName.has(kName)){
                kernel=kName;
                pipelineSyncedKernel=kName;
                kernelExplicitlyChosen=true;
                if(branch!=='all'&&kernelByName.get(kernel).branch!==branch)branch='all';
              }
              if(blk!=null)block=blk;
            },
            onViewChange:patch=>{
              if(patch.tokens!=null)tokens=patch.tokens;
              if(patch.tileIndex!=null)tileIndex=patch.tileIndex;
              if(patch.branch!=null)branch=patch.branch;
            },
            onNavigateTab:(dest,ctx)=>{
              if(ctx?.kernel&&kernelByName.has(ctx.kernel)){
                kernel=ctx.kernel;
                pipelineSyncedKernel=ctx.kernel;
                kernelExplicitlyChosen=true;
                if(branch!=='all'&&kernelByName.get(kernel).branch!==branch)branch='all';
              }
              if(ctx?.block!=null)block=ctx.block;
              if(ctx?.rank!=null&&ctx.rank!==rank){rank=ctx.rank;onRank(rank);}
              onTab(dest,graphSelection());
            },
          });
        }
        customPipelineHost.hidden=false;
        const payload=pendingPipelineState||{rank,kernel,branch,block,tokens,tileIndex,syncFilter:kernelExplicitlyChosen&&kernel!==pipelineSyncedKernel};
        pendingPipelineState=null;
        pipelineSyncedKernel=payload.kernel||kernel;
        customPipelineCtrl.update(payload);
        return;
      }
      if(customPipelineHost)customPipelineHost.hidden=true;
      if(existingShell)existingShell.hidden=false;
      const nextKey=keyFor(), changed=lastKey!==nextKey;
      if(changed){remember();zoom=cameras.get(nextKey)?.zoom??1;}
      const inspector = inspectorOpen ? detail() : '';
      if(mode==='graph')return onTab('graph');
      const screen = ({assignment,pipeline})[mode]();
      const canZoom=mode!=='assignment'||scope==='inter'||!!(renderTiles&&model.isMatmul(info()));
      const content = mode === 'graph' || mode === 'pipeline'||scope==='intra'&&!!renderTiles
        ? `<div class="csa-compressor-zoom-content" style="--compressor-zoom:${zoom}">${screen}</div>` : screen;
      const markup = `<div class="csa-sv-shell"><header class="csa-sv-header"><div class="csa-sv-title-group">
        <div class="csa-sv-title-row"><h2>${title}</h2></div><p class="csa-sv-sub">${escape(model.subtitle||(model.cases?'Rank-local Q':'Main + Indexer'))} · ${stage.kernels.length} Kernels · ${Object.values(current().kernels).flat().length} 实测切片 / Rank</p></div>
        <div class="csa-sv-controls">
        <div class="segmented-control segmented-control-muted" role="group" aria-label="Rank">${[0,1].map(r => button(`Rank ${r}`, 'data-kvsp-rank', r, r === rank)).join('')}</div>
        <div class="segmented-control segmented-control-muted" role="group" aria-label="核间核内">${button('核间', 'data-kvsp-scope', 'inter', scope === 'inter')}${button('核内', 'data-kvsp-scope', 'intra', scope === 'intra')}</div>
        <label>分支 <select data-kvsp-branch>
          <option value="all" ${branch==='all'?'selected':''}>全部</option>
          ${Object.entries(layoutBranches).map(([id,name])=>`<option value="${id}" ${branch===id?'selected':''}>${name}</option>`).join('')}</select></label>
        <label>Kernel <select data-kvsp-kernel-picker>${stage.kernels.filter(k=>branch==='all'||kernelByName.get(k.name).branch===branch).map(k=>`<option ${k.name===kernel?'selected':''}>${k.name}</option>`).join('')}</select></label>
        ${caseOptions.length&&(mode!=='pipeline'||scope==='intra')?`<label>${escape(model.caseLabel||'Q 行数')} <select data-kvsp-case>
          ${caseOptions.map(c=>`<option value="${c.tokens}" ${c.tokens===tokens?'selected':''}>${c.label}</option>`).join('')}</select></label>`:''}
        ${scope === 'intra' ? `<label>Block <select data-kvsp-block-picker>${ticks(blocks()).map(i => `<option ${i === block ? 'selected' : ''}>${i}</option>`).join('')}</select></label>` : ''}
        ${scope==='intra'&&model.cases&&tile(block,kernel).tiles.length>1?`<label>M Tile <select data-kvsp-tile-picker>
          ${tile(block,kernel).tiles.map((m,i)=>`<option value="${i}" ${i===tileIndex?'selected':''}>${m.path} ${range(m.start,m.valid)}</option>`).join('')}</select></label>`:''}
        ${mode==='graph'&&scope==='intra'&&scale==='instances'&&model.cases&&tile(block,kernel).tiles.length>1
          ? phaseButton('全部 M Tiles','data-kvsp-all-tiles','',allTiles):''}
        ${(mode==='graph'||mode==='pipeline'&&scope==='intra') ? `<div class="segmented-control segmented-control-muted" role="group" aria-label="核内观察尺度">${button('Structure', 'data-kvsp-scale', 'structure', scale === 'structure')}${button('Instances', 'data-kvsp-scale', 'instances', scale === 'instances')}</div>` : ''}
        ${navigation.length?button('← 返回原位置','data-kvsp-back',''):''}
        ${button('返回 CSA', 'data-kvsp-exit', '')}</div></header>
        <div class="csa-compressor-canvas"><div class="csa-sv-main csa-kvsp-main" tabindex="0" aria-label="${title} 画布">${content}</div>
        <footer class="csa-compressor-footer" ${canZoom?'':'hidden'}><span>${mode==='pipeline'&&scope==='inter'?'实测 µs · AIC / AIV':mode==='assignment'?'逻辑任务 / 实测分配':'静态计算'}</span>
          <div class="viz2-instance-tools">${button('−','data-kvsp-zoom','out')}<output>${Math.round(zoom*100)}%</output>${button('＋','data-kvsp-zoom','in')}${button('适应','data-kvsp-zoom','fit')}</div></footer></div>
        <aside class="pto-ide-frame__pane csa-sv-inspector" ${inspector ? '' : 'hidden'}>
          <header class="pto-ide-frame__pane-header"><b>对象详情</b><button type="button" class="btn btn-ghost btn-sm" data-kvsp-close aria-label="关闭详情">×</button></header>
          <div class="pto-ide-frame__pane-body">${inspector}</div></aside></div>`;
      const template=document.createElement('template');
      template.innerHTML=markup;
      const next=template.content.firstElementChild, mounted=host.querySelector(':scope > .csa-sv-shell');
      if(!mounted)host.append(next);
      else {
        const focus=document.activeElement;
        const container=viewport(), newContainer=next.querySelector('.csa-kvsp-main');
        const morph=mode==='assignment'&&scope==='inter'&&container.querySelector('[data-kvsp-logical-layer]');
        const body=mounted.querySelector('aside .pto-ide-frame__pane-body'), top=body.scrollTop;
        if(changed&&!morph)container.replaceChildren(...newContainer.childNodes);
        else syncElement(container,newContainer);
        syncElement(mounted.querySelector('header'),next.querySelector('header'));
        syncElement(mounted.querySelector('footer'),next.querySelector('footer'));
        syncElement(mounted.querySelector('aside'),next.querySelector('aside'));
        body.scrollTop=top;
        if(!focus?.isConnected&&savedFocus)host.querySelector(savedFocus)?.focus({preventScroll:true});
      }
      lastKey=nextKey;
      updateZoom();
      if(changed) {
        const camera=cameras.get(lastKey);
        viewport().scrollTo(camera?.left??0,camera?.top??0);
        if(!camera&&mode!=='assignment')zoomBy('fit');
        if(!reduceMotion&&!viewport().querySelector('[data-kvsp-logical-layer]')) {
          viewport().firstElementChild?.animate([{opacity:.5,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],
            {duration:240,easing:'cubic-bezier(.2,.7,.2,1)'});
        }
      }
      if(focusSelector){frame(focusSelector);focusSelector=null;}
    }

    const selector=(type,value)=>'[data-kvsp-'+type+'="'+CSS.escape(String(value))+'"]';

    function inspectSlice(id) {
      const owner=Object.keys(current().kernels).find(name=>current().kernels[name].some(slice=>slice[0]===Number(id)));
      setKernel(owner);
      sliceId=Number(id);inspectorOpen=true;
      focusObject(selector('slice',sliceId));render();
    }

    function setPhase(next) {
      phase=next;
      if(phase!=='assignment'&&blockSelected)focusObject(selector('logical',block));
      render();
    }

    function focusOperation(targetMode=mode) {
      if(!selected?.op)return;
      const graph=localGraph();
      const node=graph.nodes.find(n=>n.id===selected.id)||graph.nodes.find(n=>n.op===selected.op&&
        (selected.split==null||selected.split===n.split)&&(scale==='structure'||selected.step==null||n.step===selected.step));
      if(node)focusObject(selector('op',node.id)+(targetMode==='graph'&&scale==='instances'&&allTiles?selector('m',tileIndex):''));
    }

    function zoomBy(direction) {
      const region=viewport(), scene=region.querySelector('.csa-kvsp-graph,.csa-kvsp-timeline,.csa-compressor-assignment,.csa-qproj-matrices');
      if(!scene)return;
      const old=zoom, anchor=[region.scrollLeft+region.clientWidth/2,region.scrollTop+region.clientHeight/2];
      const rect=scene.getBoundingClientRect();
      const topOffset=scene.matches('.csa-kvsp-timeline')
        ? rect.top-region.getBoundingClientRect().top+region.scrollTop+22 : 70;
      zoom=direction==='fit'?Math.max(.15,Math.min(1,(region.clientWidth-48)*old/rect.width,(region.clientHeight-topOffset)*old/rect.height))
        :Math.max(.15,Math.min(2.5,old*(direction==='in'?1.25:.8)));
      updateZoom();
      region.scrollTo(...(direction==='fit'?[0,0]:[anchor[0]*zoom/old-region.clientWidth/2,anchor[1]*zoom/old-region.clientHeight/2]));
      remember();
    }

    const interactive='[data-kvsp-slice],[data-kvsp-logical],[data-kvsp-core],[data-kvsp-tile],[data-kvsp-k]';
    host.addEventListener('keydown',event=>{
      if(event.target.matches(interactive)&&event.target instanceof SVGElement&&['Enter',' '].includes(event.key)) {
        event.preventDefault();
        event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));
      }
    });

    host.addEventListener('click',event=>{
      if(event.target.closest('.csa-cmp-pipeline-host'))return;
      const action=event.target.closest('button,'+interactive);
      if(!action)return;
      const d=action.dataset, attr=[...action.attributes].find(item=>item.name.startsWith('data-kvsp-'));
      savedFocus=attr?'['+attr.name+'="'+CSS.escape(attr.value)+'"]':null;
      if(d.kvspExit!=null){stop();return onExit();}
      if(d.kvspBack!=null){
        const previous=navigation.pop();
        if(!previous)return;
        ({scope,scale,block,kernel,sliceId,selected,blockSelected,inspectorOpen,phase,coreId,filter,branch,tokens,tileIndex,selectedK,overviewId,allTiles}=previous);
        if(previous.mode!==mode)return onTab(previous.mode);
        return render();
      }
      if(d.kvspNav){
        const [destination,next,kind]=d.kvspNav.split(':');
        navigation.push(snapshot());stop();scope=next;
        if(kind==='slice'){
          if(destination==='assignment')phase='assignment';
          if(destination==='pipeline'){filter=kernel;pendingPipelineState={rank,kernel,level:'lane',overviewMode:'runtime',filter:kernel,sliceId};}
          if(destination==='graph'&&model.overview){
            overviewId=kernel;focusObject(selector('overview',overviewId));
          }else if(destination!=='graph')focusObject(selector('slice',sliceId));
        } else if(destination==='pipeline'){
          pendingPipelineState=next==='intra'
            ? {rank,kernel,block,level:'block',intraScale:scale==='structure'?'structure':'instances'}
            : {rank,kernel,block,level:'lane',overviewMode:'lane'};
        }
        if(kind==='core'&&destination==='assignment')phase='assignment';
        if(scope==='intra'){if(mode==='assignment'&&selected?.step!=null)scale='instances';focusOperation();}
        return onTab(destination,graphSelection());
      }
      if(d.kvspLogicalList!=null){
        navigation.push(snapshot());stop();
        scope='inter';phase='generation';sliceId=null;coreId=null;inspectorOpen=blockSelected;
        if(mode==='assignment')return render();
        return onTab('assignment');
      }
      if(d.kvspClose!=null){
        inspectorOpen=false;selected=null;sliceId=null;coreId=null;blockSelected=false;return render();
      }
      if(d.kvspSource){
        const ref=d.kvspSource.split(':');
        const file=ref.length===3?ref.shift():info().file;
        return onSource(file,...ref.map(Number));
      }
      if(d.kvspOverview){
        const node=model.overview(tokens).nodes.find(i=>i.id===d.kvspOverview);
        if(node.kernelLink)setKernel(node.kernelLink);
        overviewId=node.id;inspectorOpen=true;focusObject(selector('overview',overviewId));return render();
      }
      if(d.kvspKernelLink){
        navigation.push(snapshot());setKernel(d.kvspKernelLink);scope='inter';inspectorOpen=true;
        overviewId=kernel;return onTab('graph');
      }
      if(d.kvspTile!=null||d.kvspK!=null){
        if(d.kvspTile!=null){tileIndex=Number(d.kvspTile);selected=null;}
        if(d.kvspK!=null)selectedK=Number(d.kvspK);
        if(model.isMatmul(info())){
          const shape=tile(block,kernel);
          selected={op:'matmul',step:selectedK,split:shape.split??0};
        }
        blockSelected=true;inspectorOpen=true;return render();
      }
      if(d.kvspPhase){stop();return setPhase(d.kvspPhase);}
      if(d.kvspPlay!=null){if(timer==null){if(phase==='assignment')phase='generation';play();}else stop();return render();}
      if(d.kvspReplay!=null){phase='generation';play();return render();}
      if(d.kvspLoop!=null){loop=!loop;return render();}
      if(d.kvspZoom)return zoomBy(d.kvspZoom);
      if(d.kvspLogical!=null){
        block=Number(d.kvspLogical);if(model.cases&&selected?.split!=null)selected={...selected,split:tile(block,kernel).split??0};blockSelected=true;sliceId=null;coreId=null;inspectorOpen=true;
        focusObject(selector('logical',block));return render();
      }
      if(d.kvspSlice)return inspectSlice(d.kvspSlice);
      if(d.kvspCore!=null){
        coreId=Number(d.kvspCore);sliceId=null;inspectorOpen=true;
        focusObject(selector('core',coreId));return render();
      }
      if(d.kvspAllCores!=null){coreId=null;return render();}
      if(d.kvspRank!=null&&rank!==Number(d.kvspRank)){
        rank=Number(d.kvspRank);sliceId=null;coreId=null;navigation=[];onRank(rank);
      }
      if(d.kvspScope&&scope!==d.kvspScope){
        navigation.push(snapshot());scope=d.kvspScope;stop();inspectorOpen=scope==='intra'&&(blockSelected||!!model.overview);
      }
      if(d.kvspKernel){setKernel(d.kvspKernel);overviewId=null;inspectorOpen=true;focusObject(selector('kernel',kernel));}
      if(d.kvspAllTiles!=null){allTiles=!allTiles;selected=null;}
      if(d.kvspScale){scale=d.kvspScale;focusOperation();}
      if(d.kvspBlock!=null){
        if(scope!=='intra')navigation.push(snapshot());
        if(d.kvspBlockKernel)setKernel(d.kvspBlockKernel);
        block=Number(d.kvspBlock);scope='intra';stop();blockSelected=true;inspectorOpen=true;
      }
      if(d.kvspOp){
        const graph=mode==='assignment'&&model.isMatmul?.(info())?operationModel(info(),true,localView()):localGraph();
        const node=graph.nodes.find(n=>n.id===d.kvspOp);
        if(!node)return;
        if(d.kvspM!=null)tileIndex=Number(d.kvspM);
        if(node.block!=null)block=node.block;
        if(node.step!=null&&model.isMatmul?.(info()))selectedK=node.step;
        inspectorOpen=true;
        const previousStep=selected?.op===node.op&&selected.split===node.split?selected.step:null;
        selected={op:node.op,step:node.step??previousStep,split:node.split,id:node.step==null&&previousStep!=null?null:node.id};
        focusObject(selector('op',node.id)+(d.kvspM==null?'':selector('m',d.kvspM)));
      }
      render();
    });
    host.addEventListener('wheel',event=>{
      if(!(event.metaKey||event.ctrlKey)||!event.target.closest('.csa-kvsp-main'))return;
      event.preventDefault();
      zoomBy(event.deltaY<0?'in':'out');
    },{passive:false});
    let drag = null;
    host.addEventListener('pointerdown', event => {
      const viewport=event.target.closest('.csa-kvsp-main');
      if (!viewport || event.button!==0 || event.target.closest('button, summary, select, table, [role="button"]')
        || !viewport.querySelector('svg')) return;
      drag={viewport,id:event.pointerId,x:event.clientX,y:event.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add('is-panning');
    });
    host.addEventListener('pointermove', event => {
      if (drag?.id!==event.pointerId) return;
      drag.viewport.scrollTo(drag.left+drag.x-event.clientX,drag.top+drag.y-event.clientY);
    });
    host.addEventListener('lostpointercapture', () => {
      drag?.viewport.classList.remove('is-panning');
      drag=null;
    });
    host.addEventListener('change', event => {
      if(event.target.closest('.csa-cmp-pipeline-host'))return;
      if(event.target.matches('[data-kvsp-case]')){
        tokens=Number(event.target.value);block=Math.min(block,blocks()-1);
        tileIndex=Math.min(tileIndex,tile(block,kernel).tiles.length-1);
        selected=null;selectedK=0;
      } else if(event.target.matches('[data-kvsp-tile-picker]')){
        tileIndex=Number(event.target.value);allTiles=false;
        focusOperation();
      } else if(event.target.matches('[data-kvsp-branch]')){
        branch=event.target.value;
        if(branch!=='all'&&kernelByName.get(kernel).branch!==branch)setKernel([...kernelByName.values()].find(k=>k.branch===branch).name);
        if(mode==='pipeline'&&scope==='inter')filter=branch;
      } else if(event.target.matches('[data-kvsp-filter]')){
        filter=event.target.value;sliceId=null;coreId=null;
        if(kernelByName.has(filter))setKernel(filter);
      } else if (event.target.matches('[data-kvsp-kernel-picker]')) {
        setKernel(event.target.value);
        if(mode==='pipeline'&&scope==='inter')filter=kernel;
      } else if(event.target.matches('[data-kvsp-block-picker]')){
        block=Number(event.target.value);
        if(model.cases&&selected?.split!=null)selected={...selected,id:null,split:tile(block,kernel).split??0};
      }
      else return;
      render();
    });
    function graphSelection(){
      const operation=selected?.id||(selected?.op?(selected.step!=null?selected.op+':'+selected.step:selected.op):null);
      return {kernel,operation,rank,block,scale,explicit:kernelExplicitlyChosen,key:JSON.stringify([kernel,operation,rank,block,scale])};
    }
    return {getSelection:graphSelection,setGraphSelection(context){
      if(!kernelByName.has(context.kernel))return;
      kernel=context.kernel;branch='all';kernelExplicitlyChosen=true;
      if(context.operation){selected={op:context.operation,step:context.iteration??null,split:context.split??null};scope='intra';scale=context.iteration==null?'structure':'instances';inspectorOpen=true;pendingPipelineState={rank,kernel,block,level:'block',intraScale:scale==='structure'?'structure':'instances',filter:kernel};}
      else {selected=null;scope='inter';pendingPipelineState={rank,kernel,block,level:'lane',overviewMode:'lane',filter:kernel};}
    }, update(nextMode, nextRank) {
      if (rank !== nextRank) sliceId = null;
      if(mode!==nextMode){stop();if(scope==='intra'){if(mode==='assignment'&&selected?.step!=null)scale='instances';focusOperation(nextMode);}}
      mode = nextMode;
      rank = nextRank;
      render();
    }, pause(){stop();remember();} };
  }

  global.PtoCsaKernelStage=Object.freeze({mount});
  global.PtoCsaCompressor=global.PtoCsaKernelStage;
})(window);
