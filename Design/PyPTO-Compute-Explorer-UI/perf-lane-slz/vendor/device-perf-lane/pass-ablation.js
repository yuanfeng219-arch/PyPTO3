(() => {
 const dialog=document.getElementById('pass-dialog');
 if(!dialog)return;
 const experiment=document.documentElement.dataset.passUi!=='integrated';
 // Classify the object represented in this snapshot, not its cross-pass selection ID.
 const objects=window.PERF_PASS_OBJECTS;
 const originalRender=objects.render.bind(objects);
 function kind(node,pane){
  if(['qk','rope'].includes(node.id))return 'task';
  if(node.id==='blocks')return 'structural'; // execution fan-out, not another task
  if(node.id==='publish')return /kernel|wrapper|submit/.test(node.sub)?'kernel':'structural';
  if(['loop','stage1'].includes(node.id)||/阶段 [01] ·/.test(node.title))return 'structural';
  if(['put0','put1','notify','fence0','fence1'].includes(node.id))return 'communication';
  if(/mem_vec_|结果$|输入分片|输出分片|scale 输出|位于 Mat/.test(node.title+' '+node.sub)||node.id==='input')return 'tensor';
  return 'op';
 }
 objects.render=(scene,selected)=>{
  const view={...scene};
  for(const side of ['before','after'])view[side]={...scene[side],edges:scene[side].edges.map(edge=>edge.to==='blocks'?{...edge,label:'执行规模'}:edge)};
  const host=document.createElement('div');host.innerHTML=originalRender(view,selected);
  host.querySelectorAll('.object-pane').forEach((pane,i)=>{
   const data=i?scene.after:scene.before;
   pane.querySelectorAll('.pass-object').forEach((node,j)=>{node.dataset.objectKind=kind(data.nodes[j],data);});
  });
  return host.innerHTML;
 };
 function glyph(kind,x,y,id=''){
  const icons=window.PASS_LEGEND_ICONS||{};
  if(kind==='tensor')return [0,1,2,3].map(i=>`<rect x="${x+8}" y="${y+7+i*10}" width="8" height="8" fill="#B0B0B0" opacity="${1-i*.2}"/>`).join('');
  if(kind==='task')return `<rect x="${x+8}" y="${y+10}" width="34" height="34" rx="3" fill="#080b05"/><path transform="translate(${x+7} ${y+9})" d="M9 10 H27 M9 18 H20 M9 26 H16 M23 22 L28 26 L23 30" stroke="#9EF04B" stroke-width="2.6" fill="none"/>`;
  if(kind==='op'||kind==='communication'){
   const key=kind==='communication'?'AllGather':id==='mul'?'MatMul':id==='weighted'?'Gate / Multiply':id==='scale'?'Reduce / Pool':null;
   return `<circle cx="${x+27}" cy="${y+27}" r="18.5" fill="#101010"/>`+(key?`<path transform="translate(${x+15} ${y+15})" d="${icons[key]}" stroke="#eee" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`:`<text x="${x+27}" y="${y+31}" text-anchor="middle" fill="#eee" style="font-size:11px">IR</text>`);
  }
  return '';
 }
 function legendGlyph(kind){return `<svg viewBox="0 0 100 54" aria-hidden="true"><rect class="legend-body" width="100" height="54"/>${glyph(kind,0,0,kind==='op'?'mul':'')}</svg>`;}
 let structureExpanded=true;
 function compileStructure(stage){
  const late=['P10','P40'].includes(stage);
  const title={P00:'前端：计算体内联在 SPMD scope 中',P09:'P09：外提计算体，scope 内保留 kernel 调用',P10:'P10：scope 降为提交，目标为 SPMD wrapper',P40:'P40：提交结构与两个显式前驱均保留'}[stage];
  const box=(x,y,w,h,label,sub,style='scope',id='publish')=>`<g role="button" tabindex="0" data-pass-object="${id}" aria-label="${label} · ${sub}" class="compile-node ${style}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${style==='op'?24:4}"/><text x="${x+16}" y="${y+25}">${label}</text><text class="compile-sub" x="${x+16}" y="${y+47}">${sub}</text></g>`;
  const edge=(d,label,x,y,call=false)=>`<path class="compile-edge ${call?'call':''}" d="${d}" marker-end="url(#compile-arrow)"/><text class="compile-edge-label" x="${x}" y="${y}">${label}</text>`;
  let body=box(30,20,220,62,'qk_pv','qk_tid','task','qk')+box(300,20,220,62,'rope_cs','attn_rope_tid','task','rope');
  body+=edge('M140 82 V110 H275 V140','deps',150,102)+edge('M410 82 V110 H275 V140','deps',420,102);
  if(!late){
   body+=box(30,145,490,structureExpanded?200:62,'pl.spmd(48)','publish_tid · deps=[qk_tid, attn_rope_tid]');
   if(structureExpanded){
    if(stage==='P00')body+=box(60,230,430,66,'内联计算体','csa_merge_pack_publish','op');
    else {body+=box(60,230,430,66,'self.csa_merge_pack_publish(...)','外提后的 kernel 调用','op');body+=edge('M490 263 H565 V205 H610','调用',547,193,true);body+=box(615,145,350,150,'csa_merge_pack_publish','外提 kernel · AIV','kernel');}
   }
  }else{
   body+=box(30,145,490,70,'pl.spmd_submit(...)','core_num=48 · 返回 publish_tid','task');
   if(structureExpanded){
    body+=edge('M520 180 H610','调用',550,167,true);
    body+=box(615,125,350,220,'csa_merge_pack_publish_spmd','SPMD wrapper','kernel');
    body+=box(640,205,300,66,'csa_merge_pack_publish','kernel 调用','op');
    body+=`<text class="compile-sub" x="640" y="310">提交规模：48 个 block</text>`;
   }
   if(stage==='P40')body+='<text class="compile-preserved" x="30" y="260">✓ 显式依赖 2 → 2 · 无新增依赖边</text>';
  }
  return `<div class="compile-heading"><strong>${title}</strong><button type="button" class="ctl" data-structure-toggle aria-expanded="${structureExpanded}">${structureExpanded?'收起结构':'展开结构'}</button></div><svg class="compile-structure" viewBox="0 0 1000 370" role="group" aria-label="${stage} 编译结构"><defs><marker id="compile-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1 L9 5 L1 9" fill="none" stroke="#aaa"/></marker></defs>${body}</svg>`;
 }
 dialog.addEventListener('click',e=>{
  if(!e.target.closest('[data-structure-toggle]'))return;
  structureExpanded=!structureExpanded;
  const stage=dialog.querySelector('[data-pass-snapshot][aria-pressed="true"]').dataset.passSnapshot;
  dialog.querySelector('.dependency-stable').innerHTML=compileStructure(stage);
 });
 const toggle=document.createElement('div');toggle.className='ablation-toggle';toggle.setAttribute('aria-label','消融实验对比');
 toggle.innerHTML='<button data-ablation="original" aria-pressed="false">原版</button><button data-ablation="reduced" aria-pressed="true">消融版</button>';
 if(experiment)dialog.querySelector('.pass-close').before(toggle);
 dialog.classList.add('pass-reduced');
 toggle.onclick=e=>{const b=e.target.closest('[data-ablation]');if(!b)return;dialog.classList.toggle('pass-reduced',b.dataset.ablation==='reduced');toggle.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));enhance();};
 function enhance(){
  const runtime=dialog.querySelector('.pass-runtime');
  if(runtime.children.length&&!runtime.querySelector('.runtime-expand')){
   const b=document.createElement('button');b.className='runtime-expand';b.textContent='更多指标';b.setAttribute('aria-expanded','false');b.onclick=()=>{const on=runtime.classList.toggle('expanded');b.textContent=on?'收起指标':'更多指标';b.setAttribute('aria-expanded',String(on));};runtime.append(b);
  }
  const stages=dialog.querySelector('.object-snapshots');
  const evidence=dialog.querySelector('.pass-evidence');
  evidence.classList.toggle('dependency-evolution',!!stages);
  if(stages&&!evidence.querySelector('.dependency-stable')){
   const panes=evidence.querySelector('.object-panes');
   if(panes){
    const stable=document.createElement('section');stable.className='dependency-stable';
    stable.innerHTML='<p class="dependency-conclusion">两个显式前驱由前端指定，贯穿 P00–P40 保留</p>';
    const stage=stages.querySelector('[aria-pressed="true"]').dataset.passSnapshot;
    stable.innerHTML=compileStructure(stage);
    stages.before(stable);
    stages.classList.add('evolution-stages');
    const labels={P00:['前端定义','内联 scope'],P09:['外提 kernel','计算体成为核函数'],P10:['形成提交','SPMD wrapper · 48 blocks'],P40:['保留依赖','已有两个前驱不变']};
    stages.querySelectorAll('button').forEach(b=>{b.dataset.originalLabel=b.textContent;const [title,detail]=labels[b.dataset.passSnapshot];b.innerHTML='<span class="stage-original">'+b.dataset.originalLabel+'</span><span class="stage-evolution"><small>'+b.dataset.passSnapshot+'</small><strong>'+title+'</strong><span>'+detail+'</span></span>';});
   }
  }
  dialog.querySelectorAll('.object-pane svg').forEach(svg=>{
   if(!svg.dataset.originalBox)svg.dataset.originalBox=svg.getAttribute('viewBox');
   const box=svg.closest('.dependency-stable')?'0 0 440 235':dialog.classList.contains('pass-reduced')?'0 0 560 365':svg.dataset.originalBox;
   if(svg.getAttribute('viewBox')!==box)svg.setAttribute('viewBox',box);
  });
  dialog.querySelectorAll('.object-edge').forEach(path=>{
   if(!path.dataset.originalD)path.dataset.originalD=path.getAttribute('d');
   const points=path.dataset.originalD.match(/-?\d+(?:\.\d+)?/g).map(Number);
   const [x,y,,mid,,,endX,endY]=points;
   path.setAttribute('d',dialog.classList.contains('pass-reduced')?`M${x} ${y} V${mid} H${endX} V${endY}`:path.dataset.originalD);
  });
  dialog.querySelectorAll('.pass-object:not([data-ablation-ready])').forEach(node=>{
   node.dataset.ablationReady='true';const id=node.dataset.passObject;
   node.classList.add('csa-'+(node.dataset.objectKind||'structural'));
   const rect=node.querySelector('rect');
   const deco=document.createElementNS('http://www.w3.org/2000/svg','g');deco.classList.add('library-glyph');deco.innerHTML=glyph(node.dataset.objectKind,Number(rect.getAttribute('x')),Number(rect.getAttribute('y')),id);node.append(deco);
   const text=node.querySelector('text');if(text){text.dataset.originalY=text.getAttribute('y');}
  });
  dialog.querySelectorAll('.pass-object>text:first-of-type').forEach(t=>{if(!t.dataset.originalY)return;const y=String(Number(t.dataset.originalY)+(dialog.classList.contains('pass-reduced')?10:0));if(t.getAttribute('y')!==y)t.setAttribute('y',y);});
  dialog.querySelectorAll('.object-legend').forEach(legend=>{if(legend.nextElementSibling?.classList.contains('ablation-legend')||legend.closest('.pass-evidence')?.querySelector('.compile-heading .ablation-legend'))return;const details=document.createElement('details');details.className='ablation-legend';details.innerHTML='<summary>图例</summary><div class=type-legend>'+[['tensor','Tensor / Tile 值'],['op','Op 操作'],['communication','通信 / 同步'],['task','前驱任务'],['kernel','Kernel / 提交'],['structural','控制 / 执行结构']].map(([type,label])=>'<span class=legend-'+type+'>'+legendGlyph(type)+label+'</span>').join('')+'</div><div>'+legend.innerHTML+'</div>';legend.after(details);});
 }
 function compactDependencyEvidence(){
  const evidence=dialog.querySelector('.dependency-evolution');
  if(!evidence)return;
  const legend=evidence.querySelector('.ablation-legend');
  const heading=evidence.querySelector('.compile-heading');
  if(legend&&heading&&legend.parentElement!==heading)heading.append(legend);
  const summary=evidence.querySelector('.object-inspector .pass-disclosure>summary');
  if(summary&&summary.textContent!=='查看关联代码')summary.textContent='查看关联代码';
 }
 const coreTip=document.createElement('div');coreTip.className='core-hover-panel';coreTip.setAttribute('role','tooltip');coreTip.id='core-hover-panel';coreTip.hidden=true;dialog.append(coreTip);
 function showCore(row){
  if(!dialog.classList.contains('pass-reduced'))return;
  const rank=dialog.querySelector('.pass-head p')?.textContent.match(/rank\d+/)?.[0];
  const runtime=window.PERF_PASS_ANALYSIS?.cases?.csa_merge_pack_publish?.setupAnalysis?.ranks?.[rank];
  const b=runtime?.blocks.find(b=>b.core===Number(row.dataset.setupCore));if(!b)return;
  const us=n=>n.toFixed(2)+' μs';
  coreTip.innerHTML='<strong>Core '+b.core+'</strong><dl>'+[['本核 qk_pv 结束',us(b.previousEnd)],['领取任务',us(b.receive)],['kernel 开始',us(b.start)],['Setup',us(b.setup)],['kernel 执行',us(b.kernel)],['记录','aicore_tasks['+b.recordIndex+']']].map(([k,v])=>'<dt>'+k+'</dt><dd>'+v+'</dd>').join('')+'</dl>';
  coreTip.hidden=false;row.setAttribute('aria-describedby',coreTip.id);
  const r=row.getBoundingClientRect(),d=dialog.getBoundingClientRect();
  coreTip.style.left=Math.max(12,Math.min(r.right-d.left-360,d.width-380))+'px';
  coreTip.style.top=Math.max(12,Math.min(r.bottom-d.top+8,d.height-coreTip.offsetHeight-12))+'px';
 }
 dialog.addEventListener('pointerover',e=>{const row=e.target.closest('.setup-row');if(row)showCore(row);});
 dialog.addEventListener('focusin',e=>{const row=e.target.closest('.setup-row');if(row)showCore(row);});
 const hideCore=()=>{coreTip.hidden=true;dialog.querySelectorAll('[aria-describedby="core-hover-panel"]').forEach(n=>n.removeAttribute('aria-describedby'));};
 dialog.addEventListener('pointerout',e=>{const row=e.target.closest('.setup-row');if(row&&!row.contains(e.relatedTarget))hideCore();});
 dialog.addEventListener('focusout',e=>{if(e.target.closest('.setup-row'))hideCore();});
 dialog.addEventListener('close',hideCore);
 function compactCoreEvidence(){
  dialog.querySelectorAll('.setup-row[title]').forEach(row=>{row.dataset.originalTitle=row.title;row.removeAttribute('title');});
  const chart=dialog.querySelector('.setup-chart');
  if(chart&&chart.nextElementSibling?.tagName==='P'&&!chart.querySelector('.timing-info')){
   const note=chart.nextElementSibling;note.classList.add('timing-note');
   const info=document.createElement('details');info.className='timing-info';info.innerHTML='<summary aria-label="参考时间说明">ⓘ</summary><div>'+note.textContent+'</div>';chart.querySelector('.setup-legend').append(info);
  }
 }
 const observer=new MutationObserver(()=>{observer.disconnect();enhance();compactDependencyEvidence();compactCoreEvidence();observer.observe(dialog,{childList:true,subtree:true});});
 observer.observe(dialog,{childList:true,subtree:true});enhance();
 // Start the experiment at the existing runtime evidence entry, without changing its handlers.
 if(experiment)document.querySelector('[data-pass-analysis]')?.click();
})();
