/* Temporary relation preview; selection and Inspector remain owned by app.js. */
window.ExploreDesignHover=(()=>{
 const relations=window.EXPLORE_DESIGN_RELATIONS,types={informs:'Informs · 设计依据',constrains:'Constrains · 工程约束',evaluated_not_triggered:'Evaluated · 未触发'};
 let clear=()=>{};
 const getRelations=id=>relations.filter(r=>r.sourceId===id||r.targetId===id);
 const name=id=>window.LINEAGE_VIEW.groups.flatMap(g=>g.items).find(i=>i[0]===id)?.[1]||id;
 function explain(host,rs,links){for(const r of rs){const row=document.createElement('p'),title=document.createElement('strong');title.textContent=name(r.sourceId)+' → '+name(r.targetId)+' · '+types[r.relationType];row.append(title,document.createElement('br'),document.createTextNode(r.reason));const sources=document.createElement('small');for(const ref of r.evidenceRefs){const a=document.createElement(links?'a':'span');a.textContent=ref.sourceId+(links?' · '+ref.location:'');if(links){a.href='evidence/'+ref.sourceId;a.target='_blank';a.rel='noopener'}sources.append(a,document.createTextNode(' '))}row.append(sources);host.append(row)}}
 function inspector(id){const rs=getRelations(id);if(!rs.length)return;const details=document.createElement('details'),summary=document.createElement('summary');details.className='cross-stage-evidence';summary.textContent='Explore ↔ Design · 关联依据与证据（'+rs.length+'）';details.append(summary);explain(details,rs,true);document.getElementById('detail').append(details)}
 function attach(svg,view,mode){clear();if(mode==='optimize')return;const scene=svg.querySelector('g[clip-path]');if(!scene)return;const NS='http://www.w3.org/2000/svg',layer=document.createElementNS(NS,'g');layer.classList.add('cross-stage-overlay');const firstCard=scene.querySelector('[data-group]');scene.insertBefore(layer,firstCard);
 const items=[...svg.querySelectorAll('[data-group="constraints"] [data-node-id],[data-group="design"] [data-node-id]')];
 const anchor=(id,side)=>{const g=view.groups.find(g=>g.items.some(i=>i[0]===id)),i=g.items.findIndex(i=>i[0]===id),box=FlowGeometry.itemBox(g,i);return {x:side==='right'?g.x+g.w:g.x,y:box.y-4}};
 function reset(){layer.replaceChildren();for(const item of items){item.classList.remove('relation-origin','relation-linked','relation-muted')}}
 function show(id){const rs=getRelations(id);reset();if(!rs.length)return;const related=new Set(rs.flatMap(r=>[r.sourceId,r.targetId]));for(const item of items){const other=item.getAttribute('data-node-id');item.classList.add(other===id?'relation-origin':related.has(other)?'relation-linked':'relation-muted')}
 const pairs=new Map();for(const r of rs){const key=r.sourceId+' '+r.targetId;if(!pairs.has(key))pairs.set(key,[]);pairs.get(key).push(r)}for(const [key,pair]of pairs){const [src,tgt]=key.split(' '),path=document.createElementNS(NS,'path');path.setAttribute('d',FlowGeometry.curve(anchor(src,'right'),anchor(tgt,'left')));path.setAttribute('class','cross-stage-link'+(pair.every(r=>r.applicabilityStatus==='not_triggered')?' not-triggered':''));path.setAttribute('data-relation-ids',pair.map(r=>r.id).join(' '));layer.append(path)}}
 for(const item of items){const id=item.getAttribute('data-node-id');item.addEventListener('pointerenter',()=>show(id));item.addEventListener('pointerleave',reset);item.addEventListener('focus',()=>show(id));item.addEventListener('blur',reset);item.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();reset()}})}
 clear=()=>{reset();layer.remove()};
 }
 return {getRelations,attach,inspector,clear:()=>clear()};
})();
