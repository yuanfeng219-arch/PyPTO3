/* Opt-in presentation adapter; semantic IDs, ports and edges remain unchanged. */
(function(root){
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function apply(scene,graph,{metadata=false,layers={}}={}){
 scene.classList.add('legend-skin');
 for(const node of graph.nodes){
  const el=[...scene.querySelectorAll('[data-pypa-node]')].find(e=>e.dataset.pypaNode===node.id);if(!el)continue;
  el.removeAttribute('title');el.style.height=`${node.h}px`;
  el.dataset.aiExpandedChild=String(node.id.includes('/'));
  if(node.expandedGroup){el.innerHTML=`<b>${esc(node.title)}</b><span class="legend-toggle" data-pypa-expand-toggle="${esc(node.id)}" aria-hidden="true">−</span>`;continue;}
  const data=node.kind==='data';el.classList.toggle('legend-data',data);
  let icon='';
  if(!data){
   const label=node.title;
   const key=/matmul|QK|PV/i.test(label)?'matmul':/softmax/i.test(label)?'softmax':/sum|max|reduce/i.test(label)?'reduce':/transpose/i.test(label)?'transpose':/reshape/i.test(label)?'reshape':null;
   const semanticKey=key||(/cache|store|load|追加|写入|缓存/i.test(label)?"cache":/scal|multiply|mul|mask/i.test(label)?"gate":/accumul|update/i.test(label)?"reduce":null);const entry=root.PtoOpIcons.find(x=>x.id===semanticKey)||root.PtoAIInfraPresentation.iconFor(node);
   icon=`<span class="legend-icon">${entry?`<svg viewBox="0 0 24 24"><path d="${entry.path}"/></svg>`:'<span>ƒ</span>'}</span>`;
  }
  const info=node.overlay||{};
  const fields=[layers.shape?info.shape:null,layers.shape&&info.valid?`Valid ${info.valid}`:null,layers.precision?info.precision:null,layers.memory?[info.memory,info.executionUnit].filter(Boolean).join(' · '):null].filter(Boolean);
  el.classList.toggle('is-control-object',!!layers.control&&!!info.control);
  el.innerHTML=icon+`<span class="legend-text"><b>${esc(node.title)}</b>${fields.length?`<small>${esc(fields.join(' · '))}</small>`:''}</span>`+(node.expandable?`<span class="legend-toggle" data-pypa-expand-toggle="${esc(node.id)}" aria-hidden="true">+</span>`:'');
  if(data){
   const role=root.PtoAIInfraPresentation.semanticRole(node);
   el.dataset.semanticRole=role.key;el.style.setProperty('--data-role',role.color);
   const caption=[layers.shape?node.overlay?.shape:null,layers.precision?node.overlay?.precision:null].filter(Boolean).join(' · ');
   el.style.height=`${caption?64:35}px`;el.style.minHeight=el.style.height;el.style.top=`${node.y+(node.h-(caption?64:35))/2}px`;
   el.innerHTML=root.PtoTensorRoleGlyph.render({id:node.id,label:node.title,metadata:caption,color:role.color,width:node.w});

  }
 }
}
root.PtoStudioGraphLegendSkin=Object.freeze({apply});
})(window);
