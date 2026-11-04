/* Shared Task identity and parameter markup; view controllers own derived data. */
(function(global){
'use strict';
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const lock='<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-label="由参数计算"><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>';
function sequence(){return parameter('Sequence Length');}
function parameter(primary='Pages') {
 const editable=primary==='Stacks'?'stacks':primary==='Pages'?'pages':'sequence';
 const names={sequence:'Sequence Length',pages:'Pages',stacks:'Stacks'};
 const aria={sequence:'KV Sequence Length',pages:'Page Count',stacks:'Stacks Count'};
 const field=kind=>{
  const active=kind===editable;
  const legacy=kind==='sequence'?'data-effective-sequence':editable==='sequence'?(kind==='pages'?'data-sequence-pages':'data-sequence-stacks'):'data-stack-count';
  return `<label class="compute-parameter-field ${active?'is-editable':'is-derived'}"><span>${names[kind]}</span>${active?`<input class="input" type="number" min="1" step="1" data-compute-pages aria-label="${aria[kind]}">`:`<output>${lock}<span ${legacy} data-parameter-value="${kind}"></span></output>`}</label>`;
 };
 return `<span data-compute-context></span><div class="compute-page-readouts">${field('sequence')}</div><div class="compute-page-stepper"><button class="btn btn-ghost btn-sm" data-page-step="-1" aria-label="减少 ${names[editable]}">−</button><input type="range" min="1" step="1" data-page-range aria-label="调整 ${editable==='pages'?'Page':names[editable]} 数量"><button class="btn btn-ghost btn-sm" data-page-step="1" aria-label="增加 ${names[editable]}">+</button></div><div class="compute-page-readouts compute-derived-readouts">${field('pages')}${field('stacks')}</div><div class="compute-parameter-status"><span data-page-summary></span><span data-page-error role="status"></span><button class="btn btn-ghost btn-sm" data-scenario-reset hidden>Reset</button></div>`;
}
function picker(model,label='Graph Task') { return `<select data-compute-task aria-label="${esc(label)}">${[...new Set(model.tasks.map(t=>t.block))].sort((a,b)=>a-b).map(block=>`<optgroup label="Block ${String(block).padStart(2,'0')}">${model.tasks.filter(t=>t.block===block).sort((a,b)=>a.id-b.id).map(t=>`<option value="${t.id}">Task ${t.id}</option>`).join('')}</optgroup>`).join('')}</select>`; }
function identity() { return '<aside class="compute-identity" aria-label="当前 Task"><h1 data-compute-identity></h1><h2 data-compute-owner></h2><p data-compute-facts></p></aside>'; }
function connect(host,scenario){
 host.addEventListener('click',event=>{if(event.target.closest('[data-scenario-reset]'))scenario.reset();});
 return ()=>{const s=scenario.get();host.querySelector('[data-scenario-reset]').hidden=!s?.isCustom;if(s)host.querySelectorAll('[data-parameter-value]').forEach(e=>{e.textContent=({sequence:s.effectiveSeqLen,pages:s.pageCount,stacks:s.stackCount})[e.dataset.parameterValue];});};
}
global.PtoComputeDetailControls=Object.freeze({identity,parameter,sequence,picker,connect});
})(window);
