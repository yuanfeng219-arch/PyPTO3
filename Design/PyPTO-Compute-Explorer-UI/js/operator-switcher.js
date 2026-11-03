/* Shared operator navigation for the independent visual explorers. */
(()=>{'use strict';
const csa=document.body.classList.contains('csa-visual');
const menu=document.createElement('details');menu.className='operator-switcher';
menu.innerHTML='<summary aria-label="切换算子"><span>'+(csa?'decode_csa.py · TP=2':'paged_attention_pypto.py')+'</span><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></summary><nav aria-label="算子"><a href="./explorer.html"'+(!csa?' aria-current="page"':'')+'><span>Paged Attention<small>paged_attention_pypto.py</small></span><span aria-hidden="true">'+(!csa?'✓':'')+'</span></a><a href="./csa.html"'+(csa?' aria-current="page"':'')+'><span>Decode CSA<small>decode_csa.py · TP=2</small></span><span aria-hidden="true">'+(csa?'✓':'')+'</span></a></nav>';
function attach(){const header=document.querySelector('.viz2:not([hidden])>.viz2-header');if(header){document.body.classList.remove('ai-booting');if(!header.contains(menu)){header.append(menu);header.classList.add('has-operator-switcher');}}}
new MutationObserver(attach).observe(document.getElementById('ideWorkarea'),{childList:true,subtree:true,attributes:true,attributeFilter:['hidden']});attach();
menu.addEventListener('toggle',()=>{menu.querySelector('summary').setAttribute('aria-expanded',String(menu.open));if(menu.open){const tip=document.getElementById('ai-ui-tooltip');if(tip)tip.hidden=true;}});
document.addEventListener('pointerdown',e=>{if(menu.open&&!menu.contains(e.target))menu.open=false;});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&menu.open){menu.open=false;menu.querySelector('summary').focus();}});
})();
