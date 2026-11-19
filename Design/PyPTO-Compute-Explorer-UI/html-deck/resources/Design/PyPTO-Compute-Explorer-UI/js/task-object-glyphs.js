/* Object glyphs shared by the compact matrix and the explanatory specimen. */
(function(root){
 'use strict';
 const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function qHead({id,x=0,y=0,w=8,h=w,detailed=false}){
   const side=Math.min(w,h);x+=(w-side)/2;y+=(h-side)/2;w=h=side;
   return `<g class="ai-object-q" data-q-head="${id}" aria-label="Q Head ${id}"><rect class="kf-pa-map__q" x="${x}" y="${y}" width="${w}" height="${h}" rx="0"/><text x="${x+w/2}" y="${y+h/2}" style="font-size:${detailed?11:4.6}px" text-anchor="middle" dominant-baseline="central">${detailed?'Q'+id:'Q'}</text></g>`;
 }
 function stack({index,tokens,pageTokens,pagesPerStack,x=0,y=0,w=10,h=7,detailed=false}){
   const side=Math.min(w,h);x+=(w-side)/2;y+=(h-side)/2;w=h=side;
   if(!detailed)return `<g class="ai-object-stack" data-stack-index="${index}" data-stack-tokens="${tokens}" aria-label="Stack ${index}, ${tokens}/${pageTokens*pagesPerStack} tokens"><rect class="ai-stack-track" x="${x}" y="${y}" width="${side}" height="${side}"/><rect class="ai-stack-fill ai-page-valid" x="${x}" y="${y}" width="${side*Math.max(0,Math.min(1,tokens/(pageTokens*pagesPerStack)))}" height="${side}"/></g>`;
   const gap=detailed?3:.6,pw=(side-gap)/2,ph=pw;
   let body=`<rect class="ai-stack-frame" x="${x}" y="${y}" width="${w}" height="${h}" rx="0"/>`;
   for(let p=0;p<pagesPerStack;p++){
     const valid=Math.max(0,Math.min(pageTokens,tokens-p*pageTokens)),px=x+(p%2)*(pw+gap),py=y+Math.floor(p/2)*(ph+gap);
     body+=`<g class="${valid===pageTokens?'is-full':valid>0?'is-partial':'is-empty'}" data-page-index="${index*pagesPerStack+p}" data-valid-tokens="${valid}"><rect class="ai-page-empty" x="${px}" y="${py}" width="${pw}" height="${ph}"/><rect class="ai-page-valid" x="${px}" y="${py}" width="${pw*valid/pageTokens}" height="${ph}"/>`;
     if(detailed)body+=`<text class="ai-page-name" x="${px+pw/2}" y="${py+ph/2}" dominant-baseline="central" text-anchor="middle" font-size="9">${valid>0?'P'+(index*pagesPerStack+p):'—'}</text>`;
     body+='</g>';
   }
   if(!detailed)body+=`<text class="ai-stack-name" x="${x+w/2}" y="${y+h/2}" style="font-size:3.8px" text-anchor="middle" dominant-baseline="central">S${index}</text>`;
   return `<g class="ai-object-stack" data-stack-index="${index}" data-stack-tokens="${tokens}" aria-label="Stack ${index}, ${tokens}/${pageTokens*pagesPerStack} tokens">${body}</g>`;
 }
 function specimen({id,batch,kv,qPerKv,seqLen,pageTokens,pagesPerStack}){
   const n=Math.ceil(seqLen/(pageTokens*pagesPerStack));
   const heads=Array.from({length:qPerKv},(_,q)=>qHead({id:kv*qPerKv+q,x:12+q*48,y:32,w:30,h:30,detailed:true})).join('');
   const stacks=Array.from({length:n},(_,s)=>{const tokens=Math.min(pageTokens*pagesPerStack,seqLen-s*pageTokens*pagesPerStack),x=12+(s%4)*70,y=118+Math.floor(s/4)*88;return stack({index:s,tokens,pageTokens,pagesPerStack,x,y,w:40,h:40,detailed:true})+`<text class="ai-specimen-label" x="${x}" y="${y-8}">S${s}</text><text class="ai-specimen-count" x="${x}" y="${y+55}">${tokens} tok</text>`;}).join('');
   const height=118+Math.ceil(Math.max(1,n)/4)*88;
   return `<svg viewBox="0 0 300 ${height}" role="img" aria-label="Task ${id}: Q Heads ${kv*qPerKv}–${(kv+1)*qPerKv-1}; ${n} Stacks; ${seqLen} tokens"><text class="ai-specimen-label" x="12" y="16">Query Heads · 1 方格 = 1 Head</text>${heads}<text class="ai-specimen-label" x="12" y="88">KV Stacks · 1 方格 = ${pagesPerStack} Pages</text>${stacks}</svg>`;
 }
 const api=Object.freeze({qHead,stack,specimen});root.PtoTaskObjectGlyphs=api;if(typeof module!=='undefined')module.exports=api;
})(typeof window==='undefined'?globalThis:window);
