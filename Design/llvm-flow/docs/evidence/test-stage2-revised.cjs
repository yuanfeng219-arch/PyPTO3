/* Static contract checks: no browser automation, compiler, or artifact execution. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const zlib = require('zlib');
const crypto = require('crypto');
const assert = require('assert/strict');
const htmlPath = path.resolve(__dirname, '../../llvmcfg-standalone.html');
const {html, resources} = require('./load-page.cjs')(htmlPath);
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)].filter(m => !m[1].includes('src='));
scripts.forEach((m,i) => new vm.Script(m[2], {filename:'inline-'+i}));
const extension = html.slice(html.indexOf('/* STAGE2_REVISED_BEGIN'), html.indexOf('/* STAGE2_REVISED_END */'));
const createElement = (type, props, ...children) => ({type, props:{...props,children:children.flat(Infinity)}, children:children.flat(Infinity)});
const React={createElement,Fragment:'fragment',Suspense:'suspense',isValidElement:e=>!!e&&typeof e==='object'&&'type' in e,
  cloneElement:(e,props,...children)=>createElement(e.type,{...e.props,...props},...(children.length?children:e.children)),Children:{map:(children,fn)=>(Array.isArray(children)?children:[children]).map(fn)}};
const classes=new Proxy({},{get:(_,key)=>key});
const window = {LLVMStage2Bridge:{React, Graph:'native-cfg', classes, stage:'source', flows:{}}, dispatchEvent(){}, DecompressionStream, confirm(){return false}};
vm.runInNewContext(resources.find(r=>r.file.endsWith('/data/deepseek-evidence.js')).content, {window});
vm.runInNewContext(extension, {window, Event, DecompressionStream, Response, Blob, Uint8Array, atob, requestAnimationFrame:cb=>cb(), document:{getElementById(){return null}}});
const UI=window.LLVMStage2, C=UI.contract, S=C.state, D=C.data;
function visit(tree,predicate) {
  if(!tree || typeof tree!=='object')return [];
  return (predicate(tree)?[tree]:[]).concat((tree.children||[]).flatMap(c=>visit(c,predicate)));
}
function texts(tree) {
  if(tree==null || typeof tree==='boolean')return '';
  if(typeof tree!=='object')return String(tree);
  return (tree.children||[]).map(texts).join(' ');
}
async function main() {
  assert.deepEqual(JSON.parse(JSON.stringify(D.totals)), {snapshots:52,comparisons:51,changed:33,identical:18});
  assert.equal(D.inputs.length,55);
  assert.equal(D.inputs.find(i=>i.name==='x_hc').shape[1],'T_DYN');
  assert.equal(D.inputs.find(i=>i.name==='x_hc').metadataShape[1],-1);
  assert.equal(S.caseId,D.case.id);
  assert.equal(UI.active('transform'),true);
  assert.equal(UI.active('source'),false,'original Source components must render');
  assert.equal(UI.caseSelector(),null);
  assert.equal(UI.canNavigate('transform'),false);
  await C.load();
  assert.equal(S.error,'');
  assert.equal(S.loaded,true);
  assert.equal(UI.sourceData().lines.length,1793);
  assert.ok(UI.sourceData().lines.some(l=>l.includes('def l3_decode_csa(')));
  // Execute the retained original source-code component, not the withdrawn pane.
  const codeStart=html.indexOf('function M(t){let{workflow:');
  const codeEnd=html.indexOf('function I(e){',codeStart);
  assert.ok(codeStart>0&&codeEnd>codeStart);
  const jsx=(type,props)=>createElement(type,props,...(Array.isArray(props.children)?props.children:[props.children]));
  const oldCode=vm.runInNewContext('('+html.slice(codeStart,codeEnd)+')',{
    window,e:{...React,useState:()=>['',()=>{}]},a:{jsx,jsxs:jsx},p:classes,c:[16,8,4,2,1],d:(a,b)=>JSON.stringify(a)===JSON.stringify(b),document:{getElementById:()=>null}
  })({workflow:'initial',appliedFactors:[16,8,4,2,1],onUndo(){}});
  assert.equal(oldCode.props.className,'codePanel codePanelFull');
  assert.ok(texts(oldCode).includes('搜索源码'));
  assert.ok(texts(oldCode).includes('定位控制区域'));
  const codeLines=visit(oldCode,t=>t.props.className?.split(' ').includes('codeLine'));
  assert.equal(codeLines.length,1793);
  codeLines.forEach((row,i)=>assert.equal(texts(visit(row,t=>t.type==='code')[0]),C.sourceLines()[0][i]||' '));
  const coldCode=vm.runInNewContext('('+html.slice(codeStart,codeEnd)+')',{
    window:{},e:{...React,useState:()=>['',()=>{}]},a:{jsx,jsxs:jsx},p:classes,c:[],d:()=>true,document:{getElementById:()=>null}
  })({workflow:'initial',appliedFactors:[],onUndo(){}});
  assert.ok(texts(coldCode).includes('正在初始化真实源码'));
  assert.equal(visit(coldCode,t=>t.props.className?.split(' ').includes('codeLine')).length,0,'no legacy source during startup');
  assert.ok(html.includes('if(!window.LLVMStage2)return null;return(0,a.jsxs)("main"'),'root waits for extension before rendering dependent panes');
  assert.ok(html.includes('e=window.LLVMStage2?.bindSourceProps(e)??e;'));
  assert.ok(html.includes('t=window.LLVMStage2?.bindSourceProps(t)??t;'));
  const effect=html.match(/const refresh=\(\)=>stage2Refresh\(v=>v\+1\);window.addEventListener\("llvm-stage2-change",refresh\);refresh\(\);return\(\)=>window.removeEventListener\("llvm-stage2-change",refresh\)/)?.[0];
  assert.ok(effect,'subscription must refresh even if registration happened before effect mount');
  for(const registeredBeforeMount of [false,true]) {
    let revision=0;const listeners=new Set();const startupWindow={addEventListener:(_,f)=>listeners.add(f),removeEventListener:(_,f)=>listeners.delete(f)};
    if(registeredBeforeMount)listeners.forEach(f=>f());
    const cleanup=vm.runInNewContext('(function(){'+effect+'})()',{window:startupWindow,stage2Refresh:update=>{revision=update(revision);}});
    assert.equal(revision,1);
    if(!registeredBeforeMount){listeners.forEach(f=>f());assert.equal(revision,2);}
    cleanup();assert.equal(listeners.size,0);
  }
  const original=createElement('aside',{className:'original-pane'},createElement('header',{},'original header'),createElement('section',{className:'reviewControls'},'old constraints'),createElement('section',{className:'runCard'},createElement('strong',{},'old scenario')));
  let leftMode='code';
  const bound=UI.bindSourceProps({stage:'source',onLeftMode:mode=>{leftMode=mode;}});
  assert.equal(bound.workflow,'initial');bound.onRun();
  assert.equal(leftMode,'review','running the review must open the constraint-review tab');
  assert.equal(UI.canNavigate('transform'),true);
  const adapted=UI.adaptExplorer(original,{stage:'source'});
  assert.equal(adapted.type,'aside');assert.equal(adapted.props.className,'original-pane');assert.ok(texts(adapted).includes('original header'));
  assert.ok(texts(adapted).includes('DeepseekV4 · 静态基线已就绪'));
  assert.ok(!texts(adapted).includes('old scenario'));
  const canvas=createElement('section',{className:'original-canvas'},createElement('div',{className:'graphArea'},'old graph'));
  assert.equal(visit(UI.adaptCenter(canvas,{stage:'source'}),t=>t.type==='native-cfg').length,1);
  const entry=C.graph().graphs[0].l3_decode_csa.nodes.find(n=>n.kind==='entry');
  C.selectNode('before',entry);
  assert.equal(S.selection.snapshot,0);
  assert.equal(UI.originalSelection({}).title,'l3_decode_csa');
  assert.ok(texts(adapted).includes('55 个'));
  assert.equal(visit(adapted,t=>t.type==='select').length,0,'constraint panel must not replace original controls with dropdowns');
  const lengthInput=visit(adapted,t=>t.props.id==='shape-t')[0];
  assert.equal(lengthInput.props.type,'number');assert.equal(lengthInput.props.value,16);
  const quick=visit(adapted,t=>t.props.className==='scenarioButtons')[0];
  assert.equal(visit(quick,t=>t.type==='button').length,6);
  visit(quick,t=>t.type==='button'&&texts(t)==='31')[0].props.onClick();
  assert.equal(S.constraintDraft.t,31);
  const edited=UI.adaptExplorer(original,{stage:'source'});
  const factors=visit(edited,t=>t.props.className==='factorButtons')[0];
  assert.equal(visit(factors,t=>t.type==='button').length,6);
  visit(factors,t=>t.type==='button'&&texts(t).replace(/\s/g,'')==='×4')[0].props.onClick();
  assert.ok(!S.constraintDraft.factors.includes(4));
  assert.equal(S.constraintDraft.t,31);assert.equal(S.pass,2);assert.equal(S.sourceFn,'l3_decode_csa');
  let navigatedStage='';window.LLVMStage2Bridge.navigate=stage=>{navigatedStage=stage;};
  const passCompile=visit(UI.adaptExplorer(original,{stage:'source'}),t=>t.type==='button'&&texts(t).trim()==='查看 Pass 编译')[0];
  assert.ok(passCompile,'constraint review must expose the Stage 2 action');
  assert.equal(passCompile.props.disabled,undefined);
  assert.equal(visit(passCompile,t=>t.type==='svg')[0].props.className,'icon','Pass action must reuse original bounded icon styling');
  passCompile.props.onClick();assert.equal(navigatedStage,'transform');
  assert.equal(UI.guardStage('transform'),false);
  window.LLVMStage2Bridge.stage='transform';
  assert.ok(!texts(UI.explorer()).includes('承接 Stage 1'));
  const recipes=JSON.parse(zlib.gunzipSync(Buffer.from(D.sourcesPacked,'base64')));
  const split=text=>text.match(/[^\n]*\n|[^\n]+$/g)||[];
  let raw=recipes[0];
  for(let i=0;i<52;i++) {
    if(i) {
      const previous=split(raw), next=[];let cursor=0;
      for(const [a,b,insert] of recipes[i]){next.push(...previous.slice(cursor,a),...split(insert));cursor=b;}
      next.push(...previous.slice(cursor)); raw=next.join('');
    }
    assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),D.snapshots[i].sha256,`snapshot ${i} lossless`);
    assert.equal(C.sourceLines()[i].join('\n'),raw.replace(/\n$/,''));
  }
  let graphCount=0;
  for(const [index,functions] of Object.entries(C.graph().graphs))for(const g of Object.values(functions)) {
    assert.equal(g.status,'analyzed');graphCount++;
    const ids=new Set(g.nodes.map(n=>n.id));
    assert.equal(ids.size,g.nodes.length);
    g.edges.forEach(e=>{assert.ok(ids.has(e.source));assert.ok(ids.has(e.target));});
    const reach=new Set([g.entry]); let size=0;
    while(reach.size!==size){size=reach.size;g.edges.forEach(e=>{if(reach.has(e.source))reach.add(e.target);});}
    assert.ok(reach.has(g.exit),`${index}/${g.name} has exit path`);
    g.nodes.filter(n=>n.kind==='loop').forEach(n=>{
      assert.equal(g.edges.filter(e=>e.source===n.id).length,2);
      assert.ok(g.edges.some(e=>e.target===n.id), 'loop header reachable');
    });
    g.nodes.filter(n=>n.kind==='return').forEach(n=>assert.equal(g.edges.find(e=>e.source===n.id).target,g.exit));
  }
  assert.equal(graphCount,104);
  for(const [name,g] of Object.entries(C.graph().graphs[2])) {
    const other=C.graph().graphs[3][name];
    assert.equal(g.nodes.length,other.nodes.length,'identical graph partition');
    assert.equal(JSON.stringify(g.edges),JSON.stringify(other.edges),'identical CFG edges');
    assert.equal(JSON.stringify(g.nodes.map(n=>[n.id,n.kind,n.start,n.end])),JSON.stringify(other.nodes.map(n=>[n.id,n.kind,n.start,n.end])));
  }
  C.pickPass(2);
  const before=C.graph().graphs[1].indexer_topk_group_wave;
  const after=C.graph().graphs[2].indexer_topk_group_wave;
  const loop=before.nodes.find(n=>n.kind==='loop'&&n.start===99);
  assert.ok(loop);
  const exitEdges=before.edges.filter(e=>e.source===loop.id);
  assert.equal(exitEdges.length,2);
  assert.ok(before.edges.some(e=>e.target===loop.id && before.nodes.find(n=>n.id===e.source).start>=99),'real loop backedge');
  assert.ok(!after.nodes.some(n=>n.kind==='loop'&&n.text.includes('group_leaf')));
  C.selectNode('before',loop);
  const matched=after.nodes.filter(n=>C.isLinked(S.selection.id,C.nodeId('after',n)));
  assert.ok(matched.some(n=>n.start>=99&&n.end<=113));
  assert.ok(matched.some(n=>n.start>=114&&n.end<=128));
  assert.ok(C.nodeMeaning(loop,'before').tags.includes('展开前 · 对应 2 个副本'));
  assert.ok(matched.some(n=>C.nodeMeaning(n,'after').tags.includes('展开副本 1/2')));
  assert.ok(matched.some(n=>C.nodeMeaning(n,'after').tags.includes('展开副本 2/2')));
  const selection=S.selection.id;
  S.view='code';assert.ok(texts(UI.center()).includes('indexer_topk_group_wave'));S.view='graph';assert.equal(S.selection.id,selection);
  C.selectLine('after',115);assert.equal(S.selection.side,'after');assert.ok(S.selection.id.includes(D.case.id));
  S.full=false;const focus=C.graphModel('before');S.full=true;const full=C.graphModel('before');
  assert.ok(full.nodes.length>focus.nodes.length);assert.ok(focus.nodes.some(n=>n.kind==='boundary'));
  C.pickPass(3);assert.equal(S.selection,null);
  const stage2Center=UI.center();
  assert.equal(visit(stage2Center,t=>t.type==='native-cfg').length,1,'zero diff defaults to single CFG');
  assert.equal(visit(stage2Center,t=>t.props.role==='tab').length,2,'view choices use tabs rather than action buttons');
  assert.ok(!texts(stage2Center).includes('03 CtrlFlowTransform'),'redundant pass summary is removed');
  S.dual=true;assert.equal(visit(UI.center(),t=>t.type==='native-cfg').length,2);
  C.pickPass(9);
  const rope=D.entities.find(e=>e.kind==='scope'&&e.name==='csa_rope_interleave');
  C.focusEntity(rope.id);assert.equal(S.selection.side,'before');
  const caller=C.graph().graphs[9].decode_csa_test;
  const submit=caller.nodes.find(n=>n.submitTarget==='csa_rope_interleave');
  assert.equal(submit.start,2462);
  assert.ok(C.isLinked(S.selection.id,C.nodeId('after',submit)));
  C.selectNode('after',submit);const returnId=S.selection.id;C.openTarget();
  assert.equal(S.afterFn,'csa_rope_interleave');assert.equal(S.selection.kind,'node');
  const addedReturn=C.graph().graphs[9].csa_rope_interleave.nodes.find(n=>n.kind==='return');
  assert.equal(addedReturn.entities.length,0,'appended return is excluded from outline body proof');
  C.goBack();assert.equal(S.afterFn,'decode_csa_test');assert.equal(S.selection.id,returnId);
  C.pickPass(4);assert.ok(texts(UI.center()).includes('尚未接入'));assert.ok(!texts(UI.inspector()).includes('10 个位置参数'));
  C.selectLine('after',10);assert.equal(S.selection.kind,'line');assert.ok(texts(UI.inspector()).includes('尚无已提取图对象'));
  // User removed timeline search/filter and disabled unavailable graph entries.
  const timeline=UI.explorer();
  const passRows=visit(timeline,t=>t.type==="button"&&t.props["aria-current"]);
  assert.equal(passRows.length,51);
  assert.equal(passRows.filter(t=>!t.props.disabled).length,4);
  assert.equal(passRows.filter(t=>t.props.disabled).every(t=>t.props.onClick===undefined),true);
  assert.equal(visit(timeline,t=>t.props.className==="s2-diff-tag").length,33);
  assert.equal(visit(timeline,t=>t.props.className==="s2-pass-description"&&t.props.title).length,51);
  assert.equal(visit(timeline,t=>t.props["aria-label"]==="搜索 Pass"||t.props["aria-label"]==="文本状态筛选").length,0);
  assert.equal(UI.guardStage('runtime'),false);
  window.LLVMStage2Bridge.stage='runtime';
  const pending=UI.adaptCenter(canvas,{stage:'runtime'});
  assert.ok(texts(pending).includes('尚未接入'));
  assert.ok(!texts(pending).includes('1570'));
  assert.equal(UI.active('runtime'),false);assert.equal(UI.active('source'),false);
  assert.ok(!html.includes('function sourceExplorer()'));
  assert.ok(!html.includes('function sourceCenter()'));
  assert.ok(!html.includes('function sourceInspector()'));
  assert.ok(!html.includes('id="deepseek-pass-diff-workspace"'));
  assert.ok(html.includes('llvmJson:T.before_json'));
  assert.ok(html.includes('llvmJson:T.after_json'));
  assert.ok(html.includes('window.LLVMStage2?.active(n)'));
  // User requested semantic nodes after rejecting bare %x nodes (2026-09-07).
  assert.ok(html.includes('reviewCfg:{nodeWidth:220,nodeHeight:136,defaultPosition:[0,0],minZoom:.05,labelType:"simple",stage2:!0}'));
  const allocation=C.nodeMeaning({id:1,kind:'statements',text:'gather_window_buf: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(4194304, pl.INT64))',entities:[]},'before');
  assert.equal(allocation.title,'分配窗口与同步缓冲区');
  assert.ok(!allocation.description.includes('pl.const'));
  assert.ok(texts(UI.renderNode({...allocation,eyebrow:'操作 · %1'})).includes('分配窗口与同步缓冲区'));
  assert.equal(C.edgeCaption('有下一项'),'继续');
  assert.equal(C.edgeCaption('迭代结束'),'结束');
  assert.equal(C.edgeCaption('False'),'不成立');
  assert.equal(C.nodeMeaning({id:1,kind:'loop',text:'for window in windows:',entities:[]},'before').title,'遍历 window');
  assert.ok(html.includes('onSelectBlock:(_,id)=>selectById(side,id), readOnly:false, showMinimap:true'));
  assert.ok(html.includes('targetHandle:E(n,r)||"b",sourceHandle:E(n,r)||"b"'));
  assert.ok(html.includes('onDoubleClick:()=>{o(!n)}'));
  assert.ok(html.includes('e.data={...e.data,layoutDirection:n}'));
  assert.ok(html.includes('width:e.width||l.nodeWidth,height:e.height||l.nodeHeight'));
  assert.ok(html.includes('onEdgeClick:l.stage2?'));
  assert.ok(html.includes('split.dataset.pixelSizes="320,0,360"'),'Inspector defaults to 360px');
  console.log(`PASS: ${scripts.length} linked scripts parse; 52 SHA-verified snapshots; 55 inputs; 51 pairs; 104 normal-flow function graphs; default Deepseek source review, stage continuity, unroll one-to-many, zero-diff, outline return, unsupported fallback, no legacy downstream data.`);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
