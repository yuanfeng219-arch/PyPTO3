/* Independent narrative projection, consumed by the existing semantic graph UI.
 * No Python algorithm, real semantic node or source dependency is rewritten. */
(function(global){
'use strict';
const cache=new WeakMap();
const source={file:'photo-narrative',ranges:[{start:1,end:1}],symbols:[]};
const provenance=[{kind:'resolved',origin:'narrative',fields:[],note:'Explicit photo-studio analogy; not a source-code fact.'}];
const ports=()=>({inputs:{default:{}},outputs:{default:{}}});
const operation=(id,name,description)=>({id,type:'operation',name,description,source,provenance,ports:ports(),operation:{semantics:description,category:'photo-narrative',api:[]}});
const edge=(s,t,label)=>({id:`${s}>${t}`,s,t,label,type:'data',sourcePort:'default',targetPort:'default'});
const stages=[
 ['qk','主题匹配',[['conditions','读取匹配条件','读取顾客选择的主题与匹配子项。'],['match','匹配照片主题','按主题组织待处理照片；每个标准照片包最多 128 张。'],['score','形成匹配结果','将匹配结果交给筛选去重。']]],
 ['softmax','筛选去重',[['select','筛选入选照片','结合本批照片的主题匹配结果确定入选照片。'],['deduplicate','去除重复照片','在叙事中合并重复选择；不声称真实源码执行照片去重。'],['handoff','整理精修清单','按处理批次交给精修工序，并保留交接记录。']]],
 ['pv','修图',[['crop','裁剪','按成品构图要求调整照片边界。'],['brightness','亮度调整','整理当前处理批次照片的明暗表现。'],['color','色彩调整','整理色彩并形成该批次的精修结果。']]],
 ['output-norm','打印出片',[['collect','汇总精修照片','收齐各处理批次的精修结果。'],['print','打印照片','将整理完成的照片交给冲印输出。'],['deliver','整理交付','按顾客与主题归档成品，完成工单。']]]
];
function project(base){
 if(cache.has(base))return cache.get(base);
 const nodes=stages.map(([id,name,children])=>{
  const childNodes=children.map(a=>operation(...a)),first=childNodes[0].id,last=childNodes.at(-1).id;
  const description=`${name}：${children.map(x=>x[1]).join(' → ')}。照片叙事投影，不代表源码照片算法。`;
  return {id,type:'composite',name,description,source,provenance,ports:ports(),composite:{semantics:description,category:'photo-narrative',api:[],subgraph:{schemaVersion:2,id:`photo-${id}`,nodes:childNodes,edges:childNodes.slice(1).map((n,i)=>edge(childNodes[i].id,n.id,'照片资料')),groups:[]},portMappings:{inputs:{default:{node:first,port:'default',relation:'view'}},outputs:{default:{node:last,port:'default',relation:'view'}}}}};
 });
 const model={schemaVersion:2,id:`photo-${base.id}`,sourceFiles:[{file:'photo-narrative',lineCount:1}],constants:base.constants,runtimeDims:[],nodes,edges:nodes.slice(1).map((n,i)=>edge(nodes[i].id,n.id,['匹配结果','入选照片','精修成品'][i])),groups:[{id:'photo-workflow',type:'region',name:'原始照片 → 照片加工（叙事投影）→ 整理完成',description:global.PtoPhotoNarrative.boundary,source,provenance,members:nodes.map(n=>n.id)}],synchronizations:[]};
 cache.set(base,model);return model;
}
function focus(graph,selection,view,adapter){
 const f=selection?.subobject,requested=f?.node||f?.stage||selection?.graphNodeId||view.nodeId;
 const root=requested?.split('/')[0],aliases={accumulate:'output-norm','context-acc':'output-norm',output:'output-norm',normalize:'output-norm',scores:'qk','k-cache':'qk',q:'qk',probability:'softmax','scale-mask':'softmax','v-cache':'pv',partial:'pv',m:'softmax',l:'softmax',rescale:'softmax',o:'output-norm'};
 const id=graph.nodes.some(n=>n.id===requested)?requested:aliases[root]||root;
 const nodeId=graph.nodes.some(n=>n.id===id)?id:null;
 return {nodeId,related:nodeId?[nodeId]:graph.nodes.map(n=>n.id),kind:selection?.kind||'none'};
}
function inspector(node,selection,task){
 const title=node?.title||'当前照片工单';
 const facts=task?global.PtoPhotoNarrative.taskSummary(task):selection?.task!=null?`工单 ${selection.task} · 顾客 ${selection.batch+1} · 主题 ${selection.kvHead} · 工作小组 ${selection.block}`:'按顾客与主题组织照片工单';
 return {title,html:`<section class="viz2-inspector-core"><h3>当前工单</h3><p>${facts}</p></section><section class="viz2-inspector-core"><h3>加工说明</h3><p>${node?.detail||'选择阶段查看处理步骤；点击 + 展开阶段内部。'}</p></section><section class="viz2-inspector-core"><h3>交接规则</h3><p>每个标准照片包最多 128 张；每个处理批次最多 4 包、512 张。处理批次依次使用周转台位 0、1、2，再循环使用；该规则表示逻辑交接，不表示槽位真实占用。</p></section><section class="viz2-inspector-core"><h3>叙事边界</h3><p>${global.PtoPhotoNarrative.boundary}</p></section>`};
}
global.PtoPhotoNarrativeGraph=Object.freeze({project,focus,inspector});
})(window);
