/* Independent narrative projection, consumed by the existing semantic graph UI.
 * No Python algorithm, real semantic node or source dependency is rewritten. */
(function(global){
'use strict';
const cache=new WeakMap(), innerCache=new WeakMap();
const source={file:'photo-narrative',ranges:[{start:1,end:1}],symbols:[]};
const provenance=[{kind:'resolved',origin:'narrative',fields:[],note:'Explicit photo-studio analogy; not a source-code fact.'}];
const ports=()=>({inputs:{default:{}},outputs:{default:{}}});
const operation=(id,name,description)=>({id,type:'operation',name,description,source,provenance,ports:ports(),operation:{semantics:description,category:'photo-narrative',api:[]}});
const edge=(s,t,label)=>({id:`${s}>${t}`,s,t,label,type:'data',sourcePort:'default',targetPort:'default'});
const {qk,softmax,pv,accumulate}=global.PtoPhotoNarrative.stageNames;
const stages=[
 ['qk',qk,[['select','筛选入选照片','按顾客选定的主题筛选本批照片。'],['deduplicate','去除重复照片','保留每组重复照片中的一张。'],['handoff','整理待修照片','将入选照片按批次交给修图工序。']]],
 ['softmax',softmax,[['adjust','调整照片','调整本批入选照片的亮度和色彩。'],['check','检查修图结果','确认照片符合工单要求。'],['handoff','整理打印清单','将修好的照片和打印清单交给打印工序。']]],
 ['pv',pv,[['prepare','准备打印','按清单确认照片和打印尺寸。'],['print','打印照片','按批次输出纸质照片。'],['check','检查打印结果','确认照片齐全并交给装帧工序。']]],
 ['output-norm',accumulate,[['collect','汇总已打印照片','收齐各处理批次打印完成的照片。'],['bind','装帧照片','按工单要求装裱照片或编入相册。'],['deliver','整理交付','按顾客与主题归档成品，完成工单。']]]
];
function project(base,{task}={}){
 const projections=task?innerCache:cache;
 if(projections.has(base))return projections.get(base);
 const nodes=stages.map(([id,name,children])=>{
  const childNodes=children.map(a=>operation(...a)),first=childNodes[0].id,last=childNodes.at(-1).id;
  const description=`${name}：${children.map(x=>x[1]).join(' → ')}。照片叙事投影，不代表源码照片算法。`;
  return {id,type:'composite',name,description,source,provenance,ports:ports(),composite:{semantics:description,category:'photo-narrative',api:[],subgraph:{schemaVersion:2,id:`photo-${id}`,nodes:childNodes,edges:childNodes.slice(1).map((n,i)=>edge(childNodes[i].id,n.id,'照片资料')),groups:[]},portMappings:{inputs:{default:{node:first,port:'default',relation:'view'}},outputs:{default:{node:last,port:'default',relation:'view'}}}}};
 });
 // Keep the loop region addressable by the same Stack IDs as the operator
 // graph. The final handoff consumes all batches and sits outside this region.
 const group=task
  ? {id:'task-compute',name:`照片加工 · ${task.stacks.length} 个处理批次`,members:nodes.slice(0,-1).map(n=>n.id)}
  : {id:'photo-workflow',name:'原始照片 → 照片加工（叙事投影）→ 整理完成',members:nodes.map(n=>n.id)};
 const model={schemaVersion:2,id:`photo-${base.id}`,sourceFiles:[{file:'photo-narrative',lineCount:1}],constants:base.constants,runtimeDims:[],nodes,edges:nodes.slice(1).map((n,i)=>edge(nodes[i].id,n.id,['待修照片','已修照片','已打印照片'][i])),groups:[{...group,type:'region',description:global.PtoPhotoNarrative.boundary,source,provenance}],synchronizations:[]};
 projections.set(base,model);return model;
}
function executionData(task){
 if(!task)return null;
 // Each Page has one visible layer on the first processing node.
 const count=task.pages.length,unit='标准照片包';
 return {task:task.id,nodes:{qk:{count,unit,label:`${count} 个${unit}`}},edges:{}};
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
global.PtoPhotoNarrativeGraph=Object.freeze({project,executionData,focus,inspector});
})(window);
