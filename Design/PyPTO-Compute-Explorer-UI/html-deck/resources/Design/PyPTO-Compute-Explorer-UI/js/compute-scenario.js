/* One immutable parameter scenario shared by the three Task Detail projections. */
(function(global){
'use strict';
const constants=()=>global.PtoPagedAttentionIntro.model.constants;
function create(task,seqLen=task.seqLen){
 const counts=global.PtoPagedAttentionIntro.model.partition(seqLen);
 return {taskId:task.id,originalSeqLen:task.seqLen,effectiveSeqLen:seqLen,pageCount:counts.pageCount,stackCount:counts.stackCount,isCustom:seqLen!==task.seqLen};
}
function limits(s){
 const c=constants(),tail=(s.effectiveSeqLen-1)%c.pageTokens+1,tailPages=(s.pageCount-1)%c.pagesPerStack+1;
 const pages=Math.floor((c.maxSeq-tail)/c.pageTokens)+1;
 return {sequence:c.maxSeq,pages,stacks:Math.floor((pages-tailPages)/c.pagesPerStack)+1};
}
function edit(s,kind,value){
 const max=limits(s)[kind],count=Number(value),labels={sequence:'KV Sequence Length',pages:'Pages',stacks:'Stacks'};
 if(!max||!/^\d+$/.test(String(value))||!Number.isSafeInteger(count)||count<1||count>max)return {error:`${labels[kind]} 必须为 1–${max} 的整数`};
 const c=constants(),tail=(s.effectiveSeqLen-1)%c.pageTokens+1,tailPages=(s.pageCount-1)%c.pagesPerStack+1;
 const pages=kind==='stacks'?(count-1)*c.pagesPerStack+tailPages:count;
 const seqLen=kind==='sequence'?count:(pages-1)*c.pageTokens+tail;
 return {scenario:create({id:s.taskId,seqLen:s.originalSeqLen},seqLen)};
}
// All renderers receive projections from this same anatomy instance. Original Tasks stay intact.
function project(s,anatomy,source,preLaunch){
 const original=anatomy.tasks[s.taskId];
 let effective=anatomy;
 if(s.isCustom){
  effective=global.PtoVisualization2Anatomy.createModel({...anatomy,seqLens:Array.from({length:original.batch+1},()=>s.effectiveSeqLen),blockTable:null});
  effective.tasks=[effective.tasks.find(t=>t.id===s.taskId)];
 }
 const task=s.isCustom?effective.tasks[0]:original;
 const local={...effective,tasks:[task]};
 const graph=global.PtoVisualization2GraphIntra.createModel({anatomy:local,source,preLaunch}).tasks[0];
 if(s.isCustom)graph.graph.id+=`-scenario-${s.effectiveSeqLen}`;
 return {assignment:task,pipeline:global.PtoVisualization2PipelineIntra.createModel({anatomy:local,preLaunch}).tasks[0],graph};
}
global.PtoComputeScenario=Object.freeze({create,limits,edit,project});
})(window);
