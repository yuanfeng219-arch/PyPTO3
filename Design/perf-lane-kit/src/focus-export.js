import {adapt} from './data.js';

// Load data, not executable scripts, from a focus export.
export async function readFocusExport(read, run='聚焦导出') {
 const paths=['mock/lane-data-rendered.json','mock/qk-inline-events.json','mock/qk-pipeline-models.json','derived/pass-analysis.json','derived/parallel-lane.json',...['callable','csa-qk-pv'].flatMap(s=>['graph','passes','isa','views'].map(f=>`derived/compile/${s}/${f}.json`))];
 const values=await Promise.all(paths.map(read));
 const files=Object.fromEntries(paths.map((p,i)=>[p,values[i]]));
 const [snapshot,events,models]=values;
 if(!Array.isArray(events)||!events.length||!models||typeof models!=='object')throw Error('聚焦导出缺少有效核内事件或流水模型');
 const scenarios=Object.fromEntries(['callable','csa-qk-pv'].map(s=>[s,Object.fromEntries(['graph','passes','isa','views'].map(f=>[f,files[`derived/compile/${s}/${f}.json`]]))]));
 for(const [name,s] of Object.entries(scenarios))if(!Array.isArray(s.graph?.nodes)||!Array.isArray(s.passes?.passes))throw Error(`编译数据无效：${name}`);
 return adapt({...snapshot,run,focusExport:{available:true,events,models,passAnalysis:files['derived/pass-analysis.json'],parallel:files['derived/parallel-lane.json'],scenarios}});
}

export function findIncore(data, rank, task, core) {
 const rows=data?.focusExport?.events||[];
 return rows.find(x=>String(x.rank)===String(rank)&&String(x.taskId)===String(task.taskId)&&x.leaf===task.leaf&&(core==null||String(x.core)===String(core)))||null;
}

export function incoreKey(rank, taskId, leaf, core) {
 return JSON.stringify([String(rank),String(taskId),String(leaf),String(core)]);
}
