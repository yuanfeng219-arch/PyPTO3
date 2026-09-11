/* Adapted from /Users/yin/pto/swimlane/app.js parseTraceTask/parseCoreTask.
 * Preserve source event arguments, but use run/rank/pid/tid identity (not threadName).
 * Chrome Trace timestamps are microseconds. core-task execStart/execEnd retain source units.
 */
(function(global){
 'use strict';
 const kind=name=>/^AIC(?:_|\b)/i.test(name)?'AIC':/^AIV(?:_|\b)/i.test(name)?'AIV':'Other';
 const hint=(text,key)=>String(text||'').match(new RegExp(key+'\\s*(?:[:=]|\\.{2,})\\s*([^\\s,;]+)','i'))?.[1]??null;
 const magics=value=>Array.isArray(value)?value:[...String(value||'').matchAll(/rawmagic["']?\s*(?:[:=]|\.+)\s*([^,\s;}\]]+)/gi)].map(m=>m[1]);
 function intervals(tasks,start=-Infinity,end=Infinity){
  const result=[];
  for(const t of tasks){const a=Math.max(start,t.start),b=Math.min(end,t.end);if(b<=a)continue;result.push([a,b]);}
  result.sort((a,b)=>a[0]-b[0]);const merged=[];
  for(const pair of result){const last=merged.at(-1);if(last&&pair[0]<=last[1])last[1]=Math.max(last[1],pair[1]);else merged.push(pair.slice());}
  return merged;
 }
 function occupied(tasks,start,end){return intervals(tasks,start,end).reduce((n,[a,b])=>n+b-a,0);}
 function parse(raw,{runId='local-run',rank=0,name='Trace',source='',annotated=false}={}){
  let events=Array.isArray(raw?.traceEvents)?raw.traceEvents:null;
  if(!events){
   const cores=Array.isArray(raw)?raw:raw?.cores;
   if(!Array.isArray(cores)||!cores.every(c=>Array.isArray(c.tasks)))throw Error('需要 Chrome Trace 的 traceEvents，或包含 coreType / tasks 的数组。');
   events=[{ph:'M',name:'process_name',pid:0,args:{name:'Worker View'}}];
   cores.forEach((c,tid)=>{events.push({ph:'M',name:'thread_name',pid:0,tid,args:{name:c.coreType||'Core_'+tid}});c.tasks.forEach(t=>events.push({ph:'X',pid:0,tid,ts:t.execStart,dur:Number(t.execEnd)-Number(t.execStart),name:t.taskName||t.name||t.label||'Task',args:t}));});
  }
  const processes=new Map(),threads=new Map();
  for(const e of events){if(e.ph!=='M')continue;if(e.name==='process_name')processes.set(e.pid,String(e.args?.name));if(e.name==='thread_name')threads.set(JSON.stringify([e.pid,e.tid]),String(e.args?.name));}
  const valid=events.filter(e=>e.ph==='X'&&Number.isFinite(e.ts)&&Number.isFinite(e.dur)&&e.dur>=0);
  if(!valid.length)throw Error('Trace 没有可展示的有效 X 事件。');
  const origin=valid.reduce((n,e)=>Math.min(n,e.ts),Infinity),lanes=new Map(),byId=new Map();
  valid.forEach((e,i)=>{
   const args=e.args||{},pid=e.pid??0,tid=e.tid??0,laneId=JSON.stringify([runId,rank,pid,tid]);
   const laneName=threads.get(JSON.stringify([pid,tid]))||'Thread '+tid,process=processes.get(pid)||'Process '+pid;
   if(!lanes.has(laneId))lanes.set(laneId,{id:laneId,rank,pid,tid,name:laneName,process,kind:kind(laneName),worker:/worker|machine/i.test(process),tasks:[]});
   const t={id:laneId+':'+i,laneId,rank,pid,tid,laneKind:kind(laneName),laneIdLabel:laneName,process,rawName:String(e.name||'Task'),label:String(e.name||'Task').split('(')[0],shortId:String(e.name||'').match(/\((r\dt\d+)\)/)?.[1]||String(args.taskId??i),taskId:String(args.taskId??i),start:e.ts-origin,end:e.ts+e.dur-origin,duration:e.dur,raw:e,args,
    rootHash:args.rootHash??hint(args['event-hint'],'rootHash'),callOpMagic:args.callOpMagic??hint(args['event-hint'],'callOpMagic'),leafHash:args.leafHash??hint(args['event-hint'],'leafHash'),inputRawMagic:magics(args.inputRawMagic??args['ioperand-hint']),outputRawMagic:magics(args.outputRawMagic??args['ooperand-hint'])};
   t.displayName=t.rawName;t.laneId=laneId;t.laneName=laneName;t.funcId=args.funcId??hint(args['event-hint'],'FuncId');
   t.colorKey=/wait|complete|retire/i.test(t.label)?'wait':/push|publish|gather|readback|copy/i.test(t.label)?'copy':t.label;
   t.fanin=[...String(args['fanin-hint']||'').matchAll(/r\dt\d+/g)].map(m=>m[0]);
   t.fanout=[...String(args['fanout-hint']||'').matchAll(/r\dt\d+/g)].map(m=>m[0]);
   lanes.get(laneId).tasks.push(t);byId.set(t.id,t);
  });
  const duration=valid.reduce((n,e)=>Math.max(n,e.ts+e.dur-origin),0)||1;
  let overlapLanes=0;
  for(const lane of lanes.values()){lane.tasks.sort((a,b)=>a.start-b.start||a.end-b.end);lane.busy=occupied(lane.tasks,0,duration);lane.utilization=100*lane.busy/duration;lane.overlap=lane.tasks.reduce((n,t)=>n+t.duration,0)-lane.busy>0.001;if(lane.overlap)overlapLanes++;}
  const ordered=[...lanes.values()].sort((a,b)=>Number(b.worker)-Number(a.worker)||a.process.localeCompare(b.process)||a.kind.localeCompare(b.kind)||a.name.localeCompare(b.name,undefined,{numeric:true}));
  return {runId,rank,name,source,annotated,origin,duration,lanes:ordered,byId,tasks:[...byId.values()],events,flowEvents:events.filter(e=>['s','t','f'].includes(e.ph)),counterEvents:events.filter(e=>e.ph==='C'),warnings:[...(valid.length!==events.filter(e=>e.ph==='X').length?['已忽略无效时间事件']:[]),...(overlapLanes?[overlapLanes+' 条泳道含重叠事件；占用按区间并集统计']:[]) ]};
 }
 function gaps(lanes,start,end,min=80){
  const merged=intervals(lanes.flatMap(l=>l.tasks),start,end),out=[];let prev=start;
  for(const [a,b] of merged){if(a-prev>=min)out.push([prev,a]);prev=Math.max(prev,b);}if(end-prev>=min)out.push([prev,end]);return out;
 }
 global.TraceModel={parse,occupied,intervals,gaps};
})(typeof window==='undefined'?globalThis:window);
