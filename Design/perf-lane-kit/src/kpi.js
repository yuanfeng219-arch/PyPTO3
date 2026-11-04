import {concurrency} from './analysis.js';

// All times are microseconds; unavailable inputs remain null rather than becoming zero.
export function calculateKpis(rank) {
  const tasks=rank.launches, windowUs=rank.windowUs, cores=rank.cores;
  const sum=(items,key)=>items.length && items.every(x=>Number.isFinite(x[key])) ? items.reduce((n,x)=>n+x[key],0) : null;
  const computeUs=sum(tasks,'computeUs'), setupUs=sum(tasks,'setupUs');
  const byType={};
  for(const type of ['AIC','AIV']) {
    const count=cores.filter(c=>c.ct===type).length, group=tasks.filter(t=>t.ct===type), total=group.length?sum(group,'computeUs'):0;
    byType[type]=count && total!==null ? total/(windowUs*count) : null;
  }
  const path=rank.criticalPath, timed=Array.isArray(path)?path.filter(p=>p.onCore&&Number.isFinite(p.startUs)&&Number.isFinite(p.endUs)):[];
  const pathUs=timed.length ? Math.max(...timed.map(p=>p.endUs))-Math.min(...timed.map(p=>p.startUs)) : null;
  let singleUs=null;
  if(pathUs>0 && timed.every(p=>Number.isFinite(p.blockNum))) {
    singleUs=0; let end=-Infinity;
    for(const p of timed.filter(p=>p.blockNum===1).sort((a,b)=>a.startUs-b.startUs)) {
      singleUs+=Math.max(0,p.endUs-Math.max(p.startUs,end)); end=Math.max(end,p.endUs);
    }
  }
  let lowUs=0, idleUs=0;
  for(const segment of concurrency(rank)) {
    const duration=Math.max(0,Math.min(windowUs,segment.end)-Math.max(0,segment.start));
    if(segment.count<=2) lowUs+=duration;
    if(segment.count===0) idleUs+=duration;
  }
  return {windowUs,taskCount:tasks.length,coreCount:cores.length,pathUs,singleUs,
    singleShare:singleUs!==null?singleUs/pathUs:null,
    occupancy:computeUs!==null&&cores.length?computeUs/(windowUs*cores.length):null,
    byType,lowUs,idleUs,lowShare:lowUs/windowUs,computeUs,setupUs,
    setupShare:setupUs!==null&&computeUs>0?setupUs/computeUs:null};
}
