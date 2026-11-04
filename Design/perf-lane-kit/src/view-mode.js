export const FEATURES=['tail','parallel','imbalance','underfill','fragmented','setup','bubble'];
export function parseViewParams(search){
 const q=new URLSearchParams(search),view=q.get('view')||'combined',feature=q.get('feature')||'none';
 if(!['combined','critical','path','lane'].includes(view))return {error:`未知视图模式：${view}`};
 if(!['none',...FEATURES].includes(feature))return {error:`未知 Task 特征：${feature}`};
 if((view==='critical'||view==='path')&&feature!=='none')return {error:'关键路径独立视图无法同时显示 Task 特征图层'};
 const result={view,feature,analysis:q.get('analysis')==='1'};
 if(q.get('base')==='1'&&view==='combined')result.base=true;
 return result;
}

export function pathLaunches(rank){
 const ids=new Set(), taskIds=new Set();
 for(const p of rank.criticalPath||[]){for(const id of p.ids||[])ids.add(String(id));if(p.taskId!=null)taskIds.add(String(p.taskId));}
 return rank.launches.filter(t=>ids.has(String(t.id))||(t.taskId!=null&&taskIds.has(String(t.taskId))));
}
