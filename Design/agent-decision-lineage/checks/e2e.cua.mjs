/** Run inside cua_repl with its documented tab/browser handles.
 * No page-state mutation or hidden app API; clicks, keyboard, scroll, DOM reads.
 */
export async function runReviewRegression({tab,browser,width,writeFile,directory}) {
 const results=[];
 const check=(name,ok,detail='')=>{results.push({name,passed:!!ok,detail});if(!ok)throw new Error(name+': '+detail)};
 const click=async name=>tab.playwright.getByRole('button',{name,exact:true}).click();
 const node=async id=>tab.playwright.locator('#graph [data-node-id="'+id+'"]').click();
 const text=async sel=>tab.playwright.locator(sel).innerText();
 const snap=async()=>tab.getAXState({emit:false});
 const shot=async name=>{await snap();await writeFile(directory+'/'+name+'-'+width+'.png',await tab.screenshot({fullPage:false}));};
 try {
 await (await browser.capabilities.get('viewport')).set({width,height:900});await tab.reload();await snap();
 check('Overview default',(await text('#scope')).includes('Overview'));
 await shot('audit-overview');
 // Keyboard enters the same public node action and moves focus to its heading.
 await tab.playwright.locator('#graph [data-node-id="d-precision"]').press('Enter');await snap();
 check('keyboard opens Precision',(await text('#detail')).includes('Precision Decision'));
 check('focus follows Inspector',await tab.playwright.evaluate(()=>document.activeElement?.id==='inspector-heading'));
 const bounds=await tab.playwright.evaluate(()=>{const a=document.querySelector('#inspector').getBoundingClientRect(),m=document.querySelector('main').getBoundingClientRect();return {a:a.top,m:m.top}});
 check('auto scroll to Inspector',bounds.a>=bounds.m-3&&bounds.a<bounds.m+60,JSON.stringify(bounds));
 await click('查看完整溯源路径');check('full trace enabled',(await tab.playwright.locator('#trace').getAttribute('aria-pressed'))==='true');await click('仅看直接关联');check('direct trace restored',(await tab.playwright.locator('#trace').getAttribute('aria-pressed'))==='false');
 await click('Exp(x−max) FP16 src → FP32 result');
 check('precision code mapping',(await text('.decision-object')).includes('vf.exp_sub'));
 check('precision actual line mapping',(await text('.decision-object')).includes('L135'));
 await click('下一步 → Single Tile');await tab.playwright.locator('.logical-tensor button').last().click();
 check('tail tile valid rows',(await text('.decision-object')).includes('[1,200]'));
 await tab.playwright.locator('.ub-bar button').nth(3).click();
 check('slot address mapping',(await text('.decision-object')).includes('0x24000 ≤ addr < 0x30000'));
 await click('TILE_ROWS = 3');check('budget verified',(await text('.decision-object')).includes('192 KiB'));
 await shot('audit-design');
 await click('下一步 → UB 资源');await click('下一步 → Scheduling');await click('Pass 2 · Exp + Sum');
 check('Scheduling source',(await text('.decision-object')).includes('FP32 reduce_sum'));
 await click('下一步 → 生成 Kernel');check('Kernel viewer',await tab.playwright.getByLabel('Kernel 源码，可独立滚动',{exact:true}).isVisible());
 const pre=tab.playwright.getByLabel('Kernel 源码，可独立滚动',{exact:true});await pre.press('PageDown');await snap();
 check('code internal scroll',await pre.evaluate(e=>e.scrollTop>0));
 await click('下一步 → 正确性 8/8');
 check('all eight correctness rows',(await tab.playwright.locator('#detail table tr').count())===9);
 check('correctness source',(await text('#detail')).includes('2.77e-06'));
 await click('下一步 → Baseline');check('Baseline comparison',(await text('#detail')).includes('Performance Comparison'));
 const expectations=[['[1,128]','+5.48%'],['[4,2048]','-14.83%'],['[32,4096]','-16.54%']];
 for(const [name,value]of expectations){await click(name);check('Shape '+name,(await text('.experiment-workspace')).includes(value));}
 await tab.playwright.locator('.experiment-workspace summary').click();
 check('formal sample sets',(await text('.experiment-workspace details')).includes('5.834, 5.926, 6.130'));
 await shot('audit-experiment');
 await click('下一步 → 性能线索');check('Finding identity',(await text('#inspector-heading')).includes('重复 update_mask'));
 await click('下一步 → 优化假设');check('Hypothesis identity',(await text('#inspector-heading')).includes('优化假设'));
 await click('下一步 → 代码修改');await click('尾寄存器 pl.range');
 check('before/after files',(await text('.code-diff')).includes('test_softmax.py.bak_stage5'));
 check('tail source mapping',(await text('.code-diff')).includes('n_tail_iters'));
 await click('Runtime if · × Rejected');check('reported compiler failure',(await text('.experiment-workspace')).includes('__inline_0_tail_mreg_fp16_2'));
 await click('Empty-tail-loop · ✓ Accepted · 属于 Mask Lifting');check('component not independent',(await text('.experiment-workspace')).includes('没有独立性能归因'));
 for(const name of ['TILE_ROWS Adjustment · ○ Not Attempted','Exp Result Reuse · ○ Not Attempted','Pass Fusion · ○ Not Attempted']){await click('Optimization Journey');await click(name);check(name,(await text('.evidence-labels')).includes('Not Attempted'));}
 await click('Optimization Journey');await click('8/8 PASS');check('PASS maps verification',(await text('#inspector-heading')).includes('编译与正确性'));
 await click('下一步 → Baseline');await click('Optimization Journey');await click('正式采集 / Final');await click('Evidence & Decision');
 await tab.playwright.locator('.experiment-workspace summary').click();check('PMU table',(await text('.profiling-table')).includes('121.50%'));
 check('sampling scope distinction',(await text('.experiment-workspace')).includes('正式基线中位数=5.926'));
 await click('下一步 → 停止审查');
 check('stop missing evidence',(await text('.stop-review')).includes('preflight'));
 check('stop derived evidence',(await text('.stop-review')).includes('not_applicable_with_dag'));
 // Wheel over the IDE workarea, not the window, to reveal the entire stop section.
 await tab.scroll([width-150,720],'down',3);await snap();
 const bottom=await tab.playwright.evaluate(()=>({end:document.querySelector('#inspector').getBoundingClientRect().bottom,main:document.querySelector('main').getBoundingClientRect().bottom,top:document.querySelector('main').scrollTop}));
 check('workarea scroll reaches Inspector end',bottom.end<=bottom.main+2&&bottom.top>0,JSON.stringify(bottom));
 await shot('audit-stop');
 await tab.playwright.locator('#review-context nav button').last().click();await snap();
 check('return keeps Stop selection',await tab.playwright.locator('#graph [data-node-id="d-stop"]').getAttribute('class').then(c=>c.includes('selected')));
 check('return Optimize scope',(await text('#scope')).includes('Optimize'));
 await shot('audit-optimize');
 await click('Focus Selection');await snap();check('Focus changes viewBox',(await tab.playwright.locator('#graph').getAttribute('viewBox'))!=='1035 95 653 405');
 await click('Fit to View');check('Fit current scope',(await tab.playwright.locator('#graph').getAttribute('viewBox'))==='1035 95 653 405');
 await click('Design');check('Design clears inspector',await tab.playwright.locator('#inspector').getAttribute('hidden')!==null);
 check('Design clears selection',await tab.playwright.locator('#graph .node.selected').count()===0);
 await node('d-precision');await click('Precision Decision');check('Design regression',(await text('#detail')).includes('FP16 max'));
 await tab.playwright.locator('#review-context nav button').last().click();await click('Overview');check('Overview clears selection',await tab.playwright.locator('#graph .node.selected').count()===0);
 await node('d-precision');await click('关闭');check('close restores graph focus',await tab.playwright.evaluate(()=>document.activeElement?.id==='graph'));
 await click('返回全局');check('global viewBox',(await tab.playwright.locator('#graph').getAttribute('viewBox'))==='0 0 1688 500');
 const sizes=await tab.playwright.evaluate(()=>({width:innerWidth,page:document.documentElement.scrollWidth,main:document.querySelector('main').clientWidth,scroll:document.querySelector('main').scrollWidth}));
 check('no horizontal overflow',sizes.width===sizes.page&&sizes.main===sizes.scroll,JSON.stringify(sizes));
 const errors=await tab.dev.logs({levels:['error','warn'],limit:30});check('Console clean',errors.length===0,JSON.stringify(errors));
 } catch(error){results.push({name:'run failure',passed:false,detail:String(error)});}
 await writeFile(directory+'/regression-'+width+'.json',JSON.stringify({width,height:900,results},null,2));
 return {width,passed:results.every(r=>r.passed),checks:results.length,failures:results.filter(r=>!r.passed)};
}
