/* Regression for a mock URL showing the real-evidence empty state. */
const {chromium}=require('playwright'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'chrome'});
 const viewport={width:Number(process.env.GAP_VIEWPORT_WIDTH)||1440,height:Number(process.env.GAP_VIEWPORT_HEIGHT)||900};
 const page=await browser.newPage({viewport}),errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 try{
  const url=(process.env.GAP_DEMO_URL||'http://127.0.0.1:8766/Design/deepseek-gap-investigator/index.html').split('?')[0];
  const waitForGraph=async({mockMode=true}={})=>{
   await page.waitForSelector('#simulationPassGraph [data-ir-side="after"] .ir-node.is-added');
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
   if(mockMode){assert.match(await page.title(),/模拟 Pass Diff/);assert.equal(new URL(page.url()).searchParams.get('demo'),'compiler');}
   assert.equal(await page.locator('#simulationPassGraph .ir-data-edge[data-source="sim-gm-pipe-buffer"][data-target="after-submit"]').count(),1);
   assert.equal(await page.locator('#simulationPassGraph .ir-data-edge[data-source="after-submit"][data-target="after-publish-tid"]').count(),1);
   assert.doesNotMatch(await page.locator('#stepContent').innerText(),/尚未定位到相关/);
  };
  await page.goto(url+'?v=24&demo=compiler');await waitForGraph();
  assert.equal(await page.locator('[data-action="simulation-mode"]').count(),0);assert.doesNotMatch(await page.locator('.case-header').innerText(),/待取证/);
  const positions=await page.locator('#simulationPassGraph').evaluate(root=>{
   const box=id=>{const el=root.querySelector(`[data-node-id="${id}"]`),r=el.getBoundingClientRect();return {left:r.left,top:r.top,right:r.right,bottom:r.bottom};};
   return Object.fromEntries(['before-inputs','before-submit','before-publish-tid','before-wait','after-inputs','after-submit','after-publish-tid','after-wait','sim-gm-pipe-buffer'].map(id=>[id,box(id)]));
  });
  for(const side of ['before','after']){
   const tops=['inputs','submit','publish-tid','wait'].map(id=>positions[`${side}-${id}`].top);
   assert.ok(Math.max(...tops)-Math.min(...tops)<2,`${side} stable chain should share one baseline`);
  }
  const cardGeometry=await page.locator('#simulationPassGraph').evaluate(root=>{
   const cards=[...root.querySelectorAll('.ir-node:not([data-node-id="sim-gm-pipe-buffer"]) .node-card')].map(el=>el.getBoundingClientRect().height);
   const labels=[...root.querySelectorAll('.op-pill-name,.tensor-rect-name')].map(el=>({text:el.textContent.trim(),fits:el.scrollWidth<=el.clientWidth+1&&el.scrollHeight<=el.clientHeight+1}));
   return {cards,labels};
  });
  assert.ok(Math.max(...cardGeometry.cards)-Math.min(...cardGeometry.cards)<2,`stable Pass nodes should have equal rendered heights (${JSON.stringify(cardGeometry.cards)})`);
  assert.ok(cardGeometry.labels.every(label=>label.fits),`Pass node labels should be fully visible (${JSON.stringify(cardGeometry.labels)})`);
  assert.ok(positions['sim-gm-pipe-buffer'].top>positions['after-inputs'].bottom,'added GM buffer should occupy a separate lower branch');
  const graphVisibility=await page.locator('#simulationPassGraph').evaluate(root=>{const step=root.closest('.step-scroll').getBoundingClientRect(),graph=root.getBoundingClientRect(),drawer=document.querySelector('#investigationDrawer').getBoundingClientRect(),caseHeader=document.querySelector('.case-header').getBoundingClientRect(),workflow=document.querySelector('.workflow').getBoundingClientRect(),heading=root.parentElement.querySelector('.step-heading').getBoundingClientRect(),legend=root.parentElement.querySelector('.simulation-pass-legend').getBoundingClientRect();return {drawerTop:drawer.top,drawerHeight:drawer.height,caseHeaderHeight:caseHeader.height,workflowHeight:workflow.height,stepTop:step.top,stepBottom:step.bottom,headingHeight:heading.height,legendHeight:legend.height,graphTop:graph.top,graphHeight:graph.height,graphBottom:graph.bottom};});
  assert.ok(graphVisibility.graphBottom<=graphVisibility.stepBottom+1,`Pass graph should be visible without scrolling the investigation step (${JSON.stringify(graphVisibility)})`);
  await page.goto(url+'?v=26');await page.waitForFunction(()=>GapApp.state.model?.tasks.length===9480);assert.equal(new URL(page.url()).searchParams.has('demo'),false);assert.match(await page.title(),/真实证据/);
  await page.locator('#markerActions .diagnostic-range__tag:not([hidden])').first().press('Enter');await page.locator('#caseList [data-preset="a2a"]').click();await page.locator('[data-step="4"]').click();await waitForGraph({mockMode:false});assert.equal(await page.locator('#stepContent [data-action="simulation-open-pass"]').count(),0);
  await page.reload();await page.waitForFunction(()=>GapApp.state.model?.tasks.length===9480);
  await page.locator('#markerActions .diagnostic-range__tag:not([hidden])').first().press('Enter');
  await page.locator('#caseList [data-preset="a2a"]').click();await page.locator('[data-step="4"]').click();
  assert.equal(await page.evaluate(()=>GapApp.simulation.active()),false);await waitForGraph({mockMode:false});
  await page.locator('#caseList [data-preset="token"]').click();assert.equal(new URL(page.url()).searchParams.has('demo'),false);assert.match(await page.title(),/真实证据/);
  await page.locator('#caseList [data-preset="a2a"]').click();await page.locator('[data-step="4"]').click();await waitForGraph({mockMode:false});
  await page.locator('[data-step="3"]').click();await page.locator('[data-step="4"]').click();await waitForGraph({mockMode:false});
  await page.screenshot({path:'/tmp/deepseek-pass-entry-v24.png'});
  assert.deepEqual(errors,[]);console.log(`PASS ${viewport.width}x${viewport.height}: inline Pass diff, real Pass basis, URL/title mode sync, reload, case switch, connected causality`);
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
