document.body.insertAdjacentHTML('afterbegin', `
<main id="frame" class="pto-ide-frame" data-host="standalone">
<div id="ideWorkarea" class="pto-ide-frame__workarea">
<article id="story" data-p09 tabindex="-1" aria-label="执行层级下钻">
  <header class="headline" data-deck-title="深入执行细节，仍能看见来路">
    <p class="subject">Paged Attention <span>／</span> Pipeline</p>
    <h1>深入执行细节，仍能看见来路</h1>
    <h2 class="strategy-subtitle" hidden>逐层深入执行细节，同时保留观察上下文</h2>
  </header>
  <div class="workspace">
    <section class="pipeline" aria-label="Pipeline">
      <nav class="path" aria-label="当前执行路径">
        <button type="button" class="btn btn-ghost" id="device">Device</button>
        <button type="button" class="btn btn-ghost" id="block"></button>
        <button type="button" class="btn btn-ghost" id="task" hidden></button>
        <span id="tick" hidden></span>
      </nav>
      <div class="viz2 canvas" id="canvas">
        <div id="view"></div>
        <div id="block-detail" hidden>
          <div class="phases"><span>执行块</span><span>准备阶段</span><span>同步</span><span>注意力任务与输出</span></div>
          <div class="block-reading">
            <p id="block-facts"></p>
          </div>
        </div>
        <p id="error" role="alert" hidden>执行视图加载失败，请重新打开此页面。</p>
      </div>
      <p id="legend" class="pipeline-legend"></p>
    </section>
    <aside class="hierarchy" aria-label="执行层级">
      <h2>执行层级</h2>
      <div class="p09-axis" role="group" aria-label="观察粒度">
        <button type="button" data-p09-level="device">Device</button>
        <button type="button" data-p09-level="ai-core">AI Core</button>
        <button type="button" data-p09-level="iteration">Iteration</button>
      </div>
      <div class="layer-context" aria-hidden="true">
        <span id="device-context"></span><span id="block-context"></span><span id="iteration-context"></span>
      </div>
    </aside>
  </div>
  <footer class="story-controls">
    <p id="caption" role="status"></p>
    <nav aria-label="演讲步骤">
      <button type="button" class="btn btn-ghost" id="up">返回上级</button>
      <button type="button" class="btn btn-ghost" id="prev">上一步</button>
      <button type="button" class="btn btn-solid" id="next">下一步</button>
      <button type="button" class="btn btn-ghost" id="replay">重播</button>
      <button type="button" class="btn btn-ghost" id="reset">重置</button>
    </nav>
  </footer>
</article>
</div>
</main>
`);
