(function (global) {
  'use strict';

  function borrowHardware(host) {
    // Read P01's source, not its live iframe. Its DOM and styling remain independent.
    const source = document.querySelector('main > [data-note-index="opening-context"]').dataset.source;
    const parsed = new DOMParser().parseFromString(source,'text/html');
    const rules = parsed.querySelector('style').textContent;
    const sheet = new CSSStyleSheet();
    sheet.replaceSync(rules);
    const selectors = /^\.(device-frame|chip-frame|die-frame|hw-entity|die-label|core-tiles|core-tile)\b/;
    const style = document.createElement('style');
    style.textContent = [...sheet.cssRules].filter(rule => selectors.test(rule.selectorText ?? ''))
      .map(rule => `${rule.selectorText.split(',').map(selector =>
        `:where([data-p05-hardware]) ${selector.trim()}`).join(',')} {${rule.style.cssText}}`).join('\n');
    host.prepend(style);
    const device = parsed.querySelector('.device-frame');
    const [die] = device.querySelectorAll('.die-frame');
    die.dataset.p05Die = '';
    return device;
  }

  function mount(host,model,{reducedMotion}) {
    const {blocks} = model.constants;
    host.innerHTML = `
      <div class="p05-intro-hardware">
        <small>DIE · 物理计算资源</small>
        <h3>3 × 8 = ${blocks} AI Cores</h3>
        <div data-p05-device></div>
        <div class="p05-mapping-rule">
          <code>Block ID = Task ID % ${blocks}</code>
          <span>Task 0、24 同属逻辑 Block 00；物理落位以运行记录为准。</span>
        </div>
      </div>`;
    host.querySelector('[data-p05-device]').append(borrowHardware(host));
    const rule = host.querySelector('.p05-mapping-rule code');
    const example = host.querySelector('.p05-mapping-rule span');
    const operatorRule = rule.textContent;
    const operatorExample = example.textContent;
    const motion = () => !reducedMotion.matches && document.body.dataset.deckRole === 'deck';
    let flight;

    function setMode(mode) {
      const isPhoto = mode === 'photo';
      rule.textContent = isPhoto ? `${blocks} 个工作小组对应 ${blocks} 个逻辑 Block` : operatorRule;
      example.textContent = isPhoto ? '0 号和 24 号工单分到同一个工作小组。' : operatorExample;
    }

    function cancelFlight() {
      if (!flight) return;
      const {matrix,animations} = flight;
      flight = undefined;
      animations.forEach(animation => animation.cancel());
      host.hidden = true;
      matrix.inert = false;
    }

    function handoff(matrix,renderAssignment) {
      const fromWidth = matrix.getBoundingClientRect().width;
      renderAssignment();
      if (!motion()) return;
      const toWidth = matrix.getBoundingClientRect().width;
      if (!fromWidth || !toWidth) return;

      host.hidden = false;
      matrix.inert = true;
      const duration = 560;
      const animations = [
        matrix.animate([{width:`${100 * fromWidth / toWidth}%`},{width:'100%'}],
          {duration,easing:'cubic-bezier(.22,.68,.22,1)',fill:'both'}),
        host.animate([{opacity:1},{opacity:0}],
          {duration:duration*.72,easing:'ease-out',fill:'both'})
      ];
      const current = flight = {matrix,animations};
      Promise.allSettled(animations.map(animation => animation.finished)).then(() => {
        if (flight === current) cancelFlight();
      });
    }

    return {setMode,handoff,cancelFlight};
  }

  global.PtoDeckHardwareStage = {mount};
})(window);
