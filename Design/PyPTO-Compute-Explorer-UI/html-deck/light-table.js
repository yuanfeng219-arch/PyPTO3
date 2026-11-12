/* The grid is read-only. Only the active presentation surface participates in flight. */
window.createDeckLightTable = function ({entries, page, formatPage, showPage, suspend, rendererUrl, session, targetOrigin}) {
  const overview = document.querySelector('#light-table');
  const grid = document.querySelector('#light-table-grid');
  const toggle = document.querySelector('#light-table-toggle');
  const closeButton = document.querySelector('#light-table-close');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const presentation = [
    document.querySelector('body > main'),
    document.querySelector('#deck-nav'),
    document.querySelector('#presenter-panel'),
  ];
  let tiles;
  let renderer, rendererReady, requested, timer;
  const remaining = new Set(entries.keys());
  const protocol = 'pyptoLightTable';

  function disposeRenderer() {
    clearTimeout(timer);
    renderer?.remove();
    renderer = undefined;
    rendererReady = false;
    requested = undefined;
  }

  function pump() {
    if (!remaining.size) { disposeRenderer(); return; }
    if (phase() !== 'open' || requested !== undefined) return;
    if (!renderer) {
      const url = new URL(rendererUrl);
      url.searchParams.set('lightTable','1');
      renderer = document.createElement('iframe');
      renderer.className = 'light-table-renderer';
      renderer.title = '幻灯片缩略图生成';
      renderer.tabIndex = -1;
      renderer.inert = true;
      renderer.setAttribute('aria-hidden','true');
      renderer.src = url.href;
      document.body.append(renderer);
      timer = setTimeout(() => {
        remaining.forEach(index => { tiles[index].status.textContent = '预览加载失败'; });
        disposeRenderer();
      },12000);
    }
    if (!rendererReady) return;

    // Start with the current page, then render the other pages in deck order.
    requested = remaining.has(origin) ? origin : remaining.values().next().value;
    renderer.contentWindow.postMessage({[protocol]:'capture',session,index:requested},targetOrigin);
    timer = setTimeout(() => {
      tiles[requested].status.textContent = '预览加载失败';
      remaining.delete(requested);
      disposeRenderer();
      pump();
    },12000);
  }

  function renderTile(index, html) {
    const {frame,status} = tiles[index];
    const preview = document.createElement('iframe');
    preview.title = entries[index].label;
    preview.className = 'light-table-preview';
    preview.setAttribute('sandbox','allow-same-origin');
    preview.setAttribute('aria-hidden','true');
    preview.tabIndex = -1;
    preview.inert = true;
    preview.addEventListener('load',() => {
      if (preview.contentDocument?.body.firstElementChild) status.remove();
    });
    const viewport = document.createElement('span');
    viewport.className = 'light-table-preview-viewport';
    viewport.append(preview);
    frame.prepend(viewport);
    preview.srcdoc = html;
  }

  addEventListener('message',event => {
    const message = event.data;
    if (!renderer || event.source !== renderer.contentWindow || message?.session !== session ||
        (targetOrigin !== '*' && event.origin !== targetOrigin)) return;
    if (message[protocol] === 'ready') {
      clearTimeout(timer);
      rendererReady = true;
      pump();
    } else if (message[protocol] === 'snapshot' && message.index === requested) {
      clearTimeout(timer);
      const index = requested;
      requested = undefined;
      remaining.delete(index);
      if (typeof message.html === 'string') renderTile(index,message.html);
      else tiles[index].status.textContent = '预览加载失败';
      pump();
    }
  });

  const frameSize = new ResizeObserver(records => {
    for (const {target,contentRect} of records) {
      if (contentRect.width) target.style.setProperty('--preview-scale',contentRect.width / 1600);
    }
  });
  let origin, selected, restoreFocus, restoreInert, resumeAnimations, flight, pendingClose;

  function buildTiles() {
    document.querySelector('#light-table-count').textContent = entries.length + ' slides';
    return entries.map(({label, chapter}, index) => {
      const card = document.createElement('div');
      card.className = 'light-table-card';
      const tile = document.createElement('button');
      tile.type = 'button';
      tile.className = 'light-table-tile';
      const frame = document.createElement('span');
      frame.className = 'light-table-frame';
      const status = document.createElement('span');
      status.className = 'light-table-status';
      status.textContent = '加载预览…';
      frame.setAttribute('aria-hidden','true');
      frame.append(status);
      frameSize.observe(frame);
      const caption = document.createElement('span');
      caption.className = 'light-table-caption';
      for (const value of [formatPage(index + 1), label]) {
        const text = document.createElement('span');
        text.textContent = value;
        caption.append(text);
      }
      tile.title = chapter ? chapter + ' · ' + label : label;
      if (chapter) {
        const section = document.createElement('small');
        section.className = 'light-table-chapter';
        section.textContent = chapter;
        caption.append(section);
      }
      tile.append(caption);
      tile.onclick = () => { if (phase() === 'open') close(index); };
      card.append(tile,frame);
      grid.append(card);
      return {tile, frame, status};
    });
  }

  const phase = () => overview.dataset.phase;
  function select(index, focus = true) {
    selected = Math.max(0, Math.min(tiles.length - 1, index));
    tiles.forEach(({tile}, i) => { tile.tabIndex = i === selected ? 0 : -1; });
    const tile = tiles[selected].tile;
    tile.scrollIntoView({block:'nearest',inline:'nearest'});
    if (focus) tile.focus({preventScroll:true});
  }

  function fly(index, entering) {
    const slide = entries[index].element;
    const frame = tiles[index].frame;
    const full = slide.getBoundingClientRect();
    const thumb = frame.getBoundingClientRect();
    if (motion.matches || !full.width || !thumb.width) return Promise.resolve();

    const base = slide.style.transform;
    const dx = thumb.left + thumb.width / 2 - full.left - full.width / 2;
    const dy = thumb.top + thumb.height / 2 - full.top - full.height / 2;
    const small = `translate(${dx}px, ${dy}px) ${base} scale(${thumb.width / full.width})`;
    frame.dataset.flight = '';
    const frames = [{transform:base}, {transform:small}];
    flight = slide.animate(entering ? frames : frames.reverse(), {
      duration:540, easing:'cubic-bezier(.45,0,.25,1)', fill:'both',
    });
    return flight.finished.catch(() => {}).finally(() => {
      flight.cancel();
      flight = undefined;
      delete frame.dataset.flight;
    });
  }

  async function open() {
    if (!overview.hidden) return;
    origin = page();
    const active = document.activeElement;
    restoreFocus = active?.contentDocument?.activeElement || active;
    restoreInert = presentation.map(element => element.inert);
    resumeAnimations = suspend(entries[origin].element);
    tiles ??= buildTiles();
    tiles.forEach(({tile}, i) => {
      if (i === origin) tile.setAttribute('aria-current', 'page');
      else tile.removeAttribute('aria-current');
    });
    overview.hidden = false;
    overview.dataset.phase = 'opening';
    toggle.setAttribute('aria-expanded', 'true');
    document.body.classList.add('light-table-flight');
    select(origin);
    presentation.forEach(element => { element.inert = true; });

    await fly(origin, true);
    document.body.classList.replace('light-table-flight', 'light-table-open');
    overview.dataset.phase = 'open';
    if (pendingClose) {
      pendingClose = false;
      close(origin);
      return;
    }
    pump();
  }

  async function close(index = origin) {
    if (phase() === 'opening') {
      pendingClose = true;
      flight?.finish();
      return;
    }
    if (phase() !== 'open') return;
    overview.dataset.phase = 'closing';
    select(index, false);
    if (index !== origin) {
      resumeAnimations();
      showPage(index);
      resumeAnimations = suspend(entries[index].element, false);
    }
    document.body.classList.replace('light-table-open', 'light-table-flight');
    await fly(index, false);

    overview.hidden = true;
    delete overview.dataset.phase;
    document.body.classList.remove('light-table-flight');
    toggle.setAttribute('aria-expanded', 'false');
    presentation.forEach((element, i) => { element.inert = restoreInert[i]; });
    resumeAnimations();
    (index === origin && restoreFocus?.isConnected ? restoreFocus : toggle).focus({preventScroll:true});
  }

  function onKey(event) {
    const editable = event.target.closest?.('input,textarea,select,[contenteditable]:not([contenteditable="false"])');
    const shortcut = event.code === 'KeyG' && !editable;
    if (overview.hidden && !shortcut) return false;
    if (event.code === 'Tab' && !overview.hidden) {
      event.preventDefault();
      (document.activeElement === closeButton ? tiles[selected].tile : closeButton).focus({preventScroll:true});
      return true;
    }
    if (!overview.hidden && event.target.closest?.('#light-table button') &&
        (event.code === 'Space' || event.code === 'Enter' && event.target === closeButton)) return true;
    event.preventDefault();
    if (shortcut && !event.repeat) (overview.hidden ? open() : close());
    else if (event.code === 'Escape') close();
    else if (phase() === 'open') {
      const columns = getComputedStyle(grid).gridTemplateColumns.split(' ').length;
      switch (event.code) {
        case 'Enter': if (!event.repeat) close(selected); break;
        case 'ArrowLeft': select(selected - 1); break;
        case 'ArrowRight': select(selected + 1); break;
        case 'ArrowUp': select(selected - columns); break;
        case 'ArrowDown': select(selected + columns); break;
        case 'Home': select(0); break;
        case 'End': select(tiles.length - 1); break;
      }
    }
    return true;
  }

  toggle.onclick = open;
  closeButton.onclick = () => close();
  addEventListener('resize', () => flight?.finish());
  motion.addEventListener('change', () => flight?.finish());
  return {onKey, get active() { return !overview.hidden; }};
};

// A single isolated preview runs the deck's normal rendering pipeline. Its
// output is copied into script-free frames, so grid tiles have no controllers.
window.serveDeckThumbnails = function ({session, targetOrigin, count, select}) {
  const protocol = 'pyptoLightTable';
  const reply = (kind, data = {}) => parent.postMessage({[protocol]:kind,session,...data},targetOrigin);
  // Safari can throttle animation frames in an invisible iframe. Give the
  // slide a layout turn without waiting on hidden-frame paints or image decode.
  const settleLayout = () => new Promise(resolve => setTimeout(resolve,40));

  async function snapshot(slide) {
    const embedded = slide.matches('iframe');
    await settleLayout();
    // Safari may report the outer preview loaded while the first srcdoc is
    // still on its initial about:blank document.
    if (embedded) {
      const deadline = performance.now() + 4000;
      while (!slide.contentDocument?.body?.firstElementChild) {
        if (performance.now() >= deadline) throw new Error('Slide document is empty');
        await settleLayout();
      }
    }
    const doc = embedded ? slide.contentDocument : document;
    const source = embedded ? doc.body : slide;
    const copy = source.cloneNode(true);
    const canvases = copy.querySelectorAll('canvas');
    source.querySelectorAll('canvas').forEach((canvas,i) => {
      const image = doc.createElement('img');
      [...canvas.attributes].forEach(({name,value}) => image.setAttribute(name,value));
      image.style.width = canvas.offsetWidth + 'px';
      image.style.height = canvas.offsetHeight + 'px';
      image.src = canvas.toDataURL();
      canvases[i].replaceWith(image);
    });
    copy.querySelectorAll('script').forEach(script => script.remove());

    const head = doc.createElement('head');
    const base = doc.createElement('base');
    base.href = doc.baseURI;
    head.append(base);
    doc.head.querySelectorAll('style,link[rel="stylesheet"]').forEach(style => head.append(style.cloneNode(true)));
    const still = doc.createElement('style');
    still.textContent = '*,*::before,*::after{animation:none!important;transition:none!important}html,body{overflow:hidden!important}';
    head.append(still);

    const html = doc.documentElement.cloneNode(false);
    if (embedded) {
      html.append(head,copy);
    } else {
      const body = doc.body.cloneNode(false);
      const stage = slide.parentElement.cloneNode(false);
      stage.append(copy);
      body.append(stage);
      html.append(head,body);
    }
    return '<!doctype html>' + html.outerHTML;
  }

  addEventListener('message',async event => {
    const message = event.data;
    if (event.source !== parent || (targetOrigin !== '*' && event.origin !== targetOrigin) ||
        message?.[protocol] !== 'capture' || message.session !== session ||
        !Number.isInteger(message.index) || message.index < 0 || message.index >= count) return;
    const index = message.index;
    try {
      const html = await snapshot(select(index));
      reply('snapshot',{index,html});
    } catch (error) {
      console.warn('Slide thumbnail capture failed',error);
      reply('snapshot',{index});
    }
  });
  if (document.readyState === 'complete') reply('ready');
  else addEventListener('load',() => reply('ready'),{once:true});
};
