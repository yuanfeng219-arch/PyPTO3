/* Copy edits are stored in the deck HTML. Story renderers still own their markup and state. */
const DeckLocalEditor = (() => {
  const chrome = 'script,style,template,pre,code,input,textarea,select,output,.deck-mast,[data-deck-control-source],.controls,.transport,.local-navigation,[data-opening-controls]';
  const copy = 'h1,h2,h3,h4,h5,h6,p,li,dt,dd,figcaption,th,td,summary,span,strong,b,em,small,label,div,button,text,tspan,[data-p11-copy]';

  function mountSurface(root, onEdit) {
    const doc = root.ownerDocument;
    const previous = new WeakMap();
    let entries = [];
    let enabled = false;
    let input;
    const style = doc.createElement('style');
    style.textContent = `
      .deck-copy-enabled [data-p11-front] [data-p11-copy] {pointer-events:bounding-box;cursor:text}
      .deck-copy-hover {outline:1px dashed #819efe;outline-offset:2px}
      .deck-copy-input {position:fixed;z-index:2147483647;box-sizing:border-box;margin:0;padding:2px 6px;
        color:#eff1f4;background:#17191f;border:1px solid #819efe;border-radius:4px}
    `;
    doc.head.append(style);

    function textNodes(element) {
      return [...element.childNodes].filter(node => node.nodeType === 3);
    }

    function locator(element) {
      if (element.dataset.p11Copy) return element.dataset.p11Copy;
      const path = [];
      for (let node = element; node !== root; node = node.parentElement) {
        path.unshift([node.localName,[...node.parentElement.children].indexOf(node)]);
      }
      return path;
    }

    function locate(path) {
      if (typeof path === 'string') return [...root.querySelectorAll('[data-p11-copy="' + path + '"]')];
      let element = root;
      for (const [tag,index] of path) {
        element = element.children[index];
        if (element?.localName !== tag) return [];
      }
      return [element];
    }

    function apply() {
      for (const {path,part,original,value} of entries) {
        for (const element of locate(path)) {
          const node = textNodes(element)[part];
          if (!node || node.data === value) continue;
          const last = previous.get(node);
          const source = last?.value === node.data ? last.original : node.data;
          if (source !== original) continue;
          previous.set(node,{original,value});
          node.data = value;
        }
      }
    }

    // Captions may be recreated as the story advances. Attributes and animation frames
    // that only change style do not trigger this observer.
    const observer = new MutationObserver(apply);
    function sync(nextEnabled, edits) {
      enabled = nextEnabled;
      root.classList.toggle('deck-copy-enabled',enabled);
      if (!enabled) {
        hovered?.classList.remove('deck-copy-hover');
        hovered = null;
        input?.blur();
      }
      observer.disconnect();
      entries = Object.values(edits);
      apply();
      if (entries.length) observer.observe(root,{subtree:true,childList:true,characterData:true});
    }

    function editable(event) {
      if (!enabled) return;
      const element = event.target.closest?.(copy);
      if (!element || !root.contains(element) || element.closest(chrome) ||
          element.closest('[hidden],[inert]') || !textNodes(element).some(node => node.data.trim())) return;
      if (!element.hasAttribute('data-p11-copy') && element.closest('a,button,[role="button"]')) return;
      return element;
    }

    let hovered;
    root.addEventListener('pointermove',event => {
      const next = editable(event);
      if (next === hovered) return;
      hovered?.classList.remove('deck-copy-hover');
      hovered = next;
      hovered?.classList.add('deck-copy-hover');
    });
    root.addEventListener('pointerleave',() => {
      hovered?.classList.remove('deck-copy-hover');
      hovered = null;
    });

    root.addEventListener('dblclick',event => {
      const element = editable(event);
      if (!element) return;
      const nodes = textNodes(element);
      const caret = doc.caretRangeFromPoint?.(event.clientX,event.clientY)?.startContainer;
      const node = nodes.includes(caret) && caret.data.trim() ? caret : nodes.find(item => item.data.trim());
      const path = locator(element);
      const part = nodes.indexOf(node);
      event.preventDefault();
      event.stopPropagation();
      const bounds = doc.createRange();
      bounds.selectNode(node);
      const textRect = bounds.getBoundingClientRect();
      const rect = textRect.width ? textRect : element.getBoundingClientRect();
      const computed = doc.defaultView.getComputedStyle(element);
      const field = doc.createElement('input');
      field.type = 'text';
      field.className = 'deck-copy-input';
      field.setAttribute('aria-label','编辑文字');
      const before = node.data;
      const last = previous.get(node);
      const original = last?.value === before ? last.original : before;
      field.value = before.trim();
      const width = Math.min(Math.max(180,rect.width + 18),doc.documentElement.clientWidth - 16);
      Object.assign(field.style,{
        left:Math.max(8,Math.min(rect.left - 4,doc.documentElement.clientWidth - width - 8))+'px',
        top:Math.max(0,rect.top - 3)+'px',width:width+'px',height:Math.max(26,rect.height + 6)+'px',
        fontFamily:computed.fontFamily,fontSize:Math.max(13,rect.height * .75)+'px',
      });
      function finish(commit) {
        if (input !== field) return;
        input = null;
        const content = field.value.trim();
        field.remove();
        if (!commit || !content || content === before.trim()) return;
        const value = before.match(/^\s*/)[0] + content + before.match(/\s*$/)[0];
        onEdit({path,part,original,value});
      }
      field.addEventListener('blur',() => finish(true));
      field.addEventListener('keydown',event => {
        event.stopPropagation();
        if (event.isComposing || event.keyCode === 229) return;
        if (!['Enter','Escape'].includes(event.key) &&
            !((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's')) return;
        event.preventDefault();
        finish(event.key !== 'Escape');
      });
      doc.body.append(field);
      input = field;
      field.focus({preventScroll:true});
      field.select();
    },true);
    return {sync};
  }

  function mountDeck({slides,role,page,bindingStore,sendP9,isPresenting = () => false}) {
    const source = document.querySelector('#deck-copy-edits');
    const edits = JSON.parse(source.textContent);
    const saved = new Map();
    const pending = new Map();
    const surfaces = new Map();
    const p9Index = slides.findIndex(slide => slide.hasAttribute('data-p09'));
    const enabledButton = document.querySelector('#deck-edit-toggle');
    const saveButton = document.querySelector('#deck-edit-bind');
    const status = document.querySelector('#deck-edit-status');
    const fileName = decodeURIComponent(location.pathname.split('/').pop());
    const bindingKey = 'p11-deck:' + location.pathname;
    const canWrite = typeof window.showOpenFilePicker === 'function';
    const block = /(<script type="application\/json" id="deck-copy-edits">)([\s\S]*?)(<\/script>)/;
    let enabled = false;
    let handle, busy = false;

    function syncPage(index) {
      const slide = slides[index];
      if (index === p9Index) {
        sendP9('copy-state',{enabled:enabled && !isPresenting() && index === page(),edits:edits[index] ?? {}});
        return;
      }
      const embedded = slide.matches('iframe');
      const root = embedded ? slide.contentDocument?.querySelector('main, #slide') : slide;
      if (!root) return;
      if (index !== page() && !edits[index] && !surfaces.has(index)) return;
      if (surfaces.get(index)?.root !== root) {
        surfaces.set(index,{root,editor:mountSurface(root,edit => stage(index,edit))});
      }
      surfaces.get(index).editor.sync(enabled && !isPresenting() && index === page(),edits[index] ?? {});
    }

    function refresh() {
      slides.forEach((_,index) => syncPage(index));
    }

    function stage(index,edit) {
      const {path,part,original,value} = edit;
      const key = JSON.stringify([path,part,original]);
      const entry = {path,part,original,value};
      const changes = edits[index] ??= {};
      const id = JSON.stringify([index,key]);
      if (!saved.has(id)) saved.set(id,changes[key]?.value);
      changes[key] = entry;
      pending.set(id,{index,key,entry});
      syncPage(index);
      if (handle) void write();
      else setStatus('待保存',canWrite ? '关联 Deck 原文件后自动保存' : '请在 Chrome 或 Edge 中打开以写入原文件');
    }

    function setStatus(label,description = label) {
      status.textContent = label;
      status.title = description;
    }

    function updateControls() {
      enabledButton.setAttribute('aria-pressed',String(enabled));
      enabledButton.title = enabled ? '双击幻灯片文字开始编辑，按 Enter 确认' : '开启编辑模式';
      saveButton.hidden = Boolean(handle) || !canWrite;
      saveButton.disabled = busy;
    }

    function readBlock(text) {
      const match = text.match(block);
      if (!match) throw new Error('请选择当前 Deck 的原 HTML 文件');
      return {match,data:JSON.parse(match[2])};
    }

    async function write() {
      if (!handle || !pending.size || busy) return;
      busy = true;
      updateControls();
      setStatus('保存中…');
      const snapshot = new Map(pending);
      let stream;
      try {
        const text = await (await handle.getFile()).text();
        const {match,data} = readBlock(text);
        for (const [id,{index,key,entry}] of snapshot) {
          const changes = data[index] ??= {};
          if (changes[key]?.value !== saved.get(id) && changes[key]?.value !== entry.value) {
            throw new Error('原文件中的这段文字已修改，请先复制当前改动，再重新打开 Deck');
          }
          changes[key] = entry;
        }
        const json = JSON.stringify(data,null,2).replace(/</g,'\\u003c');
        const next = text.slice(0,match.index) + match[1] + '\n' + json + '\n' + match[3] +
          text.slice(match.index + match[0].length);
        if (next !== text) {
          stream = await handle.createWritable({mode:'exclusive'});
          if (await (await handle.getFile()).text() !== text) throw new Error('保存时原文件发生变化，请重新关联');
          await stream.write(next);
          await stream.close();
        }
        for (const [id,change] of snapshot) {
          saved.set(id,change.entry.value);
          if (pending.get(id) === change) pending.delete(id);
        }
        setStatus('已保存','已保存到 Deck 原文件');
      } catch (error) {
        await stream?.abort().catch(() => {});
        handle = null;
        setStatus('保存失败',error.name === 'Error' ? error.message : '修改仍在页面中，请重新关联原文件');
      } finally {
        busy = false;
        updateControls();
        if (handle && pending.size) void write();
      }
    }

    if (role === 'deck') {
      enabledButton.onclick = () => {
        enabled = !enabled;
        updateControls();
        refresh();
      };
      saveButton.onclick = async () => {
        busy = true;
        updateControls();
        try {
          const [candidate] = await window.showOpenFilePicker({
            id:'pypto-deck',types:[{description:'HTML Deck',accept:{'text/html':['.html']}}],
          });
          if (candidate.name !== fileName) throw new Error('请选择 ' + fileName);
          if (await candidate.requestPermission({mode:'readwrite'}) !== 'granted') {
            setStatus('未授权','需要写入权限才能保存到原文件');
            return;
          }
          readBlock(await (await candidate.getFile()).text());
          handle = candidate;
          setStatus('自动保存','双击幻灯片文字可编辑，确认后自动保存');
          bindingStore('readwrite',store => store.put(candidate,bindingKey)).catch(() => {});
        } catch (error) {
          if (error.name !== 'AbortError') setStatus('关联失败',error.message);
        } finally {
          busy = false;
          updateControls();
          void write();
        }
      };
      addEventListener('beforeunload',event => {
        if (pending.size) {
          event.preventDefault();
          event.returnValue = true;
        }
      });
      setStatus(canWrite ? '未关联' : '无法写入',canWrite ? '双击文字编辑，关联原文件后自动保存' : '请在 Chrome 或 Edge 中打开以写入原文件');
      if (canWrite) bindingStore('readonly',store => store.get(bindingKey)).then(async candidate => {
        if (!candidate || candidate.name !== fileName ||
            await candidate.queryPermission({mode:'readwrite'}) !== 'granted' || busy || handle) return;
        handle = candidate;
        updateControls();
        setStatus('自动保存','双击幻灯片文字可编辑，确认后自动保存');
        void write();
      }).catch(() => {});
      updateControls();
    }

    slides.forEach((slide,index) => {
      if (slide.matches('iframe')) slide.addEventListener('load',() => syncPage(index));
    });
    return {
      refresh,
      editP9(edit) {
        if (role !== 'deck' || isPresenting() || !Array.isArray(edit?.path) ||
            !edit.path.every(step => Array.isArray(step) && typeof step[0] === 'string' && Number.isInteger(step[1])) ||
            !Number.isInteger(edit.part) || typeof edit.original !== 'string' || typeof edit.value !== 'string') return;
        stage(p9Index,edit);
      },
      syncP9() { syncPage(p9Index); },
    };
  }
  return {mountDeck,mountSurface};
})();
