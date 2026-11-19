(function registerPtoMemoryReuseViewer(global) {
  'use strict';

  const KIND_COLORS = {
    resident: '#58a6ff',
    temp: '#3fb950',
    loop: '#f0883e',
  };

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function createDemoSourceFile() {
    const lines = Array.from({ length: 96 }, () => '');
    lines[0] = '#include "kernel_operator.h"';
    lines[1] = 'using namespace AscendC;';
    lines[3] = 'class MatMulAddReluMixAicKernel {';
    lines[4] = 'public:';
    lines[5] = '  __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR y) {';
    lines[6] = '    xGm.SetGlobalBuffer((__gm__ half*)x);';
    lines[7] = '    wGm.SetGlobalBuffer((__gm__ half*)w);';
    lines[8] = '    biasGm.SetGlobalBuffer((__gm__ float*)bias);';
    lines[9] = '    yGm.SetGlobalBuffer((__gm__ half*)y);';
    lines[10] = '  }';
    lines[12] = '  __aicore__ inline void Process(uint32_t tileId) {';
    lines[13] = '    uint32_t offset = tileId * TILE_M * TILE_K;';
    lines[14] = '    uint32_t woff = tileId * TILE_K * TILE_N;';
    lines[16] = '    LoadL1(tileId, woff);';
    lines[17] = '    LoadUbInputs(offset, woff);';
    lines[18] = '    ComputeMatMul();';
    lines[19] = '    AddBiasAndRelu(offset);';
    lines[20] = '    StoreOutput(offset);';
    lines[21] = '  }';
    lines[23] = 'private:';
    lines[24] = '  GlobalTensor<half> xGm;';
    lines[25] = '  GlobalTensor<half> wGm;';
    lines[26] = '  GlobalTensor<float> biasGm;';
    lines[27] = '  GlobalTensor<half> yGm;';
    lines[29] = '  __aicore__ inline void LoadL1(uint32_t tileId, uint32_t woff) {';
    lines[30] = '    LocalTensor<half> wL1 = l1Buf.Get<half>();';
    lines[31] = '    DataCopy(wL1, wGm[woff], {TILE_K, TILE_N, 0, 0});';
    lines[32] = '    l1Buf.EnQue(wL1);';
    lines[33] = '    LocalTensor<half> wReady = l1Buf.DeQue<half>();';
    lines[34] = '  }';
    lines[35] = '  __aicore__ inline void ReloadL1(uint32_t aOff) {';
    lines[36] = '    LocalTensor<half> aL1 = l1Buf.Get<half>();';
    lines[37] = '    DataCopy(aL1, xGm[aOff], {TILE_M, TILE_K, 0, 0});';
    lines[38] = '    l1Buf.EnQue(aL1);';
    lines[39] = '  }';
    lines[40] = '  __aicore__ inline void LoadUbInputs(uint32_t offset, uint32_t woff) {';
    lines[41] = '    LocalTensor<half> xUb = inQueueX.AllocTensor<half>();';
    lines[42] = '    DataCopy(xUb, xGm[offset], {1, blockLen, 0, 0});';
    lines[43] = '    inQueueX.EnQue(xUb);';
    lines[44] = '    xUb = inQueueX.DeQue<half>();';
    lines[45] = '    Mmad(cMatrix, xUb, wUb, mmolParams);';
    lines[46] = '    inQueueX.FreeTensor(xUb);';
    lines[47] = '';
    lines[48] = '    LocalTensor<half> wUb = inQueueW.AllocTensor<half>();';
    lines[49] = '    DataCopy(wUb, wGm[woff], {nBlk, blockLen, 0, 0});';
    lines[50] = '    inQueueW.EnQue(wUb);';
    lines[51] = '    wUb = inQueueW.DeQue<half>();';
    lines[52] = '    inQueueW.FreeTensor(wUb);';
    lines[53] = '  }';
    lines[54] = '  __aicore__ inline void ComputeMatMul() {';
    lines[55] = '    LocalTensor<float> cMatrix = l0cBuf.Get<float>();';
    lines[56] = '    Mmad(cMatrix, xUb, wUb, {TILE_M, TILE_K, TILE_N, true});';
    lines[57] = '    l0cBuf.EnQue(cMatrix);';
    lines[58] = '  }';
    lines[59] = '  __aicore__ inline void LoadBias() {';
    lines[60] = '    LocalTensor<float> biasUb = biasBuf.Get<float>();';
    lines[61] = '    DataCopy(biasUb, biasGm, n);';
    lines[62] = '    pipe_barrier(PIPE_V);';
    lines[63] = '  }';
    lines[64] = '';
    lines[65] = '  __aicore__ inline void AddBiasAndRelu(uint32_t offset) {';
    lines[66] = '    LocalTensor<float> cUb = ubBuf.Get<float>();';
    lines[67] = '    DataCopy(cUb, cMatrix, {TILE_M, TILE_N, 0, 0});';
    lines[68] = '    Add(cUb, cUb, biasUb, mask);';
    lines[69] = '    ubBuf.EnQue(cUb);';
    lines[70] = '    LocalTensor<float> cUbReady = ubBuf.DeQue<float>();';
    lines[71] = '';
    lines[72] = '    LocalTensor<float> reluUb = ubBuf.Get<float>();';
    lines[73] = '    Relu(reluUb, cUbReady);';
    lines[74] = '    ubBuf.EnQue(reluUb);';
    lines[75] = '  }';
    lines[76] = '';
    lines[77] = '  __aicore__ inline void StoreOutput(uint32_t offset) {';
    lines[78] = '    LocalTensor<half> yUb = outQueue.AllocTensor<half>();';
    lines[79] = '    Cast(yUb, reluUb, RoundMode::CAST_RINT, len);';
    lines[80] = '    DataCopy(yGm[offset], yUb, {1, blockLen, 0, 0});';
    lines[81] = '    outQueue.FreeTensor(yUb);';
    lines[82] = '  }';
    lines[83] = '  __aicore__ inline void LoopScale() {';
    lines[84] = '    LocalTensor<float> scratch = ubBuf.Get<float>();';
    lines[85] = '    for (int i = 0; i < nLoop; ++i) {';
    lines[86] = '      Muls(scratch, yUb, scaleArr[i]);';
    lines[87] = '    }';
    lines[88] = '  }';
    lines[89] = '};';
    lines[91] = 'extern "C" __global__ __aicore__ void MatMulAddRelu_mix_aic__kernel0(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR y) {';
    lines[92] = '  MatMulAddReluMixAicKernel op;';
    lines[93] = '  op.Init(x, w, bias, y);';
    lines[94] = '  op.Process(GetBlockIdx());';
    lines[95] = '}';
    return {
      path: 'matmul_add_relu.cce',
      language: 'cpp',
      text: lines.join('\n'),
    };
  }

  function createDemoData(options = {}) {
    const coreLabel = options.coreTitle || options.coreId || 'AIV UB';
    return {
      kernel: `${coreLabel} · MatMulAddRelu_mix_aic__kernel0`,
      ticks: 120,
      sourceFiles: [createDemoSourceFile()],
      buffers: [
        { name: 'UB', capacity: 290816 },
        { name: 'L1', capacity: 524288 },
        { name: 'L0C', capacity: 131072 },
      ],
      tensors: [
        {
          id: 't1',
          name: 'xGm_in',
          buffer: 'UB',
          offset: 0,
          size: 65536,
          allocTick: 2,
          freeTick: 40,
          kind: 'resident',
          reuseOf: null,
          reusedBy: ['t7'],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 40,
          srcLineEnd: 46,
          srcHotLine: 43,
          code: 'LocalTensor<half> xUb = inQueueX.AllocTensor<half>();\nDataCopy(xUb, xGm[offset], {1, blockLen, 0, 0});\ninQueueX.EnQue(xUb);\nxUb = inQueueX.DeQue<half>();\nMmad(cMatrix, xUb, wUb, mmolParams);',
          cce: '// xGm_in @UB+0x00000 64KB\nLD.global.b128 %ub0,[%gm_x+0]\nWAIT_FLAG MTE2->M\nMMAD %l0c0,%ub0,%ub_w',
        },
        {
          id: 't2',
          name: 'wGm_in',
          buffer: 'UB',
          offset: 65536,
          size: 98304,
          allocTick: 3,
          freeTick: 58,
          kind: 'resident',
          reuseOf: null,
          reusedBy: ['t6'],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 48,
          srcLineEnd: 52,
          srcHotLine: 50,
          code: 'LocalTensor<half> wUb = inQueueW.AllocTensor<half>();\nDataCopy(wUb, wGm[woff], {nBlk, blockLen, 0, 0});\ninQueueW.EnQue(wUb);\nwUb = inQueueW.DeQue<half>();',
          cce: '// wGm_in @UB+0x10000 96KB\nLD.global.b128 %ub_w,[%gm_w+0]\nWAIT_FLAG MTE2->M',
        },
        {
          id: 't3',
          name: 'cMatrix',
          buffer: 'L0C',
          offset: 0,
          size: 131072,
          allocTick: 6,
          freeTick: 74,
          kind: 'resident',
          reuseOf: null,
          reusedBy: [],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 54,
          srcLineEnd: 58,
          srcHotLine: 56,
          code: 'Mmad(cMatrix, xUb, wUb, {M,K,N,true});',
          cce: '// cMatrix @L0C+0x0 128KB\nMMAD %l0c0,%ub0,%ub_w,init=1',
        },
        {
          id: 't4',
          name: 'biasUb',
          buffer: 'UB',
          offset: 163840,
          size: 16384,
          allocTick: 10,
          freeTick: 30,
          kind: 'temp',
          reuseOf: null,
          reusedBy: [],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 60,
          srcLineEnd: 63,
          srcHotLine: 61,
          code: 'LocalTensor<float> biasUb = biasBuf.Get<float>();\nDataCopy(biasUb, biasGm, n);',
          cce: '// biasUb @UB+0x28000 16KB\nLD.global.b32 %ub_b,[%gm_bias]',
        },
        {
          id: 't5',
          name: 'cUb_fp32',
          buffer: 'UB',
          offset: 180224,
          size: 65536,
          allocTick: 62,
          freeTick: 92,
          kind: 'temp',
          reuseOf: null,
          reusedBy: ['t8'],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 66,
          srcLineEnd: 70,
          srcHotLine: 68,
          code: 'LocalTensor<float> cUb = ubBuf.Get<float>();\nDataCopy(cUb, cMatrix, {TILE_M, TILE_N, 0, 0});\nAdd(cUb, cUb, biasUb, mask);',
          cce: '// cUb_fp32 @UB+0x2c000 64KB\nWAIT_FLAG M->V\nDataCopy %ub_c,%l0c0\nVADD %ub_c,%ub_c,%ub_b',
        },
        {
          id: 't6',
          name: 'reluUb',
          buffer: 'UB',
          offset: 65536,
          size: 65536,
          allocTick: 78,
          freeTick: 104,
          kind: 'temp',
          reuseOf: 't2',
          reusedBy: [],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 72,
          srcLineEnd: 75,
          srcHotLine: 73,
          code: 'Relu(reluUb, cUb);\n// reluUb address reuses wGm_in after tick #58',
          cce: '// reluUb @UB+0x10000 64KB (reuse of wGm_in)\nVMAX %ub_r,%ub_c,#0',
        },
        {
          id: 't7',
          name: 'yUb_out',
          buffer: 'UB',
          offset: 0,
          size: 65536,
          allocTick: 94,
          freeTick: 118,
          kind: 'temp',
          reuseOf: 't1',
          reusedBy: [],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 78,
          srcLineEnd: 82,
          srcHotLine: 80,
          code: 'Cast(yUb, reluUb, RoundMode::CAST_RINT, len);\nDataCopy(yGm[offset], yUb, {1, blockLen, 0, 0});',
          cce: '// yUb_out @UB+0x0 64KB (reuse of xGm_in)\nVCONV %ub_y,%ub_r,f32->f16\nST.global.b128 [%gm_y],%ub_y',
        },
        {
          id: 't8',
          name: 'scratchUb',
          buffer: 'UB',
          offset: 180224,
          size: 32768,
          allocTick: 96,
          freeTick: 112,
          kind: 'loop',
          reuseOf: 't5',
          reusedBy: [],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 84,
          srcLineEnd: 87,
          srcHotLine: 85,
          code: 'for (int i = 0; i < nLoop; ++i) {\n  Muls(scratch, yUb, scaleArr[i]);\n}',
          cce: '// scratchUb @UB+0x2c000 32KB (reuse cUb)\nLOOP i\n  VMULS %ub_s,%ub_y,%scale\nENDLOOP',
        },
        {
          id: 't9',
          name: 'wL1_stage',
          buffer: 'L1',
          offset: 0,
          size: 262144,
          allocTick: 1,
          freeTick: 60,
          kind: 'resident',
          reuseOf: null,
          reusedBy: ['t10'],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 30,
          srcLineEnd: 34,
          srcHotLine: 32,
          code: 'DataCopy(wL1, wGm, {nBlk, blockLen, 0, 0});\nl1Buf.EnQue(wL1);',
          cce: '// wL1_stage @L1+0x0 256KB\nLD.global %l1_w,[%gm_w]',
        },
        {
          id: 't10',
          name: 'aL1_stage',
          buffer: 'L1',
          offset: 0,
          size: 262144,
          allocTick: 64,
          freeTick: 110,
          kind: 'loop',
          reuseOf: 't9',
          reusedBy: [],
          srcFile: 'matmul_add_relu.cce',
          srcLineStart: 36,
          srcLineEnd: 38,
          srcHotLine: 37,
          code: 'DataCopy(aL1, aGm[aOff], {1, blockLen, 0, 0});\nl1Buf.EnQue(aL1);',
          cce: '// aL1_stage @L1+0x0 256KB (reuse wL1_stage)\nLD.global %l1_a,[%gm_a]',
        },
      ],
    };
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function fmtKB(bytes) {
    const kb = bytes / 1024;
    return Number.isInteger(kb) ? `${kb}KB` : `${kb.toFixed(1)}KB`;
  }

  function fmtHex(bytes) {
    return `0x${Number(bytes || 0).toString(16).toUpperCase().padStart(5, '0')}`;
  }

  function nameOf(data, id) {
    const tensor = data.tensors.find((item) => item.id === id);
    return tensor ? tensor.name : id;
  }

  function render(container, data, options = {}) {
    if (!container) return null;
    const viewer = new MemoryReuseViewer(container, data || createDemoData(), options);
    viewer.mount();
    return viewer;
  }

  class MemoryReuseViewer {
    constructor(container, data, options) {
      this.container = container;
      this.data = clone(data);
      this.options = options || {};
      this.state = {
        buffer: this.options.initialBuffer || this.data.buffers?.[0]?.name || 'UB',
        showPeak: true,
        showGrid: true,
        filter: 'all',
        selected: null,
        hover: null,
        ox: 0,
        oy: 0,
        scale: 1,
      };
      this.rects = [];
      this.listeners = [];
      this.resizeObserver = null;
      this.drag = null;
      this.suppressClick = false;
      this.margin = { left: 58, right: 96, top: 54, bottom: 24 };
    }

    mount() {
      this.container.innerHTML = '';
      this.root = document.createElement('section');
      this.root.className = 'pto-memory-reuse-viewer';
      this.root.innerHTML = `
        <header class="pto-memory-reuse-viewer__toolbar">
          <div class="pto-memory-reuse-viewer__title">
            <span data-reuse-kernel></span>
          </div>
          <div class="pto-memory-reuse-viewer__controls">
            <select class="pto-memory-reuse-viewer__select" data-reuse-buffer></select>
            <details class="pto-memory-reuse-viewer__options" data-reuse-options>
              <summary class="pto-memory-reuse-viewer__button">选项</summary>
              <div class="pto-memory-reuse-viewer__options-panel">
                <div class="pto-memory-reuse-viewer__option-group">
                  <span class="pto-memory-reuse-viewer__option-label">显示</span>
                  <button class="pto-memory-reuse-viewer__button is-active" type="button" data-reuse-toggle="peak">峰值曲线</button>
                  <button class="pto-memory-reuse-viewer__button is-active" type="button" data-reuse-toggle="grid">网格</button>
                </div>
                <div class="pto-memory-reuse-viewer__option-group">
                  <span class="pto-memory-reuse-viewer__option-label">Tensor</span>
                  <div class="pto-memory-reuse-viewer__filter" data-reuse-filter>
                    <button class="is-active" type="button" data-kind="all">全部</button>
                    <button type="button" data-kind="resident">常驻</button>
                    <button type="button" data-kind="temp">临时</button>
                    <button type="button" data-kind="reused">仅复用</button>
                  </div>
                </div>
                <button class="pto-memory-reuse-viewer__button" type="button" data-reuse-reset>重置视图</button>
              </div>
            </details>
            <button class="pto-memory-reuse-viewer__button" type="button" data-reuse-open-source>打开源码</button>
            <button class="pto-memory-reuse-viewer__icon-button" type="button" data-reuse-info aria-expanded="false" aria-label="视图说明" title="视图说明">i</button>
          </div>
          <div class="pto-memory-reuse-viewer__metrics">
            <span class="pto-memory-reuse-viewer__metric">容量 <b data-reuse-capacity>--</b></span>
            <span class="pto-memory-reuse-viewer__metric is-peak">峰值 <b data-reuse-peak>--</b></span>
            <span class="pto-memory-reuse-viewer__metric">利用率 <b data-reuse-util>--</b></span>
            <span class="pto-memory-reuse-viewer__metric">Tensors <b data-reuse-count>--</b></span>
          </div>
          <div class="pto-memory-reuse-viewer__info-popover" data-reuse-info-popover hidden>
            <strong>视图说明</strong>
            <p>横轴为当前 buffer 的地址 offset / 容量范围，矩形宽度表示 tensor 地址区间和占用大小。纵轴为 kernel 生命周期 tick，时间从上到下推进，矩形高度表示 alloc 到 free 的存活区间。红色曲线是每个 tick 的 live usage 峰值投影。按住 Command + 滚轮缩放图表。</p>
            <div class="pto-memory-reuse-viewer__legend">
              <span><i class="pto-memory-reuse-viewer__swatch" style="background:#58a6ff"></i>常驻</span>
              <span><i class="pto-memory-reuse-viewer__swatch" style="background:#3fb950"></i>临时</span>
              <span><i class="pto-memory-reuse-viewer__swatch" style="background:#f0883e"></i>跨循环</span>
              <span><i class="pto-memory-reuse-viewer__swatch" style="background:#ff4b7b"></i>峰值线</span>
            </div>
          </div>
        </header>
        <div class="pto-memory-reuse-viewer__body">
          <section class="pto-memory-reuse-viewer__stage" data-reuse-stage>
            <canvas class="pto-memory-reuse-viewer__canvas" data-reuse-canvas></canvas>
            <div class="pto-memory-reuse-viewer__tip" data-reuse-tip></div>
            <div class="pto-memory-reuse-viewer__detail-popover" data-reuse-detail-popover hidden></div>
          </section>
        </div>`;
      this.container.appendChild(this.root);
      this.bindEls();
      this.populateControls();
      this.bindEvents();
      this.resize();
      this.observeResize();
    }

    bindEls() {
      this.stage = this.root.querySelector('[data-reuse-stage]');
      this.canvas = this.root.querySelector('[data-reuse-canvas]');
      this.ctx = this.canvas.getContext('2d');
      this.tip = this.root.querySelector('[data-reuse-tip]');
      this.bufferSelect = this.root.querySelector('[data-reuse-buffer]');
      this.optionsMenu = this.root.querySelector('[data-reuse-options]');
      this.openSourceButton = this.root.querySelector('[data-reuse-open-source]');
      this.infoButton = this.root.querySelector('[data-reuse-info]');
      this.infoPopover = this.root.querySelector('[data-reuse-info-popover]');
      this.detailPopover = this.root.querySelector('[data-reuse-detail-popover]');
      this.kernelEl = this.root.querySelector('[data-reuse-kernel]');
      this.kernelEl.textContent = this.data.kernel || '';
    }

    listen(target, type, handler, options) {
      if (!target) return;
      target.addEventListener(type, handler, options);
      this.listeners.push(() => target.removeEventListener(type, handler, options));
    }

    cssVar(name, fallback) {
      return getComputedStyle(this.root).getPropertyValue(name).trim() || fallback;
    }

    selectedTensor() {
      return this.data.tensors.find((tensor) => tensor.id === this.state.selected) || null;
    }

    emit(name, detail = {}) {
      this.container.dispatchEvent(new CustomEvent(name, {
        detail,
        bubbles: true,
      }));
    }

    emitTensorSelect(tensor) {
      this.emit('pto-memory-reuse-tensor-select', {
        tensor,
        buffer: this.state.buffer,
        data: this.data,
      });
    }

    requestSourceOpen() {
      this.emit('pto-memory-reuse-open-source', {
        tensor: this.selectedTensor() || this.visibleTensors()[0] || null,
        buffer: this.state.buffer,
        data: this.data,
      });
    }

    populateControls() {
      this.bufferSelect.innerHTML = '';
      this.data.buffers.forEach((buffer) => {
        const option = document.createElement('option');
        option.value = buffer.name;
        option.textContent = `${buffer.name} · ${fmtKB(buffer.capacity)}`;
        this.bufferSelect.appendChild(option);
      });
      this.bufferSelect.value = this.state.buffer;
    }

    bindEvents() {
      this.listen(this.bufferSelect, 'change', () => {
        this.state.buffer = this.bufferSelect.value;
        this.state.selected = null;
        this.resetView(false);
        this.hideDetailPopover(false);
        this.draw();
      });
      this.listen(this.root.querySelector('[data-reuse-toggle="peak"]'), 'click', (event) => {
        this.state.showPeak = !this.state.showPeak;
        event.currentTarget.classList.toggle('is-active', this.state.showPeak);
        this.draw();
      });
      this.listen(this.root.querySelector('[data-reuse-toggle="grid"]'), 'click', (event) => {
        this.state.showGrid = !this.state.showGrid;
        event.currentTarget.classList.toggle('is-active', this.state.showGrid);
        this.draw();
      });
      this.listen(this.root.querySelector('[data-reuse-reset]'), 'click', () => this.resetView());
      this.listen(this.openSourceButton, 'click', () => this.requestSourceOpen());
      this.listen(this.infoButton, 'click', (event) => {
        event.stopPropagation();
        this.toggleInfoPopover();
      });
      this.listen(document, 'click', (event) => {
        if (this.optionsMenu?.open && !this.optionsMenu.contains(event.target)) {
          this.optionsMenu.open = false;
        }
        if (this.infoPopover.hidden) return;
        if (this.infoPopover.contains(event.target) || this.infoButton.contains(event.target)) return;
        this.toggleInfoPopover(false);
      });
      this.root.querySelectorAll('[data-reuse-filter] button').forEach((button) => {
        this.listen(button, 'click', () => {
          this.root.querySelectorAll('[data-reuse-filter] button').forEach((item) => item.classList.remove('is-active'));
          button.classList.add('is-active');
          this.state.filter = button.dataset.kind || 'all';
          this.hideDetailPopover(false);
          this.draw();
        });
      });
      this.listen(this.stage, 'mousemove', (event) => this.onPointerMove(event));
      this.listen(this.stage, 'mouseleave', () => {
        this.state.hover = null;
        this.stage.classList.remove('is-tensor-hovered');
        this.tip.style.display = 'none';
        this.draw();
      });
      this.listen(this.stage, 'click', (event) => {
        if (this.suppressClick) return;
        const point = this.eventPoint(event);
        const tensor = this.hitTest(point.x, point.y);
        if (tensor) {
          this.selectTensor(tensor.id, point);
          return;
        }
        this.state.selected = null;
        this.hideDetailPopover(false);
        this.draw();
      });
      this.listen(this.stage, 'mousedown', (event) => {
        if (event.button != null && event.button !== 0) return;
        this.drag = { x: event.clientX, y: event.clientY, ox: this.state.ox, oy: this.state.oy, moved: false };
      });
      this.listen(window, 'mousemove', (event) => {
        if (!this.drag) return;
        if (Math.hypot(event.clientX - this.drag.x, event.clientY - this.drag.y) > 3) this.drag.moved = true;
        if (!this.drag.moved) return;
        this.state.ox = this.drag.ox + event.clientX - this.drag.x;
        this.state.oy = this.drag.oy + event.clientY - this.drag.y;
        this.hideDetailPopover(false);
        this.draw();
      });
      this.listen(window, 'mouseup', () => {
        if (this.drag?.moved) {
          this.suppressClick = true;
          window.setTimeout(() => {
            this.suppressClick = false;
          }, 0);
        }
        this.drag = null;
      });
      this.listen(this.stage, 'wheel', (event) => {
        if (!event.metaKey) return;
        event.preventDefault();
        const point = this.eventPoint(event);
        const next = event.deltaY < 0 ? 1.12 : 1 / 1.12;
        this.state.ox = point.x - (point.x - this.state.ox) * next;
        this.state.oy = point.y - (point.y - this.state.oy) * next;
        this.state.scale = Math.max(0.45, Math.min(10, this.state.scale * next));
        this.hideDetailPopover(false);
        this.draw();
      }, { passive: false });
      this.listen(this.detailPopover, 'click', (event) => {
        if (!event.target.closest('[data-reuse-detail-close]')) return;
        event.stopPropagation();
        this.state.selected = null;
        this.hideDetailPopover();
      });
    }

    observeResize() {
      if ('ResizeObserver' in window) {
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.stage);
      } else {
        this.listen(window, 'resize', () => this.resize());
      }
    }

    resetView(redraw = true) {
      this.state.ox = 0;
      this.state.oy = 0;
      this.state.scale = 1;
      this.hideDetailPopover(false);
      if (redraw) this.draw();
    }

    toggleInfoPopover(force) {
      const open = typeof force === 'boolean' ? force : this.infoPopover.hidden;
      this.infoPopover.hidden = !open;
      this.infoButton.setAttribute('aria-expanded', open ? 'true' : 'false');
    }

    hideDetailPopover(redraw = true) {
      if (!this.detailPopover) return;
      this.detailPopover.hidden = true;
      this.detailPopover.innerHTML = '';
      if (redraw) this.draw();
    }

    selectTensor(tensorId, point = null) {
      const tensor = this.data.tensors.find((item) => item.id === tensorId);
      if (!tensor) return;
      if (tensor.buffer !== this.state.buffer) {
        this.state.buffer = tensor.buffer;
        if (this.bufferSelect) this.bufferSelect.value = tensor.buffer;
        this.resetView(false);
      }
      this.state.selected = tensor.id;
      const rect = this.rects.find((item) => item.tensor.id === tensor.id);
      const popPoint = point || (rect
        ? { x: rect.x + Math.min(rect.w, 28), y: rect.y + Math.min(rect.h, 28) }
        : { x: this.margin.left + 18, y: this.margin.top + 18 });
      this.showDetailPopover(tensor, popPoint);
      this.emitTensorSelect(tensor);
      this.draw();
    }

    showDetailPopover(tensor, point) {
      if (!tensor || !this.detailPopover) return;
      this.renderDetailPopover(tensor);
      this.detailPopover.hidden = false;
      this.tip.style.display = 'none';
      window.requestAnimationFrame(() => this.positionDetailPopover(point));
    }

    positionDetailPopover(point) {
      if (!this.detailPopover || !point) return;
      const stageRect = this.stage.getBoundingClientRect();
      const popRect = this.detailPopover.getBoundingClientRect();
      const margin = 12;
      const preferredX = point.x + 16;
      const preferredY = point.y + 16;
      const maxX = Math.max(margin, stageRect.width - popRect.width - margin);
      const maxY = Math.max(margin, stageRect.height - popRect.height - margin);
      const x = Math.max(margin, Math.min(maxX, preferredX));
      const y = Math.max(margin, Math.min(maxY, preferredY));
      this.detailPopover.style.left = `${x}px`;
      this.detailPopover.style.top = `${y}px`;
    }

    currentBuffer() {
      return this.data.buffers.find((buffer) => buffer.name === this.state.buffer) || this.data.buffers[0];
    }

    bufferTensors() {
      return this.data.tensors.filter((tensor) => tensor.buffer === this.state.buffer);
    }

    visibleTensors() {
      let tensors = this.bufferTensors();
      if (this.state.filter === 'resident') tensors = tensors.filter((tensor) => tensor.kind === 'resident');
      if (this.state.filter === 'temp') tensors = tensors.filter((tensor) => tensor.kind !== 'resident');
      if (this.state.filter === 'reused') tensors = tensors.filter((tensor) => tensor.reuseOf || tensor.reusedBy?.length);
      return tensors;
    }

    computePeak() {
      const usage = new Array((this.data.ticks || 0) + 1).fill(0);
      this.bufferTensors().forEach((tensor) => {
        for (let tick = tensor.allocTick; tick < tensor.freeTick; tick += 1) {
          usage[tick] += tensor.size;
        }
      });
      let peak = 0;
      let peakTick = 0;
      usage.forEach((value, index) => {
        if (value > peak) {
          peak = value;
          peakTick = index;
        }
      });
      return { usage, peak, peakTick };
    }

    plotRect() {
      const width = this.canvas.clientWidth || 1;
      const height = this.canvas.clientHeight || 1;
      return {
        x: this.margin.left,
        y: this.margin.top,
        w: Math.max(80, width - this.margin.left - this.margin.right),
        h: Math.max(80, height - this.margin.top - this.margin.bottom),
      };
    }

    xOf(bytes) {
      const plot = this.plotRect();
      const capacity = this.currentBuffer()?.capacity || 1;
      return plot.x + (bytes / capacity) * plot.w * this.state.scale + this.state.ox;
    }

    yOf(tick) {
      const plot = this.plotRect();
      const ticks = this.data.ticks || 1;
      return plot.y + (tick / ticks) * plot.h * this.state.scale + this.state.oy;
    }

    eventPoint(event) {
      const rect = this.canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    resize() {
      const rect = this.stage.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));
      this.canvas.width = width * dpr;
      this.canvas.height = height * dpr;
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.draw();
    }

    draw() {
      if (!this.ctx || !this.canvas) return;
      const ctx = this.ctx;
      const width = this.canvas.clientWidth || 1;
      const height = this.canvas.clientHeight || 1;
      const plot = this.plotRect();
      const buffer = this.currentBuffer();
      const capacity = buffer?.capacity || 1;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = this.cssVar('--reuse-viewer-canvas-bg', '#101010');
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = this.cssVar('--reuse-viewer-plot-bg', '#0b0d10');
      ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
      this.drawGrid(ctx, plot, capacity);
      this.drawTensors(ctx, plot);
      if (this.state.showPeak) this.drawPeak(ctx);
      this.drawMasksAndRuler(ctx, plot, capacity, width, height);
      this.updateStats();
    }

    drawGrid(ctx, plot, capacity) {
      if (!this.state.showGrid) return;
      ctx.save();
      ctx.strokeStyle = this.cssVar('--reuse-viewer-grid', 'rgba(255,255,255,0.08)');
      ctx.fillStyle = this.cssVar('--reuse-viewer-grid-text', 'rgba(255,255,255,0.46)');
      ctx.font = '10px sans-serif';
      const stepX = this.rulerStep(plot, capacity);
      for (let bytes = 0; bytes <= capacity; bytes += stepX) {
        const x = this.xOf(bytes);
        if (x < plot.x - 1 || x > plot.x + plot.w + 1) continue;
        ctx.beginPath();
        ctx.moveTo(x, plot.y);
        ctx.lineTo(x, plot.y + plot.h);
        ctx.stroke();
      }
      for (let tick = 0; tick <= this.data.ticks; tick += 20) {
        const y = this.yOf(tick);
        if (y < plot.y - 1 || y > plot.y + plot.h + 1) continue;
        ctx.beginPath();
        ctx.moveTo(plot.x, y);
        ctx.lineTo(plot.x + plot.w, y);
        ctx.stroke();
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(`#${tick}`, plot.x - 6, y);
      }
      ctx.restore();
    }

    drawTensors(ctx) {
      this.rects = [];
      const visible = new Set(this.visibleTensors().map((tensor) => tensor.id));
      this.bufferTensors().forEach((tensor) => {
        const x = this.xOf(tensor.offset);
        const x2 = this.xOf(tensor.offset + tensor.size);
        const y = this.yOf(tensor.allocTick);
        const y2 = this.yOf(tensor.freeTick);
        const w = x2 - x;
        const h = y2 - y;
        const dim = !visible.has(tensor.id);
        const selected = this.state.selected === tensor.id;
        const hovered = this.state.hover === tensor.id;
        const color = KIND_COLORS[tensor.kind] || '#999';
        ctx.save();
        ctx.globalAlpha = dim ? 0.12 : selected ? 1 : 0.82;
        ctx.fillStyle = color;
        this.roundRect(ctx, x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2), 3);
        ctx.fill();
        ctx.globalAlpha = dim ? 0.16 : 1;
        ctx.strokeStyle = selected
          ? this.cssVar('--reuse-viewer-tensor-selected', '#ffaa3b')
          : hovered
            ? this.cssVar('--reuse-viewer-tensor-hover', '#f5f5f5')
            : this.cssVar('--reuse-viewer-tensor-stroke', '#0b0d10');
        ctx.lineWidth = selected ? 2.4 : hovered ? 2 : 1;
        this.roundRect(ctx, x + 1, y + 1, Math.max(1, w - 2), Math.max(1, h - 2), 3);
        ctx.stroke();
        if (tensor.reuseOf && !dim) {
          ctx.globalAlpha = 0.9;
          ctx.strokeStyle = this.cssVar('--reuse-viewer-tensor-stroke', '#0b0d10');
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x + 3, y + 4);
          ctx.lineTo(x + Math.min(22, w - 3), y + 4);
          ctx.stroke();
        }
        if (w > 50 && h > 17 && !dim) {
          ctx.globalAlpha = 1;
          ctx.fillStyle = this.cssVar('--reuse-viewer-tensor-label', '#0b0d10');
          ctx.font = '11px sans-serif';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          const maxChars = Math.max(1, Math.floor((w - 10) / 6.4));
          const label = tensor.name.length > maxChars ? `${tensor.name.slice(0, maxChars - 1)}...` : tensor.name;
          ctx.fillText(label, x + 6, y + 5);
          if (h > 30) {
            ctx.globalAlpha = 0.72;
            ctx.fillText(fmtKB(tensor.size), x + 6, y + 19);
          }
        }
        ctx.restore();
        this.rects.push({ tensor, x, y, w, h });
      });
    }

    drawPeak(ctx) {
      const { usage, peak, peakTick } = this.computePeak();
      ctx.save();
      ctx.strokeStyle = this.cssVar('--reuse-viewer-peak', '#ff4b7b');
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.92;
      ctx.beginPath();
      usage.forEach((value, tick) => {
        const x = this.xOf(value);
        const y = this.yOf(tick);
        if (tick === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      const peakX = this.xOf(peak);
      const peakY = this.yOf(peakTick);
      ctx.globalAlpha = 1;
      ctx.fillStyle = this.cssVar('--reuse-viewer-peak', '#ff4b7b');
      ctx.beginPath();
      ctx.arc(peakX, peakY, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`peak ${fmtKB(peak)}`, peakX + 6, peakY - 2);
      ctx.restore();
    }

    drawMasksAndRuler(ctx, plot, capacity, width, height) {
      ctx.save();
      ctx.fillStyle = this.cssVar('--reuse-viewer-canvas-bg', '#101010');
      ctx.fillRect(0, 0, plot.x, height);
      ctx.fillRect(plot.x + plot.w, 0, width, height);
      ctx.fillRect(0, 0, width, plot.y);
      ctx.fillRect(0, plot.y + plot.h, width, height);
      ctx.strokeStyle = this.cssVar('--reuse-viewer-plot-stroke', 'rgba(255,255,255,0.24)');
      ctx.beginPath();
      ctx.moveTo(plot.x, plot.y - 0.5);
      ctx.lineTo(plot.x + plot.w, plot.y - 0.5);
      ctx.stroke();
      const step = this.rulerStep(plot, capacity);
      ctx.font = '10px sans-serif';
      for (let bytes = 0; bytes <= capacity; bytes += step) {
        const x = this.xOf(Math.min(bytes, capacity));
        if (x < plot.x - 2 || x > plot.x + plot.w + 2) continue;
        ctx.strokeStyle = this.cssVar('--reuse-viewer-ruler', 'rgba(255,255,255,0.38)');
        ctx.beginPath();
        ctx.moveTo(x, plot.y);
        ctx.lineTo(x, plot.y - 7);
        ctx.stroke();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillStyle = this.cssVar('--reuse-viewer-grid-text', 'rgba(255,255,255,0.48)');
        ctx.fillText(fmtHex(bytes), x, plot.y - 9);
        ctx.fillStyle = this.cssVar('--reuse-viewer-ruler-text', 'rgba(255,255,255,0.72)');
        ctx.fillText(fmtKB(bytes), x, plot.y - 21);
      }
      ctx.fillStyle = this.cssVar('--reuse-viewer-axis-text', 'rgba(255,255,255,0.58)');
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('地址 / Buffer offset ->', plot.x, plot.y - 35);
      ctx.save();
      ctx.translate(14, plot.y + plot.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText('指令生命周期 tick', 0, 0);
      ctx.restore();
      ctx.restore();
    }

    rulerStep(plot, capacity) {
      const steps = [4, 8, 16, 32, 64, 128, 256, 512, 1024].map((kb) => kb * 1024);
      const pxPerByte = (plot.w * this.state.scale) / capacity;
      return steps.find((step) => step * pxPerByte >= 64) || steps[steps.length - 1];
    }

    roundRect(ctx, x, y, w, h, radius) {
      const r = Math.min(radius, w / 2, h / 2);
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    hitTest(x, y) {
      for (let index = this.rects.length - 1; index >= 0; index -= 1) {
        const rect = this.rects[index];
        if (x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h) {
          return rect.tensor;
        }
      }
      return null;
    }

    onPointerMove(event) {
      const point = this.eventPoint(event);
      const tensor = this.hitTest(point.x, point.y);
      this.state.hover = tensor ? tensor.id : null;
      this.stage.classList.toggle('is-tensor-hovered', !!tensor);
      if (tensor) {
        const reused = tensor.reusedBy?.length ? `<div>被复用 -> <b>${tensor.reusedBy.map((id) => nameOf(this.data, id)).join(', ')}</b></div>` : '';
        const reuse = tensor.reuseOf ? `<div>复用自 <b>${nameOf(this.data, tensor.reuseOf)}</b></div>` : '';
        this.tip.innerHTML = `
          <div class="pto-memory-reuse-viewer__tip-name">${escapeHtml(tensor.name)}</div>
          <div>地址 <b>${fmtHex(tensor.offset)}-${fmtHex(tensor.offset + tensor.size)}</b></div>
          <div>大小 <b>${fmtKB(tensor.size)}</b> · 区间 <b>#${tensor.allocTick}-#${tensor.freeTick}</b></div>
          ${reuse}${reused}`;
        const stageRect = this.stage.getBoundingClientRect();
        this.tip.style.display = 'block';
        this.tip.style.left = `${Math.min(point.x + 14, stageRect.width - 310)}px`;
        this.tip.style.top = `${Math.min(point.y + 14, stageRect.height - 110)}px`;
      } else {
        this.tip.style.display = 'none';
      }
      this.draw();
    }

    updateStats() {
      const buffer = this.currentBuffer();
      const { peak } = this.computePeak();
      this.root.querySelector('[data-reuse-capacity]').textContent = fmtKB(buffer?.capacity || 0);
      this.root.querySelector('[data-reuse-peak]').textContent = fmtKB(peak);
      this.root.querySelector('[data-reuse-util]').textContent = `${((peak / (buffer?.capacity || 1)) * 100).toFixed(1)}%`;
      this.root.querySelector('[data-reuse-count]').textContent = String(this.bufferTensors().length);
    }

    renderDetailPopover(tensor) {
      const tags = [];
      if (tensor.reuseOf) tags.push(`<span class="pto-memory-reuse-viewer__tag">复用自 ${escapeHtml(nameOf(this.data, tensor.reuseOf))}</span>`);
      (tensor.reusedBy || []).forEach((id) => {
        tags.push(`<span class="pto-memory-reuse-viewer__tag">被 ${escapeHtml(nameOf(this.data, id))} 复用</span>`);
      });
      this.detailPopover.innerHTML = `
        <div class="pto-memory-reuse-viewer__detail-head">
          <div>
            <strong>${escapeHtml(tensor.name)}</strong>
            <span>${escapeHtml(tensor.buffer)} · ${fmtHex(tensor.offset)} - ${fmtHex(tensor.offset + tensor.size)}</span>
          </div>
          <button class="pto-memory-reuse-viewer__detail-close" type="button" data-reuse-detail-close aria-label="关闭 Tensor 详情">×</button>
        </div>
        <dl class="pto-memory-reuse-viewer__kv">
          <dt>Buffer</dt><dd>${escapeHtml(tensor.buffer)}</dd>
          <dt>地址区间</dt><dd>${fmtHex(tensor.offset)} - ${fmtHex(tensor.offset + tensor.size)}</dd>
          <dt>占用大小</dt><dd>${fmtKB(tensor.size)} (${tensor.size} B)</dd>
          <dt>存活区间</dt><dd>#${tensor.allocTick} -> #${tensor.freeTick}</dd>
          <dt>分类</dt><dd>${escapeHtml(tensor.kind)}</dd>
          <dt>源文件</dt><dd>${escapeHtml(tensor.srcFile || '')}:${tensor.srcLineStart || ''}-${tensor.srcLineEnd || ''}</dd>
        </dl>
        ${tags.length ? `<div class="pto-memory-reuse-viewer__reuse-tags">${tags.join('')}</div>` : ''}
        <p class="pto-memory-reuse-viewer__code-label">源码 · ${escapeHtml(tensor.srcFile || '')}</p>
        <pre class="pto-memory-reuse-viewer__code">${escapeHtml(tensor.code || '')}</pre>
        <p class="pto-memory-reuse-viewer__code-label">CCE 指令</p>
        <pre class="pto-memory-reuse-viewer__code">${escapeHtml(tensor.cce || '')}</pre>`;
    }

    destroy() {
      this.listeners.splice(0).forEach((cleanup) => cleanup());
      this.resizeObserver?.disconnect?.();
      this.container.innerHTML = '';
    }
  }

  global.PtoMemoryReuseViewer = {
    render,
    createDemoData,
    sampleData: createDemoData(),
  };
})(window);
