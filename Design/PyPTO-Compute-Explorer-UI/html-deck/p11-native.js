(() => {
const root = window.document.querySelector('[data-p11]');
const document = {
getElementById: id => root.querySelector('#p11-' + id),
querySelectorAll: selector => root.querySelectorAll(selector),
createElementNS: (...args) => window.document.createElementNS(...args),
};

    const SVG_NS = 'http://www.w3.org/2000/svg';

    function createSVGElement(tag, attrs = {}) {
      const el = document.createElementNS(SVG_NS, tag);
      for (const [key, val] of Object.entries(attrs)) {
        el.setAttribute(key, val);
      }
      return el;
    }

    const SCENES = [
      {
        code: 'ACT 01 / 04',
        shortTitle: '独立视角隔离',
        caption: '传统性能调优依赖三类割裂工具独立观察算子，缺乏跨维度的上下文关联。',
        dotColor: 'bg-indigo-400',
        card: null
      },
      {
        code: 'ACT 02 / 04',
        shortTitle: '视角交汇 · UX 融合',
        caption: '打破工具孤岛：让 Graph、Assignment 与 Pipeline 扩张交汇，在交集中精准定位跨视角计算对象。',
        dotColor: 'bg-emerald-400',
        card: {
          tag: 'UX INTEGRATION',
          metric: '3-WAY SYNC',
          title: '跨视角计算对象同源追踪',
          desc: '通过计算×数据、计算×时间、数据×时间三大交叉域，实现图节点、存储块与流水线气泡的毫秒级联动定位。'
        }
      },
      {
        code: 'ACT 03 / 04',
        shortTitle: '多维克隆与视角压扁',
        caption: '将融合后的三位一体观测切面横向克隆为多粒度实例，由 2D 俯视图平滑压缩为立体侧视剖面。',
        dotColor: 'bg-amber-400',
        card: {
          tag: 'PERSPECTIVE MORPH',
          metric: '3 SCALES',
          title: '从平面融合切面到多层级观测切片',
          desc: '每一个侧视切面均完整保留三视角交汇关系，为接下来的执行层级纵向爆炸下钻做好空间编排。'
        }
      },
      {
        code: 'ACT 04 / 04',
        shortTitle: '执行层级爆炸下钻',
        caption: '沿执行轴线纵向展开，自上而下逐级下钻（Device → AI Core → Iteration），点击右侧标注可检视对应层级细节。',
        dotColor: 'bg-sky-400',
        card: {
          tag: 'HIERARCHY DRILL-DOWN',
          metric: 'DEVICE → ITER',
          title: '全栈执行层级同轴下钻',
          desc: '层级越深观测对象越聚焦（面积逐级收敛），同时在任一层级都能同步透视计算依赖、空间分配与流水节拍。'
        }
      }
    ];

    const TIER_DETAILS = [
      {
        tag: 'TIER 01 · DEVICE',
        metric: 'SCALE 100%',
        title: 'Device 设备级全局视图',
        desc: '观测多卡/多核间的计算图切分（DAG）、HBM 全局显存分片矩阵（4×3）以及多流并发调度甘特图。'
      },
      {
        tag: 'TIER 02 · AI CORE',
        metric: 'SCALE 78%',
        title: 'AI Core 核内执行视图',
        desc: '下钻至单核内部：追踪 MatMul/Softmax 算子流、片上 L1/UB 缓冲分区（3×2）与 MTE/Cube/Vector 三级流水重叠。'
      },
      {
        tag: 'TIER 03 · ITERATION',
        metric: 'SCALE 60%',
        title: 'Iteration 单次迭代微视图',
        desc: '聚焦最细粒度执行循环：检视单步指令依赖弧线、寄存器 Tile 分块（2×2）与单周期指令发射节拍。'
      }
    ];

    const instancesLayer = document.getElementById('instances-layer');
    const vennInstances = [];
    let currentScene = 0;
    let activeTier = -1;
    let autoPlayTimer = null;

    /**
     * Build DOM structure for each of the 3 Venn assemblies
     */
    function initInstances() {
      for (let i = 0; i < 3; i++) {
        const root = createSVGElement('g', { id: `venn-instance-${i}` });

        // 3 Ellipses: Graph (Top), Assignment (Bottom-Left), Pipeline (Bottom-Right)
        const elGraph = createSVGElement('ellipse', {
          fill: 'url(#p11-grad-graph)',
          stroke: '#6384FF',
          'stroke-width': '1.6',
          'stroke-opacity': '0.85'
        });

        const elAssign = createSVGElement('ellipse', {
          fill: 'url(#p11-grad-assign)',
          stroke: '#4ADE80',
          'stroke-width': '1.6',
          'stroke-opacity': '0.85'
        });

        const elPipe = createSVGElement('ellipse', {
          fill: 'url(#p11-grad-pipe)',
          stroke: '#F59E0B',
          'stroke-width': '1.6',
          'stroke-opacity': '0.85'
        });

        // Labels for each circle/ellipse (Full meta in Scene 1-2, Icon + Title in Scene 3-4)
        const labelGraph = createSVGElement('g', { class: 'label-graph' });
        labelGraph.innerHTML = `
          <g class="detail-meta">
            <use href="#p11-view-icon-graph" x="-64" y="-36" width="22" height="22" stroke="#E5E7EB" aria-hidden="true"/>
            <text x="-34" y="-20" fill="#F9FAFB" font-size="21" font-weight="700">Graph</text>
            <text x="-34" y="-3" fill="#9CA3AF" font-size="11">计算依赖关系网络化</text>
            <text x="0" y="28" text-anchor="middle" fill="rgba(255,255,255,0.16)" font-size="26" font-weight="700">What/Why</text>
          </g>
          <g class="compact-title" opacity="0">
            <use href="#p11-view-icon-graph" x="-42" y="-11" width="17" height="17" stroke="#818CF8" aria-hidden="true"/>
            <text x="-18" y="3" fill="#818CF8" font-size="17" font-weight="700">Graph</text>
          </g>
        `;

        const labelAssign = createSVGElement('g', { class: 'label-assign' });
        labelAssign.innerHTML = `
          <g class="detail-meta">
            <use href="#p11-view-icon-assignment" x="-82" y="-24" width="22" height="22" stroke="#E5E7EB" aria-hidden="true"/>
            <text x="-52" y="-9" fill="#F9FAFB" font-size="21" font-weight="700">Assignment</text>
            <text x="-52" y="8" fill="#9CA3AF" font-size="11">数据与任务分配空间化</text>
            <text x="-10" y="38" text-anchor="middle" fill="rgba(255,255,255,0.16)" font-size="25" font-weight="700">Who/Where</text>
          </g>
          <g class="compact-title" opacity="0">
            <use href="#p11-view-icon-assignment" x="-64" y="-9" width="17" height="17" stroke="#4ADE80" aria-hidden="true"/>
            <text x="-40" y="5" fill="#4ADE80" font-size="16" font-weight="700">Assignment</text>
          </g>
        `;

        const labelPipe = createSVGElement('g', { class: 'label-pipe' });
        labelPipe.innerHTML = `
          <g class="detail-meta">
            <use href="#p11-view-icon-pipeline" x="-61" y="-27" width="26" height="26" stroke="#E5E7EB" aria-hidden="true"/>
            <text x="-28" y="-9" fill="#F9FAFB" font-size="21" font-weight="700">Pipeline</text>
            <text x="-28" y="8" fill="#9CA3AF" font-size="11">计算组织过程可视化</text>
            <text x="12" y="38" text-anchor="middle" fill="rgba(255,255,255,0.16)" font-size="25" font-weight="700">How/When</text>
          </g>
          <g class="compact-title" opacity="0">
            <use href="#p11-view-icon-pipeline" x="-51" y="-11" width="20" height="20" stroke="#FBBF24" aria-hidden="true"/>
            <text x="-25" y="5" fill="#FBBF24" font-size="16" font-weight="700">Pipeline</text>
          </g>
        `;

        // Central Operator Core Node ("Paged Attention" in Scene 1 & 2, cross-view target reticle in Scene 3 & 4)
        const coreGroup = createSVGElement('g', { class: 'operator-core core-glow' });
        const coreHalo = createSVGElement('circle', {
          cx: 0, cy: 0, r: 46,
          fill: 'url(#p11-grad-core)'
        });
        const coreRadar = createSVGElement('g', { opacity: '0' });
        const coreRadarRing = createSVGElement('circle', {
          cx: 0, cy: 0, r: 16,
          fill: 'none',
          stroke: '#93C5FD',
          'stroke-width': '1.2',
          class: 'core-radar'
        });
        coreRadar.appendChild(coreRadarRing);
        const coreRing = createSVGElement('circle', {
          cx: 0, cy: 0, r: 36,
          fill: 'rgba(15, 17, 21, 0.88)',
          stroke: 'rgba(255, 255, 255, 0.45)',
          'stroke-width': '1.5'
        });
        const coreText = createSVGElement('g', { class: 'core-text' });
        coreText.innerHTML = `
          <text x="0" y="-4" text-anchor="middle" fill="#FFFFFF" font-size="13" font-weight="700" letter-spacing="0.3">Paged</text>
          <text x="0" y="13" text-anchor="middle" fill="#E5E7EB" font-size="13" font-weight="600" letter-spacing="0.3">Attention</text>
        `;
        const coreDiamond = createSVGElement('polygon', {
          points: '0,-6 6,0 0,6 -6,0',
          fill: '#F8FAFC',
          opacity: '0'
        });

        coreGroup.appendChild(coreHalo);
        coreGroup.appendChild(coreRadar);
        coreGroup.appendChild(coreRing);
        coreGroup.appendChild(coreText);
        coreGroup.appendChild(coreDiamond);

        // Assemble instance cleanly
        root.appendChild(elGraph);
        root.appendChild(elAssign);
        root.appendChild(elPipe);
        root.appendChild(labelGraph);
        root.appendChild(labelAssign);
        root.appendChild(labelPipe);
        root.appendChild(coreGroup);

        // Allow clicking layer in Scene 4
        root.addEventListener('click', () => {
          if (currentScene === 3) {
            selectTier(i);
          }
        });

        instancesLayer.appendChild(root);

        vennInstances.push({
          index: i,
          root,
          elGraph,
          elAssign,
          elPipe,
          labelGraph,
          labelAssign,
          labelPipe,
          coreGroup,
          coreHalo,
          coreRadar,
          coreRing,
          coreText,
          coreDiamond,
          // Current interpolated state
          state: null
        });
      }
    }

    /**
     * Define exact target geometry for each instance in each of the 4 scenes
     */
    function getTargetStates(sceneIdx) {
      // Scene 0: Isolated Circles pointing toward central Paged Attention
      if (sceneIdx === 0) {
        return [
          {
            x: 720, y: 385, scale: 1.0, opacity: 1,
            // Circle 1: Top (Graph)
            gCx: 0, gCy: -215, gRx: 105, gRy: 105,
            // Circle 2: Bottom-Left (Assignment)
            aCx: -215, aCy: 125, aRx: 105, aRy: 105,
            // Circle 3: Bottom-Right (Pipeline)
            pCx: 215, pCy: 125, pRx: 105, pRy: 105,
            // Labels
            lgX: 0, lgY: -212,
            laX: -212, laY: 125,
            lpX: 218, lpY: 125,
            detailAlpha: 1,
            compactAlpha: 0,
            coreR: 42,
            coreTextAlpha: 1,
            diamondAlpha: 0,
            telemetryAlpha: 0
          },
          // Clones 1 & 2 stay hidden beneath instance 0
          {
            x: 720, y: 385, scale: 0.9, opacity: 0,
            gCx: 0, gCy: -85, gRx: 175, gRy: 175,
            aCx: -82, aCy: 55, aRx: 175, aRy: 175,
            pCx: 82, pCy: 55, pRx: 175, pRy: 175,
            lgX: 0, lgY: -125, laX: -125, laY: 75, lpX: 125, lpY: 75,
            detailAlpha: 0, compactAlpha: 1, coreR: 16, coreTextAlpha: 0, diamondAlpha: 1, telemetryAlpha: 0
          },
          {
            x: 720, y: 385, scale: 0.9, opacity: 0,
            gCx: 0, gCy: -85, gRx: 175, gRy: 175,
            aCx: -82, aCy: 55, aRx: 175, aRy: 175,
            pCx: 82, pCy: 55, pRx: 175, pRy: 175,
            lgX: 0, lgY: -125, laX: -125, laY: 75, lpX: 125, lpY: 75,
            detailAlpha: 0, compactAlpha: 1, coreR: 16, coreTextAlpha: 0, diamondAlpha: 1, telemetryAlpha: 0
          }
        ];
      }

      // Scene 1: Intersecting Venn Diagram (Exact match to Reference Image 1)
      if (sceneIdx === 1) {
        return [
          {
            x: 720, y: 385, scale: 1.0, opacity: 1,
            // Equilateral triangle centers: Top (0, -113), BL (-98, 57), BR (98, 57), R = 185
            gCx: 0, gCy: -113, gRx: 185, gRy: 185,
            aCx: -98, aCy: 57, aRx: 185, aRy: 185,
            pCx: 98, pCy: 57, pRx: 185, pRy: 185,
            // Center the full labels in the three single-circle regions.
            lgX: 0, lgY: -174,
            laX: -170, laY: 72,
            lpX: 168, lpY: 72,
            detailAlpha: 1,
            compactAlpha: 0,
            coreR: 34,
            coreTextAlpha: 1,
            diamondAlpha: 0,
            telemetryAlpha: 0
          },
          {
            x: 720, y: 385, scale: 0.95, opacity: 0,
            gCx: 0, gCy: -113, gRx: 185, gRy: 185,
            aCx: -98, aCy: 57, aRx: 185, aRy: 185,
            pCx: 98, pCy: 57, pRx: 185, pRy: 185,
            lgX: 0, lgY: -174, laX: -170, laY: 72, lpX: 168, lpY: 72,
            detailAlpha: 0, compactAlpha: 1, coreR: 15, coreTextAlpha: 0, diamondAlpha: 1, telemetryAlpha: 0
          },
          {
            x: 720, y: 385, scale: 0.95, opacity: 0,
            gCx: 0, gCy: -113, gRx: 185, gRy: 185,
            aCx: -98, aCy: 57, aRx: 185, aRy: 185,
            pCx: 98, pCy: 57, pRx: 185, pRy: 185,
            lgX: 0, lgY: -174, laX: -170, laY: 72, lpX: 168, lpY: 72,
            detailAlpha: 0, compactAlpha: 1, coreR: 15, coreTextAlpha: 0, diamondAlpha: 1, telemetryAlpha: 0
          }
        ];
      }

      // Scene 2: Horizontal Multiplication & Perspective Compression (3 Side-by-Side Tilted Assemblies)
      if (sceneIdx === 2) {
        const xPositions = [360, 720, 1080];
        const scales = [0.76, 0.76, 0.76];
        return [0, 1, 2].map(idx => ({
          x: xPositions[idx],
          y: 380,
          scale: scales[idx],
          opacity: 1,
          // Compressed into isometric/side-view ellipses
          gCx: 0, gCy: -32, gRx: 175, gRy: 54,
          aCx: -95, aCy: 24, aRx: 175, aRy: 54,
          pCx: 95, pCy: 24, pRx: 175, pRy: 54,
          // Centered icon + title positions inside each ellipse
          lgX: 0, lgY: -48,
          laX: -158, laY: 28,
          lpX: 162, lpY: 28,
          detailAlpha: 0,
          compactAlpha: 1,
          coreR: 15,
          coreTextAlpha: 0,
          diamondAlpha: 1,
          telemetryAlpha: 0
        }));
      }

      // Scene 3: Vertical Exploded Stack — Side View Drill-Down (Image 2)
      // Stacked vertically at x=650, progressively scaling down: 1.0 -> 0.78 -> 0.60
      const yPositions = [185, 390, 585];
      const tierScales = [1.0, 0.78, 0.60];
      return [0, 1, 2].map(idx => {
        const isDimmed = activeTier !== -1 && activeTier !== idx;
        return {
          x: 650,
          y: yPositions[idx],
          scale: tierScales[idx],
          opacity: isDimmed ? 0.35 : 1.0,
          gCx: 0, gCy: -32, gRx: 185, gRy: 52,
          aCx: -102, aCy: 24, aRx: 185, aRy: 52,
          pCx: 102, pCy: 24, pRx: 185, pRy: 52,
          // Centered icon + title positions inside each ellipse
          lgX: 0, lgY: -48,
          laX: -165, laY: 28,
          lpX: 168, lpY: 28,
          detailAlpha: 0,
          compactAlpha: 1,
          coreR: 15,
          coreTextAlpha: 0,
          diamondAlpha: 1,
          telemetryAlpha: 0
        };
      });
    }

    /**
     * Render a single interpolated state frame onto the SVG elements
     */
    function applyState(inst, s) {
      inst.root.setAttribute('transform', `translate(${s.x.toFixed(2)}, ${s.y.toFixed(2)}) scale(${s.scale.toFixed(4)})`);
      inst.root.setAttribute('opacity', s.opacity.toFixed(3));

      // Ellipses
      inst.elGraph.setAttribute('cx', s.gCx.toFixed(2));
      inst.elGraph.setAttribute('cy', s.gCy.toFixed(2));
      inst.elGraph.setAttribute('rx', s.gRx.toFixed(2));
      inst.elGraph.setAttribute('ry', s.gRy.toFixed(2));

      inst.elAssign.setAttribute('cx', s.aCx.toFixed(2));
      inst.elAssign.setAttribute('cy', s.aCy.toFixed(2));
      inst.elAssign.setAttribute('rx', s.aRx.toFixed(2));
      inst.elAssign.setAttribute('ry', s.aRy.toFixed(2));

      inst.elPipe.setAttribute('cx', s.pCx.toFixed(2));
      inst.elPipe.setAttribute('cy', s.pCy.toFixed(2));
      inst.elPipe.setAttribute('rx', s.pRx.toFixed(2));
      inst.elPipe.setAttribute('ry', s.pRy.toFixed(2));

      // Labels
      inst.labelGraph.setAttribute('transform', `translate(${s.lgX.toFixed(2)}, ${s.lgY.toFixed(2)})`);
      inst.labelAssign.setAttribute('transform', `translate(${s.laX.toFixed(2)}, ${s.laY.toFixed(2)})`);
      inst.labelPipe.setAttribute('transform', `translate(${s.lpX.toFixed(2)}, ${s.lpY.toFixed(2)})`);

      // Toggle rich detail meta vs compact side-view titles
      inst.root.querySelectorAll('.detail-meta').forEach(el => el.setAttribute('opacity', s.detailAlpha.toFixed(3)));
      inst.root.querySelectorAll('.compact-title').forEach(el => el.setAttribute('opacity', s.compactAlpha.toFixed(3)));

      // Core Node
      inst.coreRing.setAttribute('r', s.coreR.toFixed(2));
      inst.coreRing.setAttribute('opacity', s.coreShellAlpha.toFixed(3));
      inst.coreHalo.setAttribute('opacity', s.coreShellAlpha.toFixed(3));
      inst.coreText.setAttribute('opacity', s.coreTextAlpha.toFixed(3));
      inst.coreDiamond.setAttribute('opacity', s.diamondAlpha.toFixed(3));
      // The circle pulses inside a group whose opacity follows the scene.
      inst.coreRadar.setAttribute('opacity', s.diamondAlpha.toFixed(3));
    }

    // Spring / Cubic-Bezier Interpolation Loop
    let animFrameId = null;
    let animStartTime = 0;
    let captionTimer = null;
    const ANIM_DURATION = 880; // ms for ultra-silky SaaS choreography
    const auxLayers = [
      { id: 'arrows-layer', scene: 0, enterDelay: 0, exitSpeed: 2.5 },
      { id: 'intersections-layer', scene: 1, enterDelay: 0.35, exitSpeed: 3 },
      { id: 'matrix-guides', scene: 2, enterDelay: 0, exitSpeed: 2.5 },
      { id: 'vertical-axis-group', scene: 3, enterDelay: 0, exitSpeed: 2.5 },
      { id: 'callouts-layer', scene: 3, enterDelay: 0.25, exitSpeed: 3 }
    ];

    function easeOutExpo(t) {
      return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    }

    function easeInOutCubic(t) {
      return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function interpolateStates(fromStates, toStates, progress) {
      for (let i = 0; i < 3; i++) {
        // Stagger clones slightly for organic kinetic wave feel
        const staggerOffset = i * 0.06;
        const localT = Math.max(0, Math.min(1, (progress - staggerOffset) / (1 - staggerOffset * 2)));
        const eased = easeInOutCubic(localT);

        const a = fromStates[i];
        const b = toStates[i];
        const current = {};
        for (const key of Object.keys(b)) {
          current[key] = lerp(a[key], b[key], eased);
        }
        vennInstances[i].state = current;
        applyState(vennInstances[i], current);
      }
    }

    function animateToScene(targetSceneIdx, immediate = false, sourceSceneIdx = targetSceneIdx, onComplete) {
      const targetStates = getTargetStates(targetSceneIdx).map(state => ({
        ...state,
        coreShellAlpha: targetSceneIdx === 1 ? 0 : 1
      }));

      if (animFrameId !== null) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
      }

      const layerTransitions = auxLayers.map(({ id, scene, enterDelay, exitSpeed }) => {
        const element = document.getElementById(id);
        const entering = scene === targetSceneIdx;
        const belongsToTransition = entering || scene === sourceSceneIdx;
        return {
          element,
          from: belongsToTransition ? Number(element.getAttribute('opacity')) : 0,
          to: entering ? 1 : 0,
          enterDelay,
          exitSpeed
        };
      });
      document.getElementById('callouts-layer').style.pointerEvents = targetSceneIdx === 3 ? 'auto' : 'none';

      if (immediate || !vennInstances[0].state) {
        for (let i = 0; i < 3; i++) {
          vennInstances[i].state = { ...targetStates[i] };
          applyState(vennInstances[i], targetStates[i]);
        }
        updateAuxLayers(layerTransitions, 1);
        onComplete?.();
        return;
      }

      updateAuxLayers(layerTransitions, 0);
      const startStates = vennInstances.map(inst => ({ ...inst.state }));
      animStartTime = performance.now();

      function tick(now) {
        const elapsed = now - animStartTime;
        const rawProgress = Math.min(1, elapsed / ANIM_DURATION);

        interpolateStates(startStates, targetStates, rawProgress);
        updateAuxLayers(layerTransitions, easeInOutCubic(rawProgress));

        if (rawProgress < 1) {
          animFrameId = requestAnimationFrame(tick);
        } else {
          animFrameId = null;
          onComplete?.();
        }
      }

      animFrameId = requestAnimationFrame(tick);
    }

    /**
     * Smoothly fade & transition auxiliary SVG layers (arrows, badges, guides, callouts)
     */
    function updateAuxLayers(transitions, t) {
      for (const { element, from, to, enterDelay, exitSpeed } of transitions) {
        const progress = to === 1
          ? Math.max(0, (t - enterDelay) / (1 - enterDelay))
          : Math.min(1, t * exitSpeed);
        element.setAttribute('opacity', lerp(from, to, progress).toFixed(3));
      }
    }

    /**
     * Scene Navigation & UI Synchronization
     */
    function goToScene(idx) {
      if (idx < 0 || idx >= SCENES.length) return;
      if (idx === 2) idx = 3;
      const sourceSceneIdx = currentScene;
      currentScene = idx;
      if (idx !== 3) activeTier = -1;

      // Update Bottom Buttons
      document.querySelectorAll('.scene-btn').forEach((btn, i) => {
        if (i === idx) {
          btn.classList.add('bg-white/15', 'text-white', 'shadow-sm');
          btn.classList.remove('text-gray-400', 'hover:text-gray-200', 'hover:bg-white/5');
        } else {
          btn.classList.remove('bg-white/15', 'text-white', 'shadow-sm');
          btn.classList.add('text-gray-400', 'hover:text-gray-200', 'hover:bg-white/5');
        }
      });

      // Update Header Metadata
      const meta = SCENES[idx];
      document.getElementById('scene-code').textContent = meta.code;
      document.getElementById('scene-short-title').textContent = meta.shortTitle;
      
      const captionEl = document.getElementById('narrative-caption');
      captionEl.style.opacity = '0';
      clearTimeout(captionTimer);
      captionTimer = setTimeout(() => {
        captionEl.textContent = meta.caption;
        captionEl.style.opacity = '1';
      }, 150);

      const dotEl = document.getElementById('status-dot');
      dotEl.className = `w-2 h-2 rounded-full animate-pulse ${meta.dotColor}`;

      // Update Contextual Telemetry Card
      updateTelemetryCard(meta.card);

      // Scene 2 is a transition pose between the two presentation stops.
      if ((sourceSceneIdx === 1 && idx === 3) || (sourceSceneIdx === 3 && idx === 1)) {
        animateToScene(2, false, sourceSceneIdx, () => animateToScene(idx, false, 2));
      } else {
        animateToScene(idx, false, sourceSceneIdx);
      }
    }

    function updateTelemetryCard(cardData) {
      const card = document.getElementById('telemetry-card');
      if (!cardData) {
        card.classList.add('opacity-0', 'translate-y-4', 'pointer-events-none');
        return;
      }
      document.getElementById('card-tag').textContent = cardData.tag;
      document.getElementById('card-metric').textContent = cardData.metric;
      document.getElementById('card-title').textContent = cardData.title;
      document.getElementById('card-desc').textContent = cardData.desc;

      card.classList.remove('opacity-0', 'translate-y-4', 'pointer-events-none');
    }

    function nextScene() {
      goToScene((currentScene + 1) % SCENES.length);
    }

    function prevScene() {
      goToScene(currentScene === 3 ? 1 : (currentScene - 1 + SCENES.length) % SCENES.length);
    }

    /**
     * Scene 4 Interactive Tier Inspection
     */
    function selectTier(tierIdx) {
      if (currentScene !== 3) return;
      activeTier = activeTier === tierIdx ? -1 : tierIdx;

      // Update callout visual highlight
      for (let i = 0; i < 3; i++) {
        const callout = document.getElementById(`callout-${i}`);
        callout.style.opacity = activeTier === -1 || activeTier === i ? '1' : '0.35';
      }

      if (activeTier === -1) {
        updateTelemetryCard(SCENES[3].card);
      } else {
        updateTelemetryCard(TIER_DETAILS[activeTier]);
      }

      animateToScene(3);
    }

    function hoverTier(tierIdx) {
      if (currentScene !== 3 || activeTier !== -1) return;
      for (let i = 0; i < 3; i++) {
        const callout = document.getElementById(`callout-${i}`);
        callout.style.opacity = tierIdx === -1 || tierIdx === i ? '1' : '0.45';
      }
    }

    /**
     * Auto-Play Presentation Mode
     */
    function toggleAutoPlay() {
      const btn = document.getElementById('autoplay-btn');
      const label = document.getElementById('autoplay-label');
      const icon = document.getElementById('play-icon');

      if (autoPlayTimer) {
        clearInterval(autoPlayTimer);
        autoPlayTimer = null;
        btn.classList.remove('text-indigo-300', 'bg-indigo-500/15');
        label.textContent = '自动演示';
        icon.innerHTML = '<path d="M8 5v14l11-7z" />';
      } else {
        btn.classList.add('text-indigo-300', 'bg-indigo-500/15');
        label.textContent = '演示中...';
        icon.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />';
        nextScene();
        autoPlayTimer = setInterval(() => {
          nextScene();
        }, 3600);
      }
    }


initInstances();
animateToScene(0,true);
function goToStep(index,{animate=true}={}) {
 if (!Number.isInteger(index) || index < 0 || index > 3) return false;
 goToScene(index === 2 ? 3 : index);
 if (!animate) animateToScene(currentScene,true);
 root.dispatchEvent(new Event('p11-step-change'));
 return true;
}
root.ptoP11Story = Object.freeze({
 goToStep,
 next: () => goToStep(currentScene+1),
 previous: () => goToStep(currentScene===3 ? 1 : currentScene-1),
 setActive(active) { if (!active) animateToScene(currentScene,true); },
 get step() { return currentScene; },
});
for (let tier=0;tier<3;tier++) {
 const callout=document.getElementById('callout-'+tier);
 callout.addEventListener('click',()=>selectTier(tier));
 callout.addEventListener('mouseenter',()=>hoverTier(tier));
 callout.addEventListener('mouseleave',()=>hoverTier(-1));
}
})();
