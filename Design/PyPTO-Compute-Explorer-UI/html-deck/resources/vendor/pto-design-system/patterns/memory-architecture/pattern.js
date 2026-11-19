(function registerPtoMemoryArchitecturePattern(global) {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const ROUTE_TONES = {
    transport: {
      line: '#ffcf59',
      fill: '#ffdf1f',
      stroke: '#ffdf1f',
      text: '#111111',
    },
    direct: {
      line: '#4d97ff',
      fill: '#2d75df',
      stroke: '#5db8ff',
      text: '#ffffff',
    },
    directReturn: {
      line: '#29c7a6',
      fill: '#29c7a6',
      stroke: '#5be5c2',
      text: '#ffffff',
    },
    simt: {
      line: '#ff9a54',
      fill: '#ffb06f',
      stroke: '#ffc18f',
      text: '#111111',
    },
    register: {
      line: '#7b61ff',
      fill: '#8f7cff',
      stroke: '#b1a4ff',
      text: '#ffffff',
    },
  };
  const ZOOM_DEFAULTS = {
    min: 0.4,
    max: 1.2,
    step: 0.1,
    defaultZoom: 0.6,
  };

  const PRESETS = {
    ascend950b: {
      id: 'ascend950b',
      name: 'Ascend 950B Memory Architecture',
      rails: [
        {
          key: 'GM',
          label: 'Global Memory',
          tone: 'memory-shell',
          grid: {
            rows: 82,
            cols: 8,
            cellSize: 12,
            gap: 4,
            shape: 'hex',
          },
        },
        {
          key: 'L2',
          label: 'L2 Cache',
          tone: 'memory-rail',
          grid: {
            rows: 82,
            cols: 4,
            cellSize: 12,
            gap: 4,
            shape: 'dot',
          },
        },
      ],
      cores: [
        {
          id: 'mem950-aiv1',
          kind: 'aiv',
          title: 'AIV 1',
          presetKey: 'aivOfficialV1',
        },
        {
          id: 'mem950-aic',
          kind: 'aic',
          title: 'AIC',
          presetKey: 'aicDraftV1',
        },
        {
          id: 'mem950-aiv2',
          kind: 'aiv',
          title: 'AIV 2',
          presetKey: 'aivOfficialV1',
        },
      ],
      routes: [
        {
          id: 'l2-to-aiv1-dcache',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aiv1 [data-aiv-node="cache:DCache"]',
          fromSide: 'right',
          toSide: 'left',
          toAnchorSelector: '.pto-aiv-core__grid',
          toBias: 0.50,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'l2-to-aiv1',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aiv1 [data-aiv-node="cache:ND-DMA Cache"]',
          fromSide: 'right',
          toSide: 'left',
          toAnchorSelector: '.pto-aiv-core__grid',
          toBias: 0.50,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'aiv1-to-l2',
          label: 'MTE3',
          tone: 'transport',
          from: '#mem950-aiv1 [data-aiv-node="buffer:UB"]',
          to: '[data-mem950-node="rail:L2"]',
          fromSide: 'left',
          toSide: 'right',
          fromAnchorSelector: '.pto-aiv-core__grid',
          fromBias: 0.82,
          sourceLaneBelowSelector: '#mem950-aiv1 [data-aiv-node="cache:ICache"]',
          sourceLaneOffset: 14,
          style: 'lane-h-source',
          labelDy: 0,
        },
        {
          id: 'gm-to-aiv1-ub',
          label: 'GM→ND-DMA',
          tone: 'transport',
          from: '[data-mem950-node="rail:GM"]',
          to: '#mem950-aiv1 [data-aiv-node="cache:ND-DMA Cache"]',
          fromSide: 'right',
          toSide: 'left',
          toAnchorSelector: '.pto-aiv-core__grid',
          toBias: 0.48,
          style: 'lane-h-target',
          labelDy: -12,
          defaultHidden: true,
        },
        {
          id: 'aiv1-ub-to-gm',
          label: 'UB→GM',
          tone: 'directReturn',
          from: '#mem950-aiv1 [data-aiv-node="buffer:UB"]',
          to: '[data-mem950-node="rail:GM"]',
          fromSide: 'left',
          toSide: 'right',
          fromAnchorSelector: '.pto-aiv-core__grid',
          fromBias: 0.82,
          sourceLaneBelowSelector: '#mem950-aiv1 [data-aiv-node="cache:ICache"]',
          sourceLaneOffset: 14,
          style: 'lane-h-source',
          labelDy: 14,
          defaultHidden: true,
        },
        {
          id: 'gm-to-aiv1-simt',
          label: 'GM→SIMT RF',
          tone: 'simt',
          from: '[data-mem950-node="rail:GM"]',
          to: '#mem950-aiv1 [data-aiv-node="exec:SIMT"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.42,
          style: 'lane-h-target',
          labelDy: -13,
          defaultHidden: true,
        },
        {
          id: 'l2-to-aic',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aic [data-aic-node="buffer:L1"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.58,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'l2-to-aic-dcache',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aic [data-aic-node="cache:DCache"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.50,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'gm-to-aic-l0a',
          label: 'GM→L0A',
          tone: 'register',
          from: '[data-mem950-node="rail:GM"]',
          to: '#mem950-aic [data-aic-node="buffer:L0A"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.46,
          style: 'lane-h-target',
          labelDy: -12,
          defaultHidden: true,
        },
        {
          id: 'gm-to-aic-l0b',
          label: 'GM→L0B',
          tone: 'register',
          from: '[data-mem950-node="rail:GM"]',
          to: '#mem950-aic [data-aic-node="buffer:L0B"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.54,
          style: 'lane-h-target',
          labelDy: 13,
          defaultHidden: true,
        },
        {
          id: 'aic-to-aiv1',
          label: 'L0C→UB',
          tone: 'direct',
          from: '#mem950-aic [data-aic-node="buffer:L0C"]',
          to: '#mem950-aiv1 [data-aiv-node="buffer:UB"]',
          fromSide: 'right',
          toSide: 'right',
          fromBias: 0.24,
          toAnchorSelector: '.pto-aiv-core__grid',
          toBias: 0.70,
          style: 'elbow-h',
          corridorRight: 40,
          labelDy: -12,
        },
        {
          id: 'aiv2-to-aic',
          label: 'UB→L1',
          tone: 'directReturn',
          from: '#mem950-aiv2 [data-aiv-node="buffer:UB"]',
          to: '#mem950-aic [data-aic-node="buffer:L1"]',
          fromSide: 'right',
          toSide: 'right',
          fromAnchorSelector: '.pto-aiv-core__grid',
          fromBias: 0.30,
          toBias: 0.74,
          style: 'elbow-h',
          corridorRight: 48,
          labelDy: -14,
        },
        {
          id: 'aic-aiv2-ssbuffer',
          label: 'SSBuffer',
          tone: 'direct',
          from: '#mem950-aic [data-aic-node="scalar:Scalar"]',
          to: '#mem950-aiv2 [data-aiv-node="scalar:Scalar"]',
          fromSide: 'bottom',
          toSide: 'top',
          style: 'elbow-v',
          defaultHidden: true,
        },
        {
          id: 'l2-to-aiv2',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aiv2 [data-aiv-node="cache:ND-DMA Cache"]',
          fromSide: 'right',
          toSide: 'left',
          toAnchorSelector: '.pto-aiv-core__grid',
          toBias: 0.66,
          style: 'lane-h-target',
          labelDy: 10,
        },
        {
          id: 'gm-to-aiv2-ub',
          label: 'GM→ND-DMA',
          tone: 'transport',
          from: '[data-mem950-node="rail:GM"]',
          to: '#mem950-aiv2 [data-aiv-node="cache:ND-DMA Cache"]',
          fromSide: 'right',
          toSide: 'left',
          toAnchorSelector: '.pto-aiv-core__grid',
          toBias: 0.48,
          style: 'lane-h-target',
          labelDy: -12,
          defaultHidden: true,
        },
        {
          id: 'aiv2-ub-to-gm',
          label: 'UB→GM',
          tone: 'directReturn',
          from: '#mem950-aiv2 [data-aiv-node="buffer:UB"]',
          to: '[data-mem950-node="rail:GM"]',
          fromSide: 'left',
          toSide: 'right',
          fromAnchorSelector: '.pto-aiv-core__grid',
          fromBias: 0.82,
          sourceLaneBelowSelector: '#mem950-aiv2 [data-aiv-node="cache:ICache"]',
          sourceLaneOffset: 14,
          style: 'lane-h-source',
          labelDy: 14,
          defaultHidden: true,
        },
        {
          id: 'gm-to-aiv2-simt',
          label: 'GM→SIMT RF',
          tone: 'simt',
          from: '[data-mem950-node="rail:GM"]',
          to: '#mem950-aiv2 [data-aiv-node="exec:SIMT"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.42,
          style: 'lane-h-target',
          labelDy: -13,
          defaultHidden: true,
        },
        {
          id: 'l2-to-aiv2-dcache',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aiv2 [data-aiv-node="cache:DCache"]',
          fromSide: 'right',
          toSide: 'left',
          toAnchorSelector: '.pto-aiv-core__grid',
          toBias: 0.82,
          style: 'lane-h-target',
          labelDy: 10,
        },
        {
          id: 'aiv2-to-l2',
          label: 'MTE3',
          tone: 'transport',
          from: '#mem950-aiv2 [data-aiv-node="buffer:UB"]',
          to: '[data-mem950-node="rail:L2"]',
          fromSide: 'left',
          toSide: 'right',
          fromAnchorSelector: '.pto-aiv-core__grid',
          fromBias: 0.82,
          sourceLaneBelowSelector: '#mem950-aiv2 [data-aiv-node="cache:ICache"]',
          sourceLaneOffset: 14,
          style: 'lane-h-source',
          labelDy: 0,
        },
      ],
      notes: [
        '1 AIC + 2 AIV memory-stage layout',
        'L2/GM → DCache, L1, or ND-DMA Cache via MTE2',
        'ND-DMA Cache → UB for GM/L2 input staging',
        'UB → L2/GM via MTE3',
        'GM → UB → SIMT RF / GM → SIMT RF',
        'GM ↔ UB ↔ Vector RF; GM → L0A/L0B',
        '950 direct CV lanes: L0C→UB and UB→L1',
      ],
      details: [
        {
          selector: '[data-aiv-node="buffer:UB"]',
          rows: [
            ['bank', '8组 x 2个/组'],
            ['单bank', '16KB'],
            ['cache', 'ND-DMA Cache / SIMT DCache'],
            ['对齐', '32B'],
            ['搬运', 'MTE2/MTE3'],
          ],
          bankGrid: { groups: 8, banksPerGroup: 2 },
        },
        {
          selector: '#mem950-aiv1 .pto-aiv-core__instruction-slot, #mem950-aiv2 .pto-aiv-core__instruction-slot',
          instructionItems: ['MTE2 指令序列', 'MTE3 指令序列', 'SIMD VF 指令序列', 'SIMT VF 指令序列'],
        },
        {
          selector: '#mem950-aic .pto-aic-core__bottom-row',
          instructionItems: ['Cube 指令序列', 'FixPipe 指令序列', 'MTE1 指令序列', 'MTE2 指令序列'],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:L1"]',
          rows: [
            ['对齐', '32B'],
            ['建议布局', 'NZ'],
          ],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:L0A"]',
          rows: [
            ['搬运对齐', '512B'],
            ['推荐布局', 'NZ'],
          ],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:L0B"]',
          rows: [
            ['搬运对齐', '512B'],
            ['推荐布局', 'ZN'],
          ],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:FP"]',
          rows: [
            ['流水', 'FixPipe'],
            ['输出', '量化/激活'],
          ],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:L0C"]',
          rows: [
            ['搬运对齐', '64B'],
            ['推荐布局', 'NZ'],
          ],
        },
      ],
      hoverTips: {
        'rail:GM': {
          title: '全局内存（GM）',
          body: '芯片级主存，是 950 显式数据通路的源端和回写端；Vector、SIMT 与 Cube 场景都要关注从 GM 到片上缓存或寄存器的搬运成本。',
        },
        'rail:L2': {
          title: '二级缓存（L2）',
          body: '共享缓存层，连接 AIV 的 DCache、ND-DMA Cache 与 AIC 的 DCache/L1。GM/L2 到 UB 的 MTE2 搬运先进入 ND-DMA Cache，再写入 UB。',
        },
        'core:AIV1': {
          title: 'AIV 1',
          body: '向量侧计算核心，包含 DCache、ICache、ND-DMA Cache、Scalar、UB、SIMT、SIMD 和 Vector Reg File，用于规则向量计算与离散 SIMT 场景。',
        },
        'core:AIC': {
          title: 'AIC',
          body: '矩阵计算侧核心，包含 L1、L0A/L0B/L0C、BT/FP、CUBE、Scalar、Dispatch 和指令队列。',
        },
        'core:AIV2': {
          title: 'AIV 2',
          body: '第二个向量侧计算核心；在折叠视图中用于表达 AIV ×2 的合并结构。',
        },
        'buffer:UB': {
          title: '统一缓冲区（UB）',
          body: 'AIV 本地数据暂存区。950 场景中，GM/L2 输入经 ND-DMA Cache 进入 UB，UB 再与 SIMT Reg File 或 Vector Reg File 形成关键通路。',
        },
        'buffer:L1': {
          title: '一级片上缓存（L1）',
          body: 'AIC 本地输入缓存，向 L0A、L0B、BT 和 FP 等下一级缓冲区供数。',
        },
        'buffer:L0A': {
          title: 'L0A 输入缓存',
          body: 'Cube 计算的矩阵 A 操作数缓存；950 差异解读中重点关注 GM 到 L0A 的主数据通路。',
        },
        'buffer:L0B': {
          title: 'L0B 输入缓存',
          body: 'Cube 计算的矩阵 B 操作数缓存；950 差异解读中重点关注 GM 到 L0B 的主数据通路。',
        },
        'buffer:L0C': {
          title: 'L0C 输出缓存',
          body: 'Cube 计算结果缓存。950 的 C-V 直连可将 L0C 结果转发到 AIV 的 UB，减少经 GM 中转的开销。',
        },
        'buffer:BT': {
          title: 'BT 缓冲区',
          body: 'AIC 侧偏置或变换相关的本地缓冲区，通常通过 MTE1 与其他本地缓存连接。',
        },
        'buffer:FP': {
          title: 'FixPipe 缓冲区',
          body: 'Cube 后处理相关的本地缓冲区，用于量化、激活或格式整理等 FixPipe 数据流。',
        },
        'cache:DCache': {
          title: '数据缓存（DCache）',
          body: '数据访问缓存端点，承接访存搬运和标量控制路径中的数据访问。',
        },
        'cache:ICache': {
          title: '指令缓存（ICache）',
          body: '指令访问缓存，为 Scalar 或调度侧控制路径提供指令流。',
        },
        'cache:ND-DMA Cache': {
          title: 'ND-DMA Cache',
          body: '面向 ND 数据搬运的 DMA 缓存单元，位于 GM/L2 到 UB 的 MTE2 输入路径上。它不是计算 buffer，而是 CopyIn 进入 UB 前的搬运缓存节点。',
        },
        'scalar:Scalar': {
          title: '标量控制单元',
          body: '负责协调本地计算、数据搬运和队列派发，是 AIC/AIV 内部控制路径的核心节点。',
        },
        'exec:SIMT': {
          title: 'SIMT 寄存器文件',
          body: '面向离散数据和多线程执行的寄存器路径，可表达 GM→UB→SIMT Reg File 或 GM→SIMT Reg File 的访问方式。',
        },
        'exec:SIMD': {
          title: 'SIMD 寄存器文件',
          body: '面向连续、规则数据的向量寄存器路径，适合规则向量计算和 RegBase 风格的数据组织。',
        },
        'vector:Vector': {
          title: 'Vector 寄存器文件',
          body: 'AIV 向量计算寄存器文件，与 UB 之间存在双向数据通路，用于向量侧暂存、计算和回写。',
        },
        'cube:CUBE': {
          title: 'Cube 计算单元',
          body: 'AIC 矩阵计算单元，主要从 L0A、L0B 和 BT 等缓存获取操作数。',
        },
        'scheduler:Dispatch': {
          title: '派发单元',
          body: 'AIC 内部派发节点，负责把工作分发到 Cube、FixPipe 和 MTE 等指令队列。',
        },
        'queue:Cube Queue': {
          title: 'Cube 指令队列',
          body: '承接 Cube 矩阵计算相关指令的队列。',
        },
        'queue:FixPipe Queue': {
          title: 'FixPipe 指令队列',
          body: '承接 Cube 后处理、格式整理、量化或激活相关指令的队列。',
        },
        'queue:MTE1 Queue': {
          title: 'MTE1 指令队列',
          body: '承接 AIC 本地缓存层级之间数据搬运工作的队列。',
        },
        'queue:MTE2 Queue': {
          title: 'MTE2 指令队列',
          body: '承接 L2 与本地缓存之间数据搬运工作的队列。',
        },
        'transport:MTE1': {
          title: 'MTE1 搬运通路',
          body: 'AIC 本地搬运通路，连接 L1、L0 缓冲区和 FixPipe 相关缓冲区。',
        },
        'transport:FixPipe': {
          title: 'FixPipe 通路',
          body: 'AIC 后处理通路，用于 Cube 结果后的格式、量化或激活处理。',
        },
      },
    },
    ascend910bLegacyDuplicate: {
      id: 'ascend910bLegacyDuplicate',
      name: 'Ascend 910B Memory Architecture (legacy duplicate)',
      rails: [
        {
          key: 'GM',
          label: 'Global Memory',
          tone: 'memory-shell',
          grid: { rows: 82, cols: 8, cellSize: 12, gap: 4, shape: 'hex' },
        },
        {
          key: 'L2',
          label: 'L2 Cache',
          tone: 'memory-rail',
          grid: { rows: 82, cols: 4, cellSize: 12, gap: 4, shape: 'dot' },
        },
      ],
      cores: [
        {
          id: 'mem950-aic',
          kind: 'aic',
          title: 'AIC',
          presetKey: 'aicDraftV1',
        },
        {
          id: 'mem950-aiv1',
          kind: 'aiv',
          title: 'AIV',
          presetKey: 'ascend910b',
        },
      ],
      routes: [
        {
          id: 'l2-to-aiv1-scalar',
          label: '',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aiv1 [data-aiv-node="scalar:Scalar"]',
          fromSide: 'right',
          fromAnchorSelector: '.pto-mem950__rail-grid',
          fromDx: -24,
          toSide: 'left',
          toBias: 0.50,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'l2-to-aiv1',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aiv1 [data-aiv-node="buffer:UB"]',
          fromSide: 'right',
          fromAnchorSelector: '.pto-mem950__rail-grid',
          fromDx: -24,
          toSide: 'left',
          toAnchorSelector: '.pto-aiv-core__grid',
          toBias: 0.40,
          style: 'lane-h-target',
          labelDy: -7,
        },
        {
          id: 'aiv1-to-l2',
          label: 'MTE3',
          tone: 'transport',
          from: '#mem950-aiv1 [data-aiv-node="buffer:UB"]',
          to: '[data-mem950-node="rail:L2"]',
          fromSide: 'left',
          toSide: 'right',
          fromAnchorSelector: '.pto-aiv-core__grid',
          toAnchorSelector: '.pto-mem950__rail-grid',
          toDx: -24,
          fromBias: 0.60,
          style: 'lane-h-source',
          labelDy: 7,
        },
        {
          id: 'l2-to-aic',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aic [data-aic-node="buffer:L1"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.24,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'l2-to-aic-dcache',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aic [data-aic-node="cache:DCache"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.50,
          style: 'lane-h-target',
          labelDy: 0,
          defaultHidden: true,
        },
      ],
      notes: [
        '1 AIC + 1 AIV memory-stage layout (910B)',
        'AIV is simplified to Scalar + UB + Vector',
        'L2/GM → Scalar, UB, or AIC L1 via MTE2',
        'UB → L2/GM via MTE3',
        'No 950 direct CV lanes; no separate SIMT/SIMD cards',
      ],
      details: [
        {
          selector: '[data-aiv-node="buffer:UB"]',
          rows: [
            ['驻留', 'AIV 本地'],
            ['执行', 'SIMD'],
            ['对齐', '32B'],
            ['搬运', 'MTE2/MTE3'],
          ],
        },
        {
          selector: '[data-aiv-node="exec:SIMD"]',
          rows: [
            ['调度', 'Vector Pipe'],
            ['SIMT', '无'],
          ],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:L1"]',
          rows: [
            ['对齐', '32B'],
            ['角色', 'AIC 输入'],
          ],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:L0A"]',
          rows: [
            ['搬运对齐', '512B'],
            ['推荐布局', 'NZ'],
          ],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:L0B"]',
          rows: [
            ['搬运对齐', '512B'],
            ['推荐布局', 'ZN'],
          ],
        },
        {
          selector: '#mem950-aic [data-aic-node="buffer:L0C"]',
          rows: [
            ['角色', 'Cube 输出'],
            ['回写', '经 GM/L2'],
          ],
        },
      ],
      hoverTips: {
        'rail:GM': {
          title: '全局内存（GM）',
          body: '910B 架构中的芯片级主存，是算子输入、输出和中间数据回写的主要位置。',
        },
        'rail:L2': {
          title: '二级缓存（L2）',
          body: '共享缓存层，通过 MTE2 等通路向 AIV 的 DCache/UB 或 AIC 的 DCache/L1 供数。',
        },
        'core:AIV': {
          title: 'AIV',
          body: '910B 的向量侧计算核心，包含 DCache、ICache、Scalar、UB、SIMD 和 Vector 通路，不包含 950 的 SIMT 结构。',
        },
        'core:AIC': {
          title: 'AIC',
          body: '矩阵计算侧核心，包含 L1、L0A/L0B/L0C、CUBE、Scalar、Dispatch 和执行队列。',
        },
        'buffer:UB': {
          title: '统一缓冲区（UB）',
          body: 'AIV 向量侧本地数据暂存区，服务于向量计算和 MTE3 回写路径。',
        },
        'buffer:L1': {
          title: '一级片上缓存（L1）',
          body: 'AIC 本地输入缓存，向 L0A、L0B、BT 和 FP 等下一级缓冲区供数。',
        },
        'buffer:L0A': {
          title: 'L0A 输入缓存',
          body: 'Cube 计算的矩阵 A 操作数缓存。',
        },
        'buffer:L0B': {
          title: 'L0B 输入缓存',
          body: 'Cube 计算的矩阵 B 操作数缓存。',
        },
        'buffer:L0C': {
          title: 'L0C 输出缓存',
          body: 'Cube 计算结果缓存，通常用于后续回写或后处理。',
        },
        'buffer:BT': {
          title: 'BT 缓冲区',
          body: 'AIC 侧偏置或变换相关的本地缓冲区，通常通过 MTE1 与其他本地缓存连接。',
        },
        'buffer:FP': {
          title: 'FixPipe 缓冲区',
          body: 'Cube 后处理相关的本地缓冲区，用于格式整理、量化或激活等数据流。',
        },
        'cache:DCache': {
          title: '数据缓存（DCache）',
          body: '数据访问缓存端点，承接访存搬运和标量控制路径中的数据访问。',
        },
        'cache:ICache': {
          title: '指令缓存（ICache）',
          body: '指令访问缓存，为 Scalar 或调度侧控制路径提供指令流。',
        },
        'scalar:Scalar': {
          title: '标量控制单元',
          body: '负责协调本地计算、数据搬运和队列派发，是 AIC/AIV 内部控制路径的核心节点。',
        },
        'exec:SIMD': {
          title: 'SIMD 执行通路',
          body: 'AIV 规则向量计算通路，连接 UB 数据暂存和向量计算输出。',
        },
        'vector:Vector': {
          title: 'Vector 计算端点',
          body: 'AIV 向量计算端点，接收 SIMD 通路的计算结果。',
        },
        'cube:CUBE': {
          title: 'Cube 计算单元',
          body: 'AIC 矩阵计算单元，主要从 L0A、L0B 和 BT 等缓存获取操作数。',
        },
        'scheduler:Dispatch': {
          title: '派发单元',
          body: 'AIC 内部派发节点，负责把工作分发到 Cube、FixPipe 和 MTE 等执行队列。',
        },
        'queue:Cube Queue': {
          title: 'Cube 指令队列',
          body: '承接 Cube 矩阵计算相关指令的队列。',
        },
        'queue:FixPipe Queue': {
          title: 'FixPipe 指令队列',
          body: '承接 Cube 后处理、格式整理、量化或激活相关指令的队列。',
        },
        'queue:MTE1 Queue': {
          title: 'MTE1 指令队列',
          body: '承接 AIC 本地缓存层级之间数据搬运工作的队列。',
        },
        'queue:MTE2 Queue': {
          title: 'MTE2 指令队列',
          body: '承接 L2 与本地缓存之间数据搬运工作的队列。',
        },
        'transport:MTE1': {
          title: 'MTE1 搬运通路',
          body: 'AIC 本地搬运通路，连接 L1、L0 缓冲区和 FixPipe 相关缓冲区。',
        },
        'transport:FixPipe': {
          title: 'FixPipe 通路',
          body: 'AIC 后处理通路，用于 Cube 结果后的格式、量化或激活处理。',
        },
      },
    },
    ascend910bObsoleteDuplicate: {
      id: 'ascend910bObsoleteDuplicate',
      name: 'Ascend 910B Memory Architecture (obsolete duplicate)',
      rails: [
        {
          key: 'GM',
          label: 'Global Memory',
          tone: 'memory-shell',
          grid: { rows: 82, cols: 8, cellSize: 12, gap: 4, shape: 'hex' },
        },
        {
          key: 'L2',
          label: 'L2 Cache',
          tone: 'memory-rail',
          grid: { rows: 82, cols: 4, cellSize: 12, gap: 4, shape: 'dot' },
        },
      ],
      cores: [
        {
          id: 'mem950-aiv1',
          kind: 'aiv',
          title: 'AIV',
          presetKey: 'aivLegacyV1',
        },
        {
          id: 'mem950-aic',
          kind: 'aic',
          title: 'AIC',
          presetKey: 'aicDraftV1',
        },
      ],
      routes: [
        {
          id: 'l2-to-aiv1-dcache',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aiv1 [data-aiv-node="cache:DCache"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.50,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'l2-to-aiv1',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aiv1 [data-aiv-node="buffer:UB"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.50,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'aiv1-to-l2',
          label: 'MTE3',
          tone: 'transport',
          from: '#mem950-aiv1 [data-aiv-node="buffer:UB"]',
          to: '[data-mem950-node="rail:L2"]',
          fromSide: 'left',
          toSide: 'right',
          fromBias: 0.82,
          style: 'lane-h-source',
          labelDy: 0,
        },
        {
          id: 'l2-to-aic',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aic [data-aic-node="buffer:L1"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.58,
          style: 'lane-h-target',
          labelDy: 0,
        },
        {
          id: 'l2-to-aic-dcache',
          label: 'MTE2',
          tone: 'transport',
          from: '[data-mem950-node="rail:L2"]',
          to: '#mem950-aic [data-aic-node="cache:DCache"]',
          fromSide: 'right',
          toSide: 'left',
          toBias: 0.50,
          style: 'lane-h-target',
          labelDy: 0,
        },
      ],
      notes: [
        '1 AIC + 1 AIV memory-stage layout (910B)',
        'L2/GM → DCache, L1, or UB via MTE2',
        'UB → L2/GM via MTE3',
        'No 950 direct CV lanes; SIMD-only vector exec (no SIMT)',
      ],
      hoverTips: {
        'rail:GM': {
          title: 'Global Memory',
          body: 'Chip-level memory source and sink for the 910B architecture.',
        },
        'rail:L2': {
          title: 'L2 Cache',
          body: 'Shared cache rail feeding the simplified AIV Scalar/UB path and AIC DCache or L1 through MTE2 paths.',
        },
        'core:AIV': {
          title: 'AIV',
          body: '910B vector-side core object simplified to Scalar, UB, and Vector lanes.',
        },
        'core:AIC': {
          title: 'AIC',
          body: 'Cube-side compute object with L1, L0 buffers, CUBE, Scalar, dispatch, and execution queues.',
        },
        'buffer:UB': {
          title: 'UB',
          body: 'Unified Buffer for AIV vector-side data staging and MTE3 return paths.',
        },
        'buffer:L1': {
          title: 'L1',
          body: 'AIC local memory feeding L0A, L0B, BT, and FP lanes.',
        },
        'buffer:L0A': {
          title: 'L0A',
          body: 'AIC matrix operand buffer for CUBE input staging.',
        },
        'buffer:L0B': {
          title: 'L0B',
          body: 'AIC matrix operand buffer for CUBE input staging.',
        },
        'buffer:L0C': {
          title: 'L0C',
          body: 'AIC CUBE output buffer.',
        },
        'buffer:BT': {
          title: 'BT',
          body: 'AIC bias or transform-side buffer lane connected through MTE1.',
        },
        'buffer:FP': {
          title: 'FP',
          body: 'AIC FixPipe buffer lane for post-CUBE data movement.',
        },
        'cache:DCache': {
          title: 'DCache',
          body: 'Data cache endpoint for memory transport and scalar-side access.',
        },
        'cache:ICache': {
          title: 'ICache',
          body: 'Instruction cache feeding the scalar or scheduler-side control path.',
        },
        'scalar:Scalar': {
          title: 'Scalar',
          body: 'Scalar control block coordinating local compute and memory movement.',
        },
        'exec:SIMD': {
          title: 'SIMD',
          body: 'AIV SIMD execution lane connected to UB data and vector output.',
        },
        'vector:Vector': {
          title: 'Vector',
          body: 'AIV vector execution endpoint receiving SIMD results.',
        },
        'cube:CUBE': {
          title: 'CUBE',
          body: 'AIC matrix compute block fed by L0A, L0B, and BT buffers.',
        },
        'scheduler:Dispatch': {
          title: 'Dispatch',
          body: 'AIC dispatch block issuing work into cube, FixPipe, and MTE queues.',
        },
        'transport:MTE1': {
          title: 'MTE1',
          body: 'Local AIC transport lane between L1 and L0 or FixPipe buffers.',
        },
        'transport:FixPipe': {
          title: 'FixPipe',
          body: 'AIC post-processing transport lane.',
        },
      },
    },
  };

  if (PRESETS.ascend910bLegacyDuplicate) {
    PRESETS.ascend910b = {
      ...PRESETS.ascend910bLegacyDuplicate,
      id: 'ascend910b',
      name: 'Ascend 910B Memory Architecture',
    };
    delete PRESETS.ascend910bLegacyDuplicate;
    delete PRESETS.ascend910bObsoleteDuplicate;
  }

  function resolvePreset(presetOrKey) {
    if (typeof presetOrKey === 'string') return PRESETS[presetOrKey] || null;
    return presetOrKey || null;
  }

  function node(tagName, className, textContent) {
    const el = document.createElement(tagName);
    if (className) el.className = className;
    if (textContent !== undefined) el.textContent = textContent;
    return el;
  }

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function readableNodeKey(key) {
    const [, label = key] = String(key || '').split(/:(.*)/);
    return label || String(key || '');
  }

  function tipForTarget(target, preset, options = {}) {
    const key = target.dataset.mem950Node || target.dataset.aicNode || target.dataset.aivNode || '';
    const custom = options.getTip?.(key, target, preset);
    const tip = custom || options.hoverTips?.[key] || preset?.hoverTips?.[key];
    if (typeof tip === 'string') {
      return {
        title: readableNodeKey(key),
        body: tip,
      };
    }
    if (tip) {
      return {
        title: tip.title || readableNodeKey(key),
        body: tip.body || tip.description || '',
      };
    }
    return {
      title: readableNodeKey(key),
      body: `该节点属于 ${preset?.name || '当前硬件架构图'}，用于表达对应的片上存储、计算或调度位置。`,
    };
  }

  function attrValue(value) {
    return String(value || '').replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  }

  function rootFor(container) {
    return container?.querySelector?.('.pto-mem950') || container || null;
  }

  function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function clamp(number, min, max) {
    return Math.max(min, Math.min(max, number));
  }

  function resolveElement(value, root = document) {
    if (!value) return null;
    if (typeof value === 'string') return root.querySelector(value);
    return value;
  }

  function createZoomController(options = {}) {
    const root = options.root || document;
    const viewport = resolveElement(options.viewport || '[data-pto-mem-arch-viewport]', root);
    const sizer = resolveElement(options.sizer || '[data-pto-mem-arch-sizer]', root);
    const canvas = resolveElement(options.canvas || '[data-pto-mem-arch-canvas]', root);
    if (!canvas) return null;

    const readout = resolveElement(options.readout || '[data-pto-mem-zoom-readout]', root);
    const outButton = resolveElement(options.outButton || '[data-pto-mem-zoom="out"]', root);
    const inButton = resolveElement(options.inButton || '[data-pto-mem-zoom="in"]', root);
    const resetButton = resolveElement(options.resetButton || '[data-pto-mem-zoom="reset"]', root);
    const min = finiteNumber(options.min, ZOOM_DEFAULTS.min);
    const max = finiteNumber(options.max, ZOOM_DEFAULTS.max);
    const step = finiteNumber(options.step, ZOOM_DEFAULTS.step);
    const defaultZoom = clamp(finiteNumber(
      options.defaultZoom ?? viewport?.dataset.defaultZoom ?? canvas.dataset.defaultZoom,
      ZOOM_DEFAULTS.defaultZoom,
    ), min, max);
    let zoom = clamp(finiteNumber(options.zoom, defaultZoom), min, max);
    let panX = finiteNumber(options.panX, 0);
    let panY = finiteNumber(options.panY, 0);
    let activePan = null;
    let frame = 0;
    const panEnabled = options.pan !== false && Boolean(viewport);
    const wheelZoomEnabled = options.wheelZoom !== false && Boolean(viewport);

    const naturalSize = () => {
      const content = canvas.firstElementChild || canvas;
      return {
        width: Math.max(canvas.scrollWidth, content.scrollWidth, 1),
        height: Math.max(canvas.scrollHeight, content.scrollHeight, 1),
      };
    };

    const syncPanVars = () => {
      const nextX = Number(panX.toFixed(2));
      const nextY = Number(panY.toFixed(2));
      canvas.style.setProperty('--pto-memory-architecture-pan-x', `${nextX}px`);
      canvas.style.setProperty('--pto-memory-architecture-pan-y', `${nextY}px`);
      canvas.dataset.ptoMemoryArchitecturePanX = String(nextX);
      canvas.dataset.ptoMemoryArchitecturePanY = String(nextY);
      if (viewport) {
        viewport.dataset.ptoMemoryArchitecturePanX = String(nextX);
        viewport.dataset.ptoMemoryArchitecturePanY = String(nextY);
      }
    };

    const apply = () => {
      const nextZoom = Number(zoom.toFixed(3));
      canvas.style.setProperty('--pto-memory-architecture-zoom', String(nextZoom));
      canvas.dataset.ptoMemoryArchitectureZoom = String(nextZoom);
      if (viewport) viewport.dataset.ptoMemoryArchitectureZoom = String(nextZoom);
      syncPanVars();

      const size = naturalSize();
      const scaledWidth = Math.max(1, Math.ceil(size.width * nextZoom));
      const scaledHeight = Math.max(1, Math.ceil(size.height * nextZoom));
      if (sizer) {
        sizer.style.width = `${scaledWidth}px`;
        sizer.style.height = `${scaledHeight}px`;
      }

      if (readout) readout.textContent = `${Math.round(nextZoom * 100)}%`;
      if (outButton) outButton.disabled = nextZoom <= min + 0.001;
      if (inButton) inButton.disabled = nextZoom >= max - 0.001;
      options.onZoom?.({
        zoom: nextZoom,
        panX,
        panY,
        width: scaledWidth,
        height: scaledHeight,
      });
    };

    const schedule = () => {
      if (frame) global.cancelAnimationFrame?.(frame);
      frame = global.requestAnimationFrame?.(() => {
        frame = 0;
        apply();
      }) || 0;
    };

    const setZoom = (next) => {
      zoom = clamp(finiteNumber(next, zoom), min, max);
      apply();
    };
    const increment = (direction) => setZoom(zoom + step * direction);
    const setPan = (nextX, nextY) => {
      panX = finiteNumber(nextX, panX);
      panY = finiteNumber(nextY, panY);
      syncPanVars();
      options.onPan?.({
        zoom: Number(zoom.toFixed(3)),
        panX,
        panY,
      });
    };
    const centerTargets = () => {
      if (!options.centerTarget) return [canvas.firstElementChild || canvas];
      const targetOption = Array.isArray(options.centerTarget)
        ? options.centerTarget
        : [options.centerTarget];
      const targets = targetOption.flatMap((target) => {
        if (!target) return [];
        if (typeof target === 'string') return Array.from(canvas.querySelectorAll(target));
        return [target];
      });
      return targets.filter((target) => target instanceof Element && canvas.contains(target));
    };
    const center = () => {
      if (!viewport) return;
      const targets = centerTargets();
      if (targets.length === 0) return;
      const rect = viewport.getBoundingClientRect();
      const boxes = targets.map((target) => target.getBoundingClientRect())
        .filter((box) => box.width > 0 && box.height > 0);
      if (boxes.length === 0) return;
      const bounds = {
        left: Math.min(...boxes.map((box) => box.left)),
        top: Math.min(...boxes.map((box) => box.top)),
        right: Math.max(...boxes.map((box) => box.right)),
        bottom: Math.max(...boxes.map((box) => box.bottom)),
      };
      const targetWidth = bounds.right - bounds.left;
      const targetHeight = bounds.bottom - bounds.top;
      const currentLeft = bounds.left - rect.left;
      const currentTop = bounds.top - rect.top;
      setPan(
        panX + Math.max(0, (rect.width - targetWidth) / 2) - currentLeft,
        panY + Math.max(0, (rect.height - targetHeight) / 2) - currentTop,
      );
    };
    const reset = () => {
      zoom = defaultZoom;
      apply();
      if (options.centerOnReset === true) {
        center();
      } else {
        setPan(0, 0);
      }
    };

    const zoomAtPoint = (next, clientX, clientY) => {
      const nextZoom = clamp(finiteNumber(next, zoom), min, max);
      if (!viewport || nextZoom === zoom) {
        setZoom(nextZoom);
        return;
      }
      const rect = viewport.getBoundingClientRect();
      const anchorX = (clientX - rect.left - panX) / zoom;
      const anchorY = (clientY - rect.top - panY) / zoom;
      zoom = nextZoom;
      panX = clientX - rect.left - anchorX * zoom;
      panY = clientY - rect.top - anchorY * zoom;
      apply();
      options.onPan?.({
        zoom: Number(zoom.toFixed(3)),
        panX,
        panY,
      });
    };

    const onOut = () => increment(-1);
    const onIn = () => increment(1);
    const onReset = () => reset();
    outButton?.addEventListener('click', onOut);
    inButton?.addEventListener('click', onIn);
    resetButton?.addEventListener('click', onReset);

    const canPanTarget = (target) => !target?.closest?.('button, a, input, select, textarea, [data-no-pan]');
    const onPointerDown = (event) => {
      if (!panEnabled || event.button !== 0 || !canPanTarget(event.target)) return;
      activePan = {
        pointerId: event.pointerId,
        clientX: event.clientX,
        clientY: event.clientY,
        panX,
        panY,
      };
      viewport.classList.add('is-panning');
      viewport.setPointerCapture?.(event.pointerId);
      event.preventDefault();
    };
    const onPointerMove = (event) => {
      if (!activePan || activePan.pointerId !== event.pointerId) return;
      setPan(
        activePan.panX + event.clientX - activePan.clientX,
        activePan.panY + event.clientY - activePan.clientY,
      );
    };
    const stopPointerPan = (event) => {
      if (!activePan || activePan.pointerId !== event.pointerId) return;
      viewport.releasePointerCapture?.(event.pointerId);
      activePan = null;
      viewport.classList.remove('is-panning');
    };
    const onWheel = (event) => {
      if (!wheelZoomEnabled || !event.metaKey) return;
      event.preventDefault();
      const direction = event.deltaY > 0 ? -1 : 1;
      const magnitude = Math.min(3, Math.max(1, Math.abs(event.deltaY) / 120));
      zoomAtPoint(zoom + step * direction * magnitude, event.clientX, event.clientY);
    };

    if (viewport) {
      viewport.addEventListener('pointerdown', onPointerDown);
      viewport.addEventListener('pointermove', onPointerMove);
      viewport.addEventListener('pointerup', stopPointerPan);
      viewport.addEventListener('pointercancel', stopPointerPan);
      viewport.addEventListener('wheel', onWheel, { passive: false });
    }

    const resizeObserver = typeof ResizeObserver === 'function'
      ? new ResizeObserver(schedule)
      : null;
    resizeObserver?.observe(canvas);
    if (canvas.firstElementChild) resizeObserver?.observe(canvas.firstElementChild);

    apply();

    return {
      getZoom: () => zoom,
      getPan: () => ({ x: panX, y: panY }),
      setZoom,
      setPan,
      center,
      zoomAtPoint,
      increment,
      reset,
      render: apply,
      destroy() {
        outButton?.removeEventListener('click', onOut);
        inButton?.removeEventListener('click', onIn);
        resetButton?.removeEventListener('click', onReset);
        viewport?.removeEventListener('pointerdown', onPointerDown);
        viewport?.removeEventListener('pointermove', onPointerMove);
        viewport?.removeEventListener('pointerup', stopPointerPan);
        viewport?.removeEventListener('pointercancel', stopPointerPan);
        viewport?.removeEventListener('wheel', onWheel);
        resizeObserver?.disconnect();
        if (frame) global.cancelAnimationFrame?.(frame);
      },
    };
  }

  function activeRouteEndpoints(routeIds, preset) {
    const ids = new Set(routeIds || []);
    return (preset?.routes || [])
      .filter((route) => ids.has(route.id))
      .flatMap((route) => [route.from, route.to])
      .filter(Boolean);
  }

  function clearPathFocus(container) {
    const root = rootFor(container);
    if (!root) return;
    root.classList.remove('is-path-focused');
    root.querySelectorAll('.is-hardware-active').forEach((el) => el.classList.remove('is-hardware-active'));
    root.querySelectorAll('.is-diagnostic-error-target').forEach((el) => el.classList.remove('is-diagnostic-error-target'));
    root.querySelectorAll('.is-route-active').forEach((el) => el.classList.remove('is-route-active'));
    root.querySelectorAll('.is-internal-route-active').forEach((el) => el.classList.remove('is-internal-route-active'));
  }

  function setDetailVisibility(container, visible = true) {
    const root = rootFor(container);
    if (!root) return null;
    const nextVisible = visible !== false;
    root.classList.toggle('is-detail-hidden', !nextVisible);
    root.dataset.detailVisibility = nextVisible ? 'bank' : 'base';
    return {
      root,
      visible: nextVisible,
    };
  }

  function setAivFolded(container, folded = false) {
    const root = rootFor(container);
    if (!root) return null;
    const nextFolded = folded === true;
    root.classList.toggle('is-aiv-folded', nextFolded);
    root.dataset.aivFolded = nextFolded ? 'true' : 'false';
    return {
      root,
      folded: nextFolded,
    };
  }

  function applyInternalRouteFocus(root) {
    root.querySelectorAll('.pto-aic-core__route[data-aic-route-from][data-aic-route-to]').forEach((routeEl) => {
      const core = routeEl.closest('.pto-aic-core');
      const fromEl = core?.querySelector(`[data-aic-node="${attrValue(routeEl.dataset.aicRouteFrom)}"]`);
      const toEl = core?.querySelector(`[data-aic-node="${attrValue(routeEl.dataset.aicRouteTo)}"]`);
      const isActive = Boolean(
        fromEl?.classList.contains('is-hardware-active') &&
        toEl?.classList.contains('is-hardware-active')
      );
      routeEl.classList.toggle('is-internal-route-active', isActive);
      if (isActive) {
        core?.querySelectorAll(`[data-aic-transport-to="${attrValue(routeEl.dataset.aicRouteTo)}"]`)
          .forEach((pill) => pill.classList.add('is-internal-route-active'));
      }
    });

    root.querySelectorAll('.pto-aiv-core__route[data-aiv-route-from][data-aiv-route-to]').forEach((routeEl) => {
      const core = routeEl.closest('.pto-aiv-core');
      const fromEl = core?.querySelector(`[data-aiv-node="${attrValue(routeEl.dataset.aivRouteFrom)}"]`);
      const toEl = core?.querySelector(`[data-aiv-node="${attrValue(routeEl.dataset.aivRouteTo)}"]`);
      const isActive = Boolean(
        fromEl?.classList.contains('is-hardware-active') &&
        toEl?.classList.contains('is-hardware-active')
      );
      routeEl.classList.toggle('is-internal-route-active', isActive);
    });
  }

  function setPathFocus(container, presetOrKey, focus = {}) {
    const preset = resolvePreset(presetOrKey);
    const root = rootFor(container);
    if (!root || !preset) return null;

    const selectors = Array.from(new Set([
      ...(focus.selectors || []),
      ...activeRouteEndpoints(focus.routes || focus.routeIds || [], preset),
    ].filter(Boolean)));
    const routeIds = Array.from(new Set((focus.routes || focus.routeIds || []).filter(Boolean)));
    const errorSelectors = Array.from(new Set((focus.errorSelectors || []).filter(Boolean)));

    clearPathFocus(root);
    root.classList.toggle('is-path-focused', selectors.length > 0 || routeIds.length > 0 || errorSelectors.length > 0);

    selectors.forEach((selector) => {
      root.querySelectorAll(selector).forEach((el) => el.classList.add('is-hardware-active'));
    });

    errorSelectors.forEach((selector) => {
      root.querySelectorAll(selector).forEach((el) => {
        el.classList.add('is-hardware-active');
        el.classList.add('is-diagnostic-error-target');
      });
    });

    routeIds.forEach((routeId) => {
      root.querySelectorAll(`[data-route-id="${attrValue(routeId)}"]`).forEach((el) => el.classList.add('is-route-active'));
    });

    applyInternalRouteFocus(root);

    return {
      root,
      selectors,
      errorSelectors,
      routes: routeIds,
    };
  }

  function selectorForHardwareTarget(target) {
    const key = target?.dataset?.mem950Node || target?.dataset?.aicNode || target?.dataset?.aivNode || '';
    if (!key) return '';
    if (target.dataset.mem950Node) return `[data-mem950-node="${attrValue(key)}"]`;
    const core = target.closest?.('.pto-mem950__core-slot');
    const corePrefix = core?.id ? `#${attrValue(core.id)} ` : '';
    if (target.dataset.aicNode) return `${corePrefix}[data-aic-node="${attrValue(key)}"]`;
    if (target.dataset.aivNode) return `${corePrefix}[data-aiv-node="${attrValue(key)}"]`;
    return '';
  }

  function routeIdsForCore(coreId) {
    if (coreId === 'mem950-aiv2') return ['gm-to-aiv2-ub', 'aiv2-ub-to-gm'];
    return ['gm-to-aiv1-ub', 'aiv1-ub-to-gm'];
  }

  function vectorFocusForCore(coreId) {
    const prefix = coreId === 'mem950-aiv2' ? '#mem950-aiv2' : '#mem950-aiv1';
    const [inRoute, outRoute] = routeIdsForCore(coreId);
    return {
      selectors: [
        '[data-mem950-node="rail:GM"]',
        `${prefix} [data-aiv-node="cache:ND-DMA Cache"]`,
        `${prefix} [data-aiv-node="buffer:UB"]`,
        `${prefix} [data-aiv-node="exec:SIMD"]`,
        `${prefix} [data-aiv-node="vector:Vector"]`,
      ],
      routes: [inRoute, outRoute],
    };
  }

  function simtFocusForCore(coreId) {
    const prefix = coreId === 'mem950-aiv2' ? '#mem950-aiv2' : '#mem950-aiv1';
    const routePrefix = coreId === 'mem950-aiv2' ? 'aiv2' : 'aiv1';
    return {
      selectors: [
        '[data-mem950-node="rail:GM"]',
        `${prefix} [data-aiv-node="cache:ND-DMA Cache"]`,
        `${prefix} [data-aiv-node="buffer:UB"]`,
        `${prefix} [data-aiv-node="exec:SIMT"]`,
      ],
      routes: [`gm-to-${routePrefix}-ub`, `gm-to-${routePrefix}-simt`],
    };
  }

  function cubeFocus() {
    return {
      selectors: [
        '[data-mem950-node="rail:GM"]',
        '#mem950-aic [data-aic-node="buffer:L0A"]',
        '#mem950-aic [data-aic-node="buffer:L0B"]',
        '#mem950-aic [data-aic-node="cube:CUBE"]',
        '#mem950-aic [data-aic-node="buffer:L0C"]',
      ],
      routes: ['gm-to-aic-l0a', 'gm-to-aic-l0b'],
    };
  }

  function directCvFocus() {
    return {
      selectors: [
        '#mem950-aic [data-aic-node="buffer:L0C"]',
        '#mem950-aiv1 [data-aiv-node="buffer:UB"]',
        '#mem950-aiv2 [data-aiv-node="buffer:UB"]',
        '#mem950-aic [data-aic-node="buffer:L1"]',
      ],
      routes: ['aic-to-aiv1', 'aiv2-to-aic'],
    };
  }

  function coreFocusForTarget(target, preset, exactSelector) {
    const coreSlot = target?.closest?.('.pto-mem950__core-slot') || target;
    const coreId = coreSlot?.id || '';
    const nodeSelector = coreSlot?.classList?.contains('is-aiv') ? '[data-aiv-node]' : '[data-aic-node]';
    const routeIds = (preset?.routes || [])
      .filter((route) => !route.defaultHidden && (
        String(route.from || '').includes(`#${coreId}`) ||
        String(route.to || '').includes(`#${coreId}`)
      ))
      .map((route) => route.id);

    return {
      selectors: [
        coreId ? `#${attrValue(coreId)} ${nodeSelector}` : '',
      ].filter(Boolean),
      routes: routeIds,
    };
  }

  function scalarFocusForCore(coreId, exactSelector) {
    if (coreId.includes('aiv')) {
      const prefix = coreId === 'mem950-aiv2' ? '#mem950-aiv2' : '#mem950-aiv1';
      return {
        selectors: [
          exactSelector,
          `${prefix} [data-aiv-node="exec:SIMT"]`,
          `${prefix} [data-aiv-node="exec:SIMD"]`,
        ],
        routes: [],
      };
    }

    if (coreId === 'mem950-aic') {
      return {
        selectors: [
          exactSelector,
          '#mem950-aic [data-aic-node="cache:DCache"]',
          '#mem950-aic [data-aic-node="cache:ICache"]',
          '#mem950-aic [data-aic-node="scheduler:Dispatch"]',
        ],
        routes: [],
      };
    }

    return exactSelector
      ? { selectors: [exactSelector], routes: [] }
      : { selectors: [], routes: [] };
  }

  function pathFocusForTarget(target, preset) {
    const key = target?.dataset?.mem950Node || target?.dataset?.aicNode || target?.dataset?.aivNode || '';
    const coreId = target?.closest?.('.pto-mem950__core-slot')?.id || '';
    const exactSelector = selectorForHardwareTarget(target);

    if (key.startsWith('core:')) {
      return coreFocusForTarget(target, preset, exactSelector);
    }

    if (key === 'rail:L2') {
      return {
        selectors: ['[data-mem950-node="rail:L2"]'],
        routes: (preset?.routes || []).filter((route) => String(route.id).startsWith('l2-to-')).map((route) => route.id),
      };
    }

    if (key === 'rail:GM') {
      return {
        selectors: ['[data-mem950-node="rail:GM"]'],
        routes: (preset?.routes || [])
          .filter((route) => String(route.id).startsWith('gm-to-') || String(route.id).endsWith('-to-gm'))
          .map((route) => route.id),
      };
    }

    if (key === 'buffer:UB') {
      const focus = vectorFocusForCore(coreId);
      if (coreId === 'mem950-aiv1') focus.routes.push('aic-to-aiv1');
      if (coreId === 'mem950-aiv2') focus.routes.push('aiv2-to-aic');
      return focus;
    }

    if (key === 'exec:SIMT') {
      return simtFocusForCore(coreId);
    }

    if (key === 'vector:Vector' || key === 'exec:SIMD') {
      return vectorFocusForCore(coreId);
    }

    if (key === 'scalar:Scalar') {
      return scalarFocusForCore(coreId, exactSelector);
    }

    if (key === 'cache:DCache' && coreId.includes('aiv')) {
      const routePrefix = coreId === 'mem950-aiv2' ? 'l2-to-aiv2' : 'l2-to-aiv1';
      return {
        selectors: ['[data-mem950-node="rail:L2"]', exactSelector],
        routes: [`${routePrefix}-dcache`],
      };
    }

    if (key === 'cache:ND-DMA Cache' && coreId.includes('aiv')) {
      const routePrefix = coreId === 'mem950-aiv2' ? 'l2-to-aiv2' : 'l2-to-aiv1';
      return {
        selectors: ['[data-mem950-node="rail:L2"]', exactSelector, `${coreId ? `#${attrValue(coreId)} ` : ''}[data-aiv-node="buffer:UB"]`],
        routes: [routePrefix],
      };
    }

    if (key === 'buffer:L1' || key === 'buffer:L0A' || key === 'buffer:L0B' || key === 'cube:CUBE') {
      return cubeFocus();
    }

    if (key === 'buffer:L0C') {
      const focus = directCvFocus();
      focus.selectors.push('#mem950-aic [data-aic-node="cube:CUBE"]');
      return focus;
    }

    if (key === 'cache:DCache' && coreId === 'mem950-aic') {
      return {
        selectors: ['[data-mem950-node="rail:L2"]', exactSelector],
        routes: ['l2-to-aic-dcache'],
      };
    }

    return exactSelector
      ? { selectors: [exactSelector], routes: [] }
      : { selectors: [], routes: [] };
  }

  function attachPathFocusInteractions(container, presetOrKey, options = {}) {
    const preset = resolvePreset(presetOrKey);
    const root = rootFor(container);
    if (!root || !preset) return null;

    const selector = options.selector || '[data-mem950-node], [data-aic-node], [data-aiv-node]';
    let activeTarget = null;

    const targetFromEvent = (event) => {
      const target = event.target?.closest?.(selector);
      return target && root.contains(target) ? target : null;
    };

    const show = (target) => {
      if (!target || target === activeTarget) return;
      activeTarget?.classList.remove('is-path-focus-source');
      activeTarget = target;
      activeTarget.classList.add('is-path-focus-source');
      const focus = options.getFocus?.(target, preset) || pathFocusForTarget(target, preset);
      setPathFocus(root, preset, focus);
    };

    const hide = () => {
      activeTarget?.classList.remove('is-path-focus-source');
      activeTarget = null;
      clearPathFocus(root);
    };

    const onPointerOver = (event) => show(targetFromEvent(event));
    const onPointerOut = (event) => {
      if (!activeTarget) return;
      if (event.relatedTarget && activeTarget.contains(event.relatedTarget)) return;
      const nextTarget = event.relatedTarget?.closest?.(selector);
      if (nextTarget && root.contains(nextTarget)) return;
      hide();
    };
    const onFocusIn = (event) => show(targetFromEvent(event));
    const onFocusOut = () => hide();

    root.addEventListener('pointerover', onPointerOver);
    root.addEventListener('pointerout', onPointerOut);
    root.addEventListener('focusin', onFocusIn);
    root.addEventListener('focusout', onFocusOut);

    return {
      destroy() {
        root.removeEventListener('pointerover', onPointerOver);
        root.removeEventListener('pointerout', onPointerOut);
        root.removeEventListener('focusin', onFocusIn);
        root.removeEventListener('focusout', onFocusOut);
        hide();
      },
    };
  }

  function activationDetailForTarget(target, preset) {
    const coreSlot = target?.closest?.('.pto-mem950__core-slot');
    const key = target?.dataset?.mem950Node || target?.dataset?.aicNode || target?.dataset?.aivNode || '';
    return {
      node: key,
      preset: preset?.id || '',
      coreId: coreSlot?.id || '',
      coreKind: coreSlot?.classList?.contains('is-aiv') ? 'aiv' : (coreSlot?.classList?.contains('is-aic') ? 'aic' : ''),
      coreTitle: coreSlot?.dataset?.coreTitle || coreSlot?.querySelector?.('.pto-mem950__core-title')?.textContent?.trim?.() || '',
      buffer: key.startsWith('buffer:') ? key.slice('buffer:'.length) : '',
    };
  }

  function attachNodeActivation(container, presetOrKey, options = {}) {
    const preset = resolvePreset(presetOrKey);
    const root = rootFor(container);
    if (!root || !preset) return null;

    const selector = options.selector || '[data-mem950-node], [data-aic-node], [data-aiv-node]';
    const onActivate = typeof options.onActivate === 'function' ? options.onActivate : null;
    const targets = Array.from(root.querySelectorAll(selector));
    const previous = new Map();

    targets.forEach((target) => {
      previous.set(target, {
        tabindex: target.getAttribute('tabindex'),
        role: target.getAttribute('role'),
        ariaLabel: target.getAttribute('aria-label'),
        title: target.getAttribute('title'),
      });
      target.classList.add('is-node-activatable');
      if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '0');
      if (!target.hasAttribute('role')) target.setAttribute('role', 'button');
      const detail = activationDetailForTarget(target, preset);
      const label = options.label?.(target, detail) || `Open ${detail.buffer || detail.node} details`;
      if (!target.hasAttribute('aria-label')) target.setAttribute('aria-label', label);
      if (!target.hasAttribute('title')) target.setAttribute('title', label);
    });

    const targetFromEvent = (event) => {
      const target = event.target?.closest?.(selector);
      return target && root.contains(target) ? target : null;
    };

    const activate = (event) => {
      const target = targetFromEvent(event);
      if (!target) return;
      onActivate?.(target, activationDetailForTarget(target, preset), event);
    };

    const onClick = (event) => activate(event);
    const onKeyDown = (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const target = targetFromEvent(event);
      if (!target) return;
      event.preventDefault();
      activate(event);
    };

    root.addEventListener('click', onClick);
    root.addEventListener('keydown', onKeyDown);

    return {
      targets,
      destroy() {
        root.removeEventListener('click', onClick);
        root.removeEventListener('keydown', onKeyDown);
        previous.forEach((attrs, target) => {
          target.classList.remove('is-node-activatable');
          if (attrs.tabindex == null) target.removeAttribute('tabindex');
          else target.setAttribute('tabindex', attrs.tabindex);
          if (attrs.role == null) target.removeAttribute('role');
          else target.setAttribute('role', attrs.role);
          if (attrs.ariaLabel == null) target.removeAttribute('aria-label');
          else target.setAttribute('aria-label', attrs.ariaLabel);
          if (attrs.title == null) target.removeAttribute('title');
          else target.setAttribute('title', attrs.title);
        });
      },
    };
  }

  function svgNode(tagName, attrs) {
    const el = document.createElementNS(SVG_NS, tagName);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      el.setAttribute(key, value);
    });
    return el;
  }

  function cloneCorePreset(coreConfig) {
    const helper = coreConfig.kind === 'aiv'
      ? global.PtoAivCorePattern
      : global.PtoAicCorePattern;
    const basePreset = helper?.resolvePreset?.(coreConfig.presetKey);
    if (!basePreset) return null;

    return {
      ...basePreset,
      id: `${basePreset.id}-${coreConfig.id}`,
      title: coreConfig.title,
    };
  }

  function buildRail(railConfig) {
    const rail = node('div', `pto-mem950__rail is-${railConfig.tone || 'memory-shell'}`);
    rail.dataset.mem950Node = `rail:${railConfig.key}`;
    rail.style.width = `${railContentWidth(railConfig.grid) + 36}px`;
    rail.appendChild(buildRailGrid(railConfig.grid));
    rail.appendChild(node('span', 'pto-mem950__rail-label', railConfig.label));
    return rail;
  }

  function railContentWidth(gridConfig) {
    const cols = Math.max(1, Number(gridConfig?.cols || 1));
    const cellSize = Math.max(4, Number(gridConfig?.cellSize || 12));
    const gap = Math.max(0, Number(gridConfig?.gap || 4));
    return cols * cellSize + Math.max(0, cols - 1) * gap;
  }

  function buildRailGrid(gridConfig) {
    const grid = node('div', 'pto-mem950__rail-grid');
    const rows = Math.max(1, Number(gridConfig?.rows || 1));
    const cols = Math.max(1, Number(gridConfig?.cols || 1));
    const cellSize = Math.max(4, Number(gridConfig?.cellSize || 12));
    const gap = Math.max(0, Number(gridConfig?.gap || 4));
    const shape = gridConfig?.shape || 'dot';

    grid.style.setProperty('--pto-mem950-rail-cols', String(cols));
    grid.style.setProperty('--pto-mem950-rail-cell-size', `${cellSize}px`);
    grid.style.setProperty('--pto-mem950-rail-gap', `${gap}px`);

    for (let index = 0; index < rows * cols; index += 1) {
      const cell = node('span', `pto-mem950__rail-cell is-${shape}`);
      grid.appendChild(cell);
    }

    return grid;
  }

  function buildCoreSlot(coreConfig) {
    const slot = node('section', `pto-mem950__core-slot is-${coreConfig.kind}`);
    slot.id = coreConfig.id;
    slot.dataset.mem950Node = `core:${coreConfig.title.replace(/\s+/g, '')}`;
    slot.dataset.coreTitle = coreConfig.title || '';
    const mount = node('div', 'pto-mem950__core-mount');
    slot.appendChild(mount);
    return { slot, mount };
  }

  function buildEnginePanel(engineConfig) {
    const panel = node('section', `pto-mem950__engine${engineConfig.region ? ` is-${engineConfig.region}` : ''}`);
    panel.dataset.mem950Node = `engine:${engineConfig.key || engineConfig.title || 'engine'}`;

    const head = node('div', 'pto-mem950__engine-head');
    if (engineConfig.kicker) {
      head.appendChild(node('div', 'pto-mem950__engine-kicker', engineConfig.kicker));
    }
    head.appendChild(node('div', 'pto-mem950__engine-title', engineConfig.title || engineConfig.key || 'Engine'));
    if (engineConfig.subtitle) {
      head.appendChild(node('div', 'pto-mem950__engine-subtitle', engineConfig.subtitle));
    }
    panel.appendChild(head);

    if (engineConfig.description) {
      panel.appendChild(node('p', 'pto-mem950__engine-description', engineConfig.description));
    }

    return panel;
  }

  function buildEngineStack(engines) {
    const list = Array.isArray(engines) ? engines : [];
    if (list.length === 0) return null;
    const stack = node('aside', 'pto-mem950__engine-stack');
    list.forEach((engineConfig) => stack.appendChild(buildEnginePanel(engineConfig)));
    return stack;
  }

  function renderCoreIntoSlot(slotMount, coreConfig) {
    const helper = coreConfig.kind === 'aiv'
      ? global.PtoAivCorePattern
      : global.PtoAicCorePattern;
    const preset = cloneCorePreset(coreConfig);
    if (!helper || !preset) {
      slotMount.appendChild(node('div', 'pto-mem950__missing', `${coreConfig.title} renderer unavailable`));
      return null;
    }
    return helper.render(slotMount, preset);
  }

  function appendDetailRows(target, rows) {
    if (!target || !Array.isArray(rows) || rows.length === 0) return;
    const list = node('div', 'detail-spec-list');
    rows.forEach(([label, value]) => {
      const row = node('div', 'detail-spec-row');
      row.appendChild(node('span', 'detail-spec-label', label));
      row.appendChild(node('span', 'detail-spec-value', value));
      list.appendChild(row);
    });
    target.appendChild(list);
  }

  function appendBankMiniGrid(target, bankGrid) {
    if (!target || !bankGrid) return;
    const groups = Math.max(1, Number(bankGrid.groups || 1));
    const banksPerGroup = Math.max(1, Number(bankGrid.banksPerGroup || 1));
    const grid = node('div', 'bank-mini-grid');
    grid.style.setProperty('--bank-mini-grid-groups', String(groups));
    for (let groupIndex = 0; groupIndex < groups; groupIndex += 1) {
      const group = node('span', 'bank-group');
      group.style.setProperty('--bank-mini-grid-bank-count', String(banksPerGroup));
      for (let bankIndex = 0; bankIndex < banksPerGroup; bankIndex += 1) {
        group.appendChild(node('span'));
      }
      grid.appendChild(group);
    }
    target.appendChild(grid);
  }

  function appendInstructionItems(target, items) {
    if (!target || !Array.isArray(items) || items.length === 0) return;
    const list = node('div', 'instruction-sequence-list');
    items.forEach((item) => list.appendChild(node('span', 'instruction-sequence-chip', item)));
    target.appendChild(list);
  }

  function applyPresetDetails(stage, preset) {
    if (!stage || !preset) return;
    (preset.details || []).forEach((detail) => {
      stage.querySelectorAll(detail.selector).forEach((target) => {
        appendDetailRows(target, detail.rows);
        appendBankMiniGrid(target, detail.bankGrid);
        appendInstructionItems(target, detail.instructionItems);
      });
    });
  }

  function renderArchitecture(container, presetOrKey) {
    const preset = resolvePreset(presetOrKey);
    if (!container || !preset) return null;

    container.innerHTML = '';
    container.dataset.ptoMemArch = 'true';
    container.dataset.ptoMemArchPreset = preset.id;

    const stage = node('section', 'pto-mem950');
    stage.dataset.ptoMemArchStage = preset.id;

    const layout = node('div', 'pto-mem950__layout');
    const rails = node('div', 'pto-mem950__rails');
    const stack = node('div', 'pto-mem950__stack');

    (preset.rails || []).forEach((railConfig) => rails.appendChild(buildRail(railConfig)));

    (preset.cores || []).forEach((coreConfig) => {
      const { slot, mount } = buildCoreSlot(coreConfig);
      stack.appendChild(slot);
      renderCoreIntoSlot(mount, coreConfig);
    });

    layout.appendChild(rails);
    const engines = buildEngineStack(preset.engines);
    if (engines) layout.appendChild(engines);
    layout.appendChild(stack);
    stage.appendChild(layout);
    applyPresetDetails(stage, preset);

    if ((preset.notes || []).length > 0) {
      const notes = node('div', 'pto-mem950__notes');
      preset.notes.forEach((item) => notes.appendChild(node('span', 'pto-mem950__note', item)));
      stage.appendChild(notes);
    }

    container.appendChild(stage);

    return {
      container,
      preset,
      stage,
    };
  }

  function elementLayoutSize(element, axis, fallback) {
    const offsetKey = axis === 'x' ? 'offsetWidth' : 'offsetHeight';
    const scrollKey = axis === 'x' ? 'scrollWidth' : 'scrollHeight';
    const clientKey = axis === 'x' ? 'clientWidth' : 'clientHeight';
    return Math.max(
      1,
      Number(element?.[offsetKey]) || 0,
      Number(element?.[scrollKey]) || 0,
      Number(element?.[clientKey]) || 0,
      Number(fallback) || 0,
    );
  }

  function overlayMetrics(root) {
    const rootRect = root.getBoundingClientRect();
    const width = elementLayoutSize(root, 'x', rootRect.width);
    const height = elementLayoutSize(root, 'y', rootRect.height);
    const scaleX = rootRect.width > 0 ? rootRect.width / width : 1;
    const scaleY = rootRect.height > 0 ? rootRect.height / height : 1;
    return {
      rootRect,
      width,
      height,
      scaleX: Number.isFinite(scaleX) && scaleX > 0 ? scaleX : 1,
      scaleY: Number.isFinite(scaleY) && scaleY > 0 ? scaleY : 1,
    };
  }

  function layoutRectInRootSpace(root, element) {
    if (!(root instanceof HTMLElement) || !(element instanceof HTMLElement)) return null;

    let left = 0;
    let top = 0;
    let current = element;
    while (current && current !== root) {
      left += current.offsetLeft || 0;
      top += current.offsetTop || 0;
      current = current.offsetParent;
      if (current && current !== root && !root.contains(current)) return null;
    }

    if (current !== root) return null;
    const width = elementLayoutSize(element, 'x', element.getBoundingClientRect().width);
    const height = elementLayoutSize(element, 'y', element.getBoundingClientRect().height);
    return {
      left,
      top,
      width,
      height,
      right: left + width,
      bottom: top + height,
    };
  }

  function rectInRootSpace(root, element, metrics = overlayMetrics(root)) {
    const layoutRect = layoutRectInRootSpace(root, element);
    if (layoutRect) return layoutRect;

    const rect = element.getBoundingClientRect();
    const left = (rect.left - metrics.rootRect.left) / metrics.scaleX;
    const top = (rect.top - metrics.rootRect.top) / metrics.scaleY;
    const width = rect.width / metrics.scaleX;
    const height = rect.height / metrics.scaleY;
    return {
      left,
      top,
      width,
      height,
      right: left + width,
      bottom: top + height,
    };
  }

  function edgePoint(root, nodeEl, side, bias, metrics = overlayMetrics(root)) {
    const rect = rectInRootSpace(root, nodeEl, metrics);
    const biasRatio = Math.max(0, Math.min(1, Number.isFinite(bias) ? bias : 0.5));
    const xAtBias = rect.left + rect.width * biasRatio;
    const yAtBias = rect.top + rect.height * biasRatio;

    if (side === 'left') return { x: rect.left, y: yAtBias };
    if (side === 'right') return { x: rect.right, y: yAtBias };
    if (side === 'top') return { x: xAtBias, y: rect.top };
    if (side === 'bottom') return { x: xAtBias, y: rect.bottom };
    return {
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2,
    };
  }

  function shiftPoint(point, dx, dy) {
    return {
      x: point.x + (Number.isFinite(dx) ? dx : 0),
      y: point.y + (Number.isFinite(dy) ? dy : 0),
    };
  }

  function endpointElement(baseEl, anchorSelector) {
    if (!baseEl || !anchorSelector) return baseEl;
    return baseEl.querySelector?.(anchorSelector) || baseEl;
  }

  function pointsToPath(points) {
    return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');
  }

  function resolveLaneX(root, route, fromPoint, toPoint, metrics = overlayMetrics(root)) {
    if (Number.isFinite(route.corridorRight)) {
      const stackEl = root.querySelector('.pto-mem950__stack');
      if (stackEl) {
        const stackRect = rectInRootSpace(root, stackEl, metrics);
        return Math.max(0, stackRect.right - route.corridorRight);
      }
      return Math.max(0, metrics.width - route.corridorRight);
    }
    if (Number.isFinite(route.corridorLeft)) return route.corridorLeft;
    return fromPoint.x + (toPoint.x - fromPoint.x) / 2;
  }

  function resolveLaneY(root, route, fromPoint, toPoint, metrics = overlayMetrics(root)) {
    const height = metrics.height;
    if (Number.isFinite(route.corridorTop)) return route.corridorTop;
    if (Number.isFinite(route.corridorBottom)) return Math.max(0, height - route.corridorBottom);
    return fromPoint.y + (toPoint.y - fromPoint.y) / 2;
  }

  function resolveSourceLaneY(root, route, fromPoint, metrics = overlayMetrics(root)) {
    let laneY = fromPoint.y;
    if (route.sourceLaneBelowSelector) {
      const laneAnchor = root.querySelector(route.sourceLaneBelowSelector);
      if (laneAnchor) {
        const anchorRect = rectInRootSpace(root, laneAnchor, metrics);
        const offset = Number.isFinite(route.sourceLaneOffset) ? route.sourceLaneOffset : 0;
        laneY = anchorRect.bottom + offset;
      }
    }
    if (route.sourceLaneAboveSelector) {
      const laneAnchor = root.querySelector(route.sourceLaneAboveSelector);
      if (laneAnchor) {
        const anchorRect = rectInRootSpace(root, laneAnchor, metrics);
        const offset = Number.isFinite(route.sourceLaneAboveOffset) ? route.sourceLaneAboveOffset : 0;
        laneY = Math.min(laneY, anchorRect.top - offset);
      }
    }
    return Math.max(0, Math.min(metrics.height, laneY));
  }

  function routeGeometry(root, route, fromPoint, toPoint, metrics = overlayMetrics(root)) {
    if (route.style === 'lane-h-target') {
      const start = { x: fromPoint.x, y: toPoint.y };
      const end = { x: toPoint.x, y: toPoint.y };
      return {
        points: [start, end],
        labelPoint: {
          x: (start.x + end.x) / 2 + (route.labelDx || 0),
          y: start.y + (route.labelDy || 0),
        },
      };
    }

    if (route.style === 'lane-h-source') {
      const laneY = resolveSourceLaneY(root, route, fromPoint, metrics);
      const start = { x: fromPoint.x, y: fromPoint.y };
      const end = { x: toPoint.x, y: laneY };
      const points = [start];
      if (Math.abs(laneY - fromPoint.y) > 0.5) {
        points.push({ x: fromPoint.x, y: laneY });
      }
      points.push(end);
      return {
        points,
        labelPoint: {
          x: (fromPoint.x + toPoint.x) / 2 + (route.labelDx || 0),
          y: laneY + (route.labelDy || 0),
        },
      };
    }

    if (route.style === 'elbow-v') {
      const laneY = resolveLaneY(root, route, fromPoint, toPoint, metrics);
      const points = [
        fromPoint,
        { x: fromPoint.x, y: laneY },
        { x: toPoint.x, y: laneY },
        toPoint,
      ];
      return {
        points,
        labelPoint: {
          x: (points[1].x + points[2].x) / 2 + (route.labelDx || 0),
          y: points[1].y + (route.labelDy || 0),
        },
      };
    }

    const laneX = resolveLaneX(root, route, fromPoint, toPoint, metrics);
    const sourceLaneY = resolveSourceLaneY(root, route, fromPoint, metrics);
    const points = [
      fromPoint,
    ];
    if (Math.abs(sourceLaneY - fromPoint.y) > 0.5) {
      points.push({ x: fromPoint.x, y: sourceLaneY });
    }
    points.push(
      { x: laneX, y: sourceLaneY },
      { x: laneX, y: toPoint.y },
      toPoint,
    );
    return {
      points,
      labelPoint: {
        x: (fromPoint.x + laneX) / 2 + (route.labelDx || 0),
        y: sourceLaneY + (route.labelDy || 0),
      },
    };
  }

  function createRouteOverlay(container, presetOrKey) {
    const preset = resolvePreset(presetOrKey);
    const root = container?.querySelector?.('.pto-mem950') || container;
    if (!root || !preset) return null;

    const svg = svgNode('svg', {
      class: 'pto-mem950__overlay',
      viewBox: '0 0 10 10',
      preserveAspectRatio: 'none',
    });
    const defs = svgNode('defs');
    svg.appendChild(defs);

    Object.entries(ROUTE_TONES).forEach(([key, tone]) => {
      const marker = svgNode('marker', {
        id: `pto-mem950-arrow-${key}`,
        markerUnits: 'userSpaceOnUse',
        markerWidth: '5.5',
        markerHeight: '5.5',
        refX: '5',
        refY: '2.75',
        orient: 'auto',
      });
      marker.appendChild(svgNode('path', {
        d: 'M1,1 L5,2.75 L1,4.5',
        fill: 'none',
        stroke: tone.line,
        'stroke-width': '1.1',
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
      }));
      defs.appendChild(marker);
    });

    const routeEls = (preset.routes || []).map((route) => {
      const groupAttrs = { 'data-route-id': route.id };
      if (route.defaultHidden) groupAttrs['data-route-default-hidden'] = 'true';
      if (route.group) groupAttrs['data-route-group'] = route.group;
      const group = svgNode('g', groupAttrs);
      const path = svgNode('path', {
        fill: 'none',
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        'stroke-width': route.strokeWidth || '1.5',
      });
      const labelGroup = svgNode('g');
      const labelBg = svgNode('rect', { rx: '11', ry: '11' });
      const labelText = svgNode('text', {
        'font-size': route.fontSize || '10',
        'font-weight': '700',
        'font-family': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
        'text-anchor': 'middle',
        'dominant-baseline': 'middle',
      });
      labelText.textContent = route.label || '';
      labelGroup.appendChild(labelBg);
      labelGroup.appendChild(labelText);
      group.appendChild(path);
      group.appendChild(labelGroup);
      svg.appendChild(group);
      return { route, path, labelGroup, labelBg, labelText };
    });

    root.appendChild(svg);

    let animationFrame = 0;
    let destroyed = false;
    const delayedUpdates = new Map();

    function update() {
      if (destroyed) return;
      const metrics = overlayMetrics(root);
      svg.setAttribute('viewBox', `0 0 ${metrics.width} ${metrics.height}`);
      svg.style.width = `${metrics.width}px`;
      svg.style.height = `${metrics.height}px`;

      routeEls.forEach((entry) => {
        const fromBaseEl = root.querySelector(entry.route.from);
        const toBaseEl = root.querySelector(entry.route.to);
        if (!fromBaseEl || !toBaseEl) {
          entry.path.style.display = 'none';
          entry.labelGroup.style.display = 'none';
          return;
        }

        const fromEl = endpointElement(fromBaseEl, entry.route.fromAnchorSelector);
        const toEl = endpointElement(toBaseEl, entry.route.toAnchorSelector);
        const fromPoint = shiftPoint(
          edgePoint(root, fromEl, entry.route.fromSide || 'right', entry.route.fromBias, metrics),
          entry.route.fromDx,
          entry.route.fromDy,
        );
        const toPoint = shiftPoint(
          edgePoint(root, toEl, entry.route.toSide || 'left', entry.route.toBias, metrics),
          entry.route.toDx,
          entry.route.toDy,
        );
        const geometry = routeGeometry(root, entry.route, fromPoint, toPoint, metrics);
        const tone = ROUTE_TONES[entry.route.tone] || ROUTE_TONES.transport;

        entry.path.style.display = '';
        entry.path.setAttribute('d', pointsToPath(geometry.points));
        entry.path.setAttribute('stroke', tone.line);
        entry.path.setAttribute('marker-end', `url(#pto-mem950-arrow-${entry.route.tone || 'transport'})`);
        if (entry.route.dashArray) {
          entry.path.setAttribute('stroke-dasharray', entry.route.dashArray);
        } else {
          entry.path.removeAttribute('stroke-dasharray');
        }

        if (!entry.route.label) {
          entry.labelGroup.style.display = 'none';
          return;
        }

        entry.labelGroup.style.display = '';
        entry.labelText.setAttribute('x', String(geometry.labelPoint.x));
        entry.labelText.setAttribute('y', String(geometry.labelPoint.y));
        entry.labelText.setAttribute('fill', tone.text);
        const textBox = entry.labelText.getBBox();
        const labelWidth = Math.max(64, textBox.width + 16);
        const labelHeight = 22;
        entry.labelBg.setAttribute('x', String(geometry.labelPoint.x - labelWidth / 2));
        entry.labelBg.setAttribute('y', String(geometry.labelPoint.y - labelHeight / 2));
        entry.labelBg.setAttribute('width', String(labelWidth));
        entry.labelBg.setAttribute('height', String(labelHeight));
        entry.labelBg.setAttribute('fill', tone.fill);
        entry.labelBg.setAttribute('stroke', tone.stroke);
        entry.labelBg.setAttribute('stroke-width', '1');
      });
    }

    function scheduleUpdate() {
      if (destroyed) return;
      if (animationFrame) global.cancelAnimationFrame?.(animationFrame);
      animationFrame = global.requestAnimationFrame?.(() => {
        animationFrame = 0;
        update();
      }) || 0;
      if (!animationFrame) update();
    }

    function scheduleDelayedUpdate(delay) {
      if (destroyed) return;
      const previous = delayedUpdates.get(delay);
      if (previous) global.clearTimeout?.(previous);
      const timeout = global.setTimeout?.(() => {
        delayedUpdates.delete(delay);
        scheduleUpdate();
      }, delay);
      if (timeout) delayedUpdates.set(delay, timeout);
    }

    function scheduleSettledUpdate() {
      scheduleUpdate();
      [48, 120, 260].forEach(scheduleDelayedUpdate);
    }

    const resizeObserver = typeof ResizeObserver === 'function'
      ? new ResizeObserver(scheduleUpdate)
      : null;
    resizeObserver?.observe(root);
    root.querySelectorAll('[data-mem950-node], [data-aiv-node], [data-aic-node]').forEach((el) => resizeObserver?.observe(el));
    scheduleSettledUpdate();
    if (document.fonts && typeof document.fonts.ready?.then === 'function') {
      document.fonts.ready.then(scheduleSettledUpdate);
    }

    return {
      svg,
      update,
      schedule: scheduleSettledUpdate,
      render() {
        update();
        scheduleSettledUpdate();
      },
      destroy() {
        destroyed = true;
        if (animationFrame) global.cancelAnimationFrame?.(animationFrame);
        delayedUpdates.forEach((timeout) => global.clearTimeout?.(timeout));
        delayedUpdates.clear();
        resizeObserver?.disconnect();
        svg.remove();
      },
    };
  }

  function attachHoverInteractions(container, presetOrKey, options = {}) {
    const preset = resolvePreset(presetOrKey);
    const root = container?.querySelector?.('.pto-mem950') || container;
    if (!root || !preset) return null;

    const selector = options.selector || '[data-mem950-node], [data-aic-node], [data-aiv-node]';
    const tooltip = node('div', 'pto-mem950__hover-tip');
    tooltip.setAttribute('role', 'tooltip');
    tooltip.setAttribute('aria-hidden', 'true');
    root.appendChild(tooltip);

    let activeTarget = null;
    let viewportScale = 1;

    const setViewportScale = (scale = 1) => {
      const numericScale = Number(scale);
      viewportScale = Number.isFinite(numericScale) && numericScale > 0 ? numericScale : 1;
      tooltip.style.setProperty('--pto-mem950-hover-tip-scale', String(1 / viewportScale));
      if (activeTarget) positionTooltip(null, null, activeTarget);
    };

    const measureRootScale = () => {
      const rootRect = root.getBoundingClientRect();
      const scaleX = root.offsetWidth ? rootRect.width / root.offsetWidth : viewportScale;
      const scaleY = root.offsetHeight ? rootRect.height / root.offsetHeight : viewportScale;
      return {
        rootRect,
        scaleX: Number.isFinite(scaleX) && scaleX > 0 ? scaleX : viewportScale,
        scaleY: Number.isFinite(scaleY) && scaleY > 0 ? scaleY : viewportScale,
      };
    };

    const positionTooltip = (clientX, clientY, fallbackTarget = activeTarget) => {
      const { rootRect, scaleX, scaleY } = measureRootScale();
      const tipRect = tooltip.getBoundingClientRect();
      tooltip.style.setProperty('--pto-mem950-hover-tip-scale', String(1 / scaleX));
      const marginX = 8 / scaleX;
      const marginY = 8 / scaleY;
      const pointerGapX = 14 / scaleX;
      const pointerGapY = 14 / scaleY;
      let x = Number.isFinite(clientX) ? (clientX - rootRect.left) / scaleX + pointerGapX : 0;
      let y = Number.isFinite(clientY) ? (clientY - rootRect.top) / scaleY + pointerGapY : 0;

      if (!Number.isFinite(clientX) && fallbackTarget) {
        const targetRect = fallbackTarget.getBoundingClientRect();
        x = (targetRect.left - rootRect.left) / scaleX + targetRect.width / scaleX / 2 + (12 / scaleX);
        y = (targetRect.top - rootRect.top) / scaleY + Math.min(targetRect.height / scaleY, 28);
      }

      const maxX = Math.max(marginX, root.offsetWidth - tipRect.width / scaleX - marginX);
      const maxY = Math.max(marginY, root.offsetHeight - tipRect.height / scaleY - marginY);
      tooltip.style.left = `${Math.max(marginX, Math.min(maxX, x))}px`;
      tooltip.style.top = `${Math.max(marginY, Math.min(maxY, y))}px`;
    };

    const renderTooltip = (target) => {
      const tip = tipForTarget(target, preset, options);
      clearChildren(tooltip);
      tooltip.appendChild(node('div', 'pto-mem950__hover-tip-title', tip.title));
      if (tip.body) tooltip.appendChild(node('div', 'pto-mem950__hover-tip-body', tip.body));
    };

    const show = (target, event = null) => {
      if (!target) return;
      if (activeTarget && activeTarget !== target) activeTarget.classList.remove('is-hovered');
      activeTarget = target;
      activeTarget.classList.add('is-hovered');
      renderTooltip(target);
      tooltip.classList.add('is-visible');
      tooltip.setAttribute('aria-hidden', 'false');
      positionTooltip(event?.clientX, event?.clientY, target);
    };

    const hide = () => {
      activeTarget?.classList.remove('is-hovered');
      activeTarget = null;
      tooltip.classList.remove('is-visible');
      tooltip.setAttribute('aria-hidden', 'true');
    };

    const targetFromEvent = (event) => {
      const target = event.target?.closest?.(selector);
      return target && root.contains(target) ? target : null;
    };

    const onPointerOver = (event) => show(targetFromEvent(event), event);
    const onPointerMove = (event) => {
      if (activeTarget) positionTooltip(event.clientX, event.clientY, activeTarget);
    };
    const onPointerOut = (event) => {
      if (!activeTarget) return;
      if (event.relatedTarget && activeTarget.contains(event.relatedTarget)) return;
      const nextTarget = event.relatedTarget?.closest?.(selector);
      if (nextTarget && root.contains(nextTarget)) return;
      hide();
    };
    const onFocusIn = (event) => show(targetFromEvent(event));
    const onFocusOut = () => hide();

    root.addEventListener('pointerover', onPointerOver);
    root.addEventListener('pointermove', onPointerMove);
    root.addEventListener('pointerout', onPointerOut);
    root.addEventListener('focusin', onFocusIn);
    root.addEventListener('focusout', onFocusOut);
    setViewportScale(options.viewportScale || options.scale || 1);

    return {
      tooltip,
      setViewportScale,
      destroy() {
        root.removeEventListener('pointerover', onPointerOver);
        root.removeEventListener('pointermove', onPointerMove);
        root.removeEventListener('pointerout', onPointerOut);
        root.removeEventListener('focusin', onFocusIn);
        root.removeEventListener('focusout', onFocusOut);
        activeTarget?.classList.remove('is-hovered');
        tooltip.remove();
      },
    };
  }

  function clearBufferBlocks(container) {
    const root = rootFor(container);
    if (!root) return null;
    global.PtoAicCorePattern?.clearBufferBlocks?.(root);
    global.PtoAivCorePattern?.clearBufferBlocks?.(root);
    return { root };
  }

  function coreSlotForBlock(root, block) {
    if (!block?.core) return root;
    return root.querySelector(`[id="${attrValue(block.core)}"]`) || root;
  }

  function setBufferBlocks(container, blocks = []) {
    const root = rootFor(container);
    if (!root) return null;
    const list = Array.isArray(blocks) ? blocks : [];
    clearBufferBlocks(root);
    let applied = 0;
    const groups = new Map();
    list.forEach((block) => {
      if (!block) return;
      const slot = coreSlotForBlock(root, block);
      const helper = slot.classList?.contains('is-aiv') || slot.querySelector?.('.pto-aiv-core')
        ? global.PtoAivCorePattern
        : global.PtoAicCorePattern;
      if (!helper?.setBufferBlocks) return;
      const key = `${slot.id || 'root'}:${helper === global.PtoAivCorePattern ? 'aiv' : 'aic'}`;
      if (!groups.has(key)) groups.set(key, { slot, helper, blocks: [] });
      groups.get(key).blocks.push(block);
    });
    groups.forEach((group) => {
      const result = group.helper.setBufferBlocks(group.slot, group.blocks);
      applied += Number(result?.applied || 0);
    });
    return { root, blocks: list, applied };
  }

  function renderBufferGrid() {
    return null;
  }

  global.PtoMemoryArchitecturePattern = {
    presets: PRESETS,
    resolvePreset,
    renderArchitecture,
    createRouteOverlay,
    attachHoverInteractions,
    attachPathFocusInteractions,
    attachNodeActivation,
    setDetailVisibility,
    setAivFolded,
    setPathFocus,
    clearPathFocus,
    setBufferBlocks,
    clearBufferBlocks,
    createZoomController,
    renderBufferGrid,
  };
})(window);
