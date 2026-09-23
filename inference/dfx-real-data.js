(function (global) {
  'use strict';

  // Keep the evidence bundle inside the static site so local HTTP serving and
  // GitHub Pages resolve the same relative path.
  var ROOT = './data/deepseekv4-flash-decode/';
  var PALETTE = {
    aic: '#c8945d',
    aiv: '#5f9b9b',
    wait: '#b86d68',
    memory: '#8a72a6',
    dispatch: '#b86d68',
    expert: '#c8945d'
  };

  var MANIFEST = {
    csa: {
      id: 'csa', label: 'CSA', kind: 'attention', path: 'CSA',
      root: 'decode_attention_csa_startpos8192/dfx_outputs',
      trace: 'merged_swimlane_20260726_222710.json',
      nameMap: 'name_map__jit_attention_csa_test_20260726_222647.json',
      hints: 'decode_attention_csa_startpos8192/report/perf_hints.log',
      memory: 'decode_attention_csa_startpos8192/report/memory_after_AllocateMemoryAddr.txt'
    },
    hca: {
      id: 'hca', label: 'HCA', kind: 'attention', path: 'HCA',
      root: 'decode_attention_hca_startpos8192/dfx_outputs',
      trace: 'merged_swimlane_20260726_222738.json',
      nameMap: 'name_map__jit_attention_hca_test_20260726_222720.json',
      hints: 'decode_attention_hca_startpos8192/report/perf_hints.log',
      memory: 'decode_attention_hca_startpos8192/report/memory_after_AllocateMemoryAddr.txt'
    },
    swa: {
      id: 'swa', label: 'SWA', kind: 'attention', path: 'SWA',
      root: 'decode_attention_swa_startpos8192/dfx_outputs',
      trace: 'merged_swimlane_20260726_222804.json',
      nameMap: 'name_map__jit_attention_swa_test_20260726_222748.json',
      hints: 'decode_attention_swa_startpos8192/report/perf_hints.log',
      memory: 'decode_attention_swa_startpos8192/report/memory_after_AllocateMemoryAddr.txt'
    },
    ep2: {
      id: 'ep2', label: 'EP2 Balanced', kind: 'moe', path: 'EP2',
      scenario: 'moe_ep2_balanced', ranks: [0, 1], timestamp: '20260726_231408',
      hints: 'moe_ep2_balanced/report/perf_hints.log',
      memory: 'moe_ep2_balanced/report/memory_after_AllocateMemoryAddr.txt',
      distributedMeta: 'moe_ep2_balanced/distributed_meta.json'
    },
    ep8: {
      id: 'ep8', label: 'EP8 Balanced', kind: 'moe', path: 'EP8',
      scenario: 'moe_ep8_balanced', ranks: [0, 1, 2, 3, 4, 5, 6, 7], timestamp: '20260726_230402',
      rankTimestamps: { 7: '20260726_230403' },
      hints: 'moe_ep8_balanced/report/perf_hints.log',
      memory: 'moe_ep8_balanced/report/memory_after_AllocateMemoryAddr.txt',
      distributedMeta: 'moe_ep8_balanced/distributed_meta.json'
    }
  };

  function getJson(path) {
    return fetch(ROOT + path).then(function (response) {
      if (!response.ok) throw new Error('DFX artifact unavailable: ' + path);
      return response.json();
    });
  }

  function getText(path) {
    if (!path) return Promise.resolve('');
    return fetch(ROOT + path).then(function (response) {
      return response.ok ? response.text() : '';
    }).catch(function () { return ''; });
  }

  function getOptionalJson(path) {
    if (!path) return Promise.resolve(null);
    return getJson(path).catch(function () { return null; });
  }

  function round(value, digits) {
    var factor = Math.pow(10, digits || 2);
    return Math.round(Number(value || 0) * factor) / factor;
  }

  function stripTraceSuffix(name) {
    return String(name || '').replace(/\(r\d+t\d+\)$/, '').replace(/_spmd$/, '');
  }

  function parseHint(value, pattern) {
    var match = String(value || '').match(pattern);
    return match ? match[1] : '';
  }

  function colourFor(name, coreType) {
    var lower = String(name || '').toLowerCase();
    if (lower.indexOf('wait') >= 0 || lower.indexOf('gather') >= 0 || lower.indexOf('meta') >= 0 || lower.indexOf('push') >= 0) return PALETTE.wait;
    if (lower.indexOf('exp_') === 0 || lower.indexOf('expert') >= 0) return PALETTE.expert;
    return coreType === 'aic' ? PALETTE.aic : PALETTE.aiv;
  }

  function parseMemoryReport(text) {
    var metrics = [];
    var sections = [];
    var current = null;
    var bufferSpace = '';
    String(text || '').split(/\r?\n/).forEach(function (line) {
      var heading = line.match(/^---\s+(.+?)\s+---$/);
      if (heading) {
        current = { kernel: heading[1], spaces: [], buffers: {} };
        sections.push(current);
        bufferSpace = '';
        return;
      }
      var bufferHeading = line.match(/^\s*Buffers \(([^)]+)\)/);
      if (bufferHeading && current) {
        bufferSpace = bufferHeading[1];
        current.buffers[bufferSpace] = current.buffers[bufferSpace] || [];
        return;
      }
      var usage = line.match(/^\s*([A-Za-z0-9_]+)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([0-9.]+)\s*%\s*\|\s*(\d+)/);
      if (usage && current && usage[1] !== 'Space') {
        var metric = { kernel: current.kernel, region: usage[1], used: usage[2].trim(), capacity: usage[3].trim(), percent: Number(usage[4]), memRefs: Number(usage[5]) };
        current.spaces.push(metric);
        metrics.push(metric);
        return;
      }
      var buffer = line.match(/^\s*([A-Za-z0-9_]+)\s*\|\s*([^|]+)\|\s*\[(\d+),\s*(\d+)\)\s*\|\s*\[(\d+),\s*(\d+)\]/);
      if (buffer && current && bufferSpace) {
        current.buffers[bufferSpace].push({
          name: buffer[1], size: buffer[2].trim(), addressStart: Number(buffer[3]), addressEnd: Number(buffer[4]),
          liveStart: Number(buffer[5]), liveEnd: Number(buffer[6])
        });
      }
    });
    var byRegion = {};
    metrics.forEach(function (item) {
      if (!byRegion[item.region] || item.percent > byRegion[item.region].percent) byRegion[item.region] = item;
    });
    var top = Object.keys(byRegion).map(function (key) { return byRegion[key]; }).sort(function (a, b) { return b.percent - a.percent; });
    var hottest = metrics.slice().sort(function (a, b) { return b.percent - a.percent; });
    return {
      metrics: top,
      sections: sections,
      hottest: hottest,
      summary: hottest.length ? hottest[0].kernel + ' · ' + hottest[0].region + ' ' + hottest[0].percent + '%' : '未采集'
    };
  }

  function parseHints(text) {
    var hints = [];
    String(text || '').split(/\r?\n/).forEach(function (line) {
      var code = parseHint(line, /^\[perf_hint ([^\]]+)\]/);
      if (!code) return;
      var source = line.match(/\bat (\/[^:]+):(\d+):(\d+)$/);
      var summary = line.replace(/^\[perf_hint [^\]]+\]\s*/, '').replace(/\s+at \/.+:\d+:\d+$/, '');
      hints.push({
        code: code,
        summary: summary,
        source: source ? source[1] : '',
        line: source ? Number(source[2]) : null,
        column: source ? Number(source[3]) : null
      });
    });
    return hints;
  }

  function groupHints(hints) {
    var groups = {};
    hints.forEach(function (hint) {
      var diagnostic = hint.summary.split(':')[0] || hint.code;
      var key = hint.code + ':' + diagnostic;
      if (!groups[key]) groups[key] = { id: key, code: hint.code, diagnostic: diagnostic, count: 0, occurrences: [], sources: [] };
      groups[key].count += 1;
      groups[key].occurrences.push(hint);
      var location = hint.source + (hint.line ? ':' + hint.line : '');
      if (location && groups[key].sources.indexOf(location) < 0) groups[key].sources.push(location);
    });
    return Object.keys(groups).map(function (key) {
      var group = groups[key];
      group.summary = group.occurrences[0] ? group.occurrences[0].summary : '';
      return group;
    }).sort(function (a, b) { return b.count - a.count; });
  }

  function threadNameMap(trace) {
    var result = {};
    (trace.traceEvents || []).forEach(function (event) {
      if (event.name === 'thread_name' && event.args && event.args.name) result[event.tid] = event.args.name;
    });
    return result;
  }

  function parseCoreAndFunc(event) {
    var hint = event.args && event.args['event-hint'] || '';
    var core = hint.match(/CoreId:\s*(\d+)/);
    var func = hint.match(/FuncId:\s*(-?\d+)/);
    return { coreId: core ? Number(core[1]) : null, funcId: func ? Number(func[1]) : null };
  }

  function normalize(config, trace, deps, l2, nameMap, hintsText, memoryText, prefix) {
    var names = nameMap && nameMap.callable_id_to_name || {};
    var threads = threadNameMap(trace);
    var coreTypes = l2 && l2.metadata && l2.metadata.core_types || [];
    var events = (trace.traceEvents || []).filter(function (event) { return event.ph === 'X'; }).map(function (event, index) {
      var parsed = parseCoreAndFunc(event);
      var coreId = parsed.coreId == null ? -1 : parsed.coreId;
      var coreType = coreTypes[coreId] || (String(threads[event.tid] || '').indexOf('AIC_') === 0 ? 'aic' : 'aiv');
      var taskId = event.args && event.args.taskId != null ? String(event.args.taskId) : 'event-' + index;
      var funcName = names[String(parsed.funcId)] || stripTraceSuffix(event.name);
      var laneName = (prefix ? prefix + ' · ' : '') + (threads[event.tid] || (coreType.toUpperCase() + '_' + coreId));
      var id = config.id + ':' + (prefix || 'run') + ':' + taskId + ':' + coreId + ':' + index;
      return {
        id: id,
        taskId: taskId,
        taskKey: (prefix || '') + taskId,
        lane: laneName,
        laneId: (prefix || '') + 'core:' + coreId,
        laneKind: coreType,
        label: funcName + ' · C' + coreId,
        funcName: funcName,
        rawName: event.name,
        funcId: parsed.funcId,
        coreId: coreId,
        coreType: coreType,
        rank: config.rank == null ? null : config.rank,
        start: Number(event.ts || 0),
        dur: Number(event.dur || 0),
        kernelDuration: Number(event.args && event.args['kernel-duration-us'] || event.dur || 0),
        setupDuration: Number(event.args && event.args.local_setup_us || 0),
        fanin: event.args && event.args['fanin-hint'] || '',
        fanout: event.args && event.args['fanout-hint'] || '',
        hint: '',
        color: colourFor(funcName, coreType),
        kind: coreType === 'aic' ? 'kernel-aic' : 'kernel-aiv'
      };
    });

    var taskRecords = (deps.tasks || []).map(function (task) {
      var kernelId = (task.kernel_ids || []).find(function (id) { return Number(id) >= 0; });
      return Object.assign({}, task, {
        taskId: String(task.task_id),
        taskKey: (prefix || '') + String(task.task_id),
        rank: config.rank == null ? null : config.rank,
        funcId: kernelId == null ? null : Number(kernelId),
        funcName: names[String(kernelId)] || (kernelId == null ? 'runtime task' : 'func_' + kernelId)
      });
    });
    var tensorRecords = (deps.tensors || []).map(function (tensor) {
      return Object.assign({}, tensor, { tensorId: String(tensor.tensor_id), tensorKey: (prefix || '') + String(tensor.tensor_id), rank: config.rank == null ? null : config.rank });
    });
    var edges = (deps.edges || []).map(function (edge, index) {
      return Object.assign({}, edge, {
        edgeId: config.id + ':' + (prefix || 'run') + ':edge:' + index,
        pred: String(edge.pred),
        succ: String(edge.succ),
        tensorId: edge.tensor_id == null ? '' : String(edge.tensor_id),
        predKey: (prefix || '') + String(edge.pred),
        succKey: (prefix || '') + String(edge.succ),
        tensorKey: edge.tensor_id == null ? '' : (prefix || '') + String(edge.tensor_id)
        ,rank: config.rank == null ? null : config.rank
      });
    });
    var taskIndex = {};
    var tensorIndex = {};
    taskRecords.forEach(function (item) { taskIndex[item.taskKey] = item; });
    tensorRecords.forEach(function (item) { tensorIndex[item.tensorKey] = item; });
    var laneMap = {};
    events.forEach(function (event) {
      if (!laneMap[event.lane]) laneMap[event.lane] = { id: event.laneId, label: event.lane, kind: event.laneKind, events: [] };
      laneMap[event.lane].events.push(event);
    });
    var lanes = Object.keys(laneMap).map(function (key) { return laneMap[key]; }).sort(function (a, b) { return a.label.localeCompare(b.label, undefined, { numeric: true }); });
    var start = events.length ? Math.min.apply(null, events.map(function (item) { return item.start; })) : 0;
    var end = events.length ? Math.max.apply(null, events.map(function (item) { return item.start + item.dur; })) : 0;
    var longest = events.slice().sort(function (a, b) { return b.dur - a.dur; })[0] || null;
    var hints = parseHints(hintsText);
    var hintGroups = groupHints(hints);
    var memory = parseMemoryReport(memoryText);
    return {
      id: config.id,
      label: config.label,
      kind: config.kind,
      path: config.path,
      window: round(end - start, 2),
      taskCount: taskRecords.length,
      tensorCount: tensorRecords.length,
      edgeCount: edges.length,
      eventCount: events.length,
      coreCount: (l2 && l2.metadata && l2.metadata.num_cores) || lanes.length,
      aicoreCount: (l2 && l2.aicore_tasks || []).length,
      aicpuCount: (l2 && l2.aicpu_tasks || []).length,
      dominant: longest ? longest.funcName : '未采集',
      dominantEventId: longest ? longest.id : '',
      memory: memory.summary,
      memoryMetrics: memory.metrics,
      memorySections: memory.sections,
      memoryHotspots: memory.hottest,
      hint: hints.length ? hints[0].code : '未采集',
      hints: hints,
      hintGroups: hintGroups,
      events: events,
      lanes: lanes,
      tasks: taskRecords,
      tensors: tensorRecords,
      edges: edges,
      taskIndex: taskIndex,
      tensorIndex: tensorIndex,
      tracePath: config.root + '/' + config.trace,
      depsPath: config.root + '/deps.json',
      l2Path: config.root + '/l2_swimlane_records.json',
      nameMapPath: config.root + '/' + (config.nameMap || 'name_map.json'),
      sourceEvidence: {
        perfHints: config.hints || '',
        memoryReport: config.memory || ''
      },
      artifactChain: [
        config.root + '/' + config.trace,
        config.root + '/deps.json',
        config.root + '/l2_swimlane_records.json',
        config.root + '/' + (config.nameMap || 'name_map.json'),
        config.memory || '',
        config.hints || ''
      ].filter(Boolean),
      rawTraceEventCount: (trace.traceEvents || []).length,
      metadata: l2 && l2.metadata || {}
    };
  }

  function loadFlat(config, prefix) {
    var root = config.root;
    return Promise.all([
      getJson(root + '/' + config.trace),
      getJson(root + '/deps.json'),
      getJson(root + '/l2_swimlane_records.json'),
      getJson(root + '/' + (config.nameMap || 'name_map.json')),
      getText(config.hints),
      getText(config.memory)
    ]).then(function (items) {
      return normalize(config, items[0], items[1], items[2], items[3], items[4], items[5], prefix || '');
    });
  }

  function rankConfig(base, rank) {
    var root = base.scenario + '/dfx_outputs/rank' + rank + '/d0';
    return {
      id: base.id + '-r' + rank,
      label: base.label + ' Rank ' + rank,
      kind: 'moe',
      path: base.path + ' · Rank ' + rank,
      rank: rank,
      root: root,
      trace: 'merged_swimlane_' + ((base.rankTimestamps && base.rankTimestamps[rank]) || base.timestamp) + '.json',
      nameMap: 'name_map.json'
    };
  }

  function rankSummary(evidence, rank) {
    var waitEvents = evidence.events.filter(function (event) { return /wait|gather|meta|push/i.test(event.funcName); });
    var computeEvents = evidence.events.filter(function (event) { return /^exp_/i.test(event.funcName); });
    var wait = waitEvents.length ? Math.max.apply(null, waitEvents.map(function (event) { return event.dur; })) : 0;
    var compute = computeEvents.length ? Math.max.apply(null, computeEvents.map(function (event) { return event.start + event.dur; })) - Math.min.apply(null, computeEvents.map(function (event) { return event.start; })) : 0;
    var longest = evidence.events.slice().sort(function (a, b) { return b.dur - a.dur; })[0];
    return {
      id: 'rank-' + rank,
      name: 'Rank ' + rank,
      rank: rank,
      window: evidence.window,
      wait: round(wait, 2),
      compute: round(compute, 2),
      eventCount: evidence.eventCount,
      taskCount: evidence.taskCount,
      tensorCount: evidence.tensorCount,
      dominant: longest ? longest.funcName : '未采集',
      dominantEventId: longest ? longest.id : ''
    };
  }

  function loadMoe(config) {
    return Promise.all([
      Promise.all(config.ranks.map(function (rank) { return loadFlat(rankConfig(config, rank), 'r' + rank + ':'); })),
      getText(config.hints),
      getText(config.memory),
      getOptionalJson(config.distributedMeta)
    ]).then(function (loaded) {
      var rankEvidence = loaded[0];
      var hints = parseHints(loaded[1]);
      var memory = parseMemoryReport(loaded[2]);
      var distributedMeta = loaded[3];
      var first = rankEvidence[0];
      var rankById = {};
      rankEvidence.forEach(function (item, index) { rankById[index] = item; });
      var allEvents = [];
      var allLanes = [];
      var allTasks = [];
      var allTensors = [];
      var allEdges = [];
      var taskIndex = {};
      var tensorIndex = {};
      rankEvidence.forEach(function (item, index) {
        allEvents = allEvents.concat(item.events);
        allLanes = allLanes.concat(item.lanes);
        allTasks = allTasks.concat(item.tasks);
        allTensors = allTensors.concat(item.tensors);
        allEdges = allEdges.concat(item.edges);
        Object.keys(item.taskIndex).forEach(function (key) { taskIndex[key] = item.taskIndex[key]; });
        Object.keys(item.tensorIndex).forEach(function (key) { tensorIndex[key] = item.tensorIndex[key]; });
        rankById[index].summary = rankSummary(item, config.ranks[index]);
      });
      var longest = allEvents.slice().sort(function (a, b) { return b.dur - a.dur; })[0] || null;
      return {
        id: config.id,
        label: config.label,
        kind: 'moe',
        path: config.path,
        window: round(Math.max.apply(null, rankEvidence.map(function (item) { return item.window; })), 2),
        taskCount: first.taskCount,
        tensorCount: first.tensorCount,
        edgeCount: first.edgeCount,
        totalTaskCount: allTasks.length,
        totalTensorCount: allTensors.length,
        totalEdgeCount: allEdges.length,
        eventCount: allEvents.length,
        rankCount: rankEvidence.length,
        coreCount: first.coreCount,
        aicoreCount: rankEvidence.reduce(function (sum, item) { return sum + item.aicoreCount; }, 0),
        aicpuCount: rankEvidence.reduce(function (sum, item) { return sum + item.aicpuCount; }, 0),
        dominant: longest ? longest.funcName : '未采集',
        dominantEventId: longest ? longest.id : '',
        memory: memory.summary,
        memoryMetrics: memory.metrics,
        memorySections: memory.sections,
        memoryHotspots: memory.hottest,
        hint: hints.length ? hints[0].code : '未采集',
        hints: hints,
        hintGroups: groupHints(hints),
        events: allEvents,
        lanes: allLanes,
        tasks: allTasks,
        tensors: allTensors,
        edges: allEdges,
        taskIndex: taskIndex,
        tensorIndex: tensorIndex,
        ranks: rankEvidence.map(function (item) { return item.summary; }),
        rankEvidence: rankById,
        tracePath: config.scenario + '/dfx_outputs/rank*/d0/merged_swimlane_' + config.timestamp + '.json',
        depsPath: config.scenario + '/dfx_outputs/rank*/d0/deps.json',
        l2Path: config.scenario + '/dfx_outputs/rank*/d0/l2_swimlane_records.json',
        nameMapPath: config.scenario + '/dfx_outputs/rank*/d0/name_map.json',
        sourceEvidence: { perfHints: config.hints, memoryReport: config.memory, distributedMeta: config.distributedMeta },
        artifactChain: [config.scenario + '/dfx_outputs/rank*/d0/merged_swimlane_*.json', config.scenario + '/dfx_outputs/rank*/d0/deps.json', config.hints, config.memory, config.distributedMeta],
        metadata: first.metadata,
        distributedMeta: distributedMeta
      };
    });
  }

  function finding(id, title, summary, severity, event, evidence, action, context) {
    return Object.assign({ id: id, title: title, summary: summary, severity: severity, event: event, evidence: evidence, action: action }, context || {});
  }

  function eventMatching(evidence, pattern) {
    return (evidence.events || []).filter(function (event) { return pattern.test(event.funcName); }).sort(function (a, b) { return b.dur - a.dur; })[0] || (evidence.events || [])[0];
  }

  function makeFindings(bundle) {
    var swa = bundle.swa;
    var ep8 = bundle.ep8;
    var f = [];
    if (swa) {
      var memoryEvent = eventMatching(swa, /q.*proj|kv_proj|hc_pre/i);
      var memoryHotspot = (swa.memoryHotspots || []).find(function (item) { return item.region === 'Right' && item.percent >= 100; }) || (swa.memoryHotspots || [])[0];
      f.push(finding('DFX-MEM-001', 'Pipeline buffer reuse 受 Right 内存共存约束', '软件流水请求 depth 2，但现有 co-resident buffer 使每级只能获得 1 / 2 buffer；' + (memoryHotspot ? memoryHotspot.kernel + ' 的 ' + memoryHotspot.region + ' 已达 ' + memoryHotspot.percent + '%。' : '需结合 buffer live range 确认。'), 'High', memoryEvent && memoryEvent.id, 'MemoryReuse → kernel buffer → live range', '打开 Memory 证据', { candidate: 'swa', evidenceTab: 'memory', traceView: 'timeline', story: 'memory' }));
      var tileHint = (swa.hintGroups || []).find(function (item) { return item.code === 'PH001'; });
      var tileEvent = eventMatching(swa, /qk|pv|attn|swa/i);
      f.push(finding('DFX-TILE-001', 'Tile innermost dimension 低于 backend 建议', tileHint ? tileHint.count + ' 个 PH001 occurrence 指向 ' + tileHint.sources.length + ' 个 source location，可从 tensor layout 与 kernel event 交叉验证。' : 'Perf hint 指向 tile granularity，需联合 tensor shape / stride 验证。', 'Medium', tileEvent && tileEvent.id, 'PH001 → tensor layout → source', '查看 Perf Hints', { candidate: 'swa', evidenceTab: 'hints', traceView: 'timeline', story: 'tile' }));
    }
    if (ep8) {
      var slow = ep8.ranks.slice().sort(function (a, b) { return b.window - a.window; })[0];
      f.push(finding('DFX-RANK-001', 'EP8 rank 执行窗口存在明显不均衡', slow.name + ' 为 ' + slow.window + ' μs，最长事件是 ' + slow.dominant + '；先在 Rank Overview 定位长尾，再展开该 Rank 的 core 与 task。', 'High', slow.dominantEventId, ep8.rankCount + ' ranks → ' + slow.name + ' → ' + slow.dominant, '展开 ' + slow.name, { candidate: 'ep8', rank: slow.name, evidenceTab: 'summary', traceView: 'timeline', story: 'rank' }));
    }
    return f;
  }

  function load() {
    return Promise.all([
      loadFlat(MANIFEST.csa),
      loadFlat(MANIFEST.hca),
      loadFlat(MANIFEST.swa),
      loadMoe(MANIFEST.ep2),
      loadMoe(MANIFEST.ep8)
    ]).then(function (items) {
      var bundle = { csa: items[0], hca: items[1], swa: items[2], ep2: items[3], ep8: items[4] };
      bundle.findings = makeFindings(bundle);
      return bundle;
    });
  }

  global.PtoDfxEvidence = { manifest: MANIFEST, load: load };
}(window));
