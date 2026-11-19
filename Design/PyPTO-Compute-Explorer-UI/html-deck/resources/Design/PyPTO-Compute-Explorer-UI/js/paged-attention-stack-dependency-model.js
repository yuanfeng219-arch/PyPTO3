/* Static, source-backed dependencies for one Intro pipeline result. */
(function (global) {
  'use strict';

  const stages = Object.freeze(['qk', 'softmax', 'pv', 'accumulate']);
  const FILE = 'paged_attention_pypto.py';
  const source = (ranges) => ({ file: FILE, ranges: ranges.map(([start, end]) => ({ start, end })) });
  const ranges = {
    scoreWrite: [[301, 305], [307, 356]], scoreRead: [[765, 776]],
    probabilityWrite: [[812, 822]], probabilityRead: [[430, 502], [522, 538], [576, 635], [662, 714]],
    pvWrite: [[503, 503], [505, 718]], pvRead: [[830, 841]],
    mlState: [[789, 801], [856, 858]], oState: [[842, 844], [856, 861]], fifo: [[754, 760], [823, 861]],
  };

  function build(pipeline, synchronizations) {
    const nodes = [], edges = [];
    const steps = new Map(pipeline.steps.map((step) => [step.tick, step]));
    const syncById = new Map(synchronizations.map((item) => [item.id, item]));
    const nodeId = (stack, stage) => `s${stack}:${stage}`;
    const addEdge = (edge) => edges.push({ versions: [], ...edge });

    for (const stack of pipeline.stacks) {
      const produce = [...steps.values()].find((step) => step.produceStack === stack.id);
      const consume = [...steps.values()].find((step) => step.consumeStack === stack.id);
      const validTokens = stack.pages.reduce((sum, page) => sum + page.validTokens, 0);
      for (const stage of stages) {
        const producing = stage === 'qk' || stage === 'softmax';
        nodes.push({ id: nodeId(stack.id, stage), stack: stack.id, stage,
          core: stage === 'qk' || stage === 'pv' ? 'AIC' : 'AIV',
          tick: (producing ? produce : consume).tick, slot: stack.slot, validTokens });
      }
      const dataEdges = [
        ['qk', 'softmax', 'score_transfer', 'QK 写入 Scores，Softmax 读取当前 Stack 的 Scores。', ranges.scoreWrite.concat(ranges.scoreRead)],
        ['softmax', 'pv', 'probability_transfer', 'Softmax 写入概率权重，PV 读取当前 Stack 的概率权重。', ranges.probabilityWrite.concat(ranges.probabilityRead)],
        ['pv', 'accumulate', 'pv_transfer', 'PV 写入 partial，Accumulate 读取当前 Stack 的 partial。', ranges.pvWrite.concat(ranges.pvRead)],
      ];
      for (const [from, to, buffer, explanation, ref] of dataEdges) {
        addEdge({ id: `data:s${stack.id}:${buffer}`, from: nodeId(stack.id, from), to: nodeId(stack.id, to), kind: 'data',
          label: buffer, explanation, source: source(ref) });
      }
      const syncEdges = [
        ['qk-ready', 'qk', 'softmax', 'QK 写完 score buffer 并 SET；Softmax 读取前 WAIT。'],
        ['softmax-ready', 'softmax', 'pv', 'Softmax 写完 probability buffer 并 SET；PV 读取前 WAIT。'],
        ['pv-ready', 'pv', 'accumulate', 'PV 写完 pv buffer 并 SET；Accumulate 读取前 WAIT。'],
      ];
      for (const [eventId, from, to, explanation] of syncEdges) {
        const synchronization = syncById.get(eventId);
        const branchIndex = synchronization.consumer.waitPageCounts?.indexOf(stack.pages.length);
        const consumerSource = branchIndex >= 0 ? { ...synchronization.consumer.source,
          ranges: [synchronization.consumer.source.ranges[branchIndex]] } : synchronization.consumer.source;
        addEdge({ id: `sync:s${stack.id}:${eventId}`, from: nodeId(stack.id, from), to: nodeId(stack.id, to), kind: 'sync',
          label: synchronization.name, explanation: explanation + (branchIndex >= 0 ? `本 Stack 含 ${stack.pages.length} Pages；源码链接定位对应的 wait 分支。` : ''), synchronization, consumerSource,
          source: source(synchronization.producer.source.ranges.concat(synchronization.consumer.source.ranges).map(({ start, end }) => [start, end])) });
      }
      if (stack.id > 0) {
        const previous = stack.id - 1;
        addEdge({ id: `state:m-l:s${previous}:s${stack.id}`, from: nodeId(previous, 'softmax'), to: nodeId(stack.id, 'softmax'), kind: 'state',
          label: 'running m / l', explanation: '下一 Stack 的 Online Softmax 读取前一 Stack 迭代后的 running maximum m 与 denominator l。',
          versions: [`S${previous} 完成后: m${previous}, l${previous}`, `S${stack.id} 完成后: m${stack.id}, l${stack.id}`], source: source(ranges.mlState) });
        addEdge({ id: `state:o:s${previous}:s${stack.id}`, from: nodeId(previous, 'accumulate'), to: nodeId(stack.id, 'accumulate'), kind: 'state',
          label: 'context o', explanation: '下一 Stack 累加前读取上一 Stack 更新后的 context accumulator o。',
          versions: [`S${previous} 完成后: o${previous}`, `S${stack.id} 完成后: o${stack.id}`], source: source(ranges.oState) });
      }
      const tick = produce.tick;
      const consumeTick = consume.tick;
      const moveTick = consumeTick - 1;
      addEdge({ id: `state:rescale:s${stack.id}`, from: nodeId(stack.id, 'softmax'), to: nodeId(stack.id, 'accumulate'), kind: 'state',
        label: 'rescale FIFO', explanation: `Stack ${stack.id} 的 rescale r${stack.id} 在 tick ${tick} 进入 pending1；tick ${moveTick} 移入 pending0；tick ${consumeTick} 消费并缩放 context。`,
        versions: [`tick ${tick}: pending1=r${stack.id}`, `tick ${moveTick}: pending0=r${stack.id}`, `tick ${consumeTick}: consume r${stack.id}`], source: source(ranges.fifo) });
    }
    for (const stack of pipeline.stacks) {
      if (stack.previousStack === null) continue;
      const previous = stack.previousStack;
      const reuse = [
        ['score_transfer', 'softmax', 'qk', '旧 Stack 的 Softmax 读取 score 后，后续 Stack 才能覆盖同一 slot。', ranges.scoreRead.concat(ranges.scoreWrite)],
        ['probability_transfer', 'pv', 'softmax', '旧 Stack 的 PV 读取 probability 后，后续 Softmax 才能覆盖同一 slot。', ranges.probabilityRead.concat(ranges.probabilityWrite)],
        ['pv_transfer', 'accumulate', 'pv', '旧 Stack 的 Accumulate 读取 pv partial 后，后续 PV 才能覆盖同一 slot。', ranges.pvRead.concat(ranges.pvWrite)],
      ];
      for (const [buffer, from, to, explanation, ref] of reuse) {
        const oldTick = from === 'softmax' ? pipeline.steps.find((s) => s.produceStack === previous).tick : pipeline.steps.find((s) => s.consumeStack === previous).tick;
        const newTick = to === 'qk' || to === 'softmax' ? pipeline.steps.find((s) => s.produceStack === stack.id).tick : pipeline.steps.find((s) => s.consumeStack === stack.id).tick;
        addEdge({ id: `reuse:${buffer}:s${previous}:s${stack.id}`, from: nodeId(previous, from), to: nodeId(stack.id, to), kind: 'reuse', buffer, versions: [`S${previous} 读取 ${buffer}[Slot ${stack.slot}]`, `S${stack.id} 覆盖 ${buffer}[Slot ${stack.slot}]`],
          label: `${buffer} slot ${stack.slot}`, explanation: `${explanation}旧读发生于 tick ${oldTick}，新写发生于 tick ${newTick}；这是静态 Buffer 覆盖约束，不表示额外 free event 或已发生冲突。`, source: source(ref) });
      }
    }
    return { ...pipeline, nodes, edges };
  }

  global.PtoPagedAttentionStackDependencyModel = { build, stages };
})(typeof window === 'undefined' ? globalThis : window);
