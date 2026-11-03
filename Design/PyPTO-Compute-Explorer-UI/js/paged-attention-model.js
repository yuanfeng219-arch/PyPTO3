/* Task assignment, logical tiling and software pipeline for paged_attention_pypto.py.
 * Source constants: 17–32; task / page / stack mapping: 261–269.
 * Pipeline source: embedded paged-attention-pypto-source.js, PRE_LAUNCH loop and READY events.
 * Sequence lengths are editable scenario inputs, not captured runtime values.
 */
(function (global) {
  'use strict';
  const constants = Object.freeze({ batch: 16, kvHeads: 8, qPerKv: 5, blocks: 24, pageTokens: 128, pagesPerStack: 4, maxSeq: 3584, preLaunch: 2, transferSlots: 3 });
  const defaultSeqLens = Object.freeze([1025, 512, 2048, 128, 1536, 3584, 3072, 256, 769, 1024, 2049, 640, 2560, 129, 1792, 3328]);
  function taskInfo(id) {
    const kv = id % constants.kvHeads;
    return { id, batch: Math.floor(id / constants.kvHeads), kv, block: id % constants.blocks,
      qStart: kv * constants.qPerKv, qEnd: (kv + 1) * constants.qPerKv - 1 };
  }
  function tasksForBlock(block) {
    return Array.from({ length: Math.ceil((constants.batch * constants.kvHeads - block) / constants.blocks) }, (_, i) => block + i * constants.blocks);
  }
  function partition(seqLen) {
    const pageCount = Math.ceil(seqLen / constants.pageTokens);
    const stackCount = Math.ceil(pageCount / constants.pagesPerStack);
    return { seqLen, pageCount, stackCount, stacks: Array.from({ length: stackCount }, (_, id) => ({ id,
      pages: Array.from({ length: Math.min(constants.pagesPerStack, pageCount - id * constants.pagesPerStack) }, (_, i) => {
        const pageId = id * constants.pagesPerStack + i;
        return { id: pageId, validTokens: Math.min(constants.pageTokens, seqLen - pageId * constants.pageTokens) };
      }),
    })) };
  }
  function pipelineFor(seqLen) {
    const tiling = partition(seqLen);
    const stepCount = tiling.stackCount + constants.preLaunch;
    return { ...tiling, stepCount,
      steps: Array.from({ length: stepCount }, (_, tick) => ({ tick,
        produceStack: tick < tiling.stackCount ? tick : null,
        consumeStack: tick >= constants.preLaunch ? tick - constants.preLaunch : null })),
      stacks: tiling.stacks.map((stack) => ({ ...stack, slot: stack.id % constants.transferSlots,
        tokenStart: stack.id * constants.pagesPerStack * constants.pageTokens,
        tokenEnd: Math.min(seqLen, (stack.id + 1) * constants.pagesPerStack * constants.pageTokens) - 1,
        previousStack: stack.id >= constants.transferSlots ? stack.id - constants.transferSlots : null })) };
  }
  // Reconcile the single context after a scenario edit; valid instances keep their tick.
  function reconcileSelection(data, selection) {
    const producing = ['qk', 'softmax'].includes(selection.stage);
    if (selection.stack !== null) {
      const stack = Math.min(selection.stack, data.stackCount - 1);
      const step = data.steps.find((item) => (producing ? item.produceStack : item.consumeStack) === stack);
      return { ...selection, stack, pipelineStep: step.tick };
    }
    const pipelineStep = Math.max(0, Math.min(selection.pipelineStep, data.stepCount - 1));
    const step = data.steps[pipelineStep];
    return { ...selection, pipelineStep, stack: producing ? step.produceStack : step.consumeStack };
  }
  global.PtoPagedAttentionIntro = {model:{constants,defaultSeqLens,taskInfo,tasksForBlock,partition,pipelineFor,reconcileSelection}};
})(window);
