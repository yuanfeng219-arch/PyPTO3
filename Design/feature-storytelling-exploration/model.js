/* Paged Attention rules from ../PyPTO-Compute-Explorer-UI/js/paged-attention-model.js
   and the corresponding paged_attention_pypto.py source snapshot, lines 17–36, 261–269. */
const model = (() => {
  const batches = 16, heads = 8, blocks = 24;
  const pageSize = 128, stackPages = 4, stackSize = pageSize * stackPages;
  const preLaunch = 2, slots = preLaunch + 1, max = 3584;
  const seqLens = [1025, 512, 2048, 128, 1536, 3584, 3072, 256, 769, 1024, 2049, 640, 2560, 129, 1792, 3328];
  const tasks = Array.from({length: batches * heads}, (_, id) => {
    const batch = Math.floor(id / heads);
    return {id, batch, head: id % heads, block: id % blocks, length: seqLens[batch]};
  });
  const groups = Array.from({length: blocks}, (_, block) => tasks.filter(task => task.block === block));
  function partition(length) {
    const pages = Array.from({length: Math.ceil(length / pageSize)}, (_, id) => ({
      id, tokens: Math.min(pageSize, length - id * pageSize)
    }));
    const stacks = Array.from({length: Math.ceil(pages.length / stackPages)}, (_, id) => ({
      id, pages: pages.slice(id * stackPages, (id + 1) * stackPages),
      slot: id % slots, produce: id, consume: id + preLaunch
    }));
    return {length, pages, stacks, ticks: stacks.length + preLaunch};
  }
  return {batches, heads, blocks, tasks, groups, pageSize, stackPages, stackSize, slots, max, partition};
})();
