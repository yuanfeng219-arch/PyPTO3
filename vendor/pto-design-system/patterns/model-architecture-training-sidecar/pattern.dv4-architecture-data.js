(function registerDeepSeekV4ArchitectureData(global) {
  'use strict';

  const PRO_CONFIG = Object.freeze({
    modelId: 'deepseek-ai/DeepSeek-V4-Pro',
    modelLabel: 'DeepSeek V4 Pro',
    numHiddenLayers: 61,
    hiddenSize: 7168,
    hcMult: 4,
    hcSinkhornIters: 20,
    slidingWindow: 128,
    csaCompression: 4,
    hcaCompression: 128,
    numHashLayers: 3,
    routedExperts: 384,
    expertsPerToken: 6,
    sharedExperts: 1,
    routerScore: 'sqrt(softplus())',
    mtpDepth: 1
  });

  function clampLayer(value) {
    const layer = Number.parseInt(value, 10);
    if (!Number.isFinite(layer)) return 0;
    return Math.max(0, Math.min(PRO_CONFIG.numHiddenLayers - 1, layer));
  }

  function attentionTypeForLayer(value) {
    const layer = clampLayer(value);
    if (layer < 2) return 'hca';
    return layer % 2 === 0 ? 'csa' : 'hca';
  }

  function routerTypeForLayer(value) {
    return clampLayer(value) < PRO_CONFIG.numHashLayers ? 'hash_moe' : 'moe';
  }

  function describeLayer(value) {
    const layer = clampLayer(value);
    const attentionType = attentionTypeForLayer(layer);
    const routerType = routerTypeForLayer(layer);
    return Object.freeze({
      layer,
      attentionType,
      attentionLabel: attentionType === 'csa'
        ? 'CSA · Compressed Sparse Attention'
        : 'HCA · Heavily Compressed Attention',
      compression: attentionType === 'csa'
        ? PRO_CONFIG.csaCompression
        : PRO_CONFIG.hcaCompression,
      routerType,
      routerLabel: routerType === 'hash_moe'
        ? 'Hash Router · frozen Token ID lookup'
        : `Learned Router · ${PRO_CONFIG.routerScore}`,
      routeSelectionLabel: routerType === 'hash_moe'
        ? `Hash Expert IDs ×${PRO_CONFIG.expertsPerToken}`
        : `Top-${PRO_CONFIG.expertsPerToken} Selection`
    });
  }

  global.DeepSeekV4ArchitectureData = Object.freeze({
    config: PRO_CONFIG,
    clampLayer,
    attentionTypeForLayer,
    routerTypeForLayer,
    describeLayer,
    provenance: Object.freeze([
      'deepseek-ai/DeepSeek-V4-Pro config.json',
      'transformers DeepseekV4Config and DeepseekV4 modeling implementation'
    ])
  });
})(window);
