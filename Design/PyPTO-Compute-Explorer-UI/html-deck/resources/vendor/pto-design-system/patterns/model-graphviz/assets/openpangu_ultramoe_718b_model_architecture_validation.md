# openPangu-Ultra-MoE-718B-V1.1 · Architecture Validation

Extraction via `model-architecture-extractor` skill. Source-of-truth = local repo
`/Users/yin/gitcode/openPangu-Ultra-MoE-718B-V1.1`.

## Sources

| id | path | role |
|---|---|---|
| `runtime_config` | `config.json` | source_of_truth (numeric facts) |
| `model_source` | `inference/model.py` | source_of_truth (module/op skeleton, forward flow) |
| `config_class` | `configuration_openpangu_moe.py` | supporting (default dataclass values) |

Weights (`*.safetensors`, 62 shards) were intentionally not downloaded; structure is derived
from source + config only, not from a checkpoint or profiling trace.

## Extraction scope

- `kind = full_source` — config + source describe the whole model.
- `full_main_layers = 61` (`config.json` `num_hidden_layers=61`).
- No profiling trace ingested; `trace_main_layers = null`.

## Validated facts (config.json ↔ model.py)

| Fact | Value | config.json | model.py |
|---|---|---|---|
| hidden_size `H` | 7680 | L13 | embed/linears |
| vocab_size `V` | 153600 | L39 | `embed_tokens` L713, `lm_head` L805 |
| num_hidden_layers `L` | 61 | L24 | `ModuleList` L716–721 |
| num_dense_layers | 3 | L11 | FFN branch L619–626 |
| → dense layers | 0–2 | — | `layer_idx < num_dense_layers` |
| → MoE layers | 3–60 (58) | — | `layer_idx >= num_dense_layers` |
| num_routed_experts `E` | 256 | L20 | `MoE.group_w1_w3` L232 |
| num_shared_experts | 1 | L21 | `shared_experts = MLP` L320 |
| num_experts_per_tok `top_k` | 8 | L23 | `MoEGate.top_k` L268, `topk` L284 |
| routed_scaling_factor | 2.5 | L32 | `topk_weight * scaling` L290 |
| num_attention_heads | 128 | L22 | `MLA.num_heads` L411 |
| attention_q_lora_dim | 1536 | L27 | q LoRA path L429–437 |
| attention_kv_lora_dim | 512 | L16 | `kv_a_proj_with_mqa` L439 |
| attention_qk_rope_dim | 64 | L29 | decoupled rope key L439/513 |
| attention_qk_dim | 128 | L28 | `q_head_dim = qk_dim+qk_rope` L422 |
| attention_v_dim | 128 | L38 | `kv_b_proj_w_v` L453 |
| intermediate_size | 18432 | L15 | Dense `MLP` L184 |
| moe_intermediate_size | 2048 | L19 | expert MLP L314 |
| sandwich_norm | true | L33 | pre/post-MLP norms L633–640 |
| rms_norm_eps | 1e-5 | L30 | RMSNorm L627 |
| rope_theta | 25600000 | L31 | RoPE L416/726 |
| max_position_embeddings `Smax` | 131072 | L17 | KV cache len L853 |
| tie_word_embeddings | false | L34 | separate `lm_head` L805 |

## Canonical forward (folded)

```
Token IDs → Parallel Embedding → [ Decoder Layer ]×61 → Final RMSNorm → LM Head → Logits

Decoder Layer (sandwich norm):
  Input RMSNorm
  → MLA: q_a→q_ln→q_b ; kv_a→kv_ln ; (k_absorb, RoPE, KV Cache) → score → softmax → context → v_absorb → o_proj → [All-Reduce]
  → Post-Attn RMSNorm → Pre-MLP RMSNorm
  → FFN branch:
       layer 0–2  : Dense MLP  (up-gate → SwiGLU → down → [All-Reduce])
       layer 3–60 : MoE        (Router[linear→sigmoid→TopK→gather→norm·scale]
                                 + init-routing → grouped up-gate → SwiGLU → grouped down → finalize → [All-Reduce]
                                 + Shared Expert MLP → Combine)
  → Post-MLP RMSNorm
```

## Modeling notes / boundaries

1. **MLA (Multi-head Latent Attention).** Q uses LoRA down/up projections; KV is compressed to a
   512-dim latent plus a 64-dim decoupled RoPE key (MQA-style). K/V are reconstructed from the latent
   via absorbed weight matmuls (`kv_b_proj_w_k`/`w_v`), not stored expanded — so `KV Cache` stores the
   compressed latent `[B,1,Smax,512+64]` (model.py:856), a real MLA memory saving.
2. **MoE routing is sigmoid + TopK** (not softmax gating), normalized then scaled by 2.5. NPU fused
   ops `npu_moe_init_routing` / `npu_grouped_matmul` / `npu_moe_finalize_routing` implement the
   dispatch → expert compute → combine path.
3. **MTP declared but not built here.** `config.num_mtp_layers=1`, but `inference/model.py` forward
   has no MTP head. MTP is therefore excluded from this graph (record it in scope notes only).
4. **Distributed primitives are conditional.** All-Reduce (attn/dense/moe, on `*_tp_size>1`) and
   All-Gather (logits, on `embed_tp_size>1`) are kept as op nodes but no-op at parallel size 1.
   Vocab is sharded for embedding/lm_head when `embed_tp_size>1`.
5. **Residual adds are fused.** `PanguUltraMoERMSNorm` uses `npu_add_rms_norm` returning
   `(normed, residual)`; residual addition is folded into the norm op rather than drawn as separate
   Add nodes.

## Validator result

`scripts/validate_model_architecture.py outputs/model_architecture.json` → **ok: True**
(49 nodes, 56 edges, 0 errors).

Remaining warnings are all on intra-block **connector edges** (e.g. `score→softmax`,
`up-gate→SwiGLU`) that carry no numeric fact; tensor payloads and provenance are attached on the
edges that do carry shapes/dtypes/constraints. These warnings are accepted by design, not defects.

## Comparison vs existing visualization

`pangu-moe-trainviz/data/graph-ultramoe-718b.js` matches this extraction on the major structure
(Embedding, Dense×3, MoE 3–60, 256 routed + 1 shared, top-8, MLA, MTP-aware). One field-name
alignment: that builder/`knowledge.md` use HF-generic `first_k_dense_replace=3`; the V1.1 official
config field is **`num_dense_layers=3`** (same semantics, layers 0–2 dense).
