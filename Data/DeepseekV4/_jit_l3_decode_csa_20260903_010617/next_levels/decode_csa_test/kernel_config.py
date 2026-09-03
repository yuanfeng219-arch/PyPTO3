# Kernel and Orchestration Configuration

from pathlib import Path

from simpler.task_interface import ArgDirection as _D

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer.
# AICPU thread count 0 selects the runtime's architecture default (a2a3: 4; a5: 5).
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 0,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "decode_csa_test.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT, _D.INOUT, _D.OUT, _D.INOUT, _D.OUT, _D.INOUT],
}

KERNELS = [
	{"func_id": 0, "name": "hc_pre_rms", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_rms.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 1, "name": "hc_pre_linear", "source": str(_ROOT_DIR / "kernels" / "aic" / "hc_pre_linear.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 2, "name": "hc_pre_linear_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_linear_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 3, "name": "split_pre_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "split_pre_post.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT, _D.INOUT, _D.INOUT]},
	{"func_id": 4, "name": "comb_sinkhorn", "source": str(_ROOT_DIR / "kernels" / "aiv" / "comb_sinkhorn.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT, _D.INOUT]},
	{"func_id": 5, "name": "mix_x", "source": str(_ROOT_DIR / "kernels" / "aiv" / "mix_x.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.INOUT, _D.IN]},
	{"func_id": 6, "name": "csa_rope_interleave", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_rope_interleave.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.OUT, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 7, "name": "rms_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rms_norm.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.IN]},
	{"func_id": 8, "name": "cp_token_allgather_push", "source": str(_ROOT_DIR / "kernels" / "aiv" / "cp_token_allgather_push.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.INOUT]},
	{"func_id": 9, "name": "cp_token_allgather_payload_wait", "source": str(_ROOT_DIR / "kernels" / "aiv" / "cp_token_allgather_payload_wait.cpp"), "core_type": "aiv", "signature": [_D.IN]},
	{"func_id": 10, "name": "cp_token_allgather_readback", "source": str(_ROOT_DIR / "kernels" / "aiv" / "cp_token_allgather_readback.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.INOUT]},
	{"func_id": 11, "name": "cp_token_allgather_readback_wait", "source": str(_ROOT_DIR / "kernels" / "aiv" / "cp_token_allgather_readback_wait.cpp"), "core_type": "aiv", "signature": [_D.IN]},
	{"func_id": 12, "name": "cp_token_allgather_retire", "source": str(_ROOT_DIR / "kernels" / "aiv" / "cp_token_allgather_retire.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT]},
	{"func_id": 13, "name": "q_rope_prepare", "source": str(_ROOT_DIR / "kernels" / "aiv" / "q_rope_prepare.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 14, "name": "q_rope_prepare_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "q_rope_prepare_0.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 15, "name": "qr_proj_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_proj_seed.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 16, "name": "qr_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qr_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 17, "name": "qr_rms_norm_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_rms_norm_quant.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.INOUT]},
	{"func_id": 18, "name": "qproj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qproj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 19, "name": "qproj_dequant_rms_nope_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qproj_dequant_rms_nope_rope.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 20, "name": "kv_proj_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_proj_seed.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 21, "name": "kv_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 22, "name": "kv_rms_norm_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_rms_norm_rope.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 23, "name": "csa_cache_writeback", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_cache_writeback.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 24, "name": "kv_score_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_score_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 25, "name": "scatter_softmax_pool", "source": str(_ROOT_DIR / "kernels" / "aiv" / "scatter_softmax_pool.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 26, "name": "compress_state_commit", "source": str(_ROOT_DIR / "kernels" / "aiv" / "compress_state_commit.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 27, "name": "rmsnorm_rope_cache_write", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rmsnorm_rope_cache_write.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.OUT, _D.OUT, _D.IN]},
	{"func_id": 28, "name": "kv_score_proj_0", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_score_proj_0.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 29, "name": "scatter_softmax_pool_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "scatter_softmax_pool_0.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 30, "name": "compress_state_commit_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "compress_state_commit_0.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 31, "name": "rmsnorm_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rmsnorm_rope.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT, _D.IN]},
	{"func_id": 32, "name": "kv_hadamard", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_hadamard.cpp"), "core_type": "aic", "signature": [_D.IN, _D.OUT, _D.IN]},
	{"func_id": 33, "name": "kv_and_cache_write", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_and_cache_write.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.OUT, _D.IN, _D.OUT]},
	{"func_id": 34, "name": "idx_qr_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "idx_qr_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 35, "name": "idx_qr_proj_dequant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "idx_qr_proj_dequant.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 36, "name": "qr_rope_swap_idx", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_rope_swap_idx.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 37, "name": "qr_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_rope.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 38, "name": "qr_hadamard_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qr_hadamard_matmul.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 39, "name": "qr_hadamard_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_hadamard_quant.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.OUT]},
	{"func_id": 40, "name": "weights_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "weights_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 41, "name": "weights_proj_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "weights_proj_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 42, "name": "indexer_score_leaf_wave_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "indexer_score_leaf_wave_aic.cpp"), "core_type": "aic", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 43, "name": "indexer_score_leaf_wave_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "indexer_score_leaf_wave_aiv.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 44, "name": "indexer_topk_group_wave", "source": str(_ROOT_DIR / "kernels" / "aiv" / "indexer_topk_group_wave.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT]},
	{"func_id": 45, "name": "indexer_topk_query_merge", "source": str(_ROOT_DIR / "kernels" / "aiv" / "indexer_topk_query_merge.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT, _D.OUT, _D.OUT]},
	{"func_id": 46, "name": "kv_touch", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_touch.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 47, "name": "csa_slots_build_valid_qk_plan", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_slots_build_valid_qk_plan.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.INOUT, _D.OUT]},
	{"func_id": 48, "name": "qk_pv_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "qk_pv_aic.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 49, "name": "qk_pv_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_pv_aiv.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 50, "name": "rope_cs", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope_cs.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 51, "name": "csa_merge_pack_publish", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_merge_pack_publish.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.INOUT]},
	{"func_id": 52, "name": "o_group_a2a_wait", "source": str(_ROOT_DIR / "kernels" / "aiv" / "o_group_a2a_wait.cpp"), "core_type": "aiv", "signature": [_D.IN]},
	{"func_id": 53, "name": "o_group_a2a_gather", "source": str(_ROOT_DIR / "kernels" / "aiv" / "o_group_a2a_gather.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
	{"func_id": 54, "name": "o_group_a2a_complete", "source": str(_ROOT_DIR / "kernels" / "aiv" / "o_group_a2a_complete.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT]},
	{"func_id": 55, "name": "tp_o_a", "source": str(_ROOT_DIR / "kernels" / "aic" / "tp_o_a.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 56, "name": "tp_o_a_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "tp_o_a_quant.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN]},
	{"func_id": 57, "name": "tp_o_b", "source": str(_ROOT_DIR / "kernels" / "aic" / "tp_o_b.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 58, "name": "tp_o_b_dequant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "tp_o_b_dequant.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 59, "name": "tp_o_b_publish", "source": str(_ROOT_DIR / "kernels" / "aiv" / "tp_o_b_publish.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.INOUT]},
	{"func_id": 60, "name": "tp_o_rs_wait", "source": str(_ROOT_DIR / "kernels" / "aiv" / "tp_o_rs_wait.cpp"), "core_type": "aiv", "signature": [_D.IN]},
	{"func_id": 61, "name": "tp_o_rs_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "tp_o_rs_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 62, "name": "tp_o_rs_complete", "source": str(_ROOT_DIR / "kernels" / "aiv" / "tp_o_rs_complete.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT]},
	{"func_id": 63, "name": "hc_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_post.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN]},
]
