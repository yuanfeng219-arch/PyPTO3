# Kernel and Orchestration Configuration

from pathlib import Path

from simpler.task_interface import ArgDirection as _D

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer.
# This runtime requires 4 AICPU threads (3 schedulers + 1 orchestrator on thread 3).
# block_dim is only emitted when the user passes compile(block_dim=...);
# otherwise the runtime default applies (simpler validates against device capacity).
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 4,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "decode_fwd_layers.cpp"),
	"function_name": "aicpu_orchestration_entry"
}

KERNELS = [
	{"func_id": 0, "name": "copy_hidden", "source": str(_ROOT_DIR / "kernels" / "aiv" / "copy_hidden.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
	{"func_id": 1, "name": "x_gamma0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "x_gamma0.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 2, "name": "rms_recip", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rms_recip.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT]},
	{"func_id": 3, "name": "q_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "q_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 4, "name": "q_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "q_proj.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 5, "name": "k_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "k_seed.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 6, "name": "k_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "k_proj.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 7, "name": "v_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "v_seed.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 8, "name": "v_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "v_proj.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 9, "name": "fa_work_build", "source": str(_ROOT_DIR / "kernels" / "aiv" / "fa_work_build.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.OUT]},
	{"func_id": 10, "name": "qk_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_norm.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 11, "name": "qk_norm_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_norm_0.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 12, "name": "qk_norm_1", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_norm_1.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 13, "name": "qk_norm_2", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_norm_2.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 14, "name": "qk_norm_3", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_norm_3.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 15, "name": "qk_norm_4", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_norm_4.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 16, "name": "qk_norm_5", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_norm_5.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 17, "name": "qk_norm_6", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_norm_6.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 18, "name": "rope_qkv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope_qkv.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 19, "name": "down_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "down_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 20, "name": "gate_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "gate_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 21, "name": "up_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "up_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 22, "name": "fa_fused_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "fa_fused_aic.cpp"), "core_type": "aic", "signature": [_D.IN, _D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 23, "name": "fa_fused_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "fa_fused_aiv.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 24, "name": "online_softmax", "source": str(_ROOT_DIR / "kernels" / "aiv" / "online_softmax.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 25, "name": "out_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "out_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 26, "name": "out_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "out_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 27, "name": "residual_rms_cast", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 28, "name": "residual_rms_cast_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast_0.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 29, "name": "residual_rms_cast_1", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast_1.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 30, "name": "residual_rms_cast_2", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast_2.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 31, "name": "residual_rms_cast_3", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast_3.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 32, "name": "post_rms_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "post_rms_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 33, "name": "gate_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "gate_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 34, "name": "up_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "up_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 35, "name": "silu", "source": str(_ROOT_DIR / "kernels" / "aiv" / "silu.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 36, "name": "down_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "down_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 37, "name": "dcr_xgamma", "source": str(_ROOT_DIR / "kernels" / "aiv" / "dcr_xgamma.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT, _D.IN, _D.INOUT]},
	{"func_id": 38, "name": "copy_out", "source": str(_ROOT_DIR / "kernels" / "aiv" / "copy_out.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
]
