import torch
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType
from pypto.runtime.tensor_arg import make_tensor_arg
from pypto.runtime.distributed_runner import _submit_chip

def _alloc_intermediates(tensors, world_size=1):
    tensors["__comm_d0_ord"] = [torch.zeros((1,), dtype=torch.int32).share_memory_() for _ in range(max(world_size, 1))]

def l3_decode_csa(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size, _domain_provider=None):
    local_t__ssa_v0 = tensors["local_t__ssa_v0"]
    T_DYN = tensors["x_hc__ssa_v0"].shape[1]
    KV_T_DYN = tensors["freqs_cos__ssa_v0"].shape[1]
    MAIN_STATE_BLOCK_NUM_DYN = tensors["compress_state__ssa_v0"].shape[1]
    KV_B_DYN = tensors["compress_state_block_table__ssa_v0"].shape[1]
    INNER_STATE_BLOCK_NUM_DYN = tensors["inner_compress_state__ssa_v0"].shape[1]
    ORI_BLOCK_NUM_DYN = tensors["kv_cache__ssa_v0"].shape[1]
    CMP_BLOCK_NUM_DYN = tensors["cmp_kv__ssa_v0"].shape[1]
    B_DYN = tensors["cmp_block_table__ssa_v0"].shape[1]
    IDX_CACHE_BLOCK_NUM_DYN = tensors["idx_kv_cache__ssa_v0"].shape[1]
    with (_domain_provider or orch.allocate_domain)(
        name="comm_d0",
        workers=[*range(world_size)],
        window_size=((((4194304 + 31) // 32) * 32)) + ((((8 + 31) // 32) * 32)) + ((((16777216 + 31) // 32) * 32)) + ((((8 + 31) // 32) * 32)) + ((((4194304 + 31) // 32) * 32)) + ((((8 + 31) // 32) * 32)),
        buffers=[
            CommBufferSpec(name="gather_window_buf__ssa_v0", dtype="opaque", count=4194304, nbytes=(((4194304 + 31) // 32) * 32)),
            CommBufferSpec(name="gather_signal_buf__ssa_v0", dtype="opaque", count=8, nbytes=(((8 + 31) // 32) * 32)),
            CommBufferSpec(name="attention_window_buf__ssa_v0", dtype="opaque", count=16777216, nbytes=(((16777216 + 31) // 32) * 32)),
            CommBufferSpec(name="attention_signal_buf__ssa_v0", dtype="opaque", count=8, nbytes=(((8 + 31) // 32) * 32)),
            CommBufferSpec(name="o_window_buf__ssa_v0", dtype="opaque", count=4194304, nbytes=(((4194304 + 31) // 32) * 32)),
            CommBufferSpec(name="o_signal_buf__ssa_v0", dtype="opaque", count=8, nbytes=(((8 + 31) // 32) * 32)),
        ],
    ) as __comm_d0:
        t__tmp_v0 = world_size
        for rank__idx_v0 in range(0, t__tmp_v0, 1):
            tensors["t__tmp_v1"] = tensors["x_hc__ssa_v0"][rank__idx_v0, 0:T_DYN, 0:4, 0:4096]
            tensors["t__tmp_v2"] = tensors["hc_attn_fn__ssa_v0"][rank__idx_v0, 0:24, 0:16384]
            tensors["t__tmp_v3"] = tensors["hc_attn_scale__ssa_v0"][rank__idx_v0, 0:3]
            tensors["t__tmp_v4"] = tensors["hc_attn_base__ssa_v0"][rank__idx_v0, 0:24]
            tensors["t__tmp_v5"] = tensors["attn_norm_w__ssa_v0"][rank__idx_v0, 0:4096]
            tensors["t__tmp_v6"] = tensors["wq_a__ssa_v0"][rank__idx_v0, 0:4096, 0:1024]
            tensors["t__tmp_v7"] = tensors["wq_b__ssa_v0"][rank__idx_v0, 0:1024, 0:32768]
            tensors["t__tmp_v8"] = tensors["wq_b_scale__ssa_v0"][rank__idx_v0, 0:32768]
            tensors["t__tmp_v9"] = tensors["wkv__ssa_v0"][rank__idx_v0, 0:4096, 0:512]
            tensors["t__tmp_v10"] = tensors["gamma_cq__ssa_v0"][rank__idx_v0, 0:1024]
            tensors["t__tmp_v11"] = tensors["gamma_ckv__ssa_v0"][rank__idx_v0, 0:512]
            tensors["t__tmp_v12"] = tensors["freqs_cos_local__ssa_v0"][rank__idx_v0, 0:T_DYN, 0:64]
            tensors["t__tmp_v13"] = tensors["freqs_sin_local__ssa_v0"][rank__idx_v0, 0:T_DYN, 0:64]
            tensors["t__tmp_v14"] = tensors["freqs_cos__ssa_v0"][rank__idx_v0, 0:KV_T_DYN, 0:64]
            tensors["t__tmp_v15"] = tensors["freqs_sin__ssa_v0"][rank__idx_v0, 0:KV_T_DYN, 0:64]
            tensors["t__tmp_v16"] = tensors["cmp_freqs_cos__ssa_v0"][rank__idx_v0, 0:KV_T_DYN, 0:64]
            tensors["t__tmp_v17"] = tensors["cmp_freqs_sin__ssa_v0"][rank__idx_v0, 0:KV_T_DYN, 0:64]
            tensors["t__tmp_v18"] = tensors["cmp_wkv__ssa_v0"][rank__idx_v0, 0:1024, 0:4096]
            tensors["t__tmp_v19"] = tensors["cmp_wgate__ssa_v0"][rank__idx_v0, 0:1024, 0:4096]
            tensors["t__tmp_v20"] = tensors["cmp_ape__ssa_v0"][rank__idx_v0, 0:4, 0:1024]
            tensors["t__tmp_v21"] = tensors["cmp_norm_w__ssa_v0"][rank__idx_v0, 0:512]
            tensors["t__tmp_v22"] = tensors["compress_state__ssa_v0"][rank__idx_v0, 0:MAIN_STATE_BLOCK_NUM_DYN, 0:2, 0:2048]
            tensors["t__tmp_v23"] = tensors["compress_state_block_table__ssa_v0"][rank__idx_v0, 0:KV_B_DYN, 0:4]
            tensors["t__tmp_v24"] = tensors["idx_wq_b__ssa_v0"][rank__idx_v0, 0:1024, 0:8192]
            tensors["t__tmp_v25"] = tensors["idx_wq_b_scale__ssa_v0"][rank__idx_v0, 0:8192]
            tensors["t__tmp_v26"] = tensors["weights_proj__ssa_v0"][rank__idx_v0, 0:4096, 0:64]
            tensors["t__tmp_v27"] = tensors["hadamard_idx__ssa_v0"][rank__idx_v0, 0:128, 0:128]
            tensors["t__tmp_v28"] = tensors["inner_wkv__ssa_v0"][rank__idx_v0, 0:256, 0:4096]
            tensors["t__tmp_v29"] = tensors["inner_wgate__ssa_v0"][rank__idx_v0, 0:256, 0:4096]
            tensors["t__tmp_v30"] = tensors["inner_ape__ssa_v0"][rank__idx_v0, 0:4, 0:256]
            tensors["t__tmp_v31"] = tensors["inner_norm_w__ssa_v0"][rank__idx_v0, 0:128]
            tensors["t__tmp_v32"] = tensors["inner_compress_state__ssa_v0"][rank__idx_v0, 0:INNER_STATE_BLOCK_NUM_DYN, 0:2, 0:512]
            tensors["t__tmp_v33"] = tensors["inner_compress_state_block_table__ssa_v0"][rank__idx_v0, 0:KV_B_DYN, 0:4]
            tensors["t__tmp_v34"] = tensors["kv_cache__ssa_v0"][rank__idx_v0, 0:ORI_BLOCK_NUM_DYN, 0:32, 0:1, 0:512]
            tensors["t__tmp_v35"] = tensors["cmp_kv__ssa_v0"][rank__idx_v0, 0:CMP_BLOCK_NUM_DYN, 0:32, 0:1, 0:512]
            tensors["t__tmp_v36"] = tensors["cmp_block_table__ssa_v0"][rank__idx_v0, 0:B_DYN, 0:8192]
            tensors["t__tmp_v37"] = tensors["idx_kv_cache__ssa_v0"][rank__idx_v0, 0:IDX_CACHE_BLOCK_NUM_DYN, 0:32, 0:1, 0:128]
            tensors["t__tmp_v38"] = tensors["idx_kv_scale__ssa_v0"][rank__idx_v0, 0:IDX_CACHE_BLOCK_NUM_DYN, 0:32, 0:1, 0:1]
            tensors["t__tmp_v39"] = tensors["idx_block_table__ssa_v0"][rank__idx_v0, 0:B_DYN, 0:8192]
            tensors["t__tmp_v40"] = tensors["ori_slot_mapping__ssa_v0"][rank__idx_v0, 0:KV_T_DYN]
            tensors["t__tmp_v41"] = tensors["window_swa_indices__ssa_v0"][rank__idx_v0, 0:T_DYN, 0:128]
            tensors["t__tmp_v42"] = tensors["window_swa_lens__ssa_v0"][rank__idx_v0, 0:T_DYN]
            tensors["t__tmp_v43"] = tensors["cmp_slot_mapping__ssa_v0"][rank__idx_v0, 0:KV_T_DYN]
            tensors["t__tmp_v44"] = tensors["idx_slot_mapping__ssa_v0"][rank__idx_v0, 0:KV_T_DYN]
            tensors["t__tmp_v45"] = tensors["state_slot_mapping__ssa_v0"][rank__idx_v0, 0:KV_T_DYN]
            tensors["t__tmp_v46"] = tensors["inner_state_slot_mapping__ssa_v0"][rank__idx_v0, 0:KV_T_DYN]
            tensors["t__tmp_v47"] = tensors["position_ids_local__ssa_v0"][rank__idx_v0, 0:T_DYN]
            tensors["t__tmp_v48"] = tensors["position_ids__ssa_v0"][rank__idx_v0, 0:KV_T_DYN]
            tensors["t__tmp_v49"] = tensors["kv_seq_lens__ssa_v0"][rank__idx_v0, 0:B_DYN]
            tensors["t__tmp_v50"] = tensors["attn_sink__ssa_v0"][rank__idx_v0, 0:64]
            tensors["t__tmp_v51"] = tensors["wo_a__ssa_v0"][rank__idx_v0, 0:4, 0:1024, 0:4096]
            tensors["t__tmp_v52"] = tensors["wo_b__ssa_v0"][rank__idx_v0, 0:4096, 0:4096]
            tensors["t__tmp_v53"] = tensors["wo_b_scale__ssa_v0"][rank__idx_v0, 0:4096]
            tensors["t__tmp_v54"] = tensors["x_out__ssa_v0"][rank__idx_v0, 0:T_DYN, 0:4, 0:4096]
            _ta_0 = TaskArgs()
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v1"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v2"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v3"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v4"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v5"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v6"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v7"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v8"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v9"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v10"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v11"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v12"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v13"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v14"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v15"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v16"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v17"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v18"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v19"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v20"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v21"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v22"]), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v23"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v24"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v25"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v26"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v27"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v28"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v29"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v30"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v31"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v32"]), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v33"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v34"]), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v35"]), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v36"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v37"]), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v38"]), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v39"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v40"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v41"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v42"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v43"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v44"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v45"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v46"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v47"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v48"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v49"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v50"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v51"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v52"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v53"]), TensorArgType.INPUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["t__tmp_v54"]), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(__comm_d0[rank__idx_v0].buffers["gather_window_buf__ssa_v0"].tensor(shapes=(512, 4096), dtype=DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(__comm_d0[rank__idx_v0].buffers["gather_signal_buf__ssa_v0"].tensor(shapes=(2, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[rank__idx_v0].buffers["attention_window_buf__ssa_v0"].tensor(shapes=(2048, 4096), dtype=DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(__comm_d0[rank__idx_v0].buffers["attention_signal_buf__ssa_v0"].tensor(shapes=(2, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[rank__idx_v0].buffers["o_window_buf__ssa_v0"].tensor(shapes=(512, 4096), dtype=DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(__comm_d0[rank__idx_v0].buffers["o_signal_buf__ssa_v0"].tensor(shapes=(2, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["__comm_d0_ord"][rank__idx_v0]), TensorArgType.INOUT)
            _ta_0.add_scalar(0)
            _ta_0.add_scalar(rank__idx_v0)
            _ta_0.add_scalar(256)
            _ta_0.add_scalar(__comm_d0[rank__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[rank__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[rank__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[rank__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[rank__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[rank__idx_v0].device_ctx)
            _keep.append(_ta_0)
            _submit_chip(orch, callables["decode_csa_test"], _ta_0, config, rank__idx_v0)

l3_decode_csa._pypto_distributed_entry = True

