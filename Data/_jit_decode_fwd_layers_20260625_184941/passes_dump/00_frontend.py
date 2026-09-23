# pypto.program: _jit_decode_fwd_layers
import pypto.language as pl

@pl.program
class _jit_decode_fwd_layers:
    @pl.function(type=pl.FunctionType.Inline)
    def _decode_layer(self, hidden_states: pl.Tensor[[16, 5120], pl.FP32], input_rms_weight: pl.Tensor[[1, 5120], pl.FP32], wq: pl.Tensor[[5120, 5120], pl.BF16], wk: pl.Tensor[[5120, 1024], pl.BF16], wv: pl.Tensor[[5120, 1024], pl.BF16], q_norm_weight: pl.Tensor[[1, 128], pl.FP32], k_norm_weight: pl.Tensor[[1, 128], pl.FP32], seq_lens: pl.Tensor[[16], pl.INT32], block_table: pl.Tensor[[512], pl.INT32], slot_mapping: pl.Tensor[[16], pl.INT32], rope_cos: pl.Tensor[[4096, 128], pl.FP32], rope_sin: pl.Tensor[[4096, 128], pl.FP32], k_cache: pl.Tensor[[524288, 128], pl.BF16], v_cache: pl.Tensor[[524288, 128], pl.BF16], wo: pl.Tensor[[5120, 5120], pl.BF16], w_gate: pl.Tensor[[5120, 17408], pl.BF16], w_up: pl.Tensor[[5120, 17408], pl.BF16], w_down: pl.Tensor[[17408, 5120], pl.BF16], post_rms_weight: pl.Tensor[[1, 5120], pl.FP32], out: pl.Tensor[[16, 5120], pl.FP32], normed_in: pl.Tensor[[16, 5120], pl.BF16], normed_out: pl.Tensor[[16, 5120], pl.BF16], layer_idx: pl.Scalar[pl.INT32], next_gamma_idx: pl.Scalar[pl.INT32], prev_out_tids: pl.Array[5, pl.TASK_ID], prev_normed_tids: pl.Array[5, pl.TASK_ID]) -> pl.Tensor[[16, 5120], pl.FP32]:
        layer_hidden_base: pl.Scalar[pl.INDEX] = pl.cast(layer_idx, pl.INDEX) * 5120
        layer_inter_base: pl.Scalar[pl.INDEX] = pl.cast(layer_idx, pl.INDEX) * 17408
        num_layers_actual: pl.Scalar[pl.INDEX] = pl.tensor.dim(input_rms_weight, 0)
        layer_cache_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(k_cache, 0) // num_layers_actual
        layer_cache_base: pl.Scalar[pl.INDEX] = pl.cast(layer_idx, pl.INDEX) * layer_cache_rows
        user_batch: pl.Scalar[pl.INDEX] = pl.tensor.dim(seq_lens, 0)
        max_blocks_per_seq: pl.Scalar[pl.INDEX] = pl.tensor.dim(block_table, 0) // user_batch
        q_norm_w: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(q_norm_weight, [1, 128], [layer_idx, 0])
        k_norm_w: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(k_norm_weight, [1, 128], [layer_idx, 0])
        down_tids: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
        q_tile_tids: pl.Array[50, pl.TASK_ID] = pl.array.create(50, dtype=pl.TASK_ID)
        k_tile_tids: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
        v_tile_tids: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
        qk_tids: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
        rope_grp_tids: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
        inv_rms_states: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        q_proj: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        k_proj: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        v_proj: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        all_q_padded: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.create([2048, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        attn_out: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        all_oi_tmp: pl.Tensor[[65536, 128], pl.FP32] = pl.tensor.create([65536, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        all_cur_mi: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.create([65536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        all_cur_li: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.create([65536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        fa_work_table: pl.Tensor[[512, 1], pl.INT32] = pl.tensor.create([512, 1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        fa_total: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            _submit_deps_buf: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_1: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf, 0, pl.array.get_element(prev_out_tids, 0))
            _submit_deps_buf_2: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_1, 1, pl.array.get_element(prev_out_tids, 1))
            _submit_deps_buf_3: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_2, 2, pl.array.get_element(prev_out_tids, 2))
            _submit_deps_buf_4: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_3, 3, pl.array.get_element(prev_out_tids, 3))
            _submit_deps_buf_5: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_4, 4, pl.array.get_element(prev_out_tids, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip", deps=[_submit_deps_buf_5]) as rms_tid:
                partial_sq: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(20, stage=4):
                    k0: pl.Scalar[pl.INDEX] = kb * 256
                    x_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(hidden_states, [16, 256], [0, k0])
                    partial_sq: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq, pl.tensor.reshape(pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)), [1, 16]))
                variance: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq, 0.00019531250000000001), 9.9999999999999995e-07), [16, 1])
                inv_rms: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance))
                inv_rms_states: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.assemble(inv_rms_states, inv_rms, [0, 0])
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed") as q_seed_tid:
                for snb in pl.pipeline(10, stage=2):
                    q_proj: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj, pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0), [0, snb * 512])
            for q_nt in pl.parallel(10):
                q_n_region: pl.Scalar[pl.INDEX] = q_nt * 512
                for q_ks in pl.range(5):
                    q_k_base: pl.Scalar[pl.INDEX] = q_ks * 1024
                    _submit_deps_buf_6: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                    _submit_deps_buf_7: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_6, 0, pl.array.get_element(prev_normed_tids, q_ks))
                    _submit_deps_buf_8: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_7, 1, q_seed_tid)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj", deps=[_submit_deps_buf_8]) as q_tid:
                        for n_sub in pl.range(2):
                            n0: pl.Scalar[pl.INDEX] = q_n_region + n_sub * 256
                            q_acc: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed_in, [16, 256], [0, q_k_base]), pl.tensor.slice(wq, [256, 256], [layer_hidden_base + q_k_base, n0]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for kc in pl.pipeline(1, 4, stage=2):
                                kk: pl.Scalar[pl.INDEX] = q_k_base + kc * 256
                                q_acc: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(q_acc, pl.tensor.slice(normed_in, [16, 256], [0, kk]), pl.tensor.slice(wq, [256, 256], [layer_hidden_base + kk, n0]), a_trans=False, b_trans=False)
                            q_proj: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj, q_acc, [0, n0], atomic=pl.AtomicType.Add)
                    q_tile_tids: pl.Array[50, pl.TASK_ID] = pl.array.update_element(q_tile_tids, q_nt * 5 + q_ks, q_tid)
            _submit_deps_buf_9: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_10: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_9, 0, pl.array.get_element(prev_out_tids, 0))
            _submit_deps_buf_11: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_10, 1, pl.array.get_element(prev_out_tids, 1))
            _submit_deps_buf_12: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_11, 2, pl.array.get_element(prev_out_tids, 2))
            _submit_deps_buf_13: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_12, 3, pl.array.get_element(prev_out_tids, 3))
            _submit_deps_buf_14: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_13, 4, pl.array.get_element(prev_out_tids, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_seed", deps=[_submit_deps_buf_14]) as k_seed_tid:
                k_proj: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj, pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0), [0, 0])
            for k_nt in pl.parallel(2):
                k_n_region: pl.Scalar[pl.INDEX] = k_nt * 512
                for k_ks in pl.range(5):
                    k_k_base: pl.Scalar[pl.INDEX] = k_ks * 1024
                    _submit_deps_buf_15: pl.Array[6, pl.TASK_ID] = pl.array.create(6, dtype=pl.TASK_ID)
                    _submit_deps_buf_16: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_15, 0, pl.array.get_element(prev_normed_tids, 0))
                    _submit_deps_buf_17: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_16, 1, pl.array.get_element(prev_normed_tids, 1))
                    _submit_deps_buf_18: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_17, 2, pl.array.get_element(prev_normed_tids, 2))
                    _submit_deps_buf_19: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_18, 3, pl.array.get_element(prev_normed_tids, 3))
                    _submit_deps_buf_20: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_19, 4, pl.array.get_element(prev_normed_tids, 4))
                    _submit_deps_buf_21: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_20, 5, k_seed_tid)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj", deps=[_submit_deps_buf_21]) as k_tid:
                        for n_sub_1 in pl.range(2):
                            n0: pl.Scalar[pl.INDEX] = k_n_region + n_sub_1 * 256
                            k_acc: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed_in, [16, 256], [0, k_k_base]), pl.tensor.slice(wk, [256, 256], [layer_hidden_base + k_k_base, n0]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for kc_1 in pl.pipeline(1, 4, stage=2):
                                kk: pl.Scalar[pl.INDEX] = k_k_base + kc_1 * 256
                                k_acc: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(k_acc, pl.tensor.slice(normed_in, [16, 256], [0, kk]), pl.tensor.slice(wk, [256, 256], [layer_hidden_base + kk, n0]), a_trans=False, b_trans=False)
                            k_proj: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj, k_acc, [0, n0], atomic=pl.AtomicType.Add)
                    k_tile_tids: pl.Array[10, pl.TASK_ID] = pl.array.update_element(k_tile_tids, k_nt * 5 + k_ks, k_tid)
            _submit_deps_buf_22: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_23: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_22, 0, pl.array.get_element(prev_out_tids, 0))
            _submit_deps_buf_24: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_23, 1, pl.array.get_element(prev_out_tids, 1))
            _submit_deps_buf_25: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_24, 2, pl.array.get_element(prev_out_tids, 2))
            _submit_deps_buf_26: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_25, 3, pl.array.get_element(prev_out_tids, 3))
            _submit_deps_buf_27: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_26, 4, pl.array.get_element(prev_out_tids, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_seed", deps=[_submit_deps_buf_27]) as v_seed_tid:
                v_proj: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(v_proj, pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0), [0, 0])
            for v_nt in pl.parallel(2):
                v_n_region: pl.Scalar[pl.INDEX] = v_nt * 512
                for v_ks in pl.range(5):
                    v_k_base: pl.Scalar[pl.INDEX] = v_ks * 1024
                    _submit_deps_buf_28: pl.Array[6, pl.TASK_ID] = pl.array.create(6, dtype=pl.TASK_ID)
                    _submit_deps_buf_29: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_28, 0, pl.array.get_element(prev_normed_tids, 0))
                    _submit_deps_buf_30: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_29, 1, pl.array.get_element(prev_normed_tids, 1))
                    _submit_deps_buf_31: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_30, 2, pl.array.get_element(prev_normed_tids, 2))
                    _submit_deps_buf_32: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_31, 3, pl.array.get_element(prev_normed_tids, 3))
                    _submit_deps_buf_33: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_32, 4, pl.array.get_element(prev_normed_tids, 4))
                    _submit_deps_buf_34: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_33, 5, v_seed_tid)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj", deps=[_submit_deps_buf_34]) as v_tid:
                        for n_sub_2 in pl.range(2):
                            n0: pl.Scalar[pl.INDEX] = v_n_region + n_sub_2 * 256
                            v_acc: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed_in, [16, 256], [0, v_k_base]), pl.tensor.slice(wv, [256, 256], [layer_hidden_base + v_k_base, n0]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for kc_2 in pl.pipeline(1, 4, stage=2):
                                kk: pl.Scalar[pl.INDEX] = v_k_base + kc_2 * 256
                                v_acc: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(v_acc, pl.tensor.slice(normed_in, [16, 256], [0, kk]), pl.tensor.slice(wv, [256, 256], [layer_hidden_base + kk, n0]), a_trans=False, b_trans=False)
                            v_proj: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(v_proj, v_acc, [0, n0], atomic=pl.AtomicType.Add)
                    v_tile_tids: pl.Array[10, pl.TASK_ID] = pl.array.update_element(v_tile_tids, v_nt * 5 + v_ks, v_tid)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_work_build") as work_tid:
                cursor: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(seq_lens, [0]), pl.INDEX) * 0
                for wb in pl.unroll(16):
                    wb_ctx: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens, [wb]), pl.INDEX) + 127) // 128
                    for wp in pl.range(wb_ctx):
                        pl.tensor.write(fa_work_table, [cursor + wp, 0], pl.cast(wb * 32 + wp, pl.INT32))
                    cursor: pl.Scalar[pl.INDEX] = cursor + wb_ctx
                pl.tensor.write(fa_total, [0, 0], pl.cast(cursor, pl.INT32))
            q_proj_norm: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            k_proj_norm: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_inv_states: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.create([640, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            k_inv_states: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.create([128, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            inv_rms_col: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(inv_rms_states, [16, 1], [0, 0])
            for h in pl.unroll(8):
                qt0: pl.Scalar[pl.INDEX] = h * 5 * 128 // 512
                kt: pl.Scalar[pl.INDEX] = h * 128 // 512
                _submit_deps_buf_35: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
                _submit_deps_buf_36: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_35, 0, pl.array.get_element(q_tile_tids, qt0 * 5 + 0))
                _submit_deps_buf_37: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_36, 1, pl.array.get_element(q_tile_tids, qt0 * 5 + 1))
                _submit_deps_buf_38: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_37, 2, pl.array.get_element(q_tile_tids, qt0 * 5 + 2))
                _submit_deps_buf_39: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_38, 3, pl.array.get_element(q_tile_tids, qt0 * 5 + 3))
                _submit_deps_buf_40: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_39, 4, pl.array.get_element(q_tile_tids, qt0 * 5 + 4))
                _submit_deps_buf_41: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_40, 5, pl.array.get_element(q_tile_tids, qt0 * 5 + 5))
                _submit_deps_buf_42: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_41, 6, pl.array.get_element(q_tile_tids, qt0 * 5 + 6))
                _submit_deps_buf_43: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_42, 7, pl.array.get_element(q_tile_tids, qt0 * 5 + 7))
                _submit_deps_buf_44: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_43, 8, pl.array.get_element(q_tile_tids, qt0 * 5 + 8))
                _submit_deps_buf_45: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_44, 9, pl.array.get_element(q_tile_tids, qt0 * 5 + 9))
                _submit_deps_buf_46: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_45, 10, pl.array.get_element(k_tile_tids, kt * 5 + 0))
                _submit_deps_buf_47: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_46, 11, pl.array.get_element(k_tile_tids, kt * 5 + 1))
                _submit_deps_buf_48: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_47, 12, pl.array.get_element(k_tile_tids, kt * 5 + 2))
                _submit_deps_buf_49: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_48, 13, pl.array.get_element(k_tile_tids, kt * 5 + 3))
                _submit_deps_buf_50: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_49, 14, pl.array.get_element(k_tile_tids, kt * 5 + 4))
                _submit_deps_buf_51: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_50, 15, rms_tid)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_51]) as qk_tid_h:
                    q0: pl.Scalar[pl.INDEX] = h * 5 * 128
                    q_slice: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj, [16, 640], [0, q0]), inv_rms_col)
                    q_chunk: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice, [80, 128])
                    q_g: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk, q_norm_w)
                    q_proj_norm: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm, pl.tensor.reshape(q_g, [16, 640]), [0, q0])
                    q_ss: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk, q_chunk))
                    q_inv: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss, 0.0078125), 9.9999999999999995e-07)))
                    q_inv_states: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states, q_inv, [h * 16 * 5, 0])
                    k0_v1: pl.Scalar[pl.INDEX] = h * 128
                    k_chunk: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj, [16, 128], [0, k0_v1]), inv_rms_col)
                    k_g: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk, k_norm_w)
                    k_proj_norm: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm, k_g, [0, k0_v1])
                    k_ss: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk, k_chunk))
                    k_inv: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss, 0.0078125), 9.9999999999999995e-07)))
                    k_inv_states: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states, k_inv, [h * 16, 0])
                qk_tids: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids, h, qk_tid_h)
            _submit_deps_buf_52: pl.Array[19, pl.TASK_ID] = pl.array.create(19, dtype=pl.TASK_ID)
            _submit_deps_buf_53: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_52, 0, pl.array.get_element(qk_tids, 0))
            _submit_deps_buf_54: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_53, 1, pl.array.get_element(qk_tids, 1))
            _submit_deps_buf_55: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_54, 2, pl.array.get_element(qk_tids, 2))
            _submit_deps_buf_56: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_55, 3, pl.array.get_element(qk_tids, 3))
            _submit_deps_buf_57: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_56, 4, pl.array.get_element(qk_tids, 4))
            _submit_deps_buf_58: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_57, 5, pl.array.get_element(qk_tids, 5))
            _submit_deps_buf_59: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_58, 6, pl.array.get_element(qk_tids, 6))
            _submit_deps_buf_60: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_59, 7, pl.array.get_element(qk_tids, 7))
            _submit_deps_buf_61: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_60, 8, rms_tid)
            _submit_deps_buf_62: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_61, 9, pl.array.get_element(v_tile_tids, 0))
            _submit_deps_buf_63: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_62, 10, pl.array.get_element(v_tile_tids, 1))
            _submit_deps_buf_64: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_63, 11, pl.array.get_element(v_tile_tids, 2))
            _submit_deps_buf_65: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_64, 12, pl.array.get_element(v_tile_tids, 3))
            _submit_deps_buf_66: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_65, 13, pl.array.get_element(v_tile_tids, 4))
            _submit_deps_buf_67: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_66, 14, pl.array.get_element(v_tile_tids, 5))
            _submit_deps_buf_68: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_67, 15, pl.array.get_element(v_tile_tids, 6))
            _submit_deps_buf_69: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_68, 16, pl.array.get_element(v_tile_tids, 7))
            _submit_deps_buf_70: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_69, 17, pl.array.get_element(v_tile_tids, 8))
            _submit_deps_buf_71: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_70, 18, pl.array.get_element(v_tile_tids, 9))
            with pl.spmd(32, name_hint="rope_qkv_spmd", deps=[_submit_deps_buf_71]) as rope_tid:
                rope_core: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                for it in pl.pipeline(4, stage=2):
                    g_idx: pl.Scalar[pl.INDEX] = rope_core * 4 + it
                    ki: pl.Scalar[pl.INDEX] = g_idx // 16
                    b: pl.Scalar[pl.INDEX] = g_idx % 16
                    ctx_len: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [b])
                    inv_rms_b: pl.Scalar[pl.FP32] = pl.tensor.read(inv_rms_states, [b, 0])
                    pos: pl.Scalar[pl.INDEX] = pl.cast(ctx_len, pl.INDEX) - 1
                    wr_slot: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slot_mapping, [b]), pl.INDEX)
                    wr_slot_block: pl.Scalar[pl.INDEX] = wr_slot // 128
                    wr_slot_offset: pl.Scalar[pl.INDEX] = wr_slot - wr_slot_block * 128
                    cos_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos, [1, 64], [pos, 0])
                    cos_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos, [1, 64], [pos, 64])
                    sin_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin, [1, 64], [pos, 0])
                    sin_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin, [1, 64], [pos, 64])
                    kv_col: pl.Scalar[pl.INDEX] = ki * 128
                    k_inv_b: pl.Scalar[pl.FP32] = pl.tensor.read(k_inv_states, [ki * 16 + b, 0])
                    k_full: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(pl.tensor.slice(k_proj_norm, [1, 128], [b, kv_col]), k_inv_b)
                    k_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full, [1, 64], [0, 0])
                    k_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full, [1, 64], [0, 64])
                    rot_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.sub(pl.tensor.col_expand_mul(k_lo, cos_lo), pl.tensor.col_expand_mul(k_hi, sin_lo))
                    rot_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(k_hi, cos_hi), pl.tensor.col_expand_mul(k_lo, sin_hi))
                    cache_row: pl.Scalar[pl.INDEX] = layer_cache_base + (wr_slot_block * 8 + ki) * 128 + wr_slot_offset
                    k_cache: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache, pl.tensor.cast(rot_lo, target_type=pl.BF16, mode='round'), [cache_row, 0])
                    k_cache: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache, pl.tensor.cast(rot_hi, target_type=pl.BF16, mode='round'), [cache_row, 64])
                    v_row_bf16: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(pl.tensor.muls(pl.tensor.slice(v_proj, [1, 128], [b, ki * 128]), inv_rms_b), target_type=pl.BF16, mode='round')
                    v_cache: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(v_cache, v_row_bf16, [cache_row, 0])
                    q_base: pl.Scalar[pl.INDEX] = ki * 5
                    q_pad_row0: pl.Scalar[pl.INDEX] = b * 8 * 16 + ki * 16
                    q_inv_base: pl.Scalar[pl.INDEX] = ki * 16 * 5 + b * 5
                    for qj in pl.range(5):
                        q_inv_bj: pl.Scalar[pl.FP32] = pl.tensor.read(q_inv_states, [q_inv_base + qj, 0])
                        q_head: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(pl.tensor.slice(q_proj_norm, [1, 128], [b, (q_base + qj) * 128]), q_inv_bj)
                        q_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_head, [1, 64], [0, 0])
                        q_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_head, [1, 64], [0, 64])
                        q_rot_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.sub(pl.tensor.col_expand_mul(q_lo, cos_lo), pl.tensor.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(q_hi, cos_hi), pl.tensor.col_expand_mul(q_lo, sin_hi))
                        all_q_padded: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded, pl.tensor.cast(q_rot_lo, target_type=pl.BF16, mode='round'), [q_pad_row0 + qj, 0])
                        all_q_padded: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded, pl.tensor.cast(q_rot_hi, target_type=pl.BF16, mode='round'), [q_pad_row0 + qj, 64])
                    q_pad_zero: pl.Tensor[[11, 128], pl.BF16] = pl.tensor.cast(pl.tensor.full([11, 128], dtype=pl.FP32, value=0.0), target_type=pl.BF16, mode='round')
                    all_q_padded: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded, q_pad_zero, [q_pad_row0 + 5, 0])
            rope_grp_tids: pl.Array[2, pl.TASK_ID] = pl.array.update_element(rope_grp_tids, 0, rope_tid)
            down_acc_all: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            gate_acc_all: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.create([16, 17408], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            up_acc_all: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.create([16, 17408], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            _submit_deps_buf_72: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_73: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_72, 0, pl.array.get_element(prev_out_tids, 0))
            _submit_deps_buf_74: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_73, 1, pl.array.get_element(prev_out_tids, 1))
            _submit_deps_buf_75: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_74, 2, pl.array.get_element(prev_out_tids, 2))
            _submit_deps_buf_76: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_75, 3, pl.array.get_element(prev_out_tids, 3))
            _submit_deps_buf_77: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_76, 4, pl.array.get_element(prev_out_tids, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_seed", deps=[_submit_deps_buf_77]) as seed_tid:
                for nb in pl.pipeline(5, stage=2):
                    n0: pl.Scalar[pl.INDEX] = nb * 1024
                    zero: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                    down_acc_all: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(down_acc_all, zero, [0, n0])
            _submit_deps_buf_78: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_79: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_78, 0, pl.array.get_element(prev_out_tids, 0))
            _submit_deps_buf_80: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_79, 1, pl.array.get_element(prev_out_tids, 1))
            _submit_deps_buf_81: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_80, 2, pl.array.get_element(prev_out_tids, 2))
            _submit_deps_buf_82: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_81, 3, pl.array.get_element(prev_out_tids, 3))
            _submit_deps_buf_83: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_82, 4, pl.array.get_element(prev_out_tids, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_seed", deps=[_submit_deps_buf_83]) as gate_seed_tid:
                for nb_1 in pl.pipeline(17, stage=2):
                    n0: pl.Scalar[pl.INDEX] = nb_1 * 1024
                    zero: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                    gate_acc_all: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(gate_acc_all, zero, [0, n0])
            _submit_deps_buf_84: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_85: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_84, 0, pl.array.get_element(prev_out_tids, 0))
            _submit_deps_buf_86: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_85, 1, pl.array.get_element(prev_out_tids, 1))
            _submit_deps_buf_87: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_86, 2, pl.array.get_element(prev_out_tids, 2))
            _submit_deps_buf_88: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_87, 3, pl.array.get_element(prev_out_tids, 3))
            _submit_deps_buf_89: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_88, 4, pl.array.get_element(prev_out_tids, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_seed", deps=[_submit_deps_buf_89]) as up_seed_tid:
                for nb_2 in pl.pipeline(17, stage=2):
                    n0: pl.Scalar[pl.INDEX] = nb_2 * 1024
                    zero: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                    up_acc_all: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(up_acc_all, zero, [0, n0])
            _submit_deps_buf_90: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
            _submit_deps_buf_91: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_90, 0, work_tid)
            _submit_deps_buf_92: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_91, 1, pl.array.get_element(rope_grp_tids, 0))
            with pl.spmd(24, name_hint="fa_fused_spmd", optimizations=[pl.split(pl.SplitMode.UP_DOWN)], deps=[_submit_deps_buf_92]) as fa_tid:
                fa_core: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                fa_total_blocks: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(fa_total, [0, 0]), pl.INDEX)
                for fa_w in pl.range(fa_core, fa_total_blocks, 24):
                    fa_enc: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(fa_work_table, [fa_w, 0]), pl.INDEX)
                    fa_b: pl.Scalar[pl.INDEX] = fa_enc // 32
                    fa_p: pl.Scalar[pl.INDEX] = fa_enc % 32
                    fa_hg: pl.Scalar[pl.INDEX] = 0
                    fa_ctx_len: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [fa_b])
                    sb: pl.Scalar[pl.INDEX] = fa_p
                    s0: pl.Scalar[pl.INDEX] = sb * 128
                    valid_len: pl.Scalar[pl.INDEX] = pl.min(128, pl.cast(fa_ctx_len, pl.INDEX) - s0)
                    fa_pbid: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)
                    for gp in pl.pipeline(8, stage=2):
                        gi: pl.Scalar[pl.INDEX] = fa_hg * 8 + gp
                        kvh: pl.Scalar[pl.INDEX] = gi
                        q_pad_row_g: pl.Scalar[pl.INDEX] = fa_b * 8 * 16 + gi * 16
                        q_padded: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.slice(all_q_padded, [16, 128], [q_pad_row_g, 0])
                        g_base: pl.Scalar[pl.INDEX] = (fa_b * 8 + gi) * 32 * 16
                        cache_row: pl.Scalar[pl.INDEX] = layer_cache_base + (fa_pbid * 8 + kvh) * 128
                        k_tile: pl.Tensor[[128, 128], pl.BF16] = pl.tensor.slice(k_cache, [128, 128], [cache_row, 0])
                        raw_scores: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(q_padded, k_tile, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        scores_scaled: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.muls(raw_scores, 0.088388347648318433)
                        scores_valid: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, valid_len], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(scores_scaled, 5, valid_len)
                        scores: pl.Tensor[[16, 128], pl.FP32, pl.TensorView()] = pl.tensor.fillpad(scores_valid, pad_value=pl.PadValue.min)
                        cur_mi: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_max(scores)
                        exp_scores: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(scores, cur_mi))
                        cur_li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(exp_scores)
                        exp_scores_bf16: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.cast(exp_scores, target_type=pl.BF16, mode='round')
                        v_tile: pl.Tensor[[128, 128], pl.BF16] = pl.tensor.slice(v_cache, [128, 128], [cache_row, 0])
                        oi_tmp: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(exp_scores_bf16, v_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        oi_tmp_v1: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(oi_tmp, 5, 128)
                        all_oi_tmp: pl.Tensor[[65536, 128], pl.FP32] = pl.tensor.assemble(all_oi_tmp, oi_tmp_v1, [g_base + sb * 16, 0])
                        all_cur_mi: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.assemble(all_cur_mi, cur_mi, [g_base + sb * 16, 0])
                        all_cur_li: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.assemble(all_cur_li, cur_li, [g_base + sb * 16, 0])
            with pl.spmd(48, name_hint="online_softmax_spmd", deps=[fa_tid]) as attn_done_tid:
                os_core: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                for os_spmd_idx in pl.range(os_core, 128, 48):
                    os_b: pl.Scalar[pl.INDEX] = os_spmd_idx // 8
                    os_gi: pl.Scalar[pl.INDEX] = os_spmd_idx % 8
                    os_ctx_len: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [os_b])
                    os_ctx_blocks: pl.Scalar[pl.INDEX] = (pl.cast(os_ctx_len, pl.INDEX) + 128 - 1) // 128
                    os_kvh: pl.Scalar[pl.INDEX] = os_gi
                    os_q_base: pl.Scalar[pl.INDEX] = os_kvh * 5
                    os_g_base: pl.Scalar[pl.INDEX] = (os_b * 8 + os_gi) * 32 * 16
                    oi: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.slice(all_oi_tmp, [16, 128], [os_g_base, 0])
                    mi: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(all_cur_mi, [16, 1], [os_g_base, 0])
                    li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(all_cur_li, [16, 1], [os_g_base, 0])
                    for sb_1 in pl.pipeline(1, os_ctx_blocks, stage=2):
                        rec: pl.Scalar[pl.INDEX] = os_g_base + sb_1 * 16
                        oi_tmp_valid: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_oi_tmp, [16, 128], [rec, 0], [5, 128])
                        online_cur_mi: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[5, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_cur_mi, [16, 1], [rec, 0], [5, 1])
                        online_cur_li: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[5, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_cur_li, [16, 1], [rec, 0], [5, 1])
                        mi_new: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.maximum(mi, online_cur_mi)
                        alpha: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi, mi_new))
                        beta: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(online_cur_mi, mi_new))
                        li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha, li), pl.tensor.mul(beta, online_cur_li))
                        oi: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(oi, alpha), pl.tensor.row_expand_mul(oi_tmp_valid, beta))
                        mi: pl.Tensor[[16, 1], pl.FP32] = mi_new
                    ctx: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_div(oi, li)
                    ctx_valid: pl.Tensor[[5, 128], pl.FP32] = pl.tensor.slice(ctx, [5, 128], [0, 0])
                    ctx_flat_bf16: pl.Tensor[[1, 640], pl.BF16] = pl.tensor.cast(pl.tensor.reshape(ctx_valid, [1, 640]), target_type=pl.BF16, mode='round')
                    attn_out: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(attn_out, ctx_flat_bf16, [os_b, os_q_base * 128])
            attn_proj_fp32: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            post_norm_partial: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            mlp_norm_in: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            inv_rms_tile: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            mlp_tile: pl.Tensor[[16, 17408], pl.BF16] = pl.tensor.create([16, 17408], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            _submit_deps_buf_93: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_94: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_93, 0, pl.array.get_element(prev_out_tids, 0))
            _submit_deps_buf_95: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_94, 1, pl.array.get_element(prev_out_tids, 1))
            _submit_deps_buf_96: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_95, 2, pl.array.get_element(prev_out_tids, 2))
            _submit_deps_buf_97: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_96, 3, pl.array.get_element(prev_out_tids, 3))
            _submit_deps_buf_98: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_97, 4, pl.array.get_element(prev_out_tids, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_seed", deps=[_submit_deps_buf_98]) as out_seed_tid:
                for nb_3 in pl.pipeline(10, stage=2):
                    out_seed_n0: pl.Scalar[pl.INDEX] = nb_3 * 512
                    out_zero: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                    attn_proj_fp32: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_proj_fp32, out_zero, [0, out_seed_n0])
            silu_tids: pl.Array[17, pl.TASK_ID] = pl.array.create(17, dtype=pl.TASK_ID)
            gate_tids: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
            up_tids: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
            cast_tids: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            out_tids: pl.Array[50, pl.TASK_ID] = pl.array.create(50, dtype=pl.TASK_ID)
            for n_out_proj in pl.parallel(10):
                n_op: pl.Scalar[pl.INDEX] = n_out_proj * 512
                for k_split_out in pl.range(5):
                    k_op: pl.Scalar[pl.INDEX] = k_split_out * 1024
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj", deps=[out_seed_tid, attn_done_tid]) as out_tid:
                        out_a0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(attn_out, [16, 64], [0, k_op])
                        out_w0: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wo, [64, 512], [layer_hidden_base + k_op, n_op])
                        out_c_acc: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.matmul(out_a0, out_w0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for out_lk in pl.pipeline(1, 16, stage=2):
                            out_ks_off: pl.Scalar[pl.INDEX] = out_lk * 64
                            out_a_k: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(attn_out, [16, 64], [0, k_op + out_ks_off])
                            out_w_k: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wo, [64, 512], [layer_hidden_base + k_op + out_ks_off, n_op])
                            out_c_acc: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.matmul_acc(out_c_acc, out_a_k, out_w_k, a_trans=False, b_trans=False)
                        attn_proj_fp32: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_proj_fp32, out_c_acc, [0, n_op], atomic=pl.AtomicType.Add)
                    out_tids: pl.Array[50, pl.TASK_ID] = pl.array.update_element(out_tids, n_out_proj * 5 + k_split_out, out_tid)
            for k_slice in pl.unroll(5):
                k_base: pl.Scalar[pl.INDEX] = k_slice * 1024
                n_split_base: pl.Scalar[pl.INDEX] = k_slice * 2
                _submit_deps_buf_99: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
                _submit_deps_buf_100: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_99, 0, pl.array.get_element(out_tids, (n_split_base + 0) * 5 + 0))
                _submit_deps_buf_101: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_100, 1, pl.array.get_element(out_tids, (n_split_base + 0) * 5 + 1))
                _submit_deps_buf_102: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_101, 2, pl.array.get_element(out_tids, (n_split_base + 0) * 5 + 2))
                _submit_deps_buf_103: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_102, 3, pl.array.get_element(out_tids, (n_split_base + 0) * 5 + 3))
                _submit_deps_buf_104: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_103, 4, pl.array.get_element(out_tids, (n_split_base + 0) * 5 + 4))
                _submit_deps_buf_105: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_104, 5, pl.array.get_element(out_tids, (n_split_base + 1) * 5 + 0))
                _submit_deps_buf_106: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_105, 6, pl.array.get_element(out_tids, (n_split_base + 1) * 5 + 1))
                _submit_deps_buf_107: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_106, 7, pl.array.get_element(out_tids, (n_split_base + 1) * 5 + 2))
                _submit_deps_buf_108: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_107, 8, pl.array.get_element(out_tids, (n_split_base + 1) * 5 + 3))
                _submit_deps_buf_109: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_108, 9, pl.array.get_element(out_tids, (n_split_base + 1) * 5 + 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_rms_cast", deps=[_submit_deps_buf_109]) as cast_tid_k:
                    for kb_1 in pl.pipeline(4, stage=2):
                        k0_v1: pl.Scalar[pl.INDEX] = k_base + kb_1 * 256
                        attn_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32, [16, 256], [0, k0_v1])
                        hidden_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(hidden_states, [16, 256], [0, k0_v1])
                        resid_fp32: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk, hidden_chunk)
                        post_norm_partial: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(post_norm_partial, resid_fp32, [0, k0_v1])
                        post_gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight, [1, 256], [layer_idx, k0_v1])
                        mlp_norm_in: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(mlp_norm_in, pl.tensor.cast(pl.tensor.col_expand_mul(resid_fp32, post_gamma), target_type=pl.BF16, mode='round'), [0, k0_v1])
                cast_tids: pl.Array[5, pl.TASK_ID] = pl.array.update_element(cast_tids, k_slice, cast_tid_k)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rms_reduce", deps=[out_tids]) as reduce_tid:
                sq_sum: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                for kb_2 in pl.pipeline(20, stage=2):
                    k0: pl.Scalar[pl.INDEX] = kb_2 * 256
                    attn_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32, [16, 256], [0, k0])
                    hidden_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(hidden_states, [16, 256], [0, k0])
                    resid_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk, hidden_chunk)
                    sq_sum: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.reshape(pl.tensor.row_sum(pl.tensor.mul(resid_chunk, resid_chunk)), [1, 16]))
                post_inv_rms: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(sq_sum, 0.00019531250000000001), 9.9999999999999995e-07)))
                post_inv_rms_col: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(post_inv_rms, [16, 1])
                inv_rms_tile: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.assemble(inv_rms_tile, post_inv_rms_col, [0, 0])
            for n_out in pl.parallel(17):
                n0: pl.Scalar[pl.INDEX] = n_out * 1024
                for k_split in pl.range(5):
                    k0: pl.Scalar[pl.INDEX] = k_split * 1024
                    _submit_deps_buf_110: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                    _submit_deps_buf_111: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_110, 0, pl.array.get_element(cast_tids, k_split))
                    _submit_deps_buf_112: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_111, 1, gate_seed_tid)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj", deps=[_submit_deps_buf_112]) as gate_tid:
                        a0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in, [16, 64], [0, k0])
                        w0: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_gate, [64, 1024], [layer_hidden_base + k0, n0])
                        c_acc: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0, w0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for lk in pl.pipeline(1, 16, stage=2):
                            ks_off: pl.Scalar[pl.INDEX] = lk * 64
                            a_k: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in, [16, 64], [0, k0 + ks_off])
                            w_k: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_gate, [64, 1024], [layer_hidden_base + k0 + ks_off, n0])
                            c_acc: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc, a_k, w_k, a_trans=False, b_trans=False)
                        gate_acc_all: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(gate_acc_all, c_acc, [0, n0], atomic=pl.AtomicType.Add)
                    gate_tids: pl.Array[85, pl.TASK_ID] = pl.array.update_element(gate_tids, n_out * 5 + k_split, gate_tid)
                    _submit_deps_buf_113: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                    _submit_deps_buf_114: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_113, 0, pl.array.get_element(cast_tids, k_split))
                    _submit_deps_buf_115: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_114, 1, up_seed_tid)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj", deps=[_submit_deps_buf_115]) as up_tid:
                        a0_v1: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in, [16, 64], [0, k0])
                        w0_v1: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_up, [64, 1024], [layer_hidden_base + k0, n0])
                        c_acc_v1: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0_v1, w0_v1, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for lk_1 in pl.pipeline(1, 16, stage=2):
                            ks_off: pl.Scalar[pl.INDEX] = lk_1 * 64
                            a_k: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in, [16, 64], [0, k0 + ks_off])
                            w_k: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_up, [64, 1024], [layer_hidden_base + k0 + ks_off, n0])
                            c_acc_v1: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc_v1, a_k, w_k, a_trans=False, b_trans=False)
                        up_acc_all: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(up_acc_all, c_acc_v1, [0, n0], atomic=pl.AtomicType.Add)
                    up_tids: pl.Array[85, pl.TASK_ID] = pl.array.update_element(up_tids, n_out * 5 + k_split, up_tid)
            for n_out_1 in pl.parallel(17):
                n0: pl.Scalar[pl.INDEX] = n_out_1 * 1024
                _submit_deps_buf_116: pl.Array[11, pl.TASK_ID] = pl.array.create(11, dtype=pl.TASK_ID)
                _submit_deps_buf_117: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_116, 0, reduce_tid)
                _submit_deps_buf_118: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_117, 1, pl.array.get_element(gate_tids, n_out_1 * 5 + 0))
                _submit_deps_buf_119: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_118, 2, pl.array.get_element(gate_tids, n_out_1 * 5 + 1))
                _submit_deps_buf_120: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_119, 3, pl.array.get_element(gate_tids, n_out_1 * 5 + 2))
                _submit_deps_buf_121: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_120, 4, pl.array.get_element(gate_tids, n_out_1 * 5 + 3))
                _submit_deps_buf_122: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_121, 5, pl.array.get_element(gate_tids, n_out_1 * 5 + 4))
                _submit_deps_buf_123: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_122, 6, pl.array.get_element(up_tids, n_out_1 * 5 + 0))
                _submit_deps_buf_124: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_123, 7, pl.array.get_element(up_tids, n_out_1 * 5 + 1))
                _submit_deps_buf_125: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_124, 8, pl.array.get_element(up_tids, n_out_1 * 5 + 2))
                _submit_deps_buf_126: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_125, 9, pl.array.get_element(up_tids, n_out_1 * 5 + 3))
                _submit_deps_buf_127: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_126, 10, pl.array.get_element(up_tids, n_out_1 * 5 + 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu", deps=[_submit_deps_buf_127]) as silu_tid:
                    inv_rms_chunk: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(inv_rms_tile, [16, 1], [0, 0])
                    for sub in pl.pipeline(4, stage=2):
                        silu_off: pl.Scalar[pl.INDEX] = n0 + sub * 256
                        gate_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(gate_acc_all, [16, 256], [0, silu_off])
                        up_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(up_acc_all, [16, 256], [0, silu_off])
                        scaled_gate: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.row_expand_mul(gate_chunk, inv_rms_chunk)
                        scaled_up: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.row_expand_mul(up_chunk, inv_rms_chunk)
                        sigmoid: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(scaled_gate)), 1.0))
                        mlp_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.mul(pl.tensor.mul(scaled_gate, sigmoid), scaled_up)
                        mlp_tile: pl.Tensor[[16, 17408], pl.BF16] = pl.tensor.assemble(mlp_tile, pl.tensor.cast(mlp_chunk, target_type=pl.BF16, mode='round'), [0, silu_off])
                silu_tids: pl.Array[17, pl.TASK_ID] = pl.array.update_element(silu_tids, n_out_1, silu_tid)
            for n_out_2 in pl.parallel(5):
                n0: pl.Scalar[pl.INDEX] = n_out_2 * 1024
                for k_split_1 in pl.range(17):
                    k0: pl.Scalar[pl.INDEX] = k_split_1 * 1024
                    _submit_deps_buf_128: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                    _submit_deps_buf_129: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_128, 0, seed_tid)
                    _submit_deps_buf_130: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_129, 1, pl.array.get_element(silu_tids, k_split_1))
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj", deps=[_submit_deps_buf_130]) as down_tid:
                        a0_v2: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_tile, [16, 64], [0, k0])
                        w0_v2: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_down, [64, 1024], [layer_inter_base + k0, n0])
                        c_acc_v2: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0_v2, w0_v2, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for lk_2 in pl.pipeline(1, 16, stage=2):
                            ks_off: pl.Scalar[pl.INDEX] = lk_2 * 64
                            a_k: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_tile, [16, 64], [0, k0 + ks_off])
                            w_k: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_down, [64, 1024], [layer_inter_base + k0 + ks_off, n0])
                            c_acc_v2: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc_v2, a_k, w_k, a_trans=False, b_trans=False)
                        down_acc_all: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(down_acc_all, c_acc_v2, [0, n0], atomic=pl.AtomicType.Add)
                    down_tids: pl.Array[85, pl.TASK_ID] = pl.array.update_element(down_tids, n_out_2 * 17 + k_split_1, down_tid)
        _submit_deps_buf_131: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
        _submit_deps_buf_132: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_131, 0, pl.array.get_element(down_tids, 0))
        _submit_deps_buf_133: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_132, 1, pl.array.get_element(down_tids, 1))
        _submit_deps_buf_134: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_133, 2, pl.array.get_element(down_tids, 2))
        _submit_deps_buf_135: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_134, 3, pl.array.get_element(down_tids, 3))
        _submit_deps_buf_136: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_135, 4, pl.array.get_element(down_tids, 4))
        _submit_deps_buf_137: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_136, 5, pl.array.get_element(down_tids, 5))
        _submit_deps_buf_138: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_137, 6, pl.array.get_element(down_tids, 6))
        _submit_deps_buf_139: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_138, 7, pl.array.get_element(down_tids, 7))
        _submit_deps_buf_140: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_139, 8, pl.array.get_element(down_tids, 8))
        _submit_deps_buf_141: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_140, 9, pl.array.get_element(down_tids, 9))
        _submit_deps_buf_142: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_141, 10, pl.array.get_element(down_tids, 10))
        _submit_deps_buf_143: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_142, 11, pl.array.get_element(down_tids, 11))
        _submit_deps_buf_144: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_143, 12, pl.array.get_element(down_tids, 12))
        _submit_deps_buf_145: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_144, 13, pl.array.get_element(down_tids, 13))
        _submit_deps_buf_146: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_145, 14, pl.array.get_element(down_tids, 14))
        _submit_deps_buf_147: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_146, 15, pl.array.get_element(down_tids, 15))
        _submit_deps_buf_148: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_147, 16, pl.array.get_element(down_tids, 16))
        _submit_deps_buf_149: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_148, 17, pl.array.get_element(down_tids, 17))
        _submit_deps_buf_150: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_149, 18, pl.array.get_element(down_tids, 18))
        _submit_deps_buf_151: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_150, 19, pl.array.get_element(down_tids, 19))
        _submit_deps_buf_152: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_151, 20, pl.array.get_element(down_tids, 20))
        _submit_deps_buf_153: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_152, 21, pl.array.get_element(down_tids, 21))
        _submit_deps_buf_154: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_153, 22, pl.array.get_element(down_tids, 22))
        _submit_deps_buf_155: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_154, 23, pl.array.get_element(down_tids, 23))
        _submit_deps_buf_156: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_155, 24, pl.array.get_element(down_tids, 24))
        _submit_deps_buf_157: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_156, 25, pl.array.get_element(down_tids, 25))
        _submit_deps_buf_158: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_157, 26, pl.array.get_element(down_tids, 26))
        _submit_deps_buf_159: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_158, 27, pl.array.get_element(down_tids, 27))
        _submit_deps_buf_160: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_159, 28, pl.array.get_element(down_tids, 28))
        _submit_deps_buf_161: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_160, 29, pl.array.get_element(down_tids, 29))
        _submit_deps_buf_162: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_161, 30, pl.array.get_element(down_tids, 30))
        _submit_deps_buf_163: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_162, 31, pl.array.get_element(down_tids, 31))
        _submit_deps_buf_164: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_163, 32, pl.array.get_element(down_tids, 32))
        _submit_deps_buf_165: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_164, 33, pl.array.get_element(down_tids, 33))
        _submit_deps_buf_166: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_165, 34, pl.array.get_element(down_tids, 34))
        _submit_deps_buf_167: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_166, 35, pl.array.get_element(down_tids, 35))
        _submit_deps_buf_168: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_167, 36, pl.array.get_element(down_tids, 36))
        _submit_deps_buf_169: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_168, 37, pl.array.get_element(down_tids, 37))
        _submit_deps_buf_170: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_169, 38, pl.array.get_element(down_tids, 38))
        _submit_deps_buf_171: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_170, 39, pl.array.get_element(down_tids, 39))
        _submit_deps_buf_172: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_171, 40, pl.array.get_element(down_tids, 40))
        _submit_deps_buf_173: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_172, 41, pl.array.get_element(down_tids, 41))
        _submit_deps_buf_174: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_173, 42, pl.array.get_element(down_tids, 42))
        _submit_deps_buf_175: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_174, 43, pl.array.get_element(down_tids, 43))
        _submit_deps_buf_176: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_175, 44, pl.array.get_element(down_tids, 44))
        _submit_deps_buf_177: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_176, 45, pl.array.get_element(down_tids, 45))
        _submit_deps_buf_178: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_177, 46, pl.array.get_element(down_tids, 46))
        _submit_deps_buf_179: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_178, 47, pl.array.get_element(down_tids, 47))
        _submit_deps_buf_180: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_179, 48, pl.array.get_element(down_tids, 48))
        _submit_deps_buf_181: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_180, 49, pl.array.get_element(down_tids, 49))
        _submit_deps_buf_182: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_181, 50, pl.array.get_element(down_tids, 50))
        _submit_deps_buf_183: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_182, 51, pl.array.get_element(down_tids, 51))
        _submit_deps_buf_184: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_183, 52, pl.array.get_element(down_tids, 52))
        _submit_deps_buf_185: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_184, 53, pl.array.get_element(down_tids, 53))
        _submit_deps_buf_186: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_185, 54, pl.array.get_element(down_tids, 54))
        _submit_deps_buf_187: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_186, 55, pl.array.get_element(down_tids, 55))
        _submit_deps_buf_188: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_187, 56, pl.array.get_element(down_tids, 56))
        _submit_deps_buf_189: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_188, 57, pl.array.get_element(down_tids, 57))
        _submit_deps_buf_190: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_189, 58, pl.array.get_element(down_tids, 58))
        _submit_deps_buf_191: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_190, 59, pl.array.get_element(down_tids, 59))
        _submit_deps_buf_192: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_191, 60, pl.array.get_element(down_tids, 60))
        _submit_deps_buf_193: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_192, 61, pl.array.get_element(down_tids, 61))
        _submit_deps_buf_194: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_193, 62, pl.array.get_element(down_tids, 62))
        _submit_deps_buf_195: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_194, 63, pl.array.get_element(down_tids, 63))
        _submit_deps_buf_196: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_195, 64, pl.array.get_element(down_tids, 64))
        _submit_deps_buf_197: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_196, 65, pl.array.get_element(down_tids, 65))
        _submit_deps_buf_198: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_197, 66, pl.array.get_element(down_tids, 66))
        _submit_deps_buf_199: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_198, 67, pl.array.get_element(down_tids, 67))
        _submit_deps_buf_200: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_199, 68, pl.array.get_element(down_tids, 68))
        _submit_deps_buf_201: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_200, 69, pl.array.get_element(down_tids, 69))
        _submit_deps_buf_202: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_201, 70, pl.array.get_element(down_tids, 70))
        _submit_deps_buf_203: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_202, 71, pl.array.get_element(down_tids, 71))
        _submit_deps_buf_204: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_203, 72, pl.array.get_element(down_tids, 72))
        _submit_deps_buf_205: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_204, 73, pl.array.get_element(down_tids, 73))
        _submit_deps_buf_206: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_205, 74, pl.array.get_element(down_tids, 74))
        _submit_deps_buf_207: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_206, 75, pl.array.get_element(down_tids, 75))
        _submit_deps_buf_208: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_207, 76, pl.array.get_element(down_tids, 76))
        _submit_deps_buf_209: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_208, 77, pl.array.get_element(down_tids, 77))
        _submit_deps_buf_210: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_209, 78, pl.array.get_element(down_tids, 78))
        _submit_deps_buf_211: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_210, 79, pl.array.get_element(down_tids, 79))
        _submit_deps_buf_212: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_211, 80, pl.array.get_element(down_tids, 80))
        _submit_deps_buf_213: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_212, 81, pl.array.get_element(down_tids, 81))
        _submit_deps_buf_214: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_213, 82, pl.array.get_element(down_tids, 82))
        _submit_deps_buf_215: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_214, 83, pl.array.get_element(down_tids, 83))
        _submit_deps_buf_216: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_215, 84, pl.array.get_element(down_tids, 84))
        with pl.spmd(5, name_hint="dcr_xgamma_spmd", deps=[_submit_deps_buf_216]) as dcr_tid:
            n_out_2: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            n0: pl.Scalar[pl.INDEX] = n_out_2 * 1024
            out_chunk: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.add(pl.tensor.slice(down_acc_all, [16, 1024], [0, n0]), pl.tensor.slice(post_norm_partial, [16, 1024], [0, n0]))
            out: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(out, out_chunk, [0, n0])
            gamma_next: pl.Tensor[[1, 1024], pl.FP32] = pl.tensor.slice(input_rms_weight, [1, 1024], [next_gamma_idx, n0])
            xg: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.col_expand_mul(out_chunk, gamma_next)
            normed_out: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(normed_out, pl.tensor.cast(xg, target_type=pl.BF16, mode='round'), [0, n0])
        for _slab in pl.unroll(5):
            prev_out_tids: pl.Array[5, pl.TASK_ID] = pl.array.update_element(prev_out_tids, _slab, dcr_tid)
            prev_normed_tids: pl.Array[5, pl.TASK_ID] = pl.array.update_element(prev_normed_tids, _slab, dcr_tid)
        return out
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def decode_fwd_layers(self, hidden_states: pl.Tensor[[16, 5120], pl.BF16], input_rms_weight: pl.Tensor[[1, 5120], pl.FP32], wq: pl.Tensor[[5120, 5120], pl.BF16], wk: pl.Tensor[[5120, 1024], pl.BF16], wv: pl.Tensor[[5120, 1024], pl.BF16], q_norm_weight: pl.Tensor[[1, 128], pl.FP32], k_norm_weight: pl.Tensor[[1, 128], pl.FP32], seq_lens: pl.Tensor[[16], pl.INT32], block_table: pl.Tensor[[512], pl.INT32], slot_mapping: pl.Tensor[[16], pl.INT32], rope_cos: pl.Tensor[[4096, 128], pl.FP32], rope_sin: pl.Tensor[[4096, 128], pl.FP32], k_cache: pl.Tensor[[524288, 128], pl.BF16], v_cache: pl.Tensor[[524288, 128], pl.BF16], wo: pl.Tensor[[5120, 5120], pl.BF16], w_gate: pl.Tensor[[5120, 17408], pl.BF16], w_up: pl.Tensor[[5120, 17408], pl.BF16], w_down: pl.Tensor[[17408, 5120], pl.BF16], post_rms_weight: pl.Tensor[[1, 5120], pl.FP32], out: pl.Out[pl.Tensor[[16, 5120], pl.BF16]]) -> pl.Tensor[[16, 5120], pl.BF16]:
        cur: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        carry_tids: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
        for cb0 in pl.parallel(0, 16, 16):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden") as ch_tid:
                for ckb in pl.range(20):
                    ck0: pl.Scalar[pl.INDEX] = ckb * 256
                    cur: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(cur, pl.tensor.cast(pl.tensor.slice(hidden_states, [16, 256], [cb0, ck0]), target_type=pl.FP32, mode='round'), [cb0, ck0])
            for cseed in pl.range(5):
                carry_tids: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_tids, cseed, ch_tid)
        normed: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        carry_normed_tids: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            for xg_n in pl.range(5):
                xg_k0: pl.Scalar[pl.INDEX] = xg_n * 1024
                _submit_deps_buf: pl.Array[1, pl.TASK_ID] = pl.array.create(1, dtype=pl.TASK_ID)
                _submit_deps_buf_1: pl.Array[1, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf, 0, pl.array.get_element(carry_tids, xg_n))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_gamma0", deps=[_submit_deps_buf_1]) as xg0_tid:
                    for kb in pl.pipeline(4, stage=2):
                        k0: pl.Scalar[pl.INDEX] = xg_k0 + kb * 256
                        x_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur, [16, 256], [0, k0])
                        gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight, [1, 256], [0, k0])
                        xg: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.col_expand_mul(x_chunk, gamma)
                        normed: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(normed, pl.tensor.cast(xg, target_type=pl.BF16, mode='round'), [0, k0])
                carry_normed_tids: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_normed_tids, xg_n, xg0_tid)
        for i in pl.range(1):
            next_hidden: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            next_normed: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            next_gamma_idx: pl.Scalar[pl.INDEX] = pl.min(i + 1, 0)
            cur: pl.Tensor[[16, 5120], pl.FP32] = self._decode_layer(cur, input_rms_weight, wq, wk, wv, q_norm_weight, k_norm_weight, seq_lens, block_table, slot_mapping, rope_cos, rope_sin, k_cache, v_cache, wo, w_gate, w_up, w_down, post_rms_weight, next_hidden, normed, next_normed, i, next_gamma_idx, carry_tids, carry_normed_tids)
            normed: pl.Tensor[[16, 5120], pl.BF16] = next_normed
        for ob0 in pl.parallel(0, 16, 16):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_out"):
                for okb in pl.range(20):
                    ok0: pl.Scalar[pl.INDEX] = okb * 256
                    out: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(out, pl.tensor.cast(pl.tensor.slice(cur, [16, 256], [ob0, ok0]), target_type=pl.BF16, mode='round'), [ob0, ok0])
        return out
