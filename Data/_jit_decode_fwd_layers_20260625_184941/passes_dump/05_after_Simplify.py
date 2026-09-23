# pypto.program: _jit_decode_fwd_layers
import pypto.language as pl

valid_len = pl.dynamic("valid_len")

@pl.program
class _jit_decode_fwd_layers:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def decode_fwd_layers(self, hidden_states__ssa_v0: pl.Tensor[[16, 5120], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], wq__ssa_v0: pl.Tensor[[5120, 5120], pl.BF16], wk__ssa_v0: pl.Tensor[[5120, 1024], pl.BF16], wv__ssa_v0: pl.Tensor[[5120, 1024], pl.BF16], q_norm_weight__ssa_v0: pl.Tensor[[1, 128], pl.FP32], k_norm_weight__ssa_v0: pl.Tensor[[1, 128], pl.FP32], seq_lens__ssa_v0: pl.Tensor[[16], pl.INT32], block_table__ssa_v0: pl.Tensor[[512], pl.INT32], slot_mapping__ssa_v0: pl.Tensor[[16], pl.INT32], rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32], rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32], k_cache__ssa_v0: pl.Tensor[[524288, 128], pl.BF16], v_cache__ssa_v0: pl.Tensor[[524288, 128], pl.BF16], wo__ssa_v0: pl.Tensor[[5120, 5120], pl.BF16], w_gate__ssa_v0: pl.Tensor[[5120, 17408], pl.BF16], w_up__ssa_v0: pl.Tensor[[5120, 17408], pl.BF16], w_down__ssa_v0: pl.Tensor[[17408, 5120], pl.BF16], post_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], out__ssa_v0: pl.Out[pl.Tensor[[16, 5120], pl.BF16]]) -> pl.Tensor[[16, 5120], pl.BF16]:
        cur__ssa_v0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        carry_tids__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
        for cb0__idx_v0, (carry_tids__iter_v1, cur__iter_v1) in pl.parallel(0, 16, 16, init_values=(carry_tids__ssa_v0, cur__ssa_v0)):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden") as ch_tid__ssa_v0:
                for ckb__idx_v0, (cur__iter_v3,) in pl.range(20, init_values=(cur__iter_v1,)):
                    ck0__ssa_v0: pl.Scalar[pl.INDEX] = ckb__idx_v0 * 256
                    cur__ssa_v5: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(cur__iter_v3, pl.tensor.cast(pl.tensor.slice(hidden_states__ssa_v0, [16, 256], [cb0__idx_v0, ck0__ssa_v0]), target_type=pl.FP32, mode='round'), [cb0__idx_v0, ck0__ssa_v0])
                    cur__rv_v4: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(cur__ssa_v5)
            for cseed__idx_v0, (carry_tids__iter_v3,) in pl.range(5, init_values=(carry_tids__iter_v1,)):
                carry_tids__ssa_v5: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_tids__iter_v3, cseed__idx_v0, ch_tid__ssa_v0)
                carry_tids__rv_v4: pl.Array[5, pl.TASK_ID] = pl.yield_(carry_tids__ssa_v5)
            carry_tids__rv_v2, cur__rv_v2 = pl.yield_(carry_tids__rv_v4, cur__rv_v4)
        normed__ssa_v0: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        carry_normed_tids__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            for xg_n__idx_v0, (carry_normed_tids__iter_v1, normed__iter_v1) in pl.range(5, init_values=(carry_normed_tids__ssa_v0, normed__ssa_v0)):
                xg_k0__ssa_v0: pl.Scalar[pl.INDEX] = xg_n__idx_v0 * 1024
                _submit_deps_buf__ssa_v0: pl.Array[1, pl.TASK_ID] = pl.array.create(1, dtype=pl.TASK_ID)
                _submit_deps_buf__ssa_v0_1: pl.Array[1, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf__ssa_v0, 0, pl.array.get_element(carry_tids__rv_v2, xg_n__idx_v0))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_gamma0", deps=[_submit_deps_buf__ssa_v0_1]) as xg0_tid__ssa_v0:
                    for kb__idx_v0, (normed__iter_v3,) in pl.pipeline(4, stage=2, init_values=(normed__iter_v1,)):
                        k0__ssa_v0: pl.Scalar[pl.INDEX] = xg_k0__ssa_v0 + kb__idx_v0 * 256
                        x_chunk__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur__rv_v2, [16, 256], [0, k0__ssa_v0])
                        gamma__ssa_v0: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight__ssa_v0, [1, 256], [0, k0__ssa_v0])
                        xg__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.col_expand_mul(x_chunk__ssa_v0, gamma__ssa_v0)
                        normed__ssa_v5: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(normed__iter_v3, pl.tensor.cast(xg__ssa_v0, target_type=pl.BF16, mode='round'), [0, k0__ssa_v0])
                        normed__rv_v4: pl.Tensor[[16, 5120], pl.BF16] = pl.yield_(normed__ssa_v5)
                carry_normed_tids__ssa_v3: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_normed_tids__iter_v1, xg_n__idx_v0, xg0_tid__ssa_v0)
                carry_normed_tids__rv_v2, normed__rv_v2 = pl.yield_(carry_normed_tids__ssa_v3, normed__rv_v4)
        next_hidden__ssa_v0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        next_normed__ssa_v0: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        layer_hidden_base_inline217__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX) * 5120
        layer_inter_base_inline296__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX) * 17408
        num_layers_actual_inline264__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(input_rms_weight__ssa_v0, 0)
        layer_cache_rows_inline282__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(k_cache__ssa_v0, 0) // num_layers_actual_inline264__ssa_v0
        layer_cache_base_inline156__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(0, pl.INDEX) * layer_cache_rows_inline282__ssa_v0
        user_batch_inline342__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(seq_lens__ssa_v0, 0)
        max_blocks_per_seq_inline401__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(block_table__ssa_v0, 0) // user_batch_inline342__ssa_v0
        q_norm_w_inline277__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(q_norm_weight__ssa_v0, [1, 128], [0, 0])
        k_norm_w_inline253__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(k_norm_weight__ssa_v0, [1, 128], [0, 0])
        down_tids_inline404__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
        q_tile_tids_inline280__ssa_v0: pl.Array[50, pl.TASK_ID] = pl.array.create(50, dtype=pl.TASK_ID)
        k_tile_tids_inline174__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
        v_tile_tids_inline438__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
        qk_tids_inline247__ssa_v0: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
        rope_grp_tids_inline266__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
        inv_rms_states_inline449__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        q_proj_inline248__ssa_v0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        k_proj_inline383__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        v_proj_inline241__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        all_q_padded_inline257__ssa_v0: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.create([2048, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        attn_out_inline203__ssa_v0: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        all_oi_tmp_inline318__ssa_v0: pl.Tensor[[65536, 128], pl.FP32] = pl.tensor.create([65536, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        all_cur_mi_inline409__ssa_v0: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.create([65536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        all_cur_li_inline294__ssa_v0: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.create([65536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        fa_work_table_inline281__ssa_v0: pl.Tensor[[512, 1], pl.INT32] = pl.tensor.create([512, 1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        fa_total_inline262__ssa_v0: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            _submit_deps_buf_inline361__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_inline362__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline361__ssa_v0, 0, pl.array.get_element(carry_tids__rv_v2, 0))
            _submit_deps_buf_inline230__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline362__ssa_v0, 1, pl.array.get_element(carry_tids__rv_v2, 1))
            _submit_deps_buf_inline307__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline230__ssa_v0, 2, pl.array.get_element(carry_tids__rv_v2, 2))
            _submit_deps_buf_inline276__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline307__ssa_v0, 3, pl.array.get_element(carry_tids__rv_v2, 3))
            _submit_deps_buf_inline310__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline276__ssa_v0, 4, pl.array.get_element(carry_tids__rv_v2, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip", deps=[_submit_deps_buf_inline310__ssa_v0]) as rms_tid_inline314__ssa_v0:
                partial_sq_inline299__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                for kb_inline317__idx_v0, (partial_sq_inline299__iter_v1,) in pl.pipeline(20, stage=4, init_values=(partial_sq_inline299__ssa_v0,)):
                    k0_inline321__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline317__idx_v0 * 256
                    x_chunk_inline271__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur__rv_v2, [16, 256], [0, k0_inline321__ssa_v0])
                    partial_sq_inline299__ssa_v3: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq_inline299__iter_v1, pl.tensor.reshape(pl.tensor.row_sum(pl.tensor.mul(x_chunk_inline271__ssa_v0, x_chunk_inline271__ssa_v0)), [1, 16]))
                    partial_sq_inline299__rv_v2: pl.Tensor[[1, 16], pl.FP32] = pl.yield_(partial_sq_inline299__ssa_v3)
                variance_inline268__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq_inline299__rv_v2, 0.00019531250000000001), 9.9999999999999995e-07), [16, 1])
                inv_rms_inline335__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance_inline268__ssa_v0))
                inv_rms_states_inline449__ssa_v1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.assemble(inv_rms_states_inline449__ssa_v0, inv_rms_inline335__ssa_v0, [0, 0])
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed") as q_seed_tid_inline180__ssa_v0:
                for snb_inline238__idx_v0, (q_proj_inline248__iter_v1,) in pl.pipeline(10, stage=2, init_values=(q_proj_inline248__ssa_v0,)):
                    q_proj_inline248__ssa_v3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_inline248__iter_v1, pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0), [0, snb_inline238__idx_v0 * 512])
                    q_proj_inline248__rv_v2: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(q_proj_inline248__ssa_v3)
            for q_nt_inline194__idx_v0, (q_proj_inline248__iter_v4, q_tile_tids_inline280__iter_v1) in pl.parallel(10, init_values=(q_proj_inline248__rv_v2, q_tile_tids_inline280__ssa_v0)):
                q_n_region_inline300__ssa_v0: pl.Scalar[pl.INDEX] = q_nt_inline194__idx_v0 * 512
                for q_ks_inline287__idx_v0, (q_proj_inline248__iter_v6, q_tile_tids_inline280__iter_v3) in pl.range(5, init_values=(q_proj_inline248__iter_v4, q_tile_tids_inline280__iter_v1)):
                    q_k_base_inline332__ssa_v0: pl.Scalar[pl.INDEX] = q_ks_inline287__idx_v0 * 1024
                    _submit_deps_buf_inline197__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline343__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline197__ssa_v0, 0, pl.array.get_element(carry_normed_tids__rv_v2, q_ks_inline287__idx_v0))
                    _submit_deps_buf_inline236__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline343__ssa_v0, 1, q_seed_tid_inline180__ssa_v0)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj", deps=[_submit_deps_buf_inline236__ssa_v0]) as q_tid_inline347__ssa_v0:
                        for n_sub_inline350__idx_v0, (q_proj_inline248__iter_v8,) in pl.range(2, init_values=(q_proj_inline248__iter_v6,)):
                            n0_inline352__ssa_v0: pl.Scalar[pl.INDEX] = q_n_region_inline300__ssa_v0 + n_sub_inline350__idx_v0 * 256
                            q_acc_inline291__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed__rv_v2, [16, 256], [0, q_k_base_inline332__ssa_v0]), pl.tensor.slice(wq__ssa_v0, [256, 256], [layer_hidden_base_inline217__ssa_v0 + q_k_base_inline332__ssa_v0, n0_inline352__ssa_v0]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for kc_inline292__idx_v0, (q_acc_inline291__iter_v1,) in pl.pipeline(1, 4, stage=2, init_values=(q_acc_inline291__ssa_v0,)):
                                kk_inline235__ssa_v0: pl.Scalar[pl.INDEX] = q_k_base_inline332__ssa_v0 + kc_inline292__idx_v0 * 256
                                q_acc_inline291__ssa_v3: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(q_acc_inline291__iter_v1, pl.tensor.slice(normed__rv_v2, [16, 256], [0, kk_inline235__ssa_v0]), pl.tensor.slice(wq__ssa_v0, [256, 256], [layer_hidden_base_inline217__ssa_v0 + kk_inline235__ssa_v0, n0_inline352__ssa_v0]), a_trans=False, b_trans=False)
                                q_acc_inline291__rv_v2: pl.Tensor[[16, 256], pl.FP32] = pl.yield_(q_acc_inline291__ssa_v3)
                            q_proj_inline248__ssa_v10: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_inline248__iter_v8, q_acc_inline291__rv_v2, [0, n0_inline352__ssa_v0], atomic=pl.AtomicType.Add)
                            q_proj_inline248__rv_v9: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(q_proj_inline248__ssa_v10)
                    q_tile_tids_inline280__ssa_v5: pl.Array[50, pl.TASK_ID] = pl.array.update_element(q_tile_tids_inline280__iter_v3, q_nt_inline194__idx_v0 * 5 + q_ks_inline287__idx_v0, q_tid_inline347__ssa_v0)
                    q_proj_inline248__rv_v7, q_tile_tids_inline280__rv_v4 = pl.yield_(q_proj_inline248__rv_v9, q_tile_tids_inline280__ssa_v5)
                q_proj_inline248__rv_v5, q_tile_tids_inline280__rv_v2 = pl.yield_(q_proj_inline248__rv_v7, q_tile_tids_inline280__rv_v4)
            _submit_deps_buf_inline338__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_inline358__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline338__ssa_v0, 0, pl.array.get_element(carry_tids__rv_v2, 0))
            _submit_deps_buf_inline370__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline358__ssa_v0, 1, pl.array.get_element(carry_tids__rv_v2, 1))
            _submit_deps_buf_inline379__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline370__ssa_v0, 2, pl.array.get_element(carry_tids__rv_v2, 2))
            _submit_deps_buf_inline380__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline379__ssa_v0, 3, pl.array.get_element(carry_tids__rv_v2, 3))
            _submit_deps_buf_inline301__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline380__ssa_v0, 4, pl.array.get_element(carry_tids__rv_v2, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_seed", deps=[_submit_deps_buf_inline301__ssa_v0]) as k_seed_tid_inline325__ssa_v0:
                k_proj_inline383__ssa_v1: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_inline383__ssa_v0, pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0), [0, 0])
            for k_nt_inline345__idx_v0, (k_proj_inline383__iter_v2, k_tile_tids_inline174__iter_v1) in pl.parallel(2, init_values=(k_proj_inline383__ssa_v1, k_tile_tids_inline174__ssa_v0)):
                k_n_region_inline229__ssa_v0: pl.Scalar[pl.INDEX] = k_nt_inline345__idx_v0 * 512
                for k_ks_inline227__idx_v0, (k_proj_inline383__iter_v4, k_tile_tids_inline174__iter_v3) in pl.range(5, init_values=(k_proj_inline383__iter_v2, k_tile_tids_inline174__iter_v1)):
                    k_k_base_inline340__ssa_v0: pl.Scalar[pl.INDEX] = k_ks_inline227__idx_v0 * 1024
                    _submit_deps_buf_inline188__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.create(6, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline319__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline188__ssa_v0, 0, pl.array.get_element(carry_normed_tids__rv_v2, 0))
                    _submit_deps_buf_inline145__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline319__ssa_v0, 1, pl.array.get_element(carry_normed_tids__rv_v2, 1))
                    _submit_deps_buf_inline222__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline145__ssa_v0, 2, pl.array.get_element(carry_normed_tids__rv_v2, 2))
                    _submit_deps_buf_inline215__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline222__ssa_v0, 3, pl.array.get_element(carry_normed_tids__rv_v2, 3))
                    _submit_deps_buf_inline214__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline215__ssa_v0, 4, pl.array.get_element(carry_normed_tids__rv_v2, 4))
                    _submit_deps_buf_inline212__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline214__ssa_v0, 5, k_seed_tid_inline325__ssa_v0)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj", deps=[_submit_deps_buf_inline212__ssa_v0]) as k_tid_inline158__ssa_v0:
                        for n_sub_inline206__idx_v0, (k_proj_inline383__iter_v6,) in pl.range(2, init_values=(k_proj_inline383__iter_v4,)):
                            n0_inline352__ssa_v1: pl.Scalar[pl.INDEX] = k_n_region_inline229__ssa_v0 + n_sub_inline206__idx_v0 * 256
                            k_acc_inline378__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed__rv_v2, [16, 256], [0, k_k_base_inline340__ssa_v0]), pl.tensor.slice(wk__ssa_v0, [256, 256], [layer_hidden_base_inline217__ssa_v0 + k_k_base_inline340__ssa_v0, n0_inline352__ssa_v1]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for kc_inline208__idx_v0, (k_acc_inline378__iter_v1,) in pl.pipeline(1, 4, stage=2, init_values=(k_acc_inline378__ssa_v0,)):
                                kk_inline235__ssa_v1: pl.Scalar[pl.INDEX] = k_k_base_inline340__ssa_v0 + kc_inline208__idx_v0 * 256
                                k_acc_inline378__ssa_v3: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(k_acc_inline378__iter_v1, pl.tensor.slice(normed__rv_v2, [16, 256], [0, kk_inline235__ssa_v1]), pl.tensor.slice(wk__ssa_v0, [256, 256], [layer_hidden_base_inline217__ssa_v0 + kk_inline235__ssa_v1, n0_inline352__ssa_v1]), a_trans=False, b_trans=False)
                                k_acc_inline378__rv_v2: pl.Tensor[[16, 256], pl.FP32] = pl.yield_(k_acc_inline378__ssa_v3)
                            k_proj_inline383__ssa_v8: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_inline383__iter_v6, k_acc_inline378__rv_v2, [0, n0_inline352__ssa_v1], atomic=pl.AtomicType.Add)
                            k_proj_inline383__rv_v7: pl.Tensor[[16, 1024], pl.FP32] = pl.yield_(k_proj_inline383__ssa_v8)
                    k_tile_tids_inline174__ssa_v5: pl.Array[10, pl.TASK_ID] = pl.array.update_element(k_tile_tids_inline174__iter_v3, k_nt_inline345__idx_v0 * 5 + k_ks_inline227__idx_v0, k_tid_inline158__ssa_v0)
                    k_proj_inline383__rv_v5, k_tile_tids_inline174__rv_v4 = pl.yield_(k_proj_inline383__rv_v7, k_tile_tids_inline174__ssa_v5)
                k_proj_inline383__rv_v3, k_tile_tids_inline174__rv_v2 = pl.yield_(k_proj_inline383__rv_v5, k_tile_tids_inline174__rv_v4)
            _submit_deps_buf_inline324__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_inline441__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline324__ssa_v0, 0, pl.array.get_element(carry_tids__rv_v2, 0))
            _submit_deps_buf_inline202__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline441__ssa_v0, 1, pl.array.get_element(carry_tids__rv_v2, 1))
            _submit_deps_buf_inline275__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline202__ssa_v0, 2, pl.array.get_element(carry_tids__rv_v2, 2))
            _submit_deps_buf_inline225__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline275__ssa_v0, 3, pl.array.get_element(carry_tids__rv_v2, 3))
            _submit_deps_buf_inline221__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline225__ssa_v0, 4, pl.array.get_element(carry_tids__rv_v2, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_seed", deps=[_submit_deps_buf_inline221__ssa_v0]) as v_seed_tid_inline250__ssa_v0:
                v_proj_inline241__ssa_v1: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(v_proj_inline241__ssa_v0, pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0), [0, 0])
            for v_nt_inline196__idx_v0, (v_proj_inline241__iter_v2, v_tile_tids_inline438__iter_v1) in pl.parallel(2, init_values=(v_proj_inline241__ssa_v1, v_tile_tids_inline438__ssa_v0)):
                v_n_region_inline267__ssa_v0: pl.Scalar[pl.INDEX] = v_nt_inline196__idx_v0 * 512
                for v_ks_inline160__idx_v0, (v_proj_inline241__iter_v4, v_tile_tids_inline438__iter_v3) in pl.range(5, init_values=(v_proj_inline241__iter_v2, v_tile_tids_inline438__iter_v1)):
                    v_k_base_inline376__ssa_v0: pl.Scalar[pl.INDEX] = v_ks_inline160__idx_v0 * 1024
                    _submit_deps_buf_inline272__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.create(6, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline154__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline272__ssa_v0, 0, pl.array.get_element(carry_normed_tids__rv_v2, 0))
                    _submit_deps_buf_inline263__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline154__ssa_v0, 1, pl.array.get_element(carry_normed_tids__rv_v2, 1))
                    _submit_deps_buf_inline305__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline263__ssa_v0, 2, pl.array.get_element(carry_normed_tids__rv_v2, 2))
                    _submit_deps_buf_inline394__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline305__ssa_v0, 3, pl.array.get_element(carry_normed_tids__rv_v2, 3))
                    _submit_deps_buf_inline190__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline394__ssa_v0, 4, pl.array.get_element(carry_normed_tids__rv_v2, 4))
                    _submit_deps_buf_inline187__ssa_v0: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline190__ssa_v0, 5, v_seed_tid_inline250__ssa_v0)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj", deps=[_submit_deps_buf_inline187__ssa_v0]) as v_tid_inline249__ssa_v0:
                        for n_sub_inline213__idx_v0, (v_proj_inline241__iter_v6,) in pl.range(2, init_values=(v_proj_inline241__iter_v4,)):
                            n0_inline352__ssa_v2: pl.Scalar[pl.INDEX] = v_n_region_inline267__ssa_v0 + n_sub_inline213__idx_v0 * 256
                            v_acc_inline326__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed__rv_v2, [16, 256], [0, v_k_base_inline376__ssa_v0]), pl.tensor.slice(wv__ssa_v0, [256, 256], [layer_hidden_base_inline217__ssa_v0 + v_k_base_inline376__ssa_v0, n0_inline352__ssa_v2]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for kc_inline186__idx_v0, (v_acc_inline326__iter_v1,) in pl.pipeline(1, 4, stage=2, init_values=(v_acc_inline326__ssa_v0,)):
                                kk_inline235__ssa_v2: pl.Scalar[pl.INDEX] = v_k_base_inline376__ssa_v0 + kc_inline186__idx_v0 * 256
                                v_acc_inline326__ssa_v3: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(v_acc_inline326__iter_v1, pl.tensor.slice(normed__rv_v2, [16, 256], [0, kk_inline235__ssa_v2]), pl.tensor.slice(wv__ssa_v0, [256, 256], [layer_hidden_base_inline217__ssa_v0 + kk_inline235__ssa_v2, n0_inline352__ssa_v2]), a_trans=False, b_trans=False)
                                v_acc_inline326__rv_v2: pl.Tensor[[16, 256], pl.FP32] = pl.yield_(v_acc_inline326__ssa_v3)
                            v_proj_inline241__ssa_v8: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(v_proj_inline241__iter_v6, v_acc_inline326__rv_v2, [0, n0_inline352__ssa_v2], atomic=pl.AtomicType.Add)
                            v_proj_inline241__rv_v7: pl.Tensor[[16, 1024], pl.FP32] = pl.yield_(v_proj_inline241__ssa_v8)
                    v_tile_tids_inline438__ssa_v5: pl.Array[10, pl.TASK_ID] = pl.array.update_element(v_tile_tids_inline438__iter_v3, v_nt_inline196__idx_v0 * 5 + v_ks_inline160__idx_v0, v_tid_inline249__ssa_v0)
                    v_proj_inline241__rv_v5, v_tile_tids_inline438__rv_v4 = pl.yield_(v_proj_inline241__rv_v7, v_tile_tids_inline438__ssa_v5)
                v_proj_inline241__rv_v3, v_tile_tids_inline438__rv_v2 = pl.yield_(v_proj_inline241__rv_v5, v_tile_tids_inline438__rv_v4)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_work_build") as work_tid_inline183__ssa_v0:
                wb_ctx_inline177__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [0]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v0 in pl.range(wb_ctx_inline177__ssa_v0):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [wp_inline304__idx_v0, 0], pl.cast(wp_inline304__idx_v0, pl.INT32))
                cursor_inline169__ssa_v1: pl.Scalar[pl.INDEX] = wb_ctx_inline177__ssa_v0
                wb_ctx_inline177__ssa_v1: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [1]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v1 in pl.range(wb_ctx_inline177__ssa_v1):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v1 + wp_inline304__idx_v1, 0], pl.cast(wp_inline304__idx_v1 + 32, pl.INT32))
                cursor_inline169__ssa_v2: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v1 + wb_ctx_inline177__ssa_v1
                wb_ctx_inline177__ssa_v2: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [2]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v2 in pl.range(wb_ctx_inline177__ssa_v2):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v2 + wp_inline304__idx_v2, 0], pl.cast(wp_inline304__idx_v2 + 64, pl.INT32))
                cursor_inline169__ssa_v3: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v2 + wb_ctx_inline177__ssa_v2
                wb_ctx_inline177__ssa_v3: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [3]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v3 in pl.range(wb_ctx_inline177__ssa_v3):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v3 + wp_inline304__idx_v3, 0], pl.cast(wp_inline304__idx_v3 + 96, pl.INT32))
                cursor_inline169__ssa_v4: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v3 + wb_ctx_inline177__ssa_v3
                wb_ctx_inline177__ssa_v4: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [4]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v4 in pl.range(wb_ctx_inline177__ssa_v4):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v4 + wp_inline304__idx_v4, 0], pl.cast(wp_inline304__idx_v4 + 128, pl.INT32))
                cursor_inline169__ssa_v5: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v4 + wb_ctx_inline177__ssa_v4
                wb_ctx_inline177__ssa_v5: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [5]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v5 in pl.range(wb_ctx_inline177__ssa_v5):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v5 + wp_inline304__idx_v5, 0], pl.cast(wp_inline304__idx_v5 + 160, pl.INT32))
                cursor_inline169__ssa_v6: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v5 + wb_ctx_inline177__ssa_v5
                wb_ctx_inline177__ssa_v6: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [6]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v6 in pl.range(wb_ctx_inline177__ssa_v6):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v6 + wp_inline304__idx_v6, 0], pl.cast(wp_inline304__idx_v6 + 192, pl.INT32))
                cursor_inline169__ssa_v7: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v6 + wb_ctx_inline177__ssa_v6
                wb_ctx_inline177__ssa_v7: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [7]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v7 in pl.range(wb_ctx_inline177__ssa_v7):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v7 + wp_inline304__idx_v7, 0], pl.cast(wp_inline304__idx_v7 + 224, pl.INT32))
                cursor_inline169__ssa_v8: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v7 + wb_ctx_inline177__ssa_v7
                wb_ctx_inline177__ssa_v8: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [8]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v8 in pl.range(wb_ctx_inline177__ssa_v8):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v8 + wp_inline304__idx_v8, 0], pl.cast(wp_inline304__idx_v8 + 256, pl.INT32))
                cursor_inline169__ssa_v9: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v8 + wb_ctx_inline177__ssa_v8
                wb_ctx_inline177__ssa_v9: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [9]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v9 in pl.range(wb_ctx_inline177__ssa_v9):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v9 + wp_inline304__idx_v9, 0], pl.cast(wp_inline304__idx_v9 + 288, pl.INT32))
                cursor_inline169__ssa_v10: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v9 + wb_ctx_inline177__ssa_v9
                wb_ctx_inline177__ssa_v10: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [10]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v10 in pl.range(wb_ctx_inline177__ssa_v10):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v10 + wp_inline304__idx_v10, 0], pl.cast(wp_inline304__idx_v10 + 320, pl.INT32))
                cursor_inline169__ssa_v11: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v10 + wb_ctx_inline177__ssa_v10
                wb_ctx_inline177__ssa_v11: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [11]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v11 in pl.range(wb_ctx_inline177__ssa_v11):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v11 + wp_inline304__idx_v11, 0], pl.cast(wp_inline304__idx_v11 + 352, pl.INT32))
                cursor_inline169__ssa_v12: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v11 + wb_ctx_inline177__ssa_v11
                wb_ctx_inline177__ssa_v12: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [12]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v12 in pl.range(wb_ctx_inline177__ssa_v12):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v12 + wp_inline304__idx_v12, 0], pl.cast(wp_inline304__idx_v12 + 384, pl.INT32))
                cursor_inline169__ssa_v13: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v12 + wb_ctx_inline177__ssa_v12
                wb_ctx_inline177__ssa_v13: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [13]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v13 in pl.range(wb_ctx_inline177__ssa_v13):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v13 + wp_inline304__idx_v13, 0], pl.cast(wp_inline304__idx_v13 + 416, pl.INT32))
                cursor_inline169__ssa_v14: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v13 + wb_ctx_inline177__ssa_v13
                wb_ctx_inline177__ssa_v14: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [14]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v14 in pl.range(wb_ctx_inline177__ssa_v14):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v14 + wp_inline304__idx_v14, 0], pl.cast(wp_inline304__idx_v14 + 448, pl.INT32))
                cursor_inline169__ssa_v15: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v14 + wb_ctx_inline177__ssa_v14
                wb_ctx_inline177__ssa_v15: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens__ssa_v0, [15]), pl.INDEX) + 127) // 128
                for wp_inline304__idx_v15 in pl.range(wb_ctx_inline177__ssa_v15):
                    pl.tensor.write(fa_work_table_inline281__ssa_v0, [cursor_inline169__ssa_v15 + wp_inline304__idx_v15, 0], pl.cast(wp_inline304__idx_v15 + 480, pl.INT32))
                cursor_inline169__ssa_v16: pl.Scalar[pl.INDEX] = cursor_inline169__ssa_v15 + wb_ctx_inline177__ssa_v15
                pl.tensor.write(fa_total_inline262__ssa_v0, [0, 0], pl.cast(cursor_inline169__ssa_v16, pl.INT32))
            q_proj_norm_inline375__ssa_v0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            k_proj_norm_inline171__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_inv_states_inline168__ssa_v0: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.create([640, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            k_inv_states_inline204__ssa_v0: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.create([128, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            inv_rms_col_inline165__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(inv_rms_states_inline449__ssa_v1, [16, 1], [0, 0])
            _submit_deps_buf_inline448__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
            _submit_deps_buf_inline323__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448__ssa_v0, 0, pl.array.get_element(q_tile_tids_inline280__rv_v2, 0))
            _submit_deps_buf_inline440__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323__ssa_v0, 1, pl.array.get_element(q_tile_tids_inline280__rv_v2, 1))
            _submit_deps_buf_inline316__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440__ssa_v0, 2, pl.array.get_element(q_tile_tids_inline280__rv_v2, 2))
            _submit_deps_buf_inline153__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316__ssa_v0, 3, pl.array.get_element(q_tile_tids_inline280__rv_v2, 3))
            _submit_deps_buf_inline339__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153__ssa_v0, 4, pl.array.get_element(q_tile_tids_inline280__rv_v2, 4))
            _submit_deps_buf_inline151__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339__ssa_v0, 5, pl.array.get_element(q_tile_tids_inline280__rv_v2, 5))
            _submit_deps_buf_inline353__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151__ssa_v0, 6, pl.array.get_element(q_tile_tids_inline280__rv_v2, 6))
            _submit_deps_buf_inline473__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353__ssa_v0, 7, pl.array.get_element(q_tile_tids_inline280__rv_v2, 7))
            _submit_deps_buf_inline148__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473__ssa_v0, 8, pl.array.get_element(q_tile_tids_inline280__rv_v2, 8))
            _submit_deps_buf_inline348__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148__ssa_v0, 9, pl.array.get_element(q_tile_tids_inline280__rv_v2, 9))
            _submit_deps_buf_inline244__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348__ssa_v0, 10, pl.array.get_element(k_tile_tids_inline174__rv_v2, 0))
            _submit_deps_buf_inline166__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244__ssa_v0, 11, pl.array.get_element(k_tile_tids_inline174__rv_v2, 1))
            _submit_deps_buf_inline176__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166__ssa_v0, 12, pl.array.get_element(k_tile_tids_inline174__rv_v2, 2))
            _submit_deps_buf_inline219__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176__ssa_v0, 13, pl.array.get_element(k_tile_tids_inline174__rv_v2, 3))
            _submit_deps_buf_inline144__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219__ssa_v0, 14, pl.array.get_element(k_tile_tids_inline174__rv_v2, 4))
            _submit_deps_buf_inline223__ssa_v0: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144__ssa_v0, 15, rms_tid_inline314__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223__ssa_v0]) as qk_tid_h_inline245__ssa_v0:
                q_slice_inline388__ssa_v0: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248__rv_v5, [16, 640], [0, 0]), inv_rms_col_inline165__ssa_v0)
                q_chunk_inline389__ssa_v0: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388__ssa_v0, [80, 128])
                q_g_inline393__ssa_v0: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389__ssa_v0, q_norm_w_inline277__ssa_v0)
                q_proj_norm_inline375__ssa_v1: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375__ssa_v0, pl.tensor.reshape(q_g_inline393__ssa_v0, [16, 640]), [0, 0])
                q_ss_inline312__ssa_v0: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389__ssa_v0, q_chunk_inline389__ssa_v0))
                q_inv_inline200__ssa_v0: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312__ssa_v0, 0.0078125), 9.9999999999999995e-07)))
                q_inv_states_inline168__ssa_v1: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168__ssa_v0, q_inv_inline200__ssa_v0, [0, 0])
                k_chunk_inline309__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383__rv_v3, [16, 128], [0, 0]), inv_rms_col_inline165__ssa_v0)
                k_g_inline397__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309__ssa_v0, k_norm_w_inline253__ssa_v0)
                k_proj_norm_inline171__ssa_v1: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171__ssa_v0, k_g_inline397__ssa_v0, [0, 0])
                k_ss_inline398__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309__ssa_v0, k_chunk_inline309__ssa_v0))
                k_inv_inline385__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398__ssa_v0, 0.0078125), 9.9999999999999995e-07)))
                k_inv_states_inline204__ssa_v1: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204__ssa_v0, k_inv_inline385__ssa_v0, [0, 0])
            qk_tids_inline247__ssa_v1: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247__ssa_v0, 0, qk_tid_h_inline245__ssa_v0)
            _submit_deps_buf_inline448__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
            _submit_deps_buf_inline323__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448__ssa_v1, 0, pl.array.get_element(q_tile_tids_inline280__rv_v2, 5))
            _submit_deps_buf_inline440__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323__ssa_v1, 1, pl.array.get_element(q_tile_tids_inline280__rv_v2, 6))
            _submit_deps_buf_inline316__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440__ssa_v1, 2, pl.array.get_element(q_tile_tids_inline280__rv_v2, 7))
            _submit_deps_buf_inline153__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316__ssa_v1, 3, pl.array.get_element(q_tile_tids_inline280__rv_v2, 8))
            _submit_deps_buf_inline339__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153__ssa_v1, 4, pl.array.get_element(q_tile_tids_inline280__rv_v2, 9))
            _submit_deps_buf_inline151__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339__ssa_v1, 5, pl.array.get_element(q_tile_tids_inline280__rv_v2, 10))
            _submit_deps_buf_inline353__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151__ssa_v1, 6, pl.array.get_element(q_tile_tids_inline280__rv_v2, 11))
            _submit_deps_buf_inline473__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353__ssa_v1, 7, pl.array.get_element(q_tile_tids_inline280__rv_v2, 12))
            _submit_deps_buf_inline148__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473__ssa_v1, 8, pl.array.get_element(q_tile_tids_inline280__rv_v2, 13))
            _submit_deps_buf_inline348__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148__ssa_v1, 9, pl.array.get_element(q_tile_tids_inline280__rv_v2, 14))
            _submit_deps_buf_inline244__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348__ssa_v1, 10, pl.array.get_element(k_tile_tids_inline174__rv_v2, 0))
            _submit_deps_buf_inline166__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244__ssa_v1, 11, pl.array.get_element(k_tile_tids_inline174__rv_v2, 1))
            _submit_deps_buf_inline176__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166__ssa_v1, 12, pl.array.get_element(k_tile_tids_inline174__rv_v2, 2))
            _submit_deps_buf_inline219__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176__ssa_v1, 13, pl.array.get_element(k_tile_tids_inline174__rv_v2, 3))
            _submit_deps_buf_inline144__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219__ssa_v1, 14, pl.array.get_element(k_tile_tids_inline174__rv_v2, 4))
            _submit_deps_buf_inline223__ssa_v1: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144__ssa_v1, 15, rms_tid_inline314__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223__ssa_v1]) as qk_tid_h_inline245__ssa_v1:
                q_slice_inline388__ssa_v1: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248__rv_v5, [16, 640], [0, 640]), inv_rms_col_inline165__ssa_v0)
                q_chunk_inline389__ssa_v1: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388__ssa_v1, [80, 128])
                q_g_inline393__ssa_v1: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389__ssa_v1, q_norm_w_inline277__ssa_v0)
                q_proj_norm_inline375__ssa_v2: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375__ssa_v1, pl.tensor.reshape(q_g_inline393__ssa_v1, [16, 640]), [0, 640])
                q_ss_inline312__ssa_v1: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389__ssa_v1, q_chunk_inline389__ssa_v1))
                q_inv_inline200__ssa_v1: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312__ssa_v1, 0.0078125), 9.9999999999999995e-07)))
                q_inv_states_inline168__ssa_v2: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168__ssa_v1, q_inv_inline200__ssa_v1, [80, 0])
                k_chunk_inline309__ssa_v1: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383__rv_v3, [16, 128], [0, 128]), inv_rms_col_inline165__ssa_v0)
                k_g_inline397__ssa_v1: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309__ssa_v1, k_norm_w_inline253__ssa_v0)
                k_proj_norm_inline171__ssa_v2: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171__ssa_v1, k_g_inline397__ssa_v1, [0, 128])
                k_ss_inline398__ssa_v1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309__ssa_v1, k_chunk_inline309__ssa_v1))
                k_inv_inline385__ssa_v1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398__ssa_v1, 0.0078125), 9.9999999999999995e-07)))
                k_inv_states_inline204__ssa_v2: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204__ssa_v1, k_inv_inline385__ssa_v1, [16, 0])
            qk_tids_inline247__ssa_v2: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247__ssa_v1, 1, qk_tid_h_inline245__ssa_v1)
            _submit_deps_buf_inline448__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
            _submit_deps_buf_inline323__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448__ssa_v2, 0, pl.array.get_element(q_tile_tids_inline280__rv_v2, 10))
            _submit_deps_buf_inline440__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323__ssa_v2, 1, pl.array.get_element(q_tile_tids_inline280__rv_v2, 11))
            _submit_deps_buf_inline316__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440__ssa_v2, 2, pl.array.get_element(q_tile_tids_inline280__rv_v2, 12))
            _submit_deps_buf_inline153__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316__ssa_v2, 3, pl.array.get_element(q_tile_tids_inline280__rv_v2, 13))
            _submit_deps_buf_inline339__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153__ssa_v2, 4, pl.array.get_element(q_tile_tids_inline280__rv_v2, 14))
            _submit_deps_buf_inline151__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339__ssa_v2, 5, pl.array.get_element(q_tile_tids_inline280__rv_v2, 15))
            _submit_deps_buf_inline353__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151__ssa_v2, 6, pl.array.get_element(q_tile_tids_inline280__rv_v2, 16))
            _submit_deps_buf_inline473__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353__ssa_v2, 7, pl.array.get_element(q_tile_tids_inline280__rv_v2, 17))
            _submit_deps_buf_inline148__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473__ssa_v2, 8, pl.array.get_element(q_tile_tids_inline280__rv_v2, 18))
            _submit_deps_buf_inline348__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148__ssa_v2, 9, pl.array.get_element(q_tile_tids_inline280__rv_v2, 19))
            _submit_deps_buf_inline244__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348__ssa_v2, 10, pl.array.get_element(k_tile_tids_inline174__rv_v2, 0))
            _submit_deps_buf_inline166__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244__ssa_v2, 11, pl.array.get_element(k_tile_tids_inline174__rv_v2, 1))
            _submit_deps_buf_inline176__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166__ssa_v2, 12, pl.array.get_element(k_tile_tids_inline174__rv_v2, 2))
            _submit_deps_buf_inline219__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176__ssa_v2, 13, pl.array.get_element(k_tile_tids_inline174__rv_v2, 3))
            _submit_deps_buf_inline144__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219__ssa_v2, 14, pl.array.get_element(k_tile_tids_inline174__rv_v2, 4))
            _submit_deps_buf_inline223__ssa_v2: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144__ssa_v2, 15, rms_tid_inline314__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223__ssa_v2]) as qk_tid_h_inline245__ssa_v2:
                q_slice_inline388__ssa_v2: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248__rv_v5, [16, 640], [0, 1280]), inv_rms_col_inline165__ssa_v0)
                q_chunk_inline389__ssa_v2: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388__ssa_v2, [80, 128])
                q_g_inline393__ssa_v2: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389__ssa_v2, q_norm_w_inline277__ssa_v0)
                q_proj_norm_inline375__ssa_v3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375__ssa_v2, pl.tensor.reshape(q_g_inline393__ssa_v2, [16, 640]), [0, 1280])
                q_ss_inline312__ssa_v2: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389__ssa_v2, q_chunk_inline389__ssa_v2))
                q_inv_inline200__ssa_v2: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312__ssa_v2, 0.0078125), 9.9999999999999995e-07)))
                q_inv_states_inline168__ssa_v3: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168__ssa_v2, q_inv_inline200__ssa_v2, [160, 0])
                k_chunk_inline309__ssa_v2: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383__rv_v3, [16, 128], [0, 256]), inv_rms_col_inline165__ssa_v0)
                k_g_inline397__ssa_v2: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309__ssa_v2, k_norm_w_inline253__ssa_v0)
                k_proj_norm_inline171__ssa_v3: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171__ssa_v2, k_g_inline397__ssa_v2, [0, 256])
                k_ss_inline398__ssa_v2: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309__ssa_v2, k_chunk_inline309__ssa_v2))
                k_inv_inline385__ssa_v2: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398__ssa_v2, 0.0078125), 9.9999999999999995e-07)))
                k_inv_states_inline204__ssa_v3: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204__ssa_v2, k_inv_inline385__ssa_v2, [32, 0])
            qk_tids_inline247__ssa_v3: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247__ssa_v2, 2, qk_tid_h_inline245__ssa_v2)
            _submit_deps_buf_inline448__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
            _submit_deps_buf_inline323__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448__ssa_v3, 0, pl.array.get_element(q_tile_tids_inline280__rv_v2, 15))
            _submit_deps_buf_inline440__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323__ssa_v3, 1, pl.array.get_element(q_tile_tids_inline280__rv_v2, 16))
            _submit_deps_buf_inline316__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440__ssa_v3, 2, pl.array.get_element(q_tile_tids_inline280__rv_v2, 17))
            _submit_deps_buf_inline153__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316__ssa_v3, 3, pl.array.get_element(q_tile_tids_inline280__rv_v2, 18))
            _submit_deps_buf_inline339__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153__ssa_v3, 4, pl.array.get_element(q_tile_tids_inline280__rv_v2, 19))
            _submit_deps_buf_inline151__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339__ssa_v3, 5, pl.array.get_element(q_tile_tids_inline280__rv_v2, 20))
            _submit_deps_buf_inline353__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151__ssa_v3, 6, pl.array.get_element(q_tile_tids_inline280__rv_v2, 21))
            _submit_deps_buf_inline473__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353__ssa_v3, 7, pl.array.get_element(q_tile_tids_inline280__rv_v2, 22))
            _submit_deps_buf_inline148__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473__ssa_v3, 8, pl.array.get_element(q_tile_tids_inline280__rv_v2, 23))
            _submit_deps_buf_inline348__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148__ssa_v3, 9, pl.array.get_element(q_tile_tids_inline280__rv_v2, 24))
            _submit_deps_buf_inline244__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348__ssa_v3, 10, pl.array.get_element(k_tile_tids_inline174__rv_v2, 0))
            _submit_deps_buf_inline166__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244__ssa_v3, 11, pl.array.get_element(k_tile_tids_inline174__rv_v2, 1))
            _submit_deps_buf_inline176__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166__ssa_v3, 12, pl.array.get_element(k_tile_tids_inline174__rv_v2, 2))
            _submit_deps_buf_inline219__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176__ssa_v3, 13, pl.array.get_element(k_tile_tids_inline174__rv_v2, 3))
            _submit_deps_buf_inline144__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219__ssa_v3, 14, pl.array.get_element(k_tile_tids_inline174__rv_v2, 4))
            _submit_deps_buf_inline223__ssa_v3: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144__ssa_v3, 15, rms_tid_inline314__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223__ssa_v3]) as qk_tid_h_inline245__ssa_v3:
                q_slice_inline388__ssa_v3: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248__rv_v5, [16, 640], [0, 1920]), inv_rms_col_inline165__ssa_v0)
                q_chunk_inline389__ssa_v3: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388__ssa_v3, [80, 128])
                q_g_inline393__ssa_v3: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389__ssa_v3, q_norm_w_inline277__ssa_v0)
                q_proj_norm_inline375__ssa_v4: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375__ssa_v3, pl.tensor.reshape(q_g_inline393__ssa_v3, [16, 640]), [0, 1920])
                q_ss_inline312__ssa_v3: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389__ssa_v3, q_chunk_inline389__ssa_v3))
                q_inv_inline200__ssa_v3: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312__ssa_v3, 0.0078125), 9.9999999999999995e-07)))
                q_inv_states_inline168__ssa_v4: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168__ssa_v3, q_inv_inline200__ssa_v3, [240, 0])
                k_chunk_inline309__ssa_v3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383__rv_v3, [16, 128], [0, 384]), inv_rms_col_inline165__ssa_v0)
                k_g_inline397__ssa_v3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309__ssa_v3, k_norm_w_inline253__ssa_v0)
                k_proj_norm_inline171__ssa_v4: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171__ssa_v3, k_g_inline397__ssa_v3, [0, 384])
                k_ss_inline398__ssa_v3: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309__ssa_v3, k_chunk_inline309__ssa_v3))
                k_inv_inline385__ssa_v3: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398__ssa_v3, 0.0078125), 9.9999999999999995e-07)))
                k_inv_states_inline204__ssa_v4: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204__ssa_v3, k_inv_inline385__ssa_v3, [48, 0])
            qk_tids_inline247__ssa_v4: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247__ssa_v3, 3, qk_tid_h_inline245__ssa_v3)
            _submit_deps_buf_inline448__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
            _submit_deps_buf_inline323__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448__ssa_v4, 0, pl.array.get_element(q_tile_tids_inline280__rv_v2, 25))
            _submit_deps_buf_inline440__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323__ssa_v4, 1, pl.array.get_element(q_tile_tids_inline280__rv_v2, 26))
            _submit_deps_buf_inline316__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440__ssa_v4, 2, pl.array.get_element(q_tile_tids_inline280__rv_v2, 27))
            _submit_deps_buf_inline153__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316__ssa_v4, 3, pl.array.get_element(q_tile_tids_inline280__rv_v2, 28))
            _submit_deps_buf_inline339__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153__ssa_v4, 4, pl.array.get_element(q_tile_tids_inline280__rv_v2, 29))
            _submit_deps_buf_inline151__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339__ssa_v4, 5, pl.array.get_element(q_tile_tids_inline280__rv_v2, 30))
            _submit_deps_buf_inline353__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151__ssa_v4, 6, pl.array.get_element(q_tile_tids_inline280__rv_v2, 31))
            _submit_deps_buf_inline473__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353__ssa_v4, 7, pl.array.get_element(q_tile_tids_inline280__rv_v2, 32))
            _submit_deps_buf_inline148__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473__ssa_v4, 8, pl.array.get_element(q_tile_tids_inline280__rv_v2, 33))
            _submit_deps_buf_inline348__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148__ssa_v4, 9, pl.array.get_element(q_tile_tids_inline280__rv_v2, 34))
            _submit_deps_buf_inline244__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348__ssa_v4, 10, pl.array.get_element(k_tile_tids_inline174__rv_v2, 5))
            _submit_deps_buf_inline166__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244__ssa_v4, 11, pl.array.get_element(k_tile_tids_inline174__rv_v2, 6))
            _submit_deps_buf_inline176__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166__ssa_v4, 12, pl.array.get_element(k_tile_tids_inline174__rv_v2, 7))
            _submit_deps_buf_inline219__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176__ssa_v4, 13, pl.array.get_element(k_tile_tids_inline174__rv_v2, 8))
            _submit_deps_buf_inline144__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219__ssa_v4, 14, pl.array.get_element(k_tile_tids_inline174__rv_v2, 9))
            _submit_deps_buf_inline223__ssa_v4: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144__ssa_v4, 15, rms_tid_inline314__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223__ssa_v4]) as qk_tid_h_inline245__ssa_v4:
                q_slice_inline388__ssa_v4: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248__rv_v5, [16, 640], [0, 2560]), inv_rms_col_inline165__ssa_v0)
                q_chunk_inline389__ssa_v4: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388__ssa_v4, [80, 128])
                q_g_inline393__ssa_v4: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389__ssa_v4, q_norm_w_inline277__ssa_v0)
                q_proj_norm_inline375__ssa_v5: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375__ssa_v4, pl.tensor.reshape(q_g_inline393__ssa_v4, [16, 640]), [0, 2560])
                q_ss_inline312__ssa_v4: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389__ssa_v4, q_chunk_inline389__ssa_v4))
                q_inv_inline200__ssa_v4: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312__ssa_v4, 0.0078125), 9.9999999999999995e-07)))
                q_inv_states_inline168__ssa_v5: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168__ssa_v4, q_inv_inline200__ssa_v4, [320, 0])
                k_chunk_inline309__ssa_v4: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383__rv_v3, [16, 128], [0, 512]), inv_rms_col_inline165__ssa_v0)
                k_g_inline397__ssa_v4: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309__ssa_v4, k_norm_w_inline253__ssa_v0)
                k_proj_norm_inline171__ssa_v5: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171__ssa_v4, k_g_inline397__ssa_v4, [0, 512])
                k_ss_inline398__ssa_v4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309__ssa_v4, k_chunk_inline309__ssa_v4))
                k_inv_inline385__ssa_v4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398__ssa_v4, 0.0078125), 9.9999999999999995e-07)))
                k_inv_states_inline204__ssa_v5: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204__ssa_v4, k_inv_inline385__ssa_v4, [64, 0])
            qk_tids_inline247__ssa_v5: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247__ssa_v4, 4, qk_tid_h_inline245__ssa_v4)
            _submit_deps_buf_inline448__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
            _submit_deps_buf_inline323__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448__ssa_v5, 0, pl.array.get_element(q_tile_tids_inline280__rv_v2, 30))
            _submit_deps_buf_inline440__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323__ssa_v5, 1, pl.array.get_element(q_tile_tids_inline280__rv_v2, 31))
            _submit_deps_buf_inline316__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440__ssa_v5, 2, pl.array.get_element(q_tile_tids_inline280__rv_v2, 32))
            _submit_deps_buf_inline153__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316__ssa_v5, 3, pl.array.get_element(q_tile_tids_inline280__rv_v2, 33))
            _submit_deps_buf_inline339__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153__ssa_v5, 4, pl.array.get_element(q_tile_tids_inline280__rv_v2, 34))
            _submit_deps_buf_inline151__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339__ssa_v5, 5, pl.array.get_element(q_tile_tids_inline280__rv_v2, 35))
            _submit_deps_buf_inline353__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151__ssa_v5, 6, pl.array.get_element(q_tile_tids_inline280__rv_v2, 36))
            _submit_deps_buf_inline473__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353__ssa_v5, 7, pl.array.get_element(q_tile_tids_inline280__rv_v2, 37))
            _submit_deps_buf_inline148__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473__ssa_v5, 8, pl.array.get_element(q_tile_tids_inline280__rv_v2, 38))
            _submit_deps_buf_inline348__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148__ssa_v5, 9, pl.array.get_element(q_tile_tids_inline280__rv_v2, 39))
            _submit_deps_buf_inline244__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348__ssa_v5, 10, pl.array.get_element(k_tile_tids_inline174__rv_v2, 5))
            _submit_deps_buf_inline166__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244__ssa_v5, 11, pl.array.get_element(k_tile_tids_inline174__rv_v2, 6))
            _submit_deps_buf_inline176__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166__ssa_v5, 12, pl.array.get_element(k_tile_tids_inline174__rv_v2, 7))
            _submit_deps_buf_inline219__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176__ssa_v5, 13, pl.array.get_element(k_tile_tids_inline174__rv_v2, 8))
            _submit_deps_buf_inline144__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219__ssa_v5, 14, pl.array.get_element(k_tile_tids_inline174__rv_v2, 9))
            _submit_deps_buf_inline223__ssa_v5: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144__ssa_v5, 15, rms_tid_inline314__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223__ssa_v5]) as qk_tid_h_inline245__ssa_v5:
                q_slice_inline388__ssa_v5: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248__rv_v5, [16, 640], [0, 3200]), inv_rms_col_inline165__ssa_v0)
                q_chunk_inline389__ssa_v5: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388__ssa_v5, [80, 128])
                q_g_inline393__ssa_v5: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389__ssa_v5, q_norm_w_inline277__ssa_v0)
                q_proj_norm_inline375__ssa_v6: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375__ssa_v5, pl.tensor.reshape(q_g_inline393__ssa_v5, [16, 640]), [0, 3200])
                q_ss_inline312__ssa_v5: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389__ssa_v5, q_chunk_inline389__ssa_v5))
                q_inv_inline200__ssa_v5: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312__ssa_v5, 0.0078125), 9.9999999999999995e-07)))
                q_inv_states_inline168__ssa_v6: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168__ssa_v5, q_inv_inline200__ssa_v5, [400, 0])
                k_chunk_inline309__ssa_v5: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383__rv_v3, [16, 128], [0, 640]), inv_rms_col_inline165__ssa_v0)
                k_g_inline397__ssa_v5: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309__ssa_v5, k_norm_w_inline253__ssa_v0)
                k_proj_norm_inline171__ssa_v6: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171__ssa_v5, k_g_inline397__ssa_v5, [0, 640])
                k_ss_inline398__ssa_v5: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309__ssa_v5, k_chunk_inline309__ssa_v5))
                k_inv_inline385__ssa_v5: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398__ssa_v5, 0.0078125), 9.9999999999999995e-07)))
                k_inv_states_inline204__ssa_v6: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204__ssa_v5, k_inv_inline385__ssa_v5, [80, 0])
            qk_tids_inline247__ssa_v6: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247__ssa_v5, 5, qk_tid_h_inline245__ssa_v5)
            _submit_deps_buf_inline448__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
            _submit_deps_buf_inline323__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448__ssa_v6, 0, pl.array.get_element(q_tile_tids_inline280__rv_v2, 35))
            _submit_deps_buf_inline440__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323__ssa_v6, 1, pl.array.get_element(q_tile_tids_inline280__rv_v2, 36))
            _submit_deps_buf_inline316__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440__ssa_v6, 2, pl.array.get_element(q_tile_tids_inline280__rv_v2, 37))
            _submit_deps_buf_inline153__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316__ssa_v6, 3, pl.array.get_element(q_tile_tids_inline280__rv_v2, 38))
            _submit_deps_buf_inline339__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153__ssa_v6, 4, pl.array.get_element(q_tile_tids_inline280__rv_v2, 39))
            _submit_deps_buf_inline151__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339__ssa_v6, 5, pl.array.get_element(q_tile_tids_inline280__rv_v2, 40))
            _submit_deps_buf_inline353__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151__ssa_v6, 6, pl.array.get_element(q_tile_tids_inline280__rv_v2, 41))
            _submit_deps_buf_inline473__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353__ssa_v6, 7, pl.array.get_element(q_tile_tids_inline280__rv_v2, 42))
            _submit_deps_buf_inline148__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473__ssa_v6, 8, pl.array.get_element(q_tile_tids_inline280__rv_v2, 43))
            _submit_deps_buf_inline348__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148__ssa_v6, 9, pl.array.get_element(q_tile_tids_inline280__rv_v2, 44))
            _submit_deps_buf_inline244__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348__ssa_v6, 10, pl.array.get_element(k_tile_tids_inline174__rv_v2, 5))
            _submit_deps_buf_inline166__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244__ssa_v6, 11, pl.array.get_element(k_tile_tids_inline174__rv_v2, 6))
            _submit_deps_buf_inline176__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166__ssa_v6, 12, pl.array.get_element(k_tile_tids_inline174__rv_v2, 7))
            _submit_deps_buf_inline219__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176__ssa_v6, 13, pl.array.get_element(k_tile_tids_inline174__rv_v2, 8))
            _submit_deps_buf_inline144__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219__ssa_v6, 14, pl.array.get_element(k_tile_tids_inline174__rv_v2, 9))
            _submit_deps_buf_inline223__ssa_v6: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144__ssa_v6, 15, rms_tid_inline314__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223__ssa_v6]) as qk_tid_h_inline245__ssa_v6:
                q_slice_inline388__ssa_v6: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248__rv_v5, [16, 640], [0, 3840]), inv_rms_col_inline165__ssa_v0)
                q_chunk_inline389__ssa_v6: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388__ssa_v6, [80, 128])
                q_g_inline393__ssa_v6: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389__ssa_v6, q_norm_w_inline277__ssa_v0)
                q_proj_norm_inline375__ssa_v7: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375__ssa_v6, pl.tensor.reshape(q_g_inline393__ssa_v6, [16, 640]), [0, 3840])
                q_ss_inline312__ssa_v6: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389__ssa_v6, q_chunk_inline389__ssa_v6))
                q_inv_inline200__ssa_v6: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312__ssa_v6, 0.0078125), 9.9999999999999995e-07)))
                q_inv_states_inline168__ssa_v7: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168__ssa_v6, q_inv_inline200__ssa_v6, [480, 0])
                k_chunk_inline309__ssa_v6: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383__rv_v3, [16, 128], [0, 768]), inv_rms_col_inline165__ssa_v0)
                k_g_inline397__ssa_v6: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309__ssa_v6, k_norm_w_inline253__ssa_v0)
                k_proj_norm_inline171__ssa_v7: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171__ssa_v6, k_g_inline397__ssa_v6, [0, 768])
                k_ss_inline398__ssa_v6: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309__ssa_v6, k_chunk_inline309__ssa_v6))
                k_inv_inline385__ssa_v6: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398__ssa_v6, 0.0078125), 9.9999999999999995e-07)))
                k_inv_states_inline204__ssa_v7: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204__ssa_v6, k_inv_inline385__ssa_v6, [96, 0])
            qk_tids_inline247__ssa_v7: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247__ssa_v6, 6, qk_tid_h_inline245__ssa_v6)
            _submit_deps_buf_inline448__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
            _submit_deps_buf_inline323__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448__ssa_v7, 0, pl.array.get_element(q_tile_tids_inline280__rv_v2, 40))
            _submit_deps_buf_inline440__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323__ssa_v7, 1, pl.array.get_element(q_tile_tids_inline280__rv_v2, 41))
            _submit_deps_buf_inline316__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440__ssa_v7, 2, pl.array.get_element(q_tile_tids_inline280__rv_v2, 42))
            _submit_deps_buf_inline153__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316__ssa_v7, 3, pl.array.get_element(q_tile_tids_inline280__rv_v2, 43))
            _submit_deps_buf_inline339__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153__ssa_v7, 4, pl.array.get_element(q_tile_tids_inline280__rv_v2, 44))
            _submit_deps_buf_inline151__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339__ssa_v7, 5, pl.array.get_element(q_tile_tids_inline280__rv_v2, 45))
            _submit_deps_buf_inline353__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151__ssa_v7, 6, pl.array.get_element(q_tile_tids_inline280__rv_v2, 46))
            _submit_deps_buf_inline473__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353__ssa_v7, 7, pl.array.get_element(q_tile_tids_inline280__rv_v2, 47))
            _submit_deps_buf_inline148__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473__ssa_v7, 8, pl.array.get_element(q_tile_tids_inline280__rv_v2, 48))
            _submit_deps_buf_inline348__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148__ssa_v7, 9, pl.array.get_element(q_tile_tids_inline280__rv_v2, 49))
            _submit_deps_buf_inline244__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348__ssa_v7, 10, pl.array.get_element(k_tile_tids_inline174__rv_v2, 5))
            _submit_deps_buf_inline166__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244__ssa_v7, 11, pl.array.get_element(k_tile_tids_inline174__rv_v2, 6))
            _submit_deps_buf_inline176__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166__ssa_v7, 12, pl.array.get_element(k_tile_tids_inline174__rv_v2, 7))
            _submit_deps_buf_inline219__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176__ssa_v7, 13, pl.array.get_element(k_tile_tids_inline174__rv_v2, 8))
            _submit_deps_buf_inline144__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219__ssa_v7, 14, pl.array.get_element(k_tile_tids_inline174__rv_v2, 9))
            _submit_deps_buf_inline223__ssa_v7: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144__ssa_v7, 15, rms_tid_inline314__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223__ssa_v7]) as qk_tid_h_inline245__ssa_v7:
                q_slice_inline388__ssa_v7: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248__rv_v5, [16, 640], [0, 4480]), inv_rms_col_inline165__ssa_v0)
                q_chunk_inline389__ssa_v7: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388__ssa_v7, [80, 128])
                q_g_inline393__ssa_v7: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389__ssa_v7, q_norm_w_inline277__ssa_v0)
                q_proj_norm_inline375__ssa_v8: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375__ssa_v7, pl.tensor.reshape(q_g_inline393__ssa_v7, [16, 640]), [0, 4480])
                q_ss_inline312__ssa_v7: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389__ssa_v7, q_chunk_inline389__ssa_v7))
                q_inv_inline200__ssa_v7: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312__ssa_v7, 0.0078125), 9.9999999999999995e-07)))
                q_inv_states_inline168__ssa_v8: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168__ssa_v7, q_inv_inline200__ssa_v7, [560, 0])
                k_chunk_inline309__ssa_v7: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383__rv_v3, [16, 128], [0, 896]), inv_rms_col_inline165__ssa_v0)
                k_g_inline397__ssa_v7: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309__ssa_v7, k_norm_w_inline253__ssa_v0)
                k_proj_norm_inline171__ssa_v8: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171__ssa_v7, k_g_inline397__ssa_v7, [0, 896])
                k_ss_inline398__ssa_v7: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309__ssa_v7, k_chunk_inline309__ssa_v7))
                k_inv_inline385__ssa_v7: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398__ssa_v7, 0.0078125), 9.9999999999999995e-07)))
                k_inv_states_inline204__ssa_v8: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204__ssa_v7, k_inv_inline385__ssa_v7, [112, 0])
            qk_tids_inline247__ssa_v8: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247__ssa_v7, 7, qk_tid_h_inline245__ssa_v7)
            _submit_deps_buf_inline302__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.create(19, dtype=pl.TASK_ID)
            _submit_deps_buf_inline320__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline302__ssa_v0, 0, pl.array.get_element(qk_tids_inline247__ssa_v8, 0))
            _submit_deps_buf_inline399__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline320__ssa_v0, 1, pl.array.get_element(qk_tids_inline247__ssa_v8, 1))
            _submit_deps_buf_inline400__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline399__ssa_v0, 2, pl.array.get_element(qk_tids_inline247__ssa_v8, 2))
            _submit_deps_buf_inline150__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline400__ssa_v0, 3, pl.array.get_element(qk_tids_inline247__ssa_v8, 3))
            _submit_deps_buf_inline283__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline150__ssa_v0, 4, pl.array.get_element(qk_tids_inline247__ssa_v8, 4))
            _submit_deps_buf_inline254__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline283__ssa_v0, 5, pl.array.get_element(qk_tids_inline247__ssa_v8, 5))
            _submit_deps_buf_inline437__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline254__ssa_v0, 6, pl.array.get_element(qk_tids_inline247__ssa_v8, 6))
            _submit_deps_buf_inline246__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline437__ssa_v0, 7, pl.array.get_element(qk_tids_inline247__ssa_v8, 7))
            _submit_deps_buf_inline313__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline246__ssa_v0, 8, rms_tid_inline314__ssa_v0)
            _submit_deps_buf_inline231__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline313__ssa_v0, 9, pl.array.get_element(v_tile_tids_inline438__rv_v2, 0))
            _submit_deps_buf_inline402__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline231__ssa_v0, 10, pl.array.get_element(v_tile_tids_inline438__rv_v2, 1))
            _submit_deps_buf_inline476__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline402__ssa_v0, 11, pl.array.get_element(v_tile_tids_inline438__rv_v2, 2))
            _submit_deps_buf_inline331__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline476__ssa_v0, 12, pl.array.get_element(v_tile_tids_inline438__rv_v2, 3))
            _submit_deps_buf_inline341__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline331__ssa_v0, 13, pl.array.get_element(v_tile_tids_inline438__rv_v2, 4))
            _submit_deps_buf_inline405__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline341__ssa_v0, 14, pl.array.get_element(v_tile_tids_inline438__rv_v2, 5))
            _submit_deps_buf_inline413__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline405__ssa_v0, 15, pl.array.get_element(v_tile_tids_inline438__rv_v2, 6))
            _submit_deps_buf_inline406__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline413__ssa_v0, 16, pl.array.get_element(v_tile_tids_inline438__rv_v2, 7))
            _submit_deps_buf_inline408__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline406__ssa_v0, 17, pl.array.get_element(v_tile_tids_inline438__rv_v2, 8))
            _submit_deps_buf_inline410__ssa_v0: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline408__ssa_v0, 18, pl.array.get_element(v_tile_tids_inline438__rv_v2, 9))
            with pl.spmd(32, name_hint="rope_qkv_spmd", deps=[_submit_deps_buf_inline410__ssa_v0]) as rope_tid_inline411__ssa_v0:
                rope_core_inline414__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                for it_inline149__idx_v0, (all_q_padded_inline257__iter_v1, k_cache__iter_v3, v_cache__iter_v3) in pl.pipeline(4, stage=2, init_values=(all_q_padded_inline257__ssa_v0, k_cache__ssa_v0, v_cache__ssa_v0)):
                    g_idx_inline418__ssa_v0: pl.Scalar[pl.INDEX] = rope_core_inline414__ssa_v0 * 4 + it_inline149__idx_v0
                    ki_inline232__ssa_v0: pl.Scalar[pl.INDEX] = g_idx_inline418__ssa_v0 // 16
                    b_inline201__ssa_v0: pl.Scalar[pl.INDEX] = g_idx_inline418__ssa_v0 % 16
                    ctx_len_inline356__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [b_inline201__ssa_v0])
                    inv_rms_b_inline420__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(inv_rms_states_inline449__ssa_v1, [b_inline201__ssa_v0, 0])
                    pos_inline327__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(ctx_len_inline356__ssa_v0, pl.INDEX) - 1
                    wr_slot_inline239__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slot_mapping__ssa_v0, [b_inline201__ssa_v0]), pl.INDEX)
                    wr_slot_block_inline421__ssa_v0: pl.Scalar[pl.INDEX] = wr_slot_inline239__ssa_v0 // 128
                    wr_slot_offset_inline295__ssa_v0: pl.Scalar[pl.INDEX] = wr_slot_inline239__ssa_v0 - wr_slot_block_inline421__ssa_v0 * 128
                    cos_lo_inline306__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos__ssa_v0, [1, 64], [pos_inline327__ssa_v0, 0])
                    cos_hi_inline456__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos__ssa_v0, [1, 64], [pos_inline327__ssa_v0, 64])
                    sin_lo_inline424__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin__ssa_v0, [1, 64], [pos_inline327__ssa_v0, 0])
                    sin_hi_inline427__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin__ssa_v0, [1, 64], [pos_inline327__ssa_v0, 64])
                    kv_col_inline233__ssa_v0: pl.Scalar[pl.INDEX] = ki_inline232__ssa_v0 * 128
                    k_inv_b_inline224__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(k_inv_states_inline204__ssa_v8, [ki_inline232__ssa_v0 * 16 + b_inline201__ssa_v0, 0])
                    k_full_inline428__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(pl.tensor.slice(k_proj_norm_inline171__ssa_v8, [1, 128], [b_inline201__ssa_v0, kv_col_inline233__ssa_v0]), k_inv_b_inline224__ssa_v0)
                    k_lo_inline382__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full_inline428__ssa_v0, [1, 64], [0, 0])
                    k_hi_inline431__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full_inline428__ssa_v0, [1, 64], [0, 64])
                    rot_lo_inline363__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.sub(pl.tensor.col_expand_mul(k_lo_inline382__ssa_v0, cos_lo_inline306__ssa_v0), pl.tensor.col_expand_mul(k_hi_inline431__ssa_v0, sin_lo_inline424__ssa_v0))
                    rot_hi_inline298__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(k_hi_inline431__ssa_v0, cos_hi_inline456__ssa_v0), pl.tensor.col_expand_mul(k_lo_inline382__ssa_v0, sin_hi_inline427__ssa_v0))
                    cache_row_inline390__ssa_v0: pl.Scalar[pl.INDEX] = layer_cache_base_inline156__ssa_v0 + (wr_slot_block_inline421__ssa_v0 * 8 + ki_inline232__ssa_v0) * 128 + wr_slot_offset_inline295__ssa_v0
                    k_cache__ssa_v5: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache__iter_v3, pl.tensor.cast(rot_lo_inline363__ssa_v0, target_type=pl.BF16, mode='round'), [cache_row_inline390__ssa_v0, 0])
                    k_cache__ssa_v6: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache__ssa_v5, pl.tensor.cast(rot_hi_inline298__ssa_v0, target_type=pl.BF16, mode='round'), [cache_row_inline390__ssa_v0, 64])
                    v_row_bf16_inline147__ssa_v0: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(pl.tensor.muls(pl.tensor.slice(v_proj_inline241__rv_v3, [1, 128], [b_inline201__ssa_v0, ki_inline232__ssa_v0 * 128]), inv_rms_b_inline420__ssa_v0), target_type=pl.BF16, mode='round')
                    v_cache__ssa_v5: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(v_cache__iter_v3, v_row_bf16_inline147__ssa_v0, [cache_row_inline390__ssa_v0, 0])
                    q_base_inline435__ssa_v0: pl.Scalar[pl.INDEX] = ki_inline232__ssa_v0 * 5
                    q_pad_row0_inline251__ssa_v0: pl.Scalar[pl.INDEX] = b_inline201__ssa_v0 * 128 + ki_inline232__ssa_v0 * 16
                    q_inv_base_inline261__ssa_v0: pl.Scalar[pl.INDEX] = ki_inline232__ssa_v0 * 80 + b_inline201__ssa_v0 * 5
                    for qj_inline432__idx_v0, (all_q_padded_inline257__iter_v3,) in pl.range(5, init_values=(all_q_padded_inline257__iter_v1,)):
                        q_inv_bj_inline433__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(q_inv_states_inline168__ssa_v8, [q_inv_base_inline261__ssa_v0 + qj_inline432__idx_v0, 0])
                        q_head_inline407__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(pl.tensor.slice(q_proj_norm_inline375__ssa_v8, [1, 128], [b_inline201__ssa_v0, (q_base_inline435__ssa_v0 + qj_inline432__idx_v0) * 128]), q_inv_bj_inline433__ssa_v0)
                        q_lo_inline242__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_head_inline407__ssa_v0, [1, 64], [0, 0])
                        q_hi_inline436__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_head_inline407__ssa_v0, [1, 64], [0, 64])
                        q_rot_lo_inline269__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.sub(pl.tensor.col_expand_mul(q_lo_inline242__ssa_v0, cos_lo_inline306__ssa_v0), pl.tensor.col_expand_mul(q_hi_inline436__ssa_v0, sin_lo_inline424__ssa_v0))
                        q_rot_hi_inline471__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(q_hi_inline436__ssa_v0, cos_hi_inline456__ssa_v0), pl.tensor.col_expand_mul(q_lo_inline242__ssa_v0, sin_hi_inline427__ssa_v0))
                        all_q_padded_inline257__ssa_v5: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded_inline257__iter_v3, pl.tensor.cast(q_rot_lo_inline269__ssa_v0, target_type=pl.BF16, mode='round'), [q_pad_row0_inline251__ssa_v0 + qj_inline432__idx_v0, 0])
                        all_q_padded_inline257__ssa_v6: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded_inline257__ssa_v5, pl.tensor.cast(q_rot_hi_inline471__ssa_v0, target_type=pl.BF16, mode='round'), [q_pad_row0_inline251__ssa_v0 + qj_inline432__idx_v0, 64])
                        all_q_padded_inline257__rv_v4: pl.Tensor[[2048, 128], pl.BF16] = pl.yield_(all_q_padded_inline257__ssa_v6)
                    q_pad_zero_inline336__ssa_v0: pl.Tensor[[11, 128], pl.BF16] = pl.tensor.cast(pl.tensor.full([11, 128], dtype=pl.FP32, value=0.0), target_type=pl.BF16, mode='round')
                    all_q_padded_inline257__ssa_v7: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded_inline257__rv_v4, q_pad_zero_inline336__ssa_v0, [q_pad_row0_inline251__ssa_v0 + 5, 0])
                    all_q_padded_inline257__rv_v2, k_cache__rv_v4, v_cache__rv_v4 = pl.yield_(all_q_padded_inline257__ssa_v7, k_cache__ssa_v6, v_cache__ssa_v5)
            rope_grp_tids_inline266__ssa_v1: pl.Array[2, pl.TASK_ID] = pl.array.update_element(rope_grp_tids_inline266__ssa_v0, 0, rope_tid_inline411__ssa_v0)
            down_acc_all_inline439__ssa_v0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            gate_acc_all_inline234__ssa_v0: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.create([16, 17408], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            up_acc_all_inline284__ssa_v0: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.create([16, 17408], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            _submit_deps_buf_inline367__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_inline396__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline367__ssa_v0, 0, pl.array.get_element(carry_tids__rv_v2, 0))
            _submit_deps_buf_inline442__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline396__ssa_v0, 1, pl.array.get_element(carry_tids__rv_v2, 1))
            _submit_deps_buf_inline445__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline442__ssa_v0, 2, pl.array.get_element(carry_tids__rv_v2, 2))
            _submit_deps_buf_inline446__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline445__ssa_v0, 3, pl.array.get_element(carry_tids__rv_v2, 3))
            _submit_deps_buf_inline447__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline446__ssa_v0, 4, pl.array.get_element(carry_tids__rv_v2, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_seed", deps=[_submit_deps_buf_inline447__ssa_v0]) as seed_tid_inline417__ssa_v0:
                for nb_inline451__idx_v0, (down_acc_all_inline439__iter_v1,) in pl.pipeline(5, stage=2, init_values=(down_acc_all_inline439__ssa_v0,)):
                    n0_inline352__ssa_v3: pl.Scalar[pl.INDEX] = nb_inline451__idx_v0 * 1024
                    zero_inline273__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                    down_acc_all_inline439__ssa_v3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(down_acc_all_inline439__iter_v1, zero_inline273__ssa_v0, [0, n0_inline352__ssa_v3])
                    down_acc_all_inline439__rv_v2: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(down_acc_all_inline439__ssa_v3)
            _submit_deps_buf_inline452__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_inline453__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline452__ssa_v0, 0, pl.array.get_element(carry_tids__rv_v2, 0))
            _submit_deps_buf_inline209__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline453__ssa_v0, 1, pl.array.get_element(carry_tids__rv_v2, 1))
            _submit_deps_buf_inline426__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline209__ssa_v0, 2, pl.array.get_element(carry_tids__rv_v2, 2))
            _submit_deps_buf_inline366__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline426__ssa_v0, 3, pl.array.get_element(carry_tids__rv_v2, 3))
            _submit_deps_buf_inline454__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline366__ssa_v0, 4, pl.array.get_element(carry_tids__rv_v2, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_seed", deps=[_submit_deps_buf_inline454__ssa_v0]) as gate_seed_tid_inline198__ssa_v0:
                for nb_inline457__idx_v0, (gate_acc_all_inline234__iter_v1,) in pl.pipeline(17, stage=2, init_values=(gate_acc_all_inline234__ssa_v0,)):
                    n0_inline352__ssa_v4: pl.Scalar[pl.INDEX] = nb_inline457__idx_v0 * 1024
                    zero_inline273__ssa_v1: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                    gate_acc_all_inline234__ssa_v3: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(gate_acc_all_inline234__iter_v1, zero_inline273__ssa_v1, [0, n0_inline352__ssa_v4])
                    gate_acc_all_inline234__rv_v2: pl.Tensor[[16, 17408], pl.FP32] = pl.yield_(gate_acc_all_inline234__ssa_v3)
            _submit_deps_buf_inline458__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_inline460__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline458__ssa_v0, 0, pl.array.get_element(carry_tids__rv_v2, 0))
            _submit_deps_buf_inline192__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline460__ssa_v0, 1, pl.array.get_element(carry_tids__rv_v2, 1))
            _submit_deps_buf_inline191__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline192__ssa_v0, 2, pl.array.get_element(carry_tids__rv_v2, 2))
            _submit_deps_buf_inline189__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline191__ssa_v0, 3, pl.array.get_element(carry_tids__rv_v2, 3))
            _submit_deps_buf_inline419__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline189__ssa_v0, 4, pl.array.get_element(carry_tids__rv_v2, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_seed", deps=[_submit_deps_buf_inline419__ssa_v0]) as up_seed_tid_inline285__ssa_v0:
                for nb_inline374__idx_v0, (up_acc_all_inline284__iter_v1,) in pl.pipeline(17, stage=2, init_values=(up_acc_all_inline284__ssa_v0,)):
                    n0_inline352__ssa_v5: pl.Scalar[pl.INDEX] = nb_inline374__idx_v0 * 1024
                    zero_inline273__ssa_v2: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                    up_acc_all_inline284__ssa_v3: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(up_acc_all_inline284__iter_v1, zero_inline273__ssa_v2, [0, n0_inline352__ssa_v5])
                    up_acc_all_inline284__rv_v2: pl.Tensor[[16, 17408], pl.FP32] = pl.yield_(up_acc_all_inline284__ssa_v3)
            _submit_deps_buf_inline462__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
            _submit_deps_buf_inline237__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline462__ssa_v0, 0, work_tid_inline183__ssa_v0)
            _submit_deps_buf_inline443__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline237__ssa_v0, 1, pl.array.get_element(rope_grp_tids_inline266__ssa_v1, 0))
            with pl.spmd(24, name_hint="fa_fused_spmd", optimizations=[pl.split(pl.SplitMode.UP_DOWN)], deps=[_submit_deps_buf_inline443__ssa_v0]) as fa_tid_inline464__ssa_v0:
                fa_core_inline466__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                fa_total_blocks_inline368__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(fa_total_inline262__ssa_v0, [0, 0]), pl.INDEX)
                for fa_w_inline322__idx_v0, (all_cur_li_inline294__iter_v1, all_cur_mi_inline409__iter_v1, all_oi_tmp_inline318__iter_v1) in pl.range(fa_core_inline466__ssa_v0, fa_total_blocks_inline368__ssa_v0, 24, init_values=(all_cur_li_inline294__ssa_v0, all_cur_mi_inline409__ssa_v0, all_oi_tmp_inline318__ssa_v0)):
                    fa_enc_inline416__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(fa_work_table_inline281__ssa_v0, [fa_w_inline322__idx_v0, 0]), pl.INDEX)
                    fa_b_inline256__ssa_v0: pl.Scalar[pl.INDEX] = fa_enc_inline416__ssa_v0 // 32
                    fa_p_inline450__ssa_v0: pl.Scalar[pl.INDEX] = fa_enc_inline416__ssa_v0 % 32
                    fa_hg_inline469__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    fa_ctx_len_inline472__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [fa_b_inline256__ssa_v0])
                    sb_inline258__ssa_v0: pl.Scalar[pl.INDEX] = fa_p_inline450__ssa_v0
                    s0_inline220__ssa_v0: pl.Scalar[pl.INDEX] = sb_inline258__ssa_v0 * 128
                    valid_len_inline474__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.cast(fa_ctx_len_inline472__ssa_v0, pl.INDEX) - s0_inline220__ssa_v0, 128)
                    fa_pbid_inline243__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table__ssa_v0, [fa_b_inline256__ssa_v0 * max_blocks_per_seq_inline401__ssa_v0 + sb_inline258__ssa_v0]), pl.INDEX)
                    for gp_inline161__idx_v0, (all_cur_li_inline294__iter_v3, all_cur_mi_inline409__iter_v3, all_oi_tmp_inline318__iter_v3) in pl.pipeline(8, stage=2, init_values=(all_cur_li_inline294__iter_v1, all_cur_mi_inline409__iter_v1, all_oi_tmp_inline318__iter_v1)):
                        gi_inline477__ssa_v0: pl.Scalar[pl.INDEX] = fa_hg_inline469__ssa_v0 * 8 + gp_inline161__idx_v0
                        kvh_inline369__ssa_v0: pl.Scalar[pl.INDEX] = gi_inline477__ssa_v0
                        q_pad_row_g_inline289__ssa_v0: pl.Scalar[pl.INDEX] = fa_b_inline256__ssa_v0 * 128 + gi_inline477__ssa_v0 * 16
                        q_padded_inline259__ssa_v0: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.slice(all_q_padded_inline257__rv_v2, [16, 128], [q_pad_row_g_inline289__ssa_v0, 0])
                        g_base_inline478__ssa_v0: pl.Scalar[pl.INDEX] = (fa_b_inline256__ssa_v0 * 8 + gi_inline477__ssa_v0) * 512
                        cache_row_inline390__ssa_v1: pl.Scalar[pl.INDEX] = layer_cache_base_inline156__ssa_v0 + (fa_pbid_inline243__ssa_v0 * 8 + kvh_inline369__ssa_v0) * 128
                        k_tile_inline279__ssa_v0: pl.Tensor[[128, 128], pl.BF16] = pl.tensor.slice(k_cache__rv_v4, [128, 128], [cache_row_inline390__ssa_v1, 0])
                        raw_scores_inline475__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(q_padded_inline259__ssa_v0, k_tile_inline279__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        scores_scaled_inline260__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.muls(raw_scores_inline475__ssa_v0, 0.088388347648318433)
                        scores_valid_inline481__ssa_v0: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, valid_len], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(scores_scaled_inline260__ssa_v0, 5, valid_len_inline474__ssa_v0)
                        scores_inline482__ssa_v0: pl.Tensor[[16, 128], pl.FP32, pl.TensorView()] = pl.tensor.fillpad(scores_valid_inline481__ssa_v0, pad_value=pl.PadValue.min)
                        cur_mi_inline365__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_max(scores_inline482__ssa_v0)
                        exp_scores_inline199__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(scores_inline482__ssa_v0, cur_mi_inline365__ssa_v0))
                        cur_li_inline484__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(exp_scores_inline199__ssa_v0)
                        exp_scores_bf16_inline211__ssa_v0: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.cast(exp_scores_inline199__ssa_v0, target_type=pl.BF16, mode='round')
                        v_tile_inline143__ssa_v0: pl.Tensor[[128, 128], pl.BF16] = pl.tensor.slice(v_cache__rv_v4, [128, 128], [cache_row_inline390__ssa_v1, 0])
                        oi_tmp_inline470__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(exp_scores_bf16_inline211__ssa_v0, v_tile_inline143__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        oi_tmp_v1_inline216__ssa_v0: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(oi_tmp_inline470__ssa_v0, 5, 128)
                        all_oi_tmp_inline318__ssa_v5: pl.Tensor[[65536, 128], pl.FP32] = pl.tensor.assemble(all_oi_tmp_inline318__iter_v3, oi_tmp_v1_inline216__ssa_v0, [g_base_inline478__ssa_v0 + sb_inline258__ssa_v0 * 16, 0])
                        all_cur_mi_inline409__ssa_v5: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.assemble(all_cur_mi_inline409__iter_v3, cur_mi_inline365__ssa_v0, [g_base_inline478__ssa_v0 + sb_inline258__ssa_v0 * 16, 0])
                        all_cur_li_inline294__ssa_v5: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.assemble(all_cur_li_inline294__iter_v3, cur_li_inline484__ssa_v0, [g_base_inline478__ssa_v0 + sb_inline258__ssa_v0 * 16, 0])
                        all_cur_li_inline294__rv_v4, all_cur_mi_inline409__rv_v4, all_oi_tmp_inline318__rv_v4 = pl.yield_(all_cur_li_inline294__ssa_v5, all_cur_mi_inline409__ssa_v5, all_oi_tmp_inline318__ssa_v5)
                    all_cur_li_inline294__rv_v2, all_cur_mi_inline409__rv_v2, all_oi_tmp_inline318__rv_v2 = pl.yield_(all_cur_li_inline294__rv_v4, all_cur_mi_inline409__rv_v4, all_oi_tmp_inline318__rv_v4)
            with pl.spmd(48, name_hint="online_softmax_spmd", deps=[fa_tid_inline464__ssa_v0]) as attn_done_tid_inline142__ssa_v0:
                os_core_inline141__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                for os_spmd_idx_inline140__idx_v0, (attn_out_inline203__iter_v1,) in pl.range(os_core_inline141__ssa_v0, 128, 48, init_values=(attn_out_inline203__ssa_v0,)):
                    os_b_inline288__ssa_v0: pl.Scalar[pl.INDEX] = os_spmd_idx_inline140__idx_v0 // 8
                    os_gi_inline344__ssa_v0: pl.Scalar[pl.INDEX] = os_spmd_idx_inline140__idx_v0 % 8
                    os_ctx_len_inline139__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [os_b_inline288__ssa_v0])
                    os_ctx_blocks_inline138__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(os_ctx_len_inline139__ssa_v0, pl.INDEX) + 127) // 128
                    os_kvh_inline392__ssa_v0: pl.Scalar[pl.INDEX] = os_gi_inline344__ssa_v0
                    os_q_base_inline137__ssa_v0: pl.Scalar[pl.INDEX] = os_kvh_inline392__ssa_v0 * 5
                    os_g_base_inline136__ssa_v0: pl.Scalar[pl.INDEX] = (os_b_inline288__ssa_v0 * 8 + os_gi_inline344__ssa_v0) * 512
                    oi_inline357__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.slice(all_oi_tmp_inline318__rv_v2, [16, 128], [os_g_base_inline136__ssa_v0, 0])
                    mi_inline330__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(all_cur_mi_inline409__rv_v2, [16, 1], [os_g_base_inline136__ssa_v0, 0])
                    li_inline135__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(all_cur_li_inline294__rv_v2, [16, 1], [os_g_base_inline136__ssa_v0, 0])
                    for sb_inline134__idx_v0, (li_inline135__iter_v1, mi_inline330__iter_v1, oi_inline357__iter_v1) in pl.pipeline(1, os_ctx_blocks_inline138__ssa_v0, stage=2, init_values=(li_inline135__ssa_v0, mi_inline330__ssa_v0, oi_inline357__ssa_v0)):
                        rec_inline133__ssa_v0: pl.Scalar[pl.INDEX] = os_g_base_inline136__ssa_v0 + sb_inline134__idx_v0 * 16
                        oi_tmp_valid_inline132__ssa_v0: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_oi_tmp_inline318__rv_v2, [16, 128], [rec_inline133__ssa_v0, 0], [5, 128])
                        online_cur_mi_inline131__ssa_v0: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[5, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_cur_mi_inline409__rv_v2, [16, 1], [rec_inline133__ssa_v0, 0], [5, 1])
                        online_cur_li_inline175__ssa_v0: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[5, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_cur_li_inline294__rv_v2, [16, 1], [rec_inline133__ssa_v0, 0], [5, 1])
                        mi_new_inline130__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.maximum(mi_inline330__iter_v1, online_cur_mi_inline131__ssa_v0)
                        alpha_inline423__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline330__iter_v1, mi_new_inline130__ssa_v0))
                        beta_inline228__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(online_cur_mi_inline131__ssa_v0, mi_new_inline130__ssa_v0))
                        li_inline135__ssa_v3: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha_inline423__ssa_v0, li_inline135__iter_v1), pl.tensor.mul(beta_inline228__ssa_v0, online_cur_li_inline175__ssa_v0))
                        oi_inline357__ssa_v3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(oi_inline357__iter_v1, alpha_inline423__ssa_v0), pl.tensor.row_expand_mul(oi_tmp_valid_inline132__ssa_v0, beta_inline228__ssa_v0))
                        mi_inline330__ssa_v3: pl.Tensor[[16, 1], pl.FP32] = mi_new_inline130__ssa_v0
                        li_inline135__rv_v2, mi_inline330__rv_v2, oi_inline357__rv_v2 = pl.yield_(li_inline135__ssa_v3, mi_inline330__ssa_v3, oi_inline357__ssa_v3)
                    ctx_inline167__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_div(oi_inline357__rv_v2, li_inline135__rv_v2)
                    ctx_valid_inline128__ssa_v0: pl.Tensor[[5, 128], pl.FP32] = pl.tensor.slice(ctx_inline167__ssa_v0, [5, 128], [0, 0])
                    ctx_flat_bf16_inline127__ssa_v0: pl.Tensor[[1, 640], pl.BF16] = pl.tensor.cast(pl.tensor.reshape(ctx_valid_inline128__ssa_v0, [1, 640]), target_type=pl.BF16, mode='round')
                    attn_out_inline203__ssa_v3: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(attn_out_inline203__iter_v1, ctx_flat_bf16_inline127__ssa_v0, [os_b_inline288__ssa_v0, os_q_base_inline137__ssa_v0 * 128])
                    attn_out_inline203__rv_v2: pl.Tensor[[16, 5120], pl.BF16] = pl.yield_(attn_out_inline203__ssa_v3)
            attn_proj_fp32_inline468__ssa_v0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            post_norm_partial_inline126__ssa_v0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            mlp_norm_in_inline355__ssa_v0: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            inv_rms_tile_inline123__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            mlp_tile_inline122__ssa_v0: pl.Tensor[[16, 17408], pl.BF16] = pl.tensor.create([16, 17408], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            _submit_deps_buf_inline121__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            _submit_deps_buf_inline119__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline121__ssa_v0, 0, pl.array.get_element(carry_tids__rv_v2, 0))
            _submit_deps_buf_inline117__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline119__ssa_v0, 1, pl.array.get_element(carry_tids__rv_v2, 1))
            _submit_deps_buf_inline114__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline117__ssa_v0, 2, pl.array.get_element(carry_tids__rv_v2, 2))
            _submit_deps_buf_inline112__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline114__ssa_v0, 3, pl.array.get_element(carry_tids__rv_v2, 3))
            _submit_deps_buf_inline179__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline112__ssa_v0, 4, pl.array.get_element(carry_tids__rv_v2, 4))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_seed", deps=[_submit_deps_buf_inline179__ssa_v0]) as out_seed_tid_inline297__ssa_v0:
                for nb_inline274__idx_v0, (attn_proj_fp32_inline468__iter_v1,) in pl.pipeline(10, stage=2, init_values=(attn_proj_fp32_inline468__ssa_v0,)):
                    out_seed_n0_inline110__ssa_v0: pl.Scalar[pl.INDEX] = nb_inline274__idx_v0 * 512
                    out_zero_inline444__ssa_v0: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                    attn_proj_fp32_inline468__ssa_v3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_proj_fp32_inline468__iter_v1, out_zero_inline444__ssa_v0, [0, out_seed_n0_inline110__ssa_v0])
                    attn_proj_fp32_inline468__rv_v2: pl.Tensor[[16, 5120], pl.FP32] = pl.yield_(attn_proj_fp32_inline468__ssa_v3)
            silu_tids_inline109__ssa_v0: pl.Array[17, pl.TASK_ID] = pl.array.create(17, dtype=pl.TASK_ID)
            gate_tids_inline108__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
            up_tids_inline107__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
            cast_tids_inline391__ssa_v0: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
            out_tids_inline104__ssa_v0: pl.Array[50, pl.TASK_ID] = pl.array.create(50, dtype=pl.TASK_ID)
            for n_out_proj_inline103__idx_v0, (attn_proj_fp32_inline468__iter_v4, out_tids_inline104__iter_v1) in pl.parallel(10, init_values=(attn_proj_fp32_inline468__rv_v2, out_tids_inline104__ssa_v0)):
                n_op_inline102__ssa_v0: pl.Scalar[pl.INDEX] = n_out_proj_inline103__idx_v0 * 512
                for k_split_out_inline100__idx_v0, (attn_proj_fp32_inline468__iter_v6, out_tids_inline104__iter_v3) in pl.range(5, init_values=(attn_proj_fp32_inline468__iter_v4, out_tids_inline104__iter_v1)):
                    k_op_inline360__ssa_v0: pl.Scalar[pl.INDEX] = k_split_out_inline100__idx_v0 * 1024
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj", deps=[out_seed_tid_inline297__ssa_v0, attn_done_tid_inline142__ssa_v0]) as out_tid_inline155__ssa_v0:
                        out_a0_inline308__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(attn_out_inline203__rv_v2, [16, 64], [0, k_op_inline360__ssa_v0])
                        out_w0_inline98__ssa_v0: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wo__ssa_v0, [64, 512], [layer_hidden_base_inline217__ssa_v0 + k_op_inline360__ssa_v0, n_op_inline102__ssa_v0])
                        out_c_acc_inline97__ssa_v0: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.matmul(out_a0_inline308__ssa_v0, out_w0_inline98__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for out_lk_inline96__idx_v0, (out_c_acc_inline97__iter_v1,) in pl.pipeline(1, 16, stage=2, init_values=(out_c_acc_inline97__ssa_v0,)):
                            out_ks_off_inline94__ssa_v0: pl.Scalar[pl.INDEX] = out_lk_inline96__idx_v0 * 64
                            out_a_k_inline329__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(attn_out_inline203__rv_v2, [16, 64], [0, k_op_inline360__ssa_v0 + out_ks_off_inline94__ssa_v0])
                            out_w_k_inline92__ssa_v0: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wo__ssa_v0, [64, 512], [layer_hidden_base_inline217__ssa_v0 + k_op_inline360__ssa_v0 + out_ks_off_inline94__ssa_v0, n_op_inline102__ssa_v0])
                            out_c_acc_inline97__ssa_v3: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.matmul_acc(out_c_acc_inline97__iter_v1, out_a_k_inline329__ssa_v0, out_w_k_inline92__ssa_v0, a_trans=False, b_trans=False)
                            out_c_acc_inline97__rv_v2: pl.Tensor[[16, 512], pl.FP32] = pl.yield_(out_c_acc_inline97__ssa_v3)
                        attn_proj_fp32_inline468__ssa_v8: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_proj_fp32_inline468__iter_v6, out_c_acc_inline97__rv_v2, [0, n_op_inline102__ssa_v0], atomic=pl.AtomicType.Add)
                    out_tids_inline104__ssa_v5: pl.Array[50, pl.TASK_ID] = pl.array.update_element(out_tids_inline104__iter_v3, n_out_proj_inline103__idx_v0 * 5 + k_split_out_inline100__idx_v0, out_tid_inline155__ssa_v0)
                    attn_proj_fp32_inline468__rv_v7, out_tids_inline104__rv_v4 = pl.yield_(attn_proj_fp32_inline468__ssa_v8, out_tids_inline104__ssa_v5)
                attn_proj_fp32_inline468__rv_v5, out_tids_inline104__rv_v2 = pl.yield_(attn_proj_fp32_inline468__rv_v7, out_tids_inline104__rv_v4)
            _submit_deps_buf_inline88__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
            _submit_deps_buf_inline87__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline88__ssa_v0, 0, pl.array.get_element(out_tids_inline104__rv_v2, 0))
            _submit_deps_buf_inline86__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline87__ssa_v0, 1, pl.array.get_element(out_tids_inline104__rv_v2, 1))
            _submit_deps_buf_inline85__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline86__ssa_v0, 2, pl.array.get_element(out_tids_inline104__rv_v2, 2))
            _submit_deps_buf_inline84__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline85__ssa_v0, 3, pl.array.get_element(out_tids_inline104__rv_v2, 3))
            _submit_deps_buf_inline95__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline84__ssa_v0, 4, pl.array.get_element(out_tids_inline104__rv_v2, 4))
            _submit_deps_buf_inline93__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline95__ssa_v0, 5, pl.array.get_element(out_tids_inline104__rv_v2, 5))
            _submit_deps_buf_inline83__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline93__ssa_v0, 6, pl.array.get_element(out_tids_inline104__rv_v2, 6))
            _submit_deps_buf_inline82__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline83__ssa_v0, 7, pl.array.get_element(out_tids_inline104__rv_v2, 7))
            _submit_deps_buf_inline463__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline82__ssa_v0, 8, pl.array.get_element(out_tids_inline104__rv_v2, 8))
            _submit_deps_buf_inline81__ssa_v0: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline463__ssa_v0, 9, pl.array.get_element(out_tids_inline104__rv_v2, 9))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_rms_cast", deps=[_submit_deps_buf_inline81__ssa_v0]) as cast_tid_k_inline79__ssa_v0:
                for kb_inline77__idx_v0, (k0_v1_inline395__iter_v8, mlp_norm_in_inline355__iter_v1, post_norm_partial_inline126__iter_v1) in pl.pipeline(4, stage=2, init_values=(896, mlp_norm_in_inline355__ssa_v0, post_norm_partial_inline126__ssa_v0)):
                    k0_v1_inline395__ssa_v10: pl.Scalar[pl.INDEX] = kb_inline77__idx_v0 * 256
                    attn_chunk_inline173__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32_inline468__rv_v5, [16, 256], [0, k0_v1_inline395__ssa_v10])
                    hidden_chunk_inline351__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur__rv_v2, [16, 256], [0, k0_v1_inline395__ssa_v10])
                    resid_fp32_inline286__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk_inline173__ssa_v0, hidden_chunk_inline351__ssa_v0)
                    post_norm_partial_inline126__ssa_v3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(post_norm_partial_inline126__iter_v1, resid_fp32_inline286__ssa_v0, [0, k0_v1_inline395__ssa_v10])
                    post_gamma_inline80__ssa_v0: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight__ssa_v0, [1, 256], [0, k0_v1_inline395__ssa_v10])
                    mlp_norm_in_inline355__ssa_v3: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(mlp_norm_in_inline355__iter_v1, pl.tensor.cast(pl.tensor.col_expand_mul(resid_fp32_inline286__ssa_v0, post_gamma_inline80__ssa_v0), target_type=pl.BF16, mode='round'), [0, k0_v1_inline395__ssa_v10])
                    k0_v1_inline395__rv_v9, mlp_norm_in_inline355__rv_v2, post_norm_partial_inline126__rv_v2 = pl.yield_(k0_v1_inline395__ssa_v10, mlp_norm_in_inline355__ssa_v3, post_norm_partial_inline126__ssa_v3)
            cast_tids_inline391__ssa_v1: pl.Array[5, pl.TASK_ID] = pl.array.update_element(cast_tids_inline391__ssa_v0, 0, cast_tid_k_inline79__ssa_v0)
            _submit_deps_buf_inline88__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
            _submit_deps_buf_inline87__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline88__ssa_v1, 0, pl.array.get_element(out_tids_inline104__rv_v2, 10))
            _submit_deps_buf_inline86__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline87__ssa_v1, 1, pl.array.get_element(out_tids_inline104__rv_v2, 11))
            _submit_deps_buf_inline85__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline86__ssa_v1, 2, pl.array.get_element(out_tids_inline104__rv_v2, 12))
            _submit_deps_buf_inline84__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline85__ssa_v1, 3, pl.array.get_element(out_tids_inline104__rv_v2, 13))
            _submit_deps_buf_inline95__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline84__ssa_v1, 4, pl.array.get_element(out_tids_inline104__rv_v2, 14))
            _submit_deps_buf_inline93__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline95__ssa_v1, 5, pl.array.get_element(out_tids_inline104__rv_v2, 15))
            _submit_deps_buf_inline83__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline93__ssa_v1, 6, pl.array.get_element(out_tids_inline104__rv_v2, 16))
            _submit_deps_buf_inline82__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline83__ssa_v1, 7, pl.array.get_element(out_tids_inline104__rv_v2, 17))
            _submit_deps_buf_inline463__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline82__ssa_v1, 8, pl.array.get_element(out_tids_inline104__rv_v2, 18))
            _submit_deps_buf_inline81__ssa_v1: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline463__ssa_v1, 9, pl.array.get_element(out_tids_inline104__rv_v2, 19))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_rms_cast", deps=[_submit_deps_buf_inline81__ssa_v1]) as cast_tid_k_inline79__ssa_v1:
                for kb_inline77__idx_v1, (k0_v1_inline395__iter_v11, mlp_norm_in_inline355__iter_v4, post_norm_partial_inline126__iter_v4) in pl.pipeline(4, stage=2, init_values=(k0_v1_inline395__rv_v9, mlp_norm_in_inline355__rv_v2, post_norm_partial_inline126__rv_v2)):
                    k0_v1_inline395__ssa_v13: pl.Scalar[pl.INDEX] = kb_inline77__idx_v1 * 256 + 1024
                    attn_chunk_inline173__ssa_v1: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32_inline468__rv_v5, [16, 256], [0, k0_v1_inline395__ssa_v13])
                    hidden_chunk_inline351__ssa_v1: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur__rv_v2, [16, 256], [0, k0_v1_inline395__ssa_v13])
                    resid_fp32_inline286__ssa_v1: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk_inline173__ssa_v1, hidden_chunk_inline351__ssa_v1)
                    post_norm_partial_inline126__ssa_v6: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(post_norm_partial_inline126__iter_v4, resid_fp32_inline286__ssa_v1, [0, k0_v1_inline395__ssa_v13])
                    post_gamma_inline80__ssa_v1: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight__ssa_v0, [1, 256], [0, k0_v1_inline395__ssa_v13])
                    mlp_norm_in_inline355__ssa_v6: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(mlp_norm_in_inline355__iter_v4, pl.tensor.cast(pl.tensor.col_expand_mul(resid_fp32_inline286__ssa_v1, post_gamma_inline80__ssa_v1), target_type=pl.BF16, mode='round'), [0, k0_v1_inline395__ssa_v13])
                    k0_v1_inline395__rv_v12, mlp_norm_in_inline355__rv_v5, post_norm_partial_inline126__rv_v5 = pl.yield_(k0_v1_inline395__ssa_v13, mlp_norm_in_inline355__ssa_v6, post_norm_partial_inline126__ssa_v6)
            cast_tids_inline391__ssa_v2: pl.Array[5, pl.TASK_ID] = pl.array.update_element(cast_tids_inline391__ssa_v1, 1, cast_tid_k_inline79__ssa_v1)
            _submit_deps_buf_inline88__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
            _submit_deps_buf_inline87__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline88__ssa_v2, 0, pl.array.get_element(out_tids_inline104__rv_v2, 20))
            _submit_deps_buf_inline86__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline87__ssa_v2, 1, pl.array.get_element(out_tids_inline104__rv_v2, 21))
            _submit_deps_buf_inline85__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline86__ssa_v2, 2, pl.array.get_element(out_tids_inline104__rv_v2, 22))
            _submit_deps_buf_inline84__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline85__ssa_v2, 3, pl.array.get_element(out_tids_inline104__rv_v2, 23))
            _submit_deps_buf_inline95__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline84__ssa_v2, 4, pl.array.get_element(out_tids_inline104__rv_v2, 24))
            _submit_deps_buf_inline93__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline95__ssa_v2, 5, pl.array.get_element(out_tids_inline104__rv_v2, 25))
            _submit_deps_buf_inline83__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline93__ssa_v2, 6, pl.array.get_element(out_tids_inline104__rv_v2, 26))
            _submit_deps_buf_inline82__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline83__ssa_v2, 7, pl.array.get_element(out_tids_inline104__rv_v2, 27))
            _submit_deps_buf_inline463__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline82__ssa_v2, 8, pl.array.get_element(out_tids_inline104__rv_v2, 28))
            _submit_deps_buf_inline81__ssa_v2: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline463__ssa_v2, 9, pl.array.get_element(out_tids_inline104__rv_v2, 29))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_rms_cast", deps=[_submit_deps_buf_inline81__ssa_v2]) as cast_tid_k_inline79__ssa_v2:
                for kb_inline77__idx_v2, (k0_v1_inline395__iter_v14, mlp_norm_in_inline355__iter_v7, post_norm_partial_inline126__iter_v7) in pl.pipeline(4, stage=2, init_values=(k0_v1_inline395__rv_v12, mlp_norm_in_inline355__rv_v5, post_norm_partial_inline126__rv_v5)):
                    k0_v1_inline395__ssa_v16: pl.Scalar[pl.INDEX] = kb_inline77__idx_v2 * 256 + 2048
                    attn_chunk_inline173__ssa_v2: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32_inline468__rv_v5, [16, 256], [0, k0_v1_inline395__ssa_v16])
                    hidden_chunk_inline351__ssa_v2: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur__rv_v2, [16, 256], [0, k0_v1_inline395__ssa_v16])
                    resid_fp32_inline286__ssa_v2: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk_inline173__ssa_v2, hidden_chunk_inline351__ssa_v2)
                    post_norm_partial_inline126__ssa_v9: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(post_norm_partial_inline126__iter_v7, resid_fp32_inline286__ssa_v2, [0, k0_v1_inline395__ssa_v16])
                    post_gamma_inline80__ssa_v2: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight__ssa_v0, [1, 256], [0, k0_v1_inline395__ssa_v16])
                    mlp_norm_in_inline355__ssa_v9: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(mlp_norm_in_inline355__iter_v7, pl.tensor.cast(pl.tensor.col_expand_mul(resid_fp32_inline286__ssa_v2, post_gamma_inline80__ssa_v2), target_type=pl.BF16, mode='round'), [0, k0_v1_inline395__ssa_v16])
                    k0_v1_inline395__rv_v15, mlp_norm_in_inline355__rv_v8, post_norm_partial_inline126__rv_v8 = pl.yield_(k0_v1_inline395__ssa_v16, mlp_norm_in_inline355__ssa_v9, post_norm_partial_inline126__ssa_v9)
            cast_tids_inline391__ssa_v3: pl.Array[5, pl.TASK_ID] = pl.array.update_element(cast_tids_inline391__ssa_v2, 2, cast_tid_k_inline79__ssa_v2)
            _submit_deps_buf_inline88__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
            _submit_deps_buf_inline87__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline88__ssa_v3, 0, pl.array.get_element(out_tids_inline104__rv_v2, 30))
            _submit_deps_buf_inline86__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline87__ssa_v3, 1, pl.array.get_element(out_tids_inline104__rv_v2, 31))
            _submit_deps_buf_inline85__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline86__ssa_v3, 2, pl.array.get_element(out_tids_inline104__rv_v2, 32))
            _submit_deps_buf_inline84__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline85__ssa_v3, 3, pl.array.get_element(out_tids_inline104__rv_v2, 33))
            _submit_deps_buf_inline95__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline84__ssa_v3, 4, pl.array.get_element(out_tids_inline104__rv_v2, 34))
            _submit_deps_buf_inline93__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline95__ssa_v3, 5, pl.array.get_element(out_tids_inline104__rv_v2, 35))
            _submit_deps_buf_inline83__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline93__ssa_v3, 6, pl.array.get_element(out_tids_inline104__rv_v2, 36))
            _submit_deps_buf_inline82__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline83__ssa_v3, 7, pl.array.get_element(out_tids_inline104__rv_v2, 37))
            _submit_deps_buf_inline463__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline82__ssa_v3, 8, pl.array.get_element(out_tids_inline104__rv_v2, 38))
            _submit_deps_buf_inline81__ssa_v3: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline463__ssa_v3, 9, pl.array.get_element(out_tids_inline104__rv_v2, 39))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_rms_cast", deps=[_submit_deps_buf_inline81__ssa_v3]) as cast_tid_k_inline79__ssa_v3:
                for kb_inline77__idx_v3, (k0_v1_inline395__iter_v17, mlp_norm_in_inline355__iter_v10, post_norm_partial_inline126__iter_v10) in pl.pipeline(4, stage=2, init_values=(k0_v1_inline395__rv_v15, mlp_norm_in_inline355__rv_v8, post_norm_partial_inline126__rv_v8)):
                    k0_v1_inline395__ssa_v19: pl.Scalar[pl.INDEX] = kb_inline77__idx_v3 * 256 + 3072
                    attn_chunk_inline173__ssa_v3: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32_inline468__rv_v5, [16, 256], [0, k0_v1_inline395__ssa_v19])
                    hidden_chunk_inline351__ssa_v3: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur__rv_v2, [16, 256], [0, k0_v1_inline395__ssa_v19])
                    resid_fp32_inline286__ssa_v3: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk_inline173__ssa_v3, hidden_chunk_inline351__ssa_v3)
                    post_norm_partial_inline126__ssa_v12: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(post_norm_partial_inline126__iter_v10, resid_fp32_inline286__ssa_v3, [0, k0_v1_inline395__ssa_v19])
                    post_gamma_inline80__ssa_v3: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight__ssa_v0, [1, 256], [0, k0_v1_inline395__ssa_v19])
                    mlp_norm_in_inline355__ssa_v12: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(mlp_norm_in_inline355__iter_v10, pl.tensor.cast(pl.tensor.col_expand_mul(resid_fp32_inline286__ssa_v3, post_gamma_inline80__ssa_v3), target_type=pl.BF16, mode='round'), [0, k0_v1_inline395__ssa_v19])
                    k0_v1_inline395__rv_v18, mlp_norm_in_inline355__rv_v11, post_norm_partial_inline126__rv_v11 = pl.yield_(k0_v1_inline395__ssa_v19, mlp_norm_in_inline355__ssa_v12, post_norm_partial_inline126__ssa_v12)
            cast_tids_inline391__ssa_v4: pl.Array[5, pl.TASK_ID] = pl.array.update_element(cast_tids_inline391__ssa_v3, 3, cast_tid_k_inline79__ssa_v3)
            _submit_deps_buf_inline88__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
            _submit_deps_buf_inline87__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline88__ssa_v4, 0, pl.array.get_element(out_tids_inline104__rv_v2, 40))
            _submit_deps_buf_inline86__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline87__ssa_v4, 1, pl.array.get_element(out_tids_inline104__rv_v2, 41))
            _submit_deps_buf_inline85__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline86__ssa_v4, 2, pl.array.get_element(out_tids_inline104__rv_v2, 42))
            _submit_deps_buf_inline84__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline85__ssa_v4, 3, pl.array.get_element(out_tids_inline104__rv_v2, 43))
            _submit_deps_buf_inline95__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline84__ssa_v4, 4, pl.array.get_element(out_tids_inline104__rv_v2, 44))
            _submit_deps_buf_inline93__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline95__ssa_v4, 5, pl.array.get_element(out_tids_inline104__rv_v2, 45))
            _submit_deps_buf_inline83__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline93__ssa_v4, 6, pl.array.get_element(out_tids_inline104__rv_v2, 46))
            _submit_deps_buf_inline82__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline83__ssa_v4, 7, pl.array.get_element(out_tids_inline104__rv_v2, 47))
            _submit_deps_buf_inline463__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline82__ssa_v4, 8, pl.array.get_element(out_tids_inline104__rv_v2, 48))
            _submit_deps_buf_inline81__ssa_v4: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline463__ssa_v4, 9, pl.array.get_element(out_tids_inline104__rv_v2, 49))
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_rms_cast", deps=[_submit_deps_buf_inline81__ssa_v4]) as cast_tid_k_inline79__ssa_v4:
                for kb_inline77__idx_v4, (k0_v1_inline395__iter_v20, mlp_norm_in_inline355__iter_v13, post_norm_partial_inline126__iter_v13) in pl.pipeline(4, stage=2, init_values=(k0_v1_inline395__rv_v18, mlp_norm_in_inline355__rv_v11, post_norm_partial_inline126__rv_v11)):
                    k0_v1_inline395__ssa_v22: pl.Scalar[pl.INDEX] = kb_inline77__idx_v4 * 256 + 4096
                    attn_chunk_inline173__ssa_v4: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32_inline468__rv_v5, [16, 256], [0, k0_v1_inline395__ssa_v22])
                    hidden_chunk_inline351__ssa_v4: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur__rv_v2, [16, 256], [0, k0_v1_inline395__ssa_v22])
                    resid_fp32_inline286__ssa_v4: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk_inline173__ssa_v4, hidden_chunk_inline351__ssa_v4)
                    post_norm_partial_inline126__ssa_v15: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(post_norm_partial_inline126__iter_v13, resid_fp32_inline286__ssa_v4, [0, k0_v1_inline395__ssa_v22])
                    post_gamma_inline80__ssa_v4: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight__ssa_v0, [1, 256], [0, k0_v1_inline395__ssa_v22])
                    mlp_norm_in_inline355__ssa_v15: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(mlp_norm_in_inline355__iter_v13, pl.tensor.cast(pl.tensor.col_expand_mul(resid_fp32_inline286__ssa_v4, post_gamma_inline80__ssa_v4), target_type=pl.BF16, mode='round'), [0, k0_v1_inline395__ssa_v22])
                    k0_v1_inline395__rv_v21, mlp_norm_in_inline355__rv_v14, post_norm_partial_inline126__rv_v14 = pl.yield_(k0_v1_inline395__ssa_v22, mlp_norm_in_inline355__ssa_v15, post_norm_partial_inline126__ssa_v15)
            cast_tids_inline391__ssa_v5: pl.Array[5, pl.TASK_ID] = pl.array.update_element(cast_tids_inline391__ssa_v4, 4, cast_tid_k_inline79__ssa_v4)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rms_reduce", deps=[out_tids_inline104__rv_v2]) as reduce_tid_inline76__ssa_v0:
                sq_sum_inline75__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                for kb_inline74__idx_v0, (sq_sum_inline75__iter_v1,) in pl.pipeline(20, stage=2, init_values=(sq_sum_inline75__ssa_v0,)):
                    k0_inline321__ssa_v1: pl.Scalar[pl.INDEX] = kb_inline74__idx_v0 * 256
                    attn_chunk_inline173__ssa_v5: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32_inline468__rv_v5, [16, 256], [0, k0_inline321__ssa_v1])
                    hidden_chunk_inline351__ssa_v5: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur__rv_v2, [16, 256], [0, k0_inline321__ssa_v1])
                    resid_chunk_inline73__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk_inline173__ssa_v5, hidden_chunk_inline351__ssa_v5)
                    sq_sum_inline75__ssa_v3: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(sq_sum_inline75__iter_v1, pl.tensor.reshape(pl.tensor.row_sum(pl.tensor.mul(resid_chunk_inline73__ssa_v0, resid_chunk_inline73__ssa_v0)), [1, 16]))
                    sq_sum_inline75__rv_v2: pl.Tensor[[1, 16], pl.FP32] = pl.yield_(sq_sum_inline75__ssa_v3)
                post_inv_rms_inline303__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(sq_sum_inline75__rv_v2, 0.00019531250000000001), 9.9999999999999995e-07)))
                post_inv_rms_col_inline334__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(post_inv_rms_inline303__ssa_v0, [16, 1])
                inv_rms_tile_inline123__ssa_v1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.assemble(inv_rms_tile_inline123__ssa_v0, post_inv_rms_col_inline334__ssa_v0, [0, 0])
            for n_out_inline346__idx_v0, (gate_acc_all_inline234__iter_v4, gate_tids_inline108__iter_v1, up_acc_all_inline284__iter_v4, up_tids_inline107__iter_v1) in pl.parallel(17, init_values=(gate_acc_all_inline234__rv_v2, gate_tids_inline108__ssa_v0, up_acc_all_inline284__rv_v2, up_tids_inline107__ssa_v0)):
                n0_inline352__ssa_v6: pl.Scalar[pl.INDEX] = n_out_inline346__idx_v0 * 1024
                for k_split_inline387__idx_v0, (gate_acc_all_inline234__iter_v6, gate_tids_inline108__iter_v3, up_acc_all_inline284__iter_v6, up_tids_inline107__iter_v3) in pl.range(5, init_values=(gate_acc_all_inline234__iter_v4, gate_tids_inline108__iter_v1, up_acc_all_inline284__iter_v4, up_tids_inline107__iter_v1)):
                    k0_inline321__ssa_v2: pl.Scalar[pl.INDEX] = k_split_inline387__idx_v0 * 1024
                    _submit_deps_buf_inline72__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline70__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline72__ssa_v0, 0, pl.array.get_element(cast_tids_inline391__ssa_v5, k_split_inline387__idx_v0))
                    _submit_deps_buf_inline69__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline70__ssa_v0, 1, gate_seed_tid_inline198__ssa_v0)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj", deps=[_submit_deps_buf_inline69__ssa_v0]) as gate_tid_inline185__ssa_v0:
                        a0_inline290__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in_inline355__rv_v14, [16, 64], [0, k0_inline321__ssa_v2])
                        w0_inline67__ssa_v0: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_gate__ssa_v0, [64, 1024], [layer_hidden_base_inline217__ssa_v0 + k0_inline321__ssa_v2, n0_inline352__ssa_v6])
                        c_acc_inline459__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0_inline290__ssa_v0, w0_inline67__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for lk_inline270__idx_v0, (c_acc_inline459__iter_v1,) in pl.pipeline(1, 16, stage=2, init_values=(c_acc_inline459__ssa_v0,)):
                            ks_off_inline66__ssa_v0: pl.Scalar[pl.INDEX] = lk_inline270__idx_v0 * 64
                            a_k_inline364__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in_inline355__rv_v14, [16, 64], [0, k0_inline321__ssa_v2 + ks_off_inline66__ssa_v0])
                            w_k_inline65__ssa_v0: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_gate__ssa_v0, [64, 1024], [layer_hidden_base_inline217__ssa_v0 + k0_inline321__ssa_v2 + ks_off_inline66__ssa_v0, n0_inline352__ssa_v6])
                            c_acc_inline459__ssa_v3: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc_inline459__iter_v1, a_k_inline364__ssa_v0, w_k_inline65__ssa_v0, a_trans=False, b_trans=False)
                            c_acc_inline459__rv_v2: pl.Tensor[[16, 1024], pl.FP32] = pl.yield_(c_acc_inline459__ssa_v3)
                        gate_acc_all_inline234__ssa_v8: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(gate_acc_all_inline234__iter_v6, c_acc_inline459__rv_v2, [0, n0_inline352__ssa_v6], atomic=pl.AtomicType.Add)
                    gate_tids_inline108__ssa_v5: pl.Array[85, pl.TASK_ID] = pl.array.update_element(gate_tids_inline108__iter_v3, n_out_inline346__idx_v0 * 5 + k_split_inline387__idx_v0, gate_tid_inline185__ssa_v0)
                    _submit_deps_buf_inline146__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline64__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline146__ssa_v0, 0, pl.array.get_element(cast_tids_inline391__ssa_v5, k_split_inline387__idx_v0))
                    _submit_deps_buf_inline62__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline64__ssa_v0, 1, up_seed_tid_inline285__ssa_v0)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj", deps=[_submit_deps_buf_inline62__ssa_v0]) as up_tid_inline415__ssa_v0:
                        a0_v1_inline61__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in_inline355__rv_v14, [16, 64], [0, k0_inline321__ssa_v2])
                        w0_v1_inline59__ssa_v0: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [64, 1024], [layer_hidden_base_inline217__ssa_v0 + k0_inline321__ssa_v2, n0_inline352__ssa_v6])
                        c_acc_v1_inline377__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0_v1_inline61__ssa_v0, w0_v1_inline59__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for lk_inline106__idx_v0, (c_acc_v1_inline377__iter_v1,) in pl.pipeline(1, 16, stage=2, init_values=(c_acc_v1_inline377__ssa_v0,)):
                            ks_off_inline66__ssa_v1: pl.Scalar[pl.INDEX] = lk_inline106__idx_v0 * 64
                            a_k_inline364__ssa_v1: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in_inline355__rv_v14, [16, 64], [0, k0_inline321__ssa_v2 + ks_off_inline66__ssa_v1])
                            w_k_inline65__ssa_v1: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [64, 1024], [layer_hidden_base_inline217__ssa_v0 + k0_inline321__ssa_v2 + ks_off_inline66__ssa_v1, n0_inline352__ssa_v6])
                            c_acc_v1_inline377__ssa_v3: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc_v1_inline377__iter_v1, a_k_inline364__ssa_v1, w_k_inline65__ssa_v1, a_trans=False, b_trans=False)
                            c_acc_v1_inline377__rv_v2: pl.Tensor[[16, 1024], pl.FP32] = pl.yield_(c_acc_v1_inline377__ssa_v3)
                        up_acc_all_inline284__ssa_v8: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(up_acc_all_inline284__iter_v6, c_acc_v1_inline377__rv_v2, [0, n0_inline352__ssa_v6], atomic=pl.AtomicType.Add)
                    up_tids_inline107__ssa_v5: pl.Array[85, pl.TASK_ID] = pl.array.update_element(up_tids_inline107__iter_v3, n_out_inline346__idx_v0 * 5 + k_split_inline387__idx_v0, up_tid_inline415__ssa_v0)
                    gate_acc_all_inline234__rv_v7, gate_tids_inline108__rv_v4, up_acc_all_inline284__rv_v7, up_tids_inline107__rv_v4 = pl.yield_(gate_acc_all_inline234__ssa_v8, gate_tids_inline108__ssa_v5, up_acc_all_inline284__ssa_v8, up_tids_inline107__ssa_v5)
                gate_acc_all_inline234__rv_v5, gate_tids_inline108__rv_v2, up_acc_all_inline284__rv_v5, up_tids_inline107__rv_v2 = pl.yield_(gate_acc_all_inline234__rv_v7, gate_tids_inline108__rv_v4, up_acc_all_inline284__rv_v7, up_tids_inline107__rv_v4)
            for n_out_inline58__idx_v0, (mlp_tile_inline122__iter_v1, silu_tids_inline109__iter_v1) in pl.parallel(17, init_values=(mlp_tile_inline122__ssa_v0, silu_tids_inline109__ssa_v0)):
                n0_inline352__ssa_v7: pl.Scalar[pl.INDEX] = n_out_inline58__idx_v0 * 1024
                _submit_deps_buf_inline480__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.create(11, dtype=pl.TASK_ID)
                _submit_deps_buf_inline328__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline480__ssa_v0, 0, reduce_tid_inline76__ssa_v0)
                _submit_deps_buf_inline207__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline328__ssa_v0, 1, pl.array.get_element(gate_tids_inline108__rv_v2, n_out_inline58__idx_v0 * 5))
                _submit_deps_buf_inline425__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline207__ssa_v0, 2, pl.array.get_element(gate_tids_inline108__rv_v2, n_out_inline58__idx_v0 * 5 + 1))
                _submit_deps_buf_inline57__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline425__ssa_v0, 3, pl.array.get_element(gate_tids_inline108__rv_v2, n_out_inline58__idx_v0 * 5 + 2))
                _submit_deps_buf_inline56__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline57__ssa_v0, 4, pl.array.get_element(gate_tids_inline108__rv_v2, n_out_inline58__idx_v0 * 5 + 3))
                _submit_deps_buf_inline55__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline56__ssa_v0, 5, pl.array.get_element(gate_tids_inline108__rv_v2, n_out_inline58__idx_v0 * 5 + 4))
                _submit_deps_buf_inline54__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline55__ssa_v0, 6, pl.array.get_element(up_tids_inline107__rv_v2, n_out_inline58__idx_v0 * 5))
                _submit_deps_buf_inline53__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline54__ssa_v0, 7, pl.array.get_element(up_tids_inline107__rv_v2, n_out_inline58__idx_v0 * 5 + 1))
                _submit_deps_buf_inline52__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline53__ssa_v0, 8, pl.array.get_element(up_tids_inline107__rv_v2, n_out_inline58__idx_v0 * 5 + 2))
                _submit_deps_buf_inline381__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline52__ssa_v0, 9, pl.array.get_element(up_tids_inline107__rv_v2, n_out_inline58__idx_v0 * 5 + 3))
                _submit_deps_buf_inline51__ssa_v0: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline381__ssa_v0, 10, pl.array.get_element(up_tids_inline107__rv_v2, n_out_inline58__idx_v0 * 5 + 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu", deps=[_submit_deps_buf_inline51__ssa_v0]) as silu_tid_inline265__ssa_v0:
                    inv_rms_chunk_inline50__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(inv_rms_tile_inline123__ssa_v1, [16, 1], [0, 0])
                    for sub_inline48__idx_v0, (mlp_tile_inline122__iter_v3,) in pl.pipeline(4, stage=2, init_values=(mlp_tile_inline122__iter_v1,)):
                        silu_off_inline63__ssa_v0: pl.Scalar[pl.INDEX] = n0_inline352__ssa_v7 + sub_inline48__idx_v0 * 256
                        gate_chunk_inline255__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(gate_acc_all_inline234__rv_v5, [16, 256], [0, silu_off_inline63__ssa_v0])
                        up_chunk_inline46__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(up_acc_all_inline284__rv_v5, [16, 256], [0, silu_off_inline63__ssa_v0])
                        scaled_gate_inline125__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.row_expand_mul(gate_chunk_inline255__ssa_v0, inv_rms_chunk_inline50__ssa_v0)
                        scaled_up_inline479__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.row_expand_mul(up_chunk_inline46__ssa_v0, inv_rms_chunk_inline50__ssa_v0)
                        sigmoid_inline293__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(scaled_gate_inline125__ssa_v0)), 1.0))
                        mlp_chunk_inline45__ssa_v0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.mul(pl.tensor.mul(scaled_gate_inline125__ssa_v0, sigmoid_inline293__ssa_v0), scaled_up_inline479__ssa_v0)
                        mlp_tile_inline122__ssa_v5: pl.Tensor[[16, 17408], pl.BF16] = pl.tensor.assemble(mlp_tile_inline122__iter_v3, pl.tensor.cast(mlp_chunk_inline45__ssa_v0, target_type=pl.BF16, mode='round'), [0, silu_off_inline63__ssa_v0])
                        mlp_tile_inline122__rv_v4: pl.Tensor[[16, 17408], pl.BF16] = pl.yield_(mlp_tile_inline122__ssa_v5)
                silu_tids_inline109__ssa_v3: pl.Array[17, pl.TASK_ID] = pl.array.update_element(silu_tids_inline109__iter_v1, n_out_inline58__idx_v0, silu_tid_inline265__ssa_v0)
                mlp_tile_inline122__rv_v2, silu_tids_inline109__rv_v2 = pl.yield_(mlp_tile_inline122__rv_v4, silu_tids_inline109__ssa_v3)
            for n_out_inline43__idx_v0, (down_acc_all_inline439__iter_v4, down_tids_inline404__iter_v1) in pl.parallel(5, init_values=(down_acc_all_inline439__rv_v2, down_tids_inline404__ssa_v0)):
                n0_inline352__ssa_v8: pl.Scalar[pl.INDEX] = n_out_inline43__idx_v0 * 1024
                for k_split_inline42__idx_v0, (down_acc_all_inline439__iter_v6, down_tids_inline404__iter_v3) in pl.range(17, init_values=(down_acc_all_inline439__iter_v4, down_tids_inline404__iter_v1)):
                    k0_inline321__ssa_v3: pl.Scalar[pl.INDEX] = k_split_inline42__idx_v0 * 1024
                    _submit_deps_buf_inline41__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline40__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline41__ssa_v0, 0, seed_tid_inline417__ssa_v0)
                    _submit_deps_buf_inline430__ssa_v0: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline40__ssa_v0, 1, pl.array.get_element(silu_tids_inline109__rv_v2, k_split_inline42__idx_v0))
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj", deps=[_submit_deps_buf_inline430__ssa_v0]) as down_tid_inline116__ssa_v0:
                        a0_v2_inline39__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_tile_inline122__rv_v2, [16, 64], [0, k0_inline321__ssa_v3])
                        w0_v2_inline311__ssa_v0: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_down__ssa_v0, [64, 1024], [layer_inter_base_inline296__ssa_v0 + k0_inline321__ssa_v3, n0_inline352__ssa_v8])
                        c_acc_v2_inline38__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0_v2_inline39__ssa_v0, w0_v2_inline311__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for lk_inline422__idx_v0, (c_acc_v2_inline38__iter_v1,) in pl.pipeline(1, 16, stage=2, init_values=(c_acc_v2_inline38__ssa_v0,)):
                            ks_off_inline66__ssa_v2: pl.Scalar[pl.INDEX] = lk_inline422__idx_v0 * 64
                            a_k_inline364__ssa_v2: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_tile_inline122__rv_v2, [16, 64], [0, k0_inline321__ssa_v3 + ks_off_inline66__ssa_v2])
                            w_k_inline65__ssa_v2: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_down__ssa_v0, [64, 1024], [layer_inter_base_inline296__ssa_v0 + k0_inline321__ssa_v3 + ks_off_inline66__ssa_v2, n0_inline352__ssa_v8])
                            c_acc_v2_inline38__ssa_v3: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc_v2_inline38__iter_v1, a_k_inline364__ssa_v2, w_k_inline65__ssa_v2, a_trans=False, b_trans=False)
                            c_acc_v2_inline38__rv_v2: pl.Tensor[[16, 1024], pl.FP32] = pl.yield_(c_acc_v2_inline38__ssa_v3)
                        down_acc_all_inline439__ssa_v8: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(down_acc_all_inline439__iter_v6, c_acc_v2_inline38__rv_v2, [0, n0_inline352__ssa_v8], atomic=pl.AtomicType.Add)
                    down_tids_inline404__ssa_v5: pl.Array[85, pl.TASK_ID] = pl.array.update_element(down_tids_inline404__iter_v3, n_out_inline43__idx_v0 * 17 + k_split_inline42__idx_v0, down_tid_inline116__ssa_v0)
                    down_acc_all_inline439__rv_v7, down_tids_inline404__rv_v4 = pl.yield_(down_acc_all_inline439__ssa_v8, down_tids_inline404__ssa_v5)
                down_acc_all_inline439__rv_v5, down_tids_inline404__rv_v2 = pl.yield_(down_acc_all_inline439__rv_v7, down_tids_inline404__rv_v4)
        _submit_deps_buf_inline36__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
        _submit_deps_buf_inline162__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline36__ssa_v0, 0, pl.array.get_element(down_tids_inline404__rv_v2, 0))
        _submit_deps_buf_inline429__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline162__ssa_v0, 1, pl.array.get_element(down_tids_inline404__rv_v2, 1))
        _submit_deps_buf_inline172__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline429__ssa_v0, 2, pl.array.get_element(down_tids_inline404__rv_v2, 2))
        _submit_deps_buf_inline467__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline172__ssa_v0, 3, pl.array.get_element(down_tids_inline404__rv_v2, 3))
        _submit_deps_buf_inline35__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline467__ssa_v0, 4, pl.array.get_element(down_tids_inline404__rv_v2, 4))
        _submit_deps_buf_inline68__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline35__ssa_v0, 5, pl.array.get_element(down_tids_inline404__rv_v2, 5))
        _submit_deps_buf_inline386__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline68__ssa_v0, 6, pl.array.get_element(down_tids_inline404__rv_v2, 6))
        _submit_deps_buf_inline37__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline386__ssa_v0, 7, pl.array.get_element(down_tids_inline404__rv_v2, 7))
        _submit_deps_buf_inline34__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline37__ssa_v0, 8, pl.array.get_element(down_tids_inline404__rv_v2, 8))
        _submit_deps_buf_inline60__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline34__ssa_v0, 9, pl.array.get_element(down_tids_inline404__rv_v2, 9))
        _submit_deps_buf_inline33__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline60__ssa_v0, 10, pl.array.get_element(down_tids_inline404__rv_v2, 10))
        _submit_deps_buf_inline164__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline33__ssa_v0, 11, pl.array.get_element(down_tids_inline404__rv_v2, 11))
        _submit_deps_buf_inline32__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline164__ssa_v0, 12, pl.array.get_element(down_tids_inline404__rv_v2, 12))
        _submit_deps_buf_inline31__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline32__ssa_v0, 13, pl.array.get_element(down_tids_inline404__rv_v2, 13))
        _submit_deps_buf_inline333__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline31__ssa_v0, 14, pl.array.get_element(down_tids_inline404__rv_v2, 14))
        _submit_deps_buf_inline30__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline333__ssa_v0, 15, pl.array.get_element(down_tids_inline404__rv_v2, 15))
        _submit_deps_buf_inline29__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline30__ssa_v0, 16, pl.array.get_element(down_tids_inline404__rv_v2, 16))
        _submit_deps_buf_inline28__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline29__ssa_v0, 17, pl.array.get_element(down_tids_inline404__rv_v2, 17))
        _submit_deps_buf_inline205__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline28__ssa_v0, 18, pl.array.get_element(down_tids_inline404__rv_v2, 18))
        _submit_deps_buf_inline27__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline205__ssa_v0, 19, pl.array.get_element(down_tids_inline404__rv_v2, 19))
        _submit_deps_buf_inline49__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline27__ssa_v0, 20, pl.array.get_element(down_tids_inline404__rv_v2, 20))
        _submit_deps_buf_inline47__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline49__ssa_v0, 21, pl.array.get_element(down_tids_inline404__rv_v2, 21))
        _submit_deps_buf_inline101__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline47__ssa_v0, 22, pl.array.get_element(down_tids_inline404__rv_v2, 22))
        _submit_deps_buf_inline99__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline101__ssa_v0, 23, pl.array.get_element(down_tids_inline404__rv_v2, 23))
        _submit_deps_buf_inline26__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline99__ssa_v0, 24, pl.array.get_element(down_tids_inline404__rv_v2, 24))
        _submit_deps_buf_inline25__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline26__ssa_v0, 25, pl.array.get_element(down_tids_inline404__rv_v2, 25))
        _submit_deps_buf_inline91__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline25__ssa_v0, 26, pl.array.get_element(down_tids_inline404__rv_v2, 26))
        _submit_deps_buf_inline24__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline91__ssa_v0, 27, pl.array.get_element(down_tids_inline404__rv_v2, 27))
        _submit_deps_buf_inline373__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline24__ssa_v0, 28, pl.array.get_element(down_tids_inline404__rv_v2, 28))
        _submit_deps_buf_inline434__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline373__ssa_v0, 29, pl.array.get_element(down_tids_inline404__rv_v2, 29))
        _submit_deps_buf_inline210__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline434__ssa_v0, 30, pl.array.get_element(down_tids_inline404__rv_v2, 30))
        _submit_deps_buf_inline78__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline210__ssa_v0, 31, pl.array.get_element(down_tids_inline404__rv_v2, 31))
        _submit_deps_buf_inline23__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline78__ssa_v0, 32, pl.array.get_element(down_tids_inline404__rv_v2, 32))
        _submit_deps_buf_inline22__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline23__ssa_v0, 33, pl.array.get_element(down_tids_inline404__rv_v2, 33))
        _submit_deps_buf_inline21__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline22__ssa_v0, 34, pl.array.get_element(down_tids_inline404__rv_v2, 34))
        _submit_deps_buf_inline403__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline21__ssa_v0, 35, pl.array.get_element(down_tids_inline404__rv_v2, 35))
        _submit_deps_buf_inline412__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline403__ssa_v0, 36, pl.array.get_element(down_tids_inline404__rv_v2, 36))
        _submit_deps_buf_inline157__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline412__ssa_v0, 37, pl.array.get_element(down_tids_inline404__rv_v2, 37))
        _submit_deps_buf_inline129__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline157__ssa_v0, 38, pl.array.get_element(down_tids_inline404__rv_v2, 38))
        _submit_deps_buf_inline159__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline129__ssa_v0, 39, pl.array.get_element(down_tids_inline404__rv_v2, 39))
        _submit_deps_buf_inline71__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline159__ssa_v0, 40, pl.array.get_element(down_tids_inline404__rv_v2, 40))
        _submit_deps_buf_inline20__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline71__ssa_v0, 41, pl.array.get_element(down_tids_inline404__rv_v2, 41))
        _submit_deps_buf_inline120__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline20__ssa_v0, 42, pl.array.get_element(down_tids_inline404__rv_v2, 42))
        _submit_deps_buf_inline118__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline120__ssa_v0, 43, pl.array.get_element(down_tids_inline404__rv_v2, 43))
        _submit_deps_buf_inline115__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline118__ssa_v0, 44, pl.array.get_element(down_tids_inline404__rv_v2, 44))
        _submit_deps_buf_inline113__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline115__ssa_v0, 45, pl.array.get_element(down_tids_inline404__rv_v2, 45))
        _submit_deps_buf_inline111__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline113__ssa_v0, 46, pl.array.get_element(down_tids_inline404__rv_v2, 46))
        _submit_deps_buf_inline178__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline111__ssa_v0, 47, pl.array.get_element(down_tids_inline404__rv_v2, 47))
        _submit_deps_buf_inline184__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline178__ssa_v0, 48, pl.array.get_element(down_tids_inline404__rv_v2, 48))
        _submit_deps_buf_inline19__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline184__ssa_v0, 49, pl.array.get_element(down_tids_inline404__rv_v2, 49))
        _submit_deps_buf_inline18__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline19__ssa_v0, 50, pl.array.get_element(down_tids_inline404__rv_v2, 50))
        _submit_deps_buf_inline354__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline18__ssa_v0, 51, pl.array.get_element(down_tids_inline404__rv_v2, 51))
        _submit_deps_buf_inline17__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline354__ssa_v0, 52, pl.array.get_element(down_tids_inline404__rv_v2, 52))
        _submit_deps_buf_inline16__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline17__ssa_v0, 53, pl.array.get_element(down_tids_inline404__rv_v2, 53))
        _submit_deps_buf_inline15__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline16__ssa_v0, 54, pl.array.get_element(down_tids_inline404__rv_v2, 54))
        _submit_deps_buf_inline170__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline15__ssa_v0, 55, pl.array.get_element(down_tids_inline404__rv_v2, 55))
        _submit_deps_buf_inline14__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline170__ssa_v0, 56, pl.array.get_element(down_tids_inline404__rv_v2, 56))
        _submit_deps_buf_inline13__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline14__ssa_v0, 57, pl.array.get_element(down_tids_inline404__rv_v2, 57))
        _submit_deps_buf_inline12__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline13__ssa_v0, 58, pl.array.get_element(down_tids_inline404__rv_v2, 58))
        _submit_deps_buf_inline11__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline12__ssa_v0, 59, pl.array.get_element(down_tids_inline404__rv_v2, 59))
        _submit_deps_buf_inline455__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline11__ssa_v0, 60, pl.array.get_element(down_tids_inline404__rv_v2, 60))
        _submit_deps_buf_inline10__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline455__ssa_v0, 61, pl.array.get_element(down_tids_inline404__rv_v2, 61))
        _submit_deps_buf_inline9__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline10__ssa_v0, 62, pl.array.get_element(down_tids_inline404__rv_v2, 62))
        _submit_deps_buf_inline193__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline9__ssa_v0, 63, pl.array.get_element(down_tids_inline404__rv_v2, 63))
        _submit_deps_buf_inline226__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline193__ssa_v0, 64, pl.array.get_element(down_tids_inline404__rv_v2, 64))
        _submit_deps_buf_inline8__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline226__ssa_v0, 65, pl.array.get_element(down_tids_inline404__rv_v2, 65))
        _submit_deps_buf_inline195__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline8__ssa_v0, 66, pl.array.get_element(down_tids_inline404__rv_v2, 66))
        _submit_deps_buf_inline7__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline195__ssa_v0, 67, pl.array.get_element(down_tids_inline404__rv_v2, 67))
        _submit_deps_buf_inline182__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline7__ssa_v0, 68, pl.array.get_element(down_tids_inline404__rv_v2, 68))
        _submit_deps_buf_inline44__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline182__ssa_v0, 69, pl.array.get_element(down_tids_inline404__rv_v2, 69))
        _submit_deps_buf_inline6__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline44__ssa_v0, 70, pl.array.get_element(down_tids_inline404__rv_v2, 70))
        _submit_deps_buf_inline315__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline6__ssa_v0, 71, pl.array.get_element(down_tids_inline404__rv_v2, 71))
        _submit_deps_buf_inline152__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline315__ssa_v0, 72, pl.array.get_element(down_tids_inline404__rv_v2, 72))
        _submit_deps_buf_inline371__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline152__ssa_v0, 73, pl.array.get_element(down_tids_inline404__rv_v2, 73))
        _submit_deps_buf_inline5__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline371__ssa_v0, 74, pl.array.get_element(down_tids_inline404__rv_v2, 74))
        _submit_deps_buf_inline278__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline5__ssa_v0, 75, pl.array.get_element(down_tids_inline404__rv_v2, 75))
        _submit_deps_buf_inline4__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline278__ssa_v0, 76, pl.array.get_element(down_tids_inline404__rv_v2, 76))
        _submit_deps_buf_inline3__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4__ssa_v0, 77, pl.array.get_element(down_tids_inline404__rv_v2, 77))
        _submit_deps_buf_inline218__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline3__ssa_v0, 78, pl.array.get_element(down_tids_inline404__rv_v2, 78))
        _submit_deps_buf_inline2__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline218__ssa_v0, 79, pl.array.get_element(down_tids_inline404__rv_v2, 79))
        _submit_deps_buf_inline1__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline2__ssa_v0, 80, pl.array.get_element(down_tids_inline404__rv_v2, 80))
        _submit_deps_buf_inline124__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline1__ssa_v0, 81, pl.array.get_element(down_tids_inline404__rv_v2, 81))
        _submit_deps_buf_inline0__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline124__ssa_v0, 82, pl.array.get_element(down_tids_inline404__rv_v2, 82))
        _submit_deps_buf_inline483__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline0__ssa_v0, 83, pl.array.get_element(down_tids_inline404__rv_v2, 83))
        _submit_deps_buf_inline240__ssa_v0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline483__ssa_v0, 84, pl.array.get_element(down_tids_inline404__rv_v2, 84))
        with pl.spmd(5, name_hint="dcr_xgamma_spmd", deps=[_submit_deps_buf_inline240__ssa_v0]) as dcr_tid_inline359__ssa_v0:
            n_out_inline43__ssa_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            n0_inline352__ssa_v9: pl.Scalar[pl.INDEX] = n_out_inline43__ssa_v1 * 1024
            out_chunk_inline105__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.add(pl.tensor.slice(down_acc_all_inline439__rv_v5, [16, 1024], [0, n0_inline352__ssa_v9]), pl.tensor.slice(post_norm_partial_inline126__rv_v14, [16, 1024], [0, n0_inline352__ssa_v9]))
            next_hidden__ssa_v1: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(next_hidden__ssa_v0, out_chunk_inline105__ssa_v0, [0, n0_inline352__ssa_v9])
            gamma_next_inline461__ssa_v0: pl.Tensor[[1, 1024], pl.FP32] = pl.tensor.slice(input_rms_weight__ssa_v0, [1, 1024], [0, n0_inline352__ssa_v9])
            xg_inline252__ssa_v0: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.col_expand_mul(out_chunk_inline105__ssa_v0, gamma_next_inline461__ssa_v0)
            next_normed__ssa_v1: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(next_normed__ssa_v0, pl.tensor.cast(xg_inline252__ssa_v0, target_type=pl.BF16, mode='round'), [0, n0_inline352__ssa_v9])
        carry_tids__ssa_v8: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_tids__rv_v2, 0, dcr_tid_inline359__ssa_v0)
        carry_normed_tids__ssa_v6: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_normed_tids__rv_v2, 0, dcr_tid_inline359__ssa_v0)
        carry_tids__ssa_v9: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_tids__ssa_v8, 1, dcr_tid_inline359__ssa_v0)
        carry_normed_tids__ssa_v7: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_normed_tids__ssa_v6, 1, dcr_tid_inline359__ssa_v0)
        carry_tids__ssa_v10: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_tids__ssa_v9, 2, dcr_tid_inline359__ssa_v0)
        carry_normed_tids__ssa_v8: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_normed_tids__ssa_v7, 2, dcr_tid_inline359__ssa_v0)
        carry_tids__ssa_v11: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_tids__ssa_v10, 3, dcr_tid_inline359__ssa_v0)
        carry_normed_tids__ssa_v9: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_normed_tids__ssa_v8, 3, dcr_tid_inline359__ssa_v0)
        carry_tids__ssa_v12: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_tids__ssa_v11, 4, dcr_tid_inline359__ssa_v0)
        carry_normed_tids__ssa_v10: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_normed_tids__ssa_v9, 4, dcr_tid_inline359__ssa_v0)
        cur__ssa_v8: pl.Tensor[[16, 5120], pl.FP32] = next_hidden__ssa_v1
        normed__ssa_v8: pl.Tensor[[16, 5120], pl.BF16] = next_normed__ssa_v1
        for ob0__idx_v0, (out__iter_v1,) in pl.parallel(0, 16, 16, init_values=(out__ssa_v0,)):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_out"):
                for okb__idx_v0, (out__iter_v3,) in pl.range(20, init_values=(out__iter_v1,)):
                    ok0__ssa_v0: pl.Scalar[pl.INDEX] = okb__idx_v0 * 256
                    out__ssa_v5: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(out__iter_v3, pl.tensor.cast(pl.tensor.slice(cur__ssa_v8, [16, 256], [ob0__idx_v0, ok0__ssa_v0]), target_type=pl.BF16, mode='round'), [ob0__idx_v0, ok0__ssa_v0])
                    out__rv_v4: pl.Tensor[[16, 5120], pl.BF16] = pl.yield_(out__ssa_v5)
            out__rv_v2: pl.Tensor[[16, 5120], pl.BF16] = pl.yield_(out__rv_v4)
        return out__rv_v2
