# pypto.program: _jit_decode_fwd_layers
import pypto.language as pl

valid_len = pl.dynamic("valid_len")

@pl.program
class _jit_decode_fwd_layers:
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
            layer_hidden_base_inline217: pl.Scalar[pl.INDEX] = pl.cast(i, pl.INDEX) * 5120
            layer_inter_base_inline296: pl.Scalar[pl.INDEX] = pl.cast(i, pl.INDEX) * 17408
            num_layers_actual_inline264: pl.Scalar[pl.INDEX] = pl.tensor.dim(input_rms_weight, 0)
            layer_cache_rows_inline282: pl.Scalar[pl.INDEX] = pl.tensor.dim(k_cache, 0) // num_layers_actual_inline264
            layer_cache_base_inline156: pl.Scalar[pl.INDEX] = pl.cast(i, pl.INDEX) * layer_cache_rows_inline282
            user_batch_inline342: pl.Scalar[pl.INDEX] = pl.tensor.dim(seq_lens, 0)
            max_blocks_per_seq_inline401: pl.Scalar[pl.INDEX] = pl.tensor.dim(block_table, 0) // user_batch_inline342
            q_norm_w_inline277: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(q_norm_weight, [1, 128], [i, 0])
            k_norm_w_inline253: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(k_norm_weight, [1, 128], [i, 0])
            down_tids_inline404: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
            q_tile_tids_inline280: pl.Array[50, pl.TASK_ID] = pl.array.create(50, dtype=pl.TASK_ID)
            k_tile_tids_inline174: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
            v_tile_tids_inline438: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
            qk_tids_inline247: pl.Array[8, pl.TASK_ID] = pl.array.create(8, dtype=pl.TASK_ID)
            rope_grp_tids_inline266: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
            inv_rms_states_inline449: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_proj_inline248: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            k_proj_inline383: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            v_proj_inline241: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            all_q_padded_inline257: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.create([2048, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            attn_out_inline203: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            all_oi_tmp_inline318: pl.Tensor[[65536, 128], pl.FP32] = pl.tensor.create([65536, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            all_cur_mi_inline409: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.create([65536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            all_cur_li_inline294: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.create([65536, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            fa_work_table_inline281: pl.Tensor[[512, 1], pl.INT32] = pl.tensor.create([512, 1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            fa_total_inline262: pl.Tensor[[1, 1], pl.INT32] = pl.tensor.create([1, 1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            with pl.scope(mode=pl.ScopeMode.MANUAL):
                _submit_deps_buf_inline361: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
                _submit_deps_buf_inline362: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline361, 0, pl.array.get_element(carry_tids, 0))
                _submit_deps_buf_inline230: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline362, 1, pl.array.get_element(carry_tids, 1))
                _submit_deps_buf_inline307: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline230, 2, pl.array.get_element(carry_tids, 2))
                _submit_deps_buf_inline276: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline307, 3, pl.array.get_element(carry_tids, 3))
                _submit_deps_buf_inline310: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline276, 4, pl.array.get_element(carry_tids, 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip", deps=[_submit_deps_buf_inline310]) as rms_tid_inline314:
                    partial_sq_inline299: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                    for kb_inline317 in pl.pipeline(20, stage=4):
                        k0_inline321: pl.Scalar[pl.INDEX] = kb_inline317 * 256
                        x_chunk_inline271: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur, [16, 256], [0, k0_inline321])
                        partial_sq_inline299: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq_inline299, pl.tensor.reshape(pl.tensor.row_sum(pl.tensor.mul(x_chunk_inline271, x_chunk_inline271)), [1, 16]))
                    variance_inline268: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq_inline299, 0.00019531250000000001), 9.9999999999999995e-07), [16, 1])
                    inv_rms_inline335: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance_inline268))
                    inv_rms_states_inline449: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.assemble(inv_rms_states_inline449, inv_rms_inline335, [0, 0])
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed") as q_seed_tid_inline180:
                    for snb_inline238 in pl.pipeline(10, stage=2):
                        q_proj_inline248: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_inline248, pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0), [0, snb_inline238 * 512])
                for q_nt_inline194 in pl.parallel(10):
                    q_n_region_inline300: pl.Scalar[pl.INDEX] = q_nt_inline194 * 512
                    for q_ks_inline287 in pl.range(5):
                        q_k_base_inline332: pl.Scalar[pl.INDEX] = q_ks_inline287 * 1024
                        _submit_deps_buf_inline197: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                        _submit_deps_buf_inline343: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline197, 0, pl.array.get_element(carry_normed_tids, q_ks_inline287))
                        _submit_deps_buf_inline236: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline343, 1, q_seed_tid_inline180)
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj", deps=[_submit_deps_buf_inline236]) as q_tid_inline347:
                            for n_sub_inline350 in pl.range(2):
                                n0_inline352: pl.Scalar[pl.INDEX] = q_n_region_inline300 + n_sub_inline350 * 256
                                q_acc_inline291: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed, [16, 256], [0, q_k_base_inline332]), pl.tensor.slice(wq, [256, 256], [layer_hidden_base_inline217 + q_k_base_inline332, n0_inline352]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                for kc_inline292 in pl.pipeline(1, 4, stage=2):
                                    kk_inline235: pl.Scalar[pl.INDEX] = q_k_base_inline332 + kc_inline292 * 256
                                    q_acc_inline291: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(q_acc_inline291, pl.tensor.slice(normed, [16, 256], [0, kk_inline235]), pl.tensor.slice(wq, [256, 256], [layer_hidden_base_inline217 + kk_inline235, n0_inline352]), a_trans=False, b_trans=False)
                                q_proj_inline248: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_inline248, q_acc_inline291, [0, n0_inline352], atomic=pl.AtomicType.Add)
                        q_tile_tids_inline280: pl.Array[50, pl.TASK_ID] = pl.array.update_element(q_tile_tids_inline280, q_nt_inline194 * 5 + q_ks_inline287, q_tid_inline347)
                _submit_deps_buf_inline338: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
                _submit_deps_buf_inline358: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline338, 0, pl.array.get_element(carry_tids, 0))
                _submit_deps_buf_inline370: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline358, 1, pl.array.get_element(carry_tids, 1))
                _submit_deps_buf_inline379: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline370, 2, pl.array.get_element(carry_tids, 2))
                _submit_deps_buf_inline380: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline379, 3, pl.array.get_element(carry_tids, 3))
                _submit_deps_buf_inline301: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline380, 4, pl.array.get_element(carry_tids, 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_seed", deps=[_submit_deps_buf_inline301]) as k_seed_tid_inline325:
                    k_proj_inline383: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_inline383, pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0), [0, 0])
                for k_nt_inline345 in pl.parallel(2):
                    k_n_region_inline229: pl.Scalar[pl.INDEX] = k_nt_inline345 * 512
                    for k_ks_inline227 in pl.range(5):
                        k_k_base_inline340: pl.Scalar[pl.INDEX] = k_ks_inline227 * 1024
                        _submit_deps_buf_inline188: pl.Array[6, pl.TASK_ID] = pl.array.create(6, dtype=pl.TASK_ID)
                        _submit_deps_buf_inline319: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline188, 0, pl.array.get_element(carry_normed_tids, 0))
                        _submit_deps_buf_inline145: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline319, 1, pl.array.get_element(carry_normed_tids, 1))
                        _submit_deps_buf_inline222: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline145, 2, pl.array.get_element(carry_normed_tids, 2))
                        _submit_deps_buf_inline215: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline222, 3, pl.array.get_element(carry_normed_tids, 3))
                        _submit_deps_buf_inline214: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline215, 4, pl.array.get_element(carry_normed_tids, 4))
                        _submit_deps_buf_inline212: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline214, 5, k_seed_tid_inline325)
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj", deps=[_submit_deps_buf_inline212]) as k_tid_inline158:
                            for n_sub_inline206 in pl.range(2):
                                n0_inline352: pl.Scalar[pl.INDEX] = k_n_region_inline229 + n_sub_inline206 * 256
                                k_acc_inline378: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed, [16, 256], [0, k_k_base_inline340]), pl.tensor.slice(wk, [256, 256], [layer_hidden_base_inline217 + k_k_base_inline340, n0_inline352]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                for kc_inline208 in pl.pipeline(1, 4, stage=2):
                                    kk_inline235: pl.Scalar[pl.INDEX] = k_k_base_inline340 + kc_inline208 * 256
                                    k_acc_inline378: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(k_acc_inline378, pl.tensor.slice(normed, [16, 256], [0, kk_inline235]), pl.tensor.slice(wk, [256, 256], [layer_hidden_base_inline217 + kk_inline235, n0_inline352]), a_trans=False, b_trans=False)
                                k_proj_inline383: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_inline383, k_acc_inline378, [0, n0_inline352], atomic=pl.AtomicType.Add)
                        k_tile_tids_inline174: pl.Array[10, pl.TASK_ID] = pl.array.update_element(k_tile_tids_inline174, k_nt_inline345 * 5 + k_ks_inline227, k_tid_inline158)
                _submit_deps_buf_inline324: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
                _submit_deps_buf_inline441: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline324, 0, pl.array.get_element(carry_tids, 0))
                _submit_deps_buf_inline202: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline441, 1, pl.array.get_element(carry_tids, 1))
                _submit_deps_buf_inline275: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline202, 2, pl.array.get_element(carry_tids, 2))
                _submit_deps_buf_inline225: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline275, 3, pl.array.get_element(carry_tids, 3))
                _submit_deps_buf_inline221: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline225, 4, pl.array.get_element(carry_tids, 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_seed", deps=[_submit_deps_buf_inline221]) as v_seed_tid_inline250:
                    v_proj_inline241: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(v_proj_inline241, pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0), [0, 0])
                for v_nt_inline196 in pl.parallel(2):
                    v_n_region_inline267: pl.Scalar[pl.INDEX] = v_nt_inline196 * 512
                    for v_ks_inline160 in pl.range(5):
                        v_k_base_inline376: pl.Scalar[pl.INDEX] = v_ks_inline160 * 1024
                        _submit_deps_buf_inline272: pl.Array[6, pl.TASK_ID] = pl.array.create(6, dtype=pl.TASK_ID)
                        _submit_deps_buf_inline154: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline272, 0, pl.array.get_element(carry_normed_tids, 0))
                        _submit_deps_buf_inline263: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline154, 1, pl.array.get_element(carry_normed_tids, 1))
                        _submit_deps_buf_inline305: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline263, 2, pl.array.get_element(carry_normed_tids, 2))
                        _submit_deps_buf_inline394: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline305, 3, pl.array.get_element(carry_normed_tids, 3))
                        _submit_deps_buf_inline190: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline394, 4, pl.array.get_element(carry_normed_tids, 4))
                        _submit_deps_buf_inline187: pl.Array[6, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline190, 5, v_seed_tid_inline250)
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj", deps=[_submit_deps_buf_inline187]) as v_tid_inline249:
                            for n_sub_inline213 in pl.range(2):
                                n0_inline352: pl.Scalar[pl.INDEX] = v_n_region_inline267 + n_sub_inline213 * 256
                                v_acc_inline326: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(normed, [16, 256], [0, v_k_base_inline376]), pl.tensor.slice(wv, [256, 256], [layer_hidden_base_inline217 + v_k_base_inline376, n0_inline352]), a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                for kc_inline186 in pl.pipeline(1, 4, stage=2):
                                    kk_inline235: pl.Scalar[pl.INDEX] = v_k_base_inline376 + kc_inline186 * 256
                                    v_acc_inline326: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.matmul_acc(v_acc_inline326, pl.tensor.slice(normed, [16, 256], [0, kk_inline235]), pl.tensor.slice(wv, [256, 256], [layer_hidden_base_inline217 + kk_inline235, n0_inline352]), a_trans=False, b_trans=False)
                                v_proj_inline241: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(v_proj_inline241, v_acc_inline326, [0, n0_inline352], atomic=pl.AtomicType.Add)
                        v_tile_tids_inline438: pl.Array[10, pl.TASK_ID] = pl.array.update_element(v_tile_tids_inline438, v_nt_inline196 * 5 + v_ks_inline160, v_tid_inline249)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_work_build") as work_tid_inline183:
                    cursor_inline169: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(seq_lens, [0]), pl.INDEX) * 0
                    for wb_inline181 in pl.unroll(16):
                        wb_ctx_inline177: pl.Scalar[pl.INDEX] = (pl.cast(pl.tensor.read(seq_lens, [wb_inline181]), pl.INDEX) + 127) // 128
                        for wp_inline304 in pl.range(wb_ctx_inline177):
                            pl.tensor.write(fa_work_table_inline281, [cursor_inline169 + wp_inline304, 0], pl.cast(wb_inline181 * 32 + wp_inline304, pl.INT32))
                        cursor_inline169: pl.Scalar[pl.INDEX] = cursor_inline169 + wb_ctx_inline177
                    pl.tensor.write(fa_total_inline262, [0, 0], pl.cast(cursor_inline169, pl.INT32))
                q_proj_norm_inline375: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                k_proj_norm_inline171: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.create([16, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                q_inv_states_inline168: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.create([640, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                k_inv_states_inline204: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.create([128, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                inv_rms_col_inline165: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(inv_rms_states_inline449, [16, 1], [0, 0])
                for h_inline163 in pl.unroll(8):
                    qt0_inline372: pl.Scalar[pl.INDEX] = h_inline163 * 5 * 128 // 512
                    kt_inline349: pl.Scalar[pl.INDEX] = h_inline163 * 128 // 512
                    _submit_deps_buf_inline448: pl.Array[16, pl.TASK_ID] = pl.array.create(16, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline323: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline448, 0, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 0))
                    _submit_deps_buf_inline440: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline323, 1, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 1))
                    _submit_deps_buf_inline316: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline440, 2, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 2))
                    _submit_deps_buf_inline153: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline316, 3, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 3))
                    _submit_deps_buf_inline339: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline153, 4, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 4))
                    _submit_deps_buf_inline151: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline339, 5, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 5))
                    _submit_deps_buf_inline353: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline151, 6, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 6))
                    _submit_deps_buf_inline473: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline353, 7, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 7))
                    _submit_deps_buf_inline148: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline473, 8, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 8))
                    _submit_deps_buf_inline348: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline148, 9, pl.array.get_element(q_tile_tids_inline280, qt0_inline372 * 5 + 9))
                    _submit_deps_buf_inline244: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline348, 10, pl.array.get_element(k_tile_tids_inline174, kt_inline349 * 5 + 0))
                    _submit_deps_buf_inline166: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline244, 11, pl.array.get_element(k_tile_tids_inline174, kt_inline349 * 5 + 1))
                    _submit_deps_buf_inline176: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline166, 12, pl.array.get_element(k_tile_tids_inline174, kt_inline349 * 5 + 2))
                    _submit_deps_buf_inline219: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline176, 13, pl.array.get_element(k_tile_tids_inline174, kt_inline349 * 5 + 3))
                    _submit_deps_buf_inline144: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline219, 14, pl.array.get_element(k_tile_tids_inline174, kt_inline349 * 5 + 4))
                    _submit_deps_buf_inline223: pl.Array[16, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline144, 15, rms_tid_inline314)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_norm", deps=[_submit_deps_buf_inline223]) as qk_tid_h_inline245:
                        q0_inline384: pl.Scalar[pl.INDEX] = h_inline163 * 5 * 128
                        q_slice_inline388: pl.Tensor[[16, 640], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_proj_inline248, [16, 640], [0, q0_inline384]), inv_rms_col_inline165)
                        q_chunk_inline389: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.reshape(q_slice_inline388, [80, 128])
                        q_g_inline393: pl.Tensor[[80, 128], pl.FP32] = pl.tensor.col_expand_mul(q_chunk_inline389, q_norm_w_inline277)
                        q_proj_norm_inline375: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_proj_norm_inline375, pl.tensor.reshape(q_g_inline393, [16, 640]), [0, q0_inline384])
                        q_ss_inline312: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(q_chunk_inline389, q_chunk_inline389))
                        q_inv_inline200: pl.Tensor[[80, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(q_ss_inline312, 0.0078125), 9.9999999999999995e-07)))
                        q_inv_states_inline168: pl.Tensor[[640, 1], pl.FP32] = pl.tensor.assemble(q_inv_states_inline168, q_inv_inline200, [h_inline163 * 16 * 5, 0])
                        k0_v1_inline395: pl.Scalar[pl.INDEX] = h_inline163 * 128
                        k_chunk_inline309: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(k_proj_inline383, [16, 128], [0, k0_v1_inline395]), inv_rms_col_inline165)
                        k_g_inline397: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.col_expand_mul(k_chunk_inline309, k_norm_w_inline253)
                        k_proj_norm_inline171: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_proj_norm_inline171, k_g_inline397, [0, k0_v1_inline395])
                        k_ss_inline398: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(k_chunk_inline309, k_chunk_inline309))
                        k_inv_inline385: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(k_ss_inline398, 0.0078125), 9.9999999999999995e-07)))
                        k_inv_states_inline204: pl.Tensor[[128, 1], pl.FP32] = pl.tensor.assemble(k_inv_states_inline204, k_inv_inline385, [h_inline163 * 16, 0])
                    qk_tids_inline247: pl.Array[8, pl.TASK_ID] = pl.array.update_element(qk_tids_inline247, h_inline163, qk_tid_h_inline245)
                _submit_deps_buf_inline302: pl.Array[19, pl.TASK_ID] = pl.array.create(19, dtype=pl.TASK_ID)
                _submit_deps_buf_inline320: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline302, 0, pl.array.get_element(qk_tids_inline247, 0))
                _submit_deps_buf_inline399: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline320, 1, pl.array.get_element(qk_tids_inline247, 1))
                _submit_deps_buf_inline400: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline399, 2, pl.array.get_element(qk_tids_inline247, 2))
                _submit_deps_buf_inline150: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline400, 3, pl.array.get_element(qk_tids_inline247, 3))
                _submit_deps_buf_inline283: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline150, 4, pl.array.get_element(qk_tids_inline247, 4))
                _submit_deps_buf_inline254: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline283, 5, pl.array.get_element(qk_tids_inline247, 5))
                _submit_deps_buf_inline437: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline254, 6, pl.array.get_element(qk_tids_inline247, 6))
                _submit_deps_buf_inline246: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline437, 7, pl.array.get_element(qk_tids_inline247, 7))
                _submit_deps_buf_inline313: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline246, 8, rms_tid_inline314)
                _submit_deps_buf_inline231: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline313, 9, pl.array.get_element(v_tile_tids_inline438, 0))
                _submit_deps_buf_inline402: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline231, 10, pl.array.get_element(v_tile_tids_inline438, 1))
                _submit_deps_buf_inline476: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline402, 11, pl.array.get_element(v_tile_tids_inline438, 2))
                _submit_deps_buf_inline331: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline476, 12, pl.array.get_element(v_tile_tids_inline438, 3))
                _submit_deps_buf_inline341: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline331, 13, pl.array.get_element(v_tile_tids_inline438, 4))
                _submit_deps_buf_inline405: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline341, 14, pl.array.get_element(v_tile_tids_inline438, 5))
                _submit_deps_buf_inline413: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline405, 15, pl.array.get_element(v_tile_tids_inline438, 6))
                _submit_deps_buf_inline406: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline413, 16, pl.array.get_element(v_tile_tids_inline438, 7))
                _submit_deps_buf_inline408: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline406, 17, pl.array.get_element(v_tile_tids_inline438, 8))
                _submit_deps_buf_inline410: pl.Array[19, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline408, 18, pl.array.get_element(v_tile_tids_inline438, 9))
                with pl.spmd(32, name_hint="rope_qkv_spmd", deps=[_submit_deps_buf_inline410]) as rope_tid_inline411:
                    rope_core_inline414: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                    for it_inline149 in pl.pipeline(4, stage=2):
                        g_idx_inline418: pl.Scalar[pl.INDEX] = rope_core_inline414 * 4 + it_inline149
                        ki_inline232: pl.Scalar[pl.INDEX] = g_idx_inline418 // 16
                        b_inline201: pl.Scalar[pl.INDEX] = g_idx_inline418 % 16
                        ctx_len_inline356: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [b_inline201])
                        inv_rms_b_inline420: pl.Scalar[pl.FP32] = pl.tensor.read(inv_rms_states_inline449, [b_inline201, 0])
                        pos_inline327: pl.Scalar[pl.INDEX] = pl.cast(ctx_len_inline356, pl.INDEX) - 1
                        wr_slot_inline239: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(slot_mapping, [b_inline201]), pl.INDEX)
                        wr_slot_block_inline421: pl.Scalar[pl.INDEX] = wr_slot_inline239 // 128
                        wr_slot_offset_inline295: pl.Scalar[pl.INDEX] = wr_slot_inline239 - wr_slot_block_inline421 * 128
                        cos_lo_inline306: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos, [1, 64], [pos_inline327, 0])
                        cos_hi_inline456: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos, [1, 64], [pos_inline327, 64])
                        sin_lo_inline424: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin, [1, 64], [pos_inline327, 0])
                        sin_hi_inline427: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin, [1, 64], [pos_inline327, 64])
                        kv_col_inline233: pl.Scalar[pl.INDEX] = ki_inline232 * 128
                        k_inv_b_inline224: pl.Scalar[pl.FP32] = pl.tensor.read(k_inv_states_inline204, [ki_inline232 * 16 + b_inline201, 0])
                        k_full_inline428: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(pl.tensor.slice(k_proj_norm_inline171, [1, 128], [b_inline201, kv_col_inline233]), k_inv_b_inline224)
                        k_lo_inline382: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full_inline428, [1, 64], [0, 0])
                        k_hi_inline431: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_full_inline428, [1, 64], [0, 64])
                        rot_lo_inline363: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.sub(pl.tensor.col_expand_mul(k_lo_inline382, cos_lo_inline306), pl.tensor.col_expand_mul(k_hi_inline431, sin_lo_inline424))
                        rot_hi_inline298: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(k_hi_inline431, cos_hi_inline456), pl.tensor.col_expand_mul(k_lo_inline382, sin_hi_inline427))
                        cache_row_inline390: pl.Scalar[pl.INDEX] = layer_cache_base_inline156 + (wr_slot_block_inline421 * 8 + ki_inline232) * 128 + wr_slot_offset_inline295
                        k_cache: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache, pl.tensor.cast(rot_lo_inline363, target_type=pl.BF16, mode='round'), [cache_row_inline390, 0])
                        k_cache: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache, pl.tensor.cast(rot_hi_inline298, target_type=pl.BF16, mode='round'), [cache_row_inline390, 64])
                        v_row_bf16_inline147: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(pl.tensor.muls(pl.tensor.slice(v_proj_inline241, [1, 128], [b_inline201, ki_inline232 * 128]), inv_rms_b_inline420), target_type=pl.BF16, mode='round')
                        v_cache: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(v_cache, v_row_bf16_inline147, [cache_row_inline390, 0])
                        q_base_inline435: pl.Scalar[pl.INDEX] = ki_inline232 * 5
                        q_pad_row0_inline251: pl.Scalar[pl.INDEX] = b_inline201 * 8 * 16 + ki_inline232 * 16
                        q_inv_base_inline261: pl.Scalar[pl.INDEX] = ki_inline232 * 16 * 5 + b_inline201 * 5
                        for qj_inline432 in pl.range(5):
                            q_inv_bj_inline433: pl.Scalar[pl.FP32] = pl.tensor.read(q_inv_states_inline168, [q_inv_base_inline261 + qj_inline432, 0])
                            q_head_inline407: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(pl.tensor.slice(q_proj_norm_inline375, [1, 128], [b_inline201, (q_base_inline435 + qj_inline432) * 128]), q_inv_bj_inline433)
                            q_lo_inline242: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_head_inline407, [1, 64], [0, 0])
                            q_hi_inline436: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_head_inline407, [1, 64], [0, 64])
                            q_rot_lo_inline269: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.sub(pl.tensor.col_expand_mul(q_lo_inline242, cos_lo_inline306), pl.tensor.col_expand_mul(q_hi_inline436, sin_lo_inline424))
                            q_rot_hi_inline471: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(q_hi_inline436, cos_hi_inline456), pl.tensor.col_expand_mul(q_lo_inline242, sin_hi_inline427))
                            all_q_padded_inline257: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded_inline257, pl.tensor.cast(q_rot_lo_inline269, target_type=pl.BF16, mode='round'), [q_pad_row0_inline251 + qj_inline432, 0])
                            all_q_padded_inline257: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded_inline257, pl.tensor.cast(q_rot_hi_inline471, target_type=pl.BF16, mode='round'), [q_pad_row0_inline251 + qj_inline432, 64])
                        q_pad_zero_inline336: pl.Tensor[[11, 128], pl.BF16] = pl.tensor.cast(pl.tensor.full([11, 128], dtype=pl.FP32, value=0.0), target_type=pl.BF16, mode='round')
                        all_q_padded_inline257: pl.Tensor[[2048, 128], pl.BF16] = pl.tensor.assemble(all_q_padded_inline257, q_pad_zero_inline336, [q_pad_row0_inline251 + 5, 0])
                rope_grp_tids_inline266: pl.Array[2, pl.TASK_ID] = pl.array.update_element(rope_grp_tids_inline266, 0, rope_tid_inline411)
                down_acc_all_inline439: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                gate_acc_all_inline234: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.create([16, 17408], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                up_acc_all_inline284: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.create([16, 17408], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                _submit_deps_buf_inline367: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
                _submit_deps_buf_inline396: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline367, 0, pl.array.get_element(carry_tids, 0))
                _submit_deps_buf_inline442: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline396, 1, pl.array.get_element(carry_tids, 1))
                _submit_deps_buf_inline445: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline442, 2, pl.array.get_element(carry_tids, 2))
                _submit_deps_buf_inline446: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline445, 3, pl.array.get_element(carry_tids, 3))
                _submit_deps_buf_inline447: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline446, 4, pl.array.get_element(carry_tids, 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_seed", deps=[_submit_deps_buf_inline447]) as seed_tid_inline417:
                    for nb_inline451 in pl.pipeline(5, stage=2):
                        n0_inline352: pl.Scalar[pl.INDEX] = nb_inline451 * 1024
                        zero_inline273: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                        down_acc_all_inline439: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(down_acc_all_inline439, zero_inline273, [0, n0_inline352])
                _submit_deps_buf_inline452: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
                _submit_deps_buf_inline453: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline452, 0, pl.array.get_element(carry_tids, 0))
                _submit_deps_buf_inline209: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline453, 1, pl.array.get_element(carry_tids, 1))
                _submit_deps_buf_inline426: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline209, 2, pl.array.get_element(carry_tids, 2))
                _submit_deps_buf_inline366: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline426, 3, pl.array.get_element(carry_tids, 3))
                _submit_deps_buf_inline454: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline366, 4, pl.array.get_element(carry_tids, 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_seed", deps=[_submit_deps_buf_inline454]) as gate_seed_tid_inline198:
                    for nb_inline457 in pl.pipeline(17, stage=2):
                        n0_inline352: pl.Scalar[pl.INDEX] = nb_inline457 * 1024
                        zero_inline273: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                        gate_acc_all_inline234: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(gate_acc_all_inline234, zero_inline273, [0, n0_inline352])
                _submit_deps_buf_inline458: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
                _submit_deps_buf_inline460: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline458, 0, pl.array.get_element(carry_tids, 0))
                _submit_deps_buf_inline192: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline460, 1, pl.array.get_element(carry_tids, 1))
                _submit_deps_buf_inline191: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline192, 2, pl.array.get_element(carry_tids, 2))
                _submit_deps_buf_inline189: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline191, 3, pl.array.get_element(carry_tids, 3))
                _submit_deps_buf_inline419: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline189, 4, pl.array.get_element(carry_tids, 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_seed", deps=[_submit_deps_buf_inline419]) as up_seed_tid_inline285:
                    for nb_inline374 in pl.pipeline(17, stage=2):
                        n0_inline352: pl.Scalar[pl.INDEX] = nb_inline374 * 1024
                        zero_inline273: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.full([16, 1024], dtype=pl.FP32, value=0.0)
                        up_acc_all_inline284: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(up_acc_all_inline284, zero_inline273, [0, n0_inline352])
                _submit_deps_buf_inline462: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                _submit_deps_buf_inline237: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline462, 0, work_tid_inline183)
                _submit_deps_buf_inline443: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline237, 1, pl.array.get_element(rope_grp_tids_inline266, 0))
                with pl.spmd(24, name_hint="fa_fused_spmd", optimizations=[pl.split(pl.SplitMode.UP_DOWN)], deps=[_submit_deps_buf_inline443]) as fa_tid_inline464:
                    fa_core_inline466: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                    fa_total_blocks_inline368: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(fa_total_inline262, [0, 0]), pl.INDEX)
                    for fa_w_inline322 in pl.range(fa_core_inline466, fa_total_blocks_inline368, 24):
                        fa_enc_inline416: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(fa_work_table_inline281, [fa_w_inline322, 0]), pl.INDEX)
                        fa_b_inline256: pl.Scalar[pl.INDEX] = fa_enc_inline416 // 32
                        fa_p_inline450: pl.Scalar[pl.INDEX] = fa_enc_inline416 % 32
                        fa_hg_inline469: pl.Scalar[pl.INDEX] = 0
                        fa_ctx_len_inline472: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [fa_b_inline256])
                        sb_inline258: pl.Scalar[pl.INDEX] = fa_p_inline450
                        s0_inline220: pl.Scalar[pl.INDEX] = sb_inline258 * 128
                        valid_len_inline474: pl.Scalar[pl.INDEX] = pl.min(128, pl.cast(fa_ctx_len_inline472, pl.INDEX) - s0_inline220)
                        fa_pbid_inline243: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(block_table, [fa_b_inline256 * max_blocks_per_seq_inline401 + sb_inline258]), pl.INDEX)
                        for gp_inline161 in pl.pipeline(8, stage=2):
                            gi_inline477: pl.Scalar[pl.INDEX] = fa_hg_inline469 * 8 + gp_inline161
                            kvh_inline369: pl.Scalar[pl.INDEX] = gi_inline477
                            q_pad_row_g_inline289: pl.Scalar[pl.INDEX] = fa_b_inline256 * 8 * 16 + gi_inline477 * 16
                            q_padded_inline259: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.slice(all_q_padded_inline257, [16, 128], [q_pad_row_g_inline289, 0])
                            g_base_inline478: pl.Scalar[pl.INDEX] = (fa_b_inline256 * 8 + gi_inline477) * 32 * 16
                            cache_row_inline390: pl.Scalar[pl.INDEX] = layer_cache_base_inline156 + (fa_pbid_inline243 * 8 + kvh_inline369) * 128
                            k_tile_inline279: pl.Tensor[[128, 128], pl.BF16] = pl.tensor.slice(k_cache, [128, 128], [cache_row_inline390, 0])
                            raw_scores_inline475: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(q_padded_inline259, k_tile_inline279, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                            scores_scaled_inline260: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.muls(raw_scores_inline475, 0.088388347648318433)
                            scores_valid_inline481: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, valid_len], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(scores_scaled_inline260, 5, valid_len_inline474)
                            scores_inline482: pl.Tensor[[16, 128], pl.FP32, pl.TensorView()] = pl.tensor.fillpad(scores_valid_inline481, pad_value=pl.PadValue.min)
                            cur_mi_inline365: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_max(scores_inline482)
                            exp_scores_inline199: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(scores_inline482, cur_mi_inline365))
                            cur_li_inline484: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(exp_scores_inline199)
                            exp_scores_bf16_inline211: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.cast(exp_scores_inline199, target_type=pl.BF16, mode='round')
                            v_tile_inline143: pl.Tensor[[128, 128], pl.BF16] = pl.tensor.slice(v_cache, [128, 128], [cache_row_inline390, 0])
                            oi_tmp_inline470: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(exp_scores_bf16_inline211, v_tile_inline143, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            oi_tmp_v1_inline216: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(oi_tmp_inline470, 5, 128)
                            all_oi_tmp_inline318: pl.Tensor[[65536, 128], pl.FP32] = pl.tensor.assemble(all_oi_tmp_inline318, oi_tmp_v1_inline216, [g_base_inline478 + sb_inline258 * 16, 0])
                            all_cur_mi_inline409: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.assemble(all_cur_mi_inline409, cur_mi_inline365, [g_base_inline478 + sb_inline258 * 16, 0])
                            all_cur_li_inline294: pl.Tensor[[65536, 1], pl.FP32] = pl.tensor.assemble(all_cur_li_inline294, cur_li_inline484, [g_base_inline478 + sb_inline258 * 16, 0])
                with pl.spmd(48, name_hint="online_softmax_spmd", deps=[fa_tid_inline464]) as attn_done_tid_inline142:
                    os_core_inline141: pl.Scalar[pl.INDEX] = pl.tensor.get_block_idx()
                    for os_spmd_idx_inline140 in pl.range(os_core_inline141, 128, 48):
                        os_b_inline288: pl.Scalar[pl.INDEX] = os_spmd_idx_inline140 // 8
                        os_gi_inline344: pl.Scalar[pl.INDEX] = os_spmd_idx_inline140 % 8
                        os_ctx_len_inline139: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [os_b_inline288])
                        os_ctx_blocks_inline138: pl.Scalar[pl.INDEX] = (pl.cast(os_ctx_len_inline139, pl.INDEX) + 128 - 1) // 128
                        os_kvh_inline392: pl.Scalar[pl.INDEX] = os_gi_inline344
                        os_q_base_inline137: pl.Scalar[pl.INDEX] = os_kvh_inline392 * 5
                        os_g_base_inline136: pl.Scalar[pl.INDEX] = (os_b_inline288 * 8 + os_gi_inline344) * 32 * 16
                        oi_inline357: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.slice(all_oi_tmp_inline318, [16, 128], [os_g_base_inline136, 0])
                        mi_inline330: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(all_cur_mi_inline409, [16, 1], [os_g_base_inline136, 0])
                        li_inline135: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(all_cur_li_inline294, [16, 1], [os_g_base_inline136, 0])
                        for sb_inline134 in pl.pipeline(1, os_ctx_blocks_inline138, stage=2):
                            rec_inline133: pl.Scalar[pl.INDEX] = os_g_base_inline136 + sb_inline134 * 16
                            oi_tmp_valid_inline132: pl.Tensor[[16, 128], pl.FP32, pl.TensorView(valid_shape=[5, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_oi_tmp_inline318, [16, 128], [rec_inline133, 0], [5, 128])
                            online_cur_mi_inline131: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[5, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_cur_mi_inline409, [16, 1], [rec_inline133, 0], [5, 1])
                            online_cur_li_inline175: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[5, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(all_cur_li_inline294, [16, 1], [rec_inline133, 0], [5, 1])
                            mi_new_inline130: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.maximum(mi_inline330, online_cur_mi_inline131)
                            alpha_inline423: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline330, mi_new_inline130))
                            beta_inline228: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(online_cur_mi_inline131, mi_new_inline130))
                            li_inline135: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha_inline423, li_inline135), pl.tensor.mul(beta_inline228, online_cur_li_inline175))
                            oi_inline357: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(oi_inline357, alpha_inline423), pl.tensor.row_expand_mul(oi_tmp_valid_inline132, beta_inline228))
                            mi_inline330: pl.Tensor[[16, 1], pl.FP32] = mi_new_inline130
                        ctx_inline167: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_div(oi_inline357, li_inline135)
                        ctx_valid_inline128: pl.Tensor[[5, 128], pl.FP32] = pl.tensor.slice(ctx_inline167, [5, 128], [0, 0])
                        ctx_flat_bf16_inline127: pl.Tensor[[1, 640], pl.BF16] = pl.tensor.cast(pl.tensor.reshape(ctx_valid_inline128, [1, 640]), target_type=pl.BF16, mode='round')
                        attn_out_inline203: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(attn_out_inline203, ctx_flat_bf16_inline127, [os_b_inline288, os_q_base_inline137 * 128])
                attn_proj_fp32_inline468: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                post_norm_partial_inline126: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                mlp_norm_in_inline355: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                inv_rms_tile_inline123: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                mlp_tile_inline122: pl.Tensor[[16, 17408], pl.BF16] = pl.tensor.create([16, 17408], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                _submit_deps_buf_inline121: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
                _submit_deps_buf_inline119: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline121, 0, pl.array.get_element(carry_tids, 0))
                _submit_deps_buf_inline117: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline119, 1, pl.array.get_element(carry_tids, 1))
                _submit_deps_buf_inline114: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline117, 2, pl.array.get_element(carry_tids, 2))
                _submit_deps_buf_inline112: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline114, 3, pl.array.get_element(carry_tids, 3))
                _submit_deps_buf_inline179: pl.Array[5, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline112, 4, pl.array.get_element(carry_tids, 4))
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_seed", deps=[_submit_deps_buf_inline179]) as out_seed_tid_inline297:
                    for nb_inline274 in pl.pipeline(10, stage=2):
                        out_seed_n0_inline110: pl.Scalar[pl.INDEX] = nb_inline274 * 512
                        out_zero_inline444: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                        attn_proj_fp32_inline468: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_proj_fp32_inline468, out_zero_inline444, [0, out_seed_n0_inline110])
                silu_tids_inline109: pl.Array[17, pl.TASK_ID] = pl.array.create(17, dtype=pl.TASK_ID)
                gate_tids_inline108: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
                up_tids_inline107: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
                cast_tids_inline391: pl.Array[5, pl.TASK_ID] = pl.array.create(5, dtype=pl.TASK_ID)
                out_tids_inline104: pl.Array[50, pl.TASK_ID] = pl.array.create(50, dtype=pl.TASK_ID)
                for n_out_proj_inline103 in pl.parallel(10):
                    n_op_inline102: pl.Scalar[pl.INDEX] = n_out_proj_inline103 * 512
                    for k_split_out_inline100 in pl.range(5):
                        k_op_inline360: pl.Scalar[pl.INDEX] = k_split_out_inline100 * 1024
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj", deps=[out_seed_tid_inline297, attn_done_tid_inline142]) as out_tid_inline155:
                            out_a0_inline308: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(attn_out_inline203, [16, 64], [0, k_op_inline360])
                            out_w0_inline98: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wo, [64, 512], [layer_hidden_base_inline217 + k_op_inline360, n_op_inline102])
                            out_c_acc_inline97: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.matmul(out_a0_inline308, out_w0_inline98, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for out_lk_inline96 in pl.pipeline(1, 16, stage=2):
                                out_ks_off_inline94: pl.Scalar[pl.INDEX] = out_lk_inline96 * 64
                                out_a_k_inline329: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(attn_out_inline203, [16, 64], [0, k_op_inline360 + out_ks_off_inline94])
                                out_w_k_inline92: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wo, [64, 512], [layer_hidden_base_inline217 + k_op_inline360 + out_ks_off_inline94, n_op_inline102])
                                out_c_acc_inline97: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.matmul_acc(out_c_acc_inline97, out_a_k_inline329, out_w_k_inline92, a_trans=False, b_trans=False)
                            attn_proj_fp32_inline468: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_proj_fp32_inline468, out_c_acc_inline97, [0, n_op_inline102], atomic=pl.AtomicType.Add)
                        out_tids_inline104: pl.Array[50, pl.TASK_ID] = pl.array.update_element(out_tids_inline104, n_out_proj_inline103 * 5 + k_split_out_inline100, out_tid_inline155)
                for k_slice_inline90 in pl.unroll(5):
                    k_base_inline89: pl.Scalar[pl.INDEX] = k_slice_inline90 * 1024
                    n_split_base_inline465: pl.Scalar[pl.INDEX] = k_slice_inline90 * 2
                    _submit_deps_buf_inline88: pl.Array[10, pl.TASK_ID] = pl.array.create(10, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline87: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline88, 0, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 0) * 5 + 0))
                    _submit_deps_buf_inline86: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline87, 1, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 0) * 5 + 1))
                    _submit_deps_buf_inline85: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline86, 2, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 0) * 5 + 2))
                    _submit_deps_buf_inline84: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline85, 3, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 0) * 5 + 3))
                    _submit_deps_buf_inline95: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline84, 4, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 0) * 5 + 4))
                    _submit_deps_buf_inline93: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline95, 5, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 1) * 5 + 0))
                    _submit_deps_buf_inline83: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline93, 6, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 1) * 5 + 1))
                    _submit_deps_buf_inline82: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline83, 7, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 1) * 5 + 2))
                    _submit_deps_buf_inline463: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline82, 8, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 1) * 5 + 3))
                    _submit_deps_buf_inline81: pl.Array[10, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline463, 9, pl.array.get_element(out_tids_inline104, (n_split_base_inline465 + 1) * 5 + 4))
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual_rms_cast", deps=[_submit_deps_buf_inline81]) as cast_tid_k_inline79:
                        for kb_inline77 in pl.pipeline(4, stage=2):
                            k0_v1_inline395: pl.Scalar[pl.INDEX] = k_base_inline89 + kb_inline77 * 256
                            attn_chunk_inline173: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32_inline468, [16, 256], [0, k0_v1_inline395])
                            hidden_chunk_inline351: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur, [16, 256], [0, k0_v1_inline395])
                            resid_fp32_inline286: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk_inline173, hidden_chunk_inline351)
                            post_norm_partial_inline126: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(post_norm_partial_inline126, resid_fp32_inline286, [0, k0_v1_inline395])
                            post_gamma_inline80: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight, [1, 256], [i, k0_v1_inline395])
                            mlp_norm_in_inline355: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(mlp_norm_in_inline355, pl.tensor.cast(pl.tensor.col_expand_mul(resid_fp32_inline286, post_gamma_inline80), target_type=pl.BF16, mode='round'), [0, k0_v1_inline395])
                    cast_tids_inline391: pl.Array[5, pl.TASK_ID] = pl.array.update_element(cast_tids_inline391, k_slice_inline90, cast_tid_k_inline79)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="post_rms_reduce", deps=[out_tids_inline104]) as reduce_tid_inline76:
                    sq_sum_inline75: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                    for kb_inline74 in pl.pipeline(20, stage=2):
                        k0_inline321: pl.Scalar[pl.INDEX] = kb_inline74 * 256
                        attn_chunk_inline173: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(attn_proj_fp32_inline468, [16, 256], [0, k0_inline321])
                        hidden_chunk_inline351: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(cur, [16, 256], [0, k0_inline321])
                        resid_chunk_inline73: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.add(attn_chunk_inline173, hidden_chunk_inline351)
                        sq_sum_inline75: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(sq_sum_inline75, pl.tensor.reshape(pl.tensor.row_sum(pl.tensor.mul(resid_chunk_inline73, resid_chunk_inline73)), [1, 16]))
                    post_inv_rms_inline303: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(pl.tensor.adds(pl.tensor.muls(sq_sum_inline75, 0.00019531250000000001), 9.9999999999999995e-07)))
                    post_inv_rms_col_inline334: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(post_inv_rms_inline303, [16, 1])
                    inv_rms_tile_inline123: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.assemble(inv_rms_tile_inline123, post_inv_rms_col_inline334, [0, 0])
                for n_out_inline346 in pl.parallel(17):
                    n0_inline352: pl.Scalar[pl.INDEX] = n_out_inline346 * 1024
                    for k_split_inline387 in pl.range(5):
                        k0_inline321: pl.Scalar[pl.INDEX] = k_split_inline387 * 1024
                        _submit_deps_buf_inline72: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                        _submit_deps_buf_inline70: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline72, 0, pl.array.get_element(cast_tids_inline391, k_split_inline387))
                        _submit_deps_buf_inline69: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline70, 1, gate_seed_tid_inline198)
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_proj", deps=[_submit_deps_buf_inline69]) as gate_tid_inline185:
                            a0_inline290: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in_inline355, [16, 64], [0, k0_inline321])
                            w0_inline67: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_gate, [64, 1024], [layer_hidden_base_inline217 + k0_inline321, n0_inline352])
                            c_acc_inline459: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0_inline290, w0_inline67, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for lk_inline270 in pl.pipeline(1, 16, stage=2):
                                ks_off_inline66: pl.Scalar[pl.INDEX] = lk_inline270 * 64
                                a_k_inline364: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in_inline355, [16, 64], [0, k0_inline321 + ks_off_inline66])
                                w_k_inline65: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_gate, [64, 1024], [layer_hidden_base_inline217 + k0_inline321 + ks_off_inline66, n0_inline352])
                                c_acc_inline459: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc_inline459, a_k_inline364, w_k_inline65, a_trans=False, b_trans=False)
                            gate_acc_all_inline234: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(gate_acc_all_inline234, c_acc_inline459, [0, n0_inline352], atomic=pl.AtomicType.Add)
                        gate_tids_inline108: pl.Array[85, pl.TASK_ID] = pl.array.update_element(gate_tids_inline108, n_out_inline346 * 5 + k_split_inline387, gate_tid_inline185)
                        _submit_deps_buf_inline146: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                        _submit_deps_buf_inline64: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline146, 0, pl.array.get_element(cast_tids_inline391, k_split_inline387))
                        _submit_deps_buf_inline62: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline64, 1, up_seed_tid_inline285)
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="up_proj", deps=[_submit_deps_buf_inline62]) as up_tid_inline415:
                            a0_v1_inline61: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in_inline355, [16, 64], [0, k0_inline321])
                            w0_v1_inline59: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_up, [64, 1024], [layer_hidden_base_inline217 + k0_inline321, n0_inline352])
                            c_acc_v1_inline377: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0_v1_inline61, w0_v1_inline59, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for lk_inline106 in pl.pipeline(1, 16, stage=2):
                                ks_off_inline66: pl.Scalar[pl.INDEX] = lk_inline106 * 64
                                a_k_inline364: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_norm_in_inline355, [16, 64], [0, k0_inline321 + ks_off_inline66])
                                w_k_inline65: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_up, [64, 1024], [layer_hidden_base_inline217 + k0_inline321 + ks_off_inline66, n0_inline352])
                                c_acc_v1_inline377: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc_v1_inline377, a_k_inline364, w_k_inline65, a_trans=False, b_trans=False)
                            up_acc_all_inline284: pl.Tensor[[16, 17408], pl.FP32] = pl.tensor.assemble(up_acc_all_inline284, c_acc_v1_inline377, [0, n0_inline352], atomic=pl.AtomicType.Add)
                        up_tids_inline107: pl.Array[85, pl.TASK_ID] = pl.array.update_element(up_tids_inline107, n_out_inline346 * 5 + k_split_inline387, up_tid_inline415)
                for n_out_inline58 in pl.parallel(17):
                    n0_inline352: pl.Scalar[pl.INDEX] = n_out_inline58 * 1024
                    _submit_deps_buf_inline480: pl.Array[11, pl.TASK_ID] = pl.array.create(11, dtype=pl.TASK_ID)
                    _submit_deps_buf_inline328: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline480, 0, reduce_tid_inline76)
                    _submit_deps_buf_inline207: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline328, 1, pl.array.get_element(gate_tids_inline108, n_out_inline58 * 5 + 0))
                    _submit_deps_buf_inline425: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline207, 2, pl.array.get_element(gate_tids_inline108, n_out_inline58 * 5 + 1))
                    _submit_deps_buf_inline57: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline425, 3, pl.array.get_element(gate_tids_inline108, n_out_inline58 * 5 + 2))
                    _submit_deps_buf_inline56: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline57, 4, pl.array.get_element(gate_tids_inline108, n_out_inline58 * 5 + 3))
                    _submit_deps_buf_inline55: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline56, 5, pl.array.get_element(gate_tids_inline108, n_out_inline58 * 5 + 4))
                    _submit_deps_buf_inline54: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline55, 6, pl.array.get_element(up_tids_inline107, n_out_inline58 * 5 + 0))
                    _submit_deps_buf_inline53: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline54, 7, pl.array.get_element(up_tids_inline107, n_out_inline58 * 5 + 1))
                    _submit_deps_buf_inline52: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline53, 8, pl.array.get_element(up_tids_inline107, n_out_inline58 * 5 + 2))
                    _submit_deps_buf_inline381: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline52, 9, pl.array.get_element(up_tids_inline107, n_out_inline58 * 5 + 3))
                    _submit_deps_buf_inline51: pl.Array[11, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline381, 10, pl.array.get_element(up_tids_inline107, n_out_inline58 * 5 + 4))
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu", deps=[_submit_deps_buf_inline51]) as silu_tid_inline265:
                        inv_rms_chunk_inline50: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(inv_rms_tile_inline123, [16, 1], [0, 0])
                        for sub_inline48 in pl.pipeline(4, stage=2):
                            silu_off_inline63: pl.Scalar[pl.INDEX] = n0_inline352 + sub_inline48 * 256
                            gate_chunk_inline255: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(gate_acc_all_inline234, [16, 256], [0, silu_off_inline63])
                            up_chunk_inline46: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(up_acc_all_inline284, [16, 256], [0, silu_off_inline63])
                            scaled_gate_inline125: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.row_expand_mul(gate_chunk_inline255, inv_rms_chunk_inline50)
                            scaled_up_inline479: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.row_expand_mul(up_chunk_inline46, inv_rms_chunk_inline50)
                            sigmoid_inline293: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(scaled_gate_inline125)), 1.0))
                            mlp_chunk_inline45: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.mul(pl.tensor.mul(scaled_gate_inline125, sigmoid_inline293), scaled_up_inline479)
                            mlp_tile_inline122: pl.Tensor[[16, 17408], pl.BF16] = pl.tensor.assemble(mlp_tile_inline122, pl.tensor.cast(mlp_chunk_inline45, target_type=pl.BF16, mode='round'), [0, silu_off_inline63])
                    silu_tids_inline109: pl.Array[17, pl.TASK_ID] = pl.array.update_element(silu_tids_inline109, n_out_inline58, silu_tid_inline265)
                for n_out_inline43 in pl.parallel(5):
                    n0_inline352: pl.Scalar[pl.INDEX] = n_out_inline43 * 1024
                    for k_split_inline42 in pl.range(17):
                        k0_inline321: pl.Scalar[pl.INDEX] = k_split_inline42 * 1024
                        _submit_deps_buf_inline41: pl.Array[2, pl.TASK_ID] = pl.array.create(2, dtype=pl.TASK_ID)
                        _submit_deps_buf_inline40: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline41, 0, seed_tid_inline417)
                        _submit_deps_buf_inline430: pl.Array[2, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline40, 1, pl.array.get_element(silu_tids_inline109, k_split_inline42))
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="down_proj", deps=[_submit_deps_buf_inline430]) as down_tid_inline116:
                            a0_v2_inline39: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_tile_inline122, [16, 64], [0, k0_inline321])
                            w0_v2_inline311: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_down, [64, 1024], [layer_inter_base_inline296 + k0_inline321, n0_inline352])
                            c_acc_v2_inline38: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul(a0_v2_inline39, w0_v2_inline311, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for lk_inline422 in pl.pipeline(1, 16, stage=2):
                                ks_off_inline66: pl.Scalar[pl.INDEX] = lk_inline422 * 64
                                a_k_inline364: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.slice(mlp_tile_inline122, [16, 64], [0, k0_inline321 + ks_off_inline66])
                                w_k_inline65: pl.Tensor[[64, 1024], pl.BF16] = pl.tensor.slice(w_down, [64, 1024], [layer_inter_base_inline296 + k0_inline321 + ks_off_inline66, n0_inline352])
                                c_acc_v2_inline38: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.matmul_acc(c_acc_v2_inline38, a_k_inline364, w_k_inline65, a_trans=False, b_trans=False)
                            down_acc_all_inline439: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(down_acc_all_inline439, c_acc_v2_inline38, [0, n0_inline352], atomic=pl.AtomicType.Add)
                        down_tids_inline404: pl.Array[85, pl.TASK_ID] = pl.array.update_element(down_tids_inline404, n_out_inline43 * 17 + k_split_inline42, down_tid_inline116)
            _submit_deps_buf_inline36: pl.Array[85, pl.TASK_ID] = pl.array.create(85, dtype=pl.TASK_ID)
            _submit_deps_buf_inline162: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline36, 0, pl.array.get_element(down_tids_inline404, 0))
            _submit_deps_buf_inline429: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline162, 1, pl.array.get_element(down_tids_inline404, 1))
            _submit_deps_buf_inline172: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline429, 2, pl.array.get_element(down_tids_inline404, 2))
            _submit_deps_buf_inline467: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline172, 3, pl.array.get_element(down_tids_inline404, 3))
            _submit_deps_buf_inline35: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline467, 4, pl.array.get_element(down_tids_inline404, 4))
            _submit_deps_buf_inline68: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline35, 5, pl.array.get_element(down_tids_inline404, 5))
            _submit_deps_buf_inline386: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline68, 6, pl.array.get_element(down_tids_inline404, 6))
            _submit_deps_buf_inline37: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline386, 7, pl.array.get_element(down_tids_inline404, 7))
            _submit_deps_buf_inline34: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline37, 8, pl.array.get_element(down_tids_inline404, 8))
            _submit_deps_buf_inline60: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline34, 9, pl.array.get_element(down_tids_inline404, 9))
            _submit_deps_buf_inline33: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline60, 10, pl.array.get_element(down_tids_inline404, 10))
            _submit_deps_buf_inline164: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline33, 11, pl.array.get_element(down_tids_inline404, 11))
            _submit_deps_buf_inline32: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline164, 12, pl.array.get_element(down_tids_inline404, 12))
            _submit_deps_buf_inline31: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline32, 13, pl.array.get_element(down_tids_inline404, 13))
            _submit_deps_buf_inline333: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline31, 14, pl.array.get_element(down_tids_inline404, 14))
            _submit_deps_buf_inline30: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline333, 15, pl.array.get_element(down_tids_inline404, 15))
            _submit_deps_buf_inline29: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline30, 16, pl.array.get_element(down_tids_inline404, 16))
            _submit_deps_buf_inline28: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline29, 17, pl.array.get_element(down_tids_inline404, 17))
            _submit_deps_buf_inline205: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline28, 18, pl.array.get_element(down_tids_inline404, 18))
            _submit_deps_buf_inline27: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline205, 19, pl.array.get_element(down_tids_inline404, 19))
            _submit_deps_buf_inline49: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline27, 20, pl.array.get_element(down_tids_inline404, 20))
            _submit_deps_buf_inline47: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline49, 21, pl.array.get_element(down_tids_inline404, 21))
            _submit_deps_buf_inline101: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline47, 22, pl.array.get_element(down_tids_inline404, 22))
            _submit_deps_buf_inline99: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline101, 23, pl.array.get_element(down_tids_inline404, 23))
            _submit_deps_buf_inline26: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline99, 24, pl.array.get_element(down_tids_inline404, 24))
            _submit_deps_buf_inline25: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline26, 25, pl.array.get_element(down_tids_inline404, 25))
            _submit_deps_buf_inline91: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline25, 26, pl.array.get_element(down_tids_inline404, 26))
            _submit_deps_buf_inline24: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline91, 27, pl.array.get_element(down_tids_inline404, 27))
            _submit_deps_buf_inline373: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline24, 28, pl.array.get_element(down_tids_inline404, 28))
            _submit_deps_buf_inline434: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline373, 29, pl.array.get_element(down_tids_inline404, 29))
            _submit_deps_buf_inline210: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline434, 30, pl.array.get_element(down_tids_inline404, 30))
            _submit_deps_buf_inline78: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline210, 31, pl.array.get_element(down_tids_inline404, 31))
            _submit_deps_buf_inline23: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline78, 32, pl.array.get_element(down_tids_inline404, 32))
            _submit_deps_buf_inline22: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline23, 33, pl.array.get_element(down_tids_inline404, 33))
            _submit_deps_buf_inline21: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline22, 34, pl.array.get_element(down_tids_inline404, 34))
            _submit_deps_buf_inline403: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline21, 35, pl.array.get_element(down_tids_inline404, 35))
            _submit_deps_buf_inline412: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline403, 36, pl.array.get_element(down_tids_inline404, 36))
            _submit_deps_buf_inline157: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline412, 37, pl.array.get_element(down_tids_inline404, 37))
            _submit_deps_buf_inline129: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline157, 38, pl.array.get_element(down_tids_inline404, 38))
            _submit_deps_buf_inline159: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline129, 39, pl.array.get_element(down_tids_inline404, 39))
            _submit_deps_buf_inline71: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline159, 40, pl.array.get_element(down_tids_inline404, 40))
            _submit_deps_buf_inline20: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline71, 41, pl.array.get_element(down_tids_inline404, 41))
            _submit_deps_buf_inline120: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline20, 42, pl.array.get_element(down_tids_inline404, 42))
            _submit_deps_buf_inline118: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline120, 43, pl.array.get_element(down_tids_inline404, 43))
            _submit_deps_buf_inline115: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline118, 44, pl.array.get_element(down_tids_inline404, 44))
            _submit_deps_buf_inline113: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline115, 45, pl.array.get_element(down_tids_inline404, 45))
            _submit_deps_buf_inline111: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline113, 46, pl.array.get_element(down_tids_inline404, 46))
            _submit_deps_buf_inline178: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline111, 47, pl.array.get_element(down_tids_inline404, 47))
            _submit_deps_buf_inline184: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline178, 48, pl.array.get_element(down_tids_inline404, 48))
            _submit_deps_buf_inline19: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline184, 49, pl.array.get_element(down_tids_inline404, 49))
            _submit_deps_buf_inline18: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline19, 50, pl.array.get_element(down_tids_inline404, 50))
            _submit_deps_buf_inline354: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline18, 51, pl.array.get_element(down_tids_inline404, 51))
            _submit_deps_buf_inline17: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline354, 52, pl.array.get_element(down_tids_inline404, 52))
            _submit_deps_buf_inline16: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline17, 53, pl.array.get_element(down_tids_inline404, 53))
            _submit_deps_buf_inline15: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline16, 54, pl.array.get_element(down_tids_inline404, 54))
            _submit_deps_buf_inline170: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline15, 55, pl.array.get_element(down_tids_inline404, 55))
            _submit_deps_buf_inline14: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline170, 56, pl.array.get_element(down_tids_inline404, 56))
            _submit_deps_buf_inline13: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline14, 57, pl.array.get_element(down_tids_inline404, 57))
            _submit_deps_buf_inline12: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline13, 58, pl.array.get_element(down_tids_inline404, 58))
            _submit_deps_buf_inline11: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline12, 59, pl.array.get_element(down_tids_inline404, 59))
            _submit_deps_buf_inline455: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline11, 60, pl.array.get_element(down_tids_inline404, 60))
            _submit_deps_buf_inline10: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline455, 61, pl.array.get_element(down_tids_inline404, 61))
            _submit_deps_buf_inline9: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline10, 62, pl.array.get_element(down_tids_inline404, 62))
            _submit_deps_buf_inline193: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline9, 63, pl.array.get_element(down_tids_inline404, 63))
            _submit_deps_buf_inline226: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline193, 64, pl.array.get_element(down_tids_inline404, 64))
            _submit_deps_buf_inline8: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline226, 65, pl.array.get_element(down_tids_inline404, 65))
            _submit_deps_buf_inline195: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline8, 66, pl.array.get_element(down_tids_inline404, 66))
            _submit_deps_buf_inline7: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline195, 67, pl.array.get_element(down_tids_inline404, 67))
            _submit_deps_buf_inline182: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline7, 68, pl.array.get_element(down_tids_inline404, 68))
            _submit_deps_buf_inline44: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline182, 69, pl.array.get_element(down_tids_inline404, 69))
            _submit_deps_buf_inline6: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline44, 70, pl.array.get_element(down_tids_inline404, 70))
            _submit_deps_buf_inline315: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline6, 71, pl.array.get_element(down_tids_inline404, 71))
            _submit_deps_buf_inline152: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline315, 72, pl.array.get_element(down_tids_inline404, 72))
            _submit_deps_buf_inline371: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline152, 73, pl.array.get_element(down_tids_inline404, 73))
            _submit_deps_buf_inline5: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline371, 74, pl.array.get_element(down_tids_inline404, 74))
            _submit_deps_buf_inline278: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline5, 75, pl.array.get_element(down_tids_inline404, 75))
            _submit_deps_buf_inline4: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline278, 76, pl.array.get_element(down_tids_inline404, 76))
            _submit_deps_buf_inline3: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline4, 77, pl.array.get_element(down_tids_inline404, 77))
            _submit_deps_buf_inline218: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline3, 78, pl.array.get_element(down_tids_inline404, 78))
            _submit_deps_buf_inline2: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline218, 79, pl.array.get_element(down_tids_inline404, 79))
            _submit_deps_buf_inline1: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline2, 80, pl.array.get_element(down_tids_inline404, 80))
            _submit_deps_buf_inline124: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline1, 81, pl.array.get_element(down_tids_inline404, 81))
            _submit_deps_buf_inline0: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline124, 82, pl.array.get_element(down_tids_inline404, 82))
            _submit_deps_buf_inline483: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline0, 83, pl.array.get_element(down_tids_inline404, 83))
            _submit_deps_buf_inline240: pl.Array[85, pl.TASK_ID] = pl.array.update_element(_submit_deps_buf_inline483, 84, pl.array.get_element(down_tids_inline404, 84))
            with pl.spmd(5, name_hint="dcr_xgamma_spmd", deps=[_submit_deps_buf_inline240]) as dcr_tid_inline359:
                n_out_inline43: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                n0_inline352: pl.Scalar[pl.INDEX] = n_out_inline43 * 1024
                out_chunk_inline105: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.add(pl.tensor.slice(down_acc_all_inline439, [16, 1024], [0, n0_inline352]), pl.tensor.slice(post_norm_partial_inline126, [16, 1024], [0, n0_inline352]))
                next_hidden: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(next_hidden, out_chunk_inline105, [0, n0_inline352])
                gamma_next_inline461: pl.Tensor[[1, 1024], pl.FP32] = pl.tensor.slice(input_rms_weight, [1, 1024], [next_gamma_idx, n0_inline352])
                xg_inline252: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.col_expand_mul(out_chunk_inline105, gamma_next_inline461)
                next_normed: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(next_normed, pl.tensor.cast(xg_inline252, target_type=pl.BF16, mode='round'), [0, n0_inline352])
            for _slab_inline337 in pl.unroll(5):
                carry_tids: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_tids, _slab_inline337, dcr_tid_inline359)
                carry_normed_tids: pl.Array[5, pl.TASK_ID] = pl.array.update_element(carry_normed_tids, _slab_inline337, dcr_tid_inline359)
            cur: pl.Tensor[[16, 5120], pl.FP32] = next_hidden
            normed: pl.Tensor[[16, 5120], pl.BF16] = next_normed
        for ob0 in pl.parallel(0, 16, 16):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_out"):
                for okb in pl.range(20):
                    ok0: pl.Scalar[pl.INDEX] = okb * 256
                    out: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(out, pl.tensor.cast(pl.tensor.slice(cur, [16, 256], [ob0, ok0]), target_type=pl.BF16, mode='round'), [ob0, ok0])
        return out
