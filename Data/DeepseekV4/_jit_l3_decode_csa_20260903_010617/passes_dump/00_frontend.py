# pypto.program: _jit_l3_decode_csa
import pypto.language as pl
import pypto.language.distributed as pld

B_DYN = pl.dynamic("B_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")
KV_B_DYN = pl.dynamic("KV_B_DYN")
KV_T_DYN = pl.dynamic("KV_T_DYN")
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic("MAIN_STATE_BLOCK_NUM_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
T_DYN = pl.dynamic("T_DYN")

@pl.program
class _jit_l3_decode_csa:
    @pl.function(type=pl.FunctionType.Inline)
    def _rms_norm_full_tile(self, x: pl.Tensor[[T_DYN, 4096], pl.BF16], norm_w: pl.Tensor[[4096], pl.BF16], x_normed: pl.Tensor[[T_DYN, 4096], pl.BF16]):
        # Run the aligned token tile through the existing Tensor-level dataflow.
        tg: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 8
        x_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
        for rms_db in pl.pipeline(32, stage=2):
            rms_d0: pl.Scalar[pl.INDEX] = rms_db * 128
            rms_x_input: pl.Tensor[[8, 128], pl.BF16] = pl.tensor.slice(x, [8, 128], [tg, rms_d0])
            rms_x_chunk: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(rms_x_input, target_type=pl.FP32, mode='round')
            rms_x_sq: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.mul(rms_x_chunk, rms_x_chunk)
            rms_x_row_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(rms_x_sq), [1, 8])
            x_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(x_sq_sum, rms_x_row_sum)
        x_inv_rms: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(x_sq_sum, 0.000244140625), 9.9999999999999995e-07), high_precision=True)
        x_inv_rms_t: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(x_inv_rms, [8, 1])
        for apply_db in pl.pipeline(32, stage=2):
            apply_d0: pl.Scalar[pl.INDEX] = apply_db * 128
            apply_x_input: pl.Tensor[[8, 128], pl.BF16] = pl.tensor.slice(x, [8, 128], [tg, apply_d0])
            apply_x_chunk: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(apply_x_input, target_type=pl.FP32, mode='round')
            norm_w_input: pl.Tensor[[128], pl.BF16] = pl.tensor.slice(norm_w, [128], [apply_d0])
            norm_w_chunk: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.reshape(norm_w_input, [1, 128]), target_type=pl.FP32, mode='round')
            x_scaled: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.row_expand_mul(apply_x_chunk, x_inv_rms_t)
            x_normed_chunk: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.col_expand_mul(x_scaled, norm_w_chunk)
            x_normed: pl.Tensor[[T_DYN, 4096], pl.BF16] = pl.tensor.assemble(x_normed, pl.tensor.cast(x_normed_chunk, target_type=pl.BF16, mode='rint'), [tg, apply_d0])
    @pl.function(type=pl.FunctionType.Inline)
    def _rms_norm_tail_tile(self, x: pl.Tensor[[T_DYN, 4096], pl.BF16], norm_w: pl.Tensor[[4096], pl.BF16], x_normed: pl.Tensor[[T_DYN, 4096], pl.BF16]):
        # Run the ragged last token tile through explicit `valid_shape` load/store.
        # 
        #         Step for step the same RMSNorm as `_rms_norm_full_tile`. The two live in
        #         separate scopes rather than in one `if`/`else` body because this path binds
        #         the shared names (`x_sq_sum`, `x_inv_rms`, …) to Vec-space Tiles while the
        #         aligned path binds them to Tensors, and a name cannot be rebound to a
        #         different type inside one kernel.
        #         
        tg: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 8
        valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, pl.tensor.dim(x, 0) - tg)
        row_reduce_tmp: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        x_sq_sum: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for rms_db in pl.pipeline(32, stage=2):
            rms_d0: pl.Scalar[pl.INDEX] = rms_db * 128
            rms_x_input: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.load(x, [tg, rms_d0], [8, 128], [valid_rows, 128], target_memory=pl.Mem.Vec)
            rms_x_chunk: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.cast(rms_x_input, target_type=pl.FP32, mode='round')
            rms_x_sq: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.mul(rms_x_chunk, rms_x_chunk)
            rms_x_row_sum: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows])] = pl.tile.reshape(pl.tile.row_sum(rms_x_sq, row_reduce_tmp), [1, 8])
            x_sq_sum: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(x_sq_sum, rms_x_row_sum)
        x_inv_rms: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(x_sq_sum, 0.000244140625), 9.9999999999999995e-07)))
        x_inv_rms_t: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(x_inv_rms, [8, 1])
        for apply_db in pl.pipeline(32, stage=2):
            apply_d0: pl.Scalar[pl.INDEX] = apply_db * 128
            apply_x_input: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.load(x, [tg, apply_d0], [8, 128], [valid_rows, 128], target_memory=pl.Mem.Vec)
            apply_x_chunk: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.cast(apply_x_input, target_type=pl.FP32, mode='round')
            norm_w_input: pl.Tile[[128], pl.BF16, pl.Mem.Vec] = pl.tile.load(norm_w, [apply_d0], [128], [128], target_memory=pl.Mem.Vec)
            norm_w_chunk: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.reshape(norm_w_input, [1, 128]), target_type=pl.FP32, mode='round')
            x_scaled: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.row_expand_mul(apply_x_chunk, x_inv_rms_t)
            x_normed_chunk: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.col_expand_mul(x_scaled, norm_w_chunk)
            x_normed_bf16: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.cast(x_normed_chunk, target_type=pl.BF16, mode='rint')
            x_normed_valid: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 128])] = pl.tile.set_validshape(x_normed_bf16, valid_rows, 128)
            pl.tile.store(x_normed_valid, [tg, apply_d0], x_normed)
    @pl.function(type=pl.FunctionType.Inline)
    def compressor_ratio4(self, x: pl.Tensor[[KV_T_DYN, 4096], pl.BF16], kv: pl.Tensor[[KV_T_DYN, 512], pl.FP32], compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32], compress_state_block_table: pl.Tensor[[KV_B_DYN, 4], pl.INT32], wkv: pl.Tensor[[1024, 4096], pl.BF16], wgate: pl.Tensor[[1024, 4096], pl.BF16], ape: pl.Tensor[[4, 1024], pl.FP32], norm_w: pl.Tensor[[512], pl.BF16], cos: pl.Tensor[[KV_T_DYN, 64], pl.FP32], sin: pl.Tensor[[KV_T_DYN, 64], pl.FP32], cmp_kv_cache: pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16], position_ids: pl.Tensor[[KV_T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], late_dep: pl.Scalar[pl.TASK_ID]) -> tuple[pl.Tensor[[KV_T_DYN, 512], pl.FP32], pl.Scalar[pl.TASK_ID]]:
        b_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table, 0)
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        s_dim: pl.Scalar[pl.INDEX] = bs // b_dim
        t_matmul: pl.Scalar[pl.INDEX] = (bs + 16 - 1) // 16 * 16
        rms_blocks: pl.Scalar[pl.INDEX] = (bs + 16 - 1) // 16
        x_flat: pl.Tensor[[KV_T_DYN, 4096], pl.BF16] = x
        cmp4_kv_proj_pad: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.create([512, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp4_score_proj_pad: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.create([512, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        compress_state_block_num: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state, 0)
        cmp_block_num: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv_cache, 0)
        compress_state_flat: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.reshape(compress_state, [compress_state_block_num * 2, 2048])
        kv_flat: pl.Tensor[[KV_T_DYN, 512], pl.FP32] = kv
        cmp_kv_cache_flat: pl.Tensor[[cmp_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(cmp_kv_cache, [cmp_block_num * 32, 512])
        with pl.spmd(t_matmul * 1024 // 1024, name_hint="kv_score_proj_spmd", deps=[late_dep]) as _kv_score_tid:
            idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            global_row0: pl.Scalar[pl.INDEX] = idx // 16 * 16
            o0: pl.Scalar[pl.INDEX] = idx % 16 * 64
            kv_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            score_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for kb in pl.pipeline(8, stage=2):
                k0: pl.Scalar[pl.INDEX] = kb * 512
                x_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - global_row0)
                x_tile: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[x_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [16, 512], [global_row0, k0], [x_rows, 512])
                wkv_tile: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wkv, [64, 512], [o0, k0])
                wgate_tile: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(wgate, [64, 512], [o0, k0])
                if k0 == 0:
                    kv_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile, wkv_tile, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                    score_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile, wgate_tile, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                else:
                    kv_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(kv_acc, x_tile, wkv_tile, a_trans=False, b_trans=True)
                    score_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(score_acc, x_tile, wgate_tile, a_trans=False, b_trans=True)
            cmp4_kv_proj_pad: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.assemble(cmp4_kv_proj_pad, kv_acc, [global_row0, o0])
            cmp4_score_proj_pad: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.assemble(cmp4_score_proj_pad, score_acc, [global_row0, o0])
        pooled_kv: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.create([512, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.spmd(b_dim, name_hint="scatter_softmax_pool_spmd", deps=[_kv_score_tid]) as pool_tid:
            c_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            first_pos_b: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [c_idx * s_dim])
            for s_idx in pl.range(s_dim):
                token: pl.Scalar[pl.INDEX] = c_idx * s_dim + s_idx
                token_pos: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [token])
                pooled_kv: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(pooled_kv, pl.tensor.full([1, 512], dtype=pl.FP32, value=0.0), [token, 0])
                if (pl.cast(token_pos, pl.INDEX) + 1) % 4 == 0:
                    window_start: pl.Scalar[pl.INDEX] = pl.cast(token_pos, pl.INDEX) - 8 + 1
                    for h0 in pl.range(0, 512, 64):
                        last_ape_row: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos, pl.INDEX) % 4, pl.INDEX)
                        mi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad, [1, 64], [token, 512 + h0]), pl.tensor.slice(ape, [1, 64], [last_ape_row, 512 + h0]))
                        li: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi, mi))
                        oi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cmp4_kv_proj_pad, [1, 64], [token, 512 + h0])
                        for state_idx in pl.range(7):
                            logical_pos: pl.Scalar[pl.INDEX] = window_start + state_idx
                            value: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0)
                            score: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                            state_half: pl.Scalar[pl.INDEX] = 0
                            if state_idx >= 4:
                                state_half: pl.Scalar[pl.INDEX] = 512
                            if logical_pos >= 0 and logical_pos < pl.cast(first_pos_b, pl.INDEX):
                                ring_row: pl.Scalar[pl.INDEX] = logical_pos % 8
                                state_page_off: pl.Scalar[pl.INDEX] = ring_row // 2
                                state_blk_id_i32: pl.Scalar[pl.INT32] = pl.tensor.read(compress_state_block_table, [c_idx, state_page_off])
                                if pl.cast(state_blk_id_i32, pl.INDEX) >= 0:
                                    state_blk_id: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32, pl.INDEX)
                                    state_row: pl.Scalar[pl.INDEX] = state_blk_id * 2 + ring_row % 2
                                    value: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat, [1, 64], [state_row, state_half + h0])
                                    score: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat, [1, 64], [state_row, 1024 + state_half + h0])
                            if logical_pos >= pl.cast(first_pos_b, pl.INDEX):
                                if logical_pos <= pl.cast(token_pos, pl.INDEX):
                                    overlay_token: pl.Scalar[pl.INDEX] = c_idx * s_dim + logical_pos - pl.cast(first_pos_b, pl.INDEX)
                                    ape_row: pl.Scalar[pl.INDEX] = pl.cast(logical_pos % 4, pl.INDEX)
                                    value: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cmp4_kv_proj_pad, [1, 64], [overlay_token, state_half + h0])
                                    score: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad, [1, 64], [overlay_token, state_half + h0]), pl.tensor.slice(ape, [1, 64], [ape_row, state_half + h0]))
                            mi_next: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(mi, score)
                            alpha: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi, mi_next))
                            beta: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(score, mi_next))
                            li: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha, li), beta)
                            oi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(oi, alpha), pl.tensor.mul(value, beta))
                            mi: pl.Tensor[[1, 64], pl.FP32] = mi_next
                        pooled_kv: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(pooled_kv, pl.tensor.div(oi, li), [token, h0])
        for c_idx_v1 in pl.spmd(b_dim, name_hint="compress_state_commit_spmd", deps=[pool_tid]):
            for s_idx_1 in pl.range(s_dim):
                token: pl.Scalar[pl.INDEX] = c_idx_v1 * s_dim + s_idx_1
                state_row_i64: pl.Scalar[pl.INT64] = pl.tensor.read(state_slot_mapping, [token])
                if state_row_i64 >= 0:
                    state_row: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64, pl.INDEX)
                    token_pos: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [token])
                    ape_row: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos, pl.INDEX) % 4, pl.INDEX)
                    compress_state_flat: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.assemble(compress_state_flat, pl.tensor.slice(cmp4_kv_proj_pad, [1, 1024], [token, 0]), [state_row, 0])
                    compress_state_flat: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.assemble(compress_state_flat, pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad, [1, 1024], [token, 0]), pl.tensor.slice(ape, [1, 1024], [ape_row, 0])), [state_row, 1024])
        normed_kv: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.create([512, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        norm_w_2d: pl.Tensor[[1, 512], pl.BF16] = pl.tensor.reshape(norm_w, [1, 512])
        with pl.spmd(rms_blocks, name_hint="rmsnorm_rope_cache_write_spmd", deps=[pool_tid]) as cache_write_tid:
            rms_blk: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            b0: pl.Scalar[pl.INDEX] = rms_blk * 16
            rms_blk_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - b0)
            cos_b: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cos, [16, 64], [b0, 0], [rms_blk_rows, 64])
            sin_b: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(sin, [16, 64], [b0, 0], [rms_blk_rows, 64])
            partial_sq: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
            for k0_1 in pl.range(0, 512, 64):
                kv_rms_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv, [16, 64], [b0, k0_1])
                kv_rms_sq: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_rms_chunk, kv_rms_chunk)
                kv_rms_rowsum: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_rms_sq), [1, 16])
                partial_sq: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq, kv_rms_rowsum)
            variance: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq, 0.001953125), 9.9999999999999995e-07), [16, 1])
            inv_rms: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance))
            for k0_2 in pl.range(0, 448, 64):
                kv_norm_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv, [16, 64], [b0, k0_2])
                gamma: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d, [1, 64], [0, k0_2]), target_type=pl.FP32, mode='round')
                normed_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                normed_kv: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(normed_kv, normed_chunk, [b0, k0_2])
            kv_rope_norm: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv, [16, 64], [b0, 448])
            gamma_rope: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d, [1, 64], [0, 448]), target_type=pl.FP32, mode='round')
            rope_normed: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
            rope_ones: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
            rope_col: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(rope_ones, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
            rope_dup_f: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(rope_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
            rope_lane: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(rope_col, pl.tensor.muls(rope_dup_f, 2.0))
            rope_swap_idx: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(rope_col, 1.0), pl.tensor.muls(rope_lane, 2.0)), target_type=pl.INT32, mode='round')
            swapped: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(rope_normed, rope_swap_idx, dim=-1)
            rope_rot: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(rope_normed, cos_b), pl.tensor.mul(swapped, sin_b))
            normed_kv: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(normed_kv, rope_rot, [b0, 448])
            for inner in pl.range(rms_blk_rows):
                token: pl.Scalar[pl.INDEX] = b0 + inner
                cache_row_i64: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_slot_mapping, [token])
                if cache_row_i64 >= 0:
                    cache_row: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64, pl.INDEX)
                    kv_row_fp32: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.slice(normed_kv, [1, 512], [token, 0])
                    kv_flat: pl.Tensor[[KV_T_DYN, 512], pl.FP32] = pl.tensor.assemble(kv_flat, kv_row_fp32, [token, 0])
                    cmp_kv_cache_flat: pl.Tensor[[cmp_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(cmp_kv_cache_flat, pl.tensor.cast(kv_row_fp32, target_type=pl.BF16, mode='rint'), [cache_row, 0])
        return kv, cache_write_tid
    @pl.function(type=pl.FunctionType.Inline)
    def decode_cp_token_allgather_step(self, hidden_local: pl.Tensor[[T_DYN, 4096], pl.BF16], group_out: pl.Tensor[[KV_T_DYN, 4096], pl.BF16], gather_window: pld.DistributedTensor[[512, 4096], pl.BF16], gather_signal: pld.DistributedTensor[[2, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32]) -> tuple[pl.Tensor[[KV_T_DYN, 4096], pl.BF16], pld.DistributedTensor[[2, 1], pl.INT32]]:
        # Gather rank-major rows and retire the complete two-phase signal epoch.
        local_rows: pl.Scalar[pl.INDEX] = pl.tensor.dim(hidden_local, 0)
        local_t: pl.Scalar[pl.INT32] = pl.cast(local_rows, pl.INT32)
        target_row: pl.Scalar[pl.INT32] = tp_rank * local_t
        full_local: pl.Scalar[pl.INDEX] = pl.cast(local_t, pl.INDEX) // 8 * 8
        with pl.spmd(16, name_hint="cp_token_allgather_push_spmd", allow_early_resolve=True) as _push_tid:
            worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            for peer_tp in pl.range(2):
                for band_row in pl.range(worker * 8, full_local, 128):
                    pld.tensor.put(gather_window, pl.cast(group_base, pl.INDEX) + peer_tp, hidden_local, [pl.cast(target_row, pl.INDEX) + band_row, 0], [band_row, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                for tail_row in pl.range(full_local + worker, local_t, 16):
                    pld.tensor.put(gather_window, pl.cast(group_base, pl.INDEX) + peer_tp, hidden_local, [pl.cast(target_row, pl.INDEX) + tail_row, 0], [tail_row, 0], [1, 4096], atomic=pl.AtomicType.None_, chunk_rows=1, chunk_cols=4096)
            for peer_tp_1 in pl.range(2):
                if peer_tp_1 != pl.cast(tp_rank, pl.INDEX):
                    pld.system.notify(gather_signal, pl.cast(group_base, pl.INDEX) + peer_tp_1, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_payload_wait", deps=[_push_tid]) as _payload_wait_tid:
            for source_tp in pl.range(2):
                if source_tp != pl.cast(tp_rank, pl.INDEX):
                    pld.system.wait(gather_signal, [source_tp, 0], pl.cast(16, pl.INT32), cmp=1)
        group_rows: pl.Scalar[pl.INDEX] = 2 * local_rows
        full_rows: pl.Scalar[pl.INDEX] = group_rows // 16 * 16
        with pl.spmd(16, name_hint="cp_token_allgather_readback_spmd", deps=[_push_tid, _payload_wait_tid]) as _readback_tid:
            worker_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            for tile_row in pl.range(worker_v1 * 16, full_rows, 256):
                window_tile: pld.DistributedTensor[[16, 4096], pl.BF16] = pl.tensor.slice(gather_window, [16, 4096], [tile_row, 0])
                group_out: pl.Tensor[[KV_T_DYN, 4096], pl.BF16] = pl.tensor.assemble(group_out, window_tile, [tile_row, 0])
            for tail_row_1 in pl.range(full_rows + worker_v1, group_rows, 16):
                window_row: pld.DistributedTensor[[1, 4096], pl.BF16] = pl.tensor.slice(gather_window, [1, 4096], [tail_row_1, 0])
                group_out: pl.Tensor[[KV_T_DYN, 4096], pl.BF16] = pl.tensor.assemble(group_out, window_row, [tail_row_1, 0])
            for peer_tp_2 in pl.range(2):
                if peer_tp_2 != pl.cast(tp_rank, pl.INDEX):
                    pld.system.notify(gather_signal, pl.cast(group_base, pl.INDEX) + peer_tp_2, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_readback_wait", deps=[_readback_tid]) as _readback_wait_tid:
            for source_tp_1 in pl.range(2):
                if source_tp_1 != pl.cast(tp_rank, pl.INDEX):
                    pld.system.wait(gather_signal, [source_tp_1, 0], pl.cast(32, pl.INT32), cmp=1)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_retire", deps=[_readback_tid, _readback_wait_tid]):
            completion_anchor: pl.Scalar[pl.BF16] = pl.tensor.read(group_out, [0, 0])
            reset_value: pl.Scalar[pl.INT32] = pl.cast(-32, pl.INT32)
            self_rank: pl.Scalar[pl.INT32] = group_base + tp_rank
            for source_tp_2 in pl.range(2):
                if source_tp_2 != pl.cast(tp_rank, pl.INDEX):
                    pld.system.notify(gather_signal, self_rank, [source_tp_2, 0], reset_value, op=0)
            pl.tensor.write(group_out, [0, 0], completion_anchor)
        return group_out, gather_signal
    @pl.function(type=pl.FunctionType.Inline)
    def hc_post(self, x: pl.Tensor[[T_DYN, 4096], pl.BF16], residual: pl.Tensor[[T_DYN, 4, 4096], pl.FP32], post: pl.Tensor[[T_DYN, 4], pl.FP32], comb: pl.Tensor[[T_DYN, 16], pl.FP32], y: pl.Tensor[[T_DYN, 4, 4096], pl.FP32]) -> pl.Tensor[[T_DYN, 4, 4096], pl.FP32]:
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        residual_flat: pl.Tensor[[t_dim, 16384], pl.FP32] = pl.tensor.reshape(residual, [t_dim, 16384])
        y_flat: pl.Tensor[[t_dim, 16384], pl.FP32] = pl.tensor.reshape(y, [t_dim, 16384])
        token_tiles: pl.Scalar[pl.INDEX] = (t_dim + 4 - 1) // 4
        for token_block in pl.spmd(token_tiles, name_hint="hc_post_spmd"):
            t0: pl.Scalar[pl.INDEX] = token_block * 4
            for t in pl.pipeline(t0, t0 + 4, stage=2):
                if t < t_dim:
                    x_row: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.cast(pl.tensor.slice(x, [1, 4096], [t, 0]), target_type=pl.FP32, mode='round')
                    for out_h in pl.unroll(4):
                        post_w: pl.Scalar[pl.FP32] = pl.tensor.read(post, [t, out_h])
                        y_row: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(x_row, post_w)
                        for in_h in pl.pipeline(4, stage=4):
                            comb_w: pl.Scalar[pl.FP32] = pl.tensor.read(comb, [t, in_h * 4 + out_h])
                            res_d: pl.Scalar[pl.INDEX] = in_h * 4096
                            res_row: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.slice(residual_flat, [1, 4096], [t, res_d])
                            weighted: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(res_row, comb_w)
                            y_row: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(y_row, weighted)
                        y_flat: pl.Tensor[[t_dim, 16384], pl.FP32] = pl.tensor.assemble(y_flat, y_row, [t, out_h * 4096])
        return y
    @pl.function(type=pl.FunctionType.Inline)
    def hc_pre(self, x: pl.Tensor[[T_DYN, 4, 4096], pl.FP32], hc_fn: pl.Tensor[[24, 16384], pl.FP32], hc_scale: pl.Tensor[[3], pl.FP32], hc_base: pl.Tensor[[24], pl.FP32], x_mixed: pl.Tensor[[T_DYN, 4096], pl.BF16], post: pl.Tensor[[T_DYN, 4], pl.FP32], comb: pl.Tensor[[T_DYN, 16], pl.FP32]) -> pl.Tensor[[T_DYN, 4096], pl.BF16]:
        # One pl.spmd task per work-type, ordered by their GM read/write dependencies.
        # 
        #         rms -> linear -> linear_reduce -> split_pre_post / comb_sinkhorn / mix_x. Cross-scope
        #         buffers are sized to t_linear, the token count padded up to whole 16-row cube tiles.
        #         
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        token_tiles: pl.Scalar[pl.INDEX] = (t_dim + 8 - 1) // 8
        t_linear: pl.Scalar[pl.INDEX] = (t_dim + 16 - 1) // 16 * 16
        x_flat: pl.Tensor[[t_dim, 16384], pl.FP32] = pl.tensor.reshape(x, [t_dim, 16384])
        scale0: pl.Scalar[pl.FP32] = pl.tensor.read(hc_scale, [0])
        scale1: pl.Scalar[pl.FP32] = pl.tensor.read(hc_scale, [1])
        scale2: pl.Scalar[pl.FP32] = pl.tensor.read(hc_scale, [2])
        hc_base_2d: pl.Tensor[[1, 24], pl.FP32] = pl.tensor.reshape(hc_base, [1, 24])
        inv_rms: pl.Tensor[[t_linear, 1], pl.FP32] = pl.tensor.create([t_linear, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for t in pl.spmd(token_tiles, name_hint="hc_pre_rms_spmd", allow_early_resolve=True):
            t0: pl.Scalar[pl.INDEX] = t * 8
            valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - t0)
            sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(32, stage=4):
                k0: pl.Scalar[pl.INDEX] = kb * 512
                if valid_rows == 8:
                    x_chunk_full: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.slice(x_flat, [8, 512], [t0, k0])
                    x_sq_full: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(x_chunk_full, x_chunk_full)
                    x_sq_row_full: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(x_sq_full), [1, 8])
                    sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(sq_sum, x_sq_row_full)
                else:
                    x_chunk_tail: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [8, 512], [t0, k0], [valid_rows, 512])
                    x_sq_tail: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.mul(x_chunk_tail, x_chunk_tail)
                    x_sq_row_tail: pl.Tensor[[1, 8], pl.FP32, pl.TensorView(valid_shape=[1, valid_rows], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.reshape(pl.tensor.row_sum(x_sq_tail), [1, 8])
                    sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(sq_sum, x_sq_row_tail)
            sq_mean: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.adds(pl.tensor.muls(sq_sum, 6.103515625e-05), 9.9999999999999995e-07)
            inv: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.rsqrt(sq_mean, high_precision=True), [8, 1])
            inv_rms: pl.Tensor[[t_linear, 1], pl.FP32] = pl.tensor.assemble(inv_rms, inv, [t0, 0])
        mixes_partials: pl.Tensor[[pl.const(4, pl.INDEX) * t_linear, 32], pl.FP32] = pl.tensor.create([4 * t_linear, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for task in pl.spmd(t_linear // 16 * 4, name_hint="hc_pre_linear_spmd", allow_early_resolve=True):
            t0: pl.Scalar[pl.INDEX] = task // 4 * 16
            linear_split: pl.Scalar[pl.INDEX] = task % 4
            k_base: pl.Scalar[pl.INDEX] = linear_split * 4096
            t_rows: pl.Scalar[pl.INDEX] = pl.min(16, t_dim - t0)
            acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for kb_1 in pl.pipeline(16, stage=2):
                k0: pl.Scalar[pl.INDEX] = k_base + kb_1 * 256
                x_linear_chunk: pl.Tensor[[16, 256], pl.FP32, pl.TensorView(valid_shape=[t_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [16, 256], [t0, k0], [t_rows, 256])
                w_chunk: pl.Tensor[[32, 256], pl.FP32, pl.TensorView(valid_shape=[24, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(hc_fn, [32, 256], [0, k0], [24, 256])
                if kb_1 == 0:
                    acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_linear_chunk, w_chunk, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                else:
                    acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(acc, x_linear_chunk, w_chunk, a_trans=False, b_trans=True)
            partial_row0: pl.Scalar[pl.INDEX] = linear_split * t_linear + t0
            mixes_partials: pl.Tensor[[pl.const(4, pl.INDEX) * t_linear, 32], pl.FP32] = pl.tensor.assemble(mixes_partials, acc, [partial_row0, 0])
        mixes_raw: pl.Tensor[[t_linear, 32], pl.FP32] = pl.tensor.create([t_linear, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for linear_block in pl.spmd(t_linear // 16, name_hint="hc_pre_linear_reduce_spmd", allow_early_resolve=True):
            linear_t0: pl.Scalar[pl.INDEX] = linear_block * 16
            mixes_total: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(mixes_partials, [16, 32], [linear_t0, 0])
            for linear_split_1 in pl.range(1, 4):
                partial_t0: pl.Scalar[pl.INDEX] = linear_split_1 * t_linear + linear_t0
                partial_tile: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(mixes_partials, [16, 32], [partial_t0, 0])
                mixes_total: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.add(mixes_total, partial_tile)
            mixes_raw: pl.Tensor[[t_linear, 32], pl.FP32] = pl.tensor.assemble(mixes_raw, mixes_total, [linear_t0, 0])
        pre_val_store: pl.Tensor[[t_linear, 8], pl.FP32] = pl.tensor.create([t_linear, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        post_tail_store: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for ob in pl.spmd(token_tiles, name_hint="split_pre_post_spmd", allow_early_resolve=True):
            t0: pl.Scalar[pl.INDEX] = ob * 8
            valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - t0)
            inv_col: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(inv_rms, [8, 1], [t0, 0])
            pre_base: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(hc_base, [8], [0]), [1, 8])
            pre_scaled: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(pl.tensor.row_expand_mul(pl.tensor.slice(mixes_raw, [8, 8], [t0, 0]), inv_col), scale0)
            pre_logits: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(pre_scaled, pl.tensor.col_expand(pre_scaled, pre_base))
            pre_sig: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(pre_logits)), 1.0))
            pre_val: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.adds(pre_sig, 9.9999999999999995e-07)
            pre_val_store: pl.Tensor[[t_linear, 8], pl.FP32] = pl.tensor.assemble(pre_val_store, pre_val, [t0, 0])
            post_base: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(hc_base, [8], [4]), [1, 8])
            post_scaled: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(pl.tensor.row_expand_mul(pl.tensor.slice(mixes_raw, [8, 8], [t0, 4]), inv_col), scale1)
            post_logits: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(post_scaled, pl.tensor.col_expand(post_scaled, post_base))
            post_sig: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(post_logits)), 1.0))
            post_pad: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(post_sig, 2.0)
            if valid_rows == 8:
                post: pl.Tensor[[T_DYN, 4], pl.FP32] = pl.tensor.assemble(post, pl.tensor.slice(post_pad, [8, 8], [0, 0], [8, 4]), [t0, 0])
            else:
                post_tail_store: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.assemble(post_tail_store, post_pad, [0, 0])
                post_tile: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(post_tail_store, [0, 0], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
                pl.tile.store(post_tile, [t0, 0], post)
        comb_tail_store: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.create([8, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for ob_1 in pl.spmd(token_tiles, name_hint="comb_sinkhorn_spmd", allow_early_resolve=True):
            t0: pl.Scalar[pl.INDEX] = ob_1 * 8
            valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - t0)
            inv_col_t: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 1])] = pl.tile.load(inv_rms, [t0, 0], [8, 1], [valid_rows, 1], target_memory=pl.Mem.Vec)
            comb_off: pl.Scalar[pl.INDEX] = 8
            mix_g0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(mixes_raw, [t0, comb_off + 0], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
            mix_g1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(mixes_raw, [t0, comb_off + 4], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
            mix_g2: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(mixes_raw, [t0, comb_off + 8], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
            mix_g3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(mixes_raw, [t0, comb_off + 12], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
            cb0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d, [0, comb_off + 0], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
            cb1: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d, [0, comb_off + 4], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
            cb2: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d, [0, comb_off + 8], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
            cb3: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d, [0, comb_off + 12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
            row0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g0, inv_col_t), scale2), pl.tile.col_expand(mix_g0, cb0))
            row1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g1, inv_col_t), scale2), pl.tile.col_expand(mix_g1, cb1))
            row2: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g2, inv_col_t), scale2), pl.tile.col_expand(mix_g2, cb2))
            row3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g3, inv_col_t), scale2), pl.tile.col_expand(mix_g3, cb3))
            row0_p: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row0, pad_value=pl.PadValue.min)
            row1_p: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row1, pad_value=pl.PadValue.min)
            row2_p: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row2, pad_value=pl.PadValue.min)
            row3_p: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row3, pad_value=pl.PadValue.min)
            row_max_tmp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            row_sum_tmp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            row0_max: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row0_p, row_max_tmp)
            row1_max: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row1_p, row_max_tmp)
            row2_max: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row2_p, row_max_tmp)
            row3_max: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row3_p, row_max_tmp)
            row0_exp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row0_p, row0_max))
            row1_exp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row1_p, row1_max))
            row2_exp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row2_p, row2_max))
            row3_exp: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row3_p, row3_max))
            row0_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row0_exp, row_sum_tmp)
            row1_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row1_exp, row_sum_tmp)
            row2_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row2_exp, row_sum_tmp)
            row3_sum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row3_exp, row_sum_tmp)
            row0_soft: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row0_exp, row0_sum), 9.9999999999999995e-07)
            row1_soft: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row1_exp, row1_sum), 9.9999999999999995e-07)
            row2_soft: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row2_exp, row2_sum), 9.9999999999999995e-07)
            row3_soft: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row3_exp, row3_sum), 9.9999999999999995e-07)
            row0_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row0_soft, 8, 4)
            row1_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row1_soft, 8, 4)
            row2_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row2_soft, 8, 4)
            row3_valid: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row3_soft, 8, 4)
            row0_eff: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row0_valid, pad_value=pl.PadValue.zero)
            row1_eff: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row1_valid, pad_value=pl.PadValue.zero)
            row2_eff: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row2_valid, pad_value=pl.PadValue.zero)
            row3_eff: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row3_valid, pad_value=pl.PadValue.zero)
            row_sum_tmp_iter: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            col_sum: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(pl.tile.add(row0_eff, row1_eff), pl.tile.add(row2_eff, row3_eff))
            col_sum_v1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum, 9.9999999999999995e-07)
            row0_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_eff, col_sum_v1)
            row1_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_eff, col_sum_v1)
            row2_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_eff, col_sum_v1)
            row3_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_eff, col_sum_v1)
            for _sk_it in pl.pipeline(19, stage=2):
                row0_rowsum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row0_cur, row_sum_tmp_iter), 9.9999999999999995e-07)
                row1_rowsum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row1_cur, row_sum_tmp_iter), 9.9999999999999995e-07)
                row2_rowsum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row2_cur, row_sum_tmp_iter), 9.9999999999999995e-07)
                row3_rowsum: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row3_cur, row_sum_tmp_iter), 9.9999999999999995e-07)
                row0_norm: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row0_cur, row0_rowsum)
                row1_norm: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row1_cur, row1_rowsum)
                row2_norm: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row2_cur, row2_rowsum)
                row3_norm: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row3_cur, row3_rowsum)
                col_sum_v1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(pl.tile.add(row0_norm, row1_norm), pl.tile.add(row2_norm, row3_norm))
                col_sum_v1: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_v1, 9.9999999999999995e-07)
                row0_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_norm, col_sum_v1)
                row1_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_norm, col_sum_v1)
                row2_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_norm, col_sum_v1)
                row3_cur: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_norm, col_sum_v1)
            if valid_rows == 8:
                row0_out: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row0_cur, 8, 4)
                row1_out: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row1_cur, 8, 4)
                row2_out: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row2_cur, 8, 4)
                row3_out: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row3_cur, 8, 4)
                pl.tile.store(row0_out, [t0, 0], comb)
                pl.tile.store(row1_out, [t0, 4], comb)
                pl.tile.store(row2_out, [t0, 8], comb)
                pl.tile.store(row3_out, [t0, 12], comb)
            else:
                pl.tile.store(row0_cur, [0, 0], comb_tail_store)
                pl.tile.store(row1_cur, [0, 8], comb_tail_store)
                pl.tile.store(row2_cur, [0, 16], comb_tail_store)
                pl.tile.store(row3_cur, [0, 24], comb_tail_store)
                row0_tail: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store, [0, 0], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
                row1_tail: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store, [0, 8], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
                row2_tail: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store, [0, 16], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
                row3_tail: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store, [0, 24], [8, 8], [valid_rows, 4], target_memory=pl.Mem.Vec)
                pl.tile.store(row0_tail, [t0, 0], comb)
                pl.tile.store(row1_tail, [t0, 4], comb)
                pl.tile.store(row2_tail, [t0, 8], comb)
                pl.tile.store(row3_tail, [t0, 12], comb)
        x_mixed_tail_store: pl.Tensor[[8, 4096], pl.BF16] = pl.tensor.create([8, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        for blk in pl.spmd(token_tiles * 1, name_hint="mix_x_spmd", allow_early_resolve=True):
            t0: pl.Scalar[pl.INDEX] = blk // 1 * 8
            d_base: pl.Scalar[pl.INDEX] = blk % 1 * 4096
            valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - t0)
            pre_tile_t: pl.Tensor[[8, 8], pl.FP32, pl.TensorView(stride=[1, 8], layout=pl.TensorLayout.DN)] = pl.tensor.transpose(pl.tensor.slice(pre_val_store, [8, 8], [t0, 0]), 0, 1)
            pre0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t, [1, 8], [0, 0]), [8, 1])
            pre1: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t, [1, 8], [1, 0]), [8, 1])
            pre2: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t, [1, 8], [2, 0]), [8, 1])
            pre3: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t, [1, 8], [3, 0]), [8, 1])
            for db in pl.pipeline(16, stage=2):
                d0: pl.Scalar[pl.INDEX] = d_base + db * 256
                x0: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [8, 256], [t0, 0 + d0], [valid_rows, 256])
                x1: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [8, 256], [t0, 4096 + d0], [valid_rows, 256])
                x2: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [8, 256], [t0, 8192 + d0], [valid_rows, 256])
                x3: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [8, 256], [t0, 12288 + d0], [valid_rows, 256])
                y0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x0, pre0)
                y1: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x1, pre1)
                y2: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x2, pre2)
                y3: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x3, pre3)
                y_tile: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.add(pl.tensor.add(y0, y1), pl.tensor.add(y2, y3))
                y_bf16: pl.Tensor[[8, 256], pl.BF16] = pl.tensor.cast(y_tile, target_type=pl.BF16, mode='rint')
                if valid_rows == 8:
                    x_mixed: pl.Tensor[[T_DYN, 4096], pl.BF16] = pl.tensor.assemble(x_mixed, y_bf16, [t0, d0])
                else:
                    x_mixed_tail_store: pl.Tensor[[8, 4096], pl.BF16] = pl.tensor.assemble(x_mixed_tail_store, y_bf16, [0, d0])
                    y_out: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(x_mixed_tail_store, [0, d0], [8, 256], [valid_rows, 256], target_memory=pl.Mem.Vec)
                    pl.tile.store(y_out, [t0, d0], x_mixed)
        return x_mixed
    @pl.function(type=pl.FunctionType.Inline)
    def indexer_topk_leaf(self, score_arena: pl.Tensor[[T_DYN, 262144], pl.FP32], pair_arena: pl.Tensor[[4192, 1024], pl.FP32], query: pl.Scalar[pl.INDEX], logical_begin: pl.Scalar[pl.INDEX], valid_count: pl.Scalar[pl.INDEX], output_slot: pl.Scalar[pl.INDEX]):
        # Sort one scored 8K leaf and store its exact Top-512 pair row.
        logical_begin_i32: pl.Scalar[pl.INT32] = pl.cast(logical_begin, pl.INT32)
        leaf_indices: pl.Tile[[1, 8192], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False), logical_begin_i32)
        leaf_scores_raw: pl.Tile[[1, 8192], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [query, logical_begin], [1, 8192], [1, valid_count])
        leaf_scores: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
        leaf_scores_v1: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores, pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38))
        pairs: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1, pl.tile.reinterpret_view(leaf_indices, dtype=pl.UINT32))
        pairs_v1: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs, pl.const(64, pl.INT32))
        pairs_v2: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1, pl.const(256, pl.INT32))
        pairs_v3: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2, pl.const(1024, pl.INT32))
        pairs_v4: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3, pl.const(4096, pl.INT32))
        pl.tile.store(pl.tile.slice(pairs_v4, [1, 1024], [0, 0]), [output_slot, 0], pair_arena)
    @pl.function(type=pl.FunctionType.Inline)
    def merge2_top512_pairs(self, pair_arena: pl.Tensor[[4192, 1024], pl.FP32], left_slot: pl.Scalar[pl.INDEX], right_slot: pl.Scalar[pl.INDEX], output_slot: pl.Scalar[pl.INDEX]):
        # Merge two arena rows and store their exact Top-512 pair row.
        left: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot, 0], [1, 1024], [1, 1024])
        right: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [right_slot, 0], [1, 1024], [1, 1024])
        merge_tmp: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
        merged_all: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left, right, merge_tmp, exhausted=False)
        merged: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all, [1, 1024], [0, 0])
        pl.tile.store(merged, [output_slot, 0], pair_arena)
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_topk_group_wave(position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], score_arena: pl.Tensor[[T_DYN, 262144], pl.FP32], pair_arena: pl.Tensor[[4192, 1024], pl.FP32]):
        # Reduce globally striped two-leaf subtrees into compact roots.
        worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids, 0)
        global_group_base: pl.Scalar[pl.INDEX] = 0
        for query in pl.range(query_count):
            batch_idx: pl.Scalar[pl.INDEX] = query // 8
            position: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [query])
            cache_len: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens, [batch_idx]), pl.INDEX) // 4
            visible_count: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len, (pl.cast(position, pl.INDEX) + 1) // 4), 262144), 0)
            leaf_count: pl.Scalar[pl.INDEX] = (visible_count + 8192 - 1) // 8192
            group_count: pl.Scalar[pl.INDEX] = (leaf_count + 2 - 1) // 2
            base_mod: pl.Scalar[pl.INDEX] = global_group_base % 48
            first_group: pl.Scalar[pl.INDEX] = (worker + base_mod) % 48
            for group in pl.range(first_group, group_count, 48):
                leaf_begin: pl.Scalar[pl.INDEX] = group * 2
                group_leaf_count: pl.Scalar[pl.INDEX] = pl.min(2, leaf_count - leaf_begin)
                group_root_slot: pl.Scalar[pl.INDEX] = query * 16 + group
                if group_leaf_count == 1:
                    logical_begin: pl.Scalar[pl.INDEX] = leaf_begin * 8192
                    valid_count: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count - logical_begin)
                    self.indexer_topk_leaf(score_arena, pair_arena, query, logical_begin, valid_count, group_root_slot)
                else:
                    scratch_base: pl.Scalar[pl.INDEX] = 4096 + worker * 2
                    for group_leaf in pl.unroll(2):
                        leaf: pl.Scalar[pl.INDEX] = leaf_begin + group_leaf
                        logical_begin: pl.Scalar[pl.INDEX] = leaf * 8192
                        valid_count: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count - logical_begin)
                        self.indexer_topk_leaf(score_arena, pair_arena, query, logical_begin, valid_count, scratch_base + group_leaf)
                    self.merge2_top512_pairs(pair_arena, scratch_base, scratch_base + 1, group_root_slot)
            global_group_base: pl.Scalar[pl.INDEX] = global_group_base + group_count
    @pl.function(type=pl.FunctionType.Inline)
    def merge_topk_level_pairs(self, pair_arena: pl.Tensor[[4192, 1024], pl.FP32], arena_base: pl.Scalar[pl.INDEX], input_count: pl.Scalar[pl.INDEX], input_base: pl.Scalar[pl.INDEX], output_base: pl.Scalar[pl.INDEX]):
        # Reduce one exact-Top-K forest level, forwarding an odd final node.
        output_count: pl.Scalar[pl.INDEX] = (input_count + 1) // 2
        for output in pl.range(output_count):
            left_slot: pl.Scalar[pl.INDEX] = arena_base + input_base + 2 * output
            right_slot: pl.Scalar[pl.INDEX] = left_slot + 1
            output_slot: pl.Scalar[pl.INDEX] = arena_base + output_base + output
            if right_slot < arena_base + input_base + input_count:
                self.merge2_top512_pairs(pair_arena, left_slot, right_slot, output_slot)
            else:
                forwarded: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot, 0], [1, 1024], [1, 1024])
                pl.tile.store(forwarded, [output_slot, 0], pair_arena)
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_topk_query_merge(position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], pair_arena: pl.Tensor[[4192, 1024], pl.FP32], topk_scores: pl.Tensor[[T_DYN, 512], pl.FP32], topk_indices: pl.Tensor[[T_DYN, 512], pl.INT32]):
        # Merge compact group roots and materialize each query's Top-512.
        query: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        batch_idx: pl.Scalar[pl.INDEX] = query // 8
        position: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [query])
        cache_len: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens, [batch_idx]), pl.INDEX) // 4
        visible_count: pl.Scalar[pl.INDEX] = pl.min(pl.min(cache_len, (pl.cast(position, pl.INDEX) + 1) // 4), 262144)
        pl.tile.store(pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38), [query, 0], topk_scores)
        pl.tile.store(pl.tile.full([1, 512], dtype=pl.INT32, value=-1), [query, 0], topk_indices)
        if visible_count > 0:
            leaf_count: pl.Scalar[pl.INDEX] = (visible_count + 8192 - 1) // 8192
            group_count: pl.Scalar[pl.INDEX] = (leaf_count + 2 - 1) // 2
            arena_base: pl.Scalar[pl.INDEX] = query * 16
            if group_count > 1:
                level1_count: pl.Scalar[pl.INDEX] = (group_count + 1) // 2
                self.merge_topk_level_pairs(pair_arena, arena_base, group_count, 0, 0)
                if level1_count > 1:
                    level2_count: pl.Scalar[pl.INDEX] = (level1_count + 1) // 2
                    self.merge_topk_level_pairs(pair_arena, arena_base, level1_count, 0, 0)
                    if level2_count > 1:
                        level3_count: pl.Scalar[pl.INDEX] = (level2_count + 1) // 2
                        self.merge_topk_level_pairs(pair_arena, arena_base, level2_count, 0, 0)
                        if level3_count > 1:
                            self.merge_topk_level_pairs(pair_arena, arena_base, level3_count, 0, 0)
            root_slot: pl.Scalar[pl.INDEX] = arena_base
            root_pairs: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [root_slot, 0], [1, 1024], [1, 1024])
            pl.tile.store(pl.tile.gather_mask(root_pairs, mask_pattern=1, output_dtype=pl.FP32), [query, 0], topk_scores)
            root_indices: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.gather_mask(root_pairs, mask_pattern=2, output_dtype=pl.INT32)
            output_indices: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
            valid_topk: pl.Scalar[pl.INDEX] = pl.min(visible_count, 512)
            for lane in pl.range(valid_topk):
                pl.tile.write(output_indices, [0, lane], pl.tile.read(root_indices, [0, lane]))
            pl.tile.store(output_indices, [query, 0], topk_indices)
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def indexer_score_topk_forest(self, qr_hadamard_i8: pl.Tensor[[16384, 128], pl.INT8], qr_hadamard_scale_dq: pl.Tensor[[16384, 1], pl.FP32], weights: pl.Tensor[[256, 64], pl.FP32], idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32], idx_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], topk_scores: pl.Tensor[[T_DYN, 512], pl.FP32], topk_idxs: pl.Tensor[[T_DYN, 512], pl.INT32], qh_quant_tid: pl.Scalar[pl.TASK_ID], weights_tid: pl.Scalar[pl.TASK_ID], cache_write_tid: pl.Scalar[pl.TASK_ID]) -> tuple[pl.Tensor[[T_DYN, 512], pl.FP32], pl.Tensor[[T_DYN, 512], pl.INT32]]:
        # Run exact Top-K with the score and pair arenas on separate rings.
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids, 0)
        b_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table, 0)
        idx_block_num: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_kv_cache, 0)
        idx_table_len: pl.Scalar[pl.INDEX] = b_dim * 8192
        kv_cache_i8_flat: pl.Tensor[[idx_block_num * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.reshape(idx_kv_cache, [idx_block_num * 32, 128])
        kv_scale_flat: pl.Tensor[[idx_block_num * pl.const(32, pl.INDEX), 1], pl.FP32] = pl.tensor.reshape(idx_kv_scale, [idx_block_num * 32, 1])
        idx_block_table_flat: pl.Tensor[[idx_table_len], pl.INT32] = pl.tensor.reshape(idx_block_table, [idx_table_len])
        pair_arena: pl.Tensor[[4192, 1024], pl.FP32] = pl.tensor.create([4192, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.scope():
            score_arena: pl.Tensor[[bs, 262144], pl.FP32] = pl.tensor.create([bs, 262144], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(24, name_hint="indexer_score_leaf_wave_spmd", optimizations=[pl.cross_core_slot(slot_num=2)], deps=[qh_quant_tid, weights_tid, cache_write_tid]) as score_tid:
                worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                query_count: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids, 0)
                global_leaf_base: pl.Scalar[pl.INDEX] = 0
                for query in pl.range(query_count):
                    batch_idx: pl.Scalar[pl.INDEX] = query // 8
                    position: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [query])
                    cache_len: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens, [batch_idx]), pl.INDEX) // 4
                    visible_count: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len, (pl.cast(position, pl.INDEX) + 1) // 4), 262144), 0)
                    leaf_count: pl.Scalar[pl.INDEX] = (visible_count + 8192 - 1) // 8192
                    base_mod: pl.Scalar[pl.INDEX] = global_leaf_base % 24
                    first_leaf: pl.Scalar[pl.INDEX] = (worker + base_mod) % 24
                    for leaf in pl.range(first_leaf, leaf_count, 24):
                        logical_begin: pl.Scalar[pl.INDEX] = leaf * 8192
                        valid_count: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count - logical_begin)
                        query_head_begin: pl.Scalar[pl.INDEX] = query * 64
                        query_vector: pl.Tensor[[64, 128], pl.INT8] = pl.tensor.slice(qr_hadamard_i8, [64, 128], [query_head_begin, 0])
                        query_scale: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(qr_hadamard_scale_dq, [64, 1], [query_head_begin, 0]), [1, 64])
                        query_weight: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(weights, [1, 64], [query, 0])
                        page_count: pl.Scalar[pl.INDEX] = (valid_count + 32 - 1) // 32
                        for page in pl.pipeline(page_count, stage=2):
                            page_begin: pl.Scalar[pl.INDEX] = page * 32
                            logical_row: pl.Scalar[pl.INDEX] = logical_begin + page_begin
                            logical_page: pl.Scalar[pl.INDEX] = logical_row // 32
                            physical_block: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_block_table_flat, [batch_idx * 8192 + logical_page]), pl.INDEX)
                            physical_row: pl.Scalar[pl.INDEX] = physical_block * 32
                            kv_i8: pl.Tensor[[32, 128], pl.INT8] = pl.tensor.slice(kv_cache_i8_flat, [32, 128], [physical_row, 0])
                            score_i32: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.matmul(kv_i8, query_vector, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.INT32)
                            score_fp32: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.cast(score_i32, target_type=pl.FP32, mode='none')
                            score_fp32_v1: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(score_fp32, query_scale)
                            score_fp32_v2: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.maximum(score_fp32_v1, 0.0)
                            score_fp32_v3: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(score_fp32_v2, query_weight)
                            kv_scale: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.slice(kv_scale_flat, [32, 1], [physical_row, 0])
                            score: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.mul(pl.tensor.row_sum(score_fp32_v3), kv_scale)
                            score_row: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.reshape(score, [1, 32])
                            valid_rows: pl.Scalar[pl.INDEX] = pl.min(32, valid_count - page_begin)
                            score_valid: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.fillpad(pl.tensor.set_validshape(score_row, 1, valid_rows), pad_value=pl.PadValue.min)
                            score_arena: pl.Tensor[[bs, 262144], pl.FP32] = pl.tensor.assemble(score_arena, score_valid, [query, logical_row])
                    global_leaf_base: pl.Scalar[pl.INDEX] = global_leaf_base + leaf_count
            with pl.spmd(48, name_hint="indexer_topk_group_wave", deps=[score_tid]) as topk_tid:
                self.indexer_topk_group_wave(position_ids, kv_seq_lens, score_arena, pair_arena)
            with pl.spmd(bs, name_hint="indexer_topk_query_merge", deps=[topk_tid]) as _score_tid:
                self.indexer_topk_query_merge(position_ids, kv_seq_lens, pair_arena, topk_scores, topk_idxs)
        return topk_scores, topk_idxs
    @pl.function(type=pl.FunctionType.Inline)
    def indexer(self, x: pl.Tensor[[T_DYN, 4096], pl.BF16], qr: pl.Tensor[[T_DYN, 1024], pl.INT8], qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32], wq_b: pl.Tensor[[1024, 8192], pl.INT8], wq_b_scale: pl.Tensor[[8192], pl.FP32], weights_proj: pl.Tensor[[4096, 64], pl.BF16], cos: pl.Tensor[[T_DYN, 64], pl.FP32], sin: pl.Tensor[[T_DYN, 64], pl.FP32], hadamard: pl.Tensor[[128, 128], pl.BF16], idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32], idx_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], topk_scores: pl.Tensor[[T_DYN, 512], pl.FP32], topk_idxs: pl.Tensor[[T_DYN, 512], pl.INT32], position_ids: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], late_dep: pl.Scalar[pl.TASK_ID], cache_write_dep: pl.Scalar[pl.TASK_ID]) -> tuple[pl.Tensor[[T_DYN, 512], pl.FP32], pl.Tensor[[T_DYN, 512], pl.INT32]]:
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        bs_heads: pl.Scalar[pl.INDEX] = bs * 64
        row_blocks: pl.Scalar[pl.INDEX] = (bs + 16 - 1) // 16
        qr_acc_pad: pl.Tensor[[256, 8192], pl.INT32] = pl.tensor.create([256, 8192], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        for qr_unit in pl.spmd(8 * row_blocks, name_hint="idx_qr_proj_matmul_spmd", allow_early_resolve=True):
            qr_rb: pl.Scalar[pl.INDEX] = qr_unit // 8
            ot: pl.Scalar[pl.INDEX] = qr_unit - qr_rb * 8
            qr_r0: pl.Scalar[pl.INDEX] = qr_rb * 16
            qr_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - qr_r0)
            o_base: pl.Scalar[pl.INDEX] = ot * 1024
            for ns in pl.range(0, 1024, 512):
                qr_acc: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.create([16, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                for kb in pl.pipeline(4, stage=2):
                    q0: pl.Scalar[pl.INDEX] = kb * 256
                    qr_tile: pl.Tensor[[16, 256], pl.INT8, pl.TensorView(valid_shape=[qr_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr, [16, 256], [qr_r0, q0], [qr_rows, 256])
                    wq_tile: pl.Tensor[[256, 512], pl.INT8] = pl.tensor.slice(wq_b, [256, 512], [q0, o_base + ns])
                    if q0 == 0:
                        qr_acc: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul(qr_tile, wq_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                    else:
                        qr_acc: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul_acc(qr_acc, qr_tile, wq_tile, a_trans=False, b_trans=False)
                qr_acc_pad: pl.Tensor[[256, 8192], pl.INT32] = pl.tensor.assemble(qr_acc_pad, qr_acc, [qr_r0, o_base + ns])
        qr_proj: pl.Tensor[[bs, 8192], pl.FP32] = pl.tensor.create([bs, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for ot_1 in pl.spmd(8, name_hint="idx_qr_proj_dequant_spmd", allow_early_resolve=True):
            o_base: pl.Scalar[pl.INDEX] = ot_1 * 1024
            wq_scale: pl.Tensor[[1, 1024], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(wq_b_scale, [1024], [o_base]), [1, 1024])
            for dq_t0 in pl.range(0, bs, 8):
                acc_fp32: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.cast(pl.tensor.slice(qr_acc_pad, [8, 1024], [dq_t0, o_base]), target_type=pl.FP32, mode='none')
                qr_scale_tile: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(qr_scale, [8, 1], [dq_t0, 0])
                qr_dequant: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
                qr_proj: pl.Tensor[[bs, 8192], pl.FP32] = pl.tensor.assemble(qr_proj, qr_dequant, [dq_t0, o_base])
        qr_proj_flat: pl.Tensor[[bs_heads, 128], pl.FP32] = pl.tensor.reshape(qr_proj, [bs_heads, 128])
        qr_bf16: pl.Tensor[[bs_heads, 128], pl.BF16] = pl.tensor.create([bs_heads, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        rope_swap_idx_t: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.create([32, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rope_swap_idx", allow_early_resolve=True):
            sw_col: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.full([32, 64], dtype=pl.FP32, value=1.0), pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
            sw_dup_f: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(sw_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
            sw_lane: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.sub(sw_col, pl.tensor.muls(sw_dup_f, 2.0))
            rope_swap_idx_t: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_t, pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(sw_col, 1.0), pl.tensor.muls(sw_lane, 2.0)), target_type=pl.INT32, mode='round'), [0, 0])
        for idx in pl.spmd(bs_heads // 32, name_hint="qr_rope_spmd", allow_early_resolve=True):
            o0: pl.Scalar[pl.INDEX] = idx * 32
            token_idx: pl.Scalar[pl.INDEX] = o0 // 64
            rope_swap_idx: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.slice(rope_swap_idx_t, [32, 64], [0, 0])
            cos_row: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos, [1, 64], [token_idx, 0])
            sin_row: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin, [1, 64], [token_idx, 0])
            qr_nope_slice: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(qr_proj_flat, [32, 64], [o0, 0])
            qr_rope_slice: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(qr_proj_flat, [32, 64], [o0, 64])
            qr_swapped: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.gather(qr_rope_slice, rope_swap_idx, dim=-1)
            rope_rot: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(qr_rope_slice, cos_row), pl.tensor.col_expand_mul(qr_swapped, sin_row))
            qr_vec: pl.Tensor[[32, 128], pl.BF16] = pl.tensor.concat(pl.tensor.cast(qr_nope_slice, target_type=pl.BF16, mode='rint'), pl.tensor.cast(rope_rot, target_type=pl.BF16, mode='rint'))
            qr_bf16: pl.Tensor[[bs_heads, 128], pl.BF16] = pl.tensor.assemble(qr_bf16, qr_vec, [o0, 0])
        qh_acc_gm: pl.Tensor[[bs_heads, 128], pl.FP32] = pl.tensor.create([bs_heads, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        for idx_1 in pl.spmd(bs_heads // 64, name_hint="qr_hadamard_matmul_spmd", allow_early_resolve=True):
            o0: pl.Scalar[pl.INDEX] = idx_1 * 64
            qh_acc: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(qr_bf16, [64, 128], [o0, 0]), hadamard, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            qh_acc_gm: pl.Tensor[[bs_heads, 128], pl.FP32] = pl.tensor.assemble(qh_acc_gm, qh_acc, [o0, 0])
        qr_hadamard_i8: pl.Tensor[[16384, 128], pl.INT8] = pl.tensor.create([16384, 128], dtype=pl.INT8, layout=pl.TensorLayout.ND)
        qr_hadamard_scale_dq: pl.Tensor[[16384, 1], pl.FP32] = pl.tensor.create([16384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.spmd(bs_heads // 64, name_hint="qr_hadamard_quant_spmd", allow_early_resolve=True) as qh_quant_tid:
            idx_1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            o0_v1: pl.Scalar[pl.INDEX] = idx_1 * 64
            qh_amax: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0001)
            for h0 in pl.range(0, 128, 64):
                qh_a_f32: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(qh_acc_gm, [64, 64], [o0_v1, h0])
                qh_a_abs: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.maximum(qh_a_f32, pl.tensor.neg(qh_a_f32))
                qh_a_max: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(qh_a_abs), [1, 64])
                qh_amax: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(qh_amax, qh_a_max)
            qh_scale_quant_row: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 64], dtype=pl.FP32, value=127.0), qh_amax)
            qh_scale_dq: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(qh_scale_quant_row), [64, 1])
            qr_hadamard_scale_dq: pl.Tensor[[16384, 1], pl.FP32] = pl.tensor.assemble(qr_hadamard_scale_dq, qh_scale_dq, [o0_v1, 0])
            qh_scale_quant: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(qh_scale_quant_row, [64, 1])
            for h1 in pl.range(0, 128, 64):
                qh_q_f32: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(qh_acc_gm, [64, 64], [o0_v1, h1])
                qh_q_scaled: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.row_expand_mul(qh_q_f32, qh_scale_quant)
                qh_q_i32: pl.Tensor[[64, 64], pl.INT32] = pl.tensor.cast(qh_q_scaled, target_type=pl.INT32, mode='rint')
                qh_q_half: pl.Tensor[[64, 64], pl.FP16] = pl.tensor.cast(qh_q_i32, target_type=pl.FP16, mode='round')
                qh_i8: pl.Tensor[[64, 64], pl.INT8] = pl.tensor.cast(qh_q_half, target_type=pl.INT8, mode='trunc')
                qr_hadamard_i8: pl.Tensor[[16384, 128], pl.INT8] = pl.tensor.assemble(qr_hadamard_i8, qh_i8, [o0_v1, h1])
        x_flat: pl.Tensor[[T_DYN, 4096], pl.BF16] = x
        weights: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        weights_partial: pl.Tensor[[1024, 64], pl.FP32] = pl.tensor.create([1024, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.spmd(4 * row_blocks, name_hint="weights_proj_spmd", deps=[late_dep]) as _weights_tid:
            w_unit: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            w_rb: pl.Scalar[pl.INDEX] = w_unit // 4
            kb: pl.Scalar[pl.INDEX] = w_unit - w_rb * 4
            w_r0: pl.Scalar[pl.INDEX] = w_rb * 16
            w_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - w_r0)
            k_base: pl.Scalar[pl.INDEX] = kb * 1024
            weights_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for db in pl.range(2):
                d0: pl.Scalar[pl.INDEX] = k_base + db * 512
                x_tile: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[w_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [16, 512], [w_r0, d0], [w_rows, 512])
                weights_proj_tile: pl.Tensor[[512, 64], pl.BF16] = pl.tensor.slice(weights_proj, [512, 64], [d0, 0])
                if db == 0:
                    weights_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile, weights_proj_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                else:
                    weights_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(weights_acc, x_tile, weights_proj_tile, a_trans=False, b_trans=False)
            weights_partial: pl.Tensor[[1024, 64], pl.FP32] = pl.tensor.assemble(weights_partial, weights_acc, [kb * 256 + w_r0, 0])
        with pl.spmd(row_blocks, name_hint="weights_proj_reduce_spmd", allow_early_resolve=True) as weights_tid:
            w_rb_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            w_r0_v1: pl.Scalar[pl.INDEX] = w_rb_v1 * 16
            w_sum: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(weights_partial, [16, 64], [w_r0_v1, 0])
            for kb_1 in pl.unroll(1, 4):
                partial_r0: pl.Scalar[pl.INDEX] = kb_1 * 256 + w_r0_v1
                w_sum: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(w_sum, pl.tensor.slice(weights_partial, [16, 64], [partial_r0, 0]))
            weights: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(weights, pl.tensor.muls(w_sum, 0.011048543456039806), [w_r0_v1, 0])
        _tuple_tmp: pl.Tuple[pl.Tensor[[T_DYN, 512], pl.FP32], pl.Tensor[[T_DYN, 512], pl.INT32]] = self.indexer_score_topk_forest(qr_hadamard_i8, qr_hadamard_scale_dq, weights, idx_kv_cache, idx_kv_scale, idx_block_table, position_ids, kv_seq_lens, topk_scores, topk_idxs, qh_quant_tid, weights_tid, cache_write_dep)
        topk_scores: pl.Tensor[[T_DYN, 512], pl.FP32] = _tuple_tmp[0]
        topk_idxs: pl.Tensor[[T_DYN, 512], pl.INT32] = _tuple_tmp[1]
        return topk_scores, topk_idxs
    @pl.function(type=pl.FunctionType.Inline)
    def indexer_compressor(self, x: pl.Tensor[[KV_T_DYN, 4096], pl.BF16], kv: pl.Tensor[[KV_T_DYN, 128], pl.FP32], compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32], compress_state_block_table: pl.Tensor[[KV_B_DYN, 4], pl.INT32], wkv: pl.Tensor[[256, 4096], pl.BF16], wgate: pl.Tensor[[256, 4096], pl.BF16], ape: pl.Tensor[[4, 256], pl.FP32], norm_w: pl.Tensor[[128], pl.BF16], cos: pl.Tensor[[KV_T_DYN, 64], pl.FP32], sin: pl.Tensor[[KV_T_DYN, 64], pl.FP32], hadamard: pl.Tensor[[128, 128], pl.BF16], idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32], position_ids: pl.Tensor[[KV_T_DYN], pl.INT32], idx_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], late_dep: pl.Scalar[pl.TASK_ID]):
        b_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table, 0)
        bs: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        s_dim: pl.Scalar[pl.INDEX] = bs // b_dim
        t_matmul: pl.Scalar[pl.INDEX] = (bs + 16 - 1) // 16 * 16
        rms_blocks: pl.Scalar[pl.INDEX] = (bs + 16 - 1) // 16
        x_flat: pl.Tensor[[KV_T_DYN, 4096], pl.BF16] = x
        kv_proj_pad: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.create([512, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        score_proj_pad: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.create([512, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        compress_state_block_num: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state, 0)
        idx_block_num: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_kv_cache, 0)
        compress_state_flat: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.reshape(compress_state, [compress_state_block_num * 2, 512])
        kv_flat: pl.Tensor[[KV_T_DYN, 128], pl.FP32] = kv
        idx_kv_cache_flat: pl.Tensor[[idx_block_num * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.reshape(idx_kv_cache, [idx_block_num * 32, 128])
        idx_kv_scale_flat: pl.Tensor[[idx_block_num * pl.const(32, pl.INDEX), 1], pl.FP32] = pl.tensor.reshape(idx_kv_scale, [idx_block_num * 32, 1])
        with pl.spmd(t_matmul * 256 // 512, name_hint="kv_score_proj_spmd", deps=[late_dep]) as _kv_score_tid:
            idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            global_row0: pl.Scalar[pl.INDEX] = idx // 8 * 16
            o0: pl.Scalar[pl.INDEX] = idx % 8 * 32
            kv_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            score_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for kb in pl.pipeline(8, stage=2):
                k0: pl.Scalar[pl.INDEX] = kb * 512
                x_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - global_row0)
                x_tile: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[x_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat, [16, 512], [global_row0, k0], [x_rows, 512])
                wkv_tile: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(wkv, [32, 512], [o0, k0])
                wgate_tile: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(wgate, [32, 512], [o0, k0])
                if k0 == 0:
                    kv_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_tile, wkv_tile, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                    score_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_tile, wgate_tile, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                else:
                    kv_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(kv_acc, x_tile, wkv_tile, a_trans=False, b_trans=True)
                    score_acc: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(score_acc, x_tile, wgate_tile, a_trans=False, b_trans=True)
            kv_proj_pad: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.assemble(kv_proj_pad, kv_acc, [global_row0, o0])
            score_proj_pad: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.assemble(score_proj_pad, score_acc, [global_row0, o0])
        pooled_kv: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.create([512, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.spmd(b_dim, name_hint="scatter_softmax_pool_spmd", deps=[_kv_score_tid]) as pool_tid:
            c_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            first_pos_b: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [c_idx * s_dim])
            for s_idx in pl.range(s_dim):
                token: pl.Scalar[pl.INDEX] = c_idx * s_dim + s_idx
                token_pos: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [token])
                pooled_kv: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(pooled_kv, pl.tensor.full([1, 128], dtype=pl.FP32, value=0.0), [token, 0])
                if (pl.cast(token_pos, pl.INDEX) + 1) % 4 == 0:
                    window_start: pl.Scalar[pl.INDEX] = pl.cast(token_pos, pl.INDEX) - 8 + 1
                    for h0 in pl.range(0, 128, 64):
                        last_ape_row: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos, pl.INDEX) % 4, pl.INDEX)
                        mi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(score_proj_pad, [1, 64], [token, 128 + h0]), pl.tensor.slice(ape, [1, 64], [last_ape_row, 128 + h0]))
                        li: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi, mi))
                        oi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(kv_proj_pad, [1, 64], [token, 128 + h0])
                        for state_idx in pl.range(7):
                            logical_pos: pl.Scalar[pl.INDEX] = window_start + state_idx
                            value: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0)
                            score: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                            state_half: pl.Scalar[pl.INDEX] = 0
                            if state_idx >= 4:
                                state_half: pl.Scalar[pl.INDEX] = 128
                            if logical_pos >= 0 and logical_pos < pl.cast(first_pos_b, pl.INDEX):
                                ring_row: pl.Scalar[pl.INDEX] = logical_pos % 8
                                state_page_off: pl.Scalar[pl.INDEX] = ring_row // 2
                                state_blk_id_i32: pl.Scalar[pl.INT32] = pl.tensor.read(compress_state_block_table, [c_idx, state_page_off])
                                if pl.cast(state_blk_id_i32, pl.INDEX) >= 0:
                                    state_blk_id: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32, pl.INDEX)
                                    state_row: pl.Scalar[pl.INDEX] = state_blk_id * 2 + ring_row % 2
                                    value: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat, [1, 64], [state_row, state_half + h0])
                                    score: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat, [1, 64], [state_row, 256 + state_half + h0])
                            if logical_pos >= pl.cast(first_pos_b, pl.INDEX):
                                if logical_pos <= pl.cast(token_pos, pl.INDEX):
                                    overlay_token: pl.Scalar[pl.INDEX] = c_idx * s_dim + logical_pos - pl.cast(first_pos_b, pl.INDEX)
                                    ape_row: pl.Scalar[pl.INDEX] = pl.cast(logical_pos % 4, pl.INDEX)
                                    value: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(kv_proj_pad, [1, 64], [overlay_token, state_half + h0])
                                    score: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(score_proj_pad, [1, 64], [overlay_token, state_half + h0]), pl.tensor.slice(ape, [1, 64], [ape_row, state_half + h0]))
                            mi_next: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(mi, score)
                            alpha: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi, mi_next))
                            beta: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(score, mi_next))
                            li: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha, li), beta)
                            oi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(oi, alpha), pl.tensor.mul(value, beta))
                            mi: pl.Tensor[[1, 64], pl.FP32] = mi_next
                        pooled_kv: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(pooled_kv, pl.tensor.div(oi, li), [token, h0])
        for c_idx_v1 in pl.spmd(b_dim, name_hint="compress_state_commit_spmd", deps=[pool_tid]):
            for s_idx_1 in pl.range(s_dim):
                token: pl.Scalar[pl.INDEX] = c_idx_v1 * s_dim + s_idx_1
                state_row_i64: pl.Scalar[pl.INT64] = pl.tensor.read(inner_state_slot_mapping, [token])
                if state_row_i64 >= 0:
                    state_row: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64, pl.INDEX)
                    token_pos: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids, [token])
                    ape_row: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos, pl.INDEX) % 4, pl.INDEX)
                    compress_state_flat: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.assemble(compress_state_flat, pl.tensor.slice(kv_proj_pad, [1, 256], [token, 0]), [state_row, 0])
                    compress_state_flat: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.assemble(compress_state_flat, pl.tensor.add(pl.tensor.slice(score_proj_pad, [1, 256], [token, 0]), pl.tensor.slice(ape, [1, 256], [ape_row, 0])), [state_row, 256])
        normed_kv: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.create([512, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        norm_w_2d: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.reshape(norm_w, [1, 128])
        with pl.spmd(rms_blocks, name_hint="rmsnorm_rope_spmd", deps=[pool_tid]) as rms_tid:
            rms_blk: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            b0: pl.Scalar[pl.INDEX] = rms_blk * 16
            rms_blk_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - b0)
            cos_b: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cos, [16, 64], [b0, 0], [rms_blk_rows, 64])
            sin_b: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(sin, [16, 64], [b0, 0], [rms_blk_rows, 64])
            partial_sq: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
            for k0_1 in pl.pipeline(0, 128, 64, stage=2):
                kv_rms_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv, [16, 64], [b0, k0_1])
                kv_rms_sq: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_rms_chunk, kv_rms_chunk)
                kv_rms_rowsum: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_rms_sq), [1, 16])
                partial_sq: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq, kv_rms_rowsum)
            variance: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq, 0.0078125), 9.9999999999999995e-07), [16, 1])
            inv_rms: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance))
            for k0_2 in pl.pipeline(0, 64, 64, stage=2):
                kv_norm_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv, [16, 64], [b0, k0_2])
                gamma: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d, [1, 64], [0, k0_2]), target_type=pl.FP32, mode='round')
                normed_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
                normed_kv: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.assemble(normed_kv, pl.tensor.cast(normed_chunk, target_type=pl.BF16, mode='rint'), [b0, k0_2])
            kv_rope_norm: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv, [16, 64], [b0, 64])
            gamma_rope: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d, [1, 64], [0, 64]), target_type=pl.FP32, mode='round')
            rope_normed: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
            rope_ones: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
            rope_col: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(rope_ones, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
            rope_dup_f: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(rope_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
            rope_lane: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(rope_col, pl.tensor.muls(rope_dup_f, 2.0))
            rope_swap_idx: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(rope_col, 1.0), pl.tensor.muls(rope_lane, 2.0)), target_type=pl.INT32, mode='round')
            swapped: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(rope_normed, rope_swap_idx, dim=-1)
            rope_rot: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(rope_normed, cos_b), pl.tensor.mul(swapped, sin_b))
            normed_kv: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.assemble(normed_kv, pl.tensor.cast(rope_rot, target_type=pl.BF16, mode='rint'), [b0, 64])
        kv_final: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.create([512, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.spmd(rms_blocks, name_hint="kv_hadamard_spmd", deps=[rms_tid]) as hadamard_tid:
            had_blk: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            had_b0: pl.Scalar[pl.INDEX] = had_blk * 16
            kv_proj_tile: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.slice(normed_kv, [16, 128], [had_b0, 0])
            for o0_1 in pl.range(0, 128, 64):
                hadamard_tile: pl.Tensor[[128, 64], pl.BF16] = pl.tensor.slice(hadamard, [128, 64], [0, o0_1])
                kv_hadamard_acc: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(kv_proj_tile, hadamard_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                kv_final: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(kv_final, kv_hadamard_acc, [had_b0, o0_1])
        with pl.spmd(rms_blocks, name_hint="kv_and_cache_write_spmd", deps=[hadamard_tid]) as _write_tid:
            wr_blk: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            wr_b0: pl.Scalar[pl.INDEX] = wr_blk * 16
            wr_blk_rows: pl.Scalar[pl.INDEX] = pl.min(16, bs - wr_b0)
            kv_blk_f32: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.slice(kv_final, [16, 128], [wr_b0, 0]), target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
            kv_amax: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(pl.tensor.abs(kv_blk_f32)), [1, 16])
            kv_amax_v1: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.maximum(kv_amax, pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0001))
            kv_scale_q_row: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 16], dtype=pl.FP32, value=127.0), kv_amax_v1)
            kv_scale_dq_col: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(kv_scale_q_row), [16, 1])
            kv_scale_q_col: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(kv_scale_q_row, [16, 1])
            kv_scaled: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(kv_blk_f32, kv_scale_q_col)
            kv_i32: pl.Tensor[[16, 128], pl.INT32] = pl.tensor.cast(kv_scaled, target_type=pl.INT32, mode='rint')
            kv_half: pl.Tensor[[16, 128], pl.FP16] = pl.tensor.cast(kv_i32, target_type=pl.FP16, mode='round')
            kv_i8_blk: pl.Tensor[[16, 128], pl.INT8] = pl.tensor.cast(kv_half, target_type=pl.INT8, mode='trunc')
            for inner in pl.range(wr_blk_rows):
                token: pl.Scalar[pl.INDEX] = wr_b0 + inner
                cache_row_i64: pl.Scalar[pl.INT64] = pl.tensor.read(idx_slot_mapping, [token])
                if cache_row_i64 >= 0:
                    cache_row: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64, pl.INDEX)
                    kv_flat: pl.Tensor[[KV_T_DYN, 128], pl.FP32] = pl.tensor.assemble(kv_flat, pl.tensor.slice(kv_final, [1, 128], [token, 0]), [token, 0])
                    idx_kv_cache_flat: pl.Tensor[[idx_block_num * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.assemble(idx_kv_cache_flat, pl.tensor.slice(kv_i8_blk, [1, 128], [inner, 0]), [cache_row, 0])
                    pl.tensor.write(idx_kv_scale_flat, [cache_row, 0], pl.tensor.read(kv_scale_dq_col, [inner, 0]))
        return _write_tid
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def kv_proj_rope(self, x: pl.Tensor[[KV_T_DYN, 4096], pl.BF16], wkv: pl.Tensor[[4096, 512], pl.BF16], gamma_ckv: pl.Tensor[[512], pl.BF16], rope_cos_il: pl.Tensor[[KV_T_DYN, 64], pl.FP32], rope_sin_signed: pl.Tensor[[KV_T_DYN, 64], pl.FP32], rope_swap_idx: pl.Tensor[[KV_T_DYN, 64], pl.INT32], kv: pl.Tensor[[KV_T_DYN, 512], pl.BF16], late_dep: pl.Scalar[pl.TASK_ID]):
        # KV LoRA, RMSNorm, and RoPE over bounded dense tiles.
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        for tile_base in pl.range(0, t_dim, 512):
            tile_rows: pl.Scalar[pl.INDEX] = pl.min(512, t_dim - tile_base)
            with pl.scope():
                x_view: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.reshape(x, [t_dim, 4096])
                t_matmul: pl.Scalar[pl.INDEX] = (tile_rows + 16 - 1) // 16 * 16
                kv_fp32: pl.Tensor[[t_matmul, 512], pl.FP32] = pl.tensor.create([t_matmul, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
                    for kts0 in pl.range(0, t_matmul, 16):
                        for kvseed0 in pl.range(0, 512, 128):
                            kv_seed: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.full([16, 128], dtype=pl.FP32, value=0.0)
                            kv_fp32: pl.Tensor[[t_matmul, 512], pl.FP32] = pl.tensor.assemble(kv_fp32, kv_seed, [kts0, kvseed0])
                with pl.spmd(32, name_hint="kv_proj_matmul_spmd", deps=[late_dep]) as _kv_tid:
                    kbg: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                    kv_col0: pl.Scalar[pl.INDEX] = kbg // 8 * 128
                    kv_k_base: pl.Scalar[pl.INDEX] = kbg // 4 % 2 * 2048
                    kv_m_group: pl.Scalar[pl.INDEX] = kbg % 4
                    for t0 in pl.range(kv_m_group * 16, t_matmul, 64):
                        kv_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        for db in pl.pipeline(8, stage=2):
                            d0: pl.Scalar[pl.INDEX] = kv_k_base + db * 256
                            kv_rows: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows - t0)
                            x_t0: pl.Scalar[pl.INDEX] = tile_base + t0
                            kv_x_chunk_bf16: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[kv_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_view, [16, 256], [x_t0, d0], [kv_rows, 256])
                            wkv_chunk: pl.Tensor[[256, 128], pl.BF16] = pl.tensor.slice(wkv, [256, 128], [d0, kv_col0])
                            if db == 0:
                                kv_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(kv_x_chunk_bf16, wkv_chunk, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            else:
                                kv_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk, a_trans=False, b_trans=False)
                        kv_fp32: pl.Tensor[[t_matmul, 512], pl.FP32] = pl.tensor.assemble(kv_fp32, kv_acc, [t0, kv_col0], atomic=pl.AtomicType.Add)
                kv_view: pl.Tensor[[t_dim, 512], pl.BF16] = pl.tensor.reshape(kv, [t_dim, 512])
                kv_token_tiles: pl.Scalar[pl.INDEX] = (tile_rows + 16 - 1) // 16
                for tg_idx in pl.spmd(kv_token_tiles, name_hint="kv_rms_norm_rope_spmd"):
                    tg: pl.Scalar[pl.INDEX] = tg_idx * 16
                    valid_rows: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows - tg)
                    out_tg: pl.Scalar[pl.INDEX] = tile_base + tg
                    if valid_rows == 16:
                        kv_sq_sum: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                        for kv_sq_col0 in pl.pipeline(0, 512, 64, stage=2):
                            kv_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32, [16, 64], [tg, kv_sq_col0])
                            kv_sq: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_chunk, kv_chunk)
                            kv_row_sum: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_sq), [1, 16])
                            kv_sq_sum: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(kv_sq_sum, kv_row_sum)
                        kv_inv_rms: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(kv_sq_sum, 0.001953125), 9.9999999999999995e-07), high_precision=True)
                        kv_inv_rms_t: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(kv_inv_rms, [16, 1])
                        for n0 in pl.pipeline(0, 448, 64, stage=2):
                            kv_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32, [16, 64], [tg, n0])
                            gamma_kv_cast: pl.Tensor[[64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_ckv, [64], [n0]), target_type=pl.FP32, mode='round')
                            gamma_kv_chunk: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(gamma_kv_cast, [1, 64])
                            kv_normed: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                            kv_normed_bf16: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(kv_normed, target_type=pl.BF16, mode='rint')
                            kv_view: pl.Tensor[[t_dim, 512], pl.BF16] = pl.tensor.assemble(kv_view, kv_normed_bf16, [out_tg, n0])
                        gamma_rope_cast: pl.Tensor[[64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_ckv, [64], [448]), target_type=pl.FP32, mode='round')
                        gamma_rope: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(gamma_rope_cast, [1, 64])
                        kv_rope_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32, [16, 64], [tg, 448])
                        kv_rope_norm_chunk: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
                        kv_cos_il_full: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(rope_cos_il, [16, 64], [out_tg, 0])
                        kv_sin_signed_full: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(rope_sin_signed, [16, 64], [out_tg, 0])
                        kv_swap_idx_full: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.slice(rope_swap_idx, [16, 64], [out_tg, 0])
                        kv_swapped_full: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(kv_rope_norm_chunk, kv_swap_idx_full, dim=-1)
                        kv_rope_rot_full: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(kv_rope_norm_chunk, kv_cos_il_full), pl.tensor.mul(kv_swapped_full, kv_sin_signed_full))
                        kv_rope_i16_full: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(kv_rope_rot_full, target_type=pl.BF16, mode='rint')
                        kv_view: pl.Tensor[[t_dim, 512], pl.BF16] = pl.tensor.assemble(kv_view, kv_rope_i16_full, [out_tg, 448])
                    else:
                        kv_reduce_tmp: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        kv_sq_sum_tail: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                        for kv_sq_col0_tail in pl.pipeline(0, 512, 64, stage=2):
                            kv_chunk_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(kv_fp32, [tg, kv_sq_col0_tail], [16, 64], [valid_rows, 64], target_memory=pl.Mem.Vec)
                            kv_sq_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.mul(kv_chunk_tail, kv_chunk_tail)
                            kv_row_sum_tail: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows])] = pl.tile.reshape(pl.tile.row_sum(kv_sq_tail, kv_reduce_tmp), [1, 16])
                            kv_sq_sum_tail: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(kv_sq_sum_tail, kv_row_sum_tail)
                        kv_inv_rms_tail: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(kv_sq_sum_tail, 0.001953125), 9.9999999999999995e-07)))
                        kv_inv_rms_t_tail: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_tail, [16, 1])
                        for n0_tail in pl.pipeline(0, 448, 64, stage=2):
                            kv_chunk_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(kv_fp32, [tg, n0_tail], [16, 64], [valid_rows, 64], target_memory=pl.Mem.Vec)
                            gamma_kv_input_tail: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(gamma_ckv, [n0_tail], [64], [64], target_memory=pl.Mem.Vec)
                            gamma_kv_cast_tail: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_kv_input_tail, target_type=pl.FP32, mode='round')
                            gamma_kv_chunk_tail: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(gamma_kv_cast_tail, [1, 64])
                            kv_normed_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_chunk_tail, kv_inv_rms_t_tail), gamma_kv_chunk_tail)
                            kv_normed_bf16_tail: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.cast(kv_normed_tail, target_type=pl.BF16, mode='rint')
                            kv_normed_valid: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.set_validshape(kv_normed_bf16_tail, valid_rows, 64)
                            pl.tile.store(kv_normed_valid, [out_tg, n0_tail], kv_view)
                        gamma_rope_input_tail: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
                        gamma_rope_cast_tail: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_rope_input_tail, target_type=pl.FP32, mode='round')
                        gamma_rope_tail: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(gamma_rope_cast_tail, [1, 64])
                        kv_rope_chunk_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(kv_fp32, [tg, 448], [16, 64], [valid_rows, 64], target_memory=pl.Mem.Vec)
                        kv_rope_norm_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_rope_chunk_tail, kv_inv_rms_t_tail), gamma_rope_tail)
                        kv_cos_il_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(rope_cos_il, [out_tg, 0], [16, 64], [valid_rows, 64], target_memory=pl.Mem.Vec)
                        kv_sin_signed_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.load(rope_sin_signed, [out_tg, 0], [16, 64], [valid_rows, 64], target_memory=pl.Mem.Vec)
                        kv_col: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([16, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                        kv_dup_f: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(kv_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                        kv_lane: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(kv_col, pl.tile.muls(kv_dup_f, 2.0))
                        kv_swap_f: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(kv_col, 1.0), pl.tile.muls(kv_lane, 2.0))
                        kv_row_seed: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                        kv_row_grid: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 16], dtype=pl.FP32, value=1.0), kv_row_seed)
                        kv_row_offset: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(kv_row_grid, 0, 1)
                        kv_swap_idx_tail: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(kv_swap_f, kv_row_offset), target_type=pl.INT32, mode='round')
                        kv_gather_tmp: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                        kv_swapped_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
                        kv_rope_rot_tail: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.add(pl.tile.mul(kv_rope_norm_tail, kv_cos_il_tail), pl.tile.mul(kv_swapped_tail, kv_sin_signed_tail))
                        kv_rope_i16_tail: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.cast(kv_rope_rot_tail, target_type=pl.BF16, mode='rint')
                        kv_rope_valid: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 64])] = pl.tile.set_validshape(kv_rope_i16_tail, valid_rows, 64)
                        pl.tile.store(kv_rope_valid, [out_tg, 448], kv_view)
    @pl.function(type=pl.FunctionType.Inline)
    def o_group_a2a(self, local_groups_out: pl.Tensor[[2048, 4096], pl.BF16], exchange_window: pld.DistributedTensor[[2048, 4096], pl.BF16], exchange_signal: pld.DistributedTensor[[2, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32], publish_dep: pl.Scalar[pl.TASK_ID], publish_count: pl.Scalar[pl.INT32]) -> tuple[pl.Tensor[[2048, 4096], pl.BF16], pld.DistributedTensor[[2, 1], pl.INT32]]:
        # Finish a non-overlapping producer-fused exchange and release its window.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_group_a2a_wait", deps=[publish_dep]) as wait_tid:
            expected: pl.Scalar[pl.INT32] = pl.cast(publish_count, pl.INT32)
            for source_tp in pl.range(2):
                if source_tp != pl.cast(tp_rank, pl.INDEX):
                    pld.system.wait(exchange_signal, [source_tp, 0], expected, cmp=1)
        group_t: pl.Scalar[pl.INDEX] = 512
        with pl.spmd(48, name_hint="o_group_a2a_gather_spmd", deps=[wait_tid]) as gather_tid:
            worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            for local_group in pl.range(4):
                group_base_row: pl.Scalar[pl.INDEX] = local_group * 512
                for group_row in pl.range(worker, group_t, 48):
                    copy_row: pl.Scalar[pl.INDEX] = group_base_row + group_row
                    local_groups_out: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(local_groups_out, pl.tensor.slice(exchange_window, [1, 4096], [copy_row, 0]), [copy_row, 0])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_group_a2a_complete", deps=[gather_tid]):
            completion_anchor: pl.Scalar[pl.BF16] = pl.tensor.read(local_groups_out, [0, 0])
            for peer_tp in pl.range(2):
                if peer_tp != pl.cast(tp_rank, pl.INDEX):
                    pld.system.notify(exchange_signal, pl.cast(group_base, pl.INDEX) + peer_tp, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
            completion_expected: pl.Scalar[pl.INT32] = pl.cast(pl.cast(publish_count, pl.INDEX) + 1, pl.INT32)
            for source_tp_1 in pl.range(2):
                if source_tp_1 != pl.cast(tp_rank, pl.INDEX):
                    pld.system.wait(exchange_signal, [source_tp_1, 0], completion_expected, cmp=1)
            reset_value: pl.Scalar[pl.INT32] = pl.cast(-completion_expected, pl.INT32)
            self_rank: pl.Scalar[pl.INT32] = group_base + tp_rank
            for source_tp_2 in pl.range(2):
                if source_tp_2 != pl.cast(tp_rank, pl.INDEX):
                    pld.system.notify(exchange_signal, self_rank, [source_tp_2, 0], reset_value, op=0)
            pl.tensor.write(local_groups_out, [0, 0], completion_anchor)
        return local_groups_out, exchange_signal
    @pl.function(type=pl.FunctionType.Inline)
    def o_proj_reduce_scatter(self, attention_local_groups: pl.Tensor[[4, 512, 4096], pl.BF16], wo_a: pl.Tensor[[4, 1024, 4096], pl.BF16], wo_b: pl.Tensor[[4096, 4096], pl.INT8], wo_b_scale: pl.Tensor[[4096], pl.FP32], local_t: pl.Scalar[pl.INT32], local_out: pl.Tensor[[T_DYN, 4096], pl.BF16], reduce_window: pld.DistributedTensor[[512, 4096], pl.BF16], reduce_signal: pld.DistributedTensor[[2, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32]) -> tuple[pl.Tensor[[T_DYN, 4096], pl.BF16], pld.DistributedTensor[[2, 1], pl.INT32]]:
        # Project O-B tiles directly into their ReduceScatter owner windows.
        group_t: pl.Scalar[pl.INDEX] = 512
        o_a_rows: pl.Scalar[pl.INDEX] = (group_t + 128 - 1) // 128
        o_b_rows: pl.Scalar[pl.INDEX] = (group_t + 128 - 1) // 128
        o_b_group_t: pl.Scalar[pl.INDEX] = o_b_rows * 128
        owner_rows: pl.Scalar[pl.INDEX] = 16
        attn_2d: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.reshape(attention_local_groups, [2048, 4096])
        wo_a_flat: pl.Tensor[[4096, 4096], pl.BF16] = pl.tensor.reshape(wo_a, [4096, 4096])
        publish_all: pl.Tensor[[512, 4096], pl.BF16] = pl.tensor.create([512, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        put_rows: pl.Scalar[pl.INDEX] = 32
        own_a_rows: pl.Scalar[pl.INDEX] = 2
        own_b_rows: pl.Scalar[pl.INDEX] = 2
        own_b_t: pl.Scalar[pl.INDEX] = own_b_rows * 128
        own_quant_blocks: pl.Scalar[pl.INDEX] = 32
        own_pad_blocks: pl.Scalar[pl.INDEX] = (own_b_t + 8 - 1) // 8
        own_act_rows: pl.Scalar[pl.INDEX] = 16
        for owner in pl.parallel(2):
            own_base: pl.Scalar[pl.INDEX] = owner * 256
            own_a_fp32: pl.Tensor[[256, 4096], pl.FP32] = pl.tensor.create([256, 4096], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            own_a_i8: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.create([256, 4096], dtype=pl.INT8, layout=pl.TensorLayout.ND)
            own_scale: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.create([4, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
            own_b_i32: pl.Tensor[[256, 16384], pl.INT32] = pl.tensor.create([256, 16384], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            for local_group in pl.parallel(4):
                attention_row: pl.Scalar[pl.INDEX] = local_group * 512 + own_base
                o_a_col: pl.Scalar[pl.INDEX] = local_group * 1024
                for pa_unit in pl.spmd(own_a_rows * 8, name_hint="tp_o_a_spmd"):
                    pa_rb: pl.Scalar[pl.INDEX] = pa_unit // 8
                    pa_nb: pl.Scalar[pl.INDEX] = pa_unit - pa_rb * 8
                    pa_t0: pl.Scalar[pl.INDEX] = pa_rb * 128
                    pa_n0: pl.Scalar[pl.INDEX] = pa_nb * 128
                    pa_rows: pl.Scalar[pl.INDEX] = pl.min(128, 256 - pa_t0)
                    pa_src: pl.Scalar[pl.INDEX] = attention_row + pa_t0
                    pa_wrow: pl.Scalar[pl.INDEX] = o_a_col + pa_n0
                    pa_x0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(attn_2d, [128, 256], [pa_src, 0], [pa_rows, 256])
                    pa_w0: pl.Tensor[[128, 256], pl.BF16] = pl.tensor.slice(wo_a_flat, [128, 256], [pa_wrow, 0])
                    pa_acc: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(pa_x0, pa_w0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                    for pa_k0 in pl.pipeline(256, 4096, 256, stage=2):
                        pa_xk: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(attn_2d, [128, 256], [pa_src, pa_k0], [pa_rows, 256])
                        pa_wk: pl.Tensor[[128, 256], pl.BF16] = pl.tensor.slice(wo_a_flat, [128, 256], [pa_wrow, pa_k0])
                        pa_acc: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul_acc(pa_acc, pa_xk, pa_wk, a_trans=False, b_trans=True)
                    pa_valid: pl.Tensor[[128, 128], pl.FP32, pl.TensorView(valid_shape=[pa_rows, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(pa_acc, pa_rows, 128)
                    own_a_fp32: pl.Tensor[[256, 4096], pl.FP32] = pl.tensor.assemble(own_a_fp32, pa_valid, [pa_t0, pa_wrow])
                for qz_worker in pl.spmd(6, name_hint="tp_o_a_quant_spmd"):
                    for qz_blk in pl.range(qz_worker, own_quant_blocks, 6):
                        qz_t: pl.Scalar[pl.INDEX] = qz_blk * 8
                        qz_rows: pl.Scalar[pl.INDEX] = pl.min(8, 256 - qz_t)
                        qz_tile: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.slice(own_a_fp32, [8, 1024], [qz_t, o_a_col])
                        qz_amax: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(pl.tensor.abs(qz_tile)), [1, 8])
                        qz_floor: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0001)
                        qz_amax_v1: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qz_floor, qz_amax)
                        qz_max: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=127.0)
                        qz_sq: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.div(qz_max, qz_amax_v1)
                        qz_sdq: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.recip(qz_sq)
                        own_scale: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.assemble(own_scale, pl.tensor.set_validshape(qz_sdq, 1, qz_rows), [local_group, qz_t])
                        qz_sq_col: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qz_sq, [8, 1])
                        qz_scaled: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.row_expand_mul(qz_tile, qz_sq_col)
                        qz_i32: pl.Tensor[[8, 1024], pl.INT32] = pl.tensor.cast(qz_scaled, target_type=pl.INT32, mode='rint')
                        qz_f16: pl.Tensor[[8, 1024], pl.FP16] = pl.tensor.cast(qz_i32, target_type=pl.FP16, mode='round')
                        qz_i8: pl.Tensor[[8, 1024], pl.INT8] = pl.tensor.cast(qz_f16, target_type=pl.INT8, mode='trunc')
                        own_a_i8: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.assemble(own_a_i8, pl.tensor.set_validshape(qz_i8, qz_rows, 1024), [qz_t, o_a_col])
                    for qz_pad in pl.range(own_quant_blocks + qz_worker, own_pad_blocks, 6):
                        qz_pt: pl.Scalar[pl.INDEX] = qz_pad * 8
                        qz_prows: pl.Scalar[pl.INDEX] = pl.min(8, own_b_t - qz_pt)
                        qz_zero: pl.Tensor[[8, 1024], pl.FP16] = pl.tensor.full([8, 1024], dtype=pl.FP16, value=0.0)
                        qz_zero_i8: pl.Tensor[[8, 1024], pl.INT8] = pl.tensor.cast(qz_zero, target_type=pl.INT8, mode='trunc')
                        own_a_i8: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.assemble(own_a_i8, pl.tensor.set_validshape(qz_zero_i8, qz_prows, 1024), [qz_pt, o_a_col])
                for pb_unit in pl.spmd(own_b_rows * 8, name_hint="tp_o_b_spmd"):
                    pb_tb: pl.Scalar[pl.INDEX] = pb_unit // 8
                    pb_db: pl.Scalar[pl.INDEX] = pb_unit - pb_tb * 8
                    pb_t0: pl.Scalar[pl.INDEX] = pb_tb * 128
                    pb_d0: pl.Scalar[pl.INDEX] = pb_db * 512
                    for pb_n0 in pl.range(pb_d0, pb_d0 + 512, 256):
                        pb_x0: pl.Tensor[[128, 256], pl.INT8] = pl.tensor.slice(own_a_i8, [128, 256], [pb_t0, o_a_col])
                        pb_w0: pl.Tensor[[256, 256], pl.INT8] = pl.tensor.slice(wo_b, [256, 256], [pb_n0, o_a_col])
                        pb_acc: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul(pb_x0, pb_w0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.INT32)
                        for pb_k0 in pl.pipeline(256, 1024, 256, stage=2):
                            pb_bk: pl.Scalar[pl.INDEX] = o_a_col + pb_k0
                            pb_xk: pl.Tensor[[128, 256], pl.INT8] = pl.tensor.slice(own_a_i8, [128, 256], [pb_t0, pb_bk])
                            pb_wk: pl.Tensor[[256, 256], pl.INT8] = pl.tensor.slice(wo_b, [256, 256], [pb_n0, pb_bk])
                            pb_acc: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul_acc(pb_acc, pb_xk, pb_wk, a_trans=False, b_trans=True)
                        pb_col: pl.Scalar[pl.INDEX] = local_group * 4096 + pb_n0
                        own_b_i32: pl.Tensor[[256, 16384], pl.INT32] = pl.tensor.assemble(own_b_i32, pb_acc, [pb_t0, pb_col])
            for dq_worker in pl.spmd(12, name_hint="tp_o_b_dequant_spmd", optimizations=[pl.cross_core_slot(slot_num=2)]):
                for dq_blk in pl.range(dq_worker, own_act_rows * 8, 12):
                    dq_rb: pl.Scalar[pl.INDEX] = dq_blk // 8
                    dq_nb: pl.Scalar[pl.INDEX] = dq_blk - dq_rb * 8
                    dq_row: pl.Scalar[pl.INDEX] = dq_rb * 16
                    dq_n0: pl.Scalar[pl.INDEX] = dq_nb * 512
                    dq_rows: pl.Scalar[pl.INDEX] = pl.min(16, 256 - dq_row)
                    dq_acc: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                    for dq_group in pl.pipeline(4, stage=2):
                        dq_col: pl.Scalar[pl.INDEX] = dq_group * 4096 + dq_n0
                        dq_i32: pl.Tensor[[16, 512], pl.INT32, pl.TensorView(valid_shape=[dq_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(own_b_i32, [16, 512], [dq_row, dq_col], [dq_rows, 512])
                        dq_fp32: pl.Tensor[[16, 512], pl.FP32, pl.TensorView(valid_shape=[dq_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.cast(dq_i32, target_type=pl.FP32, mode='none')
                        dq_srow: pl.Tensor[[1, 16], pl.FP32, pl.TensorView(valid_shape=[1, dq_rows], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(own_scale, [1, 16], [dq_group, dq_row], [1, dq_rows])
                        dq_scol: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[dq_rows, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.reshape(dq_srow, [16, 1])
                        dq_acc: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.add(dq_acc, pl.tensor.row_expand_mul(dq_fp32, dq_scol))
                    dq_wscale: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(wo_b_scale, [512], [dq_n0]), [1, 512])
                    dq_bf16: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.cast(pl.tensor.col_expand_mul(dq_acc, dq_wscale), target_type=pl.BF16, mode='rint')
                    dq_stage: pl.Scalar[pl.INDEX] = owner * 256 + dq_row
                    publish_all: pl.Tensor[[512, 4096], pl.BF16] = pl.tensor.assemble(publish_all, pl.tensor.set_validshape(dq_bf16, dq_rows, 512), [dq_stage, dq_n0])
        with pl.spmd(24, name_hint="tp_o_b_publish_spmd") as publish_tid:
            pub_worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            for pub_blk in pl.range(pub_worker, 2 * put_rows, 24):
                pub_owner: pl.Scalar[pl.INDEX] = pub_blk // put_rows
                pub_row_block: pl.Scalar[pl.INDEX] = pub_blk - pub_owner * put_rows
                pub_owner_row: pl.Scalar[pl.INDEX] = pub_row_block * 8
                pub_rows: pl.Scalar[pl.INDEX] = pl.min(8, 256 - pub_owner_row)
                pub_src_row: pl.Scalar[pl.INDEX] = pub_owner * 256 + pub_owner_row
                pub_dst_row: pl.Scalar[pl.INDEX] = pl.cast(tp_rank, pl.INDEX) * 256 + pub_owner_row
                pld.tensor.put(reduce_window, pl.cast(group_base, pl.INDEX) + pub_owner, publish_all, [pub_dst_row, 0], [pub_src_row, 0], [pub_rows, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
            for notify_owner in pl.range(2):
                if notify_owner != pl.cast(tp_rank, pl.INDEX):
                    pld.system.notify(reduce_signal, pl.cast(group_base, pl.INDEX) + notify_owner, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_rs_wait", deps=[publish_tid]) as wait_tid:
            expected: pl.Scalar[pl.INT32] = pl.cast(24, pl.INT32)
            for source_tp in pl.range(2):
                if source_tp != pl.cast(tp_rank, pl.INDEX):
                    pld.system.wait(reduce_signal, [source_tp, 0], expected, cmp=1)
        with pl.spmd(48, name_hint="tp_o_rs_reduce_spmd", deps=[wait_tid]) as reduce_tid:
            worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            for block in pl.range(worker, 256, 48):
                local_row: pl.Scalar[pl.INDEX] = block // 1
                d_block: pl.Scalar[pl.INDEX] = block - local_row * 1
                d0: pl.Scalar[pl.INDEX] = d_block * 4096
                own_partial: pl.Tile[[1, 4096], pl.BF16] = pl.tile.load(reduce_window, [local_row, d0], [1, 4096], [1, 4096])
                reduce_acc: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(own_partial, target_type=pl.FP32, mode='none')
                for source_tp_1 in pl.range(1, 2):
                    source_row: pl.Scalar[pl.INDEX] = source_tp_1 * 256 + local_row
                    source_partial: pl.Tile[[1, 4096], pl.BF16] = pl.tile.load(reduce_window, [source_row, d0], [1, 4096], [1, 4096])
                    source_fp32: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(source_partial, target_type=pl.FP32, mode='none')
                    reduce_acc: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(reduce_acc, source_fp32)
                reduced: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(reduce_acc, target_type=pl.BF16, mode='rint')
                pl.tile.store(reduced, [local_row, d0], local_out)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_rs_complete", deps=[reduce_tid]):
            completion_anchor: pl.Scalar[pl.BF16] = pl.tensor.read(local_out, [0, 0])
            for peer_tp in pl.range(2):
                if peer_tp != pl.cast(tp_rank, pl.INDEX):
                    pld.system.notify(reduce_signal, pl.cast(group_base, pl.INDEX) + peer_tp, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
            completion_expected: pl.Scalar[pl.INT32] = pl.cast(25, pl.INT32)
            for source_tp_2 in pl.range(2):
                if source_tp_2 != pl.cast(tp_rank, pl.INDEX):
                    pld.system.wait(reduce_signal, [source_tp_2, 0], completion_expected, cmp=1)
            reset_value: pl.Scalar[pl.INT32] = pl.cast(-25, pl.INT32)
            self_rank: pl.Scalar[pl.INT32] = group_base + tp_rank
            for source_tp_3 in pl.range(2):
                if source_tp_3 != pl.cast(tp_rank, pl.INDEX):
                    pld.system.notify(reduce_signal, self_rank, [source_tp_3, 0], reset_value, op=0)
            pl.tensor.write(local_out, [0, 0], completion_anchor)
        return local_out, reduce_signal
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def q_proj_rope(self, x: pl.Tensor[[T_DYN, 4096], pl.BF16], wq_a: pl.Tensor[[4096, 1024], pl.BF16], wq_b: pl.Tensor[[1024, 32768], pl.INT8], wq_b_scale: pl.Tensor[[32768], pl.FP32], gamma_cq: pl.Tensor[[1024], pl.BF16], rope_cos_il: pl.Tensor[[T_DYN, 64], pl.FP32], rope_sin_signed: pl.Tensor[[T_DYN, 64], pl.FP32], rope_swap_idx: pl.Tensor[[T_DYN, 64], pl.INT32], q: pl.Tensor[[T_DYN, 64, 512], pl.BF16], qr: pl.Tensor[[T_DYN, 1024], pl.INT8], qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32]):
        # Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles.
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        for tile_base in pl.range(0, t_dim, 512):
            tile_rows: pl.Scalar[pl.INDEX] = pl.min(512, t_dim - tile_base)
            with pl.scope():
                x_view: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.reshape(x, [t_dim, 4096])
                qr_t_matmul: pl.Scalar[pl.INDEX] = (tile_rows + 16 - 1) // 16 * 16
                qproj_t_matmul: pl.Scalar[pl.INDEX] = (tile_rows + 16 - 1) // 16 * 16
                qproj_full_rows: pl.Scalar[pl.INDEX] = tile_rows // 64 * 64
                qr_fp32: pl.Tensor[[qr_t_matmul, 1024], pl.FP32] = pl.tensor.create([qr_t_matmul, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
                    for ts0 in pl.range(0, qr_t_matmul, 16):
                        for nseed0 in pl.range(0, 1024, 128):
                            qr_seed: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.full([16, 128], dtype=pl.FP32, value=0.0)
                            qr_fp32: pl.Tensor[[qr_t_matmul, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32, qr_seed, [ts0, nseed0])
                for qbg_idx in pl.spmd(16, name_hint="qr_proj_matmul_spmd", allow_early_resolve=True):
                    q_a_col0: pl.Scalar[pl.INDEX] = qbg_idx // 2 * 128
                    qr_k_base: pl.Scalar[pl.INDEX] = qbg_idx % 2 * 2048
                    for t0 in pl.range(0, qr_t_matmul, 16):
                        q_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        for db in pl.pipeline(8, stage=2):
                            qr_d0: pl.Scalar[pl.INDEX] = qr_k_base + db * 256
                            qr_rows: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows - t0)
                            x_t0: pl.Scalar[pl.INDEX] = tile_base + t0
                            q_x_chunk_bf16: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[qr_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_view, [16, 256], [x_t0, qr_d0], [qr_rows, 256])
                            w_chunk: pl.Tensor[[256, 128], pl.BF16] = pl.tensor.slice(wq_a, [256, 128], [qr_d0, q_a_col0])
                            if db == 0:
                                q_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(q_x_chunk_bf16, w_chunk, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            else:
                                q_acc: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk, a_trans=False, b_trans=False)
                        qr_fp32: pl.Tensor[[qr_t_matmul, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=pl.AtomicType.Add)
                qr_view: pl.Tensor[[t_dim, 1024], pl.INT8] = pl.tensor.reshape(qr, [t_dim, 1024])
                qr_scale_view: pl.Tensor[[t_dim, 1], pl.FP32] = pl.tensor.reshape(qr_scale, [t_dim, 1])
                qr_i8_matmul: pl.Tensor[[qproj_t_matmul, 1024], pl.INT8] = pl.tensor.create([qproj_t_matmul, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                qr_scale_pad_store: pl.Tensor[[qproj_t_matmul, 1], pl.FP32] = pl.tensor.create([qproj_t_matmul, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                qr_token_tiles: pl.Scalar[pl.INDEX] = (tile_rows + 8 - 1) // 8
                for tg_idx in pl.spmd(qr_token_tiles, name_hint="qr_rms_norm_quant_spmd", allow_early_resolve=True):
                    tg: pl.Scalar[pl.INDEX] = tg_idx * 8
                    valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, tile_rows - tg)
                    out_tg: pl.Scalar[pl.INDEX] = tile_base + tg
                    qr_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                    qr_amax_g: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                    for qr_rms_col0 in pl.pipeline(0, 1024, 256, stage=2):
                        qr_rms_chunk: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.slice(qr_fp32, [8, 256], [tg, qr_rms_col0])
                        qr_rms_sq: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.mul(qr_rms_chunk, qr_rms_chunk)
                        qr_rms_row_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(qr_rms_sq), [1, 8])
                        qr_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(qr_sq_sum, qr_rms_row_sum)
                        gamma_rms_cast: pl.Tensor[[256], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_cq, [256], [qr_rms_col0]), target_type=pl.FP32, mode='round')
                        gamma_rms_chunk: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.reshape(gamma_rms_cast, [1, 256])
                        qr_g: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
                        qr_g_abs: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.abs(qr_g)
                        qr_g_row_max: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(qr_g_abs), [1, 8])
                        qr_amax_g: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qr_amax_g, qr_g_row_max)
                    qr_inv_rms: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(qr_sq_sum, 0.0009765625), 9.9999999999999995e-07), high_precision=True)
                    qr_inv_rms_t: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qr_inv_rms, [8, 1])
                    qr_amax_floor: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0001)
                    qr_amax_normed: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.mul(qr_inv_rms, qr_amax_g)
                    qr_tile_amax: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qr_amax_floor, qr_amax_normed)
                    qr_scale_quant_row: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 8], dtype=pl.FP32, value=127.0), qr_tile_amax)
                    qr_scale_quant_t: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qr_scale_quant_row, [8, 1])
                    qr_tile_scale_dq: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(qr_scale_quant_row), [8, 1])
                    qr_scale_pad_store: pl.Tensor[[qproj_t_matmul, 1], pl.FP32] = pl.tensor.assemble(qr_scale_pad_store, qr_tile_scale_dq, [tg, 0])
                    if valid_rows == 8:
                        qr_scale_view: pl.Tensor[[t_dim, 1], pl.FP32] = pl.tensor.assemble(qr_scale_view, qr_tile_scale_dq, [out_tg, 0])
                    else:
                        qr_scale_tail: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 1])] = pl.tile.load(qr_scale_pad_store, [tg, 0], [8, 1], [valid_rows, 1], target_memory=pl.Mem.Vec)
                        pl.tile.store(qr_scale_tail, [out_tg, 0], qr_scale_view)
                    for qa in pl.pipeline(0, 1024, 256, stage=2):
                        qr_chunk: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.slice(qr_fp32, [8, 256], [tg, qa])
                        gamma_q_cast: pl.Tensor[[256], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_cq, [256], [qa]), target_type=pl.FP32, mode='round')
                        gamma_q_chunk: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.reshape(gamma_q_cast, [1, 256])
                        qr_q_normed: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
                        qr_q_scaled: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(qr_q_normed, qr_scale_quant_t)
                        qr_q_i32: pl.Tensor[[8, 256], pl.INT32] = pl.tensor.cast(qr_q_scaled, target_type=pl.INT32, mode='rint')
                        qr_q_half: pl.Tensor[[8, 256], pl.FP16] = pl.tensor.cast(qr_q_i32, target_type=pl.FP16, mode='round')
                        qr_q_i8: pl.Tensor[[8, 256], pl.INT8] = pl.tensor.cast(qr_q_half, target_type=pl.INT8, mode='trunc')
                        qr_i8_matmul: pl.Tensor[[qproj_t_matmul, 1024], pl.INT8] = pl.tensor.assemble(qr_i8_matmul, qr_q_i8, [tg, qa])
                        if valid_rows == 8:
                            qr_view: pl.Tensor[[t_dim, 1024], pl.INT8] = pl.tensor.assemble(qr_view, qr_q_i8, [out_tg, qa])
                        else:
                            qr_q_tail: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(qr_i8_matmul, [tg, qa], [8, 256], [valid_rows, 256], target_memory=pl.Mem.Vec)
                            pl.tile.store(qr_q_tail, [out_tg, qa], qr_view)
                q_proj_i32: pl.Tensor[[qproj_t_matmul, 32768], pl.INT32] = pl.tensor.create([qproj_t_matmul, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                for qproj_n_idx in pl.spmd(64, name_hint="qproj_matmul_spmd"):
                    w_col0: pl.Scalar[pl.INDEX] = qproj_n_idx * 512
                    for t0_1 in pl.range(0, qproj_full_rows, 64):
                        col_acc: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.create([64, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                        for qr_proj_col0 in pl.pipeline(0, 1024, 128, stage=2):
                            qr_i8_chunk: pl.Tensor[[64, 128], pl.INT8] = pl.tensor.slice(qr_i8_matmul, [64, 128], [t0_1, qr_proj_col0])
                            wq_chunk: pl.Tensor[[128, 512], pl.INT8] = pl.tensor.slice(wq_b, [128, 512], [qr_proj_col0, w_col0])
                            if qr_proj_col0 == 0:
                                col_acc: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.matmul(qr_i8_chunk, wq_chunk, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                            else:
                                col_acc: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.matmul_acc(col_acc, qr_i8_chunk, wq_chunk, a_trans=False, b_trans=False)
                        q_proj_i32: pl.Tensor[[qproj_t_matmul, 32768], pl.INT32] = pl.tensor.assemble(q_proj_i32, col_acc, [t0_1, w_col0])
                    tail_w_col0: pl.Scalar[pl.INDEX] = w_col0
                    for tail_t0 in pl.range(qproj_full_rows, qproj_t_matmul, 16):
                        qproj_tail_rows: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows - tail_t0)
                        tail_acc: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.create([16, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                        for tail_qr_col0 in pl.pipeline(0, 1024, 128, stage=2):
                            qr_i8_tail: pl.Tensor[[16, 128], pl.INT8, pl.TensorView(valid_shape=[qproj_tail_rows, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr_i8_matmul, [16, 128], [tail_t0, tail_qr_col0], [qproj_tail_rows, 128])
                            wq_tail: pl.Tensor[[128, 512], pl.INT8] = pl.tensor.slice(wq_b, [128, 512], [tail_qr_col0, tail_w_col0])
                            if tail_qr_col0 == 0:
                                tail_acc: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul(qr_i8_tail, wq_tail, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                            else:
                                tail_acc: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul_acc(tail_acc, qr_i8_tail, wq_tail, a_trans=False, b_trans=False)
                        q_proj_i32: pl.Tensor[[qproj_t_matmul, 32768], pl.INT32] = pl.tensor.assemble(q_proj_i32, tail_acc, [tail_t0, tail_w_col0])
                q_flat: pl.Tensor[[t_dim, 32768], pl.BF16] = pl.tensor.reshape(q, [t_dim, 32768])
                for hg_idx in pl.spmd(16, name_hint="qproj_dequant_rms_nope_rope_spmd", allow_early_resolve=True):
                    hg: pl.Scalar[pl.INDEX] = hg_idx * 4
                    for tg_1 in pl.range(0, tile_rows, 8):
                        out_tg: pl.Scalar[pl.INDEX] = tile_base + tg_1
                        if tg_1 + 8 <= tile_rows:
                            qr_scale_dq_t: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(qr_scale_pad_store, [8, 1], [tg_1, 0])
                            q_cos_il: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(rope_cos_il, [8, 64], [out_tg, 0])
                            q_sin_signed: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(rope_sin_signed, [8, 64], [out_tg, 0])
                            q_swap_idx: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.slice(rope_swap_idx, [8, 64], [out_tg, 0])
                            for h_inner in pl.pipeline(4, stage=2):
                                h: pl.Scalar[pl.INDEX] = hg + h_inner
                                h0: pl.Scalar[pl.INDEX] = h * 512
                                q_head_acc: pl.Tensor[[8, 512], pl.INT32] = pl.tensor.slice(q_proj_i32, [8, 512], [tg_1, h0])
                                q_head_scale: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(wq_b_scale, [512], [h0]), [1, 512])
                                q_head_acc_fp32: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.cast(q_head_acc, target_type=pl.FP32, mode='none')
                                q_head_row_scaled: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                                q_head_dq: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.col_expand_mul(q_head_row_scaled, q_head_scale)
                                q_head_sq: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(q_head_dq, q_head_dq)
                                q_head_sq_row: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_sum(q_head_sq)
                                q_head_sq_sum: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(q_head_sq_row, [1, 8])
                                q_head_sq_mean: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.muls(q_head_sq_sum, 0.001953125)
                                q_head_var: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.adds(q_head_sq_mean, 9.9999999999999995e-07)
                                q_head_inv_rms: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(q_head_var, high_precision=True)
                                q_head_inv_rms_t: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(q_head_inv_rms, [8, 1])
                                q_nope_normed: pl.Tensor[[8, 448], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_head_dq, [8, 448], [0, 0]), q_head_inv_rms_t)
                                q_nope_bf16: pl.Tensor[[8, 448], pl.BF16] = pl.tensor.cast(q_nope_normed, target_type=pl.BF16, mode='rint')
                                q_flat: pl.Tensor[[t_dim, 32768], pl.BF16] = pl.tensor.assemble(q_flat, q_nope_bf16, [out_tg, h0])
                                q_rope_chunk_raw: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(q_head_dq, [8, 64], [0, 448])
                                q_rope_chunk: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                                q_rope_swapped: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(q_rope_chunk, q_swap_idx, dim=-1)
                                q_rope_base: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(q_rope_chunk, q_cos_il)
                                q_rope_delta: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(q_rope_swapped, q_sin_signed)
                                q_rope_rot: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(q_rope_base, q_rope_delta)
                                q_rope_bf16: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.cast(q_rope_rot, target_type=pl.BF16, mode='rint')
                                q_flat: pl.Tensor[[t_dim, 32768], pl.BF16] = pl.tensor.assemble(q_flat, q_rope_bf16, [out_tg, h0 + 448])
                        else:
                            valid_tail_rows: pl.Scalar[pl.INDEX] = tile_rows - tg_1
                            qr_scale_dq_tail: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(qr_scale_pad_store, [tg_1, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                            q_cos_il_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.load(rope_cos_il, [out_tg, 0], [8, 64], [valid_tail_rows, 64], target_memory=pl.Mem.Vec)
                            q_sin_signed_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.load(rope_sin_signed, [out_tg, 0], [8, 64], [valid_tail_rows, 64], target_memory=pl.Mem.Vec)
                            q_col: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                            q_dup_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(q_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                            q_lane: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(q_col, pl.tile.muls(q_dup_f, 2.0))
                            q_swap_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(q_col, 1.0), pl.tile.muls(q_lane, 2.0))
                            q_row_seed: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                            q_row_grid: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), q_row_seed)
                            q_row_offset: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(q_row_grid, 0, 1)
                            q_swap_idx_tail: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(q_swap_f, q_row_offset), target_type=pl.INT32, mode='round')
                            q_head_reduce_tmp: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            q_gather_tmp: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                            for h_inner_tail in pl.range(4):
                                h_tail: pl.Scalar[pl.INDEX] = hg + h_inner_tail
                                h0_tail: pl.Scalar[pl.INDEX] = h_tail * 512
                                q_head_acc_tail: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(q_proj_i32, [tg_1, h0_tail], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                                q_head_scale_input_tail: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(wq_b_scale, [h0_tail], [512], [512], target_memory=pl.Mem.Vec)
                                q_head_scale_tail: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_head_scale_input_tail, [1, 512])
                                q_head_acc_fp32_tail: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(q_head_acc_tail, target_type=pl.FP32, mode='none')
                                q_head_row_scaled_tail: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_head_acc_fp32_tail, qr_scale_dq_tail)
                                q_head_dq_tail: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_head_row_scaled_tail, q_head_scale_tail)
                                q_head_sq_tail: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_head_dq_tail, q_head_dq_tail)
                                q_head_sq_sum_tail: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_tail, q_head_reduce_tmp)
                                q_head_inv_rms_tail: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(q_head_sq_sum_tail, 0.001953125), 9.9999999999999995e-07)))
                                q_nope_normed_tail: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(pl.tile.slice(q_head_dq_tail, [8, 448], [0, 0]), q_head_inv_rms_tail)
                                q_nope_bf16_tail: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_nope_normed_tail, target_type=pl.BF16, mode='rint')
                                q_nope_valid: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 448])] = pl.tile.set_validshape(q_nope_bf16_tail, valid_tail_rows, 448)
                                pl.tile.store(q_nope_valid, [out_tg, h0_tail], q_flat)
                                q_rope_chunk_raw_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail, [8, 64], [0, 448])
                                q_rope_chunk_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_rope_chunk_raw_tail, q_head_inv_rms_tail)
                                q_rope_swapped_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(q_rope_chunk_tail, q_swap_idx_tail, q_gather_tmp)
                                q_rope_base_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_rope_chunk_tail, q_cos_il_tail)
                                q_rope_delta_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_rope_swapped_tail, q_sin_signed_tail)
                                q_rope_rot_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(q_rope_base_tail, q_rope_delta_tail)
                                q_rope_bf16_tail: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_rope_rot_tail, target_type=pl.BF16, mode='rint')
                                q_rope_valid: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.set_validshape(q_rope_bf16_tail, valid_tail_rows, 64)
                                pl.tile.store(q_rope_valid, [out_tg, h0_tail + 448], q_flat)
    @pl.function(type=pl.FunctionType.Inline)
    def rms_norm(self, x: pl.Tensor[[T_DYN, 4096], pl.BF16], norm_w: pl.Tensor[[4096], pl.BF16], x_normed: pl.Tensor[[T_DYN, 4096], pl.BF16]):
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)
        token_tiles: pl.Scalar[pl.INDEX] = (t_dim + 8 - 1) // 8
        with pl.spmd(token_tiles, name_hint="rms_norm_spmd", allow_early_resolve=True) as rms_tid:
            tg_idx: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            tg: pl.Scalar[pl.INDEX] = tg_idx * 8
            valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - tg)
            if valid_rows == 8:
                self._rms_norm_full_tile(x, norm_w, x_normed)
            else:
                self._rms_norm_tail_tile(x, norm_w, x_normed)
        return rms_tid
    @pl.function(type=pl.FunctionType.Inline)
    def rope_prepare(self, rope_cos: pl.Tensor[[KV_T_DYN, 64], pl.BF16], rope_sin: pl.Tensor[[KV_T_DYN, 64], pl.BF16], rope_cos_il: pl.Tensor[[KV_T_DYN, 64], pl.FP32], rope_sin_signed: pl.Tensor[[KV_T_DYN, 64], pl.FP32], rope_swap_idx: pl.Tensor[[KV_T_DYN, 64], pl.INT32]):
        # Build the head-invariant interleaved cos / sign-folded sin / swap-index rope rows.
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(rope_cos, 0)
        rope_cos_view: pl.Tensor[[t_dim, 64], pl.BF16] = pl.tensor.reshape(rope_cos, [t_dim, 64])
        rope_sin_view: pl.Tensor[[t_dim, 64], pl.BF16] = pl.tensor.reshape(rope_sin, [t_dim, 64])
        rope_cos_il_view: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.reshape(rope_cos_il, [t_dim, 64])
        rope_sin_signed_view: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.reshape(rope_sin_signed, [t_dim, 64])
        rope_swap_idx_view: pl.Tensor[[t_dim, 64], pl.INT32] = pl.tensor.reshape(rope_swap_idx, [t_dim, 64])
        token_tiles: pl.Scalar[pl.INDEX] = (t_dim + 8 - 1) // 8
        for qrp_idx in pl.spmd(token_tiles, name_hint="q_rope_prepare_spmd", allow_early_resolve=True):
            qrp_t0: pl.Scalar[pl.INDEX] = qrp_idx * 8
            qrp_valid_rows: pl.Scalar[pl.INDEX] = pl.min(8, t_dim - qrp_t0)
            qrp_ones: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
            qrp_idx_i32: pl.Tensor[[1, 64], pl.INT32] = pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            qrp_idx_fp32: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(qrp_idx_i32, target_type=pl.FP32, mode='round')
            qrp_col: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(qrp_ones, qrp_idx_fp32)
            qrp_half: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_col, 0.5)
            qrp_dup_i32: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_half, target_type=pl.INT32, mode='trunc')
            qrp_dup_f: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_dup_i32, target_type=pl.FP32, mode='round')
            qrp_dup_idx: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_dup_f, target_type=pl.INT32, mode='round')
            qrp_lane: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_col, pl.tensor.muls(qrp_dup_f, 2.0))
            qrp_next_col: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(qrp_col, 1.0)
            qrp_lane_offset: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_lane, 2.0)
            qrp_swap_f: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_next_col, qrp_lane_offset)
            qrp_swap_idx: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_swap_f, target_type=pl.INT32, mode='round')
            qrp_sign: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(qrp_lane, 2.0), 1.0)
            if qrp_valid_rows == 8:
                qrp_cos_rows_full: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_cos_view, [8, 64], [qrp_t0, 0])
                qrp_sin_rows_full: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_sin_view, [8, 64], [qrp_t0, 0])
                qrp_cos_full: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_cos_rows_full, target_type=pl.FP32, mode='round')
                qrp_sin_full: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_sin_rows_full, target_type=pl.FP32, mode='round')
                qrp_cos_il_full: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_cos_full, qrp_dup_idx, dim=-1)
                qrp_sin_il_full: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_sin_full, qrp_dup_idx, dim=-1)
                qrp_sin_signed_full: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(qrp_sin_il_full, qrp_sign)
                rope_cos_il_view: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.assemble(rope_cos_il_view, qrp_cos_il_full, [qrp_t0, 0])
                rope_sin_signed_view: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed_view, qrp_sin_signed_full, [qrp_t0, 0])
                rope_swap_idx_view: pl.Tensor[[t_dim, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_view, qrp_swap_idx, [qrp_t0, 0])
            else:
                qrp_cos_rows_tail: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.load(rope_cos_view, [qrp_t0, 0], [8, 64], [qrp_valid_rows, 64], target_memory=pl.Mem.Vec)
                qrp_sin_rows_tail: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.load(rope_sin_view, [qrp_t0, 0], [8, 64], [qrp_valid_rows, 64], target_memory=pl.Mem.Vec)
                qrp_tail_col: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                qrp_tail_dup_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(qrp_tail_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                qrp_tail_lane: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(qrp_tail_col, pl.tile.muls(qrp_tail_dup_f, 2.0))
                qrp_tail_swap_f: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(qrp_tail_col, 1.0), pl.tile.muls(qrp_tail_lane, 2.0))
                qrp_row_seed: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                qrp_row_grid: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), qrp_row_seed)
                qrp_row_offset: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(qrp_row_grid, 0, 1)
                qrp_dup_idx_tail: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(qrp_tail_dup_f, qrp_row_offset), target_type=pl.INT32, mode='round')
                qrp_gather_tmp: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                qrp_cos_il_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_cos_rows_tail, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail, qrp_gather_tmp)
                qrp_sin_il_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_sin_rows_tail, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail, qrp_gather_tmp)
                qrp_tail_sign: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(pl.tile.muls(qrp_tail_lane, 2.0), 1.0)
                qrp_sin_signed_tail: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_tail, qrp_tail_sign)
                pl.tile.store(pl.tile.set_validshape(qrp_cos_il_tail, qrp_valid_rows, 64), [qrp_t0, 0], rope_cos_il_view)
                pl.tile.store(pl.tile.set_validshape(qrp_sin_signed_tail, qrp_valid_rows, 64), [qrp_t0, 0], rope_sin_signed_view)
                pl.tile.store(pl.tile.set_validshape(pl.tile.cast(qrp_tail_swap_f, target_type=pl.INT32, mode='round'), qrp_valid_rows, 64), [qrp_t0, 0], rope_swap_idx_view)
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def sparse_attn_csa(self, q: pl.Tensor[[T_DYN, 64, 512], pl.BF16], ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16], window_swa_indices: pl.Tensor[[T_DYN, 128], pl.INT32], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16], cmp_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], idx_topk: pl.Tensor[[T_DYN, 512], pl.INT32], position_ids: pl.Tensor[[T_DYN, 1], pl.INT32], freqs_cos: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_sin: pl.Tensor[[T_DYN, 64], pl.BF16], cache_ready_dep: pl.Scalar[pl.TASK_ID]):
        # Plan and run CSA QK/PV over sparse blocks, and build inverse-RoPE metadata.
        ori_block_num: pl.Scalar[pl.INDEX] = pl.tensor.dim(ori_kv, 0)
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(q, 0)
        t_heads: pl.Scalar[pl.INDEX] = t_dim * 64
        t_blk: pl.Scalar[pl.INDEX] = t_dim * 4 * 5 * 16
        qk_items: pl.Scalar[pl.INDEX] = t_dim * 5
        rope_cs_blocks: pl.Scalar[pl.INDEX] = t_dim // 8
        ori_kv_flat: pl.Tensor[[ori_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(ori_kv, [ori_block_num * 32, 512])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_touch", allow_early_resolve=True):
            ori_kv_flat: pl.Tensor[[ori_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(ori_kv_flat, pl.tensor.slice(ori_kv_flat, [1, 512], [0, 0]), [0, 0])
        sparse_bias: pl.Tensor[[t_dim, 640], pl.FP32] = pl.tensor.create([t_dim, 640], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_sparse_indices: pl.Tensor[[t_dim, 512], pl.INT32] = pl.tensor.create([t_dim, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        valid_block_mask: pl.Tensor[[t_dim, 5], pl.INT32] = pl.tensor.create([t_dim, 5], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        qk_order: pl.Tensor[[1280], pl.INT32] = pl.tensor.create([1280], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        qk_wcur: pl.Tensor[[1], pl.INT32] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_slots_build_valid_qk_plan") as qk_plan_tid:
            for bias_t0 in pl.range(0, t_dim, 8):
                c_raw: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.cast(pl.tensor.slice(idx_topk, [8, 512], [bias_t0, 0]), target_type=pl.FP32, mode='round')
                c_pos: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.cast(pl.tensor.slice(position_ids, [8, 1], [bias_t0, 0]), target_type=pl.FP32, mode='round')
                c_pos_scaled: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.muls(pl.tensor.adds(c_pos, 1.0), 0.25)
                c_pos_i32: pl.Tensor[[8, 1], pl.INT32] = pl.tensor.cast(c_pos_scaled, target_type=pl.INT32, mode='trunc')
                c_pos_q: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.cast(c_pos_i32, target_type=pl.FP32, mode='round')
                c_upper_b: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.full([8, 512], dtype=pl.FP32, value=1.0), c_pos_q)
                c_ge: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.adds(c_raw, 1.0), 0.0), 1.0)
                c_lt: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.sub(c_upper_b, c_raw), 0.0), 1.0)
                c_mask: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(c_ge, c_lt)
                c_out: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.subs(pl.tensor.mul(c_mask, pl.tensor.adds(c_raw, 1.0)), 1.0)
                cmp_sparse_indices: pl.Tensor[[t_dim, 512], pl.INT32] = pl.tensor.assemble(cmp_sparse_indices, pl.tensor.cast(c_out, target_type=pl.INT32, mode='round'), [bias_t0, 0])
                for c_t0 in pl.range(8):
                    pl.tensor.write(valid_block_mask, [bias_t0 + c_t0, 0], pl.cast(1, pl.INT32))
                for c_sb in pl.range(1, 5):
                    c_s0: pl.Scalar[pl.INDEX] = (c_sb - 1) * 128
                    c_blk_valid: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_max(pl.tensor.slice(c_mask, [8, 128], [0, c_s0]))
                    for c_dt in pl.range(8):
                        c_valid: pl.Scalar[pl.INT32] = pl.cast(pl.tensor.read(c_blk_valid, [c_dt, 0]), pl.INT32)
                        pl.tensor.write(valid_block_mask, [bias_t0 + c_dt, c_sb], c_valid)
                v_win_f: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(window_swa_indices, [8, 128], [bias_t0, 0]), target_type=pl.FP32, mode='round')
                v_win_valid: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.adds(v_win_f, 1.0), 0.0), 1.0)
                sparse_bias: pl.Tensor[[t_dim, 640], pl.FP32] = pl.tensor.assemble(sparse_bias, pl.tensor.muls(pl.tensor.subs(v_win_valid, 1.0), 1e+20), [bias_t0, 0])
                sparse_bias: pl.Tensor[[t_dim, 640], pl.FP32] = pl.tensor.assemble(sparse_bias, pl.tensor.muls(pl.tensor.minimum(c_out, 0.0), 1e+20), [bias_t0, 128])
                if pl.const(640, pl.INDEX) > pl.const(640, pl.INDEX):
                    sparse_bias: pl.Tensor[[t_dim, 640], pl.FP32] = pl.tensor.assemble(sparse_bias, pl.tensor.full([8, 0], dtype=pl.FP32, value=-1e+20), [bias_t0, 640])
            pl.tensor.write(qk_wcur, [0], pl.cast(0, pl.INT32))
            for plan_t in pl.range(t_dim):
                for plan_sb in pl.range(5):
                    if pl.cast(pl.tensor.read(valid_block_mask, [plan_t, plan_sb]), pl.INDEX) > 0:
                        plan_w: pl.Scalar[pl.INT32] = pl.tensor.read(qk_wcur, [0])
                        pl.tensor.write(qk_order, [plan_w], pl.cast(plan_t * 5 + plan_sb, pl.INT32))
                        pl.tensor.write(qk_wcur, [0], pl.cast(pl.cast(plan_w, pl.INDEX) + 1, pl.INT32))
            for plan_t_1 in pl.range(t_dim):
                for plan_sb_1 in pl.range(5):
                    if pl.cast(pl.tensor.read(valid_block_mask, [plan_t_1, plan_sb_1]), pl.INDEX) <= 0:
                        plan_w_v1: pl.Scalar[pl.INT32] = pl.tensor.read(qk_wcur, [0])
                        pl.tensor.write(qk_order, [plan_w_v1], pl.cast(plan_t_1 * 5 + plan_sb_1, pl.INT32))
                        pl.tensor.write(qk_wcur, [0], pl.cast(pl.cast(plan_w_v1, pl.INDEX) + 1, pl.INT32))
        cmp_block_num: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv, 0)
        cmp_kv_flat: pl.Tensor[[cmp_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(cmp_kv, [cmp_block_num * 32, 512])
        q_flat: pl.Tensor[[t_heads, 512], pl.BF16] = pl.tensor.reshape(q, [t_heads, 512])
        sparse_blk_mi: pl.Tensor[[t_blk, 1], pl.FP32] = pl.tensor.create([t_blk, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        sparse_blk_li: pl.Tensor[[t_blk, 1], pl.FP32] = pl.tensor.create([t_blk, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        sparse_blk_oi: pl.Tensor[[t_blk, 512], pl.FP32] = pl.tensor.create([t_blk, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.spmd(24, name_hint="qk_pv_spmd", deps=[qk_plan_tid, cache_ready_dep], allow_early_resolve=True) as qk_tid:
            qk_core: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            qk_lane_iters: pl.Scalar[pl.INDEX] = (qk_items - qk_core + 24 - 1) // 24
            for qk_it in pl.range(qk_lane_iters):
                qk_flat: pl.Scalar[pl.INDEX] = qk_core + qk_it * 24
                qk_item: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(qk_order, [qk_flat]), pl.INDEX)
                qk_t: pl.Scalar[pl.INDEX] = qk_item // 5
                qk_sb: pl.Scalar[pl.INDEX] = qk_item - qk_t * 5
                qk_b: pl.Scalar[pl.INDEX] = qk_t // 8
                qk_token_base: pl.Scalar[pl.INDEX] = qk_t * 4 * 5 * 16
                qk_s0: pl.Scalar[pl.INDEX] = qk_sb * 128
                qk_bias_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(sparse_bias, [1, 128], [qk_t, qk_s0])
                qk_block_valid: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask, [qk_t, qk_sb])
                if pl.cast(qk_block_valid, pl.INDEX) > 0:
                    qk_kv: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.create_l1([128, 512], dtype=pl.BF16, transpose=False)
                    qk_win_rows: pl.Scalar[pl.INDEX] = pl.min(pl.max(128 - qk_s0, 0), 128)
                    if qk_win_rows > 0:
                        qk_pos: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(position_ids, [qk_t, 0]), pl.INDEX)
                        qk_win_len: pl.Scalar[pl.INDEX] = pl.min(qk_pos + 1, 128)
                        qk_win_start: pl.Scalar[pl.INDEX] = qk_pos - qk_win_len + 1
                        qk_run_rows: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len - qk_s0, 0), qk_win_rows)
                        qk_head: pl.Scalar[pl.INDEX] = (qk_win_start + qk_s0) % 32
                        for qk_run in pl.unroll(5):
                            qk_run_lo: pl.Scalar[pl.INDEX] = pl.max(qk_run * 32 - qk_head, 0)
                            qk_run_hi: pl.Scalar[pl.INDEX] = pl.min((qk_run + 1) * 32 - qk_head, qk_run_rows)
                            if qk_run_hi > qk_run_lo:
                                qk_run_raw: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices, [qk_t, qk_s0 + qk_run_lo])
                                qk_run_src: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw, pl.INDEX), 0), pl.INDEX)
                                qk_kv: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv, ori_kv_flat, [qk_run_lo, 0], [qk_run_src, 0], [128, 512], valid_shape=[qk_run_hi - qk_run_lo, 512], transpose=False)
                        qk_tail_n: pl.Scalar[pl.INDEX] = qk_win_rows - qk_run_rows
                        if qk_tail_n > 0:
                            qk_kv: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv, ori_kv_flat, [qk_run_rows, 0], [0, 0], [128, 512], valid_shape=[qk_tail_n, 512], transpose=False)
                    for qk_r in pl.range(qk_win_rows, 128):
                        qk_cmp_k: pl.Scalar[pl.INDEX] = qk_s0 + qk_r - 128
                        if qk_cmp_k < 512:
                            qk_ridx: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices, [qk_t, qk_cmp_k])
                            if pl.cast(qk_ridx, pl.INDEX) >= 0:
                                qk_slot: pl.Scalar[pl.INT32] = qk_ridx
                                qk_cblk: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(cmp_block_table, [qk_b, pl.cast(qk_slot, pl.INDEX) // 32]), pl.INDEX)
                                qk_csrc: pl.Scalar[pl.INDEX] = qk_cblk * 32 + pl.cast(qk_slot, pl.INDEX) % 32
                                qk_kv: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv, cmp_kv_flat, [qk_r, 0], [qk_csrc, 0], [1, 512], transpose=False)
                            else:
                                qk_kv: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, 512], transpose=False)
                        else:
                            qk_kv: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, 512], transpose=False)
                    for qk_hb in pl.pipeline(2, stage=2):
                        qk_h0: pl.Scalar[pl.INDEX] = qk_hb * 32
                        qk_head_row: pl.Scalar[pl.INDEX] = qk_t * 64 + qk_h0
                        qk_q_tile: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(q_flat, [32, 512], [qk_head_row, 0])
                        qk_raw: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.matmul(qk_q_tile, qk_kv, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        qk_scaled: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.muls(qk_raw, 0.044194173824159223)
                        qk_scores: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.col_expand_add(qk_scaled, qk_bias_row)
                        qk_mi: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.row_max(qk_scores)
                        qk_exp: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(qk_scores, qk_mi))
                        qk_li: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.row_sum(qk_exp)
                        qk_exp_bf16: pl.Tensor[[32, 128], pl.BF16] = pl.tensor.cast(qk_exp, target_type=pl.BF16, mode='rint')
                        qk_oi: pl.Tensor[[32, 512], pl.FP32] = pl.tensor.matmul(qk_exp_bf16, qk_kv, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        for qk_sub in pl.unroll(2):
                            qk_h_idx: pl.Scalar[pl.INDEX] = qk_hb * 2 + qk_sub
                            qk_r0: pl.Scalar[pl.INDEX] = qk_sub * 16
                            qk_blk_base: pl.Scalar[pl.INDEX] = qk_token_base + qk_h_idx * 5 * 16
                            qk_row: pl.Scalar[pl.INDEX] = qk_blk_base + qk_sb * 16
                            sparse_blk_mi: pl.Tensor[[t_blk, 1], pl.FP32] = pl.tensor.assemble(sparse_blk_mi, pl.tensor.slice(qk_mi, [16, 1], [qk_r0, 0]), [qk_row, 0])
                            sparse_blk_li: pl.Tensor[[t_blk, 1], pl.FP32] = pl.tensor.assemble(sparse_blk_li, pl.tensor.slice(qk_li, [16, 1], [qk_r0, 0]), [qk_row, 0])
                            sparse_blk_oi: pl.Tensor[[t_blk, 512], pl.FP32] = pl.tensor.assemble(sparse_blk_oi, pl.tensor.slice(qk_oi, [16, 512], [qk_r0, 0]), [qk_row, 0])
                else:
                    qk_oi_zero: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                    for qk_h_idx_1 in pl.range(4):
                        qk_blk_base: pl.Scalar[pl.INDEX] = qk_token_base + qk_h_idx_1 * 5 * 16
                        qk_row: pl.Scalar[pl.INDEX] = qk_blk_base + qk_sb * 16
                        for qk_hr in pl.range(16):
                            pl.tensor.write(sparse_blk_mi, [qk_row + qk_hr, 0], -3.0000000000000001e+38)
                            pl.tensor.write(sparse_blk_li, [qk_row + qk_hr, 0], 0.0)
                        sparse_blk_oi: pl.Tensor[[t_blk, 512], pl.FP32] = pl.tensor.assemble(sparse_blk_oi, qk_oi_zero, [qk_row, 0])
        rope_cos_il: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        rope_sin_signed: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        rope_swap_idx: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.create([16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True) as rope_tid:
            sw_ones: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
            sw_idx_f: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round')
            sw_col: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(sw_ones, sw_idx_f)
            sw_dup_i32: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.muls(sw_col, 0.5), target_type=pl.INT32, mode='trunc')
            sw_dup_f: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(sw_dup_i32, target_type=pl.FP32, mode='round')
            sw_lane: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(sw_col, pl.tensor.muls(sw_dup_f, 2.0))
            sw_swap_f: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(pl.tensor.adds(sw_col, 1.0), pl.tensor.muls(sw_lane, 2.0))
            rope_swap_idx: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx, pl.tensor.cast(sw_swap_f, target_type=pl.INT32, mode='round'), [0, 0])
            cs_ones: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
            cs_idx_f: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round')
            cs_col: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(cs_ones, cs_idx_f)
            cs_dup_i32: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(pl.tensor.muls(cs_col, 0.5), target_type=pl.INT32, mode='trunc')
            cs_dup_f: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(cs_dup_i32, target_type=pl.FP32, mode='round')
            cs_dup_idx: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(cs_dup_f, target_type=pl.INT32, mode='round')
            cs_lane: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(cs_col, pl.tensor.muls(cs_dup_f, 2.0))
            cs_sign: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.neg(pl.tensor.subs(pl.tensor.muls(cs_lane, 2.0), 1.0))
            for cs_rb in pl.range(rope_cs_blocks):
                cs_t0: pl.Scalar[pl.INDEX] = cs_rb * 8
                cs_cos: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.cast(pl.tensor.slice(freqs_cos, [8, 32], [cs_t0, 0]), target_type=pl.FP32, mode='round')
                cs_sin: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.cast(pl.tensor.slice(freqs_sin, [8, 32], [cs_t0, 0]), target_type=pl.FP32, mode='round')
                rope_cos_il: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(rope_cos_il, pl.tensor.gather(cs_cos, cs_dup_idx, dim=-1), [cs_t0, 0])
                cs_sin_il: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(cs_sin, cs_dup_idx, dim=-1)
                rope_sin_signed: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed, pl.tensor.mul(cs_sin_il, cs_sign), [cs_t0, 0])
        return sparse_blk_mi, sparse_blk_li, sparse_blk_oi, rope_cos_il, rope_sin_signed, rope_swap_idx, qk_tid, rope_tid
    @pl.function(type=pl.FunctionType.Inline, auto_scope=False)
    def decode_csa(self, x_hc: pl.Tensor[[T_DYN, 4, 4096], pl.FP32], hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[24], pl.FP32], attn_norm_w: pl.Tensor[[4096], pl.BF16], wq_a: pl.Tensor[[4096, 1024], pl.BF16], wq_b: pl.Tensor[[1024, 32768], pl.INT8], wq_b_scale: pl.Tensor[[32768], pl.FP32], wkv: pl.Tensor[[4096, 512], pl.BF16], gamma_cq: pl.Tensor[[1024], pl.BF16], gamma_ckv: pl.Tensor[[512], pl.BF16], freqs_cos_local: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_sin_local: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_cos: pl.Tensor[[KV_T_DYN, 64], pl.BF16], freqs_sin: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_freqs_cos: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_freqs_sin: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], cmp_ape: pl.Tensor[[4, 1024], pl.FP32], cmp_norm_w: pl.Tensor[[512], pl.BF16], compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32], compress_state_block_table: pl.Tensor[[KV_B_DYN, 4], pl.INT32], idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], weights_proj: pl.Tensor[[4096, 64], pl.BF16], hadamard_idx: pl.Tensor[[128, 128], pl.BF16], inner_wkv: pl.Tensor[[256, 4096], pl.BF16], inner_wgate: pl.Tensor[[256, 4096], pl.BF16], inner_ape: pl.Tensor[[4, 256], pl.FP32], inner_norm_w: pl.Tensor[[128], pl.BF16], inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32], inner_compress_state_block_table: pl.Tensor[[KV_B_DYN, 4], pl.INT32], kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16], cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16], cmp_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8], idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32], idx_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], window_swa_indices: pl.Tensor[[T_DYN, 128], pl.INT32], window_swa_lens: pl.Tensor[[T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], idx_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], position_ids_local: pl.Tensor[[T_DYN], pl.INT32], position_ids: pl.Tensor[[KV_T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], attn_sink: pl.Tensor[[64], pl.FP32], wo_a: pl.Tensor[[4, 1024, 4096], pl.BF16], wo_b: pl.Tensor[[4096, 4096], pl.INT8], wo_b_scale: pl.Tensor[[4096], pl.FP32], x_out: pl.Tensor[[T_DYN, 4, 4096], pl.FP32], gather_window: pld.DistributedTensor[[512, 4096], pl.BF16], gather_signal: pld.DistributedTensor[[2, 1], pl.INT32], attention_window: pld.DistributedTensor[[2048, 4096], pl.BF16], attention_signal: pld.DistributedTensor[[2, 1], pl.INT32], o_window: pld.DistributedTensor[[512, 4096], pl.BF16], o_signal: pld.DistributedTensor[[2, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32]) -> pl.Tensor[[T_DYN, 4, 4096], pl.FP32]:
        # Run one rank of the context-parallel CSA layer.
        t_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc, 0)
        kv_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(ori_slot_mapping, 0)
        kv_b_dim: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table, 0)
        q: pl.Tensor[[t_dim, 64, 512], pl.BF16] = pl.tensor.create([t_dim, 64, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        idx_topk_scores: pl.Tensor[[t_dim, 512], pl.FP32] = pl.tensor.create([t_dim, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        idx_topk: pl.Tensor[[t_dim, 512], pl.INT32] = pl.tensor.create([t_dim, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        post_t: pl.Tensor[[t_dim, 4], pl.FP32] = pl.tensor.create([t_dim, 4], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        comb_t: pl.Tensor[[t_dim, 16], pl.FP32] = pl.tensor.create([t_dim, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        x_mixed: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.create([t_dim, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            self.hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)
        idx_cos_il: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.create([t_dim, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        idx_sin_signed: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.create([t_dim, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_cos_il_full: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.create([kv_dim, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_sin_signed_full: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.create([kv_dim, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_interleave") as rope_tid:
            il_ones: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.full([4, 64], dtype=pl.FP32, value=1.0)
            il_col: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.col_expand_mul(il_ones, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
            il_dup_f: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(il_col, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
            il_dup_idx: pl.Tensor[[4, 64], pl.INT32] = pl.tensor.cast(il_dup_f, target_type=pl.INT32, mode='round')
            il_lane: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.sub(il_col, pl.tensor.muls(il_dup_f, 2.0))
            il_sign: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(il_lane, 2.0), 1.0)
            for rope_t0 in pl.range(0, t_dim, 4):
                idx_cos_il: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.assemble(idx_cos_il, pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(freqs_cos_local, [4, 32], [rope_t0, 0]), target_type=pl.FP32, mode='round'), il_dup_idx, dim=-1), [rope_t0, 0])
                idx_sin_signed: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.assemble(idx_sin_signed, pl.tensor.mul(pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(freqs_sin_local, [4, 32], [rope_t0, 0]), target_type=pl.FP32, mode='round'), il_dup_idx, dim=-1), il_sign), [rope_t0, 0])
            for cmp_t0 in pl.range(0, kv_dim, 4):
                cmp_cos_il_full: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.assemble(cmp_cos_il_full, pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(cmp_freqs_cos, [4, 32], [cmp_t0, 0]), target_type=pl.FP32, mode='round'), il_dup_idx, dim=-1), [cmp_t0, 0])
                cmp_sin_signed_full: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.assemble(cmp_sin_signed_full, pl.tensor.mul(pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(cmp_freqs_sin, [4, 32], [cmp_t0, 0]), target_type=pl.FP32, mode='round'), il_dup_idx, dim=-1), il_sign), [cmp_t0, 0])
        x_normed_t: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.create([t_dim, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            self.rms_norm(x_mixed, attn_norm_w, x_normed_t)
        kv_wb_blocks: pl.Scalar[pl.INDEX] = kv_dim // 8
        x_normed_full: pl.Tensor[[kv_dim, 4096], pl.BF16] = pl.tensor.create([kv_dim, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            _tuple_tmp: pl.Tuple[pl.Tensor[[kv_dim, 4096], pl.BF16], pld.DistributedTensor[[2, 1], pl.INT32]] = self.decode_cp_token_allgather_step(x_normed_t, x_normed_full, gather_window, gather_signal, group_base, tp_rank)
            _gathered_normed: pl.Tensor[[kv_dim, 4096], pl.BF16] = _tuple_tmp[0]
            gather_signal: pld.DistributedTensor[[2, 1], pl.INT32] = _tuple_tmp[1]
        kv_full: pl.Tensor[[kv_dim, 512], pl.BF16] = pl.tensor.create([kv_dim, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        qr: pl.Tensor[[t_dim, 1024], pl.INT8] = pl.tensor.create([t_dim, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
        qr_scale: pl.Tensor[[t_dim, 1], pl.FP32] = pl.tensor.create([t_dim, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        position_ids_t1: pl.Tensor[[t_dim, 1], pl.INT32] = pl.tensor.reshape(position_ids_local, [t_dim, 1])
        attention_local_flat: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.create([2048, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        attn_out: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.create([t_dim, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            late_dep: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[rope_tid])
            kv_cos_il: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.create([kv_dim, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_sin_signed: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.create([kv_dim, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_swap_idx: pl.Tensor[[kv_dim, 64], pl.INT32] = pl.tensor.create([kv_dim, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            self.rope_prepare(freqs_cos, freqs_sin, kv_cos_il, kv_sin_signed, kv_swap_idx)
            q_cos_il: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.create([t_dim, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_sin_signed: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.create([t_dim, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_swap_idx: pl.Tensor[[t_dim, 64], pl.INT32] = pl.tensor.create([t_dim, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            self.rope_prepare(freqs_cos_local, freqs_sin_local, q_cos_il, q_sin_signed, q_swap_idx)
            self.q_proj_rope(x_normed_t, wq_a, wq_b, wq_b_scale, gamma_cq, q_cos_il, q_sin_signed, q_swap_idx, q, qr, qr_scale)
            self.kv_proj_rope(x_normed_full, wkv, gamma_ckv, kv_cos_il, kv_sin_signed, kv_swap_idx, kv_full, late_dep)
            ori_block_num: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache, 0)
            kv_cache_flat: pl.Tensor[[ori_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(kv_cache, [ori_block_num * 32, 512])
            with pl.spmd(kv_wb_blocks, name_hint="csa_cache_writeback_spmd") as ori_cache_write_tid:
                wb_blk: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                wb_t0: pl.Scalar[pl.INDEX] = wb_blk * 8
                for write_dt in pl.range(8):
                    write_t: pl.Scalar[pl.INDEX] = wb_t0 + write_dt
                    write_row_i64: pl.Scalar[pl.INT64] = pl.tensor.read(ori_slot_mapping, [write_t])
                    if write_row_i64 >= 0:
                        write_row: pl.Scalar[pl.INDEX] = pl.cast(write_row_i64, pl.INDEX)
                        kv_cache_flat: pl.Tensor[[ori_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(kv_cache_flat, pl.tensor.slice(kv_full, [1, 512], [write_t, 0]), [write_row, 0])
            cmp_positions: pl.Tensor[[kv_dim], pl.INT32] = pl.tensor.reshape(position_ids, [kv_dim])
            cmp_slots: pl.Tensor[[kv_dim], pl.INT64] = pl.tensor.reshape(cmp_slot_mapping, [kv_dim])
            cmp_state_slots: pl.Tensor[[kv_dim], pl.INT64] = pl.tensor.reshape(state_slot_mapping, [kv_dim])
            idx_slots: pl.Tensor[[kv_dim], pl.INT64] = pl.tensor.reshape(idx_slot_mapping, [kv_dim])
            idx_positions: pl.Tensor[[t_dim], pl.INT32] = pl.tensor.reshape(position_ids_local, [t_dim])
            inner_state_slots: pl.Tensor[[kv_dim], pl.INT64] = pl.tensor.reshape(inner_state_slot_mapping, [kv_dim])
            cmp_state_table: pl.Tensor[[kv_b_dim, 4], pl.INT32] = pl.tensor.reshape(compress_state_block_table, [kv_b_dim, 4])
            inner_state_table: pl.Tensor[[kv_b_dim, 4], pl.INT32] = pl.tensor.reshape(inner_compress_state_block_table, [kv_b_dim, 4])
            cmp_out: pl.Tensor[[kv_dim, 512], pl.FP32] = pl.tensor.create([kv_dim, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            _tuple_tmp_1: pl.Tuple[pl.Tensor[[kv_dim, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = self.compressor_ratio4(x_normed_full, cmp_out, compress_state, cmp_state_table, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, cmp_cos_il_full, cmp_sin_signed_full, cmp_kv, cmp_positions, cmp_slots, cmp_state_slots, late_dep)
            cmp_out: pl.Tensor[[kv_dim, 512], pl.FP32] = _tuple_tmp_1[0]
            cmp_cache_write_tid: pl.Scalar[pl.TASK_ID] = _tuple_tmp_1[1]
            cache_ready_dep: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[ori_cache_write_tid, cmp_cache_write_tid])
            idx_kv_unused: pl.Tensor[[kv_dim, 128], pl.FP32] = pl.tensor.create([kv_dim, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            idx_cache_write_tid: pl.Scalar[pl.TASK_ID] = self.indexer_compressor(x_normed_full, idx_kv_unused, inner_compress_state, inner_state_table, inner_wkv, inner_wgate, inner_ape, inner_norm_w, cmp_cos_il_full, cmp_sin_signed_full, hadamard_idx, idx_kv_cache, idx_kv_scale, cmp_positions, idx_slots, inner_state_slots, late_dep)
            _tuple_tmp_2: pl.Tuple[pl.Tensor[[t_dim, 512], pl.FP32], pl.Tensor[[t_dim, 512], pl.INT32]] = self.indexer(x_normed_t, qr, qr_scale, idx_wq_b, idx_wq_b_scale, weights_proj, idx_cos_il, idx_sin_signed, hadamard_idx, idx_kv_cache, idx_kv_scale, idx_block_table, idx_topk_scores, idx_topk, idx_positions, kv_seq_lens, late_dep, idx_cache_write_tid)
            idx_topk_scores: pl.Tensor[[t_dim, 512], pl.FP32] = _tuple_tmp_2[0]
            idx_topk: pl.Tensor[[t_dim, 512], pl.INT32] = _tuple_tmp_2[1]
            _tuple_tmp_3: pl.Tuple[pl.Tensor[[t_blk, 1], pl.FP32], pl.Tensor[[t_blk, 1], pl.FP32], pl.Tensor[[t_blk, 512], pl.FP32], pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[16, 64], pl.INT32], pl.Scalar[pl.TASK_ID], pl.Scalar[pl.TASK_ID]] = self.sparse_attn_csa(q, kv_cache, window_swa_indices, cmp_kv, cmp_block_table, idx_topk, position_ids_t1, freqs_cos_local, freqs_sin_local, cache_ready_dep)
            sparse_blk_mi: pl.Tensor[[t_blk, 1], pl.FP32] = _tuple_tmp_3[0]
            sparse_blk_li: pl.Tensor[[t_blk, 1], pl.FP32] = _tuple_tmp_3[1]
            sparse_blk_oi: pl.Tensor[[t_blk, 512], pl.FP32] = _tuple_tmp_3[2]
            rope_cos_il: pl.Tensor[[256, 64], pl.FP32] = _tuple_tmp_3[3]
            rope_sin_signed: pl.Tensor[[256, 64], pl.FP32] = _tuple_tmp_3[4]
            rope_swap_idx: pl.Tensor[[16, 64], pl.INT32] = _tuple_tmp_3[5]
            qk_tid: pl.Scalar[pl.TASK_ID] = _tuple_tmp_3[6]
            attn_rope_tid: pl.Scalar[pl.TASK_ID] = _tuple_tmp_3[7]
            attention_grouped: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.create([2048, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            pack_work_count: pl.Scalar[pl.INDEX] = t_dim // 8 * 4
            with pl.spmd(48, name_hint="csa_merge_pack_publish_spmd", deps=[qk_tid, attn_rope_tid]) as publish_tid:
                worker: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for pack_work in pl.range(worker, pack_work_count, 48):
                    token_block: pl.Scalar[pl.INDEX] = pack_work // 4
                    m_h_idx: pl.Scalar[pl.INDEX] = pack_work - token_block * 4
                    m_t0: pl.Scalar[pl.INDEX] = token_block * 8
                    m_h0: pl.Scalar[pl.INDEX] = m_h_idx * 16
                    global_group0: pl.Scalar[pl.INDEX] = m_h0 // 8
                    destination_rank: pl.Scalar[pl.INDEX] = global_group0 // 4
                    local_group0: pl.Scalar[pl.INDEX] = global_group0 - destination_rank * 4
                    for m_dt in pl.range(8):
                        m_t: pl.Scalar[pl.INDEX] = m_t0 + m_dt
                        m_idx: pl.Scalar[pl.INDEX] = m_t * 4 + m_h_idx
                        m_blk_base: pl.Scalar[pl.INDEX] = m_idx * 5 * 16
                        m_mi: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi, [16, 1], [m_blk_base, 0])
                        m_li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li, [16, 1], [m_blk_base, 0])
                        m_oi: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi, [16, 512], [m_blk_base, 0])
                        for m_sb in pl.pipeline(1, 5, stage=2):
                            m_row: pl.Scalar[pl.INDEX] = m_blk_base + m_sb * 16
                            m_cur_mi: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi, [16, 1], [m_row, 0])
                            m_cur_li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li, [16, 1], [m_row, 0])
                            m_cur_oi: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi, [16, 512], [m_row, 0])
                            m_mi_new: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.maximum(m_mi, m_cur_mi)
                            m_alpha: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_mi, m_mi_new))
                            m_beta: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_cur_mi, m_mi_new))
                            m_li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(m_alpha, m_li), pl.tensor.mul(m_beta, m_cur_li))
                            m_oi: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(m_oi, m_alpha), pl.tensor.row_expand_mul(m_cur_oi, m_beta))
                            m_mi: pl.Tensor[[16, 1], pl.FP32] = m_mi_new
                        n_sink_bias: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(attn_sink, [16], [m_h0]), [16, 1])
                        n_sink_tile: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.sub(m_mi, m_mi), n_sink_bias)
                        n_denom: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(m_li, pl.tensor.exp(pl.tensor.sub(n_sink_tile, m_mi)))
                        n_full: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(pl.tensor.row_expand_div(m_oi, n_denom), [16, 512], [0, 0])
                        n_bf16: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.cast(n_full, target_type=pl.BF16, mode='rint')
                        m_rope: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(n_full, [16, 64], [0, 448])
                        m_cos_il: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos_il, [1, 64], [m_t, 0])
                        m_sin_signed: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin_signed, [1, 64], [m_t, 0])
                        m_swapped: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(m_rope, pl.tensor.slice(rope_swap_idx, [16, 64], [0, 0]), dim=-1)
                        m_rot: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(m_rope, m_cos_il), pl.tensor.col_expand_mul(m_swapped, m_sin_signed))
                        n_rope_bf16: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(m_rot, target_type=pl.BF16, mode='rint')
                        n_full_bf16: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.concat(pl.tensor.slice(n_bf16, [16, 448], [0, 0]), n_rope_bf16)
                        for n_hi in pl.unroll(16):
                            n_head: pl.Scalar[pl.INDEX] = m_h0 + n_hi
                            source_row: pl.Scalar[pl.INDEX] = n_head // 8 * 256 + m_t
                            source_col: pl.Scalar[pl.INDEX] = n_head % 8 * 512
                            attention_grouped: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped, pl.tensor.slice(n_full_bf16, [1, 512], [n_hi, 0]), [source_row, source_col])
                    for group_slot in pl.unroll(2):
                        source_row: pl.Scalar[pl.INDEX] = (global_group0 + group_slot) * 256 + m_t0
                        target_row: pl.Scalar[pl.INDEX] = (local_group0 + group_slot) * 512 + pl.cast(tp_rank, pl.INDEX) * 256 + m_t0
                        pld.tensor.put(attention_window, pl.cast(group_base, pl.INDEX) + destination_rank, attention_grouped, [target_row, 0], [source_row, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                for peer_tp in pl.range(2):
                    if peer_tp != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(attention_signal, pl.cast(group_base, pl.INDEX) + peer_tp, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
            _tuple_tmp_4: pl.Tuple[pl.Tensor[[2048, 4096], pl.BF16], pld.DistributedTensor[[2, 1], pl.INT32]] = self.o_group_a2a(attention_local_flat, attention_window, attention_signal, group_base, tp_rank, 256, publish_tid, 48)
            attention_local_flat: pl.Tensor[[2048, 4096], pl.BF16] = _tuple_tmp_4[0]
            attention_signal: pld.DistributedTensor[[2, 1], pl.INT32] = _tuple_tmp_4[1]
            attention_local_groups: pl.Tensor[[4, 512, 4096], pl.BF16] = pl.tensor.reshape(attention_local_flat, [4, 512, 4096])
            _tuple_tmp_5: pl.Tuple[pl.Tensor[[t_dim, 4096], pl.BF16], pld.DistributedTensor[[2, 1], pl.INT32]] = self.o_proj_reduce_scatter(attention_local_groups, wo_a, wo_b, wo_b_scale, 256, attn_out, o_window, o_signal, group_base, tp_rank)
            _o_reduced: pl.Tensor[[t_dim, 4096], pl.BF16] = _tuple_tmp_5[0]
            o_signal: pld.DistributedTensor[[2, 1], pl.INT32] = _tuple_tmp_5[1]
        with pl.scope():
            self.hc_post(attn_out, x_hc, post_t, comb_t, x_out)
        return x_out
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def decode_csa_test(self, x_hc: pl.Tensor[[T_DYN, 4, 4096], pl.FP32], hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[24], pl.FP32], attn_norm_w: pl.Tensor[[4096], pl.BF16], wq_a: pl.Tensor[[4096, 1024], pl.BF16], wq_b: pl.Tensor[[1024, 32768], pl.INT8], wq_b_scale: pl.Tensor[[32768], pl.FP32], wkv: pl.Tensor[[4096, 512], pl.BF16], gamma_cq: pl.Tensor[[1024], pl.BF16], gamma_ckv: pl.Tensor[[512], pl.BF16], freqs_cos_local: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_sin_local: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_cos: pl.Tensor[[KV_T_DYN, 64], pl.BF16], freqs_sin: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_freqs_cos: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_freqs_sin: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], cmp_ape: pl.Tensor[[4, 1024], pl.FP32], cmp_norm_w: pl.Tensor[[512], pl.BF16], compress_state: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32]], compress_state_block_table: pl.Tensor[[KV_B_DYN, 4], pl.INT32], idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], weights_proj: pl.Tensor[[4096, 64], pl.BF16], hadamard_idx: pl.Tensor[[128, 128], pl.BF16], inner_wkv: pl.Tensor[[256, 4096], pl.BF16], inner_wgate: pl.Tensor[[256, 4096], pl.BF16], inner_ape: pl.Tensor[[4, 256], pl.FP32], inner_norm_w: pl.Tensor[[128], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[KV_B_DYN, 4], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], window_swa_indices: pl.Tensor[[T_DYN, 128], pl.INT32], window_swa_lens: pl.Tensor[[T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], idx_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], position_ids_local: pl.Tensor[[T_DYN], pl.INT32], position_ids: pl.Tensor[[KV_T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], attn_sink: pl.Tensor[[64], pl.FP32], wo_a: pl.Tensor[[4, 1024, 4096], pl.BF16], wo_b: pl.Tensor[[4096, 4096], pl.INT8], wo_b_scale: pl.Tensor[[4096], pl.FP32], x_out: pl.Out[pl.Tensor[[T_DYN, 4, 4096], pl.FP32]], gather_window: pld.DistributedTensor[[512, 4096], pl.BF16], gather_signal: pld.DistributedTensor[[2, 1], pl.INT32], attention_window: pld.DistributedTensor[[2048, 4096], pl.BF16], attention_signal: pld.DistributedTensor[[2, 1], pl.INT32], o_window: pld.DistributedTensor[[512, 4096], pl.BF16], o_signal: pld.DistributedTensor[[2, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32]):
        # Compile one rank of the complete tensor-parallel CSA layer.
        return self.decode_csa(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, freqs_cos_local, freqs_sin_local, freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx, inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_compress_state, inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, ori_slot_mapping, window_swa_indices, window_swa_lens, cmp_slot_mapping, idx_slot_mapping, state_slot_mapping, inner_state_slot_mapping, position_ids_local, position_ids, kv_seq_lens, attn_sink, wo_a, wo_b, wo_b_scale, x_out, gather_window, gather_signal, attention_window, attention_signal, o_window, o_signal, group_base, tp_rank, 256)
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def l3_decode_csa(self, x_hc: pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32], hc_attn_fn: pl.Tensor[[2, 24, 16384], pl.FP32], hc_attn_scale: pl.Tensor[[2, 3], pl.FP32], hc_attn_base: pl.Tensor[[2, 24], pl.FP32], attn_norm_w: pl.Tensor[[2, 4096], pl.BF16], wq_a: pl.Tensor[[2, 4096, 1024], pl.BF16], wq_b: pl.Tensor[[2, 1024, 32768], pl.INT8], wq_b_scale: pl.Tensor[[2, 32768], pl.FP32], wkv: pl.Tensor[[2, 4096, 512], pl.BF16], gamma_cq: pl.Tensor[[2, 1024], pl.BF16], gamma_ckv: pl.Tensor[[2, 512], pl.BF16], freqs_cos_local: pl.Tensor[[2, T_DYN, 64], pl.BF16], freqs_cos: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16], freqs_sin_local: pl.Tensor[[2, T_DYN, 64], pl.BF16], freqs_sin: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16], cmp_freqs_cos: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16], cmp_freqs_sin: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16], cmp_wkv: pl.Tensor[[2, 1024, 4096], pl.BF16], cmp_wgate: pl.Tensor[[2, 1024, 4096], pl.BF16], cmp_ape: pl.Tensor[[2, 4, 1024], pl.FP32], cmp_norm_w: pl.Tensor[[2, 512], pl.BF16], compress_state: pl.InOut[pl.Tensor[[2, MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32]], compress_state_block_table: pl.Tensor[[2, KV_B_DYN, 4], pl.INT32], idx_wq_b: pl.Tensor[[2, 1024, 8192], pl.INT8], idx_wq_b_scale: pl.Tensor[[2, 8192], pl.FP32], weights_proj: pl.Tensor[[2, 4096, 64], pl.BF16], hadamard_idx: pl.Tensor[[2, 128, 128], pl.BF16], inner_wkv: pl.Tensor[[2, 256, 4096], pl.BF16], inner_wgate: pl.Tensor[[2, 256, 4096], pl.BF16], inner_ape: pl.Tensor[[2, 4, 256], pl.FP32], inner_norm_w: pl.Tensor[[2, 128], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor[[2, INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[2, KV_B_DYN, 4], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[2, ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[2, CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_block_table: pl.Tensor[[2, B_DYN, 8192], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[2, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[2, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[2, B_DYN, 8192], pl.INT32], ori_slot_mapping: pl.Tensor[[2, KV_T_DYN], pl.INT64], window_swa_indices: pl.Tensor[[2, T_DYN, 128], pl.INT32], window_swa_lens: pl.Tensor[[2, T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[2, KV_T_DYN], pl.INT64], idx_slot_mapping: pl.Tensor[[2, KV_T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[2, KV_T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[2, KV_T_DYN], pl.INT64], position_ids_local: pl.Tensor[[2, T_DYN], pl.INT32], position_ids: pl.Tensor[[2, KV_T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[2, B_DYN], pl.INT32], attn_sink: pl.Tensor[[2, 64], pl.FP32], wo_a: pl.Tensor[[2, 4, 1024, 4096], pl.BF16], wo_b: pl.Tensor[[2, 4096, 4096], pl.INT8], wo_b_scale: pl.Tensor[[2, 4096], pl.FP32], x_out: pl.Out[pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32]], local_t: pl.Scalar[pl.INT32]) -> pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32]:
        # Launch the complete CSA layer on one physical TP group.
        gather_window_buf: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(4194304, pl.INT64))
        gather_signal_buf: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        attention_window_buf: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(16777216, pl.INT64))
        attention_signal_buf: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        o_window_buf: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(4194304, pl.INT64))
        o_signal_buf: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        for rank in pl.range(pld.system.world_size()):
            gather_window: pld.DistributedTensor[[512, 4096], pl.BF16] = pld.tensor.window(gather_window_buf, [512, 4096], dtype=pl.BF16)
            gather_signal: pld.DistributedTensor[[2, 1], pl.INT32] = pld.tensor.window(gather_signal_buf, [2, 1], dtype=pl.INT32)
            attention_window: pld.DistributedTensor[[2048, 4096], pl.BF16] = pld.tensor.window(attention_window_buf, [2048, 4096], dtype=pl.BF16)
            attention_signal: pld.DistributedTensor[[2, 1], pl.INT32] = pld.tensor.window(attention_signal_buf, [2, 1], dtype=pl.INT32)
            o_window: pld.DistributedTensor[[512, 4096], pl.BF16] = pld.tensor.window(o_window_buf, [512, 4096], dtype=pl.BF16)
            o_signal: pld.DistributedTensor[[2, 1], pl.INT32] = pld.tensor.window(o_signal_buf, [2, 1], dtype=pl.INT32)
            self.decode_csa_test(pl.tensor.slice(x_hc, [1, T_DYN, 4, 4096], [rank, 0, 0, 0], [], [0]), pl.tensor.slice(hc_attn_fn, [1, 24, 16384], [rank, 0, 0], [], [0]), pl.tensor.slice(hc_attn_scale, [1, 3], [rank, 0], [], [0]), pl.tensor.slice(hc_attn_base, [1, 24], [rank, 0], [], [0]), pl.tensor.slice(attn_norm_w, [1, 4096], [rank, 0], [], [0]), pl.tensor.slice(wq_a, [1, 4096, 1024], [rank, 0, 0], [], [0]), pl.tensor.slice(wq_b, [1, 1024, 32768], [rank, 0, 0], [], [0]), pl.tensor.slice(wq_b_scale, [1, 32768], [rank, 0], [], [0]), pl.tensor.slice(wkv, [1, 4096, 512], [rank, 0, 0], [], [0]), pl.tensor.slice(gamma_cq, [1, 1024], [rank, 0], [], [0]), pl.tensor.slice(gamma_ckv, [1, 512], [rank, 0], [], [0]), pl.tensor.slice(freqs_cos_local, [1, T_DYN, 64], [rank, 0, 0], [], [0]), pl.tensor.slice(freqs_sin_local, [1, T_DYN, 64], [rank, 0, 0], [], [0]), pl.tensor.slice(freqs_cos, [1, KV_T_DYN, 64], [rank, 0, 0], [], [0]), pl.tensor.slice(freqs_sin, [1, KV_T_DYN, 64], [rank, 0, 0], [], [0]), pl.tensor.slice(cmp_freqs_cos, [1, KV_T_DYN, 64], [rank, 0, 0], [], [0]), pl.tensor.slice(cmp_freqs_sin, [1, KV_T_DYN, 64], [rank, 0, 0], [], [0]), pl.tensor.slice(cmp_wkv, [1, 1024, 4096], [rank, 0, 0], [], [0]), pl.tensor.slice(cmp_wgate, [1, 1024, 4096], [rank, 0, 0], [], [0]), pl.tensor.slice(cmp_ape, [1, 4, 1024], [rank, 0, 0], [], [0]), pl.tensor.slice(cmp_norm_w, [1, 512], [rank, 0], [], [0]), pl.tensor.slice(compress_state, [1, MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], [rank, 0, 0, 0], [], [0]), pl.tensor.slice(compress_state_block_table, [1, KV_B_DYN, 4], [rank, 0, 0], [], [0]), pl.tensor.slice(idx_wq_b, [1, 1024, 8192], [rank, 0, 0], [], [0]), pl.tensor.slice(idx_wq_b_scale, [1, 8192], [rank, 0], [], [0]), pl.tensor.slice(weights_proj, [1, 4096, 64], [rank, 0, 0], [], [0]), pl.tensor.slice(hadamard_idx, [1, 128, 128], [rank, 0, 0], [], [0]), pl.tensor.slice(inner_wkv, [1, 256, 4096], [rank, 0, 0], [], [0]), pl.tensor.slice(inner_wgate, [1, 256, 4096], [rank, 0, 0], [], [0]), pl.tensor.slice(inner_ape, [1, 4, 256], [rank, 0, 0], [], [0]), pl.tensor.slice(inner_norm_w, [1, 128], [rank, 0], [], [0]), pl.tensor.slice(inner_compress_state, [1, INNER_STATE_BLOCK_NUM_DYN, 2, 512], [rank, 0, 0, 0], [], [0]), pl.tensor.slice(inner_compress_state_block_table, [1, KV_B_DYN, 4], [rank, 0, 0], [], [0]), pl.tensor.slice(kv_cache, [1, ORI_BLOCK_NUM_DYN, 32, 1, 512], [rank, 0, 0, 0, 0], [], [0]), pl.tensor.slice(cmp_kv, [1, CMP_BLOCK_NUM_DYN, 32, 1, 512], [rank, 0, 0, 0, 0], [], [0]), pl.tensor.slice(cmp_block_table, [1, B_DYN, 8192], [rank, 0, 0], [], [0]), pl.tensor.slice(idx_kv_cache, [1, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], [rank, 0, 0, 0, 0], [], [0]), pl.tensor.slice(idx_kv_scale, [1, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], [rank, 0, 0, 0, 0], [], [0]), pl.tensor.slice(idx_block_table, [1, B_DYN, 8192], [rank, 0, 0], [], [0]), pl.tensor.slice(ori_slot_mapping, [1, KV_T_DYN], [rank, 0], [], [0]), pl.tensor.slice(window_swa_indices, [1, T_DYN, 128], [rank, 0, 0], [], [0]), pl.tensor.slice(window_swa_lens, [1, T_DYN], [rank, 0], [], [0]), pl.tensor.slice(cmp_slot_mapping, [1, KV_T_DYN], [rank, 0], [], [0]), pl.tensor.slice(idx_slot_mapping, [1, KV_T_DYN], [rank, 0], [], [0]), pl.tensor.slice(state_slot_mapping, [1, KV_T_DYN], [rank, 0], [], [0]), pl.tensor.slice(inner_state_slot_mapping, [1, KV_T_DYN], [rank, 0], [], [0]), pl.tensor.slice(position_ids_local, [1, T_DYN], [rank, 0], [], [0]), pl.tensor.slice(position_ids, [1, KV_T_DYN], [rank, 0], [], [0]), pl.tensor.slice(kv_seq_lens, [1, B_DYN], [rank, 0], [], [0]), pl.tensor.slice(attn_sink, [1, 64], [rank, 0], [], [0]), pl.tensor.slice(wo_a, [1, 4, 1024, 4096], [rank, 0, 0, 0], [], [0]), pl.tensor.slice(wo_b, [1, 4096, 4096], [rank, 0, 0], [], [0]), pl.tensor.slice(wo_b_scale, [1, 4096], [rank, 0], [], [0]), pl.tensor.slice(x_out, [1, T_DYN, 4, 4096], [rank, 0, 0, 0], [], [0]), gather_window, gather_signal, attention_window, attention_signal, o_window, o_signal, 0, rank, 256, device=rank)
        return x_out
