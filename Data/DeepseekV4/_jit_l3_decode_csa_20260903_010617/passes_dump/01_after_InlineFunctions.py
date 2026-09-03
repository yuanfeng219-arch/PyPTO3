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
bs = pl.dynamic("bs")
bs_1 = pl.dynamic("bs_1")
bs_heads = pl.dynamic("bs_heads")
cmp_block_num = pl.dynamic("cmp_block_num")
cmp_block_num_1 = pl.dynamic("cmp_block_num_1")
compress_state_block_num = pl.dynamic("compress_state_block_num")
compress_state_block_num_1 = pl.dynamic("compress_state_block_num_1")
dq_rows = pl.dynamic("dq_rows")
idx_block_num = pl.dynamic("idx_block_num")
idx_block_num_1 = pl.dynamic("idx_block_num_1")
idx_table_len = pl.dynamic("idx_table_len")
kv_b_dim = pl.dynamic("kv_b_dim")
kv_dim = pl.dynamic("kv_dim")
kv_rows = pl.dynamic("kv_rows")
ori_block_num = pl.dynamic("ori_block_num")
ori_block_num_1 = pl.dynamic("ori_block_num_1")
pa_rows = pl.dynamic("pa_rows")
qproj_t_matmul = pl.dynamic("qproj_t_matmul")
qproj_tail_rows = pl.dynamic("qproj_tail_rows")
qr_rows = pl.dynamic("qr_rows")
qr_rows_1 = pl.dynamic("qr_rows_1")
qr_t_matmul = pl.dynamic("qr_t_matmul")
qrp_valid_rows = pl.dynamic("qrp_valid_rows")
rms_blk_rows = pl.dynamic("rms_blk_rows")
rms_blk_rows_1 = pl.dynamic("rms_blk_rows_1")
t_blk = pl.dynamic("t_blk")
t_dim = pl.dynamic("t_dim")
t_dim_1 = pl.dynamic("t_dim_1")
t_dim_2 = pl.dynamic("t_dim_2")
t_dim_3 = pl.dynamic("t_dim_3")
t_dim_4 = pl.dynamic("t_dim_4")
t_dim_5 = pl.dynamic("t_dim_5")
t_dim_6 = pl.dynamic("t_dim_6")
t_heads = pl.dynamic("t_heads")
t_linear = pl.dynamic("t_linear")
t_matmul = pl.dynamic("t_matmul")
t_rows = pl.dynamic("t_rows")
valid_count = pl.dynamic("valid_count")
valid_rows = pl.dynamic("valid_rows")
valid_rows_1 = pl.dynamic("valid_rows_1")
valid_rows_2 = pl.dynamic("valid_rows_2")
valid_rows_3 = pl.dynamic("valid_rows_3")
valid_tail_rows = pl.dynamic("valid_tail_rows")
w_rows = pl.dynamic("w_rows")
x_rows = pl.dynamic("x_rows")
x_rows_1 = pl.dynamic("x_rows_1")

@pl.program
class _jit_l3_decode_csa:
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
                    # Sort one scored 8K leaf and store its exact Top-512 pair row.
                    logical_begin_i32_inline126: pl.Scalar[pl.INT32] = pl.cast(logical_begin, pl.INT32)
                    leaf_indices_inline125: pl.Tile[[1, 8192], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False), logical_begin_i32_inline126)
                    leaf_scores_raw_inline124: pl.Tile[[1, 8192], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [query, logical_begin], [1, 8192], [1, valid_count])
                    leaf_scores_inline121: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw_inline124, pad_value=pl.PadValue.min)
                    leaf_scores_v1_inline118: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores_inline121, pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38))
                    pairs_inline120: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1_inline118, pl.tile.reinterpret_view(leaf_indices_inline125, dtype=pl.UINT32))
                    pairs_v1_inline119: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_inline120, pl.const(64, pl.INT32))
                    pairs_v2_inline123: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1_inline119, pl.const(256, pl.INT32))
                    pairs_v3_inline117: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2_inline123, pl.const(1024, pl.INT32))
                    pairs_v4_inline122: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3_inline117, pl.const(4096, pl.INT32))
                    pl.tile.store(pl.tile.slice(pairs_v4_inline122, [1, 1024], [0, 0]), [group_root_slot, 0], pair_arena)
                else:
                    scratch_base: pl.Scalar[pl.INDEX] = 4096 + worker * 2
                    for group_leaf in pl.unroll(2):
                        leaf: pl.Scalar[pl.INDEX] = leaf_begin + group_leaf
                        logical_begin: pl.Scalar[pl.INDEX] = leaf * 8192
                        valid_count: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count - logical_begin)
                        # Sort one scored 8K leaf and store its exact Top-512 pair row.
                        logical_begin_i32_inline136: pl.Scalar[pl.INT32] = pl.cast(logical_begin, pl.INT32)
                        leaf_indices_inline135: pl.Tile[[1, 8192], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False), logical_begin_i32_inline136)
                        leaf_scores_raw_inline134: pl.Tile[[1, 8192], pl.FP32, pl.TileView(valid_shape=[1, valid_count])] = pl.tile.load(score_arena, [query, logical_begin], [1, 8192], [1, valid_count])
                        leaf_scores_inline131: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw_inline134, pad_value=pl.PadValue.min)
                        leaf_scores_v1_inline128: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores_inline131, pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38))
                        pairs_inline130: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1_inline128, pl.tile.reinterpret_view(leaf_indices_inline135, dtype=pl.UINT32))
                        pairs_v1_inline129: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_inline130, pl.const(64, pl.INT32))
                        pairs_v2_inline133: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1_inline129, pl.const(256, pl.INT32))
                        pairs_v3_inline127: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2_inline133, pl.const(1024, pl.INT32))
                        pairs_v4_inline132: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3_inline127, pl.const(4096, pl.INT32))
                        pl.tile.store(pl.tile.slice(pairs_v4_inline132, [1, 1024], [0, 0]), [scratch_base + group_leaf, 0], pair_arena)
                    # Merge two arena rows and store their exact Top-512 pair row.
                    left_inline141: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [scratch_base, 0], [1, 1024], [1, 1024])
                    right_inline140: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [scratch_base + 1, 0], [1, 1024], [1, 1024])
                    merge_tmp_inline138: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                    merged_all_inline139: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline141, right_inline140, merge_tmp_inline138, exhausted=False)
                    merged_inline137: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline139, [1, 1024], [0, 0])
                    pl.tile.store(merged_inline137, [group_root_slot, 0], pair_arena)
            global_group_base: pl.Scalar[pl.INDEX] = global_group_base + group_count
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
                # Reduce one exact-Top-K forest level, forwarding an odd final node.
                output_count_inline10: pl.Scalar[pl.INDEX] = (group_count + 1) // 2
                for output_inline9 in pl.range(output_count_inline10):
                    left_slot_inline8: pl.Scalar[pl.INDEX] = arena_base + 0 + 2 * output_inline9
                    right_slot_inline6: pl.Scalar[pl.INDEX] = left_slot_inline8 + 1
                    output_slot_inline7: pl.Scalar[pl.INDEX] = arena_base + 0 + output_inline9
                    if right_slot_inline6 < arena_base + 0 + group_count:
                        # Merge two arena rows and store their exact Top-512 pair row.
                        left_inline1329: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot_inline8, 0], [1, 1024], [1, 1024])
                        right_inline1328: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [right_slot_inline6, 0], [1, 1024], [1, 1024])
                        merge_tmp_inline1326: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                        merged_all_inline1327: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1329, right_inline1328, merge_tmp_inline1326, exhausted=False)
                        merged_inline1325: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline1327, [1, 1024], [0, 0])
                        pl.tile.store(merged_inline1325, [output_slot_inline7, 0], pair_arena)
                    else:
                        forwarded_inline5: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot_inline8, 0], [1, 1024], [1, 1024])
                        pl.tile.store(forwarded_inline5, [output_slot_inline7, 0], pair_arena)
                if level1_count > 1:
                    level2_count: pl.Scalar[pl.INDEX] = (level1_count + 1) // 2
                    # Reduce one exact-Top-K forest level, forwarding an odd final node.
                    output_count_inline16: pl.Scalar[pl.INDEX] = (level1_count + 1) // 2
                    for output_inline15 in pl.range(output_count_inline16):
                        left_slot_inline14: pl.Scalar[pl.INDEX] = arena_base + 0 + 2 * output_inline15
                        right_slot_inline12: pl.Scalar[pl.INDEX] = left_slot_inline14 + 1
                        output_slot_inline13: pl.Scalar[pl.INDEX] = arena_base + 0 + output_inline15
                        if right_slot_inline12 < arena_base + 0 + level1_count:
                            # Merge two arena rows and store their exact Top-512 pair row.
                            left_inline1334: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot_inline14, 0], [1, 1024], [1, 1024])
                            right_inline1333: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [right_slot_inline12, 0], [1, 1024], [1, 1024])
                            merge_tmp_inline1331: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                            merged_all_inline1332: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1334, right_inline1333, merge_tmp_inline1331, exhausted=False)
                            merged_inline1330: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline1332, [1, 1024], [0, 0])
                            pl.tile.store(merged_inline1330, [output_slot_inline13, 0], pair_arena)
                        else:
                            forwarded_inline11: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot_inline14, 0], [1, 1024], [1, 1024])
                            pl.tile.store(forwarded_inline11, [output_slot_inline13, 0], pair_arena)
                    if level2_count > 1:
                        level3_count: pl.Scalar[pl.INDEX] = (level2_count + 1) // 2
                        # Reduce one exact-Top-K forest level, forwarding an odd final node.
                        output_count_inline22: pl.Scalar[pl.INDEX] = (level2_count + 1) // 2
                        for output_inline21 in pl.range(output_count_inline22):
                            left_slot_inline20: pl.Scalar[pl.INDEX] = arena_base + 0 + 2 * output_inline21
                            right_slot_inline18: pl.Scalar[pl.INDEX] = left_slot_inline20 + 1
                            output_slot_inline19: pl.Scalar[pl.INDEX] = arena_base + 0 + output_inline21
                            if right_slot_inline18 < arena_base + 0 + level2_count:
                                # Merge two arena rows and store their exact Top-512 pair row.
                                left_inline1339: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot_inline20, 0], [1, 1024], [1, 1024])
                                right_inline1338: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [right_slot_inline18, 0], [1, 1024], [1, 1024])
                                merge_tmp_inline1336: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                                merged_all_inline1337: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1339, right_inline1338, merge_tmp_inline1336, exhausted=False)
                                merged_inline1335: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline1337, [1, 1024], [0, 0])
                                pl.tile.store(merged_inline1335, [output_slot_inline19, 0], pair_arena)
                            else:
                                forwarded_inline17: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot_inline20, 0], [1, 1024], [1, 1024])
                                pl.tile.store(forwarded_inline17, [output_slot_inline19, 0], pair_arena)
                        if level3_count > 1:
                            # Reduce one exact-Top-K forest level, forwarding an odd final node.
                            output_count_inline28: pl.Scalar[pl.INDEX] = (level3_count + 1) // 2
                            for output_inline27 in pl.range(output_count_inline28):
                                left_slot_inline26: pl.Scalar[pl.INDEX] = arena_base + 0 + 2 * output_inline27
                                right_slot_inline24: pl.Scalar[pl.INDEX] = left_slot_inline26 + 1
                                output_slot_inline25: pl.Scalar[pl.INDEX] = arena_base + 0 + output_inline27
                                if right_slot_inline24 < arena_base + 0 + level3_count:
                                    # Merge two arena rows and store their exact Top-512 pair row.
                                    left_inline1344: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot_inline26, 0], [1, 1024], [1, 1024])
                                    right_inline1343: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [right_slot_inline24, 0], [1, 1024], [1, 1024])
                                    merge_tmp_inline1341: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                                    merged_all_inline1342: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1344, right_inline1343, merge_tmp_inline1341, exhausted=False)
                                    merged_inline1340: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline1342, [1, 1024], [0, 0])
                                    pl.tile.store(merged_inline1340, [output_slot_inline25, 0], pair_arena)
                                else:
                                    forwarded_inline23: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [left_slot_inline26, 0], [1, 1024], [1, 1024])
                                    pl.tile.store(forwarded_inline23, [output_slot_inline25, 0], pair_arena)
            root_slot: pl.Scalar[pl.INDEX] = arena_base
            root_pairs: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena, [root_slot, 0], [1, 1024], [1, 1024])
            pl.tile.store(pl.tile.gather_mask(root_pairs, mask_pattern=1, output_dtype=pl.FP32), [query, 0], topk_scores)
            root_indices: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.gather_mask(root_pairs, mask_pattern=2, output_dtype=pl.INT32)
            output_indices: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
            valid_topk: pl.Scalar[pl.INDEX] = pl.min(visible_count, 512)
            for lane in pl.range(valid_topk):
                pl.tile.write(output_indices, [0, lane], pl.tile.read(root_indices, [0, lane]))
            pl.tile.store(output_indices, [query, 0], topk_indices)
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def decode_csa_test(self, x_hc: pl.Tensor[[T_DYN, 4, 4096], pl.FP32], hc_attn_fn: pl.Tensor[[24, 16384], pl.FP32], hc_attn_scale: pl.Tensor[[3], pl.FP32], hc_attn_base: pl.Tensor[[24], pl.FP32], attn_norm_w: pl.Tensor[[4096], pl.BF16], wq_a: pl.Tensor[[4096, 1024], pl.BF16], wq_b: pl.Tensor[[1024, 32768], pl.INT8], wq_b_scale: pl.Tensor[[32768], pl.FP32], wkv: pl.Tensor[[4096, 512], pl.BF16], gamma_cq: pl.Tensor[[1024], pl.BF16], gamma_ckv: pl.Tensor[[512], pl.BF16], freqs_cos_local: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_sin_local: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_cos: pl.Tensor[[KV_T_DYN, 64], pl.BF16], freqs_sin: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_freqs_cos: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_freqs_sin: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_wkv: pl.Tensor[[1024, 4096], pl.BF16], cmp_wgate: pl.Tensor[[1024, 4096], pl.BF16], cmp_ape: pl.Tensor[[4, 1024], pl.FP32], cmp_norm_w: pl.Tensor[[512], pl.BF16], compress_state: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32]], compress_state_block_table: pl.Tensor[[KV_B_DYN, 4], pl.INT32], idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8], idx_wq_b_scale: pl.Tensor[[8192], pl.FP32], weights_proj: pl.Tensor[[4096, 64], pl.BF16], hadamard_idx: pl.Tensor[[128, 128], pl.BF16], inner_wkv: pl.Tensor[[256, 4096], pl.BF16], inner_wgate: pl.Tensor[[256, 4096], pl.BF16], inner_ape: pl.Tensor[[4, 256], pl.FP32], inner_norm_w: pl.Tensor[[128], pl.BF16], inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32]], inner_compress_state_block_table: pl.Tensor[[KV_B_DYN, 4], pl.INT32], kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8]], idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32]], idx_block_table: pl.Tensor[[B_DYN, 8192], pl.INT32], ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], window_swa_indices: pl.Tensor[[T_DYN, 128], pl.INT32], window_swa_lens: pl.Tensor[[T_DYN], pl.INT32], cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], idx_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], inner_state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64], position_ids_local: pl.Tensor[[T_DYN], pl.INT32], position_ids: pl.Tensor[[KV_T_DYN], pl.INT32], kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32], attn_sink: pl.Tensor[[64], pl.FP32], wo_a: pl.Tensor[[4, 1024, 4096], pl.BF16], wo_b: pl.Tensor[[4096, 4096], pl.INT8], wo_b_scale: pl.Tensor[[4096], pl.FP32], x_out: pl.Out[pl.Tensor[[T_DYN, 4, 4096], pl.FP32]], gather_window: pld.DistributedTensor[[512, 4096], pl.BF16], gather_signal: pld.DistributedTensor[[2, 1], pl.INT32], attention_window: pld.DistributedTensor[[2048, 4096], pl.BF16], attention_signal: pld.DistributedTensor[[2, 1], pl.INT32], o_window: pld.DistributedTensor[[512, 4096], pl.BF16], o_signal: pld.DistributedTensor[[2, 1], pl.INT32], group_base: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32], local_t: pl.Scalar[pl.INT32]):
        # Run one rank of the context-parallel CSA layer.
        t_dim_inline1251: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc, 0)
        kv_dim_inline1261: pl.Scalar[pl.INDEX] = pl.tensor.dim(ori_slot_mapping, 0)
        kv_b_dim_inline1264: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table, 0)
        q_inline1246: pl.Tensor[[t_dim, 64, 512], pl.BF16] = pl.tensor.create([t_dim_inline1251, 64, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        idx_topk_scores_inline1271: pl.Tensor[[t_dim, 512], pl.FP32] = pl.tensor.create([t_dim_inline1251, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        idx_topk_inline1280: pl.Tensor[[t_dim, 512], pl.INT32] = pl.tensor.create([t_dim_inline1251, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        post_t_inline1277: pl.Tensor[[t_dim, 4], pl.FP32] = pl.tensor.create([t_dim_inline1251, 4], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        comb_t_inline1267: pl.Tensor[[t_dim, 16], pl.FP32] = pl.tensor.create([t_dim_inline1251, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        x_mixed_inline1253: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.create([t_dim_inline1251, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            # One pl.spmd task per work-type, ordered by their GM read/write dependencies.
            # 
            #         rms -> linear -> linear_reduce -> split_pre_post / comb_sinkhorn / mix_x. Cross-scope
            #         buffers are sized to t_linear, the token count padded up to whole 16-row cube tiles.
            #         
            t_dim_inline1568: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc, 0)
            token_tiles_inline1492: pl.Scalar[pl.INDEX] = (t_dim_inline1568 + 8 - 1) // 8
            t_linear_inline1486: pl.Scalar[pl.INDEX] = (t_dim_inline1568 + 16 - 1) // 16 * 16
            x_flat_inline1497: pl.Tensor[[t_dim_1, 16384], pl.FP32] = pl.tensor.reshape(x_hc, [t_dim_inline1568, 16384])
            scale0_inline1499: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale, [0])
            scale1_inline1530: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale, [1])
            scale2_inline1480: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale, [2])
            hc_base_2d_inline1467: pl.Tensor[[1, 24], pl.FP32] = pl.tensor.reshape(hc_attn_base, [1, 24])
            inv_rms_inline1463: pl.Tensor[[t_linear, 1], pl.FP32] = pl.tensor.create([t_linear_inline1486, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for t_inline1518 in pl.spmd(token_tiles_inline1492, name_hint="hc_pre_rms_spmd", allow_early_resolve=True):
                t0_inline1476: pl.Scalar[pl.INDEX] = t_inline1518 * 8
                valid_rows_inline1507: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1568 - t0_inline1476)
                sq_sum_inline1490: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                for kb_inline1493 in pl.pipeline(32, stage=4):
                    k0_inline1501: pl.Scalar[pl.INDEX] = kb_inline1493 * 512
                    if valid_rows_inline1507 == 8:
                        x_chunk_full_inline1502: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.slice(x_flat_inline1497, [8, 512], [t0_inline1476, k0_inline1501])
                        x_sq_full_inline1510: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(x_chunk_full_inline1502, x_chunk_full_inline1502)
                        x_sq_row_full_inline1511: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(x_sq_full_inline1510), [1, 8])
                        sq_sum_inline1490: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(sq_sum_inline1490, x_sq_row_full_inline1511)
                    else:
                        x_chunk_tail_inline1513: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497, [8, 512], [t0_inline1476, k0_inline1501], [valid_rows_inline1507, 512])
                        x_sq_tail_inline1503: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.mul(x_chunk_tail_inline1513, x_chunk_tail_inline1513)
                        x_sq_row_tail_inline1537: pl.Tensor[[1, 8], pl.FP32, pl.TensorView(valid_shape=[1, valid_rows], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.reshape(pl.tensor.row_sum(x_sq_tail_inline1503), [1, 8])
                        sq_sum_inline1490: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(sq_sum_inline1490, x_sq_row_tail_inline1537)
                sq_mean_inline1514: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.adds(pl.tensor.muls(sq_sum_inline1490, 6.103515625e-05), 9.9999999999999995e-07)
                inv_inline1481: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.rsqrt(sq_mean_inline1514, high_precision=True), [8, 1])
                inv_rms_inline1463: pl.Tensor[[t_linear, 1], pl.FP32] = pl.tensor.assemble(inv_rms_inline1463, inv_inline1481, [t0_inline1476, 0])
            mixes_partials_inline1475: pl.Tensor[[pl.const(4, pl.INDEX) * t_linear, 32], pl.FP32] = pl.tensor.create([4 * t_linear_inline1486, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for task_inline1516 in pl.spmd(t_linear_inline1486 // 16 * 4, name_hint="hc_pre_linear_spmd", allow_early_resolve=True):
                t0_inline1476: pl.Scalar[pl.INDEX] = task_inline1516 // 4 * 16
                linear_split_inline1509: pl.Scalar[pl.INDEX] = task_inline1516 % 4
                k_base_inline1494: pl.Scalar[pl.INDEX] = linear_split_inline1509 * 4096
                t_rows_inline1520: pl.Scalar[pl.INDEX] = pl.min(16, t_dim_inline1568 - t0_inline1476)
                acc_inline1524: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for kb_inline1488 in pl.pipeline(16, stage=2):
                    k0_inline1501: pl.Scalar[pl.INDEX] = k_base_inline1494 + kb_inline1488 * 256
                    x_linear_chunk_inline1459: pl.Tensor[[16, 256], pl.FP32, pl.TensorView(valid_shape=[t_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497, [16, 256], [t0_inline1476, k0_inline1501], [t_rows_inline1520, 256])
                    w_chunk_inline1519: pl.Tensor[[32, 256], pl.FP32, pl.TensorView(valid_shape=[24, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(hc_attn_fn, [32, 256], [0, k0_inline1501], [24, 256])
                    if kb_inline1488 == 0:
                        acc_inline1524: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_linear_chunk_inline1459, w_chunk_inline1519, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                    else:
                        acc_inline1524: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(acc_inline1524, x_linear_chunk_inline1459, w_chunk_inline1519, a_trans=False, b_trans=True)
                partial_row0_inline1471: pl.Scalar[pl.INDEX] = linear_split_inline1509 * t_linear_inline1486 + t0_inline1476
                mixes_partials_inline1475: pl.Tensor[[pl.const(4, pl.INDEX) * t_linear, 32], pl.FP32] = pl.tensor.assemble(mixes_partials_inline1475, acc_inline1524, [partial_row0_inline1471, 0])
            mixes_raw_inline1505: pl.Tensor[[t_linear, 32], pl.FP32] = pl.tensor.create([t_linear_inline1486, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for linear_block_inline1567 in pl.spmd(t_linear_inline1486 // 16, name_hint="hc_pre_linear_reduce_spmd", allow_early_resolve=True):
                linear_t0_inline1457: pl.Scalar[pl.INDEX] = linear_block_inline1567 * 16
                mixes_total_inline1570: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(mixes_partials_inline1475, [16, 32], [linear_t0_inline1457, 0])
                for linear_split_inline1456 in pl.range(1, 4):
                    partial_t0_inline1464: pl.Scalar[pl.INDEX] = linear_split_inline1456 * t_linear_inline1486 + linear_t0_inline1457
                    partial_tile_inline1561: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(mixes_partials_inline1475, [16, 32], [partial_t0_inline1464, 0])
                    mixes_total_inline1570: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.add(mixes_total_inline1570, partial_tile_inline1561)
                mixes_raw_inline1505: pl.Tensor[[t_linear, 32], pl.FP32] = pl.tensor.assemble(mixes_raw_inline1505, mixes_total_inline1570, [linear_t0_inline1457, 0])
            pre_val_store_inline1529: pl.Tensor[[t_linear, 8], pl.FP32] = pl.tensor.create([t_linear_inline1486, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            post_tail_store_inline1544: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for ob_inline1554 in pl.spmd(token_tiles_inline1492, name_hint="split_pre_post_spmd", allow_early_resolve=True):
                t0_inline1476: pl.Scalar[pl.INDEX] = ob_inline1554 * 8
                valid_rows_inline1507: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1568 - t0_inline1476)
                inv_col_inline1450: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(inv_rms_inline1463, [8, 1], [t0_inline1476, 0])
                pre_base_inline1470: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(hc_attn_base, [8], [0]), [1, 8])
                pre_scaled_inline1447: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(pl.tensor.row_expand_mul(pl.tensor.slice(mixes_raw_inline1505, [8, 8], [t0_inline1476, 0]), inv_col_inline1450), scale0_inline1499)
                pre_logits_inline1455: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(pre_scaled_inline1447, pl.tensor.col_expand(pre_scaled_inline1447, pre_base_inline1470))
                pre_sig_inline1452: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(pre_logits_inline1455)), 1.0))
                pre_val_inline1448: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.adds(pre_sig_inline1452, 9.9999999999999995e-07)
                pre_val_store_inline1529: pl.Tensor[[t_linear, 8], pl.FP32] = pl.tensor.assemble(pre_val_store_inline1529, pre_val_inline1448, [t0_inline1476, 0])
                post_base_inline1472: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(hc_attn_base, [8], [4]), [1, 8])
                post_scaled_inline1461: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(pl.tensor.row_expand_mul(pl.tensor.slice(mixes_raw_inline1505, [8, 8], [t0_inline1476, 4]), inv_col_inline1450), scale1_inline1530)
                post_logits_inline1506: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(post_scaled_inline1461, pl.tensor.col_expand(post_scaled_inline1461, post_base_inline1472))
                post_sig_inline1444: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(post_logits_inline1506)), 1.0))
                post_pad_inline1489: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(post_sig_inline1444, 2.0)
                if valid_rows_inline1507 == 8:
                    post_t_inline1277: pl.Tensor[[t_dim, 4], pl.FP32] = pl.tensor.assemble(post_t_inline1277, pl.tensor.slice(post_pad_inline1489, [8, 8], [0, 0], [8, 4]), [t0_inline1476, 0])
                else:
                    post_tail_store_inline1544: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.assemble(post_tail_store_inline1544, post_pad_inline1489, [0, 0])
                    post_tile_inline1483: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(post_tail_store_inline1544, [0, 0], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                    pl.tile.store(post_tile_inline1483, [t0_inline1476, 0], post_t_inline1277)
            comb_tail_store_inline1523: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.create([8, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for ob_inline1522 in pl.spmd(token_tiles_inline1492, name_hint="comb_sinkhorn_spmd", allow_early_resolve=True):
                t0_inline1476: pl.Scalar[pl.INDEX] = ob_inline1522 * 8
                valid_rows_inline1507: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1568 - t0_inline1476)
                inv_col_t_inline1560: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 1])] = pl.tile.load(inv_rms_inline1463, [t0_inline1476, 0], [8, 1], [valid_rows_inline1507, 1], target_memory=pl.Mem.Vec)
                comb_off_inline1572: pl.Scalar[pl.INDEX] = 8
                mix_g0_inline1525: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(mixes_raw_inline1505, [t0_inline1476, comb_off_inline1572 + 0], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                mix_g1_inline1528: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(mixes_raw_inline1505, [t0_inline1476, comb_off_inline1572 + 4], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                mix_g2_inline1531: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(mixes_raw_inline1505, [t0_inline1476, comb_off_inline1572 + 8], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                mix_g3_inline1552: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(mixes_raw_inline1505, [t0_inline1476, comb_off_inline1572 + 12], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                cb0_inline1458: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467, [0, comb_off_inline1572 + 0], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb1_inline1485: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467, [0, comb_off_inline1572 + 4], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb2_inline1532: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467, [0, comb_off_inline1572 + 8], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb3_inline1533: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467, [0, comb_off_inline1572 + 12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                row0_inline1534: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g0_inline1525, inv_col_t_inline1560), scale2_inline1480), pl.tile.col_expand(mix_g0_inline1525, cb0_inline1458))
                row1_inline1575: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g1_inline1528, inv_col_t_inline1560), scale2_inline1480), pl.tile.col_expand(mix_g1_inline1528, cb1_inline1485))
                row2_inline1468: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g2_inline1531, inv_col_t_inline1560), scale2_inline1480), pl.tile.col_expand(mix_g2_inline1531, cb2_inline1532))
                row3_inline1484: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g3_inline1552, inv_col_t_inline1560), scale2_inline1480), pl.tile.col_expand(mix_g3_inline1552, cb3_inline1533))
                row0_p_inline1478: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row0_inline1534, pad_value=pl.PadValue.min)
                row1_p_inline1495: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row1_inline1575, pad_value=pl.PadValue.min)
                row2_p_inline1535: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row2_inline1468, pad_value=pl.PadValue.min)
                row3_p_inline1445: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row3_inline1484, pad_value=pl.PadValue.min)
                row_max_tmp_inline1487: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                row_sum_tmp_inline1498: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                row0_max_inline1538: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row0_p_inline1478, row_max_tmp_inline1487)
                row1_max_inline1473: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row1_p_inline1495, row_max_tmp_inline1487)
                row2_max_inline1539: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row2_p_inline1535, row_max_tmp_inline1487)
                row3_max_inline1540: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row3_p_inline1445, row_max_tmp_inline1487)
                row0_exp_inline1527: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row0_p_inline1478, row0_max_inline1538))
                row1_exp_inline1543: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row1_p_inline1495, row1_max_inline1473))
                row2_exp_inline1545: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row2_p_inline1535, row2_max_inline1539))
                row3_exp_inline1547: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row3_p_inline1445, row3_max_inline1540))
                row0_sum_inline1512: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row0_exp_inline1527, row_sum_tmp_inline1498)
                row1_sum_inline1482: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row1_exp_inline1543, row_sum_tmp_inline1498)
                row2_sum_inline1546: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row2_exp_inline1545, row_sum_tmp_inline1498)
                row3_sum_inline1453: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row3_exp_inline1547, row_sum_tmp_inline1498)
                row0_soft_inline1548: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row0_exp_inline1527, row0_sum_inline1512), 9.9999999999999995e-07)
                row1_soft_inline1515: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row1_exp_inline1543, row1_sum_inline1482), 9.9999999999999995e-07)
                row2_soft_inline1477: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row2_exp_inline1545, row2_sum_inline1546), 9.9999999999999995e-07)
                row3_soft_inline1549: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row3_exp_inline1547, row3_sum_inline1453), 9.9999999999999995e-07)
                row0_valid_inline1508: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row0_soft_inline1548, 8, 4)
                row1_valid_inline1551: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row1_soft_inline1515, 8, 4)
                row2_valid_inline1517: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row2_soft_inline1477, 8, 4)
                row3_valid_inline1553: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row3_soft_inline1549, 8, 4)
                row0_eff_inline1555: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row0_valid_inline1508, pad_value=pl.PadValue.zero)
                row1_eff_inline1557: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row1_valid_inline1551, pad_value=pl.PadValue.zero)
                row2_eff_inline1479: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row2_valid_inline1517, pad_value=pl.PadValue.zero)
                row3_eff_inline1559: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row3_valid_inline1553, pad_value=pl.PadValue.zero)
                row_sum_tmp_iter_inline1562: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                col_sum_inline1563: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(pl.tile.add(row0_eff_inline1555, row1_eff_inline1557), pl.tile.add(row2_eff_inline1479, row3_eff_inline1559))
                col_sum_v1_inline1564: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_inline1563, 9.9999999999999995e-07)
                row0_cur_inline1565: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_eff_inline1555, col_sum_v1_inline1564)
                row1_cur_inline1449: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_eff_inline1557, col_sum_v1_inline1564)
                row2_cur_inline1566: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_eff_inline1479, col_sum_v1_inline1564)
                row3_cur_inline1569: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_eff_inline1559, col_sum_v1_inline1564)
                for _sk_it_inline1460 in pl.pipeline(19, stage=2):
                    row0_rowsum_inline1571: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row0_cur_inline1565, row_sum_tmp_iter_inline1562), 9.9999999999999995e-07)
                    row1_rowsum_inline1469: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row1_cur_inline1449, row_sum_tmp_iter_inline1562), 9.9999999999999995e-07)
                    row2_rowsum_inline1573: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row2_cur_inline1566, row_sum_tmp_iter_inline1562), 9.9999999999999995e-07)
                    row3_rowsum_inline1574: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row3_cur_inline1569, row_sum_tmp_iter_inline1562), 9.9999999999999995e-07)
                    row0_norm_inline1576: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row0_cur_inline1565, row0_rowsum_inline1571)
                    row1_norm_inline1504: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row1_cur_inline1449, row1_rowsum_inline1469)
                    row2_norm_inline1466: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row2_cur_inline1566, row2_rowsum_inline1573)
                    row3_norm_inline1577: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row3_cur_inline1569, row3_rowsum_inline1574)
                    col_sum_v1_inline1564: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(pl.tile.add(row0_norm_inline1576, row1_norm_inline1504), pl.tile.add(row2_norm_inline1466, row3_norm_inline1577))
                    col_sum_v1_inline1564: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_v1_inline1564, 9.9999999999999995e-07)
                    row0_cur_inline1565: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_norm_inline1576, col_sum_v1_inline1564)
                    row1_cur_inline1449: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_norm_inline1504, col_sum_v1_inline1564)
                    row2_cur_inline1566: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_norm_inline1466, col_sum_v1_inline1564)
                    row3_cur_inline1569: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_norm_inline1577, col_sum_v1_inline1564)
                if valid_rows_inline1507 == 8:
                    row0_out_inline1541: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row0_cur_inline1565, 8, 4)
                    row1_out_inline1536: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row1_cur_inline1449, 8, 4)
                    row2_out_inline1558: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row2_cur_inline1566, 8, 4)
                    row3_out_inline1474: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row3_cur_inline1569, 8, 4)
                    pl.tile.store(row0_out_inline1541, [t0_inline1476, 0], comb_t_inline1267)
                    pl.tile.store(row1_out_inline1536, [t0_inline1476, 4], comb_t_inline1267)
                    pl.tile.store(row2_out_inline1558, [t0_inline1476, 8], comb_t_inline1267)
                    pl.tile.store(row3_out_inline1474, [t0_inline1476, 12], comb_t_inline1267)
                else:
                    pl.tile.store(row0_cur_inline1565, [0, 0], comb_tail_store_inline1523)
                    pl.tile.store(row1_cur_inline1449, [0, 8], comb_tail_store_inline1523)
                    pl.tile.store(row2_cur_inline1566, [0, 16], comb_tail_store_inline1523)
                    pl.tile.store(row3_cur_inline1569, [0, 24], comb_tail_store_inline1523)
                    row0_tail_inline1556: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store_inline1523, [0, 0], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                    row1_tail_inline1578: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store_inline1523, [0, 8], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                    row2_tail_inline1500: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store_inline1523, [0, 16], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                    row3_tail_inline1451: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 4])] = pl.tile.load(comb_tail_store_inline1523, [0, 24], [8, 8], [valid_rows_inline1507, 4], target_memory=pl.Mem.Vec)
                    pl.tile.store(row0_tail_inline1556, [t0_inline1476, 0], comb_t_inline1267)
                    pl.tile.store(row1_tail_inline1578, [t0_inline1476, 4], comb_t_inline1267)
                    pl.tile.store(row2_tail_inline1500, [t0_inline1476, 8], comb_t_inline1267)
                    pl.tile.store(row3_tail_inline1451, [t0_inline1476, 12], comb_t_inline1267)
            x_mixed_tail_store_inline1462: pl.Tensor[[8, 4096], pl.BF16] = pl.tensor.create([8, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            for blk_inline1491 in pl.spmd(token_tiles_inline1492 * 1, name_hint="mix_x_spmd", allow_early_resolve=True):
                t0_inline1476: pl.Scalar[pl.INDEX] = blk_inline1491 // 1 * 8
                d_base_inline1454: pl.Scalar[pl.INDEX] = blk_inline1491 % 1 * 4096
                valid_rows_inline1507: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1568 - t0_inline1476)
                pre_tile_t_inline1443: pl.Tensor[[8, 8], pl.FP32, pl.TensorView(stride=[1, 8], layout=pl.TensorLayout.DN)] = pl.tensor.transpose(pl.tensor.slice(pre_val_store_inline1529, [8, 8], [t0_inline1476, 0]), 0, 1)
                pre0_inline1526: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t_inline1443, [1, 8], [0, 0]), [8, 1])
                pre1_inline1442: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t_inline1443, [1, 8], [1, 0]), [8, 1])
                pre2_inline1441: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t_inline1443, [1, 8], [2, 0]), [8, 1])
                pre3_inline1440: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t_inline1443, [1, 8], [3, 0]), [8, 1])
                for db_inline1465 in pl.pipeline(16, stage=2):
                    d0_inline1496: pl.Scalar[pl.INDEX] = d_base_inline1454 + db_inline1465 * 256
                    x0_inline1439: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497, [8, 256], [t0_inline1476, 0 + d0_inline1496], [valid_rows_inline1507, 256])
                    x1_inline1446: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497, [8, 256], [t0_inline1476, 4096 + d0_inline1496], [valid_rows_inline1507, 256])
                    x2_inline1542: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497, [8, 256], [t0_inline1476, 8192 + d0_inline1496], [valid_rows_inline1507, 256])
                    x3_inline1438: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497, [8, 256], [t0_inline1476, 12288 + d0_inline1496], [valid_rows_inline1507, 256])
                    y0_inline1437: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x0_inline1439, pre0_inline1526)
                    y1_inline1550: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x1_inline1446, pre1_inline1442)
                    y2_inline1436: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x2_inline1542, pre2_inline1441)
                    y3_inline1435: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x3_inline1438, pre3_inline1440)
                    y_tile_inline1434: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.add(pl.tensor.add(y0_inline1437, y1_inline1550), pl.tensor.add(y2_inline1436, y3_inline1435))
                    y_bf16_inline1433: pl.Tensor[[8, 256], pl.BF16] = pl.tensor.cast(y_tile_inline1434, target_type=pl.BF16, mode='rint')
                    if valid_rows_inline1507 == 8:
                        x_mixed_inline1253: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.assemble(x_mixed_inline1253, y_bf16_inline1433, [t0_inline1476, d0_inline1496])
                    else:
                        x_mixed_tail_store_inline1462: pl.Tensor[[8, 4096], pl.BF16] = pl.tensor.assemble(x_mixed_tail_store_inline1462, y_bf16_inline1433, [0, d0_inline1496])
                        y_out_inline1521: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows, 256])] = pl.tile.load(x_mixed_tail_store_inline1462, [0, d0_inline1496], [8, 256], [valid_rows_inline1507, 256], target_memory=pl.Mem.Vec)
                        pl.tile.store(y_out_inline1521, [t0_inline1476, d0_inline1496], x_mixed_inline1253)
        idx_cos_il_inline1282: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.create([t_dim_inline1251, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        idx_sin_signed_inline1307: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.create([t_dim_inline1251, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_cos_il_full_inline1249: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.create([kv_dim_inline1261, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_sin_signed_full_inline1263: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.create([kv_dim_inline1261, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_interleave") as rope_tid_inline1259:
            il_ones_inline1242: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.full([4, 64], dtype=pl.FP32, value=1.0)
            il_col_inline1252: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.col_expand_mul(il_ones_inline1242, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
            il_dup_f_inline1250: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(il_col_inline1252, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
            il_dup_idx_inline1272: pl.Tensor[[4, 64], pl.INT32] = pl.tensor.cast(il_dup_f_inline1250, target_type=pl.INT32, mode='round')
            il_lane_inline1300: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.sub(il_col_inline1252, pl.tensor.muls(il_dup_f_inline1250, 2.0))
            il_sign_inline1298: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(il_lane_inline1300, 2.0), 1.0)
            for rope_t0_inline1256 in pl.range(0, t_dim_inline1251, 4):
                idx_cos_il_inline1282: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.assemble(idx_cos_il_inline1282, pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(freqs_cos_local, [4, 32], [rope_t0_inline1256, 0]), target_type=pl.FP32, mode='round'), il_dup_idx_inline1272, dim=-1), [rope_t0_inline1256, 0])
                idx_sin_signed_inline1307: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.assemble(idx_sin_signed_inline1307, pl.tensor.mul(pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(freqs_sin_local, [4, 32], [rope_t0_inline1256, 0]), target_type=pl.FP32, mode='round'), il_dup_idx_inline1272, dim=-1), il_sign_inline1298), [rope_t0_inline1256, 0])
            for cmp_t0_inline1248 in pl.range(0, kv_dim_inline1261, 4):
                cmp_cos_il_full_inline1249: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.assemble(cmp_cos_il_full_inline1249, pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(cmp_freqs_cos, [4, 32], [cmp_t0_inline1248, 0]), target_type=pl.FP32, mode='round'), il_dup_idx_inline1272, dim=-1), [cmp_t0_inline1248, 0])
                cmp_sin_signed_full_inline1263: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.assemble(cmp_sin_signed_full_inline1263, pl.tensor.mul(pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(cmp_freqs_sin, [4, 32], [cmp_t0_inline1248, 0]), target_type=pl.FP32, mode='round'), il_dup_idx_inline1272, dim=-1), il_sign_inline1298), [cmp_t0_inline1248, 0])
        x_normed_t_inline1243: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.create([t_dim_inline1251, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            t_dim_inline1611: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_mixed_inline1253, 0)
            token_tiles_inline1607: pl.Scalar[pl.INDEX] = (t_dim_inline1611 + 8 - 1) // 8
            with pl.spmd(token_tiles_inline1607, name_hint="rms_norm_spmd", allow_early_resolve=True) as rms_tid_inline1605:
                tg_idx_inline1603: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                tg_inline1598: pl.Scalar[pl.INDEX] = tg_idx_inline1603 * 8
                valid_rows_inline1600: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1611 - tg_inline1598)
                if valid_rows_inline1600 == 8:
                    # Run the aligned token tile through the existing Tensor-level dataflow.
                    tg_inline80_inline1597: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 8
                    x_sq_sum_inline83_inline1612: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                    for rms_db_inline82_inline1602 in pl.pipeline(32, stage=2):
                        rms_d0_inline87_inline1595: pl.Scalar[pl.INDEX] = rms_db_inline82_inline1602 * 128
                        rms_x_input_inline84_inline1593: pl.Tensor[[8, 128], pl.BF16] = pl.tensor.slice(x_mixed_inline1253, [8, 128], [tg_inline80_inline1597, rms_d0_inline87_inline1595])
                        rms_x_chunk_inline88_inline1608: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(rms_x_input_inline84_inline1593, target_type=pl.FP32, mode='round')
                        rms_x_sq_inline89_inline1619: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.mul(rms_x_chunk_inline88_inline1608, rms_x_chunk_inline88_inline1608)
                        rms_x_row_sum_inline81_inline1609: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(rms_x_sq_inline89_inline1619), [1, 8])
                        x_sq_sum_inline83_inline1612: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(x_sq_sum_inline83_inline1612, rms_x_row_sum_inline81_inline1609)
                    x_inv_rms_inline85_inline1610: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(x_sq_sum_inline83_inline1612, 0.000244140625), 9.9999999999999995e-07), high_precision=True)
                    x_inv_rms_t_inline93_inline1594: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(x_inv_rms_inline85_inline1610, [8, 1])
                    for apply_db_inline86_inline1614 in pl.pipeline(32, stage=2):
                        apply_d0_inline91_inline1615: pl.Scalar[pl.INDEX] = apply_db_inline86_inline1614 * 128
                        apply_x_input_inline94_inline1613: pl.Tensor[[8, 128], pl.BF16] = pl.tensor.slice(x_mixed_inline1253, [8, 128], [tg_inline80_inline1597, apply_d0_inline91_inline1615])
                        apply_x_chunk_inline78_inline1620: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(apply_x_input_inline94_inline1613, target_type=pl.FP32, mode='round')
                        norm_w_input_inline77_inline1617: pl.Tensor[[128], pl.BF16] = pl.tensor.slice(attn_norm_w, [128], [apply_d0_inline91_inline1615])
                        norm_w_chunk_inline92_inline1604: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.reshape(norm_w_input_inline77_inline1617, [1, 128]), target_type=pl.FP32, mode='round')
                        x_scaled_inline90_inline1622: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.row_expand_mul(apply_x_chunk_inline78_inline1620, x_inv_rms_t_inline93_inline1594)
                        x_normed_chunk_inline79_inline1592: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.col_expand_mul(x_scaled_inline90_inline1622, norm_w_chunk_inline92_inline1604)
                        x_normed_t_inline1243: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.assemble(x_normed_t_inline1243, pl.tensor.cast(x_normed_chunk_inline79_inline1592, target_type=pl.BF16, mode='rint'), [tg_inline80_inline1597, apply_d0_inline91_inline1615])
                else:
                    # Run the ragged last token tile through explicit `valid_shape` load/store.
                    # 
                    #         Step for step the same RMSNorm as `_rms_norm_full_tile`. The two live in
                    #         separate scopes rather than in one `if`/`else` body because this path binds
                    #         the shared names (`x_sq_sum`, `x_inv_rms`, …) to Vec-space Tiles while the
                    #         aligned path binds them to Tensors, and a name cannot be rebound to a
                    #         different type inside one kernel.
                    #         
                    tg_inline103_inline1596: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 8
                    valid_rows_inline113_inline1601: pl.Scalar[pl.INDEX] = pl.min(8, pl.tensor.dim(x_mixed_inline1253, 0) - tg_inline103_inline1596)
                    row_reduce_tmp_inline101_inline1623: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    x_sq_sum_inline104_inline1624: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                    for rms_db_inline105_inline1591 in pl.pipeline(32, stage=2):
                        rms_d0_inline106_inline1589: pl.Scalar[pl.INDEX] = rms_db_inline105_inline1591 * 128
                        rms_x_input_inline107_inline1587: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.load(x_mixed_inline1253, [tg_inline103_inline1596, rms_d0_inline106_inline1589], [8, 128], [valid_rows_inline113_inline1601, 128], target_memory=pl.Mem.Vec)
                        rms_x_chunk_inline108_inline1586: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.cast(rms_x_input_inline107_inline1587, target_type=pl.FP32, mode='round')
                        rms_x_sq_inline111_inline1585: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.mul(rms_x_chunk_inline108_inline1586, rms_x_chunk_inline108_inline1586)
                        rms_x_row_sum_inline114_inline1584: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_1])] = pl.tile.reshape(pl.tile.row_sum(rms_x_sq_inline111_inline1585, row_reduce_tmp_inline101_inline1623), [1, 8])
                        x_sq_sum_inline104_inline1624: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(x_sq_sum_inline104_inline1624, rms_x_row_sum_inline114_inline1584)
                    x_inv_rms_inline115_inline1618: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(x_sq_sum_inline104_inline1624, 0.000244140625), 9.9999999999999995e-07)))
                    x_inv_rms_t_inline116_inline1621: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(x_inv_rms_inline115_inline1618, [8, 1])
                    for apply_db_inline110_inline1583 in pl.pipeline(32, stage=2):
                        apply_d0_inline100_inline1582: pl.Scalar[pl.INDEX] = apply_db_inline110_inline1583 * 128
                        apply_x_input_inline99_inline1616: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.load(x_mixed_inline1253, [tg_inline103_inline1596, apply_d0_inline100_inline1582], [8, 128], [valid_rows_inline113_inline1601, 128], target_memory=pl.Mem.Vec)
                        apply_x_chunk_inline98_inline1599: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.cast(apply_x_input_inline99_inline1616, target_type=pl.FP32, mode='round')
                        norm_w_input_inline96_inline1588: pl.Tile[[128], pl.BF16, pl.Mem.Vec] = pl.tile.load(attn_norm_w, [apply_d0_inline100_inline1582], [128], [128], target_memory=pl.Mem.Vec)
                        norm_w_chunk_inline102_inline1581: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.reshape(norm_w_input_inline96_inline1588, [1, 128]), target_type=pl.FP32, mode='round')
                        x_scaled_inline109_inline1606: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.row_expand_mul(apply_x_chunk_inline98_inline1599, x_inv_rms_t_inline116_inline1621)
                        x_normed_chunk_inline112_inline1580: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.col_expand_mul(x_scaled_inline109_inline1606, norm_w_chunk_inline102_inline1581)
                        x_normed_bf16_inline95_inline1590: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.cast(x_normed_chunk_inline112_inline1580, target_type=pl.BF16, mode='rint')
                        x_normed_valid_inline97_inline1579: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_1, 128])] = pl.tile.set_validshape(x_normed_bf16_inline95_inline1590, valid_rows_inline113_inline1601, 128)
                        pl.tile.store(x_normed_valid_inline97_inline1579, [tg_inline103_inline1596, apply_d0_inline100_inline1582], x_normed_t_inline1243)
        kv_wb_blocks_inline1274: pl.Scalar[pl.INDEX] = kv_dim_inline1261 // 8
        x_normed_full_inline1240: pl.Tensor[[kv_dim, 4096], pl.BF16] = pl.tensor.create([kv_dim_inline1261, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            # Gather rank-major rows and retire the complete two-phase signal epoch.
            local_rows_inline1634: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243, 0)
            local_t_inline1640: pl.Scalar[pl.INT32] = pl.cast(local_rows_inline1634, pl.INT32)
            target_row_inline1642: pl.Scalar[pl.INT32] = tp_rank * local_t_inline1640
            full_local_inline1644: pl.Scalar[pl.INDEX] = pl.cast(local_t_inline1640, pl.INDEX) // 8 * 8
            with pl.spmd(16, name_hint="cp_token_allgather_push_spmd", allow_early_resolve=True) as _push_tid_inline1646:
                worker_inline1647: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for peer_tp_inline1635 in pl.range(2):
                    for band_row_inline1648 in pl.range(worker_inline1647 * 8, full_local_inline1644, 128):
                        pld.tensor.put(gather_window, pl.cast(group_base, pl.INDEX) + peer_tp_inline1635, x_normed_t_inline1243, [pl.cast(target_row_inline1642, pl.INDEX) + band_row_inline1648, 0], [band_row_inline1648, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                    for tail_row_inline1651 in pl.range(full_local_inline1644 + worker_inline1647, local_t_inline1640, 16):
                        pld.tensor.put(gather_window, pl.cast(group_base, pl.INDEX) + peer_tp_inline1635, x_normed_t_inline1243, [pl.cast(target_row_inline1642, pl.INDEX) + tail_row_inline1651, 0], [tail_row_inline1651, 0], [1, 4096], atomic=pl.AtomicType.None_, chunk_rows=1, chunk_cols=4096)
                for peer_tp_inline1636 in pl.range(2):
                    if peer_tp_inline1636 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(gather_signal, pl.cast(group_base, pl.INDEX) + peer_tp_inline1636, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_payload_wait", deps=[_push_tid_inline1646]) as _payload_wait_tid_inline1643:
                for source_tp_inline1637 in pl.range(2):
                    if source_tp_inline1637 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.wait(gather_signal, [source_tp_inline1637, 0], pl.cast(16, pl.INT32), cmp=1)
            group_rows_inline1639: pl.Scalar[pl.INDEX] = 2 * local_rows_inline1634
            full_rows_inline1641: pl.Scalar[pl.INDEX] = group_rows_inline1639 // 16 * 16
            with pl.spmd(16, name_hint="cp_token_allgather_readback_spmd", deps=[_push_tid_inline1646, _payload_wait_tid_inline1643]) as _readback_tid_inline1633:
                worker_v1_inline1638: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for tile_row_inline1632 in pl.range(worker_v1_inline1638 * 16, full_rows_inline1641, 256):
                    window_tile_inline1650: pld.DistributedTensor[[16, 4096], pl.BF16] = pl.tensor.slice(gather_window, [16, 4096], [tile_row_inline1632, 0])
                    x_normed_full_inline1240: pl.Tensor[[kv_dim, 4096], pl.BF16] = pl.tensor.assemble(x_normed_full_inline1240, window_tile_inline1650, [tile_row_inline1632, 0])
                for tail_row_inline1629 in pl.range(full_rows_inline1641 + worker_v1_inline1638, group_rows_inline1639, 16):
                    window_row_inline1631: pld.DistributedTensor[[1, 4096], pl.BF16] = pl.tensor.slice(gather_window, [1, 4096], [tail_row_inline1629, 0])
                    x_normed_full_inline1240: pl.Tensor[[kv_dim, 4096], pl.BF16] = pl.tensor.assemble(x_normed_full_inline1240, window_row_inline1631, [tail_row_inline1629, 0])
                for peer_tp_inline1628 in pl.range(2):
                    if peer_tp_inline1628 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(gather_signal, pl.cast(group_base, pl.INDEX) + peer_tp_inline1628, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_readback_wait", deps=[_readback_tid_inline1633]) as _readback_wait_tid_inline1630:
                for source_tp_inline1627 in pl.range(2):
                    if source_tp_inline1627 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.wait(gather_signal, [source_tp_inline1627, 0], pl.cast(32, pl.INT32), cmp=1)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_retire", deps=[_readback_tid_inline1633, _readback_wait_tid_inline1630]):
                completion_anchor_inline1626: pl.Scalar[pl.BF16] = pl.tensor.read(x_normed_full_inline1240, [0, 0])
                reset_value_inline1649: pl.Scalar[pl.INT32] = pl.cast(-32, pl.INT32)
                self_rank_inline1625: pl.Scalar[pl.INT32] = group_base + tp_rank
                for source_tp_inline1645 in pl.range(2):
                    if source_tp_inline1645 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(gather_signal, self_rank_inline1625, [source_tp_inline1645, 0], reset_value_inline1649, op=0)
                pl.tensor.write(x_normed_full_inline1240, [0, 0], completion_anchor_inline1626)
            _gathered_normed_inline1281: pl.Tensor[[kv_dim, 4096], pl.BF16] = x_normed_full_inline1240
            gather_signal: pld.DistributedTensor[[2, 1], pl.INT32] = gather_signal
        kv_full_inline1265: pl.Tensor[[kv_dim, 512], pl.BF16] = pl.tensor.create([kv_dim_inline1261, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        qr_inline1255: pl.Tensor[[t_dim, 1024], pl.INT8] = pl.tensor.create([t_dim_inline1251, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
        qr_scale_inline1310: pl.Tensor[[t_dim, 1], pl.FP32] = pl.tensor.create([t_dim_inline1251, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        position_ids_t1_inline1288: pl.Tensor[[t_dim, 1], pl.INT32] = pl.tensor.reshape(position_ids_local, [t_dim_inline1251, 1])
        attention_local_flat_inline1292: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.create([2048, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        attn_out_inline1284: pl.Tensor[[t_dim, 4096], pl.BF16] = pl.tensor.create([t_dim_inline1251, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            late_dep_inline1297: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[rope_tid_inline1259])
            kv_cos_il_inline1258: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.create([kv_dim_inline1261, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_sin_signed_inline1301: pl.Tensor[[kv_dim, 64], pl.FP32] = pl.tensor.create([kv_dim_inline1261, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_swap_idx_inline1305: pl.Tensor[[kv_dim, 64], pl.INT32] = pl.tensor.create([kv_dim_inline1261, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            # Build the head-invariant interleaved cos / sign-folded sin / swap-index rope rows.
            t_dim_inline1682: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_cos, 0)
            rope_cos_view_inline1679: pl.Tensor[[t_dim_2, 64], pl.BF16] = pl.tensor.reshape(freqs_cos, [t_dim_inline1682, 64])
            rope_sin_view_inline1674: pl.Tensor[[t_dim_2, 64], pl.BF16] = pl.tensor.reshape(freqs_sin, [t_dim_inline1682, 64])
            rope_cos_il_view_inline1670: pl.Tensor[[t_dim_2, 64], pl.FP32] = pl.tensor.reshape(kv_cos_il_inline1258, [t_dim_inline1682, 64])
            rope_sin_signed_view_inline1668: pl.Tensor[[t_dim_2, 64], pl.FP32] = pl.tensor.reshape(kv_sin_signed_inline1301, [t_dim_inline1682, 64])
            rope_swap_idx_view_inline1694: pl.Tensor[[t_dim_2, 64], pl.INT32] = pl.tensor.reshape(kv_swap_idx_inline1305, [t_dim_inline1682, 64])
            token_tiles_inline1673: pl.Scalar[pl.INDEX] = (t_dim_inline1682 + 8 - 1) // 8
            for qrp_idx_inline1681 in pl.spmd(token_tiles_inline1673, name_hint="q_rope_prepare_spmd", allow_early_resolve=True):
                qrp_t0_inline1672: pl.Scalar[pl.INDEX] = qrp_idx_inline1681 * 8
                qrp_valid_rows_inline1667: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1682 - qrp_t0_inline1672)
                qrp_ones_inline1686: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
                qrp_idx_i32_inline1666: pl.Tensor[[1, 64], pl.INT32] = pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                qrp_idx_fp32_inline1661: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(qrp_idx_i32_inline1666, target_type=pl.FP32, mode='round')
                qrp_col_inline1675: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(qrp_ones_inline1686, qrp_idx_fp32_inline1661)
                qrp_half_inline1683: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_col_inline1675, 0.5)
                qrp_dup_i32_inline1685: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_half_inline1683, target_type=pl.INT32, mode='trunc')
                qrp_dup_f_inline1663: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_dup_i32_inline1685, target_type=pl.FP32, mode='round')
                qrp_dup_idx_inline1680: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_dup_f_inline1663, target_type=pl.INT32, mode='round')
                qrp_lane_inline1678: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_col_inline1675, pl.tensor.muls(qrp_dup_f_inline1663, 2.0))
                qrp_next_col_inline1689: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(qrp_col_inline1675, 1.0)
                qrp_lane_offset_inline1692: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_lane_inline1678, 2.0)
                qrp_swap_f_inline1690: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_next_col_inline1689, qrp_lane_offset_inline1692)
                qrp_swap_idx_inline1695: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_swap_f_inline1690, target_type=pl.INT32, mode='round')
                qrp_sign_inline1687: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(qrp_lane_inline1678, 2.0), 1.0)
                if qrp_valid_rows_inline1667 == 8:
                    qrp_cos_rows_full_inline1693: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_cos_view_inline1679, [8, 64], [qrp_t0_inline1672, 0])
                    qrp_sin_rows_full_inline1696: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_sin_view_inline1674, [8, 64], [qrp_t0_inline1672, 0])
                    qrp_cos_full_inline1676: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_cos_rows_full_inline1693, target_type=pl.FP32, mode='round')
                    qrp_sin_full_inline1669: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_sin_rows_full_inline1696, target_type=pl.FP32, mode='round')
                    qrp_cos_il_full_inline1697: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_cos_full_inline1676, qrp_dup_idx_inline1680, dim=-1)
                    qrp_sin_il_full_inline1671: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_sin_full_inline1669, qrp_dup_idx_inline1680, dim=-1)
                    qrp_sin_signed_full_inline1691: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(qrp_sin_il_full_inline1671, qrp_sign_inline1687)
                    rope_cos_il_view_inline1670: pl.Tensor[[t_dim_2, 64], pl.FP32] = pl.tensor.assemble(rope_cos_il_view_inline1670, qrp_cos_il_full_inline1697, [qrp_t0_inline1672, 0])
                    rope_sin_signed_view_inline1668: pl.Tensor[[t_dim_2, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed_view_inline1668, qrp_sin_signed_full_inline1691, [qrp_t0_inline1672, 0])
                    rope_swap_idx_view_inline1694: pl.Tensor[[t_dim_2, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_view_inline1694, qrp_swap_idx_inline1695, [qrp_t0_inline1672, 0])
                else:
                    qrp_cos_rows_tail_inline1684: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.load(rope_cos_view_inline1679, [qrp_t0_inline1672, 0], [8, 64], [qrp_valid_rows_inline1667, 64], target_memory=pl.Mem.Vec)
                    qrp_sin_rows_tail_inline1660: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.load(rope_sin_view_inline1674, [qrp_t0_inline1672, 0], [8, 64], [qrp_valid_rows_inline1667, 64], target_memory=pl.Mem.Vec)
                    qrp_tail_col_inline1659: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                    qrp_tail_dup_f_inline1657: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(qrp_tail_col_inline1659, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                    qrp_tail_lane_inline1662: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(qrp_tail_col_inline1659, pl.tile.muls(qrp_tail_dup_f_inline1657, 2.0))
                    qrp_tail_swap_f_inline1665: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(qrp_tail_col_inline1659, 1.0), pl.tile.muls(qrp_tail_lane_inline1662, 2.0))
                    qrp_row_seed_inline1664: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                    qrp_row_grid_inline1656: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), qrp_row_seed_inline1664)
                    qrp_row_offset_inline1655: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(qrp_row_grid_inline1656, 0, 1)
                    qrp_dup_idx_tail_inline1658: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(qrp_tail_dup_f_inline1657, qrp_row_offset_inline1655), target_type=pl.INT32, mode='round')
                    qrp_gather_tmp_inline1653: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    qrp_cos_il_tail_inline1652: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_cos_rows_tail_inline1684, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail_inline1658, qrp_gather_tmp_inline1653)
                    qrp_sin_il_tail_inline1654: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_sin_rows_tail_inline1660, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail_inline1658, qrp_gather_tmp_inline1653)
                    qrp_tail_sign_inline1677: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(pl.tile.muls(qrp_tail_lane_inline1662, 2.0), 1.0)
                    qrp_sin_signed_tail_inline1688: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_tail_inline1654, qrp_tail_sign_inline1677)
                    pl.tile.store(pl.tile.set_validshape(qrp_cos_il_tail_inline1652, qrp_valid_rows_inline1667, 64), [qrp_t0_inline1672, 0], rope_cos_il_view_inline1670)
                    pl.tile.store(pl.tile.set_validshape(qrp_sin_signed_tail_inline1688, qrp_valid_rows_inline1667, 64), [qrp_t0_inline1672, 0], rope_sin_signed_view_inline1668)
                    pl.tile.store(pl.tile.set_validshape(pl.tile.cast(qrp_tail_swap_f_inline1665, target_type=pl.INT32, mode='round'), qrp_valid_rows_inline1667, 64), [qrp_t0_inline1672, 0], rope_swap_idx_view_inline1694)
            q_cos_il_inline1311: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.create([t_dim_inline1251, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_sin_signed_inline1295: pl.Tensor[[t_dim, 64], pl.FP32] = pl.tensor.create([t_dim_inline1251, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_swap_idx_inline1313: pl.Tensor[[t_dim, 64], pl.INT32] = pl.tensor.create([t_dim_inline1251, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            # Build the head-invariant interleaved cos / sign-folded sin / swap-index rope rows.
            t_dim_inline1728: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_cos_local, 0)
            rope_cos_view_inline1725: pl.Tensor[[t_dim_2, 64], pl.BF16] = pl.tensor.reshape(freqs_cos_local, [t_dim_inline1728, 64])
            rope_sin_view_inline1720: pl.Tensor[[t_dim_2, 64], pl.BF16] = pl.tensor.reshape(freqs_sin_local, [t_dim_inline1728, 64])
            rope_cos_il_view_inline1716: pl.Tensor[[t_dim_2, 64], pl.FP32] = pl.tensor.reshape(q_cos_il_inline1311, [t_dim_inline1728, 64])
            rope_sin_signed_view_inline1714: pl.Tensor[[t_dim_2, 64], pl.FP32] = pl.tensor.reshape(q_sin_signed_inline1295, [t_dim_inline1728, 64])
            rope_swap_idx_view_inline1740: pl.Tensor[[t_dim_2, 64], pl.INT32] = pl.tensor.reshape(q_swap_idx_inline1313, [t_dim_inline1728, 64])
            token_tiles_inline1719: pl.Scalar[pl.INDEX] = (t_dim_inline1728 + 8 - 1) // 8
            for qrp_idx_inline1727 in pl.spmd(token_tiles_inline1719, name_hint="q_rope_prepare_spmd", allow_early_resolve=True):
                qrp_t0_inline1718: pl.Scalar[pl.INDEX] = qrp_idx_inline1727 * 8
                qrp_valid_rows_inline1713: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1728 - qrp_t0_inline1718)
                qrp_ones_inline1732: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
                qrp_idx_i32_inline1712: pl.Tensor[[1, 64], pl.INT32] = pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                qrp_idx_fp32_inline1707: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(qrp_idx_i32_inline1712, target_type=pl.FP32, mode='round')
                qrp_col_inline1721: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(qrp_ones_inline1732, qrp_idx_fp32_inline1707)
                qrp_half_inline1729: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_col_inline1721, 0.5)
                qrp_dup_i32_inline1731: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_half_inline1729, target_type=pl.INT32, mode='trunc')
                qrp_dup_f_inline1709: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_dup_i32_inline1731, target_type=pl.FP32, mode='round')
                qrp_dup_idx_inline1726: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_dup_f_inline1709, target_type=pl.INT32, mode='round')
                qrp_lane_inline1724: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_col_inline1721, pl.tensor.muls(qrp_dup_f_inline1709, 2.0))
                qrp_next_col_inline1735: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(qrp_col_inline1721, 1.0)
                qrp_lane_offset_inline1738: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_lane_inline1724, 2.0)
                qrp_swap_f_inline1736: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_next_col_inline1735, qrp_lane_offset_inline1738)
                qrp_swap_idx_inline1741: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_swap_f_inline1736, target_type=pl.INT32, mode='round')
                qrp_sign_inline1733: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(qrp_lane_inline1724, 2.0), 1.0)
                if qrp_valid_rows_inline1713 == 8:
                    qrp_cos_rows_full_inline1739: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_cos_view_inline1725, [8, 64], [qrp_t0_inline1718, 0])
                    qrp_sin_rows_full_inline1742: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_sin_view_inline1720, [8, 64], [qrp_t0_inline1718, 0])
                    qrp_cos_full_inline1722: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_cos_rows_full_inline1739, target_type=pl.FP32, mode='round')
                    qrp_sin_full_inline1715: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_sin_rows_full_inline1742, target_type=pl.FP32, mode='round')
                    qrp_cos_il_full_inline1743: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_cos_full_inline1722, qrp_dup_idx_inline1726, dim=-1)
                    qrp_sin_il_full_inline1717: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_sin_full_inline1715, qrp_dup_idx_inline1726, dim=-1)
                    qrp_sin_signed_full_inline1737: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(qrp_sin_il_full_inline1717, qrp_sign_inline1733)
                    rope_cos_il_view_inline1716: pl.Tensor[[t_dim_2, 64], pl.FP32] = pl.tensor.assemble(rope_cos_il_view_inline1716, qrp_cos_il_full_inline1743, [qrp_t0_inline1718, 0])
                    rope_sin_signed_view_inline1714: pl.Tensor[[t_dim_2, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed_view_inline1714, qrp_sin_signed_full_inline1737, [qrp_t0_inline1718, 0])
                    rope_swap_idx_view_inline1740: pl.Tensor[[t_dim_2, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_view_inline1740, qrp_swap_idx_inline1741, [qrp_t0_inline1718, 0])
                else:
                    qrp_cos_rows_tail_inline1730: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.load(rope_cos_view_inline1725, [qrp_t0_inline1718, 0], [8, 64], [qrp_valid_rows_inline1713, 64], target_memory=pl.Mem.Vec)
                    qrp_sin_rows_tail_inline1706: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows, 64])] = pl.tile.load(rope_sin_view_inline1720, [qrp_t0_inline1718, 0], [8, 64], [qrp_valid_rows_inline1713, 64], target_memory=pl.Mem.Vec)
                    qrp_tail_col_inline1705: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                    qrp_tail_dup_f_inline1703: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(qrp_tail_col_inline1705, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                    qrp_tail_lane_inline1708: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(qrp_tail_col_inline1705, pl.tile.muls(qrp_tail_dup_f_inline1703, 2.0))
                    qrp_tail_swap_f_inline1711: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(qrp_tail_col_inline1705, 1.0), pl.tile.muls(qrp_tail_lane_inline1708, 2.0))
                    qrp_row_seed_inline1710: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                    qrp_row_grid_inline1702: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), qrp_row_seed_inline1710)
                    qrp_row_offset_inline1701: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(qrp_row_grid_inline1702, 0, 1)
                    qrp_dup_idx_tail_inline1704: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(qrp_tail_dup_f_inline1703, qrp_row_offset_inline1701), target_type=pl.INT32, mode='round')
                    qrp_gather_tmp_inline1699: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    qrp_cos_il_tail_inline1698: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_cos_rows_tail_inline1730, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail_inline1704, qrp_gather_tmp_inline1699)
                    qrp_sin_il_tail_inline1700: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_sin_rows_tail_inline1706, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail_inline1704, qrp_gather_tmp_inline1699)
                    qrp_tail_sign_inline1723: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(pl.tile.muls(qrp_tail_lane_inline1708, 2.0), 1.0)
                    qrp_sin_signed_tail_inline1734: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_tail_inline1700, qrp_tail_sign_inline1723)
                    pl.tile.store(pl.tile.set_validshape(qrp_cos_il_tail_inline1698, qrp_valid_rows_inline1713, 64), [qrp_t0_inline1718, 0], rope_cos_il_view_inline1716)
                    pl.tile.store(pl.tile.set_validshape(qrp_sin_signed_tail_inline1734, qrp_valid_rows_inline1713, 64), [qrp_t0_inline1718, 0], rope_sin_signed_view_inline1714)
                    pl.tile.store(pl.tile.set_validshape(pl.tile.cast(qrp_tail_swap_f_inline1711, target_type=pl.INT32, mode='round'), qrp_valid_rows_inline1713, 64), [qrp_t0_inline1718, 0], rope_swap_idx_view_inline1740)
            # Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles.
            t_dim_inline1813: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243, 0)
            for tile_base_inline1799 in pl.range(0, t_dim_inline1813, 512):
                tile_rows_inline1798: pl.Scalar[pl.INDEX] = pl.min(512, t_dim_inline1813 - tile_base_inline1799)
                with pl.scope():
                    x_view_inline1797: pl.Tensor[[t_dim_3, 4096], pl.BF16] = pl.tensor.reshape(x_normed_t_inline1243, [t_dim_inline1813, 4096])
                    qr_t_matmul_inline1793: pl.Scalar[pl.INDEX] = (tile_rows_inline1798 + 16 - 1) // 16 * 16
                    qproj_t_matmul_inline1791: pl.Scalar[pl.INDEX] = (tile_rows_inline1798 + 16 - 1) // 16 * 16
                    qproj_full_rows_inline1804: pl.Scalar[pl.INDEX] = tile_rows_inline1798 // 64 * 64
                    qr_fp32_inline1834: pl.Tensor[[qr_t_matmul, 1024], pl.FP32] = pl.tensor.create([qr_t_matmul_inline1793, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
                        for ts0_inline1784 in pl.range(0, qr_t_matmul_inline1793, 16):
                            for nseed0_inline1855 in pl.range(0, 1024, 128):
                                qr_seed_inline1790: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.full([16, 128], dtype=pl.FP32, value=0.0)
                                qr_fp32_inline1834: pl.Tensor[[qr_t_matmul, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32_inline1834, qr_seed_inline1790, [ts0_inline1784, nseed0_inline1855])
                    for qbg_idx_inline1807 in pl.spmd(16, name_hint="qr_proj_matmul_spmd", allow_early_resolve=True):
                        q_a_col0_inline1795: pl.Scalar[pl.INDEX] = qbg_idx_inline1807 // 2 * 128
                        qr_k_base_inline1819: pl.Scalar[pl.INDEX] = qbg_idx_inline1807 % 2 * 2048
                        for t0_inline1823 in pl.range(0, qr_t_matmul_inline1793, 16):
                            q_acc_inline1824: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            for db_inline1815 in pl.pipeline(8, stage=2):
                                qr_d0_inline1822: pl.Scalar[pl.INDEX] = qr_k_base_inline1819 + db_inline1815 * 256
                                qr_rows_inline1808: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows_inline1798 - t0_inline1823)
                                x_t0_inline1766: pl.Scalar[pl.INDEX] = tile_base_inline1799 + t0_inline1823
                                q_x_chunk_bf16_inline1785: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[qr_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_view_inline1797, [16, 256], [x_t0_inline1766, qr_d0_inline1822], [qr_rows_inline1808, 256])
                                w_chunk_inline1827: pl.Tensor[[256, 128], pl.BF16] = pl.tensor.slice(wq_a, [256, 128], [qr_d0_inline1822, q_a_col0_inline1795])
                                if db_inline1815 == 0:
                                    q_acc_inline1824: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(q_x_chunk_bf16_inline1785, w_chunk_inline1827, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                else:
                                    q_acc_inline1824: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(q_acc_inline1824, q_x_chunk_bf16_inline1785, w_chunk_inline1827, a_trans=False, b_trans=False)
                            qr_fp32_inline1834: pl.Tensor[[qr_t_matmul, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32_inline1834, q_acc_inline1824, [t0_inline1823, q_a_col0_inline1795], atomic=pl.AtomicType.Add)
                    qr_view_inline1775: pl.Tensor[[t_dim_3, 1024], pl.INT8] = pl.tensor.reshape(qr_inline1255, [t_dim_inline1813, 1024])
                    qr_scale_view_inline1796: pl.Tensor[[t_dim_3, 1], pl.FP32] = pl.tensor.reshape(qr_scale_inline1310, [t_dim_inline1813, 1])
                    qr_i8_matmul_inline1787: pl.Tensor[[qproj_t_matmul, 1024], pl.INT8] = pl.tensor.create([qproj_t_matmul_inline1791, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                    qr_scale_pad_store_inline1814: pl.Tensor[[qproj_t_matmul, 1], pl.FP32] = pl.tensor.create([qproj_t_matmul_inline1791, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                    qr_token_tiles_inline1878: pl.Scalar[pl.INDEX] = (tile_rows_inline1798 + 8 - 1) // 8
                    for tg_idx_inline1757 in pl.spmd(qr_token_tiles_inline1878, name_hint="qr_rms_norm_quant_spmd", allow_early_resolve=True):
                        tg_inline1758: pl.Scalar[pl.INDEX] = tg_idx_inline1757 * 8
                        valid_rows_inline1818: pl.Scalar[pl.INDEX] = pl.min(8, tile_rows_inline1798 - tg_inline1758)
                        out_tg_inline1826: pl.Scalar[pl.INDEX] = tile_base_inline1799 + tg_inline1758
                        qr_sq_sum_inline1817: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                        qr_amax_g_inline1782: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                        for qr_rms_col0_inline1886 in pl.pipeline(0, 1024, 256, stage=2):
                            qr_rms_chunk_inline1781: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.slice(qr_fp32_inline1834, [8, 256], [tg_inline1758, qr_rms_col0_inline1886])
                            qr_rms_sq_inline1779: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.mul(qr_rms_chunk_inline1781, qr_rms_chunk_inline1781)
                            qr_rms_row_sum_inline1821: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(qr_rms_sq_inline1779), [1, 8])
                            qr_sq_sum_inline1817: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(qr_sq_sum_inline1817, qr_rms_row_sum_inline1821)
                            gamma_rms_cast_inline1774: pl.Tensor[[256], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_cq, [256], [qr_rms_col0_inline1886]), target_type=pl.FP32, mode='round')
                            gamma_rms_chunk_inline1772: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.reshape(gamma_rms_cast_inline1774, [1, 256])
                            qr_g_inline1786: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.col_expand_mul(qr_rms_chunk_inline1781, gamma_rms_chunk_inline1772)
                            qr_g_abs_inline1836: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.abs(qr_g_inline1786)
                            qr_g_row_max_inline1770: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(qr_g_abs_inline1836), [1, 8])
                            qr_amax_g_inline1782: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qr_amax_g_inline1782, qr_g_row_max_inline1770)
                        qr_inv_rms_inline1769: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(qr_sq_sum_inline1817, 0.0009765625), 9.9999999999999995e-07), high_precision=True)
                        qr_inv_rms_t_inline1802: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qr_inv_rms_inline1769, [8, 1])
                        qr_amax_floor_inline1829: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0001)
                        qr_amax_normed_inline1792: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.mul(qr_inv_rms_inline1769, qr_amax_g_inline1782)
                        qr_tile_amax_inline1828: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qr_amax_floor_inline1829, qr_amax_normed_inline1792)
                        qr_scale_quant_row_inline1767: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 8], dtype=pl.FP32, value=127.0), qr_tile_amax_inline1828)
                        qr_scale_quant_t_inline1765: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qr_scale_quant_row_inline1767, [8, 1])
                        qr_tile_scale_dq_inline1764: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(qr_scale_quant_row_inline1767), [8, 1])
                        qr_scale_pad_store_inline1814: pl.Tensor[[qproj_t_matmul, 1], pl.FP32] = pl.tensor.assemble(qr_scale_pad_store_inline1814, qr_tile_scale_dq_inline1764, [tg_inline1758, 0])
                        if valid_rows_inline1818 == 8:
                            qr_scale_view_inline1796: pl.Tensor[[t_dim_3, 1], pl.FP32] = pl.tensor.assemble(qr_scale_view_inline1796, qr_tile_scale_dq_inline1764, [out_tg_inline1826, 0])
                        else:
                            qr_scale_tail_inline1777: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_2, 1])] = pl.tile.load(qr_scale_pad_store_inline1814, [tg_inline1758, 0], [8, 1], [valid_rows_inline1818, 1], target_memory=pl.Mem.Vec)
                            pl.tile.store(qr_scale_tail_inline1777, [out_tg_inline1826, 0], qr_scale_view_inline1796)
                        for qa_inline1761 in pl.pipeline(0, 1024, 256, stage=2):
                            qr_chunk_inline1780: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.slice(qr_fp32_inline1834, [8, 256], [tg_inline1758, qa_inline1761])
                            gamma_q_cast_inline1776: pl.Tensor[[256], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_cq, [256], [qa_inline1761]), target_type=pl.FP32, mode='round')
                            gamma_q_chunk_inline1794: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.reshape(gamma_q_cast_inline1776, [1, 256])
                            qr_q_normed_inline1810: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(qr_chunk_inline1780, qr_inv_rms_t_inline1802), gamma_q_chunk_inline1794)
                            qr_q_scaled_inline1805: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(qr_q_normed_inline1810, qr_scale_quant_t_inline1765)
                            qr_q_i32_inline1762: pl.Tensor[[8, 256], pl.INT32] = pl.tensor.cast(qr_q_scaled_inline1805, target_type=pl.INT32, mode='rint')
                            qr_q_half_inline1756: pl.Tensor[[8, 256], pl.FP16] = pl.tensor.cast(qr_q_i32_inline1762, target_type=pl.FP16, mode='round')
                            qr_q_i8_inline1831: pl.Tensor[[8, 256], pl.INT8] = pl.tensor.cast(qr_q_half_inline1756, target_type=pl.INT8, mode='trunc')
                            qr_i8_matmul_inline1787: pl.Tensor[[qproj_t_matmul, 1024], pl.INT8] = pl.tensor.assemble(qr_i8_matmul_inline1787, qr_q_i8_inline1831, [tg_inline1758, qa_inline1761])
                            if valid_rows_inline1818 == 8:
                                qr_view_inline1775: pl.Tensor[[t_dim_3, 1024], pl.INT8] = pl.tensor.assemble(qr_view_inline1775, qr_q_i8_inline1831, [out_tg_inline1826, qa_inline1761])
                            else:
                                qr_q_tail_inline1832: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_2, 256])] = pl.tile.load(qr_i8_matmul_inline1787, [tg_inline1758, qa_inline1761], [8, 256], [valid_rows_inline1818, 256], target_memory=pl.Mem.Vec)
                                pl.tile.store(qr_q_tail_inline1832, [out_tg_inline1826, qa_inline1761], qr_view_inline1775)
                    q_proj_i32_inline1835: pl.Tensor[[qproj_t_matmul, 32768], pl.INT32] = pl.tensor.create([qproj_t_matmul_inline1791, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    for qproj_n_idx_inline1841 in pl.spmd(64, name_hint="qproj_matmul_spmd"):
                        w_col0_inline1842: pl.Scalar[pl.INDEX] = qproj_n_idx_inline1841 * 512
                        for t0_inline1861 in pl.range(0, qproj_full_rows_inline1804, 64):
                            col_acc_inline1847: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.create([64, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                            for qr_proj_col0_inline1848 in pl.pipeline(0, 1024, 128, stage=2):
                                qr_i8_chunk_inline1879: pl.Tensor[[64, 128], pl.INT8] = pl.tensor.slice(qr_i8_matmul_inline1787, [64, 128], [t0_inline1861, qr_proj_col0_inline1848])
                                wq_chunk_inline1845: pl.Tensor[[128, 512], pl.INT8] = pl.tensor.slice(wq_b, [128, 512], [qr_proj_col0_inline1848, w_col0_inline1842])
                                if qr_proj_col0_inline1848 == 0:
                                    col_acc_inline1847: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.matmul(qr_i8_chunk_inline1879, wq_chunk_inline1845, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                                else:
                                    col_acc_inline1847: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.matmul_acc(col_acc_inline1847, qr_i8_chunk_inline1879, wq_chunk_inline1845, a_trans=False, b_trans=False)
                            q_proj_i32_inline1835: pl.Tensor[[qproj_t_matmul, 32768], pl.INT32] = pl.tensor.assemble(q_proj_i32_inline1835, col_acc_inline1847, [t0_inline1861, w_col0_inline1842])
                        tail_w_col0_inline1809: pl.Scalar[pl.INDEX] = w_col0_inline1842
                        for tail_t0_inline1850 in pl.range(qproj_full_rows_inline1804, qproj_t_matmul_inline1791, 16):
                            qproj_tail_rows_inline1851: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows_inline1798 - tail_t0_inline1850)
                            tail_acc_inline1801: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.create([16, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                            for tail_qr_col0_inline1852 in pl.pipeline(0, 1024, 128, stage=2):
                                qr_i8_tail_inline1763: pl.Tensor[[16, 128], pl.INT8, pl.TensorView(valid_shape=[qproj_tail_rows, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr_i8_matmul_inline1787, [16, 128], [tail_t0_inline1850, tail_qr_col0_inline1852], [qproj_tail_rows_inline1851, 128])
                                wq_tail_inline1853: pl.Tensor[[128, 512], pl.INT8] = pl.tensor.slice(wq_b, [128, 512], [tail_qr_col0_inline1852, tail_w_col0_inline1809])
                                if tail_qr_col0_inline1852 == 0:
                                    tail_acc_inline1801: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul(qr_i8_tail_inline1763, wq_tail_inline1853, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                                else:
                                    tail_acc_inline1801: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul_acc(tail_acc_inline1801, qr_i8_tail_inline1763, wq_tail_inline1853, a_trans=False, b_trans=False)
                            q_proj_i32_inline1835: pl.Tensor[[qproj_t_matmul, 32768], pl.INT32] = pl.tensor.assemble(q_proj_i32_inline1835, tail_acc_inline1801, [tail_t0_inline1850, tail_w_col0_inline1809])
                    q_flat_inline1856: pl.Tensor[[t_dim_3, 32768], pl.BF16] = pl.tensor.reshape(q_inline1246, [t_dim_inline1813, 32768])
                    for hg_idx_inline1857 in pl.spmd(16, name_hint="qproj_dequant_rms_nope_rope_spmd", allow_early_resolve=True):
                        hg_inline1858: pl.Scalar[pl.INDEX] = hg_idx_inline1857 * 4
                        for tg_inline1844 in pl.range(0, tile_rows_inline1798, 8):
                            out_tg_inline1826: pl.Scalar[pl.INDEX] = tile_base_inline1799 + tg_inline1844
                            if tg_inline1844 + 8 <= tile_rows_inline1798:
                                qr_scale_dq_t_inline1860: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(qr_scale_pad_store_inline1814, [8, 1], [tg_inline1844, 0])
                                q_cos_il_inline1849: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(q_cos_il_inline1311, [8, 64], [out_tg_inline1826, 0])
                                q_sin_signed_inline1862: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(q_sin_signed_inline1295, [8, 64], [out_tg_inline1826, 0])
                                q_swap_idx_inline1788: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.slice(q_swap_idx_inline1313, [8, 64], [out_tg_inline1826, 0])
                                for h_inner_inline1863 in pl.pipeline(4, stage=2):
                                    h_inline1768: pl.Scalar[pl.INDEX] = hg_inline1858 + h_inner_inline1863
                                    h0_inline1800: pl.Scalar[pl.INDEX] = h_inline1768 * 512
                                    q_head_acc_inline1865: pl.Tensor[[8, 512], pl.INT32] = pl.tensor.slice(q_proj_i32_inline1835, [8, 512], [tg_inline1844, h0_inline1800])
                                    q_head_scale_inline1843: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(wq_b_scale, [512], [h0_inline1800]), [1, 512])
                                    q_head_acc_fp32_inline1866: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.cast(q_head_acc_inline1865, target_type=pl.FP32, mode='none')
                                    q_head_row_scaled_inline1867: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.row_expand_mul(q_head_acc_fp32_inline1866, qr_scale_dq_t_inline1860)
                                    q_head_dq_inline1825: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.col_expand_mul(q_head_row_scaled_inline1867, q_head_scale_inline1843)
                                    q_head_sq_inline1868: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(q_head_dq_inline1825, q_head_dq_inline1825)
                                    q_head_sq_row_inline1887: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_sum(q_head_sq_inline1868)
                                    q_head_sq_sum_inline1870: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(q_head_sq_row_inline1887, [1, 8])
                                    q_head_sq_mean_inline1846: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.muls(q_head_sq_sum_inline1870, 0.001953125)
                                    q_head_var_inline1864: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.adds(q_head_sq_mean_inline1846, 9.9999999999999995e-07)
                                    q_head_inv_rms_inline1871: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(q_head_var_inline1864, high_precision=True)
                                    q_head_inv_rms_t_inline1771: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(q_head_inv_rms_inline1871, [8, 1])
                                    q_nope_normed_inline1872: pl.Tensor[[8, 448], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_head_dq_inline1825, [8, 448], [0, 0]), q_head_inv_rms_t_inline1771)
                                    q_nope_bf16_inline1873: pl.Tensor[[8, 448], pl.BF16] = pl.tensor.cast(q_nope_normed_inline1872, target_type=pl.BF16, mode='rint')
                                    q_flat_inline1856: pl.Tensor[[t_dim_3, 32768], pl.BF16] = pl.tensor.assemble(q_flat_inline1856, q_nope_bf16_inline1873, [out_tg_inline1826, h0_inline1800])
                                    q_rope_chunk_raw_inline1837: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(q_head_dq_inline1825, [8, 64], [0, 448])
                                    q_rope_chunk_inline1811: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.row_expand_mul(q_rope_chunk_raw_inline1837, q_head_inv_rms_t_inline1771)
                                    q_rope_swapped_inline1778: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(q_rope_chunk_inline1811, q_swap_idx_inline1788, dim=-1)
                                    q_rope_base_inline1838: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(q_rope_chunk_inline1811, q_cos_il_inline1849)
                                    q_rope_delta_inline1874: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(q_rope_swapped_inline1778, q_sin_signed_inline1862)
                                    q_rope_rot_inline1875: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(q_rope_base_inline1838, q_rope_delta_inline1874)
                                    q_rope_bf16_inline1833: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.cast(q_rope_rot_inline1875, target_type=pl.BF16, mode='rint')
                                    q_flat_inline1856: pl.Tensor[[t_dim_3, 32768], pl.BF16] = pl.tensor.assemble(q_flat_inline1856, q_rope_bf16_inline1833, [out_tg_inline1826, h0_inline1800 + 448])
                            else:
                                valid_tail_rows_inline1859: pl.Scalar[pl.INDEX] = tile_rows_inline1798 - tg_inline1844
                                qr_scale_dq_tail_inline1876: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(qr_scale_pad_store_inline1814, [tg_inline1844, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                                q_cos_il_tail_inline1880: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.load(q_cos_il_inline1311, [out_tg_inline1826, 0], [8, 64], [valid_tail_rows_inline1859, 64], target_memory=pl.Mem.Vec)
                                q_sin_signed_tail_inline1882: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.load(q_sin_signed_inline1295, [out_tg_inline1826, 0], [8, 64], [valid_tail_rows_inline1859, 64], target_memory=pl.Mem.Vec)
                                q_col_inline1883: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                                q_dup_f_inline1884: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(q_col_inline1883, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                                q_lane_inline1869: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(q_col_inline1883, pl.tile.muls(q_dup_f_inline1884, 2.0))
                                q_swap_f_inline1803: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(q_col_inline1883, 1.0), pl.tile.muls(q_lane_inline1869, 2.0))
                                q_row_seed_inline1759: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                                q_row_grid_inline1820: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), q_row_seed_inline1759)
                                q_row_offset_inline1885: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(q_row_grid_inline1820, 0, 1)
                                q_swap_idx_tail_inline1877: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(q_swap_f_inline1803, q_row_offset_inline1885), target_type=pl.INT32, mode='round')
                                q_head_reduce_tmp_inline1760: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                q_gather_tmp_inline1888: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                                for h_inner_tail_inline1773 in pl.range(4):
                                    h_tail_inline1789: pl.Scalar[pl.INDEX] = hg_inline1858 + h_inner_tail_inline1773
                                    h0_tail_inline1881: pl.Scalar[pl.INDEX] = h_tail_inline1789 * 512
                                    q_head_acc_tail_inline1839: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(q_proj_i32_inline1835, [tg_inline1844, h0_tail_inline1881], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                                    q_head_scale_input_tail_inline1840: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(wq_b_scale, [h0_tail_inline1881], [512], [512], target_memory=pl.Mem.Vec)
                                    q_head_scale_tail_inline1816: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_head_scale_input_tail_inline1840, [1, 512])
                                    q_head_acc_fp32_tail_inline1755: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(q_head_acc_tail_inline1839, target_type=pl.FP32, mode='none')
                                    q_head_row_scaled_tail_inline1754: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_head_acc_fp32_tail_inline1755, qr_scale_dq_tail_inline1876)
                                    q_head_dq_tail_inline1806: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_head_row_scaled_tail_inline1754, q_head_scale_tail_inline1816)
                                    q_head_sq_tail_inline1753: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_head_dq_tail_inline1806, q_head_dq_tail_inline1806)
                                    q_head_sq_sum_tail_inline1752: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_tail_inline1753, q_head_reduce_tmp_inline1760)
                                    q_head_inv_rms_tail_inline1783: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(q_head_sq_sum_tail_inline1752, 0.001953125), 9.9999999999999995e-07)))
                                    q_nope_normed_tail_inline1751: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(pl.tile.slice(q_head_dq_tail_inline1806, [8, 448], [0, 0]), q_head_inv_rms_tail_inline1783)
                                    q_nope_bf16_tail_inline1750: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_nope_normed_tail_inline1751, target_type=pl.BF16, mode='rint')
                                    q_nope_valid_inline1749: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 448])] = pl.tile.set_validshape(q_nope_bf16_tail_inline1750, valid_tail_rows_inline1859, 448)
                                    pl.tile.store(q_nope_valid_inline1749, [out_tg_inline1826, h0_tail_inline1881], q_flat_inline1856)
                                    q_rope_chunk_raw_tail_inline1748: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail_inline1806, [8, 64], [0, 448])
                                    q_rope_chunk_tail_inline1812: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_rope_chunk_raw_tail_inline1748, q_head_inv_rms_tail_inline1783)
                                    q_rope_swapped_tail_inline1747: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(q_rope_chunk_tail_inline1812, q_swap_idx_tail_inline1877, q_gather_tmp_inline1888)
                                    q_rope_base_tail_inline1746: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_rope_chunk_tail_inline1812, q_cos_il_tail_inline1880)
                                    q_rope_delta_tail_inline1745: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_rope_swapped_tail_inline1747, q_sin_signed_tail_inline1882)
                                    q_rope_rot_tail_inline1854: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(q_rope_base_tail_inline1746, q_rope_delta_tail_inline1745)
                                    q_rope_bf16_tail_inline1744: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_rope_rot_tail_inline1854, target_type=pl.BF16, mode='rint')
                                    q_rope_valid_inline1830: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows, 64])] = pl.tile.set_validshape(q_rope_bf16_tail_inline1744, valid_tail_rows_inline1859, 64)
                                    pl.tile.store(q_rope_valid_inline1830, [out_tg_inline1826, h0_tail_inline1881 + 448], q_flat_inline1856)
            # KV LoRA, RMSNorm, and RoPE over bounded dense tiles.
            t_dim_inline1923: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240, 0)
            for tile_base_inline1954 in pl.range(0, t_dim_inline1923, 512):
                tile_rows_inline1928: pl.Scalar[pl.INDEX] = pl.min(512, t_dim_inline1923 - tile_base_inline1954)
                with pl.scope():
                    x_view_inline1914: pl.Tensor[[t_dim_4, 4096], pl.BF16] = pl.tensor.reshape(x_normed_full_inline1240, [t_dim_inline1923, 4096])
                    t_matmul_inline1930: pl.Scalar[pl.INDEX] = (tile_rows_inline1928 + 16 - 1) // 16 * 16
                    kv_fp32_inline1920: pl.Tensor[[t_matmul, 512], pl.FP32] = pl.tensor.create([t_matmul_inline1930, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
                        for kts0_inline1934 in pl.range(0, t_matmul_inline1930, 16):
                            for kvseed0_inline1936 in pl.range(0, 512, 128):
                                kv_seed_inline1926: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.full([16, 128], dtype=pl.FP32, value=0.0)
                                kv_fp32_inline1920: pl.Tensor[[t_matmul, 512], pl.FP32] = pl.tensor.assemble(kv_fp32_inline1920, kv_seed_inline1926, [kts0_inline1934, kvseed0_inline1936])
                    with pl.spmd(32, name_hint="kv_proj_matmul_spmd", deps=[late_dep_inline1297]) as _kv_tid_inline1935:
                        kbg_inline1940: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                        kv_col0_inline1937: pl.Scalar[pl.INDEX] = kbg_inline1940 // 8 * 128
                        kv_k_base_inline1962: pl.Scalar[pl.INDEX] = kbg_inline1940 // 4 % 2 * 2048
                        kv_m_group_inline1922: pl.Scalar[pl.INDEX] = kbg_inline1940 % 4
                        for t0_inline1943 in pl.range(kv_m_group_inline1922 * 16, t_matmul_inline1930, 64):
                            kv_acc_inline1938: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            for db_inline1932 in pl.pipeline(8, stage=2):
                                d0_inline1913: pl.Scalar[pl.INDEX] = kv_k_base_inline1962 + db_inline1932 * 256
                                kv_rows_inline1911: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows_inline1928 - t0_inline1943)
                                x_t0_inline1918: pl.Scalar[pl.INDEX] = tile_base_inline1954 + t0_inline1943
                                kv_x_chunk_bf16_inline1910: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[kv_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_view_inline1914, [16, 256], [x_t0_inline1918, d0_inline1913], [kv_rows_inline1911, 256])
                                wkv_chunk_inline1933: pl.Tensor[[256, 128], pl.BF16] = pl.tensor.slice(wkv, [256, 128], [d0_inline1913, kv_col0_inline1937])
                                if db_inline1932 == 0:
                                    kv_acc_inline1938: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(kv_x_chunk_bf16_inline1910, wkv_chunk_inline1933, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                else:
                                    kv_acc_inline1938: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(kv_acc_inline1938, kv_x_chunk_bf16_inline1910, wkv_chunk_inline1933, a_trans=False, b_trans=False)
                            kv_fp32_inline1920: pl.Tensor[[t_matmul, 512], pl.FP32] = pl.tensor.assemble(kv_fp32_inline1920, kv_acc_inline1938, [t0_inline1943, kv_col0_inline1937], atomic=pl.AtomicType.Add)
                    kv_view_inline1909: pl.Tensor[[t_dim_4, 512], pl.BF16] = pl.tensor.reshape(kv_full_inline1265, [t_dim_inline1923, 512])
                    kv_token_tiles_inline1972: pl.Scalar[pl.INDEX] = (tile_rows_inline1928 + 16 - 1) // 16
                    for tg_idx_inline1916 in pl.spmd(kv_token_tiles_inline1972, name_hint="kv_rms_norm_rope_spmd"):
                        tg_inline1915: pl.Scalar[pl.INDEX] = tg_idx_inline1916 * 16
                        valid_rows_inline1931: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows_inline1928 - tg_inline1915)
                        out_tg_inline1924: pl.Scalar[pl.INDEX] = tile_base_inline1954 + tg_inline1915
                        if valid_rows_inline1931 == 16:
                            kv_sq_sum_inline1961: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                            for kv_sq_col0_inline1944 in pl.pipeline(0, 512, 64, stage=2):
                                kv_chunk_inline1941: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32_inline1920, [16, 64], [tg_inline1915, kv_sq_col0_inline1944])
                                kv_sq_inline1968: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_chunk_inline1941, kv_chunk_inline1941)
                                kv_row_sum_inline1946: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_sq_inline1968), [1, 16])
                                kv_sq_sum_inline1961: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(kv_sq_sum_inline1961, kv_row_sum_inline1946)
                            kv_inv_rms_inline1942: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(kv_sq_sum_inline1961, 0.001953125), 9.9999999999999995e-07), high_precision=True)
                            kv_inv_rms_t_inline1949: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(kv_inv_rms_inline1942, [16, 1])
                            for n0_inline1950 in pl.pipeline(0, 448, 64, stage=2):
                                kv_chunk_inline1941: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32_inline1920, [16, 64], [tg_inline1915, n0_inline1950])
                                gamma_kv_cast_inline1948: pl.Tensor[[64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_ckv, [64], [n0_inline1950]), target_type=pl.FP32, mode='round')
                                gamma_kv_chunk_inline1951: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(gamma_kv_cast_inline1948, [1, 64])
                                kv_normed_inline1925: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_chunk_inline1941, kv_inv_rms_t_inline1949), gamma_kv_chunk_inline1951)
                                kv_normed_bf16_inline1908: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(kv_normed_inline1925, target_type=pl.BF16, mode='rint')
                                kv_view_inline1909: pl.Tensor[[t_dim_4, 512], pl.BF16] = pl.tensor.assemble(kv_view_inline1909, kv_normed_bf16_inline1908, [out_tg_inline1924, n0_inline1950])
                            gamma_rope_cast_inline1955: pl.Tensor[[64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_ckv, [64], [448]), target_type=pl.FP32, mode='round')
                            gamma_rope_inline1957: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(gamma_rope_cast_inline1955, [1, 64])
                            kv_rope_chunk_inline1959: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32_inline1920, [16, 64], [tg_inline1915, 448])
                            kv_rope_norm_chunk_inline1945: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_chunk_inline1959, kv_inv_rms_t_inline1949), gamma_rope_inline1957)
                            kv_cos_il_full_inline1963: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_cos_il_inline1258, [16, 64], [out_tg_inline1924, 0])
                            kv_sin_signed_full_inline1956: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_sin_signed_inline1301, [16, 64], [out_tg_inline1924, 0])
                            kv_swap_idx_full_inline1953: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.slice(kv_swap_idx_inline1305, [16, 64], [out_tg_inline1924, 0])
                            kv_swapped_full_inline1912: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(kv_rope_norm_chunk_inline1945, kv_swap_idx_full_inline1953, dim=-1)
                            kv_rope_rot_full_inline1921: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(kv_rope_norm_chunk_inline1945, kv_cos_il_full_inline1963), pl.tensor.mul(kv_swapped_full_inline1912, kv_sin_signed_full_inline1956))
                            kv_rope_i16_full_inline1965: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(kv_rope_rot_full_inline1921, target_type=pl.BF16, mode='rint')
                            kv_view_inline1909: pl.Tensor[[t_dim_4, 512], pl.BF16] = pl.tensor.assemble(kv_view_inline1909, kv_rope_i16_full_inline1965, [out_tg_inline1924, 448])
                        else:
                            kv_reduce_tmp_inline1967: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            kv_sq_sum_tail_inline1970: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                            for kv_sq_col0_tail_inline1969 in pl.pipeline(0, 512, 64, stage=2):
                                kv_chunk_tail_inline1939: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.load(kv_fp32_inline1920, [tg_inline1915, kv_sq_col0_tail_inline1969], [16, 64], [valid_rows_inline1931, 64], target_memory=pl.Mem.Vec)
                                kv_sq_tail_inline1971: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.mul(kv_chunk_tail_inline1939, kv_chunk_tail_inline1939)
                                kv_row_sum_tail_inline1973: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_3])] = pl.tile.reshape(pl.tile.row_sum(kv_sq_tail_inline1971, kv_reduce_tmp_inline1967), [1, 16])
                                kv_sq_sum_tail_inline1970: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(kv_sq_sum_tail_inline1970, kv_row_sum_tail_inline1973)
                            kv_inv_rms_tail_inline1958: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(kv_sq_sum_tail_inline1970, 0.001953125), 9.9999999999999995e-07)))
                            kv_inv_rms_t_tail_inline1947: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_tail_inline1958, [16, 1])
                            for n0_tail_inline1964 in pl.pipeline(0, 448, 64, stage=2):
                                kv_chunk_tail_inline1939: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.load(kv_fp32_inline1920, [tg_inline1915, n0_tail_inline1964], [16, 64], [valid_rows_inline1931, 64], target_memory=pl.Mem.Vec)
                                gamma_kv_input_tail_inline1960: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(gamma_ckv, [n0_tail_inline1964], [64], [64], target_memory=pl.Mem.Vec)
                                gamma_kv_cast_tail_inline1905: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_kv_input_tail_inline1960, target_type=pl.FP32, mode='round')
                                gamma_kv_chunk_tail_inline1919: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(gamma_kv_cast_tail_inline1905, [1, 64])
                                kv_normed_tail_inline1904: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_chunk_tail_inline1939, kv_inv_rms_t_tail_inline1947), gamma_kv_chunk_tail_inline1919)
                                kv_normed_bf16_tail_inline1903: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.cast(kv_normed_tail_inline1904, target_type=pl.BF16, mode='rint')
                                kv_normed_valid_inline1902: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1903, valid_rows_inline1931, 64)
                                pl.tile.store(kv_normed_valid_inline1902, [out_tg_inline1924, n0_tail_inline1964], kv_view_inline1909)
                            gamma_rope_input_tail_inline1901: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(gamma_ckv, [448], [64], [64], target_memory=pl.Mem.Vec)
                            gamma_rope_cast_tail_inline1900: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_rope_input_tail_inline1901, target_type=pl.FP32, mode='round')
                            gamma_rope_tail_inline1899: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(gamma_rope_cast_tail_inline1900, [1, 64])
                            kv_rope_chunk_tail_inline1907: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.load(kv_fp32_inline1920, [tg_inline1915, 448], [16, 64], [valid_rows_inline1931, 64], target_memory=pl.Mem.Vec)
                            kv_rope_norm_tail_inline1898: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_rope_chunk_tail_inline1907, kv_inv_rms_t_tail_inline1947), gamma_rope_tail_inline1899)
                            kv_cos_il_tail_inline1929: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.load(kv_cos_il_inline1258, [out_tg_inline1924, 0], [16, 64], [valid_rows_inline1931, 64], target_memory=pl.Mem.Vec)
                            kv_sin_signed_tail_inline1917: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.load(kv_sin_signed_inline1301, [out_tg_inline1924, 0], [16, 64], [valid_rows_inline1931, 64], target_memory=pl.Mem.Vec)
                            kv_col_inline1897: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([16, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                            kv_dup_f_inline1952: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(kv_col_inline1897, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                            kv_lane_inline1896: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(kv_col_inline1897, pl.tile.muls(kv_dup_f_inline1952, 2.0))
                            kv_swap_f_inline1895: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(kv_col_inline1897, 1.0), pl.tile.muls(kv_lane_inline1896, 2.0))
                            kv_row_seed_inline1966: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                            kv_row_grid_inline1893: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 16], dtype=pl.FP32, value=1.0), kv_row_seed_inline1966)
                            kv_row_offset_inline1892: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(kv_row_grid_inline1893, 0, 1)
                            kv_swap_idx_tail_inline1891: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(kv_swap_f_inline1895, kv_row_offset_inline1892), target_type=pl.INT32, mode='round')
                            kv_gather_tmp_inline1890: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                            kv_swapped_tail_inline1894: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(kv_rope_norm_tail_inline1898, kv_swap_idx_tail_inline1891, kv_gather_tmp_inline1890)
                            kv_rope_rot_tail_inline1927: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.add(pl.tile.mul(kv_rope_norm_tail_inline1898, kv_cos_il_tail_inline1929), pl.tile.mul(kv_swapped_tail_inline1894, kv_sin_signed_tail_inline1917))
                            kv_rope_i16_tail_inline1889: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.cast(kv_rope_rot_tail_inline1927, target_type=pl.BF16, mode='rint')
                            kv_rope_valid_inline1906: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_3, 64])] = pl.tile.set_validshape(kv_rope_i16_tail_inline1889, valid_rows_inline1931, 64)
                            pl.tile.store(kv_rope_valid_inline1906, [out_tg_inline1924, 448], kv_view_inline1909)
            ori_block_num_inline1291: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache, 0)
            kv_cache_flat_inline1312: pl.Tensor[[ori_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(kv_cache, [ori_block_num_inline1291 * 32, 512])
            with pl.spmd(kv_wb_blocks_inline1274, name_hint="csa_cache_writeback_spmd") as ori_cache_write_tid_inline1279:
                wb_blk_inline1315: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                wb_t0_inline1317: pl.Scalar[pl.INDEX] = wb_blk_inline1315 * 8
                for write_dt_inline1318 in pl.range(8):
                    write_t_inline1308: pl.Scalar[pl.INDEX] = wb_t0_inline1317 + write_dt_inline1318
                    write_row_i64_inline1319: pl.Scalar[pl.INT64] = pl.tensor.read(ori_slot_mapping, [write_t_inline1308])
                    if write_row_i64_inline1319 >= 0:
                        write_row_inline1268: pl.Scalar[pl.INDEX] = pl.cast(write_row_i64_inline1319, pl.INDEX)
                        kv_cache_flat_inline1312: pl.Tensor[[ori_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(kv_cache_flat_inline1312, pl.tensor.slice(kv_full_inline1265, [1, 512], [write_t_inline1308, 0]), [write_row_inline1268, 0])
            cmp_positions_inline1320: pl.Tensor[[kv_dim], pl.INT32] = pl.tensor.reshape(position_ids, [kv_dim_inline1261])
            cmp_slots_inline1296: pl.Tensor[[kv_dim], pl.INT64] = pl.tensor.reshape(cmp_slot_mapping, [kv_dim_inline1261])
            cmp_state_slots_inline1247: pl.Tensor[[kv_dim], pl.INT64] = pl.tensor.reshape(state_slot_mapping, [kv_dim_inline1261])
            idx_slots_inline1322: pl.Tensor[[kv_dim], pl.INT64] = pl.tensor.reshape(idx_slot_mapping, [kv_dim_inline1261])
            idx_positions_inline1323: pl.Tensor[[t_dim], pl.INT32] = pl.tensor.reshape(position_ids_local, [t_dim_inline1251])
            inner_state_slots_inline1257: pl.Tensor[[kv_dim], pl.INT64] = pl.tensor.reshape(inner_state_slot_mapping, [kv_dim_inline1261])
            cmp_state_table_inline1275: pl.Tensor[[kv_b_dim, 4], pl.INT32] = pl.tensor.reshape(compress_state_block_table, [kv_b_dim_inline1264, 4])
            inner_state_table_inline1324: pl.Tensor[[kv_b_dim, 4], pl.INT32] = pl.tensor.reshape(inner_compress_state_block_table, [kv_b_dim_inline1264, 4])
            cmp_out_inline1299: pl.Tensor[[kv_dim, 512], pl.FP32] = pl.tensor.create([kv_dim_inline1261, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            b_dim_inline2018: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_state_table_inline1275, 0)
            bs_inline2038: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240, 0)
            s_dim_inline2020: pl.Scalar[pl.INDEX] = bs_inline2038 // b_dim_inline2018
            t_matmul_inline2013: pl.Scalar[pl.INDEX] = (bs_inline2038 + 16 - 1) // 16 * 16
            rms_blocks_inline2061: pl.Scalar[pl.INDEX] = (bs_inline2038 + 16 - 1) // 16
            x_flat_inline2021: pl.Tensor[[KV_T_DYN, 4096], pl.BF16] = x_normed_full_inline1240
            cmp4_kv_proj_pad_inline2031: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.create([512, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            cmp4_score_proj_pad_inline2019: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.create([512, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            compress_state_block_num_inline2051: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state, 0)
            cmp_block_num_inline2055: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv, 0)
            compress_state_flat_inline2023: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.reshape(compress_state, [compress_state_block_num_inline2051 * 2, 2048])
            kv_flat_inline2039: pl.Tensor[[KV_T_DYN, 512], pl.FP32] = cmp_out_inline1299
            cmp_kv_cache_flat_inline2036: pl.Tensor[[cmp_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(cmp_kv, [cmp_block_num_inline2055 * 32, 512])
            with pl.spmd(t_matmul_inline2013 * 1024 // 1024, name_hint="kv_score_proj_spmd", deps=[late_dep_inline1297]) as _kv_score_tid_inline2015:
                idx_inline2012: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                global_row0_inline2011: pl.Scalar[pl.INDEX] = idx_inline2012 // 16 * 16
                o0_inline2017: pl.Scalar[pl.INDEX] = idx_inline2012 % 16 * 64
                kv_acc_inline2010: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                score_acc_inline2014: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for kb_inline2001 in pl.pipeline(8, stage=2):
                    k0_inline2035: pl.Scalar[pl.INDEX] = kb_inline2001 * 512
                    x_rows_inline2000: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2038 - global_row0_inline2011)
                    x_tile_inline2033: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[x_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline2021, [16, 512], [global_row0_inline2011, k0_inline2035], [x_rows_inline2000, 512])
                    wkv_tile_inline2007: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(cmp_wkv, [64, 512], [o0_inline2017, k0_inline2035])
                    wgate_tile_inline1998: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(cmp_wgate, [64, 512], [o0_inline2017, k0_inline2035])
                    if k0_inline2035 == 0:
                        kv_acc_inline2010: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile_inline2033, wkv_tile_inline2007, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        score_acc_inline2014: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile_inline2033, wgate_tile_inline1998, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                    else:
                        kv_acc_inline2010: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(kv_acc_inline2010, x_tile_inline2033, wkv_tile_inline2007, a_trans=False, b_trans=True)
                        score_acc_inline2014: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(score_acc_inline2014, x_tile_inline2033, wgate_tile_inline1998, a_trans=False, b_trans=True)
                cmp4_kv_proj_pad_inline2031: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.assemble(cmp4_kv_proj_pad_inline2031, kv_acc_inline2010, [global_row0_inline2011, o0_inline2017])
                cmp4_score_proj_pad_inline2019: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.assemble(cmp4_score_proj_pad_inline2019, score_acc_inline2014, [global_row0_inline2011, o0_inline2017])
            pooled_kv_inline2008: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.create([512, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(b_dim_inline2018, name_hint="scatter_softmax_pool_spmd", deps=[_kv_score_tid_inline2015]) as pool_tid_inline1994:
                c_idx_inline2049: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                first_pos_b_inline2028: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320, [c_idx_inline2049 * s_dim_inline2020])
                for s_idx_inline2026 in pl.range(s_dim_inline2020):
                    token_inline2040: pl.Scalar[pl.INDEX] = c_idx_inline2049 * s_dim_inline2020 + s_idx_inline2026
                    token_pos_inline2041: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320, [token_inline2040])
                    pooled_kv_inline2008: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(pooled_kv_inline2008, pl.tensor.full([1, 512], dtype=pl.FP32, value=0.0), [token_inline2040, 0])
                    if (pl.cast(token_pos_inline2041, pl.INDEX) + 1) % 4 == 0:
                        window_start_inline1995: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline2041, pl.INDEX) - 8 + 1
                        for h0_inline2042 in pl.range(0, 512, 64):
                            last_ape_row_inline2044: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2041, pl.INDEX) % 4, pl.INDEX)
                            mi_inline2045: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad_inline2019, [1, 64], [token_inline2040, 512 + h0_inline2042]), pl.tensor.slice(cmp_ape, [1, 64], [last_ape_row_inline2044, 512 + h0_inline2042]))
                            li_inline2029: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline2045, mi_inline2045))
                            oi_inline2046: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cmp4_kv_proj_pad_inline2031, [1, 64], [token_inline2040, 512 + h0_inline2042])
                            for state_idx_inline2032 in pl.range(7):
                                logical_pos_inline2022: pl.Scalar[pl.INDEX] = window_start_inline1995 + state_idx_inline2032
                                value_inline2050: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0)
                                score_inline2003: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                                state_half_inline2052: pl.Scalar[pl.INDEX] = 0
                                if state_idx_inline2032 >= 4:
                                    state_half_inline2052: pl.Scalar[pl.INDEX] = 512
                                if logical_pos_inline2022 >= 0 and logical_pos_inline2022 < pl.cast(first_pos_b_inline2028, pl.INDEX):
                                    ring_row_inline2004: pl.Scalar[pl.INDEX] = logical_pos_inline2022 % 8
                                    state_page_off_inline2054: pl.Scalar[pl.INDEX] = ring_row_inline2004 // 2
                                    state_blk_id_i32_inline1996: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_state_table_inline1275, [c_idx_inline2049, state_page_off_inline2054])
                                    if pl.cast(state_blk_id_i32_inline1996, pl.INDEX) >= 0:
                                        state_blk_id_inline2056: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline1996, pl.INDEX)
                                        state_row_inline2058: pl.Scalar[pl.INDEX] = state_blk_id_inline2056 * 2 + ring_row_inline2004 % 2
                                        value_inline2050: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat_inline2023, [1, 64], [state_row_inline2058, state_half_inline2052 + h0_inline2042])
                                        score_inline2003: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat_inline2023, [1, 64], [state_row_inline2058, 1024 + state_half_inline2052 + h0_inline2042])
                                if logical_pos_inline2022 >= pl.cast(first_pos_b_inline2028, pl.INDEX):
                                    if logical_pos_inline2022 <= pl.cast(token_pos_inline2041, pl.INDEX):
                                        overlay_token_inline2005: pl.Scalar[pl.INDEX] = c_idx_inline2049 * s_dim_inline2020 + logical_pos_inline2022 - pl.cast(first_pos_b_inline2028, pl.INDEX)
                                        ape_row_inline2034: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline2022 % 4, pl.INDEX)
                                        value_inline2050: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cmp4_kv_proj_pad_inline2031, [1, 64], [overlay_token_inline2005, state_half_inline2052 + h0_inline2042])
                                        score_inline2003: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad_inline2019, [1, 64], [overlay_token_inline2005, state_half_inline2052 + h0_inline2042]), pl.tensor.slice(cmp_ape, [1, 64], [ape_row_inline2034, state_half_inline2052 + h0_inline2042]))
                                mi_next_inline2059: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(mi_inline2045, score_inline2003)
                                alpha_inline2027: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline2045, mi_next_inline2059))
                                beta_inline1999: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(score_inline2003, mi_next_inline2059))
                                li_inline2029: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha_inline2027, li_inline2029), beta_inline1999)
                                oi_inline2046: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(oi_inline2046, alpha_inline2027), pl.tensor.mul(value_inline2050, beta_inline1999))
                                mi_inline2045: pl.Tensor[[1, 64], pl.FP32] = mi_next_inline2059
                            pooled_kv_inline2008: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(pooled_kv_inline2008, pl.tensor.div(oi_inline2046, li_inline2029), [token_inline2040, h0_inline2042])
            for c_idx_v1_inline2030 in pl.spmd(b_dim_inline2018, name_hint="compress_state_commit_spmd", deps=[pool_tid_inline1994]):
                for s_idx_inline2009 in pl.range(s_dim_inline2020):
                    token_inline2040: pl.Scalar[pl.INDEX] = c_idx_v1_inline2030 * s_dim_inline2020 + s_idx_inline2009
                    state_row_i64_inline2006: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_state_slots_inline1247, [token_inline2040])
                    if state_row_i64_inline2006 >= 0:
                        state_row_inline2058: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline2006, pl.INDEX)
                        token_pos_inline2041: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320, [token_inline2040])
                        ape_row_inline2034: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2041, pl.INDEX) % 4, pl.INDEX)
                        compress_state_flat_inline2023: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.assemble(compress_state_flat_inline2023, pl.tensor.slice(cmp4_kv_proj_pad_inline2031, [1, 1024], [token_inline2040, 0]), [state_row_inline2058, 0])
                        compress_state_flat_inline2023: pl.Tensor[[compress_state_block_num * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.assemble(compress_state_flat_inline2023, pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad_inline2019, [1, 1024], [token_inline2040, 0]), pl.tensor.slice(cmp_ape, [1, 1024], [ape_row_inline2034, 0])), [state_row_inline2058, 1024])
            normed_kv_inline2016: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.create([512, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            norm_w_2d_inline2060: pl.Tensor[[1, 512], pl.BF16] = pl.tensor.reshape(cmp_norm_w, [1, 512])
            with pl.spmd(rms_blocks_inline2061, name_hint="rmsnorm_rope_cache_write_spmd", deps=[pool_tid_inline1994]) as cache_write_tid_inline2062:
                rms_blk_inline1993: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                b0_inline1992: pl.Scalar[pl.INDEX] = rms_blk_inline1993 * 16
                rms_blk_rows_inline1991: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2038 - b0_inline1992)
                cos_b_inline2057: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cmp_cos_il_full_inline1249, [16, 64], [b0_inline1992, 0], [rms_blk_rows_inline1991, 64])
                sin_b_inline1989: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cmp_sin_signed_full_inline1263, [16, 64], [b0_inline1992, 0], [rms_blk_rows_inline1991, 64])
                partial_sq_inline2025: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                for k0_inline1987 in pl.range(0, 512, 64):
                    kv_rms_chunk_inline1986: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2008, [16, 64], [b0_inline1992, k0_inline1987])
                    kv_rms_sq_inline1988: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_rms_chunk_inline1986, kv_rms_chunk_inline1986)
                    kv_rms_rowsum_inline2053: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_rms_sq_inline1988), [1, 16])
                    partial_sq_inline2025: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq_inline2025, kv_rms_rowsum_inline2053)
                variance_inline1990: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq_inline2025, 0.001953125), 9.9999999999999995e-07), [16, 1])
                inv_rms_inline1984: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance_inline1990))
                for k0_inline1983 in pl.range(0, 448, 64):
                    kv_norm_chunk_inline2037: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2008, [16, 64], [b0_inline1992, k0_inline1983])
                    gamma_inline2024: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d_inline2060, [1, 64], [0, k0_inline1983]), target_type=pl.FP32, mode='round')
                    normed_chunk_inline1982: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_norm_chunk_inline2037, inv_rms_inline1984), gamma_inline2024)
                    normed_kv_inline2016: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(normed_kv_inline2016, normed_chunk_inline1982, [b0_inline1992, k0_inline1983])
                kv_rope_norm_inline1981: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2008, [16, 64], [b0_inline1992, 448])
                gamma_rope_inline1980: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d_inline2060, [1, 64], [0, 448]), target_type=pl.FP32, mode='round')
                rope_normed_inline1979: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_norm_inline1981, inv_rms_inline1984), gamma_rope_inline1980)
                rope_ones_inline1978: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
                rope_col_inline1977: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(rope_ones_inline1978, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                rope_dup_f_inline1976: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(rope_col_inline1977, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                rope_lane_inline2048: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(rope_col_inline1977, pl.tensor.muls(rope_dup_f_inline1976, 2.0))
                rope_swap_idx_inline1997: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(rope_col_inline1977, 1.0), pl.tensor.muls(rope_lane_inline2048, 2.0)), target_type=pl.INT32, mode='round')
                swapped_inline1975: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(rope_normed_inline1979, rope_swap_idx_inline1997, dim=-1)
                rope_rot_inline2002: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(rope_normed_inline1979, cos_b_inline2057), pl.tensor.mul(swapped_inline1975, sin_b_inline1989))
                normed_kv_inline2016: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(normed_kv_inline2016, rope_rot_inline2002, [b0_inline1992, 448])
                for inner_inline1985 in pl.range(rms_blk_rows_inline1991):
                    token_inline2040: pl.Scalar[pl.INDEX] = b0_inline1992 + inner_inline1985
                    cache_row_i64_inline1974: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_slots_inline1296, [token_inline2040])
                    if cache_row_i64_inline1974 >= 0:
                        cache_row_inline2047: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline1974, pl.INDEX)
                        kv_row_fp32_inline2043: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.slice(normed_kv_inline2016, [1, 512], [token_inline2040, 0])
                        kv_flat_inline2039: pl.Tensor[[KV_T_DYN, 512], pl.FP32] = pl.tensor.assemble(kv_flat_inline2039, kv_row_fp32_inline2043, [token_inline2040, 0])
                        cmp_kv_cache_flat_inline2036: pl.Tensor[[cmp_block_num * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(cmp_kv_cache_flat_inline2036, pl.tensor.cast(kv_row_fp32_inline2043, target_type=pl.BF16, mode='rint'), [cache_row_inline2047, 0])
            cmp_out_inline1299: pl.Tensor[[kv_dim, 512], pl.FP32] = cmp_out_inline1299
            cmp_cache_write_tid_inline1237: pl.Scalar[pl.TASK_ID] = cache_write_tid_inline2062
            cache_ready_dep_inline1304: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[ori_cache_write_tid_inline1279, cmp_cache_write_tid_inline1237])
            idx_kv_unused_inline1241: pl.Tensor[[kv_dim, 128], pl.FP32] = pl.tensor.create([kv_dim_inline1261, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            b_dim_inline2158: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_state_table_inline1324, 0)
            bs_inline2140: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240, 0)
            s_dim_inline2127: pl.Scalar[pl.INDEX] = bs_inline2140 // b_dim_inline2158
            t_matmul_inline2119: pl.Scalar[pl.INDEX] = (bs_inline2140 + 16 - 1) // 16 * 16
            rms_blocks_inline2124: pl.Scalar[pl.INDEX] = (bs_inline2140 + 16 - 1) // 16
            x_flat_inline2162: pl.Tensor[[KV_T_DYN, 4096], pl.BF16] = x_normed_full_inline1240
            kv_proj_pad_inline2129: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.create([512, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            score_proj_pad_inline2143: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.create([512, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            compress_state_block_num_inline2109: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_compress_state, 0)
            idx_block_num_inline2101: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_kv_cache, 0)
            compress_state_flat_inline2139: pl.Tensor[[compress_state_block_num_1 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.reshape(inner_compress_state, [compress_state_block_num_inline2109 * 2, 512])
            kv_flat_inline2098: pl.Tensor[[KV_T_DYN, 128], pl.FP32] = idx_kv_unused_inline1241
            idx_kv_cache_flat_inline2161: pl.Tensor[[idx_block_num * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.reshape(idx_kv_cache, [idx_block_num_inline2101 * 32, 128])
            idx_kv_scale_flat_inline2116: pl.Tensor[[idx_block_num * pl.const(32, pl.INDEX), 1], pl.FP32] = pl.tensor.reshape(idx_kv_scale, [idx_block_num_inline2101 * 32, 1])
            with pl.spmd(t_matmul_inline2119 * 256 // 512, name_hint="kv_score_proj_spmd", deps=[late_dep_inline1297]) as _kv_score_tid_inline2110:
                idx_inline2108: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                global_row0_inline2104: pl.Scalar[pl.INDEX] = idx_inline2108 // 8 * 16
                o0_inline2117: pl.Scalar[pl.INDEX] = idx_inline2108 % 8 * 32
                kv_acc_inline2107: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                score_acc_inline2103: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for kb_inline2105 in pl.pipeline(8, stage=2):
                    k0_inline2172: pl.Scalar[pl.INDEX] = kb_inline2105 * 512
                    x_rows_inline2097: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2140 - global_row0_inline2104)
                    x_tile_inline2096: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[x_rows_1, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline2162, [16, 512], [global_row0_inline2104, k0_inline2172], [x_rows_inline2097, 512])
                    wkv_tile_inline2114: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(inner_wkv, [32, 512], [o0_inline2117, k0_inline2172])
                    wgate_tile_inline2100: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(inner_wgate, [32, 512], [o0_inline2117, k0_inline2172])
                    if k0_inline2172 == 0:
                        kv_acc_inline2107: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_tile_inline2096, wkv_tile_inline2114, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        score_acc_inline2103: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_tile_inline2096, wgate_tile_inline2100, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                    else:
                        kv_acc_inline2107: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(kv_acc_inline2107, x_tile_inline2096, wkv_tile_inline2114, a_trans=False, b_trans=True)
                        score_acc_inline2103: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(score_acc_inline2103, x_tile_inline2096, wgate_tile_inline2100, a_trans=False, b_trans=True)
                kv_proj_pad_inline2129: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.assemble(kv_proj_pad_inline2129, kv_acc_inline2107, [global_row0_inline2104, o0_inline2117])
                score_proj_pad_inline2143: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.assemble(score_proj_pad_inline2143, score_acc_inline2103, [global_row0_inline2104, o0_inline2117])
            pooled_kv_inline2131: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.create([512, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(b_dim_inline2158, name_hint="scatter_softmax_pool_spmd", deps=[_kv_score_tid_inline2110]) as pool_tid_inline2128:
                c_idx_inline2094: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                first_pos_b_inline2144: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320, [c_idx_inline2094 * s_dim_inline2127])
                for s_idx_inline2147 in pl.range(s_dim_inline2127):
                    token_inline2123: pl.Scalar[pl.INDEX] = c_idx_inline2094 * s_dim_inline2127 + s_idx_inline2147
                    token_pos_inline2120: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320, [token_inline2123])
                    pooled_kv_inline2131: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(pooled_kv_inline2131, pl.tensor.full([1, 128], dtype=pl.FP32, value=0.0), [token_inline2123, 0])
                    if (pl.cast(token_pos_inline2120, pl.INDEX) + 1) % 4 == 0:
                        window_start_inline2142: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline2120, pl.INDEX) - 8 + 1
                        for h0_inline2125 in pl.range(0, 128, 64):
                            last_ape_row_inline2113: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2120, pl.INDEX) % 4, pl.INDEX)
                            mi_inline2145: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(score_proj_pad_inline2143, [1, 64], [token_inline2123, 128 + h0_inline2125]), pl.tensor.slice(inner_ape, [1, 64], [last_ape_row_inline2113, 128 + h0_inline2125]))
                            li_inline2148: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline2145, mi_inline2145))
                            oi_inline2132: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(kv_proj_pad_inline2129, [1, 64], [token_inline2123, 128 + h0_inline2125])
                            for state_idx_inline2149 in pl.range(7):
                                logical_pos_inline2150: pl.Scalar[pl.INDEX] = window_start_inline2142 + state_idx_inline2149
                                value_inline2163: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0)
                                score_inline2156: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                                state_half_inline2160: pl.Scalar[pl.INDEX] = 0
                                if state_idx_inline2149 >= 4:
                                    state_half_inline2160: pl.Scalar[pl.INDEX] = 128
                                if logical_pos_inline2150 >= 0 and logical_pos_inline2150 < pl.cast(first_pos_b_inline2144, pl.INDEX):
                                    ring_row_inline2159: pl.Scalar[pl.INDEX] = logical_pos_inline2150 % 8
                                    state_page_off_inline2154: pl.Scalar[pl.INDEX] = ring_row_inline2159 // 2
                                    state_blk_id_i32_inline2130: pl.Scalar[pl.INT32] = pl.tensor.read(inner_state_table_inline1324, [c_idx_inline2094, state_page_off_inline2154])
                                    if pl.cast(state_blk_id_i32_inline2130, pl.INDEX) >= 0:
                                        state_blk_id_inline2168: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline2130, pl.INDEX)
                                        state_row_inline2166: pl.Scalar[pl.INDEX] = state_blk_id_inline2168 * 2 + ring_row_inline2159 % 2
                                        value_inline2163: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat_inline2139, [1, 64], [state_row_inline2166, state_half_inline2160 + h0_inline2125])
                                        score_inline2156: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat_inline2139, [1, 64], [state_row_inline2166, 256 + state_half_inline2160 + h0_inline2125])
                                if logical_pos_inline2150 >= pl.cast(first_pos_b_inline2144, pl.INDEX):
                                    if logical_pos_inline2150 <= pl.cast(token_pos_inline2120, pl.INDEX):
                                        overlay_token_inline2169: pl.Scalar[pl.INDEX] = c_idx_inline2094 * s_dim_inline2127 + logical_pos_inline2150 - pl.cast(first_pos_b_inline2144, pl.INDEX)
                                        ape_row_inline2136: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline2150 % 4, pl.INDEX)
                                        value_inline2163: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(kv_proj_pad_inline2129, [1, 64], [overlay_token_inline2169, state_half_inline2160 + h0_inline2125])
                                        score_inline2156: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(score_proj_pad_inline2143, [1, 64], [overlay_token_inline2169, state_half_inline2160 + h0_inline2125]), pl.tensor.slice(inner_ape, [1, 64], [ape_row_inline2136, state_half_inline2160 + h0_inline2125]))
                                mi_next_inline2126: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(mi_inline2145, score_inline2156)
                                alpha_inline2115: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline2145, mi_next_inline2126))
                                beta_inline2170: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(score_inline2156, mi_next_inline2126))
                                li_inline2148: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha_inline2115, li_inline2148), beta_inline2170)
                                oi_inline2132: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(oi_inline2132, alpha_inline2115), pl.tensor.mul(value_inline2163, beta_inline2170))
                                mi_inline2145: pl.Tensor[[1, 64], pl.FP32] = mi_next_inline2126
                            pooled_kv_inline2131: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(pooled_kv_inline2131, pl.tensor.div(oi_inline2132, li_inline2148), [token_inline2123, h0_inline2125])
            for c_idx_v1_inline2153 in pl.spmd(b_dim_inline2158, name_hint="compress_state_commit_spmd", deps=[pool_tid_inline2128]):
                for s_idx_inline2121 in pl.range(s_dim_inline2127):
                    token_inline2123: pl.Scalar[pl.INDEX] = c_idx_v1_inline2153 * s_dim_inline2127 + s_idx_inline2121
                    state_row_i64_inline2137: pl.Scalar[pl.INT64] = pl.tensor.read(inner_state_slots_inline1257, [token_inline2123])
                    if state_row_i64_inline2137 >= 0:
                        state_row_inline2166: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline2137, pl.INDEX)
                        token_pos_inline2120: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320, [token_inline2123])
                        ape_row_inline2136: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2120, pl.INDEX) % 4, pl.INDEX)
                        compress_state_flat_inline2139: pl.Tensor[[compress_state_block_num_1 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.assemble(compress_state_flat_inline2139, pl.tensor.slice(kv_proj_pad_inline2129, [1, 256], [token_inline2123, 0]), [state_row_inline2166, 0])
                        compress_state_flat_inline2139: pl.Tensor[[compress_state_block_num_1 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.assemble(compress_state_flat_inline2139, pl.tensor.add(pl.tensor.slice(score_proj_pad_inline2143, [1, 256], [token_inline2123, 0]), pl.tensor.slice(inner_ape, [1, 256], [ape_row_inline2136, 0])), [state_row_inline2166, 256])
            normed_kv_inline2164: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.create([512, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            norm_w_2d_inline2173: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.reshape(inner_norm_w, [1, 128])
            with pl.spmd(rms_blocks_inline2124, name_hint="rmsnorm_rope_spmd", deps=[pool_tid_inline2128]) as rms_tid_inline2093:
                rms_blk_inline2092: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                b0_inline2090: pl.Scalar[pl.INDEX] = rms_blk_inline2092 * 16
                rms_blk_rows_inline2141: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2140 - b0_inline2090)
                cos_b_inline2087: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows_1, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cmp_cos_il_full_inline1249, [16, 64], [b0_inline2090, 0], [rms_blk_rows_inline2141, 64])
                sin_b_inline2085: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows_1, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cmp_sin_signed_full_inline1263, [16, 64], [b0_inline2090, 0], [rms_blk_rows_inline2141, 64])
                partial_sq_inline2155: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                for k0_inline2152 in pl.pipeline(0, 128, 64, stage=2):
                    kv_rms_chunk_inline2084: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2131, [16, 64], [b0_inline2090, k0_inline2152])
                    kv_rms_sq_inline2083: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_rms_chunk_inline2084, kv_rms_chunk_inline2084)
                    kv_rms_rowsum_inline2165: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_rms_sq_inline2083), [1, 16])
                    partial_sq_inline2155: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq_inline2155, kv_rms_rowsum_inline2165)
                variance_inline2082: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq_inline2155, 0.0078125), 9.9999999999999995e-07), [16, 1])
                inv_rms_inline2122: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance_inline2082))
                for k0_inline2081 in pl.pipeline(0, 64, 64, stage=2):
                    kv_norm_chunk_inline2080: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2131, [16, 64], [b0_inline2090, k0_inline2081])
                    gamma_inline2078: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d_inline2173, [1, 64], [0, k0_inline2081]), target_type=pl.FP32, mode='round')
                    normed_chunk_inline2102: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_norm_chunk_inline2080, inv_rms_inline2122), gamma_inline2078)
                    normed_kv_inline2164: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.assemble(normed_kv_inline2164, pl.tensor.cast(normed_chunk_inline2102, target_type=pl.BF16, mode='rint'), [b0_inline2090, k0_inline2081])
                kv_rope_norm_inline2135: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2131, [16, 64], [b0_inline2090, 64])
                gamma_rope_inline2077: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d_inline2173, [1, 64], [0, 64]), target_type=pl.FP32, mode='round')
                rope_normed_inline2138: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_norm_inline2135, inv_rms_inline2122), gamma_rope_inline2077)
                rope_ones_inline2089: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
                rope_col_inline2076: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(rope_ones_inline2089, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                rope_dup_f_inline2074: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(rope_col_inline2076, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                rope_lane_inline2073: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(rope_col_inline2076, pl.tensor.muls(rope_dup_f_inline2074, 2.0))
                rope_swap_idx_inline2079: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(rope_col_inline2076, 1.0), pl.tensor.muls(rope_lane_inline2073, 2.0)), target_type=pl.INT32, mode='round')
                swapped_inline2071: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(rope_normed_inline2138, rope_swap_idx_inline2079, dim=-1)
                rope_rot_inline2070: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(rope_normed_inline2138, cos_b_inline2087), pl.tensor.mul(swapped_inline2071, sin_b_inline2085))
                normed_kv_inline2164: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.assemble(normed_kv_inline2164, pl.tensor.cast(rope_rot_inline2070, target_type=pl.BF16, mode='rint'), [b0_inline2090, 64])
            kv_final_inline2118: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.create([512, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(rms_blocks_inline2124, name_hint="kv_hadamard_spmd", deps=[rms_tid_inline2093]) as hadamard_tid_inline2075:
                had_blk_inline2088: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                had_b0_inline2069: pl.Scalar[pl.INDEX] = had_blk_inline2088 * 16
                kv_proj_tile_inline2072: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.slice(normed_kv_inline2164, [16, 128], [had_b0_inline2069, 0])
                for o0_inline2068 in pl.range(0, 128, 64):
                    hadamard_tile_inline2134: pl.Tensor[[128, 64], pl.BF16] = pl.tensor.slice(hadamard_idx, [128, 64], [0, o0_inline2068])
                    kv_hadamard_acc_inline2151: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(kv_proj_tile_inline2072, hadamard_tile_inline2134, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    kv_final_inline2118: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(kv_final_inline2118, kv_hadamard_acc_inline2151, [had_b0_inline2069, o0_inline2068])
            with pl.spmd(rms_blocks_inline2124, name_hint="kv_and_cache_write_spmd", deps=[hadamard_tid_inline2075]) as _write_tid_inline2106:
                wr_blk_inline2171: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                wr_b0_inline2157: pl.Scalar[pl.INDEX] = wr_blk_inline2171 * 16
                wr_blk_rows_inline2167: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2140 - wr_b0_inline2157)
                kv_blk_f32_inline2067: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.slice(kv_final_inline2118, [16, 128], [wr_b0_inline2157, 0]), target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                kv_amax_inline2091: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(pl.tensor.abs(kv_blk_f32_inline2067)), [1, 16])
                kv_amax_v1_inline2066: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.maximum(kv_amax_inline2091, pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0001))
                kv_scale_q_row_inline2112: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 16], dtype=pl.FP32, value=127.0), kv_amax_v1_inline2066)
                kv_scale_dq_col_inline2065: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(kv_scale_q_row_inline2112), [16, 1])
                kv_scale_q_col_inline2086: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(kv_scale_q_row_inline2112, [16, 1])
                kv_scaled_inline2111: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(kv_blk_f32_inline2067, kv_scale_q_col_inline2086)
                kv_i32_inline2133: pl.Tensor[[16, 128], pl.INT32] = pl.tensor.cast(kv_scaled_inline2111, target_type=pl.INT32, mode='rint')
                kv_half_inline2064: pl.Tensor[[16, 128], pl.FP16] = pl.tensor.cast(kv_i32_inline2133, target_type=pl.FP16, mode='round')
                kv_i8_blk_inline2095: pl.Tensor[[16, 128], pl.INT8] = pl.tensor.cast(kv_half_inline2064, target_type=pl.INT8, mode='trunc')
                for inner_inline2063 in pl.range(wr_blk_rows_inline2167):
                    token_inline2123: pl.Scalar[pl.INDEX] = wr_b0_inline2157 + inner_inline2063
                    cache_row_i64_inline2099: pl.Scalar[pl.INT64] = pl.tensor.read(idx_slots_inline1322, [token_inline2123])
                    if cache_row_i64_inline2099 >= 0:
                        cache_row_inline2146: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline2099, pl.INDEX)
                        kv_flat_inline2098: pl.Tensor[[KV_T_DYN, 128], pl.FP32] = pl.tensor.assemble(kv_flat_inline2098, pl.tensor.slice(kv_final_inline2118, [1, 128], [token_inline2123, 0]), [token_inline2123, 0])
                        idx_kv_cache_flat_inline2161: pl.Tensor[[idx_block_num * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.assemble(idx_kv_cache_flat_inline2161, pl.tensor.slice(kv_i8_blk_inline2095, [1, 128], [inner_inline2063, 0]), [cache_row_inline2146, 0])
                        pl.tensor.write(idx_kv_scale_flat_inline2116, [cache_row_inline2146, 0], pl.tensor.read(kv_scale_dq_col_inline2065, [inner_inline2063, 0]))
            idx_cache_write_tid_inline1316: pl.Scalar[pl.TASK_ID] = _write_tid_inline2106
            bs_inline2301: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243, 0)
            bs_heads_inline2228: pl.Scalar[pl.INDEX] = bs_inline2301 * 64
            row_blocks_inline2258: pl.Scalar[pl.INDEX] = (bs_inline2301 + 16 - 1) // 16
            qr_acc_pad_inline2225: pl.Tensor[[256, 8192], pl.INT32] = pl.tensor.create([256, 8192], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            for qr_unit_inline2215 in pl.spmd(8 * row_blocks_inline2258, name_hint="idx_qr_proj_matmul_spmd", allow_early_resolve=True):
                qr_rb_inline2194: pl.Scalar[pl.INDEX] = qr_unit_inline2215 // 8
                ot_inline2200: pl.Scalar[pl.INDEX] = qr_unit_inline2215 - qr_rb_inline2194 * 8
                qr_r0_inline2195: pl.Scalar[pl.INDEX] = qr_rb_inline2194 * 16
                qr_rows_inline2208: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2301 - qr_r0_inline2195)
                o_base_inline2220: pl.Scalar[pl.INDEX] = ot_inline2200 * 1024
                for ns_inline2239 in pl.range(0, 1024, 512):
                    qr_acc_inline2212: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.create([16, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    for kb_inline2210 in pl.pipeline(4, stage=2):
                        q0_inline2232: pl.Scalar[pl.INDEX] = kb_inline2210 * 256
                        qr_tile_inline2201: pl.Tensor[[16, 256], pl.INT8, pl.TensorView(valid_shape=[qr_rows_1, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr_inline1255, [16, 256], [qr_r0_inline2195, q0_inline2232], [qr_rows_inline2208, 256])
                        wq_tile_inline2230: pl.Tensor[[256, 512], pl.INT8] = pl.tensor.slice(idx_wq_b, [256, 512], [q0_inline2232, o_base_inline2220 + ns_inline2239])
                        if q0_inline2232 == 0:
                            qr_acc_inline2212: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul(qr_tile_inline2201, wq_tile_inline2230, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                        else:
                            qr_acc_inline2212: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul_acc(qr_acc_inline2212, qr_tile_inline2201, wq_tile_inline2230, a_trans=False, b_trans=False)
                    qr_acc_pad_inline2225: pl.Tensor[[256, 8192], pl.INT32] = pl.tensor.assemble(qr_acc_pad_inline2225, qr_acc_inline2212, [qr_r0_inline2195, o_base_inline2220 + ns_inline2239])
            qr_proj_inline2268: pl.Tensor[[bs, 8192], pl.FP32] = pl.tensor.create([bs_inline2301, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for ot_inline2284 in pl.spmd(8, name_hint="idx_qr_proj_dequant_spmd", allow_early_resolve=True):
                o_base_inline2220: pl.Scalar[pl.INDEX] = ot_inline2284 * 1024
                wq_scale_inline2262: pl.Tensor[[1, 1024], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(idx_wq_b_scale, [1024], [o_base_inline2220]), [1, 1024])
                for dq_t0_inline2203 in pl.range(0, bs_inline2301, 8):
                    acc_fp32_inline2257: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.cast(pl.tensor.slice(qr_acc_pad_inline2225, [8, 1024], [dq_t0_inline2203, o_base_inline2220]), target_type=pl.FP32, mode='none')
                    qr_scale_tile_inline2236: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(qr_scale_inline1310, [8, 1], [dq_t0_inline2203, 0])
                    qr_dequant_inline2240: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(acc_fp32_inline2257, qr_scale_tile_inline2236), wq_scale_inline2262)
                    qr_proj_inline2268: pl.Tensor[[bs, 8192], pl.FP32] = pl.tensor.assemble(qr_proj_inline2268, qr_dequant_inline2240, [dq_t0_inline2203, o_base_inline2220])
            qr_proj_flat_inline2295: pl.Tensor[[bs_heads, 128], pl.FP32] = pl.tensor.reshape(qr_proj_inline2268, [bs_heads_inline2228, 128])
            qr_bf16_inline2223: pl.Tensor[[bs_heads, 128], pl.BF16] = pl.tensor.create([bs_heads_inline2228, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            rope_swap_idx_t_inline2189: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.create([32, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rope_swap_idx", allow_early_resolve=True):
                sw_col_inline2204: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.full([32, 64], dtype=pl.FP32, value=1.0), pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                sw_dup_f_inline2263: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(sw_col_inline2204, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                sw_lane_inline2241: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.sub(sw_col_inline2204, pl.tensor.muls(sw_dup_f_inline2263, 2.0))
                rope_swap_idx_t_inline2189: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_t_inline2189, pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(sw_col_inline2204, 1.0), pl.tensor.muls(sw_lane_inline2241, 2.0)), target_type=pl.INT32, mode='round'), [0, 0])
            for idx_inline2276 in pl.spmd(bs_heads_inline2228 // 32, name_hint="qr_rope_spmd", allow_early_resolve=True):
                o0_inline2221: pl.Scalar[pl.INDEX] = idx_inline2276 * 32
                token_idx_inline2272: pl.Scalar[pl.INDEX] = o0_inline2221 // 64
                rope_swap_idx_inline2190: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.slice(rope_swap_idx_t_inline2189, [32, 64], [0, 0])
                cos_row_inline2192: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(idx_cos_il_inline1282, [1, 64], [token_idx_inline2272, 0])
                sin_row_inline2231: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(idx_sin_signed_inline1307, [1, 64], [token_idx_inline2272, 0])
                qr_nope_slice_inline2206: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(qr_proj_flat_inline2295, [32, 64], [o0_inline2221, 0])
                qr_rope_slice_inline2222: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(qr_proj_flat_inline2295, [32, 64], [o0_inline2221, 64])
                qr_swapped_inline2205: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.gather(qr_rope_slice_inline2222, rope_swap_idx_inline2190, dim=-1)
                rope_rot_inline2185: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(qr_rope_slice_inline2222, cos_row_inline2192), pl.tensor.col_expand_mul(qr_swapped_inline2205, sin_row_inline2231))
                qr_vec_inline2181: pl.Tensor[[32, 128], pl.BF16] = pl.tensor.concat(pl.tensor.cast(qr_nope_slice_inline2206, target_type=pl.BF16, mode='rint'), pl.tensor.cast(rope_rot_inline2185, target_type=pl.BF16, mode='rint'))
                qr_bf16_inline2223: pl.Tensor[[bs_heads, 128], pl.BF16] = pl.tensor.assemble(qr_bf16_inline2223, qr_vec_inline2181, [o0_inline2221, 0])
            qh_acc_gm_inline2179: pl.Tensor[[bs_heads, 128], pl.FP32] = pl.tensor.create([bs_heads_inline2228, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for idx_inline2227 in pl.spmd(bs_heads_inline2228 // 64, name_hint="qr_hadamard_matmul_spmd", allow_early_resolve=True):
                o0_inline2221: pl.Scalar[pl.INDEX] = idx_inline2227 * 64
                qh_acc_inline2178: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(qr_bf16_inline2223, [64, 128], [o0_inline2221, 0]), hadamard_idx, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                qh_acc_gm_inline2179: pl.Tensor[[bs_heads, 128], pl.FP32] = pl.tensor.assemble(qh_acc_gm_inline2179, qh_acc_inline2178, [o0_inline2221, 0])
            qr_hadamard_i8_inline2177: pl.Tensor[[16384, 128], pl.INT8] = pl.tensor.create([16384, 128], dtype=pl.INT8, layout=pl.TensorLayout.ND)
            qr_hadamard_scale_dq_inline2234: pl.Tensor[[16384, 1], pl.FP32] = pl.tensor.create([16384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(bs_heads_inline2228 // 64, name_hint="qr_hadamard_quant_spmd", allow_early_resolve=True) as qh_quant_tid_inline2182:
                idx_inline2227: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                o0_v1_inline2218: pl.Scalar[pl.INDEX] = idx_inline2227 * 64
                qh_amax_inline2278: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0001)
                for h0_inline2198 in pl.range(0, 128, 64):
                    qh_a_f32_inline2175: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(qh_acc_gm_inline2179, [64, 64], [o0_v1_inline2218, h0_inline2198])
                    qh_a_abs_inline2196: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.maximum(qh_a_f32_inline2175, pl.tensor.neg(qh_a_f32_inline2175))
                    qh_a_max_inline2211: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(qh_a_abs_inline2196), [1, 64])
                    qh_amax_inline2278: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(qh_amax_inline2278, qh_a_max_inline2211)
                qh_scale_quant_row_inline2183: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 64], dtype=pl.FP32, value=127.0), qh_amax_inline2278)
                qh_scale_dq_inline2176: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(qh_scale_quant_row_inline2183), [64, 1])
                qr_hadamard_scale_dq_inline2234: pl.Tensor[[16384, 1], pl.FP32] = pl.tensor.assemble(qr_hadamard_scale_dq_inline2234, qh_scale_dq_inline2176, [o0_v1_inline2218, 0])
                qh_scale_quant_inline2202: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(qh_scale_quant_row_inline2183, [64, 1])
                for h1_inline2188 in pl.range(0, 128, 64):
                    qh_q_f32_inline2226: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(qh_acc_gm_inline2179, [64, 64], [o0_v1_inline2218, h1_inline2188])
                    qh_q_scaled_inline2191: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.row_expand_mul(qh_q_f32_inline2226, qh_scale_quant_inline2202)
                    qh_q_i32_inline2199: pl.Tensor[[64, 64], pl.INT32] = pl.tensor.cast(qh_q_scaled_inline2191, target_type=pl.INT32, mode='rint')
                    qh_q_half_inline2290: pl.Tensor[[64, 64], pl.FP16] = pl.tensor.cast(qh_q_i32_inline2199, target_type=pl.FP16, mode='round')
                    qh_i8_inline2233: pl.Tensor[[64, 64], pl.INT8] = pl.tensor.cast(qh_q_half_inline2290, target_type=pl.INT8, mode='trunc')
                    qr_hadamard_i8_inline2177: pl.Tensor[[16384, 128], pl.INT8] = pl.tensor.assemble(qr_hadamard_i8_inline2177, qh_i8_inline2233, [o0_v1_inline2218, h1_inline2188])
            x_flat_inline2242: pl.Tensor[[T_DYN, 4096], pl.BF16] = x_normed_t_inline1243
            weights_inline2244: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            weights_partial_inline2245: pl.Tensor[[1024, 64], pl.FP32] = pl.tensor.create([1024, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(4 * row_blocks_inline2258, name_hint="weights_proj_spmd", deps=[late_dep_inline1297]) as _weights_tid_inline2286:
                w_unit_inline2246: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                w_rb_inline2237: pl.Scalar[pl.INDEX] = w_unit_inline2246 // 4
                kb_inline2210: pl.Scalar[pl.INDEX] = w_unit_inline2246 - w_rb_inline2237 * 4
                w_r0_inline2207: pl.Scalar[pl.INDEX] = w_rb_inline2237 * 16
                w_rows_inline2229: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2301 - w_r0_inline2207)
                k_base_inline2248: pl.Scalar[pl.INDEX] = kb_inline2210 * 1024
                weights_acc_inline2217: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for db_inline2297 in pl.range(2):
                    d0_inline2250: pl.Scalar[pl.INDEX] = k_base_inline2248 + db_inline2297 * 512
                    x_tile_inline2253: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[w_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline2242, [16, 512], [w_r0_inline2207, d0_inline2250], [w_rows_inline2229, 512])
                    weights_proj_tile_inline2255: pl.Tensor[[512, 64], pl.BF16] = pl.tensor.slice(weights_proj, [512, 64], [d0_inline2250, 0])
                    if db_inline2297 == 0:
                        weights_acc_inline2217: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile_inline2253, weights_proj_tile_inline2255, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    else:
                        weights_acc_inline2217: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(weights_acc_inline2217, x_tile_inline2253, weights_proj_tile_inline2255, a_trans=False, b_trans=False)
                weights_partial_inline2245: pl.Tensor[[1024, 64], pl.FP32] = pl.tensor.assemble(weights_partial_inline2245, weights_acc_inline2217, [kb_inline2210 * 256 + w_r0_inline2207, 0])
            with pl.spmd(row_blocks_inline2258, name_hint="weights_proj_reduce_spmd", allow_early_resolve=True) as weights_tid_inline2256:
                w_rb_v1_inline2235: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                w_r0_v1_inline2213: pl.Scalar[pl.INDEX] = w_rb_v1_inline2235 * 16
                w_sum_inline2209: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(weights_partial_inline2245, [16, 64], [w_r0_v1_inline2213, 0])
                for kb_inline2259 in pl.unroll(1, 4):
                    partial_r0_inline2260: pl.Scalar[pl.INDEX] = kb_inline2259 * 256 + w_r0_v1_inline2213
                    w_sum_inline2209: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(w_sum_inline2209, pl.tensor.slice(weights_partial_inline2245, [16, 64], [partial_r0_inline2260, 0]))
                weights_inline2244: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(weights_inline2244, pl.tensor.muls(w_sum_inline2209, 0.011048543456039806), [w_r0_v1_inline2213, 0])
            # Run exact Top-K with the score and pair arenas on separate rings.
            bs_inline61_inline2238: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_positions_inline1323, 0)
            b_dim_inline57_inline2261: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table, 0)
            idx_block_num_inline53_inline2264: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_kv_cache, 0)
            idx_table_len_inline55_inline2193: pl.Scalar[pl.INDEX] = b_dim_inline57_inline2261 * 8192
            kv_cache_i8_flat_inline46_inline2265: pl.Tensor[[idx_block_num_1 * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.reshape(idx_kv_cache, [idx_block_num_inline53_inline2264 * 32, 128])
            kv_scale_flat_inline50_inline2214: pl.Tensor[[idx_block_num_1 * pl.const(32, pl.INDEX), 1], pl.FP32] = pl.tensor.reshape(idx_kv_scale, [idx_block_num_inline53_inline2264 * 32, 1])
            idx_block_table_flat_inline47_inline2186: pl.Tensor[[idx_table_len], pl.INT32] = pl.tensor.reshape(idx_block_table, [idx_table_len_inline55_inline2193])
            pair_arena_inline71_inline2266: pl.Tensor[[4192, 1024], pl.FP32] = pl.tensor.create([4192, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.scope():
                score_arena_inline44_inline2267: pl.Tensor[[bs_1, 262144], pl.FP32] = pl.tensor.create([bs_inline61_inline2238, 262144], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                with pl.spmd(24, name_hint="indexer_score_leaf_wave_spmd", optimizations=[pl.cross_core_slot(slot_num=2)], deps=[qh_quant_tid_inline2182, weights_tid_inline2256, idx_cache_write_tid_inline1316]) as score_tid_inline45_inline2269:
                    worker_inline75_inline2270: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                    query_count_inline56_inline2271: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_positions_inline1323, 0)
                    global_leaf_base_inline51_inline2273: pl.Scalar[pl.INDEX] = 0
                    for query_inline65_inline2274 in pl.range(query_count_inline56_inline2271):
                        batch_idx_inline60_inline2249: pl.Scalar[pl.INDEX] = query_inline65_inline2274 // 8
                        position_inline54_inline2275: pl.Scalar[pl.INT32] = pl.tensor.read(idx_positions_inline1323, [query_inline65_inline2274])
                        cache_len_inline64_inline2279: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens, [batch_idx_inline60_inline2249]), pl.INDEX) // 4
                        visible_count_inline49_inline2280: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len_inline64_inline2279, (pl.cast(position_inline54_inline2275, pl.INDEX) + 1) // 4), 262144), 0)
                        leaf_count_inline66_inline2281: pl.Scalar[pl.INDEX] = (visible_count_inline49_inline2280 + 8192 - 1) // 8192
                        base_mod_inline52_inline2283: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273 % 24
                        first_leaf_inline67_inline2285: pl.Scalar[pl.INDEX] = (worker_inline75_inline2270 + base_mod_inline52_inline2283) % 24
                        for leaf_inline48_inline2288 in pl.range(first_leaf_inline67_inline2285, leaf_count_inline66_inline2281, 24):
                            logical_begin_inline63_inline2224: pl.Scalar[pl.INDEX] = leaf_inline48_inline2288 * 8192
                            valid_count_inline68_inline2219: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count_inline49_inline2280 - logical_begin_inline63_inline2224)
                            query_head_begin_inline69_inline2289: pl.Scalar[pl.INDEX] = query_inline65_inline2274 * 64
                            query_vector_inline70_inline2291: pl.Tensor[[64, 128], pl.INT8] = pl.tensor.slice(qr_hadamard_i8_inline2177, [64, 128], [query_head_begin_inline69_inline2289, 0])
                            query_scale_inline73_inline2197: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(qr_hadamard_scale_dq_inline2234, [64, 1], [query_head_begin_inline69_inline2289, 0]), [1, 64])
                            query_weight_inline76_inline2243: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(weights_inline2244, [1, 64], [query_inline65_inline2274, 0])
                            page_count_inline72_inline2252: pl.Scalar[pl.INDEX] = (valid_count_inline68_inline2219 + 32 - 1) // 32
                            for page_inline43_inline2292 in pl.pipeline(page_count_inline72_inline2252, stage=2):
                                page_begin_inline42_inline2287: pl.Scalar[pl.INDEX] = page_inline43_inline2292 * 32
                                logical_row_inline41_inline2282: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224 + page_begin_inline42_inline2287
                                logical_page_inline74_inline2293: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282 // 32
                                physical_block_inline40_inline2298: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_block_table_flat_inline47_inline2186, [batch_idx_inline60_inline2249 * 8192 + logical_page_inline74_inline2293]), pl.INDEX)
                                physical_row_inline59_inline2299: pl.Scalar[pl.INDEX] = physical_block_inline40_inline2298 * 32
                                kv_i8_inline38_inline2300: pl.Tensor[[32, 128], pl.INT8] = pl.tensor.slice(kv_cache_i8_flat_inline46_inline2265, [32, 128], [physical_row_inline59_inline2299, 0])
                                score_i32_inline37_inline2302: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.matmul(kv_i8_inline38_inline2300, query_vector_inline70_inline2291, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.INT32)
                                score_fp32_inline35_inline2254: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.cast(score_i32_inline37_inline2302, target_type=pl.FP32, mode='none')
                                score_fp32_v1_inline34_inline2251: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(score_fp32_inline35_inline2254, query_scale_inline73_inline2197)
                                score_fp32_v2_inline62_inline2294: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.maximum(score_fp32_v1_inline34_inline2251, 0.0)
                                score_fp32_v3_inline33_inline2187: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(score_fp32_v2_inline62_inline2294, query_weight_inline76_inline2243)
                                kv_scale_inline32_inline2184: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.slice(kv_scale_flat_inline50_inline2214, [32, 1], [physical_row_inline59_inline2299, 0])
                                score_inline58_inline2180: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.mul(pl.tensor.row_sum(score_fp32_v3_inline33_inline2187), kv_scale_inline32_inline2184)
                                score_row_inline31_inline2296: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.reshape(score_inline58_inline2180, [1, 32])
                                valid_rows_inline39_inline2216: pl.Scalar[pl.INDEX] = pl.min(32, valid_count_inline68_inline2219 - page_begin_inline42_inline2287)
                                score_valid_inline30_inline2277: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.fillpad(pl.tensor.set_validshape(score_row_inline31_inline2296, 1, valid_rows_inline39_inline2216), pad_value=pl.PadValue.min)
                                score_arena_inline44_inline2267: pl.Tensor[[bs_1, 262144], pl.FP32] = pl.tensor.assemble(score_arena_inline44_inline2267, score_valid_inline30_inline2277, [query_inline65_inline2274, logical_row_inline41_inline2282])
                        global_leaf_base_inline51_inline2273: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273 + leaf_count_inline66_inline2281
                with pl.spmd(48, name_hint="indexer_topk_group_wave", deps=[score_tid_inline45_inline2269]) as topk_tid_inline36_inline2174:
                    self.indexer_topk_group_wave(idx_positions_inline1323, kv_seq_lens, score_arena_inline44_inline2267, pair_arena_inline71_inline2266)
                with pl.spmd(bs_inline61_inline2238, name_hint="indexer_topk_query_merge", deps=[topk_tid_inline36_inline2174]) as _score_tid_inline29_inline2247:
                    self.indexer_topk_query_merge(idx_positions_inline1323, kv_seq_lens, pair_arena_inline71_inline2266, idx_topk_scores_inline1271, idx_topk_inline1280)
            idx_topk_scores_inline1271: pl.Tensor[[t_dim, 512], pl.FP32] = idx_topk_scores_inline1271
            idx_topk_inline1280: pl.Tensor[[t_dim, 512], pl.INT32] = idx_topk_inline1280
            idx_topk_scores_inline1271: pl.Tensor[[t_dim, 512], pl.FP32] = idx_topk_scores_inline1271
            idx_topk_inline1280: pl.Tensor[[t_dim, 512], pl.INT32] = idx_topk_inline1280
            # Plan and run CSA QK/PV over sparse blocks, and build inverse-RoPE metadata.
            ori_block_num_inline2362: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache, 0)
            t_dim_inline2369: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_inline1246, 0)
            t_heads_inline2364: pl.Scalar[pl.INDEX] = t_dim_inline2369 * 64
            t_blk_inline2373: pl.Scalar[pl.INDEX] = t_dim_inline2369 * 4 * 5 * 16
            qk_items_inline2347: pl.Scalar[pl.INDEX] = t_dim_inline2369 * 5
            rope_cs_blocks_inline2380: pl.Scalar[pl.INDEX] = t_dim_inline2369 // 8
            ori_kv_flat_inline2344: pl.Tensor[[ori_block_num_1 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(kv_cache, [ori_block_num_inline2362 * 32, 512])
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_touch", allow_early_resolve=True):
                ori_kv_flat_inline2344: pl.Tensor[[ori_block_num_1 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(ori_kv_flat_inline2344, pl.tensor.slice(ori_kv_flat_inline2344, [1, 512], [0, 0]), [0, 0])
            sparse_bias_inline2381: pl.Tensor[[t_dim_5, 640], pl.FP32] = pl.tensor.create([t_dim_inline2369, 640], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            cmp_sparse_indices_inline2383: pl.Tensor[[t_dim_5, 512], pl.INT32] = pl.tensor.create([t_dim_inline2369, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            valid_block_mask_inline2385: pl.Tensor[[t_dim_5, 5], pl.INT32] = pl.tensor.create([t_dim_inline2369, 5], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            qk_order_inline2351: pl.Tensor[[1280], pl.INT32] = pl.tensor.create([1280], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            qk_wcur_inline2412: pl.Tensor[[1], pl.INT32] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_slots_build_valid_qk_plan") as qk_plan_tid_inline2387:
                for bias_t0_inline2361 in pl.range(0, t_dim_inline2369, 8):
                    c_raw_inline2382: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.cast(pl.tensor.slice(idx_topk_inline1280, [8, 512], [bias_t0_inline2361, 0]), target_type=pl.FP32, mode='round')
                    c_pos_inline2359: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.cast(pl.tensor.slice(position_ids_t1_inline1288, [8, 1], [bias_t0_inline2361, 0]), target_type=pl.FP32, mode='round')
                    c_pos_scaled_inline2417: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.muls(pl.tensor.adds(c_pos_inline2359, 1.0), 0.25)
                    c_pos_i32_inline2358: pl.Tensor[[8, 1], pl.INT32] = pl.tensor.cast(c_pos_scaled_inline2417, target_type=pl.INT32, mode='trunc')
                    c_pos_q_inline2367: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.cast(c_pos_i32_inline2358, target_type=pl.FP32, mode='round')
                    c_upper_b_inline2379: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.full([8, 512], dtype=pl.FP32, value=1.0), c_pos_q_inline2367)
                    c_ge_inline2421: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.adds(c_raw_inline2382, 1.0), 0.0), 1.0)
                    c_lt_inline2354: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.sub(c_upper_b_inline2379, c_raw_inline2382), 0.0), 1.0)
                    c_mask_inline2370: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(c_ge_inline2421, c_lt_inline2354)
                    c_out_inline2410: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.subs(pl.tensor.mul(c_mask_inline2370, pl.tensor.adds(c_raw_inline2382, 1.0)), 1.0)
                    cmp_sparse_indices_inline2383: pl.Tensor[[t_dim_5, 512], pl.INT32] = pl.tensor.assemble(cmp_sparse_indices_inline2383, pl.tensor.cast(c_out_inline2410, target_type=pl.INT32, mode='round'), [bias_t0_inline2361, 0])
                    for c_t0_inline2340 in pl.range(8):
                        pl.tensor.write(valid_block_mask_inline2385, [bias_t0_inline2361 + c_t0_inline2340, 0], pl.cast(1, pl.INT32))
                    for c_sb_inline2407 in pl.range(1, 5):
                        c_s0_inline2342: pl.Scalar[pl.INDEX] = (c_sb_inline2407 - 1) * 128
                        c_blk_valid_inline2350: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_max(pl.tensor.slice(c_mask_inline2370, [8, 128], [0, c_s0_inline2342]))
                        for c_dt_inline2378 in pl.range(8):
                            c_valid_inline2389: pl.Scalar[pl.INT32] = pl.cast(pl.tensor.read(c_blk_valid_inline2350, [c_dt_inline2378, 0]), pl.INT32)
                            pl.tensor.write(valid_block_mask_inline2385, [bias_t0_inline2361 + c_dt_inline2378, c_sb_inline2407], c_valid_inline2389)
                    v_win_f_inline2375: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(window_swa_indices, [8, 128], [bias_t0_inline2361, 0]), target_type=pl.FP32, mode='round')
                    v_win_valid_inline2414: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.adds(v_win_f_inline2375, 1.0), 0.0), 1.0)
                    sparse_bias_inline2381: pl.Tensor[[t_dim_5, 640], pl.FP32] = pl.tensor.assemble(sparse_bias_inline2381, pl.tensor.muls(pl.tensor.subs(v_win_valid_inline2414, 1.0), 1e+20), [bias_t0_inline2361, 0])
                    sparse_bias_inline2381: pl.Tensor[[t_dim_5, 640], pl.FP32] = pl.tensor.assemble(sparse_bias_inline2381, pl.tensor.muls(pl.tensor.minimum(c_out_inline2410, 0.0), 1e+20), [bias_t0_inline2361, 128])
                    if pl.const(640, pl.INDEX) > pl.const(640, pl.INDEX):
                        sparse_bias_inline2381: pl.Tensor[[t_dim_5, 640], pl.FP32] = pl.tensor.assemble(sparse_bias_inline2381, pl.tensor.full([8, 0], dtype=pl.FP32, value=-1e+20), [bias_t0_inline2361, 640])
                pl.tensor.write(qk_wcur_inline2412, [0], pl.cast(0, pl.INT32))
                for plan_t_inline2392 in pl.range(t_dim_inline2369):
                    for plan_sb_inline2386 in pl.range(5):
                        if pl.cast(pl.tensor.read(valid_block_mask_inline2385, [plan_t_inline2392, plan_sb_inline2386]), pl.INDEX) > 0:
                            plan_w_inline2348: pl.Scalar[pl.INT32] = pl.tensor.read(qk_wcur_inline2412, [0])
                            pl.tensor.write(qk_order_inline2351, [plan_w_inline2348], pl.cast(plan_t_inline2392 * 5 + plan_sb_inline2386, pl.INT32))
                            pl.tensor.write(qk_wcur_inline2412, [0], pl.cast(pl.cast(plan_w_inline2348, pl.INDEX) + 1, pl.INT32))
                for plan_t_inline2394 in pl.range(t_dim_inline2369):
                    for plan_sb_inline2395 in pl.range(5):
                        if pl.cast(pl.tensor.read(valid_block_mask_inline2385, [plan_t_inline2394, plan_sb_inline2395]), pl.INDEX) <= 0:
                            plan_w_v1_inline2399: pl.Scalar[pl.INT32] = pl.tensor.read(qk_wcur_inline2412, [0])
                            pl.tensor.write(qk_order_inline2351, [plan_w_v1_inline2399], pl.cast(plan_t_inline2394 * 5 + plan_sb_inline2395, pl.INT32))
                            pl.tensor.write(qk_wcur_inline2412, [0], pl.cast(pl.cast(plan_w_v1_inline2399, pl.INDEX) + 1, pl.INT32))
            cmp_block_num_inline2376: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv, 0)
            cmp_kv_flat_inline2401: pl.Tensor[[cmp_block_num_1 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(cmp_kv, [cmp_block_num_inline2376 * 32, 512])
            q_flat_inline2355: pl.Tensor[[t_heads, 512], pl.BF16] = pl.tensor.reshape(q_inline1246, [t_heads_inline2364, 512])
            sparse_blk_mi_inline2404: pl.Tensor[[t_blk, 1], pl.FP32] = pl.tensor.create([t_blk_inline2373, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            sparse_blk_li_inline2405: pl.Tensor[[t_blk, 1], pl.FP32] = pl.tensor.create([t_blk_inline2373, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            sparse_blk_oi_inline2398: pl.Tensor[[t_blk, 512], pl.FP32] = pl.tensor.create([t_blk_inline2373, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(24, name_hint="qk_pv_spmd", deps=[qk_plan_tid_inline2387, cache_ready_dep_inline1304], allow_early_resolve=True) as qk_tid_inline2349:
                qk_core_inline2368: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                qk_lane_iters_inline2408: pl.Scalar[pl.INDEX] = (qk_items_inline2347 - qk_core_inline2368 + 24 - 1) // 24
                for qk_it_inline2413 in pl.range(qk_lane_iters_inline2408):
                    qk_flat_inline2365: pl.Scalar[pl.INDEX] = qk_core_inline2368 + qk_it_inline2413 * 24
                    qk_item_inline2403: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(qk_order_inline2351, [qk_flat_inline2365]), pl.INDEX)
                    qk_t_inline2411: pl.Scalar[pl.INDEX] = qk_item_inline2403 // 5
                    qk_sb_inline2374: pl.Scalar[pl.INDEX] = qk_item_inline2403 - qk_t_inline2411 * 5
                    qk_b_inline2371: pl.Scalar[pl.INDEX] = qk_t_inline2411 // 8
                    qk_token_base_inline2391: pl.Scalar[pl.INDEX] = qk_t_inline2411 * 4 * 5 * 16
                    qk_s0_inline2418: pl.Scalar[pl.INDEX] = qk_sb_inline2374 * 128
                    qk_bias_row_inline2419: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(sparse_bias_inline2381, [1, 128], [qk_t_inline2411, qk_s0_inline2418])
                    qk_block_valid_inline2422: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2385, [qk_t_inline2411, qk_sb_inline2374])
                    if pl.cast(qk_block_valid_inline2422, pl.INDEX) > 0:
                        qk_kv_inline2415: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.create_l1([128, 512], dtype=pl.BF16, transpose=False)
                        qk_win_rows_inline2406: pl.Scalar[pl.INDEX] = pl.min(pl.max(128 - qk_s0_inline2418, 0), 128)
                        if qk_win_rows_inline2406 > 0:
                            qk_pos_inline2339: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(position_ids_t1_inline1288, [qk_t_inline2411, 0]), pl.INDEX)
                            qk_win_len_inline2337: pl.Scalar[pl.INDEX] = pl.min(qk_pos_inline2339 + 1, 128)
                            qk_win_start_inline2333: pl.Scalar[pl.INDEX] = qk_pos_inline2339 - qk_win_len_inline2337 + 1
                            qk_run_rows_inline2332: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len_inline2337 - qk_s0_inline2418, 0), qk_win_rows_inline2406)
                            qk_head_inline2353: pl.Scalar[pl.INDEX] = (qk_win_start_inline2333 + qk_s0_inline2418) % 32
                            for qk_run_inline2331 in pl.unroll(5):
                                qk_run_lo_inline2372: pl.Scalar[pl.INDEX] = pl.max(qk_run_inline2331 * 32 - qk_head_inline2353, 0)
                                qk_run_hi_inline2330: pl.Scalar[pl.INDEX] = pl.min((qk_run_inline2331 + 1) * 32 - qk_head_inline2353, qk_run_rows_inline2332)
                                if qk_run_hi_inline2330 > qk_run_lo_inline2372:
                                    qk_run_raw_inline2357: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices, [qk_t_inline2411, qk_s0_inline2418 + qk_run_lo_inline2372])
                                    qk_run_src_inline2384: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357, pl.INDEX), 0), pl.INDEX)
                                    qk_kv_inline2415: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415, ori_kv_flat_inline2344, [qk_run_lo_inline2372, 0], [qk_run_src_inline2384, 0], [128, 512], valid_shape=[qk_run_hi_inline2330 - qk_run_lo_inline2372, 512], transpose=False)
                            qk_tail_n_inline2329: pl.Scalar[pl.INDEX] = qk_win_rows_inline2406 - qk_run_rows_inline2332
                            if qk_tail_n_inline2329 > 0:
                                qk_kv_inline2415: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415, ori_kv_flat_inline2344, [qk_run_rows_inline2332, 0], [0, 0], [128, 512], valid_shape=[qk_tail_n_inline2329, 512], transpose=False)
                        for qk_r_inline2390 in pl.range(qk_win_rows_inline2406, 128):
                            qk_cmp_k_inline2328: pl.Scalar[pl.INDEX] = qk_s0_inline2418 + qk_r_inline2390 - 128
                            if qk_cmp_k_inline2328 < 512:
                                qk_ridx_inline2377: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices_inline2383, [qk_t_inline2411, qk_cmp_k_inline2328])
                                if pl.cast(qk_ridx_inline2377, pl.INDEX) >= 0:
                                    qk_slot_inline2388: pl.Scalar[pl.INT32] = qk_ridx_inline2377
                                    qk_cblk_inline2327: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(cmp_block_table, [qk_b_inline2371, pl.cast(qk_slot_inline2388, pl.INDEX) // 32]), pl.INDEX)
                                    qk_csrc_inline2400: pl.Scalar[pl.INDEX] = qk_cblk_inline2327 * 32 + pl.cast(qk_slot_inline2388, pl.INDEX) % 32
                                    qk_kv_inline2415: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415, cmp_kv_flat_inline2401, [qk_r_inline2390, 0], [qk_csrc_inline2400, 0], [1, 512], transpose=False)
                                else:
                                    qk_kv_inline2415: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415, ori_kv_flat_inline2344, [qk_r_inline2390, 0], [0, 0], [1, 512], transpose=False)
                            else:
                                qk_kv_inline2415: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415, ori_kv_flat_inline2344, [qk_r_inline2390, 0], [0, 0], [1, 512], transpose=False)
                        for qk_hb_inline2336 in pl.pipeline(2, stage=2):
                            qk_h0_inline2393: pl.Scalar[pl.INDEX] = qk_hb_inline2336 * 32
                            qk_head_row_inline2326: pl.Scalar[pl.INDEX] = qk_t_inline2411 * 64 + qk_h0_inline2393
                            qk_q_tile_inline2338: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(q_flat_inline2355, [32, 512], [qk_head_row_inline2326, 0])
                            qk_raw_inline2397: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.matmul(qk_q_tile_inline2338, qk_kv_inline2415, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                            qk_scaled_inline2416: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.muls(qk_raw_inline2397, 0.044194173824159223)
                            qk_scores_inline2325: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.col_expand_add(qk_scaled_inline2416, qk_bias_row_inline2419)
                            qk_mi_inline2360: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.row_max(qk_scores_inline2325)
                            qk_exp_inline2324: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(qk_scores_inline2325, qk_mi_inline2360))
                            qk_li_inline2323: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.row_sum(qk_exp_inline2324)
                            qk_exp_bf16_inline2322: pl.Tensor[[32, 128], pl.BF16] = pl.tensor.cast(qk_exp_inline2324, target_type=pl.BF16, mode='rint')
                            qk_oi_inline2335: pl.Tensor[[32, 512], pl.FP32] = pl.tensor.matmul(qk_exp_bf16_inline2322, qk_kv_inline2415, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            for qk_sub_inline2321 in pl.unroll(2):
                                qk_h_idx_inline2320: pl.Scalar[pl.INDEX] = qk_hb_inline2336 * 2 + qk_sub_inline2321
                                qk_r0_inline2343: pl.Scalar[pl.INDEX] = qk_sub_inline2321 * 16
                                qk_blk_base_inline2319: pl.Scalar[pl.INDEX] = qk_token_base_inline2391 + qk_h_idx_inline2320 * 5 * 16
                                qk_row_inline2356: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319 + qk_sb_inline2374 * 16
                                sparse_blk_mi_inline2404: pl.Tensor[[t_blk, 1], pl.FP32] = pl.tensor.assemble(sparse_blk_mi_inline2404, pl.tensor.slice(qk_mi_inline2360, [16, 1], [qk_r0_inline2343, 0]), [qk_row_inline2356, 0])
                                sparse_blk_li_inline2405: pl.Tensor[[t_blk, 1], pl.FP32] = pl.tensor.assemble(sparse_blk_li_inline2405, pl.tensor.slice(qk_li_inline2323, [16, 1], [qk_r0_inline2343, 0]), [qk_row_inline2356, 0])
                                sparse_blk_oi_inline2398: pl.Tensor[[t_blk, 512], pl.FP32] = pl.tensor.assemble(sparse_blk_oi_inline2398, pl.tensor.slice(qk_oi_inline2335, [16, 512], [qk_r0_inline2343, 0]), [qk_row_inline2356, 0])
                    else:
                        qk_oi_zero_inline2318: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                        for qk_h_idx_inline2317 in pl.range(4):
                            qk_blk_base_inline2319: pl.Scalar[pl.INDEX] = qk_token_base_inline2391 + qk_h_idx_inline2317 * 5 * 16
                            qk_row_inline2356: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319 + qk_sb_inline2374 * 16
                            for qk_hr_inline2346 in pl.range(16):
                                pl.tensor.write(sparse_blk_mi_inline2404, [qk_row_inline2356 + qk_hr_inline2346, 0], -3.0000000000000001e+38)
                                pl.tensor.write(sparse_blk_li_inline2405, [qk_row_inline2356 + qk_hr_inline2346, 0], 0.0)
                            sparse_blk_oi_inline2398: pl.Tensor[[t_blk, 512], pl.FP32] = pl.tensor.assemble(sparse_blk_oi_inline2398, qk_oi_zero_inline2318, [qk_row_inline2356, 0])
            rope_cos_il_inline2316: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            rope_sin_signed_inline2315: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            rope_swap_idx_inline2314: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.create([16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True) as rope_tid_inline2402:
                sw_ones_inline2420: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
                sw_idx_f_inline2366: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round')
                sw_col_inline2313: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(sw_ones_inline2420, sw_idx_f_inline2366)
                sw_dup_i32_inline2312: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.muls(sw_col_inline2313, 0.5), target_type=pl.INT32, mode='trunc')
                sw_dup_f_inline2311: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(sw_dup_i32_inline2312, target_type=pl.FP32, mode='round')
                sw_lane_inline2310: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(sw_col_inline2313, pl.tensor.muls(sw_dup_f_inline2311, 2.0))
                sw_swap_f_inline2363: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(pl.tensor.adds(sw_col_inline2313, 1.0), pl.tensor.muls(sw_lane_inline2310, 2.0))
                rope_swap_idx_inline2314: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_inline2314, pl.tensor.cast(sw_swap_f_inline2363, target_type=pl.INT32, mode='round'), [0, 0])
                cs_ones_inline2309: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
                cs_idx_f_inline2345: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round')
                cs_col_inline2308: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(cs_ones_inline2309, cs_idx_f_inline2345)
                cs_dup_i32_inline2334: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(pl.tensor.muls(cs_col_inline2308, 0.5), target_type=pl.INT32, mode='trunc')
                cs_dup_f_inline2307: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(cs_dup_i32_inline2334, target_type=pl.FP32, mode='round')
                cs_dup_idx_inline2396: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(cs_dup_f_inline2307, target_type=pl.INT32, mode='round')
                cs_lane_inline2352: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(cs_col_inline2308, pl.tensor.muls(cs_dup_f_inline2307, 2.0))
                cs_sign_inline2306: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.neg(pl.tensor.subs(pl.tensor.muls(cs_lane_inline2352, 2.0), 1.0))
                for cs_rb_inline2341 in pl.range(rope_cs_blocks_inline2380):
                    cs_t0_inline2305: pl.Scalar[pl.INDEX] = cs_rb_inline2341 * 8
                    cs_cos_inline2304: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.cast(pl.tensor.slice(freqs_cos_local, [8, 32], [cs_t0_inline2305, 0]), target_type=pl.FP32, mode='round')
                    cs_sin_inline2303: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.cast(pl.tensor.slice(freqs_sin_local, [8, 32], [cs_t0_inline2305, 0]), target_type=pl.FP32, mode='round')
                    rope_cos_il_inline2316: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(rope_cos_il_inline2316, pl.tensor.gather(cs_cos_inline2304, cs_dup_idx_inline2396, dim=-1), [cs_t0_inline2305, 0])
                    cs_sin_il_inline2409: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(cs_sin_inline2303, cs_dup_idx_inline2396, dim=-1)
                    rope_sin_signed_inline2315: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed_inline2315, pl.tensor.mul(cs_sin_il_inline2409, cs_sign_inline2306), [cs_t0_inline2305, 0])
            sparse_blk_mi_inline1234: pl.Tensor[[t_blk, 1], pl.FP32] = sparse_blk_mi_inline2404
            sparse_blk_li_inline1283: pl.Tensor[[t_blk, 1], pl.FP32] = sparse_blk_li_inline2405
            sparse_blk_oi_inline1233: pl.Tensor[[t_blk, 512], pl.FP32] = sparse_blk_oi_inline2398
            rope_cos_il_inline1232: pl.Tensor[[256, 64], pl.FP32] = rope_cos_il_inline2316
            rope_sin_signed_inline1231: pl.Tensor[[256, 64], pl.FP32] = rope_sin_signed_inline2315
            rope_swap_idx_inline1230: pl.Tensor[[16, 64], pl.INT32] = rope_swap_idx_inline2314
            qk_tid_inline1229: pl.Scalar[pl.TASK_ID] = qk_tid_inline2349
            attn_rope_tid_inline1294: pl.Scalar[pl.TASK_ID] = rope_tid_inline2402
            attention_grouped_inline1276: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.create([2048, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            pack_work_count_inline1228: pl.Scalar[pl.INDEX] = t_dim_inline1251 // 8 * 4
            with pl.spmd(48, name_hint="csa_merge_pack_publish_spmd", deps=[qk_tid_inline1229, attn_rope_tid_inline1294]) as publish_tid_inline1260:
                worker_inline1227: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for pack_work_inline1270 in pl.range(worker_inline1227, pack_work_count_inline1228, 48):
                    token_block_inline1226: pl.Scalar[pl.INDEX] = pack_work_inline1270 // 4
                    m_h_idx_inline1303: pl.Scalar[pl.INDEX] = pack_work_inline1270 - token_block_inline1226 * 4
                    m_t0_inline1278: pl.Scalar[pl.INDEX] = token_block_inline1226 * 8
                    m_h0_inline1224: pl.Scalar[pl.INDEX] = m_h_idx_inline1303 * 16
                    global_group0_inline1314: pl.Scalar[pl.INDEX] = m_h0_inline1224 // 8
                    destination_rank_inline1223: pl.Scalar[pl.INDEX] = global_group0_inline1314 // 4
                    local_group0_inline1222: pl.Scalar[pl.INDEX] = global_group0_inline1314 - destination_rank_inline1223 * 4
                    for m_dt_inline1287 in pl.range(8):
                        m_t_inline1290: pl.Scalar[pl.INDEX] = m_t0_inline1278 + m_dt_inline1287
                        m_idx_inline1289: pl.Scalar[pl.INDEX] = m_t_inline1290 * 4 + m_h_idx_inline1303
                        m_blk_base_inline1306: pl.Scalar[pl.INDEX] = m_idx_inline1289 * 5 * 16
                        m_mi_inline1221: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi_inline1234, [16, 1], [m_blk_base_inline1306, 0])
                        m_li_inline1302: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li_inline1283, [16, 1], [m_blk_base_inline1306, 0])
                        m_oi_inline1220: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi_inline1233, [16, 512], [m_blk_base_inline1306, 0])
                        for m_sb_inline1219 in pl.pipeline(1, 5, stage=2):
                            m_row_inline1218: pl.Scalar[pl.INDEX] = m_blk_base_inline1306 + m_sb_inline1219 * 16
                            m_cur_mi_inline1217: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi_inline1234, [16, 1], [m_row_inline1218, 0])
                            m_cur_li_inline1293: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li_inline1283, [16, 1], [m_row_inline1218, 0])
                            m_cur_oi_inline1216: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi_inline1233, [16, 512], [m_row_inline1218, 0])
                            m_mi_new_inline1215: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.maximum(m_mi_inline1221, m_cur_mi_inline1217)
                            m_alpha_inline1214: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_mi_inline1221, m_mi_new_inline1215))
                            m_beta_inline1213: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_cur_mi_inline1217, m_mi_new_inline1215))
                            m_li_inline1302: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(m_alpha_inline1214, m_li_inline1302), pl.tensor.mul(m_beta_inline1213, m_cur_li_inline1293))
                            m_oi_inline1220: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(m_oi_inline1220, m_alpha_inline1214), pl.tensor.row_expand_mul(m_cur_oi_inline1216, m_beta_inline1213))
                            m_mi_inline1221: pl.Tensor[[16, 1], pl.FP32] = m_mi_new_inline1215
                        n_sink_bias_inline1266: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(attn_sink, [16], [m_h0_inline1224]), [16, 1])
                        n_sink_tile_inline1212: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.sub(m_mi_inline1221, m_mi_inline1221), n_sink_bias_inline1266)
                        n_denom_inline1254: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(m_li_inline1302, pl.tensor.exp(pl.tensor.sub(n_sink_tile_inline1212, m_mi_inline1221)))
                        n_full_inline1211: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(pl.tensor.row_expand_div(m_oi_inline1220, n_denom_inline1254), [16, 512], [0, 0])
                        n_bf16_inline1210: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.cast(n_full_inline1211, target_type=pl.BF16, mode='rint')
                        m_rope_inline1208: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(n_full_inline1211, [16, 64], [0, 448])
                        m_cos_il_inline1286: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos_il_inline1232, [1, 64], [m_t_inline1290, 0])
                        m_sin_signed_inline1225: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin_signed_inline1231, [1, 64], [m_t_inline1290, 0])
                        m_swapped_inline1207: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(m_rope_inline1208, pl.tensor.slice(rope_swap_idx_inline1230, [16, 64], [0, 0]), dim=-1)
                        m_rot_inline1206: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(m_rope_inline1208, m_cos_il_inline1286), pl.tensor.col_expand_mul(m_swapped_inline1207, m_sin_signed_inline1225))
                        n_rope_bf16_inline1205: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(m_rot_inline1206, target_type=pl.BF16, mode='rint')
                        n_full_bf16_inline1285: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.concat(pl.tensor.slice(n_bf16_inline1210, [16, 448], [0, 0]), n_rope_bf16_inline1205)
                        for n_hi_inline1209 in pl.unroll(16):
                            n_head_inline1244: pl.Scalar[pl.INDEX] = m_h0_inline1224 + n_hi_inline1209
                            source_row_inline1238: pl.Scalar[pl.INDEX] = n_head_inline1244 // 8 * 256 + m_t_inline1290
                            source_col_inline1204: pl.Scalar[pl.INDEX] = n_head_inline1244 % 8 * 512
                            attention_grouped_inline1276: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276, pl.tensor.slice(n_full_bf16_inline1285, [1, 512], [n_hi_inline1209, 0]), [source_row_inline1238, source_col_inline1204])
                    for group_slot_inline1262 in pl.unroll(2):
                        source_row_inline1238: pl.Scalar[pl.INDEX] = (global_group0_inline1314 + group_slot_inline1262) * 256 + m_t0_inline1278
                        target_row_inline1203: pl.Scalar[pl.INDEX] = (local_group0_inline1222 + group_slot_inline1262) * 512 + pl.cast(tp_rank, pl.INDEX) * 256 + m_t0_inline1278
                        pld.tensor.put(attention_window, pl.cast(group_base, pl.INDEX) + destination_rank_inline1223, attention_grouped_inline1276, [target_row_inline1203, 0], [source_row_inline1238, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                for peer_tp_inline1273 in pl.range(2):
                    if peer_tp_inline1273 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(attention_signal, pl.cast(group_base, pl.INDEX) + peer_tp_inline1273, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
            # Finish a non-overlapping producer-fused exchange and release its window.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_group_a2a_wait", deps=[publish_tid_inline1260]) as wait_tid_inline2436:
                expected_inline2426: pl.Scalar[pl.INT32] = pl.cast(48, pl.INT32)
                for source_tp_inline2427 in pl.range(2):
                    if source_tp_inline2427 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.wait(attention_signal, [source_tp_inline2427, 0], expected_inline2426, cmp=1)
            group_t_inline2428: pl.Scalar[pl.INDEX] = 512
            with pl.spmd(48, name_hint="o_group_a2a_gather_spmd", deps=[wait_tid_inline2436]) as gather_tid_inline2431:
                worker_inline2433: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for local_group_inline2435 in pl.range(4):
                    group_base_row_inline2437: pl.Scalar[pl.INDEX] = local_group_inline2435 * 512
                    for group_row_inline2432 in pl.range(worker_inline2433, group_t_inline2428, 48):
                        copy_row_inline2430: pl.Scalar[pl.INDEX] = group_base_row_inline2437 + group_row_inline2432
                        attention_local_flat_inline1292: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_local_flat_inline1292, pl.tensor.slice(attention_window, [1, 4096], [copy_row_inline2430, 0]), [copy_row_inline2430, 0])
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_group_a2a_complete", deps=[gather_tid_inline2431]):
                completion_anchor_inline2439: pl.Scalar[pl.BF16] = pl.tensor.read(attention_local_flat_inline1292, [0, 0])
                for peer_tp_inline2438 in pl.range(2):
                    if peer_tp_inline2438 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(attention_signal, pl.cast(group_base, pl.INDEX) + peer_tp_inline2438, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
                completion_expected_inline2425: pl.Scalar[pl.INT32] = pl.cast(pl.cast(48, pl.INDEX) + 1, pl.INT32)
                for source_tp_inline2424 in pl.range(2):
                    if source_tp_inline2424 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.wait(attention_signal, [source_tp_inline2424, 0], completion_expected_inline2425, cmp=1)
                reset_value_inline2434: pl.Scalar[pl.INT32] = pl.cast(-completion_expected_inline2425, pl.INT32)
                self_rank_inline2423: pl.Scalar[pl.INT32] = group_base + tp_rank
                for source_tp_inline2429 in pl.range(2):
                    if source_tp_inline2429 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(attention_signal, self_rank_inline2423, [source_tp_inline2429, 0], reset_value_inline2434, op=0)
                pl.tensor.write(attention_local_flat_inline1292, [0, 0], completion_anchor_inline2439)
            attention_local_flat_inline1292: pl.Tensor[[2048, 4096], pl.BF16] = attention_local_flat_inline1292
            attention_signal: pld.DistributedTensor[[2, 1], pl.INT32] = attention_signal
            attention_local_groups_inline1321: pl.Tensor[[4, 512, 4096], pl.BF16] = pl.tensor.reshape(attention_local_flat_inline1292, [4, 512, 4096])
            # Project O-B tiles directly into their ReduceScatter owner windows.
            group_t_inline2530: pl.Scalar[pl.INDEX] = 512
            o_a_rows_inline2513: pl.Scalar[pl.INDEX] = (group_t_inline2530 + 128 - 1) // 128
            o_b_rows_inline2534: pl.Scalar[pl.INDEX] = (group_t_inline2530 + 128 - 1) // 128
            o_b_group_t_inline2487: pl.Scalar[pl.INDEX] = o_b_rows_inline2534 * 128
            owner_rows_inline2509: pl.Scalar[pl.INDEX] = 16
            attn_2d_inline2548: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.reshape(attention_local_groups_inline1321, [2048, 4096])
            wo_a_flat_inline2521: pl.Tensor[[4096, 4096], pl.BF16] = pl.tensor.reshape(wo_a, [4096, 4096])
            publish_all_inline2525: pl.Tensor[[512, 4096], pl.BF16] = pl.tensor.create([512, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            put_rows_inline2508: pl.Scalar[pl.INDEX] = 32
            own_a_rows_inline2516: pl.Scalar[pl.INDEX] = 2
            own_b_rows_inline2529: pl.Scalar[pl.INDEX] = 2
            own_b_t_inline2531: pl.Scalar[pl.INDEX] = own_b_rows_inline2529 * 128
            own_quant_blocks_inline2490: pl.Scalar[pl.INDEX] = 32
            own_pad_blocks_inline2511: pl.Scalar[pl.INDEX] = (own_b_t_inline2531 + 8 - 1) // 8
            own_act_rows_inline2505: pl.Scalar[pl.INDEX] = 16
            for owner_inline2504 in pl.parallel(2):
                own_base_inline2502: pl.Scalar[pl.INDEX] = owner_inline2504 * 256
                own_a_fp32_inline2500: pl.Tensor[[256, 4096], pl.FP32] = pl.tensor.create([256, 4096], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                own_a_i8_inline2497: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.create([256, 4096], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                own_scale_inline2494: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.create([4, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                own_b_i32_inline2528: pl.Tensor[[256, 16384], pl.INT32] = pl.tensor.create([256, 16384], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                for local_group_inline2514 in pl.parallel(4):
                    attention_row_inline2526: pl.Scalar[pl.INDEX] = local_group_inline2514 * 512 + own_base_inline2502
                    o_a_col_inline2496: pl.Scalar[pl.INDEX] = local_group_inline2514 * 1024
                    for pa_unit_inline2493 in pl.spmd(own_a_rows_inline2516 * 8, name_hint="tp_o_a_spmd"):
                        pa_rb_inline2491: pl.Scalar[pl.INDEX] = pa_unit_inline2493 // 8
                        pa_nb_inline2499: pl.Scalar[pl.INDEX] = pa_unit_inline2493 - pa_rb_inline2491 * 8
                        pa_t0_inline2489: pl.Scalar[pl.INDEX] = pa_rb_inline2491 * 128
                        pa_n0_inline2485: pl.Scalar[pl.INDEX] = pa_nb_inline2499 * 128
                        pa_rows_inline2532: pl.Scalar[pl.INDEX] = pl.min(128, 256 - pa_t0_inline2489)
                        pa_src_inline2495: pl.Scalar[pl.INDEX] = attention_row_inline2526 + pa_t0_inline2489
                        pa_wrow_inline2533: pl.Scalar[pl.INDEX] = o_a_col_inline2496 + pa_n0_inline2485
                        pa_x0_inline2535: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(attn_2d_inline2548, [128, 256], [pa_src_inline2495, 0], [pa_rows_inline2532, 256])
                        pa_w0_inline2523: pl.Tensor[[128, 256], pl.BF16] = pl.tensor.slice(wo_a_flat_inline2521, [128, 256], [pa_wrow_inline2533, 0])
                        pa_acc_inline2537: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(pa_x0_inline2535, pa_w0_inline2523, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        for pa_k0_inline2561 in pl.pipeline(256, 4096, 256, stage=2):
                            pa_xk_inline2539: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(attn_2d_inline2548, [128, 256], [pa_src_inline2495, pa_k0_inline2561], [pa_rows_inline2532, 256])
                            pa_wk_inline2540: pl.Tensor[[128, 256], pl.BF16] = pl.tensor.slice(wo_a_flat_inline2521, [128, 256], [pa_wrow_inline2533, pa_k0_inline2561])
                            pa_acc_inline2537: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul_acc(pa_acc_inline2537, pa_xk_inline2539, pa_wk_inline2540, a_trans=False, b_trans=True)
                        pa_valid_inline2542: pl.Tensor[[128, 128], pl.FP32, pl.TensorView(valid_shape=[pa_rows, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(pa_acc_inline2537, pa_rows_inline2532, 128)
                        own_a_fp32_inline2500: pl.Tensor[[256, 4096], pl.FP32] = pl.tensor.assemble(own_a_fp32_inline2500, pa_valid_inline2542, [pa_t0_inline2489, pa_wrow_inline2533])
                    for qz_worker_inline2544 in pl.spmd(6, name_hint="tp_o_a_quant_spmd"):
                        for qz_blk_inline2546 in pl.range(qz_worker_inline2544, own_quant_blocks_inline2490, 6):
                            qz_t_inline2547: pl.Scalar[pl.INDEX] = qz_blk_inline2546 * 8
                            qz_rows_inline2501: pl.Scalar[pl.INDEX] = pl.min(8, 256 - qz_t_inline2547)
                            qz_tile_inline2560: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.slice(own_a_fp32_inline2500, [8, 1024], [qz_t_inline2547, o_a_col_inline2496])
                            qz_amax_inline2524: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(pl.tensor.abs(qz_tile_inline2560)), [1, 8])
                            qz_floor_inline2536: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0001)
                            qz_amax_v1_inline2549: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qz_floor_inline2536, qz_amax_inline2524)
                            qz_max_inline2512: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=127.0)
                            qz_sq_inline2527: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.div(qz_max_inline2512, qz_amax_v1_inline2549)
                            qz_sdq_inline2550: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.recip(qz_sq_inline2527)
                            own_scale_inline2494: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.assemble(own_scale_inline2494, pl.tensor.set_validshape(qz_sdq_inline2550, 1, qz_rows_inline2501), [local_group_inline2514, qz_t_inline2547])
                            qz_sq_col_inline2519: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qz_sq_inline2527, [8, 1])
                            qz_scaled_inline2538: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.row_expand_mul(qz_tile_inline2560, qz_sq_col_inline2519)
                            qz_i32_inline2551: pl.Tensor[[8, 1024], pl.INT32] = pl.tensor.cast(qz_scaled_inline2538, target_type=pl.INT32, mode='rint')
                            qz_f16_inline2552: pl.Tensor[[8, 1024], pl.FP16] = pl.tensor.cast(qz_i32_inline2551, target_type=pl.FP16, mode='round')
                            qz_i8_inline2503: pl.Tensor[[8, 1024], pl.INT8] = pl.tensor.cast(qz_f16_inline2552, target_type=pl.INT8, mode='trunc')
                            own_a_i8_inline2497: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.assemble(own_a_i8_inline2497, pl.tensor.set_validshape(qz_i8_inline2503, qz_rows_inline2501, 1024), [qz_t_inline2547, o_a_col_inline2496])
                        for qz_pad_inline2554 in pl.range(own_quant_blocks_inline2490 + qz_worker_inline2544, own_pad_blocks_inline2511, 6):
                            qz_pt_inline2558: pl.Scalar[pl.INDEX] = qz_pad_inline2554 * 8
                            qz_prows_inline2562: pl.Scalar[pl.INDEX] = pl.min(8, own_b_t_inline2531 - qz_pt_inline2558)
                            qz_zero_inline2563: pl.Tensor[[8, 1024], pl.FP16] = pl.tensor.full([8, 1024], dtype=pl.FP16, value=0.0)
                            qz_zero_i8_inline2492: pl.Tensor[[8, 1024], pl.INT8] = pl.tensor.cast(qz_zero_inline2563, target_type=pl.INT8, mode='trunc')
                            own_a_i8_inline2497: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.assemble(own_a_i8_inline2497, pl.tensor.set_validshape(qz_zero_i8_inline2492, qz_prows_inline2562, 1024), [qz_pt_inline2558, o_a_col_inline2496])
                    for pb_unit_inline2483 in pl.spmd(own_b_rows_inline2529 * 8, name_hint="tp_o_b_spmd"):
                        pb_tb_inline2510: pl.Scalar[pl.INDEX] = pb_unit_inline2483 // 8
                        pb_db_inline2480: pl.Scalar[pl.INDEX] = pb_unit_inline2483 - pb_tb_inline2510 * 8
                        pb_t0_inline2557: pl.Scalar[pl.INDEX] = pb_tb_inline2510 * 128
                        pb_d0_inline2479: pl.Scalar[pl.INDEX] = pb_db_inline2480 * 512
                        for pb_n0_inline2478 in pl.range(pb_d0_inline2479, pb_d0_inline2479 + 512, 256):
                            pb_x0_inline2498: pl.Tensor[[128, 256], pl.INT8] = pl.tensor.slice(own_a_i8_inline2497, [128, 256], [pb_t0_inline2557, o_a_col_inline2496])
                            pb_w0_inline2522: pl.Tensor[[256, 256], pl.INT8] = pl.tensor.slice(wo_b, [256, 256], [pb_n0_inline2478, o_a_col_inline2496])
                            pb_acc_inline2507: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul(pb_x0_inline2498, pb_w0_inline2522, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.INT32)
                            for pb_k0_inline2477 in pl.pipeline(256, 1024, 256, stage=2):
                                pb_bk_inline2515: pl.Scalar[pl.INDEX] = o_a_col_inline2496 + pb_k0_inline2477
                                pb_xk_inline2520: pl.Tensor[[128, 256], pl.INT8] = pl.tensor.slice(own_a_i8_inline2497, [128, 256], [pb_t0_inline2557, pb_bk_inline2515])
                                pb_wk_inline2476: pl.Tensor[[256, 256], pl.INT8] = pl.tensor.slice(wo_b, [256, 256], [pb_n0_inline2478, pb_bk_inline2515])
                                pb_acc_inline2507: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul_acc(pb_acc_inline2507, pb_xk_inline2520, pb_wk_inline2476, a_trans=False, b_trans=True)
                            pb_col_inline2474: pl.Scalar[pl.INDEX] = local_group_inline2514 * 4096 + pb_n0_inline2478
                            own_b_i32_inline2528: pl.Tensor[[256, 16384], pl.INT32] = pl.tensor.assemble(own_b_i32_inline2528, pb_acc_inline2507, [pb_t0_inline2557, pb_col_inline2474])
                for dq_worker_inline2469 in pl.spmd(12, name_hint="tp_o_b_dequant_spmd", optimizations=[pl.cross_core_slot(slot_num=2)]):
                    for dq_blk_inline2556 in pl.range(dq_worker_inline2469, own_act_rows_inline2505 * 8, 12):
                        dq_rb_inline2484: pl.Scalar[pl.INDEX] = dq_blk_inline2556 // 8
                        dq_nb_inline2468: pl.Scalar[pl.INDEX] = dq_blk_inline2556 - dq_rb_inline2484 * 8
                        dq_row_inline2467: pl.Scalar[pl.INDEX] = dq_rb_inline2484 * 16
                        dq_n0_inline2545: pl.Scalar[pl.INDEX] = dq_nb_inline2468 * 512
                        dq_rows_inline2466: pl.Scalar[pl.INDEX] = pl.min(16, 256 - dq_row_inline2467)
                        dq_acc_inline2464: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                        for dq_group_inline2463 in pl.pipeline(4, stage=2):
                            dq_col_inline2488: pl.Scalar[pl.INDEX] = dq_group_inline2463 * 4096 + dq_n0_inline2545
                            dq_i32_inline2461: pl.Tensor[[16, 512], pl.INT32, pl.TensorView(valid_shape=[dq_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(own_b_i32_inline2528, [16, 512], [dq_row_inline2467, dq_col_inline2488], [dq_rows_inline2466, 512])
                            dq_fp32_inline2459: pl.Tensor[[16, 512], pl.FP32, pl.TensorView(valid_shape=[dq_rows, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.cast(dq_i32_inline2461, target_type=pl.FP32, mode='none')
                            dq_srow_inline2518: pl.Tensor[[1, 16], pl.FP32, pl.TensorView(valid_shape=[1, dq_rows], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(own_scale_inline2494, [1, 16], [dq_group_inline2463, dq_row_inline2467], [1, dq_rows_inline2466])
                            dq_scol_inline2462: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[dq_rows, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.reshape(dq_srow_inline2518, [16, 1])
                            dq_acc_inline2464: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.add(dq_acc_inline2464, pl.tensor.row_expand_mul(dq_fp32_inline2459, dq_scol_inline2462))
                        dq_wscale_inline2458: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(wo_b_scale, [512], [dq_n0_inline2545]), [1, 512])
                        dq_bf16_inline2457: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.cast(pl.tensor.col_expand_mul(dq_acc_inline2464, dq_wscale_inline2458), target_type=pl.BF16, mode='rint')
                        dq_stage_inline2456: pl.Scalar[pl.INDEX] = owner_inline2504 * 256 + dq_row_inline2467
                        publish_all_inline2525: pl.Tensor[[512, 4096], pl.BF16] = pl.tensor.assemble(publish_all_inline2525, pl.tensor.set_validshape(dq_bf16_inline2457, dq_rows_inline2466, 512), [dq_stage_inline2456, dq_n0_inline2545])
            with pl.spmd(24, name_hint="tp_o_b_publish_spmd") as publish_tid_inline2454:
                pub_worker_inline2453: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for pub_blk_inline2559 in pl.range(pub_worker_inline2453, 2 * put_rows_inline2508, 24):
                    pub_owner_inline2541: pl.Scalar[pl.INDEX] = pub_blk_inline2559 // put_rows_inline2508
                    pub_row_block_inline2460: pl.Scalar[pl.INDEX] = pub_blk_inline2559 - pub_owner_inline2541 * put_rows_inline2508
                    pub_owner_row_inline2482: pl.Scalar[pl.INDEX] = pub_row_block_inline2460 * 8
                    pub_rows_inline2452: pl.Scalar[pl.INDEX] = pl.min(8, 256 - pub_owner_row_inline2482)
                    pub_src_row_inline2486: pl.Scalar[pl.INDEX] = pub_owner_inline2541 * 256 + pub_owner_row_inline2482
                    pub_dst_row_inline2451: pl.Scalar[pl.INDEX] = pl.cast(tp_rank, pl.INDEX) * 256 + pub_owner_row_inline2482
                    pld.tensor.put(o_window, pl.cast(group_base, pl.INDEX) + pub_owner_inline2541, publish_all_inline2525, [pub_dst_row_inline2451, 0], [pub_src_row_inline2486, 0], [pub_rows_inline2452, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                for notify_owner_inline2517 in pl.range(2):
                    if notify_owner_inline2517 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(o_signal, pl.cast(group_base, pl.INDEX) + notify_owner_inline2517, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_rs_wait", deps=[publish_tid_inline2454]) as wait_tid_inline2450:
                expected_inline2475: pl.Scalar[pl.INT32] = pl.cast(24, pl.INT32)
                for source_tp_inline2449 in pl.range(2):
                    if source_tp_inline2449 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.wait(o_signal, [source_tp_inline2449, 0], expected_inline2475, cmp=1)
            with pl.spmd(48, name_hint="tp_o_rs_reduce_spmd", deps=[wait_tid_inline2450]) as reduce_tid_inline2543:
                worker_inline2455: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for block_inline2473 in pl.range(worker_inline2455, 256, 48):
                    local_row_inline2555: pl.Scalar[pl.INDEX] = block_inline2473 // 1
                    d_block_inline2465: pl.Scalar[pl.INDEX] = block_inline2473 - local_row_inline2555 * 1
                    d0_inline2447: pl.Scalar[pl.INDEX] = d_block_inline2465 * 4096
                    own_partial_inline2446: pl.Tile[[1, 4096], pl.BF16] = pl.tile.load(o_window, [local_row_inline2555, d0_inline2447], [1, 4096], [1, 4096])
                    reduce_acc_inline2445: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(own_partial_inline2446, target_type=pl.FP32, mode='none')
                    for source_tp_inline2444 in pl.range(1, 2):
                        source_row_inline2481: pl.Scalar[pl.INDEX] = source_tp_inline2444 * 256 + local_row_inline2555
                        source_partial_inline2443: pl.Tile[[1, 4096], pl.BF16] = pl.tile.load(o_window, [source_row_inline2481, d0_inline2447], [1, 4096], [1, 4096])
                        source_fp32_inline2506: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(source_partial_inline2443, target_type=pl.FP32, mode='none')
                        reduce_acc_inline2445: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(reduce_acc_inline2445, source_fp32_inline2506)
                    reduced_inline2553: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(reduce_acc_inline2445, target_type=pl.BF16, mode='rint')
                    pl.tile.store(reduced_inline2553, [local_row_inline2555, d0_inline2447], attn_out_inline1284)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_rs_complete", deps=[reduce_tid_inline2543]):
                completion_anchor_inline2442: pl.Scalar[pl.BF16] = pl.tensor.read(attn_out_inline1284, [0, 0])
                for peer_tp_inline2472 in pl.range(2):
                    if peer_tp_inline2472 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(o_signal, pl.cast(group_base, pl.INDEX) + peer_tp_inline2472, [tp_rank, 0], pl.const(1, pl.INT32), op=0)
                completion_expected_inline2441: pl.Scalar[pl.INT32] = pl.cast(25, pl.INT32)
                for source_tp_inline2448 in pl.range(2):
                    if source_tp_inline2448 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.wait(o_signal, [source_tp_inline2448, 0], completion_expected_inline2441, cmp=1)
                reset_value_inline2471: pl.Scalar[pl.INT32] = pl.cast(-25, pl.INT32)
                self_rank_inline2440: pl.Scalar[pl.INT32] = group_base + tp_rank
                for source_tp_inline2470 in pl.range(2):
                    if source_tp_inline2470 != pl.cast(tp_rank, pl.INDEX):
                        pld.system.notify(o_signal, self_rank_inline2440, [source_tp_inline2470, 0], reset_value_inline2471, op=0)
                pl.tensor.write(attn_out_inline1284, [0, 0], completion_anchor_inline2442)
            _o_reduced_inline1269: pl.Tensor[[t_dim, 4096], pl.BF16] = attn_out_inline1284
            o_signal: pld.DistributedTensor[[2, 1], pl.INT32] = o_signal
        with pl.scope():
            t_dim_inline2576: pl.Scalar[pl.INDEX] = pl.tensor.dim(attn_out_inline1284, 0)
            residual_flat_inline2567: pl.Tensor[[t_dim_6, 16384], pl.FP32] = pl.tensor.reshape(x_hc, [t_dim_inline2576, 16384])
            y_flat_inline2568: pl.Tensor[[t_dim_6, 16384], pl.FP32] = pl.tensor.reshape(x_out, [t_dim_inline2576, 16384])
            token_tiles_inline2571: pl.Scalar[pl.INDEX] = (t_dim_inline2576 + 4 - 1) // 4
            for token_block_inline2572 in pl.spmd(token_tiles_inline2571, name_hint="hc_post_spmd"):
                t0_inline2573: pl.Scalar[pl.INDEX] = token_block_inline2572 * 4
                for t_inline2574 in pl.pipeline(t0_inline2573, t0_inline2573 + 4, stage=2):
                    if t_inline2574 < t_dim_inline2576:
                        x_row_inline2578: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.cast(pl.tensor.slice(attn_out_inline1284, [1, 4096], [t_inline2574, 0]), target_type=pl.FP32, mode='round')
                        for out_h_inline2575 in pl.unroll(4):
                            post_w_inline2577: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277, [t_inline2574, out_h_inline2575])
                            y_row_inline2569: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(x_row_inline2578, post_w_inline2577)
                            for in_h_inline2570 in pl.pipeline(4, stage=4):
                                comb_w_inline2579: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267, [t_inline2574, in_h_inline2570 * 4 + out_h_inline2575])
                                res_d_inline2566: pl.Scalar[pl.INDEX] = in_h_inline2570 * 4096
                                res_row_inline2565: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.slice(residual_flat_inline2567, [1, 4096], [t_inline2574, res_d_inline2566])
                                weighted_inline2564: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(res_row_inline2565, comb_w_inline2579)
                                y_row_inline2569: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(y_row_inline2569, weighted_inline2564)
                            y_flat_inline2568: pl.Tensor[[t_dim_6, 16384], pl.FP32] = pl.tensor.assemble(y_flat_inline2568, y_row_inline2569, [t_inline2574, out_h_inline2575 * 4096])
        return x_out
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
