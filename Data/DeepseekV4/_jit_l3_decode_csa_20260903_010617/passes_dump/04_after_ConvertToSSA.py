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
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_topk_group_wave(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], score_arena__ssa_v0: pl.Tensor[[T_DYN, 262144], pl.FP32], pair_arena__ssa_v0: pl.Tensor[[4192, 1024], pl.FP32]):
        # Reduce globally striped two-leaf subtrees into compact roots.
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        global_group_base__ssa_v0: pl.Scalar[pl.INDEX] = 0
        for query__idx_v0, (global_group_base__iter_v1,) in pl.range(query_count__ssa_v0, init_values=(global_group_base__ssa_v0,)):
            batch_idx__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 // 8
            position__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query__idx_v0])
            cache_len__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx__ssa_v0]), pl.INDEX) // 4
            visible_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len__ssa_v0, (pl.cast(position__ssa_v0, pl.INDEX) + 1) // 4), 262144), 0)
            leaf_count__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count__ssa_v0 + 8192 - 1) // 8192
            group_count__ssa_v0: pl.Scalar[pl.INDEX] = (leaf_count__ssa_v0 + 2 - 1) // 2
            base_mod__ssa_v0: pl.Scalar[pl.INDEX] = global_group_base__iter_v1 % 48
            first_group__ssa_v0: pl.Scalar[pl.INDEX] = (worker__ssa_v0 + base_mod__ssa_v0) % 48
            for group__idx_v0 in pl.range(first_group__ssa_v0, group_count__ssa_v0, 48):
                leaf_begin__ssa_v0: pl.Scalar[pl.INDEX] = group__idx_v0 * 2
                group_leaf_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(2, leaf_count__ssa_v0 - leaf_begin__ssa_v0)
                group_root_slot__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 * 16 + group__idx_v0
                if group_leaf_count__ssa_v0 == 1:
                    logical_begin__ssa_v0: pl.Scalar[pl.INDEX] = leaf_begin__ssa_v0 * 8192
                    valid_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count__ssa_v0 - logical_begin__ssa_v0)
                    # Sort one scored 8K leaf and store its exact Top-512 pair row.
                    logical_begin_i32_inline126__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(logical_begin__ssa_v0, pl.INT32)
                    leaf_indices_inline125__ssa_v0: pl.Tile[[1, 8192], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False), logical_begin_i32_inline126__ssa_v0)
                    leaf_scores_raw_inline124__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.TileView(valid_shape=[1, valid_count__ssa_v0])] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, logical_begin__ssa_v0], [1, 8192], [1, valid_count__ssa_v0])
                    leaf_scores_inline121__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw_inline124__ssa_v0, pad_value=pl.PadValue.min)
                    leaf_scores_v1_inline118__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores_inline121__ssa_v0, pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38))
                    pairs_inline120__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1_inline118__ssa_v0, pl.tile.reinterpret_view(leaf_indices_inline125__ssa_v0, dtype=pl.UINT32))
                    pairs_v1_inline119__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_inline120__ssa_v0, pl.const(64, pl.INT32))
                    pairs_v2_inline123__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1_inline119__ssa_v0, pl.const(256, pl.INT32))
                    pairs_v3_inline117__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2_inline123__ssa_v0, pl.const(1024, pl.INT32))
                    pairs_v4_inline122__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3_inline117__ssa_v0, pl.const(4096, pl.INT32))
                    pl.tile.store(pl.tile.slice(pairs_v4_inline122__ssa_v0, [1, 1024], [0, 0]), [group_root_slot__ssa_v0, 0], pair_arena__ssa_v0)
                    logical_begin__phi_v3, valid_count__phi_v3 = pl.yield_(logical_begin__ssa_v0, valid_count__ssa_v0)
                else:
                    scratch_base__ssa_v0: pl.Scalar[pl.INDEX] = 4096 + worker__ssa_v0 * 2
                    leaf__ssa_v0: pl.Scalar[pl.INDEX] = leaf_begin__ssa_v0 + 0
                    logical_begin__ssa_v1: pl.Scalar[pl.INDEX] = leaf__ssa_v0 * 8192
                    valid_count__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count__ssa_v0 - logical_begin__ssa_v1)
                    # Sort one scored 8K leaf and store its exact Top-512 pair row.
                    logical_begin_i32_inline136__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(logical_begin__ssa_v1, pl.INT32)
                    leaf_indices_inline135__ssa_v0: pl.Tile[[1, 8192], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False), logical_begin_i32_inline136__ssa_v0)
                    leaf_scores_raw_inline134__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.TileView(valid_shape=[1, valid_count__ssa_v1])] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, logical_begin__ssa_v1], [1, 8192], [1, valid_count__ssa_v1])
                    leaf_scores_inline131__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw_inline134__ssa_v0, pad_value=pl.PadValue.min)
                    leaf_scores_v1_inline128__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores_inline131__ssa_v0, pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38))
                    pairs_inline130__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1_inline128__ssa_v0, pl.tile.reinterpret_view(leaf_indices_inline135__ssa_v0, dtype=pl.UINT32))
                    pairs_v1_inline129__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_inline130__ssa_v0, pl.const(64, pl.INT32))
                    pairs_v2_inline133__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1_inline129__ssa_v0, pl.const(256, pl.INT32))
                    pairs_v3_inline127__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2_inline133__ssa_v0, pl.const(1024, pl.INT32))
                    pairs_v4_inline132__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3_inline127__ssa_v0, pl.const(4096, pl.INT32))
                    pl.tile.store(pl.tile.slice(pairs_v4_inline132__ssa_v0, [1, 1024], [0, 0]), [scratch_base__ssa_v0 + 0, 0], pair_arena__ssa_v0)
                    leaf__ssa_v1: pl.Scalar[pl.INDEX] = leaf_begin__ssa_v0 + 1
                    logical_begin__ssa_v2: pl.Scalar[pl.INDEX] = leaf__ssa_v1 * 8192
                    valid_count__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count__ssa_v0 - logical_begin__ssa_v2)
                    # Sort one scored 8K leaf and store its exact Top-512 pair row.
                    logical_begin_i32_inline136__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(logical_begin__ssa_v2, pl.INT32)
                    leaf_indices_inline135__ssa_v1: pl.Tile[[1, 8192], pl.INT32, pl.Mem.Vec] = pl.tile.adds(pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False), logical_begin_i32_inline136__ssa_v1)
                    leaf_scores_raw_inline134__ssa_v1: pl.Tile[[1, 8192], pl.FP32, pl.TileView(valid_shape=[1, valid_count__ssa_v2])] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, logical_begin__ssa_v2], [1, 8192], [1, valid_count__ssa_v2])
                    leaf_scores_inline131__ssa_v1: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw_inline134__ssa_v1, pad_value=pl.PadValue.min)
                    leaf_scores_v1_inline128__ssa_v1: pl.Tile[[1, 8192], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores_inline131__ssa_v1, pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38))
                    pairs_inline130__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1_inline128__ssa_v1, pl.tile.reinterpret_view(leaf_indices_inline135__ssa_v1, dtype=pl.UINT32))
                    pairs_v1_inline129__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_inline130__ssa_v1, pl.const(64, pl.INT32))
                    pairs_v2_inline133__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1_inline129__ssa_v1, pl.const(256, pl.INT32))
                    pairs_v3_inline127__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2_inline133__ssa_v1, pl.const(1024, pl.INT32))
                    pairs_v4_inline132__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3_inline127__ssa_v1, pl.const(4096, pl.INT32))
                    pl.tile.store(pl.tile.slice(pairs_v4_inline132__ssa_v1, [1, 1024], [0, 0]), [scratch_base__ssa_v0 + 1, 0], pair_arena__ssa_v0)
                    # Merge two arena rows and store their exact Top-512 pair row.
                    left_inline141__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [scratch_base__ssa_v0, 0], [1, 1024], [1, 1024])
                    right_inline140__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [scratch_base__ssa_v0 + 1, 0], [1, 1024], [1, 1024])
                    merge_tmp_inline138__ssa_v0: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                    merged_all_inline139__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline141__ssa_v0, right_inline140__ssa_v0, merge_tmp_inline138__ssa_v0, exhausted=False)
                    merged_inline137__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline139__ssa_v0, [1, 1024], [0, 0])
                    pl.tile.store(merged_inline137__ssa_v0, [group_root_slot__ssa_v0, 0], pair_arena__ssa_v0)
                    logical_begin__phi_v3, valid_count__phi_v3 = pl.yield_(logical_begin__ssa_v2, valid_count__ssa_v2)
            global_group_base__ssa_v3: pl.Scalar[pl.INDEX] = global_group_base__iter_v1 + group_count__ssa_v0
            global_group_base__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(global_group_base__ssa_v3)
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def indexer_topk_query_merge(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], pair_arena__ssa_v0: pl.Tensor[[4192, 1024], pl.FP32], topk_scores__ssa_v0: pl.Tensor[[T_DYN, 512], pl.FP32], topk_indices__ssa_v0: pl.Tensor[[T_DYN, 512], pl.INT32]):
        # Merge compact group roots and materialize each query's Top-512.
        query__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        batch_idx__ssa_v0: pl.Scalar[pl.INDEX] = query__ssa_v0 // 8
        position__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query__ssa_v0])
        cache_len__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx__ssa_v0]), pl.INDEX) // 4
        visible_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.min(cache_len__ssa_v0, (pl.cast(position__ssa_v0, pl.INDEX) + 1) // 4), 262144)
        pl.tile.store(pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38), [query__ssa_v0, 0], topk_scores__ssa_v0)
        pl.tile.store(pl.tile.full([1, 512], dtype=pl.INT32, value=-1), [query__ssa_v0, 0], topk_indices__ssa_v0)
        if visible_count__ssa_v0 > 0:
            leaf_count__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count__ssa_v0 + 8192 - 1) // 8192
            group_count__ssa_v0: pl.Scalar[pl.INDEX] = (leaf_count__ssa_v0 + 2 - 1) // 2
            arena_base__ssa_v0: pl.Scalar[pl.INDEX] = query__ssa_v0 * 16
            if group_count__ssa_v0 > 1:
                level1_count__ssa_v0: pl.Scalar[pl.INDEX] = (group_count__ssa_v0 + 1) // 2
                # Reduce one exact-Top-K forest level, forwarding an odd final node.
                output_count_inline10__ssa_v0: pl.Scalar[pl.INDEX] = (group_count__ssa_v0 + 1) // 2
                for output_inline9__idx_v0 in pl.range(output_count_inline10__ssa_v0):
                    left_slot_inline8__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + 0 + 2 * output_inline9__idx_v0
                    right_slot_inline6__ssa_v0: pl.Scalar[pl.INDEX] = left_slot_inline8__ssa_v0 + 1
                    output_slot_inline7__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + 0 + output_inline9__idx_v0
                    if right_slot_inline6__ssa_v0 < arena_base__ssa_v0 + 0 + group_count__ssa_v0:
                        # Merge two arena rows and store their exact Top-512 pair row.
                        left_inline1329__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline8__ssa_v0, 0], [1, 1024], [1, 1024])
                        right_inline1328__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [right_slot_inline6__ssa_v0, 0], [1, 1024], [1, 1024])
                        merge_tmp_inline1326__ssa_v0: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                        merged_all_inline1327__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1329__ssa_v0, right_inline1328__ssa_v0, merge_tmp_inline1326__ssa_v0, exhausted=False)
                        merged_inline1325__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline1327__ssa_v0, [1, 1024], [0, 0])
                        pl.tile.store(merged_inline1325__ssa_v0, [output_slot_inline7__ssa_v0, 0], pair_arena__ssa_v0)
                    else:
                        forwarded_inline5__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline8__ssa_v0, 0], [1, 1024], [1, 1024])
                        pl.tile.store(forwarded_inline5__ssa_v0, [output_slot_inline7__ssa_v0, 0], pair_arena__ssa_v0)
                if level1_count__ssa_v0 > 1:
                    level2_count__ssa_v0: pl.Scalar[pl.INDEX] = (level1_count__ssa_v0 + 1) // 2
                    # Reduce one exact-Top-K forest level, forwarding an odd final node.
                    output_count_inline16__ssa_v0: pl.Scalar[pl.INDEX] = (level1_count__ssa_v0 + 1) // 2
                    for output_inline15__idx_v0 in pl.range(output_count_inline16__ssa_v0):
                        left_slot_inline14__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + 0 + 2 * output_inline15__idx_v0
                        right_slot_inline12__ssa_v0: pl.Scalar[pl.INDEX] = left_slot_inline14__ssa_v0 + 1
                        output_slot_inline13__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + 0 + output_inline15__idx_v0
                        if right_slot_inline12__ssa_v0 < arena_base__ssa_v0 + 0 + level1_count__ssa_v0:
                            # Merge two arena rows and store their exact Top-512 pair row.
                            left_inline1334__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline14__ssa_v0, 0], [1, 1024], [1, 1024])
                            right_inline1333__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [right_slot_inline12__ssa_v0, 0], [1, 1024], [1, 1024])
                            merge_tmp_inline1331__ssa_v0: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                            merged_all_inline1332__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1334__ssa_v0, right_inline1333__ssa_v0, merge_tmp_inline1331__ssa_v0, exhausted=False)
                            merged_inline1330__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline1332__ssa_v0, [1, 1024], [0, 0])
                            pl.tile.store(merged_inline1330__ssa_v0, [output_slot_inline13__ssa_v0, 0], pair_arena__ssa_v0)
                        else:
                            forwarded_inline11__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline14__ssa_v0, 0], [1, 1024], [1, 1024])
                            pl.tile.store(forwarded_inline11__ssa_v0, [output_slot_inline13__ssa_v0, 0], pair_arena__ssa_v0)
                    if level2_count__ssa_v0 > 1:
                        level3_count__ssa_v0: pl.Scalar[pl.INDEX] = (level2_count__ssa_v0 + 1) // 2
                        # Reduce one exact-Top-K forest level, forwarding an odd final node.
                        output_count_inline22__ssa_v0: pl.Scalar[pl.INDEX] = (level2_count__ssa_v0 + 1) // 2
                        for output_inline21__idx_v0 in pl.range(output_count_inline22__ssa_v0):
                            left_slot_inline20__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + 0 + 2 * output_inline21__idx_v0
                            right_slot_inline18__ssa_v0: pl.Scalar[pl.INDEX] = left_slot_inline20__ssa_v0 + 1
                            output_slot_inline19__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + 0 + output_inline21__idx_v0
                            if right_slot_inline18__ssa_v0 < arena_base__ssa_v0 + 0 + level2_count__ssa_v0:
                                # Merge two arena rows and store their exact Top-512 pair row.
                                left_inline1339__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline20__ssa_v0, 0], [1, 1024], [1, 1024])
                                right_inline1338__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [right_slot_inline18__ssa_v0, 0], [1, 1024], [1, 1024])
                                merge_tmp_inline1336__ssa_v0: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                                merged_all_inline1337__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1339__ssa_v0, right_inline1338__ssa_v0, merge_tmp_inline1336__ssa_v0, exhausted=False)
                                merged_inline1335__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline1337__ssa_v0, [1, 1024], [0, 0])
                                pl.tile.store(merged_inline1335__ssa_v0, [output_slot_inline19__ssa_v0, 0], pair_arena__ssa_v0)
                            else:
                                forwarded_inline17__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline20__ssa_v0, 0], [1, 1024], [1, 1024])
                                pl.tile.store(forwarded_inline17__ssa_v0, [output_slot_inline19__ssa_v0, 0], pair_arena__ssa_v0)
                        if level3_count__ssa_v0 > 1:
                            # Reduce one exact-Top-K forest level, forwarding an odd final node.
                            output_count_inline28__ssa_v0: pl.Scalar[pl.INDEX] = (level3_count__ssa_v0 + 1) // 2
                            for output_inline27__idx_v0 in pl.range(output_count_inline28__ssa_v0):
                                left_slot_inline26__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + 0 + 2 * output_inline27__idx_v0
                                right_slot_inline24__ssa_v0: pl.Scalar[pl.INDEX] = left_slot_inline26__ssa_v0 + 1
                                output_slot_inline25__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + 0 + output_inline27__idx_v0
                                if right_slot_inline24__ssa_v0 < arena_base__ssa_v0 + 0 + level3_count__ssa_v0:
                                    # Merge two arena rows and store their exact Top-512 pair row.
                                    left_inline1344__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline26__ssa_v0, 0], [1, 1024], [1, 1024])
                                    right_inline1343__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [right_slot_inline24__ssa_v0, 0], [1, 1024], [1, 1024])
                                    merge_tmp_inline1341__ssa_v0: pl.Tile[[1, 2048], pl.FP32] = pl.tile.create([1, 2048], dtype=pl.FP32)
                                    merged_all_inline1342__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1344__ssa_v0, right_inline1343__ssa_v0, merge_tmp_inline1341__ssa_v0, exhausted=False)
                                    merged_inline1340__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.Mem.Vec] = pl.tile.slice(merged_all_inline1342__ssa_v0, [1, 1024], [0, 0])
                                    pl.tile.store(merged_inline1340__ssa_v0, [output_slot_inline25__ssa_v0, 0], pair_arena__ssa_v0)
                                else:
                                    forwarded_inline23__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline26__ssa_v0, 0], [1, 1024], [1, 1024])
                                    pl.tile.store(forwarded_inline23__ssa_v0, [output_slot_inline25__ssa_v0, 0], pair_arena__ssa_v0)
            root_slot__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0
            root_pairs__ssa_v0: pl.Tile[[1, 1024], pl.FP32] = pl.tile.load(pair_arena__ssa_v0, [root_slot__ssa_v0, 0], [1, 1024], [1, 1024])
            pl.tile.store(pl.tile.gather_mask(root_pairs__ssa_v0, mask_pattern=1, output_dtype=pl.FP32), [query__ssa_v0, 0], topk_scores__ssa_v0)
            root_indices__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.gather_mask(root_pairs__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
            output_indices__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
            valid_topk__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count__ssa_v0, 512)
            for lane__idx_v0 in pl.range(valid_topk__ssa_v0):
                pl.tile.write(output_indices__ssa_v0, [0, lane__idx_v0], pl.tile.read(root_indices__ssa_v0, [0, lane__idx_v0]))
            pl.tile.store(output_indices__ssa_v0, [query__ssa_v0, 0], topk_indices__ssa_v0)
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def decode_csa_test(self, x_hc__ssa_v0: pl.Tensor[[T_DYN, 4, 4096], pl.FP32], hc_attn_fn__ssa_v0: pl.Tensor[[24, 16384], pl.FP32], hc_attn_scale__ssa_v0: pl.Tensor[[3], pl.FP32], hc_attn_base__ssa_v0: pl.Tensor[[24], pl.FP32], attn_norm_w__ssa_v0: pl.Tensor[[4096], pl.BF16], wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16], wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8], wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32], wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16], gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16], gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16], freqs_cos_local__ssa_v0: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_sin_local__ssa_v0: pl.Tensor[[T_DYN, 64], pl.BF16], freqs_cos__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16], freqs_sin__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_freqs_cos__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_freqs_sin__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16], cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16], cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32], cmp_norm_w__ssa_v0: pl.Tensor[[512], pl.BF16], compress_state__ssa_v0: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32]], compress_state_block_table__ssa_v0: pl.Tensor[[KV_B_DYN, 4], pl.INT32], idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8], idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32], weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16], hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16], inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16], inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32], inner_norm_w__ssa_v0: pl.Tensor[[128], pl.BF16], inner_compress_state__ssa_v0: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32]], inner_compress_state_block_table__ssa_v0: pl.Tensor[[KV_B_DYN, 4], pl.INT32], kv_cache__ssa_v0: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_kv__ssa_v0: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, 8192], pl.INT32], idx_kv_cache__ssa_v0: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8]], idx_kv_scale__ssa_v0: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32]], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, 8192], pl.INT32], ori_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64], window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32], window_swa_lens__ssa_v0: pl.Tensor[[T_DYN], pl.INT32], cmp_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64], idx_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64], state_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64], inner_state_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64], position_ids_local__ssa_v0: pl.Tensor[[T_DYN], pl.INT32], position_ids__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT32], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32], attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32], wo_a__ssa_v0: pl.Tensor[[4, 1024, 4096], pl.BF16], wo_b__ssa_v0: pl.Tensor[[4096, 4096], pl.INT8], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32], x_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4, 4096], pl.FP32]], gather_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16], gather_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32], attention_window__ssa_v0: pld.DistributedTensor[[2048, 4096], pl.BF16], attention_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32], o_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16], o_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32], group_base__ssa_v0: pl.Scalar[pl.INT32], tp_rank__ssa_v0: pl.Scalar[pl.INT32], local_t__ssa_v0: pl.Scalar[pl.INT32]):
        # Run one rank of the context-parallel CSA layer.
        t_dim_inline1251__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc__ssa_v0, 0)
        kv_dim_inline1261__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(ori_slot_mapping__ssa_v0, 0)
        kv_b_dim_inline1264__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table__ssa_v0, 0)
        q_inline1246__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64, 512], pl.BF16] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        idx_topk_scores_inline1271__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.FP32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        idx_topk_inline1280__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.INT32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
        post_t_inline1277__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 4], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        comb_t_inline1267__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        x_mixed_inline1253__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16] = pl.tensor.create([t_dim_inline1251__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            # One pl.spmd task per work-type, ordered by their GM read/write dependencies.
            # 
            #         rms -> linear -> linear_reduce -> split_pre_post / comb_sinkhorn / mix_x. Cross-scope
            #         buffers are sized to t_linear, the token count padded up to whole 16-row cube tiles.
            #         
            t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc__ssa_v0, 0)
            token_tiles_inline1492__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1568__ssa_v0 + 8 - 1) // 8
            t_linear_inline1486__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1568__ssa_v0 + 16 - 1) // 16 * 16
            x_flat_inline1497__ssa_v0: pl.Tensor[[t_dim_inline1568__ssa_v0, 16384], pl.FP32] = pl.tensor.reshape(x_hc__ssa_v0, [t_dim_inline1568__ssa_v0, 16384])
            scale0_inline1499__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale__ssa_v0, [0])
            scale1_inline1530__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale__ssa_v0, [1])
            scale2_inline1480__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale__ssa_v0, [2])
            hc_base_2d_inline1467__ssa_v0: pl.Tensor[[1, 24], pl.FP32] = pl.tensor.reshape(hc_attn_base__ssa_v0, [1, 24])
            inv_rms_inline1463__ssa_v0: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32] = pl.tensor.create([t_linear_inline1486__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for t_inline1518__ssa_v0 in pl.spmd(token_tiles_inline1492__ssa_v0, name_hint="hc_pre_rms_spmd", allow_early_resolve=True):
                t0_inline1476__ssa_v0: pl.Scalar[pl.INDEX] = t_inline1518__ssa_v0 * 8
                valid_rows_inline1507__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v0)
                sq_sum_inline1490__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                for kb_inline1493__idx_v0, (sq_sum_inline1490__iter_v1,) in pl.pipeline(32, stage=4, init_values=(sq_sum_inline1490__ssa_v0,)):
                    k0_inline1501__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline1493__idx_v0 * 512
                    if valid_rows_inline1507__ssa_v0 == 8:
                        x_chunk_full_inline1502__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.slice(x_flat_inline1497__ssa_v0, [8, 512], [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0])
                        x_sq_full_inline1510__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(x_chunk_full_inline1502__ssa_v0, x_chunk_full_inline1502__ssa_v0)
                        x_sq_row_full_inline1511__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(x_sq_full_inline1510__ssa_v0), [1, 8])
                        sq_sum_inline1490__ssa_v3: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(sq_sum_inline1490__iter_v1, x_sq_row_full_inline1511__ssa_v0)
                        sq_sum_inline1490__phi_v5: pl.Tensor[[1, 8], pl.FP32] = pl.yield_(sq_sum_inline1490__ssa_v3)
                    else:
                        x_chunk_tail_inline1513__ssa_v0: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[valid_rows_inline1507__ssa_v0, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497__ssa_v0, [8, 512], [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0], [valid_rows_inline1507__ssa_v0, 512])
                        x_sq_tail_inline1503__ssa_v0: pl.Tensor[[8, 512], pl.FP32, pl.TensorView(valid_shape=[valid_rows_inline1507__ssa_v0, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.mul(x_chunk_tail_inline1513__ssa_v0, x_chunk_tail_inline1513__ssa_v0)
                        x_sq_row_tail_inline1537__ssa_v0: pl.Tensor[[1, 8], pl.FP32, pl.TensorView(valid_shape=[1, valid_rows_inline1507__ssa_v0], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.reshape(pl.tensor.row_sum(x_sq_tail_inline1503__ssa_v0), [1, 8])
                        sq_sum_inline1490__ssa_v4: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(sq_sum_inline1490__iter_v1, x_sq_row_tail_inline1537__ssa_v0)
                        sq_sum_inline1490__phi_v5: pl.Tensor[[1, 8], pl.FP32] = pl.yield_(sq_sum_inline1490__ssa_v4)
                    sq_sum_inline1490__rv_v2: pl.Tensor[[1, 8], pl.FP32] = pl.yield_(sq_sum_inline1490__phi_v5)
                sq_mean_inline1514__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.adds(pl.tensor.muls(sq_sum_inline1490__rv_v2, 6.103515625e-05), 9.9999999999999995e-07)
                inv_inline1481__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.rsqrt(sq_mean_inline1514__ssa_v0, high_precision=True), [8, 1])
                inv_rms_inline1463__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32] = pl.tensor.assemble(inv_rms_inline1463__ssa_v0, inv_inline1481__ssa_v0, [t0_inline1476__ssa_v0, 0])
            mixes_partials_inline1475__ssa_v0: pl.Tensor[[pl.const(4, pl.INDEX) * t_linear_inline1486__ssa_v0, 32], pl.FP32] = pl.tensor.create([4 * t_linear_inline1486__ssa_v0, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for task_inline1516__ssa_v0 in pl.spmd(t_linear_inline1486__ssa_v0 // 16 * 4, name_hint="hc_pre_linear_spmd", allow_early_resolve=True):
                t0_inline1476__ssa_v1: pl.Scalar[pl.INDEX] = task_inline1516__ssa_v0 // 4 * 16
                linear_split_inline1509__ssa_v0: pl.Scalar[pl.INDEX] = task_inline1516__ssa_v0 % 4
                k_base_inline1494__ssa_v0: pl.Scalar[pl.INDEX] = linear_split_inline1509__ssa_v0 * 4096
                t_rows_inline1520__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v1)
                acc_inline1524__ssa_v0: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for kb_inline1488__idx_v0, (acc_inline1524__iter_v1,) in pl.pipeline(16, stage=2, init_values=(acc_inline1524__ssa_v0,)):
                    k0_inline1501__ssa_v1: pl.Scalar[pl.INDEX] = k_base_inline1494__ssa_v0 + kb_inline1488__idx_v0 * 256
                    x_linear_chunk_inline1459__ssa_v0: pl.Tensor[[16, 256], pl.FP32, pl.TensorView(valid_shape=[t_rows_inline1520__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497__ssa_v0, [16, 256], [t0_inline1476__ssa_v1, k0_inline1501__ssa_v1], [t_rows_inline1520__ssa_v0, 256])
                    w_chunk_inline1519__ssa_v0: pl.Tensor[[32, 256], pl.FP32, pl.TensorView(valid_shape=[24, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(hc_attn_fn__ssa_v0, [32, 256], [0, k0_inline1501__ssa_v1], [24, 256])
                    if kb_inline1488__idx_v0 == 0:
                        acc_inline1524__ssa_v3: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_linear_chunk_inline1459__ssa_v0, w_chunk_inline1519__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        acc_inline1524__phi_v5: pl.Tensor[[16, 32], pl.FP32] = pl.yield_(acc_inline1524__ssa_v3)
                    else:
                        acc_inline1524__ssa_v4: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(acc_inline1524__iter_v1, x_linear_chunk_inline1459__ssa_v0, w_chunk_inline1519__ssa_v0, a_trans=False, b_trans=True)
                        acc_inline1524__phi_v5: pl.Tensor[[16, 32], pl.FP32] = pl.yield_(acc_inline1524__ssa_v4)
                    acc_inline1524__rv_v2: pl.Tensor[[16, 32], pl.FP32] = pl.yield_(acc_inline1524__phi_v5)
                partial_row0_inline1471__ssa_v0: pl.Scalar[pl.INDEX] = linear_split_inline1509__ssa_v0 * t_linear_inline1486__ssa_v0 + t0_inline1476__ssa_v1
                mixes_partials_inline1475__ssa_v1: pl.Tensor[[pl.const(4, pl.INDEX) * t_linear_inline1486__ssa_v0, 32], pl.FP32] = pl.tensor.assemble(mixes_partials_inline1475__ssa_v0, acc_inline1524__rv_v2, [partial_row0_inline1471__ssa_v0, 0])
            mixes_raw_inline1505__ssa_v0: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32] = pl.tensor.create([t_linear_inline1486__ssa_v0, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for linear_block_inline1567__ssa_v0 in pl.spmd(t_linear_inline1486__ssa_v0 // 16, name_hint="hc_pre_linear_reduce_spmd", allow_early_resolve=True):
                linear_t0_inline1457__ssa_v0: pl.Scalar[pl.INDEX] = linear_block_inline1567__ssa_v0 * 16
                mixes_total_inline1570__ssa_v0: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(mixes_partials_inline1475__ssa_v1, [16, 32], [linear_t0_inline1457__ssa_v0, 0])
                for linear_split_inline1456__idx_v0, (mixes_total_inline1570__iter_v1,) in pl.range(1, 4, init_values=(mixes_total_inline1570__ssa_v0,)):
                    partial_t0_inline1464__ssa_v0: pl.Scalar[pl.INDEX] = linear_split_inline1456__idx_v0 * t_linear_inline1486__ssa_v0 + linear_t0_inline1457__ssa_v0
                    partial_tile_inline1561__ssa_v0: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.slice(mixes_partials_inline1475__ssa_v1, [16, 32], [partial_t0_inline1464__ssa_v0, 0])
                    mixes_total_inline1570__ssa_v3: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.add(mixes_total_inline1570__iter_v1, partial_tile_inline1561__ssa_v0)
                    mixes_total_inline1570__rv_v2: pl.Tensor[[16, 32], pl.FP32] = pl.yield_(mixes_total_inline1570__ssa_v3)
                mixes_raw_inline1505__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32] = pl.tensor.assemble(mixes_raw_inline1505__ssa_v0, mixes_total_inline1570__rv_v2, [linear_t0_inline1457__ssa_v0, 0])
            pre_val_store_inline1529__ssa_v0: pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32] = pl.tensor.create([t_linear_inline1486__ssa_v0, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            post_tail_store_inline1544__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for ob_inline1554__ssa_v0 in pl.spmd(token_tiles_inline1492__ssa_v0, name_hint="split_pre_post_spmd", allow_early_resolve=True):
                t0_inline1476__ssa_v2: pl.Scalar[pl.INDEX] = ob_inline1554__ssa_v0 * 8
                valid_rows_inline1507__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v2)
                inv_col_inline1450__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(inv_rms_inline1463__ssa_v1, [8, 1], [t0_inline1476__ssa_v2, 0])
                pre_base_inline1470__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(hc_attn_base__ssa_v0, [8], [0]), [1, 8])
                pre_scaled_inline1447__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(pl.tensor.row_expand_mul(pl.tensor.slice(mixes_raw_inline1505__ssa_v1, [8, 8], [t0_inline1476__ssa_v2, 0]), inv_col_inline1450__ssa_v0), scale0_inline1499__ssa_v0)
                pre_logits_inline1455__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(pre_scaled_inline1447__ssa_v0, pl.tensor.col_expand(pre_scaled_inline1447__ssa_v0, pre_base_inline1470__ssa_v0))
                pre_sig_inline1452__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(pre_logits_inline1455__ssa_v0)), 1.0))
                pre_val_inline1448__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.adds(pre_sig_inline1452__ssa_v0, 9.9999999999999995e-07)
                pre_val_store_inline1529__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32] = pl.tensor.assemble(pre_val_store_inline1529__ssa_v0, pre_val_inline1448__ssa_v0, [t0_inline1476__ssa_v2, 0])
                post_base_inline1472__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(hc_attn_base__ssa_v0, [8], [4]), [1, 8])
                post_scaled_inline1461__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(pl.tensor.row_expand_mul(pl.tensor.slice(mixes_raw_inline1505__ssa_v1, [8, 8], [t0_inline1476__ssa_v2, 4]), inv_col_inline1450__ssa_v0), scale1_inline1530__ssa_v0)
                post_logits_inline1506__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(post_scaled_inline1461__ssa_v0, pl.tensor.col_expand(post_scaled_inline1461__ssa_v0, post_base_inline1472__ssa_v0))
                post_sig_inline1444__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(post_logits_inline1506__ssa_v0)), 1.0))
                post_pad_inline1489__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(post_sig_inline1444__ssa_v0, 2.0)
                if valid_rows_inline1507__ssa_v1 == 8:
                    post_t_inline1277__ssa_v1: pl.Tensor[[T_DYN, 4], pl.FP32] = pl.tensor.assemble(post_t_inline1277__ssa_v0, pl.tensor.slice(post_pad_inline1489__ssa_v0, [8, 8], [0, 0], [8, 4]), [t0_inline1476__ssa_v2, 0])
                    post_t_inline1277__phi_v2, post_tail_store_inline1544__phi_v2 = pl.yield_(post_t_inline1277__ssa_v1, post_tail_store_inline1544__ssa_v0)
                else:
                    post_tail_store_inline1544__ssa_v1: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.assemble(post_tail_store_inline1544__ssa_v0, post_pad_inline1489__ssa_v0, [0, 0])
                    post_tile_inline1483__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v1, 4])] = pl.tile.load(post_tail_store_inline1544__ssa_v1, [0, 0], [8, 8], [valid_rows_inline1507__ssa_v1, 4], target_memory=pl.Mem.Vec)
                    pl.tile.store(post_tile_inline1483__ssa_v0, [t0_inline1476__ssa_v2, 0], post_t_inline1277__ssa_v0)
                    post_t_inline1277__phi_v2, post_tail_store_inline1544__phi_v2 = pl.yield_(post_t_inline1277__ssa_v0, post_tail_store_inline1544__ssa_v1)
            comb_tail_store_inline1523__ssa_v0: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.create([8, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for ob_inline1522__ssa_v0 in pl.spmd(token_tiles_inline1492__ssa_v0, name_hint="comb_sinkhorn_spmd", allow_early_resolve=True):
                t0_inline1476__ssa_v3: pl.Scalar[pl.INDEX] = ob_inline1522__ssa_v0 * 8
                valid_rows_inline1507__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v3)
                inv_col_t_inline1560__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 1])] = pl.tile.load(inv_rms_inline1463__ssa_v1, [t0_inline1476__ssa_v3, 0], [8, 1], [valid_rows_inline1507__ssa_v2, 1], target_memory=pl.Mem.Vec)
                comb_off_inline1572__ssa_v0: pl.Scalar[pl.INDEX] = 8
                mix_g0_inline1525__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v3, comb_off_inline1572__ssa_v0 + 0], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
                mix_g1_inline1528__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v3, comb_off_inline1572__ssa_v0 + 4], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
                mix_g2_inline1531__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v3, comb_off_inline1572__ssa_v0 + 8], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
                mix_g3_inline1552__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v3, comb_off_inline1572__ssa_v0 + 12], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
                cb0_inline1458__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467__ssa_v0, [0, comb_off_inline1572__ssa_v0 + 0], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb1_inline1485__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467__ssa_v0, [0, comb_off_inline1572__ssa_v0 + 4], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb2_inline1532__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467__ssa_v0, [0, comb_off_inline1572__ssa_v0 + 8], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                cb3_inline1533__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467__ssa_v0, [0, comb_off_inline1572__ssa_v0 + 12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
                row0_inline1534__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g0_inline1525__ssa_v0, inv_col_t_inline1560__ssa_v0), scale2_inline1480__ssa_v0), pl.tile.col_expand(mix_g0_inline1525__ssa_v0, cb0_inline1458__ssa_v0))
                row1_inline1575__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g1_inline1528__ssa_v0, inv_col_t_inline1560__ssa_v0), scale2_inline1480__ssa_v0), pl.tile.col_expand(mix_g1_inline1528__ssa_v0, cb1_inline1485__ssa_v0))
                row2_inline1468__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g2_inline1531__ssa_v0, inv_col_t_inline1560__ssa_v0), scale2_inline1480__ssa_v0), pl.tile.col_expand(mix_g2_inline1531__ssa_v0, cb2_inline1532__ssa_v0))
                row3_inline1484__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.add(pl.tile.muls(pl.tile.row_expand_mul(mix_g3_inline1552__ssa_v0, inv_col_t_inline1560__ssa_v0), scale2_inline1480__ssa_v0), pl.tile.col_expand(mix_g3_inline1552__ssa_v0, cb3_inline1533__ssa_v0))
                row0_p_inline1478__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row0_inline1534__ssa_v0, pad_value=pl.PadValue.min)
                row1_p_inline1495__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row1_inline1575__ssa_v0, pad_value=pl.PadValue.min)
                row2_p_inline1535__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row2_inline1468__ssa_v0, pad_value=pl.PadValue.min)
                row3_p_inline1445__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row3_inline1484__ssa_v0, pad_value=pl.PadValue.min)
                row_max_tmp_inline1487__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                row_sum_tmp_inline1498__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                row0_max_inline1538__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row0_p_inline1478__ssa_v0, row_max_tmp_inline1487__ssa_v0)
                row1_max_inline1473__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row1_p_inline1495__ssa_v0, row_max_tmp_inline1487__ssa_v0)
                row2_max_inline1539__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row2_p_inline1535__ssa_v0, row_max_tmp_inline1487__ssa_v0)
                row3_max_inline1540__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_max(row3_p_inline1445__ssa_v0, row_max_tmp_inline1487__ssa_v0)
                row0_exp_inline1527__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row0_p_inline1478__ssa_v0, row0_max_inline1538__ssa_v0))
                row1_exp_inline1543__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row1_p_inline1495__ssa_v0, row1_max_inline1473__ssa_v0))
                row2_exp_inline1545__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row2_p_inline1535__ssa_v0, row2_max_inline1539__ssa_v0))
                row3_exp_inline1547__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(pl.tile.row_expand_sub(row3_p_inline1445__ssa_v0, row3_max_inline1540__ssa_v0))
                row0_sum_inline1512__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row0_exp_inline1527__ssa_v0, row_sum_tmp_inline1498__ssa_v0)
                row1_sum_inline1482__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row1_exp_inline1543__ssa_v0, row_sum_tmp_inline1498__ssa_v0)
                row2_sum_inline1546__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row2_exp_inline1545__ssa_v0, row_sum_tmp_inline1498__ssa_v0)
                row3_sum_inline1453__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(row3_exp_inline1547__ssa_v0, row_sum_tmp_inline1498__ssa_v0)
                row0_soft_inline1548__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row0_exp_inline1527__ssa_v0, row0_sum_inline1512__ssa_v0), 9.9999999999999995e-07)
                row1_soft_inline1515__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row1_exp_inline1543__ssa_v0, row1_sum_inline1482__ssa_v0), 9.9999999999999995e-07)
                row2_soft_inline1477__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row2_exp_inline1545__ssa_v0, row2_sum_inline1546__ssa_v0), 9.9999999999999995e-07)
                row3_soft_inline1549__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(pl.tile.row_expand_div(row3_exp_inline1547__ssa_v0, row3_sum_inline1453__ssa_v0), 9.9999999999999995e-07)
                row0_valid_inline1508__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row0_soft_inline1548__ssa_v0, 8, 4)
                row1_valid_inline1551__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row1_soft_inline1515__ssa_v0, 8, 4)
                row2_valid_inline1517__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row2_soft_inline1477__ssa_v0, 8, 4)
                row3_valid_inline1553__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row3_soft_inline1549__ssa_v0, 8, 4)
                row0_eff_inline1555__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row0_valid_inline1508__ssa_v0, pad_value=pl.PadValue.zero)
                row1_eff_inline1557__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row1_valid_inline1551__ssa_v0, pad_value=pl.PadValue.zero)
                row2_eff_inline1479__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row2_valid_inline1517__ssa_v0, pad_value=pl.PadValue.zero)
                row3_eff_inline1559__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row3_valid_inline1553__ssa_v0, pad_value=pl.PadValue.zero)
                row_sum_tmp_iter_inline1562__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                col_sum_inline1563__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(pl.tile.add(row0_eff_inline1555__ssa_v0, row1_eff_inline1557__ssa_v0), pl.tile.add(row2_eff_inline1479__ssa_v0, row3_eff_inline1559__ssa_v0))
                col_sum_v1_inline1564__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_inline1563__ssa_v0, 9.9999999999999995e-07)
                row0_cur_inline1565__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_eff_inline1555__ssa_v0, col_sum_v1_inline1564__ssa_v0)
                row1_cur_inline1449__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_eff_inline1557__ssa_v0, col_sum_v1_inline1564__ssa_v0)
                row2_cur_inline1566__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_eff_inline1479__ssa_v0, col_sum_v1_inline1564__ssa_v0)
                row3_cur_inline1569__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_eff_inline1559__ssa_v0, col_sum_v1_inline1564__ssa_v0)
                for _sk_it_inline1460__idx_v0, (col_sum_v1_inline1564__iter_v1, row0_cur_inline1565__iter_v1, row1_cur_inline1449__iter_v1, row2_cur_inline1566__iter_v1, row3_cur_inline1569__iter_v1) in pl.pipeline(19, stage=2, init_values=(col_sum_v1_inline1564__ssa_v0, row0_cur_inline1565__ssa_v0, row1_cur_inline1449__ssa_v0, row2_cur_inline1566__ssa_v0, row3_cur_inline1569__ssa_v0)):
                    row0_rowsum_inline1571__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row0_cur_inline1565__iter_v1, row_sum_tmp_iter_inline1562__ssa_v0), 9.9999999999999995e-07)
                    row1_rowsum_inline1469__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row1_cur_inline1449__iter_v1, row_sum_tmp_iter_inline1562__ssa_v0), 9.9999999999999995e-07)
                    row2_rowsum_inline1573__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row2_cur_inline1566__iter_v1, row_sum_tmp_iter_inline1562__ssa_v0), 9.9999999999999995e-07)
                    row3_rowsum_inline1574__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.adds(pl.tile.row_sum(row3_cur_inline1569__iter_v1, row_sum_tmp_iter_inline1562__ssa_v0), 9.9999999999999995e-07)
                    row0_norm_inline1576__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row0_cur_inline1565__iter_v1, row0_rowsum_inline1571__ssa_v0)
                    row1_norm_inline1504__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row1_cur_inline1449__iter_v1, row1_rowsum_inline1469__ssa_v0)
                    row2_norm_inline1466__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row2_cur_inline1566__iter_v1, row2_rowsum_inline1573__ssa_v0)
                    row3_norm_inline1577__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row3_cur_inline1569__iter_v1, row3_rowsum_inline1574__ssa_v0)
                    col_sum_v1_inline1564__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(pl.tile.add(row0_norm_inline1576__ssa_v0, row1_norm_inline1504__ssa_v0), pl.tile.add(row2_norm_inline1466__ssa_v0, row3_norm_inline1577__ssa_v0))
                    col_sum_v1_inline1564__ssa_v4: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_v1_inline1564__ssa_v3, 9.9999999999999995e-07)
                    row0_cur_inline1565__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_norm_inline1576__ssa_v0, col_sum_v1_inline1564__ssa_v4)
                    row1_cur_inline1449__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_norm_inline1504__ssa_v0, col_sum_v1_inline1564__ssa_v4)
                    row2_cur_inline1566__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_norm_inline1466__ssa_v0, col_sum_v1_inline1564__ssa_v4)
                    row3_cur_inline1569__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_norm_inline1577__ssa_v0, col_sum_v1_inline1564__ssa_v4)
                    col_sum_v1_inline1564__rv_v2, row0_cur_inline1565__rv_v2, row1_cur_inline1449__rv_v2, row2_cur_inline1566__rv_v2, row3_cur_inline1569__rv_v2 = pl.yield_(col_sum_v1_inline1564__ssa_v4, row0_cur_inline1565__ssa_v3, row1_cur_inline1449__ssa_v3, row2_cur_inline1566__ssa_v3, row3_cur_inline1569__ssa_v3)
                if valid_rows_inline1507__ssa_v2 == 8:
                    row0_out_inline1541__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row0_cur_inline1565__rv_v2, 8, 4)
                    row1_out_inline1536__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row1_cur_inline1449__rv_v2, 8, 4)
                    row2_out_inline1558__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row2_cur_inline1566__rv_v2, 8, 4)
                    row3_out_inline1474__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row3_cur_inline1569__rv_v2, 8, 4)
                    pl.tile.store(row0_out_inline1541__ssa_v0, [t0_inline1476__ssa_v3, 0], comb_t_inline1267__ssa_v0)
                    pl.tile.store(row1_out_inline1536__ssa_v0, [t0_inline1476__ssa_v3, 4], comb_t_inline1267__ssa_v0)
                    pl.tile.store(row2_out_inline1558__ssa_v0, [t0_inline1476__ssa_v3, 8], comb_t_inline1267__ssa_v0)
                    pl.tile.store(row3_out_inline1474__ssa_v0, [t0_inline1476__ssa_v3, 12], comb_t_inline1267__ssa_v0)
                else:
                    pl.tile.store(row0_cur_inline1565__rv_v2, [0, 0], comb_tail_store_inline1523__ssa_v0)
                    pl.tile.store(row1_cur_inline1449__rv_v2, [0, 8], comb_tail_store_inline1523__ssa_v0)
                    pl.tile.store(row2_cur_inline1566__rv_v2, [0, 16], comb_tail_store_inline1523__ssa_v0)
                    pl.tile.store(row3_cur_inline1569__rv_v2, [0, 24], comb_tail_store_inline1523__ssa_v0)
                    row0_tail_inline1556__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(comb_tail_store_inline1523__ssa_v0, [0, 0], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
                    row1_tail_inline1578__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(comb_tail_store_inline1523__ssa_v0, [0, 8], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
                    row2_tail_inline1500__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(comb_tail_store_inline1523__ssa_v0, [0, 16], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
                    row3_tail_inline1451__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(comb_tail_store_inline1523__ssa_v0, [0, 24], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
                    pl.tile.store(row0_tail_inline1556__ssa_v0, [t0_inline1476__ssa_v3, 0], comb_t_inline1267__ssa_v0)
                    pl.tile.store(row1_tail_inline1578__ssa_v0, [t0_inline1476__ssa_v3, 4], comb_t_inline1267__ssa_v0)
                    pl.tile.store(row2_tail_inline1500__ssa_v0, [t0_inline1476__ssa_v3, 8], comb_t_inline1267__ssa_v0)
                    pl.tile.store(row3_tail_inline1451__ssa_v0, [t0_inline1476__ssa_v3, 12], comb_t_inline1267__ssa_v0)
            x_mixed_tail_store_inline1462__ssa_v0: pl.Tensor[[8, 4096], pl.BF16] = pl.tensor.create([8, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            for blk_inline1491__ssa_v0 in pl.spmd(token_tiles_inline1492__ssa_v0 * 1, name_hint="mix_x_spmd", allow_early_resolve=True):
                t0_inline1476__ssa_v4: pl.Scalar[pl.INDEX] = blk_inline1491__ssa_v0 // 1 * 8
                d_base_inline1454__ssa_v0: pl.Scalar[pl.INDEX] = blk_inline1491__ssa_v0 % 1 * 4096
                valid_rows_inline1507__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v4)
                pre_tile_t_inline1443__ssa_v0: pl.Tensor[[8, 8], pl.FP32, pl.TensorView(stride=[1, 8], layout=pl.TensorLayout.DN)] = pl.tensor.transpose(pl.tensor.slice(pre_val_store_inline1529__ssa_v1, [8, 8], [t0_inline1476__ssa_v4, 0]), 0, 1)
                pre0_inline1526__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t_inline1443__ssa_v0, [1, 8], [0, 0]), [8, 1])
                pre1_inline1442__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t_inline1443__ssa_v0, [1, 8], [1, 0]), [8, 1])
                pre2_inline1441__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t_inline1443__ssa_v0, [1, 8], [2, 0]), [8, 1])
                pre3_inline1440__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(pre_tile_t_inline1443__ssa_v0, [1, 8], [3, 0]), [8, 1])
                for db_inline1465__idx_v0, (x_mixed_inline1253__iter_v1, x_mixed_tail_store_inline1462__iter_v1) in pl.pipeline(16, stage=2, init_values=(x_mixed_inline1253__ssa_v0, x_mixed_tail_store_inline1462__ssa_v0)):
                    d0_inline1496__ssa_v0: pl.Scalar[pl.INDEX] = d_base_inline1454__ssa_v0 + db_inline1465__idx_v0 * 256
                    x0_inline1439__ssa_v0: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows_inline1507__ssa_v3, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497__ssa_v0, [8, 256], [t0_inline1476__ssa_v4, 0 + d0_inline1496__ssa_v0], [valid_rows_inline1507__ssa_v3, 256])
                    x1_inline1446__ssa_v0: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows_inline1507__ssa_v3, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497__ssa_v0, [8, 256], [t0_inline1476__ssa_v4, 4096 + d0_inline1496__ssa_v0], [valid_rows_inline1507__ssa_v3, 256])
                    x2_inline1542__ssa_v0: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows_inline1507__ssa_v3, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497__ssa_v0, [8, 256], [t0_inline1476__ssa_v4, 8192 + d0_inline1496__ssa_v0], [valid_rows_inline1507__ssa_v3, 256])
                    x3_inline1438__ssa_v0: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(valid_shape=[valid_rows_inline1507__ssa_v3, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline1497__ssa_v0, [8, 256], [t0_inline1476__ssa_v4, 12288 + d0_inline1496__ssa_v0], [valid_rows_inline1507__ssa_v3, 256])
                    y0_inline1437__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x0_inline1439__ssa_v0, pre0_inline1526__ssa_v0)
                    y1_inline1550__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x1_inline1446__ssa_v0, pre1_inline1442__ssa_v0)
                    y2_inline1436__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x2_inline1542__ssa_v0, pre2_inline1441__ssa_v0)
                    y3_inline1435__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(x3_inline1438__ssa_v0, pre3_inline1440__ssa_v0)
                    y_tile_inline1434__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.add(pl.tensor.add(y0_inline1437__ssa_v0, y1_inline1550__ssa_v0), pl.tensor.add(y2_inline1436__ssa_v0, y3_inline1435__ssa_v0))
                    y_bf16_inline1433__ssa_v0: pl.Tensor[[8, 256], pl.BF16] = pl.tensor.cast(y_tile_inline1434__ssa_v0, target_type=pl.BF16, mode='rint')
                    if valid_rows_inline1507__ssa_v3 == 8:
                        x_mixed_inline1253__ssa_v3: pl.Tensor[[T_DYN, 4096], pl.BF16] = pl.tensor.assemble(x_mixed_inline1253__iter_v1, y_bf16_inline1433__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0])
                        x_mixed_inline1253__phi_v4, x_mixed_tail_store_inline1462__phi_v4 = pl.yield_(x_mixed_inline1253__ssa_v3, x_mixed_tail_store_inline1462__iter_v1)
                    else:
                        x_mixed_tail_store_inline1462__ssa_v3: pl.Tensor[[8, 4096], pl.BF16] = pl.tensor.assemble(x_mixed_tail_store_inline1462__iter_v1, y_bf16_inline1433__ssa_v0, [0, d0_inline1496__ssa_v0])
                        y_out_inline1521__ssa_v0: pl.Tile[[8, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_mixed_tail_store_inline1462__ssa_v3, [0, d0_inline1496__ssa_v0], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
                        pl.tile.store(y_out_inline1521__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0], x_mixed_inline1253__iter_v1)
                        x_mixed_inline1253__phi_v4, x_mixed_tail_store_inline1462__phi_v4 = pl.yield_(x_mixed_inline1253__iter_v1, x_mixed_tail_store_inline1462__ssa_v3)
                    x_mixed_inline1253__rv_v2, x_mixed_tail_store_inline1462__rv_v2 = pl.yield_(x_mixed_inline1253__phi_v4, x_mixed_tail_store_inline1462__phi_v4)
        idx_cos_il_inline1282__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        idx_sin_signed_inline1307__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_cos_il_full_inline1249__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        cmp_sin_signed_full_inline1263__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_interleave") as rope_tid_inline1259__ssa_v0:
            il_ones_inline1242__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.full([4, 64], dtype=pl.FP32, value=1.0)
            il_col_inline1252__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.col_expand_mul(il_ones_inline1242__ssa_v0, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
            il_dup_f_inline1250__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(il_col_inline1252__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
            il_dup_idx_inline1272__ssa_v0: pl.Tensor[[4, 64], pl.INT32] = pl.tensor.cast(il_dup_f_inline1250__ssa_v0, target_type=pl.INT32, mode='round')
            il_lane_inline1300__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.sub(il_col_inline1252__ssa_v0, pl.tensor.muls(il_dup_f_inline1250__ssa_v0, 2.0))
            il_sign_inline1298__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(il_lane_inline1300__ssa_v0, 2.0), 1.0)
            for rope_t0_inline1256__idx_v0, (idx_cos_il_inline1282__iter_v1, idx_sin_signed_inline1307__iter_v1) in pl.range(0, t_dim_inline1251__ssa_v0, 4, init_values=(idx_cos_il_inline1282__ssa_v0, idx_sin_signed_inline1307__ssa_v0)):
                idx_cos_il_inline1282__ssa_v3: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32] = pl.tensor.assemble(idx_cos_il_inline1282__iter_v1, pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(freqs_cos_local__ssa_v0, [4, 32], [rope_t0_inline1256__idx_v0, 0]), target_type=pl.FP32, mode='round'), il_dup_idx_inline1272__ssa_v0, dim=-1), [rope_t0_inline1256__idx_v0, 0])
                idx_sin_signed_inline1307__ssa_v3: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32] = pl.tensor.assemble(idx_sin_signed_inline1307__iter_v1, pl.tensor.mul(pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(freqs_sin_local__ssa_v0, [4, 32], [rope_t0_inline1256__idx_v0, 0]), target_type=pl.FP32, mode='round'), il_dup_idx_inline1272__ssa_v0, dim=-1), il_sign_inline1298__ssa_v0), [rope_t0_inline1256__idx_v0, 0])
                idx_cos_il_inline1282__rv_v2, idx_sin_signed_inline1307__rv_v2 = pl.yield_(idx_cos_il_inline1282__ssa_v3, idx_sin_signed_inline1307__ssa_v3)
            for cmp_t0_inline1248__idx_v0, (cmp_cos_il_full_inline1249__iter_v1, cmp_sin_signed_full_inline1263__iter_v1) in pl.range(0, kv_dim_inline1261__ssa_v0, 4, init_values=(cmp_cos_il_full_inline1249__ssa_v0, cmp_sin_signed_full_inline1263__ssa_v0)):
                cmp_cos_il_full_inline1249__ssa_v3: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32] = pl.tensor.assemble(cmp_cos_il_full_inline1249__iter_v1, pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(cmp_freqs_cos__ssa_v0, [4, 32], [cmp_t0_inline1248__idx_v0, 0]), target_type=pl.FP32, mode='round'), il_dup_idx_inline1272__ssa_v0, dim=-1), [cmp_t0_inline1248__idx_v0, 0])
                cmp_sin_signed_full_inline1263__ssa_v3: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32] = pl.tensor.assemble(cmp_sin_signed_full_inline1263__iter_v1, pl.tensor.mul(pl.tensor.gather(pl.tensor.cast(pl.tensor.slice(cmp_freqs_sin__ssa_v0, [4, 32], [cmp_t0_inline1248__idx_v0, 0]), target_type=pl.FP32, mode='round'), il_dup_idx_inline1272__ssa_v0, dim=-1), il_sign_inline1298__ssa_v0), [cmp_t0_inline1248__idx_v0, 0])
                cmp_cos_il_full_inline1249__rv_v2, cmp_sin_signed_full_inline1263__rv_v2 = pl.yield_(cmp_cos_il_full_inline1249__ssa_v3, cmp_sin_signed_full_inline1263__ssa_v3)
        x_normed_t_inline1243__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16] = pl.tensor.create([t_dim_inline1251__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            t_dim_inline1611__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_mixed_inline1253__rv_v2, 0)
            token_tiles_inline1607__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1611__ssa_v0 + 8 - 1) // 8
            with pl.spmd(token_tiles_inline1607__ssa_v0, name_hint="rms_norm_spmd", allow_early_resolve=True) as rms_tid_inline1605__ssa_v0:
                tg_idx_inline1603__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                tg_inline1598__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1603__ssa_v0 * 8
                valid_rows_inline1600__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1611__ssa_v0 - tg_inline1598__ssa_v0)
                if valid_rows_inline1600__ssa_v0 == 8:
                    # Run the aligned token tile through the existing Tensor-level dataflow.
                    tg_inline80_inline1597__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 8
                    x_sq_sum_inline83_inline1612__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                    for rms_db_inline82_inline1602__idx_v0, (x_sq_sum_inline83_inline1612__iter_v1,) in pl.pipeline(32, stage=2, init_values=(x_sq_sum_inline83_inline1612__ssa_v0,)):
                        rms_d0_inline87_inline1595__ssa_v0: pl.Scalar[pl.INDEX] = rms_db_inline82_inline1602__idx_v0 * 128
                        rms_x_input_inline84_inline1593__ssa_v0: pl.Tensor[[8, 128], pl.BF16] = pl.tensor.slice(x_mixed_inline1253__rv_v2, [8, 128], [tg_inline80_inline1597__ssa_v0, rms_d0_inline87_inline1595__ssa_v0])
                        rms_x_chunk_inline88_inline1608__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(rms_x_input_inline84_inline1593__ssa_v0, target_type=pl.FP32, mode='round')
                        rms_x_sq_inline89_inline1619__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.mul(rms_x_chunk_inline88_inline1608__ssa_v0, rms_x_chunk_inline88_inline1608__ssa_v0)
                        rms_x_row_sum_inline81_inline1609__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(rms_x_sq_inline89_inline1619__ssa_v0), [1, 8])
                        x_sq_sum_inline83_inline1612__ssa_v3: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(x_sq_sum_inline83_inline1612__iter_v1, rms_x_row_sum_inline81_inline1609__ssa_v0)
                        x_sq_sum_inline83_inline1612__rv_v2: pl.Tensor[[1, 8], pl.FP32] = pl.yield_(x_sq_sum_inline83_inline1612__ssa_v3)
                    x_inv_rms_inline85_inline1610__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(x_sq_sum_inline83_inline1612__rv_v2, 0.000244140625), 9.9999999999999995e-07), high_precision=True)
                    x_inv_rms_t_inline93_inline1594__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(x_inv_rms_inline85_inline1610__ssa_v0, [8, 1])
                    for apply_db_inline86_inline1614__idx_v0, (x_normed_t_inline1243__iter_v1,) in pl.pipeline(32, stage=2, init_values=(x_normed_t_inline1243__ssa_v0,)):
                        apply_d0_inline91_inline1615__ssa_v0: pl.Scalar[pl.INDEX] = apply_db_inline86_inline1614__idx_v0 * 128
                        apply_x_input_inline94_inline1613__ssa_v0: pl.Tensor[[8, 128], pl.BF16] = pl.tensor.slice(x_mixed_inline1253__rv_v2, [8, 128], [tg_inline80_inline1597__ssa_v0, apply_d0_inline91_inline1615__ssa_v0])
                        apply_x_chunk_inline78_inline1620__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(apply_x_input_inline94_inline1613__ssa_v0, target_type=pl.FP32, mode='round')
                        norm_w_input_inline77_inline1617__ssa_v0: pl.Tensor[[128], pl.BF16] = pl.tensor.slice(attn_norm_w__ssa_v0, [128], [apply_d0_inline91_inline1615__ssa_v0])
                        norm_w_chunk_inline92_inline1604__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.reshape(norm_w_input_inline77_inline1617__ssa_v0, [1, 128]), target_type=pl.FP32, mode='round')
                        x_scaled_inline90_inline1622__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.row_expand_mul(apply_x_chunk_inline78_inline1620__ssa_v0, x_inv_rms_t_inline93_inline1594__ssa_v0)
                        x_normed_chunk_inline79_inline1592__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.col_expand_mul(x_scaled_inline90_inline1622__ssa_v0, norm_w_chunk_inline92_inline1604__ssa_v0)
                        x_normed_t_inline1243__ssa_v3: pl.Tensor[[T_DYN, 4096], pl.BF16] = pl.tensor.assemble(x_normed_t_inline1243__iter_v1, pl.tensor.cast(x_normed_chunk_inline79_inline1592__ssa_v0, target_type=pl.BF16, mode='rint'), [tg_inline80_inline1597__ssa_v0, apply_d0_inline91_inline1615__ssa_v0])
                        x_normed_t_inline1243__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16] = pl.yield_(x_normed_t_inline1243__ssa_v3)
                    x_normed_t_inline1243__phi_v4: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16] = pl.yield_(x_normed_t_inline1243__rv_v2)
                else:
                    # Run the ragged last token tile through explicit `valid_shape` load/store.
                    # 
                    #         Step for step the same RMSNorm as `_rms_norm_full_tile`. The two live in
                    #         separate scopes rather than in one `if`/`else` body because this path binds
                    #         the shared names (`x_sq_sum`, `x_inv_rms`, …) to Vec-space Tiles while the
                    #         aligned path binds them to Tensors, and a name cannot be rebound to a
                    #         different type inside one kernel.
                    #         
                    tg_inline103_inline1596__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx() * 8
                    valid_rows_inline113_inline1601__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, pl.tensor.dim(x_mixed_inline1253__rv_v2, 0) - tg_inline103_inline1596__ssa_v0)
                    row_reduce_tmp_inline101_inline1623__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    x_sq_sum_inline104_inline1624__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                    for rms_db_inline105_inline1591__idx_v0, (x_sq_sum_inline104_inline1624__iter_v1,) in pl.pipeline(32, stage=2, init_values=(x_sq_sum_inline104_inline1624__ssa_v0,)):
                        rms_d0_inline106_inline1589__ssa_v0: pl.Scalar[pl.INDEX] = rms_db_inline105_inline1591__idx_v0 * 128
                        rms_x_input_inline107_inline1587__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline103_inline1596__ssa_v0, rms_d0_inline106_inline1589__ssa_v0], [8, 128], [valid_rows_inline113_inline1601__ssa_v0, 128], target_memory=pl.Mem.Vec)
                        rms_x_chunk_inline108_inline1586__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(rms_x_input_inline107_inline1587__ssa_v0, target_type=pl.FP32, mode='round')
                        rms_x_sq_inline111_inline1585__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.mul(rms_x_chunk_inline108_inline1586__ssa_v0, rms_x_chunk_inline108_inline1586__ssa_v0)
                        rms_x_row_sum_inline114_inline1584__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline113_inline1601__ssa_v0])] = pl.tile.reshape(pl.tile.row_sum(rms_x_sq_inline111_inline1585__ssa_v0, row_reduce_tmp_inline101_inline1623__ssa_v0), [1, 8])
                        x_sq_sum_inline104_inline1624__ssa_v3: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.add(x_sq_sum_inline104_inline1624__iter_v1, rms_x_row_sum_inline114_inline1584__ssa_v0)
                        x_sq_sum_inline104_inline1624__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.yield_(x_sq_sum_inline104_inline1624__ssa_v3)
                    x_inv_rms_inline115_inline1618__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(x_sq_sum_inline104_inline1624__rv_v2, 0.000244140625), 9.9999999999999995e-07)))
                    x_inv_rms_t_inline116_inline1621__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(x_inv_rms_inline115_inline1618__ssa_v0, [8, 1])
                    for apply_db_inline110_inline1583__idx_v0 in pl.pipeline(32, stage=2):
                        apply_d0_inline100_inline1582__ssa_v0: pl.Scalar[pl.INDEX] = apply_db_inline110_inline1583__idx_v0 * 128
                        apply_x_input_inline99_inline1616__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline103_inline1596__ssa_v0, apply_d0_inline100_inline1582__ssa_v0], [8, 128], [valid_rows_inline113_inline1601__ssa_v0, 128], target_memory=pl.Mem.Vec)
                        apply_x_chunk_inline98_inline1599__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(apply_x_input_inline99_inline1616__ssa_v0, target_type=pl.FP32, mode='round')
                        norm_w_input_inline96_inline1588__ssa_v0: pl.Tile[[128], pl.BF16, pl.Mem.Vec] = pl.tile.load(attn_norm_w__ssa_v0, [apply_d0_inline100_inline1582__ssa_v0], [128], [128], target_memory=pl.Mem.Vec)
                        norm_w_chunk_inline102_inline1581__ssa_v0: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.reshape(norm_w_input_inline96_inline1588__ssa_v0, [1, 128]), target_type=pl.FP32, mode='round')
                        x_scaled_inline109_inline1606__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.row_expand_mul(apply_x_chunk_inline98_inline1599__ssa_v0, x_inv_rms_t_inline116_inline1621__ssa_v0)
                        x_normed_chunk_inline112_inline1580__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.col_expand_mul(x_scaled_inline109_inline1606__ssa_v0, norm_w_chunk_inline102_inline1581__ssa_v0)
                        x_normed_bf16_inline95_inline1590__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(x_normed_chunk_inline112_inline1580__ssa_v0, target_type=pl.BF16, mode='rint')
                        x_normed_valid_inline97_inline1579__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.set_validshape(x_normed_bf16_inline95_inline1590__ssa_v0, valid_rows_inline113_inline1601__ssa_v0, 128)
                        pl.tile.store(x_normed_valid_inline97_inline1579__ssa_v0, [tg_inline103_inline1596__ssa_v0, apply_d0_inline100_inline1582__ssa_v0], x_normed_t_inline1243__ssa_v0)
                    x_normed_t_inline1243__phi_v4: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16] = pl.yield_(x_normed_t_inline1243__ssa_v0)
        kv_wb_blocks_inline1274__ssa_v0: pl.Scalar[pl.INDEX] = kv_dim_inline1261__ssa_v0 // 8
        x_normed_full_inline1240__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            # Gather rank-major rows and retire the complete two-phase signal epoch.
            local_rows_inline1634__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243__phi_v4, 0)
            local_t_inline1640__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(local_rows_inline1634__ssa_v0, pl.INT32)
            target_row_inline1642__ssa_v0: pl.Scalar[pl.INT32] = tp_rank__ssa_v0 * local_t_inline1640__ssa_v0
            full_local_inline1644__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(local_t_inline1640__ssa_v0, pl.INDEX) // 8 * 8
            with pl.spmd(16, name_hint="cp_token_allgather_push_spmd", allow_early_resolve=True) as _push_tid_inline1646__ssa_v0:
                worker_inline1647__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for peer_tp_inline1635__idx_v0 in pl.range(2):
                    for band_row_inline1648__idx_v0 in pl.range(worker_inline1647__ssa_v0 * 8, full_local_inline1644__ssa_v0, 128):
                        pld.tensor.put(gather_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1635__idx_v0, x_normed_t_inline1243__phi_v4, [pl.cast(target_row_inline1642__ssa_v0, pl.INDEX) + band_row_inline1648__idx_v0, 0], [band_row_inline1648__idx_v0, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                    for tail_row_inline1651__idx_v0 in pl.range(full_local_inline1644__ssa_v0 + worker_inline1647__ssa_v0, local_t_inline1640__ssa_v0, 16):
                        pld.tensor.put(gather_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1635__idx_v0, x_normed_t_inline1243__phi_v4, [pl.cast(target_row_inline1642__ssa_v0, pl.INDEX) + tail_row_inline1651__idx_v0, 0], [tail_row_inline1651__idx_v0, 0], [1, 4096], atomic=pl.AtomicType.None_, chunk_rows=1, chunk_cols=4096)
                for peer_tp_inline1636__idx_v0 in pl.range(2):
                    if peer_tp_inline1636__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(gather_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1636__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_payload_wait", deps=[_push_tid_inline1646__ssa_v0]) as _payload_wait_tid_inline1643__ssa_v0:
                for source_tp_inline1637__idx_v0 in pl.range(2):
                    if source_tp_inline1637__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.wait(gather_signal__ssa_v0, [source_tp_inline1637__idx_v0, 0], pl.cast(16, pl.INT32), cmp=1)
            group_rows_inline1639__ssa_v0: pl.Scalar[pl.INDEX] = 2 * local_rows_inline1634__ssa_v0
            full_rows_inline1641__ssa_v0: pl.Scalar[pl.INDEX] = group_rows_inline1639__ssa_v0 // 16 * 16
            with pl.spmd(16, name_hint="cp_token_allgather_readback_spmd", deps=[_push_tid_inline1646__ssa_v0, _payload_wait_tid_inline1643__ssa_v0]) as _readback_tid_inline1633__ssa_v0:
                worker_v1_inline1638__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for tile_row_inline1632__idx_v0, (x_normed_full_inline1240__iter_v1,) in pl.range(worker_v1_inline1638__ssa_v0 * 16, full_rows_inline1641__ssa_v0, 256, init_values=(x_normed_full_inline1240__ssa_v0,)):
                    window_tile_inline1650__ssa_v0: pld.DistributedTensor[[16, 4096], pl.BF16] = pl.tensor.slice(gather_window__ssa_v0, [16, 4096], [tile_row_inline1632__idx_v0, 0])
                    x_normed_full_inline1240__ssa_v3: pl.Tensor[[KV_T_DYN, 4096], pl.BF16] = pl.tensor.assemble(x_normed_full_inline1240__iter_v1, window_tile_inline1650__ssa_v0, [tile_row_inline1632__idx_v0, 0])
                    x_normed_full_inline1240__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16] = pl.yield_(x_normed_full_inline1240__ssa_v3)
                for tail_row_inline1629__idx_v0, (x_normed_full_inline1240__iter_v4,) in pl.range(full_rows_inline1641__ssa_v0 + worker_v1_inline1638__ssa_v0, group_rows_inline1639__ssa_v0, 16, init_values=(x_normed_full_inline1240__rv_v2,)):
                    window_row_inline1631__ssa_v0: pld.DistributedTensor[[1, 4096], pl.BF16] = pl.tensor.slice(gather_window__ssa_v0, [1, 4096], [tail_row_inline1629__idx_v0, 0])
                    x_normed_full_inline1240__ssa_v6: pl.Tensor[[KV_T_DYN, 4096], pl.BF16] = pl.tensor.assemble(x_normed_full_inline1240__iter_v4, window_row_inline1631__ssa_v0, [tail_row_inline1629__idx_v0, 0])
                    x_normed_full_inline1240__rv_v5: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16] = pl.yield_(x_normed_full_inline1240__ssa_v6)
                for peer_tp_inline1628__idx_v0 in pl.range(2):
                    if peer_tp_inline1628__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(gather_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1628__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_readback_wait", deps=[_readback_tid_inline1633__ssa_v0]) as _readback_wait_tid_inline1630__ssa_v0:
                for source_tp_inline1627__idx_v0 in pl.range(2):
                    if source_tp_inline1627__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.wait(gather_signal__ssa_v0, [source_tp_inline1627__idx_v0, 0], pl.cast(32, pl.INT32), cmp=1)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_token_allgather_retire", deps=[_readback_tid_inline1633__ssa_v0, _readback_wait_tid_inline1630__ssa_v0]):
                completion_anchor_inline1626__ssa_v0: pl.Scalar[pl.BF16] = pl.tensor.read(x_normed_full_inline1240__rv_v5, [0, 0])
                reset_value_inline1649__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(-32, pl.INT32)
                self_rank_inline1625__ssa_v0: pl.Scalar[pl.INT32] = group_base__ssa_v0 + tp_rank__ssa_v0
                for source_tp_inline1645__idx_v0 in pl.range(2):
                    if source_tp_inline1645__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(gather_signal__ssa_v0, self_rank_inline1625__ssa_v0, [source_tp_inline1645__idx_v0, 0], reset_value_inline1649__ssa_v0, op=0)
                pl.tensor.write(x_normed_full_inline1240__rv_v5, [0, 0], completion_anchor_inline1626__ssa_v0)
            _gathered_normed_inline1281__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16] = x_normed_full_inline1240__rv_v5
            gather_signal__ssa_v1: pld.DistributedTensor[[2, 1], pl.INT32] = gather_signal__ssa_v0
        kv_full_inline1265__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.BF16] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        qr_inline1255__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1024], pl.INT8] = pl.tensor.create([t_dim_inline1251__ssa_v0, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
        qr_scale_inline1310__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.FP32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        position_ids_t1_inline1288__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.INT32] = pl.tensor.reshape(position_ids_local__ssa_v0, [t_dim_inline1251__ssa_v0, 1])
        attention_local_flat_inline1292__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.create([2048, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        attn_out_inline1284__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16] = pl.tensor.create([t_dim_inline1251__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        with pl.scope():
            late_dep_inline1297__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[rope_tid_inline1259__ssa_v0])
            kv_cos_il_inline1258__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_sin_signed_inline1301__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            kv_swap_idx_inline1305__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.INT32] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            # Build the head-invariant interleaved cos / sign-folded sin / swap-index rope rows.
            t_dim_inline1682__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_cos__ssa_v0, 0)
            rope_cos_view_inline1679__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.BF16] = pl.tensor.reshape(freqs_cos__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
            rope_sin_view_inline1674__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.BF16] = pl.tensor.reshape(freqs_sin__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
            rope_cos_il_view_inline1670__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32] = pl.tensor.reshape(kv_cos_il_inline1258__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
            rope_sin_signed_view_inline1668__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32] = pl.tensor.reshape(kv_sin_signed_inline1301__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
            rope_swap_idx_view_inline1694__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32] = pl.tensor.reshape(kv_swap_idx_inline1305__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
            token_tiles_inline1673__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1682__ssa_v0 + 8 - 1) // 8
            for qrp_idx_inline1681__ssa_v0 in pl.spmd(token_tiles_inline1673__ssa_v0, name_hint="q_rope_prepare_spmd", allow_early_resolve=True):
                qrp_t0_inline1672__ssa_v0: pl.Scalar[pl.INDEX] = qrp_idx_inline1681__ssa_v0 * 8
                qrp_valid_rows_inline1667__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1682__ssa_v0 - qrp_t0_inline1672__ssa_v0)
                qrp_ones_inline1686__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
                qrp_idx_i32_inline1666__ssa_v0: pl.Tensor[[1, 64], pl.INT32] = pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                qrp_idx_fp32_inline1661__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(qrp_idx_i32_inline1666__ssa_v0, target_type=pl.FP32, mode='round')
                qrp_col_inline1675__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(qrp_ones_inline1686__ssa_v0, qrp_idx_fp32_inline1661__ssa_v0)
                qrp_half_inline1683__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_col_inline1675__ssa_v0, 0.5)
                qrp_dup_i32_inline1685__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_half_inline1683__ssa_v0, target_type=pl.INT32, mode='trunc')
                qrp_dup_f_inline1663__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_dup_i32_inline1685__ssa_v0, target_type=pl.FP32, mode='round')
                qrp_dup_idx_inline1680__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_dup_f_inline1663__ssa_v0, target_type=pl.INT32, mode='round')
                qrp_lane_inline1678__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_col_inline1675__ssa_v0, pl.tensor.muls(qrp_dup_f_inline1663__ssa_v0, 2.0))
                qrp_next_col_inline1689__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(qrp_col_inline1675__ssa_v0, 1.0)
                qrp_lane_offset_inline1692__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_lane_inline1678__ssa_v0, 2.0)
                qrp_swap_f_inline1690__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_next_col_inline1689__ssa_v0, qrp_lane_offset_inline1692__ssa_v0)
                qrp_swap_idx_inline1695__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_swap_f_inline1690__ssa_v0, target_type=pl.INT32, mode='round')
                qrp_sign_inline1687__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(qrp_lane_inline1678__ssa_v0, 2.0), 1.0)
                if qrp_valid_rows_inline1667__ssa_v0 == 8:
                    qrp_cos_rows_full_inline1693__ssa_v0: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_cos_view_inline1679__ssa_v0, [8, 64], [qrp_t0_inline1672__ssa_v0, 0])
                    qrp_sin_rows_full_inline1696__ssa_v0: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_sin_view_inline1674__ssa_v0, [8, 64], [qrp_t0_inline1672__ssa_v0, 0])
                    qrp_cos_full_inline1676__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_cos_rows_full_inline1693__ssa_v0, target_type=pl.FP32, mode='round')
                    qrp_sin_full_inline1669__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_sin_rows_full_inline1696__ssa_v0, target_type=pl.FP32, mode='round')
                    qrp_cos_il_full_inline1697__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_cos_full_inline1676__ssa_v0, qrp_dup_idx_inline1680__ssa_v0, dim=-1)
                    qrp_sin_il_full_inline1671__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_sin_full_inline1669__ssa_v0, qrp_dup_idx_inline1680__ssa_v0, dim=-1)
                    qrp_sin_signed_full_inline1691__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(qrp_sin_il_full_inline1671__ssa_v0, qrp_sign_inline1687__ssa_v0)
                    rope_cos_il_view_inline1670__ssa_v1: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32] = pl.tensor.assemble(rope_cos_il_view_inline1670__ssa_v0, qrp_cos_il_full_inline1697__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0])
                    rope_sin_signed_view_inline1668__ssa_v1: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed_view_inline1668__ssa_v0, qrp_sin_signed_full_inline1691__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0])
                    rope_swap_idx_view_inline1694__ssa_v1: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_view_inline1694__ssa_v0, qrp_swap_idx_inline1695__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0])
                    rope_cos_il_view_inline1670__phi_v2, rope_sin_signed_view_inline1668__phi_v2, rope_swap_idx_view_inline1694__phi_v2 = pl.yield_(rope_cos_il_view_inline1670__ssa_v1, rope_sin_signed_view_inline1668__ssa_v1, rope_swap_idx_view_inline1694__ssa_v1)
                else:
                    qrp_cos_rows_tail_inline1684__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.load(rope_cos_view_inline1679__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1667__ssa_v0, 64], target_memory=pl.Mem.Vec)
                    qrp_sin_rows_tail_inline1660__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.load(rope_sin_view_inline1674__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1667__ssa_v0, 64], target_memory=pl.Mem.Vec)
                    qrp_tail_col_inline1659__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                    qrp_tail_dup_f_inline1657__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(qrp_tail_col_inline1659__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                    qrp_tail_lane_inline1662__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(qrp_tail_col_inline1659__ssa_v0, pl.tile.muls(qrp_tail_dup_f_inline1657__ssa_v0, 2.0))
                    qrp_tail_swap_f_inline1665__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(qrp_tail_col_inline1659__ssa_v0, 1.0), pl.tile.muls(qrp_tail_lane_inline1662__ssa_v0, 2.0))
                    qrp_row_seed_inline1664__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                    qrp_row_grid_inline1656__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), qrp_row_seed_inline1664__ssa_v0)
                    qrp_row_offset_inline1655__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(qrp_row_grid_inline1656__ssa_v0, 0, 1)
                    qrp_dup_idx_tail_inline1658__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(qrp_tail_dup_f_inline1657__ssa_v0, qrp_row_offset_inline1655__ssa_v0), target_type=pl.INT32, mode='round')
                    qrp_gather_tmp_inline1653__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    qrp_cos_il_tail_inline1652__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_cos_rows_tail_inline1684__ssa_v0, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail_inline1658__ssa_v0, qrp_gather_tmp_inline1653__ssa_v0)
                    qrp_sin_il_tail_inline1654__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_sin_rows_tail_inline1660__ssa_v0, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail_inline1658__ssa_v0, qrp_gather_tmp_inline1653__ssa_v0)
                    qrp_tail_sign_inline1677__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(pl.tile.muls(qrp_tail_lane_inline1662__ssa_v0, 2.0), 1.0)
                    qrp_sin_signed_tail_inline1688__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_tail_inline1654__ssa_v0, qrp_tail_sign_inline1677__ssa_v0)
                    pl.tile.store(pl.tile.set_validshape(qrp_cos_il_tail_inline1652__ssa_v0, qrp_valid_rows_inline1667__ssa_v0, 64), [qrp_t0_inline1672__ssa_v0, 0], rope_cos_il_view_inline1670__ssa_v0)
                    pl.tile.store(pl.tile.set_validshape(qrp_sin_signed_tail_inline1688__ssa_v0, qrp_valid_rows_inline1667__ssa_v0, 64), [qrp_t0_inline1672__ssa_v0, 0], rope_sin_signed_view_inline1668__ssa_v0)
                    pl.tile.store(pl.tile.set_validshape(pl.tile.cast(qrp_tail_swap_f_inline1665__ssa_v0, target_type=pl.INT32, mode='round'), qrp_valid_rows_inline1667__ssa_v0, 64), [qrp_t0_inline1672__ssa_v0, 0], rope_swap_idx_view_inline1694__ssa_v0)
                    rope_cos_il_view_inline1670__phi_v2, rope_sin_signed_view_inline1668__phi_v2, rope_swap_idx_view_inline1694__phi_v2 = pl.yield_(rope_cos_il_view_inline1670__ssa_v0, rope_sin_signed_view_inline1668__ssa_v0, rope_swap_idx_view_inline1694__ssa_v0)
            q_cos_il_inline1311__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_sin_signed_inline1295__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            q_swap_idx_inline1313__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.INT32] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            # Build the head-invariant interleaved cos / sign-folded sin / swap-index rope rows.
            t_dim_inline1728__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_cos_local__ssa_v0, 0)
            rope_cos_view_inline1725__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.BF16] = pl.tensor.reshape(freqs_cos_local__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
            rope_sin_view_inline1720__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.BF16] = pl.tensor.reshape(freqs_sin_local__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
            rope_cos_il_view_inline1716__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32] = pl.tensor.reshape(q_cos_il_inline1311__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
            rope_sin_signed_view_inline1714__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32] = pl.tensor.reshape(q_sin_signed_inline1295__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
            rope_swap_idx_view_inline1740__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32] = pl.tensor.reshape(q_swap_idx_inline1313__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
            token_tiles_inline1719__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1728__ssa_v0 + 8 - 1) // 8
            for qrp_idx_inline1727__ssa_v0 in pl.spmd(token_tiles_inline1719__ssa_v0, name_hint="q_rope_prepare_spmd", allow_early_resolve=True):
                qrp_t0_inline1718__ssa_v0: pl.Scalar[pl.INDEX] = qrp_idx_inline1727__ssa_v0 * 8
                qrp_valid_rows_inline1713__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, t_dim_inline1728__ssa_v0 - qrp_t0_inline1718__ssa_v0)
                qrp_ones_inline1732__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
                qrp_idx_i32_inline1712__ssa_v0: pl.Tensor[[1, 64], pl.INT32] = pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                qrp_idx_fp32_inline1707__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(qrp_idx_i32_inline1712__ssa_v0, target_type=pl.FP32, mode='round')
                qrp_col_inline1721__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(qrp_ones_inline1732__ssa_v0, qrp_idx_fp32_inline1707__ssa_v0)
                qrp_half_inline1729__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_col_inline1721__ssa_v0, 0.5)
                qrp_dup_i32_inline1731__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_half_inline1729__ssa_v0, target_type=pl.INT32, mode='trunc')
                qrp_dup_f_inline1709__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_dup_i32_inline1731__ssa_v0, target_type=pl.FP32, mode='round')
                qrp_dup_idx_inline1726__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_dup_f_inline1709__ssa_v0, target_type=pl.INT32, mode='round')
                qrp_lane_inline1724__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_col_inline1721__ssa_v0, pl.tensor.muls(qrp_dup_f_inline1709__ssa_v0, 2.0))
                qrp_next_col_inline1735__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.adds(qrp_col_inline1721__ssa_v0, 1.0)
                qrp_lane_offset_inline1738__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.muls(qrp_lane_inline1724__ssa_v0, 2.0)
                qrp_swap_f_inline1736__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(qrp_next_col_inline1735__ssa_v0, qrp_lane_offset_inline1738__ssa_v0)
                qrp_swap_idx_inline1741__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(qrp_swap_f_inline1736__ssa_v0, target_type=pl.INT32, mode='round')
                qrp_sign_inline1733__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.subs(pl.tensor.muls(qrp_lane_inline1724__ssa_v0, 2.0), 1.0)
                if qrp_valid_rows_inline1713__ssa_v0 == 8:
                    qrp_cos_rows_full_inline1739__ssa_v0: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_cos_view_inline1725__ssa_v0, [8, 64], [qrp_t0_inline1718__ssa_v0, 0])
                    qrp_sin_rows_full_inline1742__ssa_v0: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.slice(rope_sin_view_inline1720__ssa_v0, [8, 64], [qrp_t0_inline1718__ssa_v0, 0])
                    qrp_cos_full_inline1722__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_cos_rows_full_inline1739__ssa_v0, target_type=pl.FP32, mode='round')
                    qrp_sin_full_inline1715__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(qrp_sin_rows_full_inline1742__ssa_v0, target_type=pl.FP32, mode='round')
                    qrp_cos_il_full_inline1743__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_cos_full_inline1722__ssa_v0, qrp_dup_idx_inline1726__ssa_v0, dim=-1)
                    qrp_sin_il_full_inline1717__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(qrp_sin_full_inline1715__ssa_v0, qrp_dup_idx_inline1726__ssa_v0, dim=-1)
                    qrp_sin_signed_full_inline1737__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(qrp_sin_il_full_inline1717__ssa_v0, qrp_sign_inline1733__ssa_v0)
                    rope_cos_il_view_inline1716__ssa_v1: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32] = pl.tensor.assemble(rope_cos_il_view_inline1716__ssa_v0, qrp_cos_il_full_inline1743__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0])
                    rope_sin_signed_view_inline1714__ssa_v1: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed_view_inline1714__ssa_v0, qrp_sin_signed_full_inline1737__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0])
                    rope_swap_idx_view_inline1740__ssa_v1: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_view_inline1740__ssa_v0, qrp_swap_idx_inline1741__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0])
                    rope_cos_il_view_inline1716__phi_v2, rope_sin_signed_view_inline1714__phi_v2, rope_swap_idx_view_inline1740__phi_v2 = pl.yield_(rope_cos_il_view_inline1716__ssa_v1, rope_sin_signed_view_inline1714__ssa_v1, rope_swap_idx_view_inline1740__ssa_v1)
                else:
                    qrp_cos_rows_tail_inline1730__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.load(rope_cos_view_inline1725__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1713__ssa_v0, 64], target_memory=pl.Mem.Vec)
                    qrp_sin_rows_tail_inline1706__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.load(rope_sin_view_inline1720__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1713__ssa_v0, 64], target_memory=pl.Mem.Vec)
                    qrp_tail_col_inline1705__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                    qrp_tail_dup_f_inline1703__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(qrp_tail_col_inline1705__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                    qrp_tail_lane_inline1708__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(qrp_tail_col_inline1705__ssa_v0, pl.tile.muls(qrp_tail_dup_f_inline1703__ssa_v0, 2.0))
                    qrp_tail_swap_f_inline1711__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(qrp_tail_col_inline1705__ssa_v0, 1.0), pl.tile.muls(qrp_tail_lane_inline1708__ssa_v0, 2.0))
                    qrp_row_seed_inline1710__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                    qrp_row_grid_inline1702__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), qrp_row_seed_inline1710__ssa_v0)
                    qrp_row_offset_inline1701__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(qrp_row_grid_inline1702__ssa_v0, 0, 1)
                    qrp_dup_idx_tail_inline1704__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(qrp_tail_dup_f_inline1703__ssa_v0, qrp_row_offset_inline1701__ssa_v0), target_type=pl.INT32, mode='round')
                    qrp_gather_tmp_inline1699__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    qrp_cos_il_tail_inline1698__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_cos_rows_tail_inline1730__ssa_v0, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail_inline1704__ssa_v0, qrp_gather_tmp_inline1699__ssa_v0)
                    qrp_sin_il_tail_inline1700__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(pl.tile.cast(qrp_sin_rows_tail_inline1706__ssa_v0, target_type=pl.FP32, mode='round'), qrp_dup_idx_tail_inline1704__ssa_v0, qrp_gather_tmp_inline1699__ssa_v0)
                    qrp_tail_sign_inline1723__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.subs(pl.tile.muls(qrp_tail_lane_inline1708__ssa_v0, 2.0), 1.0)
                    qrp_sin_signed_tail_inline1734__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_tail_inline1700__ssa_v0, qrp_tail_sign_inline1723__ssa_v0)
                    pl.tile.store(pl.tile.set_validshape(qrp_cos_il_tail_inline1698__ssa_v0, qrp_valid_rows_inline1713__ssa_v0, 64), [qrp_t0_inline1718__ssa_v0, 0], rope_cos_il_view_inline1716__ssa_v0)
                    pl.tile.store(pl.tile.set_validshape(qrp_sin_signed_tail_inline1734__ssa_v0, qrp_valid_rows_inline1713__ssa_v0, 64), [qrp_t0_inline1718__ssa_v0, 0], rope_sin_signed_view_inline1714__ssa_v0)
                    pl.tile.store(pl.tile.set_validshape(pl.tile.cast(qrp_tail_swap_f_inline1711__ssa_v0, target_type=pl.INT32, mode='round'), qrp_valid_rows_inline1713__ssa_v0, 64), [qrp_t0_inline1718__ssa_v0, 0], rope_swap_idx_view_inline1740__ssa_v0)
                    rope_cos_il_view_inline1716__phi_v2, rope_sin_signed_view_inline1714__phi_v2, rope_swap_idx_view_inline1740__phi_v2 = pl.yield_(rope_cos_il_view_inline1716__ssa_v0, rope_sin_signed_view_inline1714__ssa_v0, rope_swap_idx_view_inline1740__ssa_v0)
            # Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles.
            t_dim_inline1813__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243__phi_v4, 0)
            for tile_base_inline1799__idx_v0 in pl.range(0, t_dim_inline1813__ssa_v0, 512):
                tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(512, t_dim_inline1813__ssa_v0 - tile_base_inline1799__idx_v0)
                with pl.scope():
                    x_view_inline1797__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 4096], pl.BF16] = pl.tensor.reshape(x_normed_t_inline1243__phi_v4, [t_dim_inline1813__ssa_v0, 4096])
                    qr_t_matmul_inline1793__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1798__ssa_v0 + 16 - 1) // 16 * 16
                    qproj_t_matmul_inline1791__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1798__ssa_v0 + 16 - 1) // 16 * 16
                    qproj_full_rows_inline1804__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline1798__ssa_v0 // 64 * 64
                    qr_fp32_inline1834__ssa_v0: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32] = pl.tensor.create([qr_t_matmul_inline1793__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
                        for ts0_inline1784__idx_v0, (qr_fp32_inline1834__iter_v1,) in pl.range(0, qr_t_matmul_inline1793__ssa_v0, 16, init_values=(qr_fp32_inline1834__ssa_v0,)):
                            for nseed0_inline1855__idx_v0, (qr_fp32_inline1834__iter_v3,) in pl.range(0, 1024, 128, init_values=(qr_fp32_inline1834__iter_v1,)):
                                qr_seed_inline1790__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.full([16, 128], dtype=pl.FP32, value=0.0)
                                qr_fp32_inline1834__ssa_v5: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32_inline1834__iter_v3, qr_seed_inline1790__ssa_v0, [ts0_inline1784__idx_v0, nseed0_inline1855__idx_v0])
                                qr_fp32_inline1834__rv_v4: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32] = pl.yield_(qr_fp32_inline1834__ssa_v5)
                            qr_fp32_inline1834__rv_v2: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32] = pl.yield_(qr_fp32_inline1834__rv_v4)
                    for qbg_idx_inline1807__ssa_v0 in pl.spmd(16, name_hint="qr_proj_matmul_spmd", allow_early_resolve=True):
                        q_a_col0_inline1795__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline1807__ssa_v0 // 2 * 128
                        qr_k_base_inline1819__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline1807__ssa_v0 % 2 * 2048
                        for t0_inline1823__idx_v0, (qr_fp32_inline1834__iter_v6,) in pl.range(0, qr_t_matmul_inline1793__ssa_v0, 16, init_values=(qr_fp32_inline1834__rv_v2,)):
                            q_acc_inline1824__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            for db_inline1815__idx_v0, (q_acc_inline1824__iter_v1,) in pl.pipeline(8, stage=2, init_values=(q_acc_inline1824__ssa_v0,)):
                                qr_d0_inline1822__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_base_inline1819__ssa_v0 + db_inline1815__idx_v0 * 256
                                qr_rows_inline1808__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows_inline1798__ssa_v0 - t0_inline1823__idx_v0)
                                x_t0_inline1766__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1799__idx_v0 + t0_inline1823__idx_v0
                                q_x_chunk_bf16_inline1785__ssa_v0: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[qr_rows_inline1808__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_view_inline1797__ssa_v0, [16, 256], [x_t0_inline1766__ssa_v0, qr_d0_inline1822__ssa_v0], [qr_rows_inline1808__ssa_v0, 256])
                                w_chunk_inline1827__ssa_v0: pl.Tensor[[256, 128], pl.BF16] = pl.tensor.slice(wq_a__ssa_v0, [256, 128], [qr_d0_inline1822__ssa_v0, q_a_col0_inline1795__ssa_v0])
                                if db_inline1815__idx_v0 == 0:
                                    q_acc_inline1824__ssa_v3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(q_x_chunk_bf16_inline1785__ssa_v0, w_chunk_inline1827__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                    q_acc_inline1824__phi_v5: pl.Tensor[[16, 128], pl.FP32] = pl.yield_(q_acc_inline1824__ssa_v3)
                                else:
                                    q_acc_inline1824__ssa_v4: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(q_acc_inline1824__iter_v1, q_x_chunk_bf16_inline1785__ssa_v0, w_chunk_inline1827__ssa_v0, a_trans=False, b_trans=False)
                                    q_acc_inline1824__phi_v5: pl.Tensor[[16, 128], pl.FP32] = pl.yield_(q_acc_inline1824__ssa_v4)
                                q_acc_inline1824__rv_v2: pl.Tensor[[16, 128], pl.FP32] = pl.yield_(q_acc_inline1824__phi_v5)
                            qr_fp32_inline1834__ssa_v8: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32] = pl.tensor.assemble(qr_fp32_inline1834__iter_v6, q_acc_inline1824__rv_v2, [t0_inline1823__idx_v0, q_a_col0_inline1795__ssa_v0], atomic=pl.AtomicType.Add)
                            qr_fp32_inline1834__rv_v7: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32] = pl.yield_(qr_fp32_inline1834__ssa_v8)
                    qr_view_inline1775__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8] = pl.tensor.reshape(qr_inline1255__ssa_v0, [t_dim_inline1813__ssa_v0, 1024])
                    qr_scale_view_inline1796__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32] = pl.tensor.reshape(qr_scale_inline1310__ssa_v0, [t_dim_inline1813__ssa_v0, 1])
                    qr_i8_matmul_inline1787__ssa_v0: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8] = pl.tensor.create([qproj_t_matmul_inline1791__ssa_v0, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                    qr_scale_pad_store_inline1814__ssa_v0: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32] = pl.tensor.create([qproj_t_matmul_inline1791__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                    qr_token_tiles_inline1878__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1798__ssa_v0 + 8 - 1) // 8
                    for tg_idx_inline1757__ssa_v0 in pl.spmd(qr_token_tiles_inline1878__ssa_v0, name_hint="qr_rms_norm_quant_spmd", allow_early_resolve=True):
                        tg_inline1758__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1757__ssa_v0 * 8
                        valid_rows_inline1818__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, tile_rows_inline1798__ssa_v0 - tg_inline1758__ssa_v0)
                        out_tg_inline1826__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1799__idx_v0 + tg_inline1758__ssa_v0
                        qr_sq_sum_inline1817__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                        qr_amax_g_inline1782__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0)
                        for qr_rms_col0_inline1886__idx_v0, (qr_amax_g_inline1782__iter_v1, qr_sq_sum_inline1817__iter_v1) in pl.pipeline(0, 1024, 256, stage=2, init_values=(qr_amax_g_inline1782__ssa_v0, qr_sq_sum_inline1817__ssa_v0)):
                            qr_rms_chunk_inline1781__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.slice(qr_fp32_inline1834__rv_v7, [8, 256], [tg_inline1758__ssa_v0, qr_rms_col0_inline1886__idx_v0])
                            qr_rms_sq_inline1779__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.mul(qr_rms_chunk_inline1781__ssa_v0, qr_rms_chunk_inline1781__ssa_v0)
                            qr_rms_row_sum_inline1821__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(qr_rms_sq_inline1779__ssa_v0), [1, 8])
                            qr_sq_sum_inline1817__ssa_v3: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.add(qr_sq_sum_inline1817__iter_v1, qr_rms_row_sum_inline1821__ssa_v0)
                            gamma_rms_cast_inline1774__ssa_v0: pl.Tensor[[256], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_cq__ssa_v0, [256], [qr_rms_col0_inline1886__idx_v0]), target_type=pl.FP32, mode='round')
                            gamma_rms_chunk_inline1772__ssa_v0: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.reshape(gamma_rms_cast_inline1774__ssa_v0, [1, 256])
                            qr_g_inline1786__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.col_expand_mul(qr_rms_chunk_inline1781__ssa_v0, gamma_rms_chunk_inline1772__ssa_v0)
                            qr_g_abs_inline1836__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.abs(qr_g_inline1786__ssa_v0)
                            qr_g_row_max_inline1770__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(qr_g_abs_inline1836__ssa_v0), [1, 8])
                            qr_amax_g_inline1782__ssa_v3: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qr_amax_g_inline1782__iter_v1, qr_g_row_max_inline1770__ssa_v0)
                            qr_amax_g_inline1782__rv_v2, qr_sq_sum_inline1817__rv_v2 = pl.yield_(qr_amax_g_inline1782__ssa_v3, qr_sq_sum_inline1817__ssa_v3)
                        qr_inv_rms_inline1769__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(qr_sq_sum_inline1817__rv_v2, 0.0009765625), 9.9999999999999995e-07), high_precision=True)
                        qr_inv_rms_t_inline1802__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qr_inv_rms_inline1769__ssa_v0, [8, 1])
                        qr_amax_floor_inline1829__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0001)
                        qr_amax_normed_inline1792__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.mul(qr_inv_rms_inline1769__ssa_v0, qr_amax_g_inline1782__rv_v2)
                        qr_tile_amax_inline1828__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qr_amax_floor_inline1829__ssa_v0, qr_amax_normed_inline1792__ssa_v0)
                        qr_scale_quant_row_inline1767__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 8], dtype=pl.FP32, value=127.0), qr_tile_amax_inline1828__ssa_v0)
                        qr_scale_quant_t_inline1765__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qr_scale_quant_row_inline1767__ssa_v0, [8, 1])
                        qr_tile_scale_dq_inline1764__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(qr_scale_quant_row_inline1767__ssa_v0), [8, 1])
                        qr_scale_pad_store_inline1814__ssa_v1: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32] = pl.tensor.assemble(qr_scale_pad_store_inline1814__ssa_v0, qr_tile_scale_dq_inline1764__ssa_v0, [tg_inline1758__ssa_v0, 0])
                        if valid_rows_inline1818__ssa_v0 == 8:
                            qr_scale_view_inline1796__ssa_v1: pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32] = pl.tensor.assemble(qr_scale_view_inline1796__ssa_v0, qr_tile_scale_dq_inline1764__ssa_v0, [out_tg_inline1826__ssa_v0, 0])
                            qr_scale_view_inline1796__phi_v2: pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32] = pl.yield_(qr_scale_view_inline1796__ssa_v1)
                        else:
                            qr_scale_tail_inline1777__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1818__ssa_v0, 1])] = pl.tile.load(qr_scale_pad_store_inline1814__ssa_v1, [tg_inline1758__ssa_v0, 0], [8, 1], [valid_rows_inline1818__ssa_v0, 1], target_memory=pl.Mem.Vec)
                            pl.tile.store(qr_scale_tail_inline1777__ssa_v0, [out_tg_inline1826__ssa_v0, 0], qr_scale_view_inline1796__ssa_v0)
                            qr_scale_view_inline1796__phi_v2: pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32] = pl.yield_(qr_scale_view_inline1796__ssa_v0)
                        for qa_inline1761__idx_v0, (qr_i8_matmul_inline1787__iter_v1, qr_view_inline1775__iter_v1) in pl.pipeline(0, 1024, 256, stage=2, init_values=(qr_i8_matmul_inline1787__ssa_v0, qr_view_inline1775__ssa_v0)):
                            qr_chunk_inline1780__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.slice(qr_fp32_inline1834__rv_v7, [8, 256], [tg_inline1758__ssa_v0, qa_inline1761__idx_v0])
                            gamma_q_cast_inline1776__ssa_v0: pl.Tensor[[256], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_cq__ssa_v0, [256], [qa_inline1761__idx_v0]), target_type=pl.FP32, mode='round')
                            gamma_q_chunk_inline1794__ssa_v0: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.reshape(gamma_q_cast_inline1776__ssa_v0, [1, 256])
                            qr_q_normed_inline1810__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(qr_chunk_inline1780__ssa_v0, qr_inv_rms_t_inline1802__ssa_v0), gamma_q_chunk_inline1794__ssa_v0)
                            qr_q_scaled_inline1805__ssa_v0: pl.Tensor[[8, 256], pl.FP32] = pl.tensor.row_expand_mul(qr_q_normed_inline1810__ssa_v0, qr_scale_quant_t_inline1765__ssa_v0)
                            qr_q_i32_inline1762__ssa_v0: pl.Tensor[[8, 256], pl.INT32] = pl.tensor.cast(qr_q_scaled_inline1805__ssa_v0, target_type=pl.INT32, mode='rint')
                            qr_q_half_inline1756__ssa_v0: pl.Tensor[[8, 256], pl.FP16] = pl.tensor.cast(qr_q_i32_inline1762__ssa_v0, target_type=pl.FP16, mode='round')
                            qr_q_i8_inline1831__ssa_v0: pl.Tensor[[8, 256], pl.INT8] = pl.tensor.cast(qr_q_half_inline1756__ssa_v0, target_type=pl.INT8, mode='trunc')
                            qr_i8_matmul_inline1787__ssa_v3: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8] = pl.tensor.assemble(qr_i8_matmul_inline1787__iter_v1, qr_q_i8_inline1831__ssa_v0, [tg_inline1758__ssa_v0, qa_inline1761__idx_v0])
                            if valid_rows_inline1818__ssa_v0 == 8:
                                qr_view_inline1775__ssa_v3: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8] = pl.tensor.assemble(qr_view_inline1775__iter_v1, qr_q_i8_inline1831__ssa_v0, [out_tg_inline1826__ssa_v0, qa_inline1761__idx_v0])
                                qr_view_inline1775__phi_v4: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8] = pl.yield_(qr_view_inline1775__ssa_v3)
                            else:
                                qr_q_tail_inline1832__ssa_v0: pl.Tile[[8, 256], pl.INT8, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1818__ssa_v0, 256])] = pl.tile.load(qr_i8_matmul_inline1787__ssa_v3, [tg_inline1758__ssa_v0, qa_inline1761__idx_v0], [8, 256], [valid_rows_inline1818__ssa_v0, 256], target_memory=pl.Mem.Vec)
                                pl.tile.store(qr_q_tail_inline1832__ssa_v0, [out_tg_inline1826__ssa_v0, qa_inline1761__idx_v0], qr_view_inline1775__iter_v1)
                                qr_view_inline1775__phi_v4: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8] = pl.yield_(qr_view_inline1775__iter_v1)
                            qr_i8_matmul_inline1787__rv_v2, qr_view_inline1775__rv_v2 = pl.yield_(qr_i8_matmul_inline1787__ssa_v3, qr_view_inline1775__phi_v4)
                    q_proj_i32_inline1835__ssa_v0: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32] = pl.tensor.create([qproj_t_matmul_inline1791__ssa_v0, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    for qproj_n_idx_inline1841__ssa_v0 in pl.spmd(64, name_hint="qproj_matmul_spmd"):
                        w_col0_inline1842__ssa_v0: pl.Scalar[pl.INDEX] = qproj_n_idx_inline1841__ssa_v0 * 512
                        for t0_inline1861__idx_v0, (q_proj_i32_inline1835__iter_v1,) in pl.range(0, qproj_full_rows_inline1804__ssa_v0, 64, init_values=(q_proj_i32_inline1835__ssa_v0,)):
                            col_acc_inline1847__ssa_v0: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.create([64, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                            for qr_proj_col0_inline1848__idx_v0, (col_acc_inline1847__iter_v1,) in pl.pipeline(0, 1024, 128, stage=2, init_values=(col_acc_inline1847__ssa_v0,)):
                                qr_i8_chunk_inline1879__ssa_v0: pl.Tensor[[64, 128], pl.INT8] = pl.tensor.slice(qr_i8_matmul_inline1787__rv_v2, [64, 128], [t0_inline1861__idx_v0, qr_proj_col0_inline1848__idx_v0])
                                wq_chunk_inline1845__ssa_v0: pl.Tensor[[128, 512], pl.INT8] = pl.tensor.slice(wq_b__ssa_v0, [128, 512], [qr_proj_col0_inline1848__idx_v0, w_col0_inline1842__ssa_v0])
                                if qr_proj_col0_inline1848__idx_v0 == 0:
                                    col_acc_inline1847__ssa_v3: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.matmul(qr_i8_chunk_inline1879__ssa_v0, wq_chunk_inline1845__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                                    col_acc_inline1847__phi_v5: pl.Tensor[[64, 512], pl.INT32] = pl.yield_(col_acc_inline1847__ssa_v3)
                                else:
                                    col_acc_inline1847__ssa_v4: pl.Tensor[[64, 512], pl.INT32] = pl.tensor.matmul_acc(col_acc_inline1847__iter_v1, qr_i8_chunk_inline1879__ssa_v0, wq_chunk_inline1845__ssa_v0, a_trans=False, b_trans=False)
                                    col_acc_inline1847__phi_v5: pl.Tensor[[64, 512], pl.INT32] = pl.yield_(col_acc_inline1847__ssa_v4)
                                col_acc_inline1847__rv_v2: pl.Tensor[[64, 512], pl.INT32] = pl.yield_(col_acc_inline1847__phi_v5)
                            q_proj_i32_inline1835__ssa_v3: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32] = pl.tensor.assemble(q_proj_i32_inline1835__iter_v1, col_acc_inline1847__rv_v2, [t0_inline1861__idx_v0, w_col0_inline1842__ssa_v0])
                            q_proj_i32_inline1835__rv_v2: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32] = pl.yield_(q_proj_i32_inline1835__ssa_v3)
                        tail_w_col0_inline1809__ssa_v0: pl.Scalar[pl.INDEX] = w_col0_inline1842__ssa_v0
                        for tail_t0_inline1850__idx_v0, (q_proj_i32_inline1835__iter_v4,) in pl.range(qproj_full_rows_inline1804__ssa_v0, qproj_t_matmul_inline1791__ssa_v0, 16, init_values=(q_proj_i32_inline1835__rv_v2,)):
                            qproj_tail_rows_inline1851__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows_inline1798__ssa_v0 - tail_t0_inline1850__idx_v0)
                            tail_acc_inline1801__ssa_v0: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.create([16, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                            for tail_qr_col0_inline1852__idx_v0, (tail_acc_inline1801__iter_v1,) in pl.pipeline(0, 1024, 128, stage=2, init_values=(tail_acc_inline1801__ssa_v0,)):
                                qr_i8_tail_inline1763__ssa_v0: pl.Tensor[[16, 128], pl.INT8, pl.TensorView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr_i8_matmul_inline1787__rv_v2, [16, 128], [tail_t0_inline1850__idx_v0, tail_qr_col0_inline1852__idx_v0], [qproj_tail_rows_inline1851__ssa_v0, 128])
                                wq_tail_inline1853__ssa_v0: pl.Tensor[[128, 512], pl.INT8] = pl.tensor.slice(wq_b__ssa_v0, [128, 512], [tail_qr_col0_inline1852__idx_v0, tail_w_col0_inline1809__ssa_v0])
                                if tail_qr_col0_inline1852__idx_v0 == 0:
                                    tail_acc_inline1801__ssa_v3: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul(qr_i8_tail_inline1763__ssa_v0, wq_tail_inline1853__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                                    tail_acc_inline1801__phi_v5: pl.Tensor[[16, 512], pl.INT32] = pl.yield_(tail_acc_inline1801__ssa_v3)
                                else:
                                    tail_acc_inline1801__ssa_v4: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul_acc(tail_acc_inline1801__iter_v1, qr_i8_tail_inline1763__ssa_v0, wq_tail_inline1853__ssa_v0, a_trans=False, b_trans=False)
                                    tail_acc_inline1801__phi_v5: pl.Tensor[[16, 512], pl.INT32] = pl.yield_(tail_acc_inline1801__ssa_v4)
                                tail_acc_inline1801__rv_v2: pl.Tensor[[16, 512], pl.INT32] = pl.yield_(tail_acc_inline1801__phi_v5)
                            q_proj_i32_inline1835__ssa_v6: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32] = pl.tensor.assemble(q_proj_i32_inline1835__iter_v4, tail_acc_inline1801__rv_v2, [tail_t0_inline1850__idx_v0, tail_w_col0_inline1809__ssa_v0])
                            q_proj_i32_inline1835__rv_v5: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32] = pl.yield_(q_proj_i32_inline1835__ssa_v6)
                    q_flat_inline1856__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16] = pl.tensor.reshape(q_inline1246__ssa_v0, [t_dim_inline1813__ssa_v0, 32768])
                    for hg_idx_inline1857__ssa_v0 in pl.spmd(16, name_hint="qproj_dequant_rms_nope_rope_spmd", allow_early_resolve=True):
                        hg_inline1858__ssa_v0: pl.Scalar[pl.INDEX] = hg_idx_inline1857__ssa_v0 * 4
                        for tg_inline1844__idx_v0, (out_tg_inline1826__iter_v1, q_flat_inline1856__iter_v1) in pl.range(0, tile_rows_inline1798__ssa_v0, 8, init_values=(out_tg_inline1826__ssa_v0, q_flat_inline1856__ssa_v0)):
                            out_tg_inline1826__ssa_v3: pl.Scalar[pl.INDEX] = tile_base_inline1799__idx_v0 + tg_inline1844__idx_v0
                            if tg_inline1844__idx_v0 + 8 <= tile_rows_inline1798__ssa_v0:
                                qr_scale_dq_t_inline1860__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(qr_scale_pad_store_inline1814__ssa_v1, [8, 1], [tg_inline1844__idx_v0, 0])
                                q_cos_il_inline1849__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(q_cos_il_inline1311__ssa_v0, [8, 64], [out_tg_inline1826__ssa_v3, 0])
                                q_sin_signed_inline1862__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(q_sin_signed_inline1295__ssa_v0, [8, 64], [out_tg_inline1826__ssa_v3, 0])
                                q_swap_idx_inline1788__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.slice(q_swap_idx_inline1313__ssa_v0, [8, 64], [out_tg_inline1826__ssa_v3, 0])
                                for h_inner_inline1863__idx_v0, (q_flat_inline1856__iter_v3,) in pl.pipeline(4, stage=2, init_values=(q_flat_inline1856__iter_v1,)):
                                    h_inline1768__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline1858__ssa_v0 + h_inner_inline1863__idx_v0
                                    h0_inline1800__ssa_v0: pl.Scalar[pl.INDEX] = h_inline1768__ssa_v0 * 512
                                    q_head_acc_inline1865__ssa_v0: pl.Tensor[[8, 512], pl.INT32] = pl.tensor.slice(q_proj_i32_inline1835__rv_v5, [8, 512], [tg_inline1844__idx_v0, h0_inline1800__ssa_v0])
                                    q_head_scale_inline1843__ssa_v0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(wq_b_scale__ssa_v0, [512], [h0_inline1800__ssa_v0]), [1, 512])
                                    q_head_acc_fp32_inline1866__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.cast(q_head_acc_inline1865__ssa_v0, target_type=pl.FP32, mode='none')
                                    q_head_row_scaled_inline1867__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.row_expand_mul(q_head_acc_fp32_inline1866__ssa_v0, qr_scale_dq_t_inline1860__ssa_v0)
                                    q_head_dq_inline1825__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.col_expand_mul(q_head_row_scaled_inline1867__ssa_v0, q_head_scale_inline1843__ssa_v0)
                                    q_head_sq_inline1868__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(q_head_dq_inline1825__ssa_v0, q_head_dq_inline1825__ssa_v0)
                                    q_head_sq_row_inline1887__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_sum(q_head_sq_inline1868__ssa_v0)
                                    q_head_sq_sum_inline1870__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(q_head_sq_row_inline1887__ssa_v0, [1, 8])
                                    q_head_sq_mean_inline1846__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.muls(q_head_sq_sum_inline1870__ssa_v0, 0.001953125)
                                    q_head_var_inline1864__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.adds(q_head_sq_mean_inline1846__ssa_v0, 9.9999999999999995e-07)
                                    q_head_inv_rms_inline1871__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.rsqrt(q_head_var_inline1864__ssa_v0, high_precision=True)
                                    q_head_inv_rms_t_inline1771__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(q_head_inv_rms_inline1871__ssa_v0, [8, 1])
                                    q_nope_normed_inline1872__ssa_v0: pl.Tensor[[8, 448], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.slice(q_head_dq_inline1825__ssa_v0, [8, 448], [0, 0]), q_head_inv_rms_t_inline1771__ssa_v0)
                                    q_nope_bf16_inline1873__ssa_v0: pl.Tensor[[8, 448], pl.BF16] = pl.tensor.cast(q_nope_normed_inline1872__ssa_v0, target_type=pl.BF16, mode='rint')
                                    q_flat_inline1856__ssa_v5: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16] = pl.tensor.assemble(q_flat_inline1856__iter_v3, q_nope_bf16_inline1873__ssa_v0, [out_tg_inline1826__ssa_v3, h0_inline1800__ssa_v0])
                                    q_rope_chunk_raw_inline1837__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.slice(q_head_dq_inline1825__ssa_v0, [8, 64], [0, 448])
                                    q_rope_chunk_inline1811__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.row_expand_mul(q_rope_chunk_raw_inline1837__ssa_v0, q_head_inv_rms_t_inline1771__ssa_v0)
                                    q_rope_swapped_inline1778__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(q_rope_chunk_inline1811__ssa_v0, q_swap_idx_inline1788__ssa_v0, dim=-1)
                                    q_rope_base_inline1838__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(q_rope_chunk_inline1811__ssa_v0, q_cos_il_inline1849__ssa_v0)
                                    q_rope_delta_inline1874__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.mul(q_rope_swapped_inline1778__ssa_v0, q_sin_signed_inline1862__ssa_v0)
                                    q_rope_rot_inline1875__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.add(q_rope_base_inline1838__ssa_v0, q_rope_delta_inline1874__ssa_v0)
                                    q_rope_bf16_inline1833__ssa_v0: pl.Tensor[[8, 64], pl.BF16] = pl.tensor.cast(q_rope_rot_inline1875__ssa_v0, target_type=pl.BF16, mode='rint')
                                    q_flat_inline1856__ssa_v6: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16] = pl.tensor.assemble(q_flat_inline1856__ssa_v5, q_rope_bf16_inline1833__ssa_v0, [out_tg_inline1826__ssa_v3, h0_inline1800__ssa_v0 + 448])
                                    q_flat_inline1856__rv_v4: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16] = pl.yield_(q_flat_inline1856__ssa_v6)
                                q_flat_inline1856__phi_v7: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16] = pl.yield_(q_flat_inline1856__rv_v4)
                            else:
                                valid_tail_rows_inline1859__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline1798__ssa_v0 - tg_inline1844__idx_v0
                                qr_scale_dq_tail_inline1876__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(qr_scale_pad_store_inline1814__ssa_v1, [tg_inline1844__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                                q_cos_il_tail_inline1880__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1859__ssa_v0, 64])] = pl.tile.load(q_cos_il_inline1311__ssa_v0, [out_tg_inline1826__ssa_v3, 0], [8, 64], [valid_tail_rows_inline1859__ssa_v0, 64], target_memory=pl.Mem.Vec)
                                q_sin_signed_tail_inline1882__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1859__ssa_v0, 64])] = pl.tile.load(q_sin_signed_inline1295__ssa_v0, [out_tg_inline1826__ssa_v3, 0], [8, 64], [valid_tail_rows_inline1859__ssa_v0, 64], target_memory=pl.Mem.Vec)
                                q_col_inline1883__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([8, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                                q_dup_f_inline1884__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(q_col_inline1883__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                                q_lane_inline1869__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(q_col_inline1883__ssa_v0, pl.tile.muls(q_dup_f_inline1884__ssa_v0, 2.0))
                                q_swap_f_inline1803__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(q_col_inline1883__ssa_v0, 1.0), pl.tile.muls(q_lane_inline1869__ssa_v0, 2.0))
                                q_row_seed_inline1759__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                                q_row_grid_inline1820__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 8], dtype=pl.FP32, value=1.0), q_row_seed_inline1759__ssa_v0)
                                q_row_offset_inline1885__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(q_row_grid_inline1820__ssa_v0, 0, 1)
                                q_swap_idx_tail_inline1877__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(q_swap_f_inline1803__ssa_v0, q_row_offset_inline1885__ssa_v0), target_type=pl.INT32, mode='round')
                                q_head_reduce_tmp_inline1760__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                q_gather_tmp_inline1888__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                                for h_inner_tail_inline1773__idx_v0 in pl.range(4):
                                    h_tail_inline1789__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline1858__ssa_v0 + h_inner_tail_inline1773__idx_v0
                                    h0_tail_inline1881__ssa_v0: pl.Scalar[pl.INDEX] = h_tail_inline1789__ssa_v0 * 512
                                    q_head_acc_tail_inline1839__ssa_v0: pl.Tile[[8, 512], pl.INT32, pl.Mem.Vec] = pl.tile.load(q_proj_i32_inline1835__rv_v5, [tg_inline1844__idx_v0, h0_tail_inline1881__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                                    q_head_scale_input_tail_inline1840__ssa_v0: pl.Tile[[512], pl.FP32, pl.Mem.Vec] = pl.tile.load(wq_b_scale__ssa_v0, [h0_tail_inline1881__ssa_v0], [512], [512], target_memory=pl.Mem.Vec)
                                    q_head_scale_tail_inline1816__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(q_head_scale_input_tail_inline1840__ssa_v0, [1, 512])
                                    q_head_acc_fp32_tail_inline1755__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.cast(q_head_acc_tail_inline1839__ssa_v0, target_type=pl.FP32, mode='none')
                                    q_head_row_scaled_tail_inline1754__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_head_acc_fp32_tail_inline1755__ssa_v0, qr_scale_dq_tail_inline1876__ssa_v0)
                                    q_head_dq_tail_inline1806__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_head_row_scaled_tail_inline1754__ssa_v0, q_head_scale_tail_inline1816__ssa_v0)
                                    q_head_sq_tail_inline1753__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_head_dq_tail_inline1806__ssa_v0, q_head_dq_tail_inline1806__ssa_v0)
                                    q_head_sq_sum_tail_inline1752__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_tail_inline1753__ssa_v0, q_head_reduce_tmp_inline1760__ssa_v0)
                                    q_head_inv_rms_tail_inline1783__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(q_head_sq_sum_tail_inline1752__ssa_v0, 0.001953125), 9.9999999999999995e-07)))
                                    q_nope_normed_tail_inline1751__ssa_v0: pl.Tile[[8, 448], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(pl.tile.slice(q_head_dq_tail_inline1806__ssa_v0, [8, 448], [0, 0]), q_head_inv_rms_tail_inline1783__ssa_v0)
                                    q_nope_bf16_tail_inline1750__ssa_v0: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_nope_normed_tail_inline1751__ssa_v0, target_type=pl.BF16, mode='rint')
                                    q_nope_valid_inline1749__ssa_v0: pl.Tile[[8, 448], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1859__ssa_v0, 448])] = pl.tile.set_validshape(q_nope_bf16_tail_inline1750__ssa_v0, valid_tail_rows_inline1859__ssa_v0, 448)
                                    pl.tile.store(q_nope_valid_inline1749__ssa_v0, [out_tg_inline1826__ssa_v3, h0_tail_inline1881__ssa_v0], q_flat_inline1856__iter_v1)
                                    q_rope_chunk_raw_tail_inline1748__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail_inline1806__ssa_v0, [8, 64], [0, 448])
                                    q_rope_chunk_tail_inline1812__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(q_rope_chunk_raw_tail_inline1748__ssa_v0, q_head_inv_rms_tail_inline1783__ssa_v0)
                                    q_rope_swapped_tail_inline1747__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(q_rope_chunk_tail_inline1812__ssa_v0, q_swap_idx_tail_inline1877__ssa_v0, q_gather_tmp_inline1888__ssa_v0)
                                    q_rope_base_tail_inline1746__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_rope_chunk_tail_inline1812__ssa_v0, q_cos_il_tail_inline1880__ssa_v0)
                                    q_rope_delta_tail_inline1745__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(q_rope_swapped_tail_inline1747__ssa_v0, q_sin_signed_tail_inline1882__ssa_v0)
                                    q_rope_rot_tail_inline1854__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(q_rope_base_tail_inline1746__ssa_v0, q_rope_delta_tail_inline1745__ssa_v0)
                                    q_rope_bf16_tail_inline1744__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_rope_rot_tail_inline1854__ssa_v0, target_type=pl.BF16, mode='rint')
                                    q_rope_valid_inline1830__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1859__ssa_v0, 64])] = pl.tile.set_validshape(q_rope_bf16_tail_inline1744__ssa_v0, valid_tail_rows_inline1859__ssa_v0, 64)
                                    pl.tile.store(q_rope_valid_inline1830__ssa_v0, [out_tg_inline1826__ssa_v3, h0_tail_inline1881__ssa_v0 + 448], q_flat_inline1856__iter_v1)
                                q_flat_inline1856__phi_v7: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16] = pl.yield_(q_flat_inline1856__iter_v1)
                            out_tg_inline1826__rv_v2, q_flat_inline1856__rv_v2 = pl.yield_(out_tg_inline1826__ssa_v3, q_flat_inline1856__phi_v7)
            # KV LoRA, RMSNorm, and RoPE over bounded dense tiles.
            t_dim_inline1923__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240__rv_v5, 0)
            for tile_base_inline1954__idx_v0 in pl.range(0, t_dim_inline1923__ssa_v0, 512):
                tile_rows_inline1928__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(512, t_dim_inline1923__ssa_v0 - tile_base_inline1954__idx_v0)
                with pl.scope():
                    x_view_inline1914__ssa_v0: pl.Tensor[[t_dim_inline1923__ssa_v0, 4096], pl.BF16] = pl.tensor.reshape(x_normed_full_inline1240__rv_v5, [t_dim_inline1923__ssa_v0, 4096])
                    t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1928__ssa_v0 + 16 - 1) // 16 * 16
                    kv_fp32_inline1920__ssa_v0: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32] = pl.tensor.create([t_matmul_inline1930__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
                        for kts0_inline1934__idx_v0, (kv_fp32_inline1920__iter_v1,) in pl.range(0, t_matmul_inline1930__ssa_v0, 16, init_values=(kv_fp32_inline1920__ssa_v0,)):
                            for kvseed0_inline1936__idx_v0, (kv_fp32_inline1920__iter_v3,) in pl.range(0, 512, 128, init_values=(kv_fp32_inline1920__iter_v1,)):
                                kv_seed_inline1926__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.full([16, 128], dtype=pl.FP32, value=0.0)
                                kv_fp32_inline1920__ssa_v5: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32] = pl.tensor.assemble(kv_fp32_inline1920__iter_v3, kv_seed_inline1926__ssa_v0, [kts0_inline1934__idx_v0, kvseed0_inline1936__idx_v0])
                                kv_fp32_inline1920__rv_v4: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32] = pl.yield_(kv_fp32_inline1920__ssa_v5)
                            kv_fp32_inline1920__rv_v2: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32] = pl.yield_(kv_fp32_inline1920__rv_v4)
                    with pl.spmd(32, name_hint="kv_proj_matmul_spmd", deps=[late_dep_inline1297__ssa_v0]) as _kv_tid_inline1935__ssa_v0:
                        kbg_inline1940__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                        kv_col0_inline1937__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1940__ssa_v0 // 8 * 128
                        kv_k_base_inline1962__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1940__ssa_v0 // 4 % 2 * 2048
                        kv_m_group_inline1922__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1940__ssa_v0 % 4
                        for t0_inline1943__idx_v0, (kv_fp32_inline1920__iter_v6,) in pl.range(kv_m_group_inline1922__ssa_v0 * 16, t_matmul_inline1930__ssa_v0, 64, init_values=(kv_fp32_inline1920__rv_v2,)):
                            kv_acc_inline1938__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            for db_inline1932__idx_v0, (kv_acc_inline1938__iter_v1,) in pl.pipeline(8, stage=2, init_values=(kv_acc_inline1938__ssa_v0,)):
                                d0_inline1913__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline1962__ssa_v0 + db_inline1932__idx_v0 * 256
                                kv_rows_inline1911__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows_inline1928__ssa_v0 - t0_inline1943__idx_v0)
                                x_t0_inline1918__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1954__idx_v0 + t0_inline1943__idx_v0
                                kv_x_chunk_bf16_inline1910__ssa_v0: pl.Tensor[[16, 256], pl.BF16, pl.TensorView(valid_shape=[kv_rows_inline1911__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_view_inline1914__ssa_v0, [16, 256], [x_t0_inline1918__ssa_v0, d0_inline1913__ssa_v0], [kv_rows_inline1911__ssa_v0, 256])
                                wkv_chunk_inline1933__ssa_v0: pl.Tensor[[256, 128], pl.BF16] = pl.tensor.slice(wkv__ssa_v0, [256, 128], [d0_inline1913__ssa_v0, kv_col0_inline1937__ssa_v0])
                                if db_inline1932__idx_v0 == 0:
                                    kv_acc_inline1938__ssa_v3: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul(kv_x_chunk_bf16_inline1910__ssa_v0, wkv_chunk_inline1933__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                    kv_acc_inline1938__phi_v5: pl.Tensor[[16, 128], pl.FP32] = pl.yield_(kv_acc_inline1938__ssa_v3)
                                else:
                                    kv_acc_inline1938__ssa_v4: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.matmul_acc(kv_acc_inline1938__iter_v1, kv_x_chunk_bf16_inline1910__ssa_v0, wkv_chunk_inline1933__ssa_v0, a_trans=False, b_trans=False)
                                    kv_acc_inline1938__phi_v5: pl.Tensor[[16, 128], pl.FP32] = pl.yield_(kv_acc_inline1938__ssa_v4)
                                kv_acc_inline1938__rv_v2: pl.Tensor[[16, 128], pl.FP32] = pl.yield_(kv_acc_inline1938__phi_v5)
                            kv_fp32_inline1920__ssa_v8: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32] = pl.tensor.assemble(kv_fp32_inline1920__iter_v6, kv_acc_inline1938__rv_v2, [t0_inline1943__idx_v0, kv_col0_inline1937__ssa_v0], atomic=pl.AtomicType.Add)
                            kv_fp32_inline1920__rv_v7: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32] = pl.yield_(kv_fp32_inline1920__ssa_v8)
                    kv_view_inline1909__ssa_v0: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16] = pl.tensor.reshape(kv_full_inline1265__ssa_v0, [t_dim_inline1923__ssa_v0, 512])
                    kv_token_tiles_inline1972__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1928__ssa_v0 + 16 - 1) // 16
                    for tg_idx_inline1916__ssa_v0 in pl.spmd(kv_token_tiles_inline1972__ssa_v0, name_hint="kv_rms_norm_rope_spmd"):
                        tg_inline1915__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1916__ssa_v0 * 16
                        valid_rows_inline1931__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, tile_rows_inline1928__ssa_v0 - tg_inline1915__ssa_v0)
                        out_tg_inline1924__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1954__idx_v0 + tg_inline1915__ssa_v0
                        if valid_rows_inline1931__ssa_v0 == 16:
                            kv_sq_sum_inline1961__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                            for kv_sq_col0_inline1944__idx_v0, (kv_sq_sum_inline1961__iter_v1,) in pl.pipeline(0, 512, 64, stage=2, init_values=(kv_sq_sum_inline1961__ssa_v0,)):
                                kv_chunk_inline1941__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32_inline1920__rv_v7, [16, 64], [tg_inline1915__ssa_v0, kv_sq_col0_inline1944__idx_v0])
                                kv_sq_inline1968__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_chunk_inline1941__ssa_v0, kv_chunk_inline1941__ssa_v0)
                                kv_row_sum_inline1946__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_sq_inline1968__ssa_v0), [1, 16])
                                kv_sq_sum_inline1961__ssa_v3: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(kv_sq_sum_inline1961__iter_v1, kv_row_sum_inline1946__ssa_v0)
                                kv_sq_sum_inline1961__rv_v2: pl.Tensor[[1, 16], pl.FP32] = pl.yield_(kv_sq_sum_inline1961__ssa_v3)
                            kv_inv_rms_inline1942__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(kv_sq_sum_inline1961__rv_v2, 0.001953125), 9.9999999999999995e-07), high_precision=True)
                            kv_inv_rms_t_inline1949__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(kv_inv_rms_inline1942__ssa_v0, [16, 1])
                            for n0_inline1950__idx_v0, (kv_view_inline1909__iter_v1,) in pl.pipeline(0, 448, 64, stage=2, init_values=(kv_view_inline1909__ssa_v0,)):
                                kv_chunk_inline1941__ssa_v1: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32_inline1920__rv_v7, [16, 64], [tg_inline1915__ssa_v0, n0_inline1950__idx_v0])
                                gamma_kv_cast_inline1948__ssa_v0: pl.Tensor[[64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_ckv__ssa_v0, [64], [n0_inline1950__idx_v0]), target_type=pl.FP32, mode='round')
                                gamma_kv_chunk_inline1951__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(gamma_kv_cast_inline1948__ssa_v0, [1, 64])
                                kv_normed_inline1925__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_chunk_inline1941__ssa_v1, kv_inv_rms_t_inline1949__ssa_v0), gamma_kv_chunk_inline1951__ssa_v0)
                                kv_normed_bf16_inline1908__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(kv_normed_inline1925__ssa_v0, target_type=pl.BF16, mode='rint')
                                kv_view_inline1909__ssa_v3: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16] = pl.tensor.assemble(kv_view_inline1909__iter_v1, kv_normed_bf16_inline1908__ssa_v0, [out_tg_inline1924__ssa_v0, n0_inline1950__idx_v0])
                                kv_view_inline1909__rv_v2: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16] = pl.yield_(kv_view_inline1909__ssa_v3)
                            gamma_rope_cast_inline1955__ssa_v0: pl.Tensor[[64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(gamma_ckv__ssa_v0, [64], [448]), target_type=pl.FP32, mode='round')
                            gamma_rope_inline1957__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(gamma_rope_cast_inline1955__ssa_v0, [1, 64])
                            kv_rope_chunk_inline1959__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_fp32_inline1920__rv_v7, [16, 64], [tg_inline1915__ssa_v0, 448])
                            kv_rope_norm_chunk_inline1945__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_chunk_inline1959__ssa_v0, kv_inv_rms_t_inline1949__ssa_v0), gamma_rope_inline1957__ssa_v0)
                            kv_cos_il_full_inline1963__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_cos_il_inline1258__ssa_v0, [16, 64], [out_tg_inline1924__ssa_v0, 0])
                            kv_sin_signed_full_inline1956__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(kv_sin_signed_inline1301__ssa_v0, [16, 64], [out_tg_inline1924__ssa_v0, 0])
                            kv_swap_idx_full_inline1953__ssa_v0: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.slice(kv_swap_idx_inline1305__ssa_v0, [16, 64], [out_tg_inline1924__ssa_v0, 0])
                            kv_swapped_full_inline1912__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(kv_rope_norm_chunk_inline1945__ssa_v0, kv_swap_idx_full_inline1953__ssa_v0, dim=-1)
                            kv_rope_rot_full_inline1921__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(kv_rope_norm_chunk_inline1945__ssa_v0, kv_cos_il_full_inline1963__ssa_v0), pl.tensor.mul(kv_swapped_full_inline1912__ssa_v0, kv_sin_signed_full_inline1956__ssa_v0))
                            kv_rope_i16_full_inline1965__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(kv_rope_rot_full_inline1921__ssa_v0, target_type=pl.BF16, mode='rint')
                            kv_view_inline1909__ssa_v4: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16] = pl.tensor.assemble(kv_view_inline1909__rv_v2, kv_rope_i16_full_inline1965__ssa_v0, [out_tg_inline1924__ssa_v0, 448])
                            kv_view_inline1909__phi_v5: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16] = pl.yield_(kv_view_inline1909__ssa_v4)
                        else:
                            kv_reduce_tmp_inline1967__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            kv_sq_sum_tail_inline1970__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                            for kv_sq_col0_tail_inline1969__idx_v0, (kv_sq_sum_tail_inline1970__iter_v1,) in pl.pipeline(0, 512, 64, stage=2, init_values=(kv_sq_sum_tail_inline1970__ssa_v0,)):
                                kv_chunk_tail_inline1939__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, kv_sq_col0_tail_inline1969__idx_v0], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                                kv_sq_tail_inline1971__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.mul(kv_chunk_tail_inline1939__ssa_v0, kv_chunk_tail_inline1939__ssa_v0)
                                kv_row_sum_tail_inline1973__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1931__ssa_v0])] = pl.tile.reshape(pl.tile.row_sum(kv_sq_tail_inline1971__ssa_v0, kv_reduce_tmp_inline1967__ssa_v0), [1, 16])
                                kv_sq_sum_tail_inline1970__ssa_v3: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.add(kv_sq_sum_tail_inline1970__iter_v1, kv_row_sum_tail_inline1973__ssa_v0)
                                kv_sq_sum_tail_inline1970__rv_v2: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.yield_(kv_sq_sum_tail_inline1970__ssa_v3)
                            kv_inv_rms_tail_inline1958__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.recip(pl.tile.sqrt(pl.tile.adds(pl.tile.muls(kv_sq_sum_tail_inline1970__rv_v2, 0.001953125), 9.9999999999999995e-07)))
                            kv_inv_rms_t_tail_inline1947__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_tail_inline1958__ssa_v0, [16, 1])
                            for n0_tail_inline1964__idx_v0 in pl.pipeline(0, 448, 64, stage=2):
                                kv_chunk_tail_inline1939__ssa_v1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, n0_tail_inline1964__idx_v0], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                                gamma_kv_input_tail_inline1960__ssa_v0: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [n0_tail_inline1964__idx_v0], [64], [64], target_memory=pl.Mem.Vec)
                                gamma_kv_cast_tail_inline1905__ssa_v0: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_kv_input_tail_inline1960__ssa_v0, target_type=pl.FP32, mode='round')
                                gamma_kv_chunk_tail_inline1919__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(gamma_kv_cast_tail_inline1905__ssa_v0, [1, 64])
                                kv_normed_tail_inline1904__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_chunk_tail_inline1939__ssa_v1, kv_inv_rms_t_tail_inline1947__ssa_v0), gamma_kv_chunk_tail_inline1919__ssa_v0)
                                kv_normed_bf16_tail_inline1903__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.cast(kv_normed_tail_inline1904__ssa_v0, target_type=pl.BF16, mode='rint')
                                kv_normed_valid_inline1902__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1903__ssa_v0, valid_rows_inline1931__ssa_v0, 64)
                                pl.tile.store(kv_normed_valid_inline1902__ssa_v0, [out_tg_inline1924__ssa_v0, n0_tail_inline1964__idx_v0], kv_view_inline1909__ssa_v0)
                            gamma_rope_input_tail_inline1901__ssa_v0: pl.Tile[[64], pl.BF16, pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec)
                            gamma_rope_cast_tail_inline1900__ssa_v0: pl.Tile[[64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(gamma_rope_input_tail_inline1901__ssa_v0, target_type=pl.FP32, mode='round')
                            gamma_rope_tail_inline1899__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(gamma_rope_cast_tail_inline1900__ssa_v0, [1, 64])
                            kv_rope_chunk_tail_inline1907__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, 448], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                            kv_rope_norm_tail_inline1898__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.col_expand_mul(pl.tile.row_expand_mul(kv_rope_chunk_tail_inline1907__ssa_v0, kv_inv_rms_t_tail_inline1947__ssa_v0), gamma_rope_tail_inline1899__ssa_v0)
                            kv_cos_il_tail_inline1929__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_cos_il_inline1258__ssa_v0, [out_tg_inline1924__ssa_v0, 0], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                            kv_sin_signed_tail_inline1917__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_sin_signed_inline1301__ssa_v0, [out_tg_inline1924__ssa_v0, 0], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                            kv_col_inline1897__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([16, 64], dtype=pl.FP32, value=1.0), pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                            kv_dup_f_inline1952__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(pl.tile.cast(pl.tile.muls(kv_col_inline1897__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                            kv_lane_inline1896__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(kv_col_inline1897__ssa_v0, pl.tile.muls(kv_dup_f_inline1952__ssa_v0, 2.0))
                            kv_swap_f_inline1895__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(pl.tile.adds(kv_col_inline1897__ssa_v0, 1.0), pl.tile.muls(kv_lane_inline1896__ssa_v0, 2.0))
                            kv_row_seed_inline1966__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(pl.tile.cast(pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'), 64.0)
                            kv_row_grid_inline1893__ssa_v0: pl.Tile[[64, 16], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(pl.tile.full([64, 16], dtype=pl.FP32, value=1.0), kv_row_seed_inline1966__ssa_v0)
                            kv_row_offset_inline1892__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.transpose(kv_row_grid_inline1893__ssa_v0, 0, 1)
                            kv_swap_idx_tail_inline1891__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.cast(pl.tile.add(kv_swap_f_inline1895__ssa_v0, kv_row_offset_inline1892__ssa_v0), target_type=pl.INT32, mode='round')
                            kv_gather_tmp_inline1890__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                            kv_swapped_tail_inline1894__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.gather(kv_rope_norm_tail_inline1898__ssa_v0, kv_swap_idx_tail_inline1891__ssa_v0, kv_gather_tmp_inline1890__ssa_v0)
                            kv_rope_rot_tail_inline1927__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.add(pl.tile.mul(kv_rope_norm_tail_inline1898__ssa_v0, kv_cos_il_tail_inline1929__ssa_v0), pl.tile.mul(kv_swapped_tail_inline1894__ssa_v0, kv_sin_signed_tail_inline1917__ssa_v0))
                            kv_rope_i16_tail_inline1889__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.cast(kv_rope_rot_tail_inline1927__ssa_v0, target_type=pl.BF16, mode='rint')
                            kv_rope_valid_inline1906__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.set_validshape(kv_rope_i16_tail_inline1889__ssa_v0, valid_rows_inline1931__ssa_v0, 64)
                            pl.tile.store(kv_rope_valid_inline1906__ssa_v0, [out_tg_inline1924__ssa_v0, 448], kv_view_inline1909__ssa_v0)
                            kv_view_inline1909__phi_v5: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16] = pl.yield_(kv_view_inline1909__ssa_v0)
            ori_block_num_inline1291__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache__ssa_v0, 0)
            kv_cache_flat_inline1312__ssa_v0: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(kv_cache__ssa_v0, [ori_block_num_inline1291__ssa_v0 * 32, 512])
            with pl.spmd(kv_wb_blocks_inline1274__ssa_v0, name_hint="csa_cache_writeback_spmd") as ori_cache_write_tid_inline1279__ssa_v0:
                wb_blk_inline1315__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                wb_t0_inline1317__ssa_v0: pl.Scalar[pl.INDEX] = wb_blk_inline1315__ssa_v0 * 8
                for write_dt_inline1318__idx_v0, (kv_cache_flat_inline1312__iter_v1,) in pl.range(8, init_values=(kv_cache_flat_inline1312__ssa_v0,)):
                    write_t_inline1308__ssa_v0: pl.Scalar[pl.INDEX] = wb_t0_inline1317__ssa_v0 + write_dt_inline1318__idx_v0
                    write_row_i64_inline1319__ssa_v0: pl.Scalar[pl.INT64] = pl.tensor.read(ori_slot_mapping__ssa_v0, [write_t_inline1308__ssa_v0])
                    if write_row_i64_inline1319__ssa_v0 >= 0:
                        write_row_inline1268__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(write_row_i64_inline1319__ssa_v0, pl.INDEX)
                        kv_cache_flat_inline1312__ssa_v3: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(kv_cache_flat_inline1312__iter_v1, pl.tensor.slice(kv_full_inline1265__ssa_v0, [1, 512], [write_t_inline1308__ssa_v0, 0]), [write_row_inline1268__ssa_v0, 0])
                        kv_cache_flat_inline1312__phi_v4: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.yield_(kv_cache_flat_inline1312__ssa_v3)
                    else:
                        kv_cache_flat_inline1312__phi_v4: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.yield_(kv_cache_flat_inline1312__iter_v1)
                    kv_cache_flat_inline1312__rv_v2: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.yield_(kv_cache_flat_inline1312__phi_v4)
            cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32] = pl.tensor.reshape(position_ids__ssa_v0, [kv_dim_inline1261__ssa_v0])
            cmp_slots_inline1296__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64] = pl.tensor.reshape(cmp_slot_mapping__ssa_v0, [kv_dim_inline1261__ssa_v0])
            cmp_state_slots_inline1247__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64] = pl.tensor.reshape(state_slot_mapping__ssa_v0, [kv_dim_inline1261__ssa_v0])
            idx_slots_inline1322__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64] = pl.tensor.reshape(idx_slot_mapping__ssa_v0, [kv_dim_inline1261__ssa_v0])
            idx_positions_inline1323__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0], pl.INT32] = pl.tensor.reshape(position_ids_local__ssa_v0, [t_dim_inline1251__ssa_v0])
            inner_state_slots_inline1257__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64] = pl.tensor.reshape(inner_state_slot_mapping__ssa_v0, [kv_dim_inline1261__ssa_v0])
            cmp_state_table_inline1275__ssa_v0: pl.Tensor[[kv_b_dim_inline1264__ssa_v0, 4], pl.INT32] = pl.tensor.reshape(compress_state_block_table__ssa_v0, [kv_b_dim_inline1264__ssa_v0, 4])
            inner_state_table_inline1324__ssa_v0: pl.Tensor[[kv_b_dim_inline1264__ssa_v0, 4], pl.INT32] = pl.tensor.reshape(inner_compress_state_block_table__ssa_v0, [kv_b_dim_inline1264__ssa_v0, 4])
            cmp_out_inline1299__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            b_dim_inline2018__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_state_table_inline1275__ssa_v0, 0)
            bs_inline2038__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240__rv_v5, 0)
            s_dim_inline2020__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2038__ssa_v0 // b_dim_inline2018__ssa_v0
            t_matmul_inline2013__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2038__ssa_v0 + 16 - 1) // 16 * 16
            rms_blocks_inline2061__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2038__ssa_v0 + 16 - 1) // 16
            x_flat_inline2021__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16] = x_normed_full_inline1240__rv_v5
            cmp4_kv_proj_pad_inline2031__ssa_v0: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.create([512, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            cmp4_score_proj_pad_inline2019__ssa_v0: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.create([512, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            compress_state_block_num_inline2051__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state__ssa_v0, 0)
            cmp_block_num_inline2055__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            compress_state_flat_inline2023__ssa_v0: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.reshape(compress_state__ssa_v0, [compress_state_block_num_inline2051__ssa_v0 * 2, 2048])
            kv_flat_inline2039__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32] = cmp_out_inline1299__ssa_v0
            cmp_kv_cache_flat_inline2036__ssa_v0: pl.Tensor[[cmp_block_num_inline2055__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline2055__ssa_v0 * 32, 512])
            with pl.spmd(t_matmul_inline2013__ssa_v0 * 1024 // 1024, name_hint="kv_score_proj_spmd", deps=[late_dep_inline1297__ssa_v0]) as _kv_score_tid_inline2015__ssa_v0:
                idx_inline2012__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                global_row0_inline2011__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2012__ssa_v0 // 16 * 16
                o0_inline2017__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2012__ssa_v0 % 16 * 64
                kv_acc_inline2010__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                score_acc_inline2014__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for kb_inline2001__idx_v0, (kv_acc_inline2010__iter_v1, score_acc_inline2014__iter_v1) in pl.pipeline(8, stage=2, init_values=(kv_acc_inline2010__ssa_v0, score_acc_inline2014__ssa_v0)):
                    k0_inline2035__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2001__idx_v0 * 512
                    x_rows_inline2000__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2038__ssa_v0 - global_row0_inline2011__ssa_v0)
                    x_tile_inline2033__ssa_v0: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[x_rows_inline2000__ssa_v0, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline2021__ssa_v0, [16, 512], [global_row0_inline2011__ssa_v0, k0_inline2035__ssa_v0], [x_rows_inline2000__ssa_v0, 512])
                    wkv_tile_inline2007__ssa_v0: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(cmp_wkv__ssa_v0, [64, 512], [o0_inline2017__ssa_v0, k0_inline2035__ssa_v0])
                    wgate_tile_inline1998__ssa_v0: pl.Tensor[[64, 512], pl.BF16] = pl.tensor.slice(cmp_wgate__ssa_v0, [64, 512], [o0_inline2017__ssa_v0, k0_inline2035__ssa_v0])
                    if k0_inline2035__ssa_v0 == 0:
                        kv_acc_inline2010__ssa_v3: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile_inline2033__ssa_v0, wkv_tile_inline2007__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        score_acc_inline2014__ssa_v3: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile_inline2033__ssa_v0, wgate_tile_inline1998__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        kv_acc_inline2010__phi_v5, score_acc_inline2014__phi_v5 = pl.yield_(kv_acc_inline2010__ssa_v3, score_acc_inline2014__ssa_v3)
                    else:
                        kv_acc_inline2010__ssa_v4: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(kv_acc_inline2010__iter_v1, x_tile_inline2033__ssa_v0, wkv_tile_inline2007__ssa_v0, a_trans=False, b_trans=True)
                        score_acc_inline2014__ssa_v4: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(score_acc_inline2014__iter_v1, x_tile_inline2033__ssa_v0, wgate_tile_inline1998__ssa_v0, a_trans=False, b_trans=True)
                        kv_acc_inline2010__phi_v5, score_acc_inline2014__phi_v5 = pl.yield_(kv_acc_inline2010__ssa_v4, score_acc_inline2014__ssa_v4)
                    kv_acc_inline2010__rv_v2, score_acc_inline2014__rv_v2 = pl.yield_(kv_acc_inline2010__phi_v5, score_acc_inline2014__phi_v5)
                cmp4_kv_proj_pad_inline2031__ssa_v1: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.assemble(cmp4_kv_proj_pad_inline2031__ssa_v0, kv_acc_inline2010__rv_v2, [global_row0_inline2011__ssa_v0, o0_inline2017__ssa_v0])
                cmp4_score_proj_pad_inline2019__ssa_v1: pl.Tensor[[512, 1024], pl.FP32] = pl.tensor.assemble(cmp4_score_proj_pad_inline2019__ssa_v0, score_acc_inline2014__rv_v2, [global_row0_inline2011__ssa_v0, o0_inline2017__ssa_v0])
            pooled_kv_inline2008__ssa_v0: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.create([512, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(b_dim_inline2018__ssa_v0, name_hint="scatter_softmax_pool_spmd", deps=[_kv_score_tid_inline2015__ssa_v0]) as pool_tid_inline1994__ssa_v0:
                c_idx_inline2049__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                first_pos_b_inline2028__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [c_idx_inline2049__ssa_v0 * s_dim_inline2020__ssa_v0])
                for s_idx_inline2026__idx_v0, (pooled_kv_inline2008__iter_v1,) in pl.range(s_dim_inline2020__ssa_v0, init_values=(pooled_kv_inline2008__ssa_v0,)):
                    token_inline2040__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2049__ssa_v0 * s_dim_inline2020__ssa_v0 + s_idx_inline2026__idx_v0
                    token_pos_inline2041__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [token_inline2040__ssa_v0])
                    pooled_kv_inline2008__ssa_v3: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(pooled_kv_inline2008__iter_v1, pl.tensor.full([1, 512], dtype=pl.FP32, value=0.0), [token_inline2040__ssa_v0, 0])
                    if (pl.cast(token_pos_inline2041__ssa_v0, pl.INDEX) + 1) % 4 == 0:
                        window_start_inline1995__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline2041__ssa_v0, pl.INDEX) - 8 + 1
                        for h0_inline2042__idx_v0, (pooled_kv_inline2008__iter_v4,) in pl.range(0, 512, 64, init_values=(pooled_kv_inline2008__ssa_v3,)):
                            last_ape_row_inline2044__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2041__ssa_v0, pl.INDEX) % 4, pl.INDEX)
                            mi_inline2045__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad_inline2019__ssa_v1, [1, 64], [token_inline2040__ssa_v0, 512 + h0_inline2042__idx_v0]), pl.tensor.slice(cmp_ape__ssa_v0, [1, 64], [last_ape_row_inline2044__ssa_v0, 512 + h0_inline2042__idx_v0]))
                            li_inline2029__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline2045__ssa_v0, mi_inline2045__ssa_v0))
                            oi_inline2046__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cmp4_kv_proj_pad_inline2031__ssa_v1, [1, 64], [token_inline2040__ssa_v0, 512 + h0_inline2042__idx_v0])
                            for state_idx_inline2032__idx_v0, (li_inline2029__iter_v1, mi_inline2045__iter_v1, oi_inline2046__iter_v1) in pl.range(7, init_values=(li_inline2029__ssa_v0, mi_inline2045__ssa_v0, oi_inline2046__ssa_v0)):
                                logical_pos_inline2022__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline1995__ssa_v0 + state_idx_inline2032__idx_v0
                                value_inline2050__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0)
                                score_inline2003__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                                state_half_inline2052__ssa_v0: pl.Scalar[pl.INDEX] = 0
                                if state_idx_inline2032__idx_v0 >= 4:
                                    state_half_inline2052__ssa_v1: pl.Scalar[pl.INDEX] = 512
                                    state_half_inline2052__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline2052__ssa_v1)
                                else:
                                    state_half_inline2052__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline2052__ssa_v0)
                                if logical_pos_inline2022__ssa_v0 >= 0 and logical_pos_inline2022__ssa_v0 < pl.cast(first_pos_b_inline2028__ssa_v0, pl.INDEX):
                                    ring_row_inline2004__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline2022__ssa_v0 % 8
                                    state_page_off_inline2054__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline2004__ssa_v0 // 2
                                    state_blk_id_i32_inline1996__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_state_table_inline1275__ssa_v0, [c_idx_inline2049__ssa_v0, state_page_off_inline2054__ssa_v0])
                                    if pl.cast(state_blk_id_i32_inline1996__ssa_v0, pl.INDEX) >= 0:
                                        state_blk_id_inline2056__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline1996__ssa_v0, pl.INDEX)
                                        state_row_inline2058__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline2056__ssa_v0 * 2 + ring_row_inline2004__ssa_v0 % 2
                                        value_inline2050__ssa_v1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat_inline2023__ssa_v0, [1, 64], [state_row_inline2058__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0])
                                        score_inline2003__ssa_v1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat_inline2023__ssa_v0, [1, 64], [state_row_inline2058__ssa_v0, 1024 + state_half_inline2052__phi_v2 + h0_inline2042__idx_v0])
                                        score_inline2003__phi_v2, value_inline2050__phi_v2 = pl.yield_(score_inline2003__ssa_v1, value_inline2050__ssa_v1)
                                    else:
                                        score_inline2003__phi_v2, value_inline2050__phi_v2 = pl.yield_(score_inline2003__ssa_v0, value_inline2050__ssa_v0)
                                    score_inline2003__phi_v3, value_inline2050__phi_v3 = pl.yield_(score_inline2003__phi_v2, value_inline2050__phi_v2)
                                else:
                                    score_inline2003__phi_v3, value_inline2050__phi_v3 = pl.yield_(score_inline2003__ssa_v0, value_inline2050__ssa_v0)
                                if logical_pos_inline2022__ssa_v0 >= pl.cast(first_pos_b_inline2028__ssa_v0, pl.INDEX):
                                    if logical_pos_inline2022__ssa_v0 <= pl.cast(token_pos_inline2041__ssa_v0, pl.INDEX):
                                        overlay_token_inline2005__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2049__ssa_v0 * s_dim_inline2020__ssa_v0 + logical_pos_inline2022__ssa_v0 - pl.cast(first_pos_b_inline2028__ssa_v0, pl.INDEX)
                                        ape_row_inline2034__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline2022__ssa_v0 % 4, pl.INDEX)
                                        value_inline2050__ssa_v4: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cmp4_kv_proj_pad_inline2031__ssa_v1, [1, 64], [overlay_token_inline2005__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0])
                                        score_inline2003__ssa_v4: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad_inline2019__ssa_v1, [1, 64], [overlay_token_inline2005__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0]), pl.tensor.slice(cmp_ape__ssa_v0, [1, 64], [ape_row_inline2034__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0]))
                                        score_inline2003__phi_v5, value_inline2050__phi_v5 = pl.yield_(score_inline2003__ssa_v4, value_inline2050__ssa_v4)
                                    else:
                                        score_inline2003__phi_v5, value_inline2050__phi_v5 = pl.yield_(score_inline2003__phi_v3, value_inline2050__phi_v3)
                                    score_inline2003__phi_v6, value_inline2050__phi_v6 = pl.yield_(score_inline2003__phi_v5, value_inline2050__phi_v5)
                                else:
                                    score_inline2003__phi_v6, value_inline2050__phi_v6 = pl.yield_(score_inline2003__phi_v3, value_inline2050__phi_v3)
                                mi_next_inline2059__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(mi_inline2045__iter_v1, score_inline2003__phi_v6)
                                alpha_inline2027__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline2045__iter_v1, mi_next_inline2059__ssa_v0))
                                beta_inline1999__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(score_inline2003__phi_v6, mi_next_inline2059__ssa_v0))
                                li_inline2029__ssa_v3: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha_inline2027__ssa_v0, li_inline2029__iter_v1), beta_inline1999__ssa_v0)
                                oi_inline2046__ssa_v3: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(oi_inline2046__iter_v1, alpha_inline2027__ssa_v0), pl.tensor.mul(value_inline2050__phi_v6, beta_inline1999__ssa_v0))
                                mi_inline2045__ssa_v3: pl.Tensor[[1, 64], pl.FP32] = mi_next_inline2059__ssa_v0
                                li_inline2029__rv_v2, mi_inline2045__rv_v2, oi_inline2046__rv_v2 = pl.yield_(li_inline2029__ssa_v3, mi_inline2045__ssa_v3, oi_inline2046__ssa_v3)
                            pooled_kv_inline2008__ssa_v6: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(pooled_kv_inline2008__iter_v4, pl.tensor.div(oi_inline2046__rv_v2, li_inline2029__rv_v2), [token_inline2040__ssa_v0, h0_inline2042__idx_v0])
                            pooled_kv_inline2008__rv_v5: pl.Tensor[[512, 512], pl.FP32] = pl.yield_(pooled_kv_inline2008__ssa_v6)
                        pooled_kv_inline2008__phi_v7: pl.Tensor[[512, 512], pl.FP32] = pl.yield_(pooled_kv_inline2008__rv_v5)
                    else:
                        pooled_kv_inline2008__phi_v7: pl.Tensor[[512, 512], pl.FP32] = pl.yield_(pooled_kv_inline2008__ssa_v3)
                    pooled_kv_inline2008__rv_v2: pl.Tensor[[512, 512], pl.FP32] = pl.yield_(pooled_kv_inline2008__phi_v7)
            for c_idx_v1_inline2030__ssa_v0 in pl.spmd(b_dim_inline2018__ssa_v0, name_hint="compress_state_commit_spmd", deps=[pool_tid_inline1994__ssa_v0]):
                for s_idx_inline2009__idx_v0, (compress_state_flat_inline2023__iter_v1,) in pl.range(s_dim_inline2020__ssa_v0, init_values=(compress_state_flat_inline2023__ssa_v0,)):
                    token_inline2040__ssa_v1: pl.Scalar[pl.INDEX] = c_idx_v1_inline2030__ssa_v0 * s_dim_inline2020__ssa_v0 + s_idx_inline2009__idx_v0
                    state_row_i64_inline2006__ssa_v0: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_state_slots_inline1247__ssa_v0, [token_inline2040__ssa_v1])
                    if state_row_i64_inline2006__ssa_v0 >= 0:
                        state_row_inline2058__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline2006__ssa_v0, pl.INDEX)
                        token_pos_inline2041__ssa_v1: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [token_inline2040__ssa_v1])
                        ape_row_inline2034__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2041__ssa_v1, pl.INDEX) % 4, pl.INDEX)
                        compress_state_flat_inline2023__ssa_v3: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.assemble(compress_state_flat_inline2023__iter_v1, pl.tensor.slice(cmp4_kv_proj_pad_inline2031__ssa_v1, [1, 1024], [token_inline2040__ssa_v1, 0]), [state_row_inline2058__ssa_v1, 0])
                        compress_state_flat_inline2023__ssa_v4: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.tensor.assemble(compress_state_flat_inline2023__ssa_v3, pl.tensor.add(pl.tensor.slice(cmp4_score_proj_pad_inline2019__ssa_v1, [1, 1024], [token_inline2040__ssa_v1, 0]), pl.tensor.slice(cmp_ape__ssa_v0, [1, 1024], [ape_row_inline2034__ssa_v1, 0])), [state_row_inline2058__ssa_v1, 1024])
                        compress_state_flat_inline2023__phi_v5: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.yield_(compress_state_flat_inline2023__ssa_v4)
                    else:
                        compress_state_flat_inline2023__phi_v5: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.yield_(compress_state_flat_inline2023__iter_v1)
                    compress_state_flat_inline2023__rv_v2: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32] = pl.yield_(compress_state_flat_inline2023__phi_v5)
            normed_kv_inline2016__ssa_v0: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.create([512, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            norm_w_2d_inline2060__ssa_v0: pl.Tensor[[1, 512], pl.BF16] = pl.tensor.reshape(cmp_norm_w__ssa_v0, [1, 512])
            with pl.spmd(rms_blocks_inline2061__ssa_v0, name_hint="rmsnorm_rope_cache_write_spmd", deps=[pool_tid_inline1994__ssa_v0]) as cache_write_tid_inline2062__ssa_v0:
                rms_blk_inline1993__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                b0_inline1992__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline1993__ssa_v0 * 16
                rms_blk_rows_inline1991__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2038__ssa_v0 - b0_inline1992__ssa_v0)
                cos_b_inline2057__ssa_v0: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows_inline1991__ssa_v0, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cmp_cos_il_full_inline1249__rv_v2, [16, 64], [b0_inline1992__ssa_v0, 0], [rms_blk_rows_inline1991__ssa_v0, 64])
                sin_b_inline1989__ssa_v0: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows_inline1991__ssa_v0, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cmp_sin_signed_full_inline1263__rv_v2, [16, 64], [b0_inline1992__ssa_v0, 0], [rms_blk_rows_inline1991__ssa_v0, 64])
                partial_sq_inline2025__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                for k0_inline1987__idx_v0, (partial_sq_inline2025__iter_v1,) in pl.range(0, 512, 64, init_values=(partial_sq_inline2025__ssa_v0,)):
                    kv_rms_chunk_inline1986__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2008__rv_v2, [16, 64], [b0_inline1992__ssa_v0, k0_inline1987__idx_v0])
                    kv_rms_sq_inline1988__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_rms_chunk_inline1986__ssa_v0, kv_rms_chunk_inline1986__ssa_v0)
                    kv_rms_rowsum_inline2053__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_rms_sq_inline1988__ssa_v0), [1, 16])
                    partial_sq_inline2025__ssa_v3: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq_inline2025__iter_v1, kv_rms_rowsum_inline2053__ssa_v0)
                    partial_sq_inline2025__rv_v2: pl.Tensor[[1, 16], pl.FP32] = pl.yield_(partial_sq_inline2025__ssa_v3)
                variance_inline1990__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq_inline2025__rv_v2, 0.001953125), 9.9999999999999995e-07), [16, 1])
                inv_rms_inline1984__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance_inline1990__ssa_v0))
                for k0_inline1983__idx_v0, (normed_kv_inline2016__iter_v1,) in pl.range(0, 448, 64, init_values=(normed_kv_inline2016__ssa_v0,)):
                    kv_norm_chunk_inline2037__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2008__rv_v2, [16, 64], [b0_inline1992__ssa_v0, k0_inline1983__idx_v0])
                    gamma_inline2024__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d_inline2060__ssa_v0, [1, 64], [0, k0_inline1983__idx_v0]), target_type=pl.FP32, mode='round')
                    normed_chunk_inline1982__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_norm_chunk_inline2037__ssa_v0, inv_rms_inline1984__ssa_v0), gamma_inline2024__ssa_v0)
                    normed_kv_inline2016__ssa_v3: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(normed_kv_inline2016__iter_v1, normed_chunk_inline1982__ssa_v0, [b0_inline1992__ssa_v0, k0_inline1983__idx_v0])
                    normed_kv_inline2016__rv_v2: pl.Tensor[[512, 512], pl.FP32] = pl.yield_(normed_kv_inline2016__ssa_v3)
                kv_rope_norm_inline1981__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2008__rv_v2, [16, 64], [b0_inline1992__ssa_v0, 448])
                gamma_rope_inline1980__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d_inline2060__ssa_v0, [1, 64], [0, 448]), target_type=pl.FP32, mode='round')
                rope_normed_inline1979__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_norm_inline1981__ssa_v0, inv_rms_inline1984__ssa_v0), gamma_rope_inline1980__ssa_v0)
                rope_ones_inline1978__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
                rope_col_inline1977__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(rope_ones_inline1978__ssa_v0, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                rope_dup_f_inline1976__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(rope_col_inline1977__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                rope_lane_inline2048__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(rope_col_inline1977__ssa_v0, pl.tensor.muls(rope_dup_f_inline1976__ssa_v0, 2.0))
                rope_swap_idx_inline1997__ssa_v0: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(rope_col_inline1977__ssa_v0, 1.0), pl.tensor.muls(rope_lane_inline2048__ssa_v0, 2.0)), target_type=pl.INT32, mode='round')
                swapped_inline1975__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(rope_normed_inline1979__ssa_v0, rope_swap_idx_inline1997__ssa_v0, dim=-1)
                rope_rot_inline2002__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(rope_normed_inline1979__ssa_v0, cos_b_inline2057__ssa_v0), pl.tensor.mul(swapped_inline1975__ssa_v0, sin_b_inline1989__ssa_v0))
                normed_kv_inline2016__ssa_v4: pl.Tensor[[512, 512], pl.FP32] = pl.tensor.assemble(normed_kv_inline2016__rv_v2, rope_rot_inline2002__ssa_v0, [b0_inline1992__ssa_v0, 448])
                for inner_inline1985__idx_v0, (cmp_kv_cache_flat_inline2036__iter_v1, kv_flat_inline2039__iter_v1) in pl.range(rms_blk_rows_inline1991__ssa_v0, init_values=(cmp_kv_cache_flat_inline2036__ssa_v0, kv_flat_inline2039__ssa_v0)):
                    token_inline2040__ssa_v2: pl.Scalar[pl.INDEX] = b0_inline1992__ssa_v0 + inner_inline1985__idx_v0
                    cache_row_i64_inline1974__ssa_v0: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_slots_inline1296__ssa_v0, [token_inline2040__ssa_v2])
                    if cache_row_i64_inline1974__ssa_v0 >= 0:
                        cache_row_inline2047__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline1974__ssa_v0, pl.INDEX)
                        kv_row_fp32_inline2043__ssa_v0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.slice(normed_kv_inline2016__ssa_v4, [1, 512], [token_inline2040__ssa_v2, 0])
                        kv_flat_inline2039__ssa_v3: pl.Tensor[[KV_T_DYN, 512], pl.FP32] = pl.tensor.assemble(kv_flat_inline2039__iter_v1, kv_row_fp32_inline2043__ssa_v0, [token_inline2040__ssa_v2, 0])
                        cmp_kv_cache_flat_inline2036__ssa_v3: pl.Tensor[[cmp_block_num_inline2055__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(cmp_kv_cache_flat_inline2036__iter_v1, pl.tensor.cast(kv_row_fp32_inline2043__ssa_v0, target_type=pl.BF16, mode='rint'), [cache_row_inline2047__ssa_v0, 0])
                        cmp_kv_cache_flat_inline2036__phi_v4, kv_flat_inline2039__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline2036__ssa_v3, kv_flat_inline2039__ssa_v3)
                    else:
                        cmp_kv_cache_flat_inline2036__phi_v4, kv_flat_inline2039__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline2036__iter_v1, kv_flat_inline2039__iter_v1)
                    cmp_kv_cache_flat_inline2036__rv_v2, kv_flat_inline2039__rv_v2 = pl.yield_(cmp_kv_cache_flat_inline2036__phi_v4, kv_flat_inline2039__phi_v4)
            cmp_out_inline1299__ssa_v1: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32] = cmp_out_inline1299__ssa_v0
            cmp_cache_write_tid_inline1237__ssa_v0: pl.Scalar[pl.TASK_ID] = cache_write_tid_inline2062__ssa_v0
            cache_ready_dep_inline1304__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[ori_cache_write_tid_inline1279__ssa_v0, cmp_cache_write_tid_inline1237__ssa_v0])
            idx_kv_unused_inline1241__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 128], pl.FP32] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            b_dim_inline2158__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_state_table_inline1324__ssa_v0, 0)
            bs_inline2140__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240__rv_v5, 0)
            s_dim_inline2127__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2140__ssa_v0 // b_dim_inline2158__ssa_v0
            t_matmul_inline2119__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2140__ssa_v0 + 16 - 1) // 16 * 16
            rms_blocks_inline2124__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2140__ssa_v0 + 16 - 1) // 16
            x_flat_inline2162__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16] = x_normed_full_inline1240__rv_v5
            kv_proj_pad_inline2129__ssa_v0: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.create([512, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            score_proj_pad_inline2143__ssa_v0: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.create([512, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            compress_state_block_num_inline2109__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_compress_state__ssa_v0, 0)
            idx_block_num_inline2101__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_kv_cache__ssa_v0, 0)
            compress_state_flat_inline2139__ssa_v0: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.reshape(inner_compress_state__ssa_v0, [compress_state_block_num_inline2109__ssa_v0 * 2, 512])
            kv_flat_inline2098__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 128], pl.FP32] = idx_kv_unused_inline1241__ssa_v0
            idx_kv_cache_flat_inline2161__ssa_v0: pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.reshape(idx_kv_cache__ssa_v0, [idx_block_num_inline2101__ssa_v0 * 32, 128])
            idx_kv_scale_flat_inline2116__ssa_v0: pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32] = pl.tensor.reshape(idx_kv_scale__ssa_v0, [idx_block_num_inline2101__ssa_v0 * 32, 1])
            with pl.spmd(t_matmul_inline2119__ssa_v0 * 256 // 512, name_hint="kv_score_proj_spmd", deps=[late_dep_inline1297__ssa_v0]) as _kv_score_tid_inline2110__ssa_v0:
                idx_inline2108__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                global_row0_inline2104__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2108__ssa_v0 // 8 * 16
                o0_inline2117__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2108__ssa_v0 % 8 * 32
                kv_acc_inline2107__ssa_v0: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                score_acc_inline2103__ssa_v0: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.create([16, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for kb_inline2105__idx_v0, (kv_acc_inline2107__iter_v1, score_acc_inline2103__iter_v1) in pl.pipeline(8, stage=2, init_values=(kv_acc_inline2107__ssa_v0, score_acc_inline2103__ssa_v0)):
                    k0_inline2172__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2105__idx_v0 * 512
                    x_rows_inline2097__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2140__ssa_v0 - global_row0_inline2104__ssa_v0)
                    x_tile_inline2096__ssa_v0: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[x_rows_inline2097__ssa_v0, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline2162__ssa_v0, [16, 512], [global_row0_inline2104__ssa_v0, k0_inline2172__ssa_v0], [x_rows_inline2097__ssa_v0, 512])
                    wkv_tile_inline2114__ssa_v0: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(inner_wkv__ssa_v0, [32, 512], [o0_inline2117__ssa_v0, k0_inline2172__ssa_v0])
                    wgate_tile_inline2100__ssa_v0: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(inner_wgate__ssa_v0, [32, 512], [o0_inline2117__ssa_v0, k0_inline2172__ssa_v0])
                    if k0_inline2172__ssa_v0 == 0:
                        kv_acc_inline2107__ssa_v3: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_tile_inline2096__ssa_v0, wkv_tile_inline2114__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        score_acc_inline2103__ssa_v3: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul(x_tile_inline2096__ssa_v0, wgate_tile_inline2100__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        kv_acc_inline2107__phi_v5, score_acc_inline2103__phi_v5 = pl.yield_(kv_acc_inline2107__ssa_v3, score_acc_inline2103__ssa_v3)
                    else:
                        kv_acc_inline2107__ssa_v4: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(kv_acc_inline2107__iter_v1, x_tile_inline2096__ssa_v0, wkv_tile_inline2114__ssa_v0, a_trans=False, b_trans=True)
                        score_acc_inline2103__ssa_v4: pl.Tensor[[16, 32], pl.FP32] = pl.tensor.matmul_acc(score_acc_inline2103__iter_v1, x_tile_inline2096__ssa_v0, wgate_tile_inline2100__ssa_v0, a_trans=False, b_trans=True)
                        kv_acc_inline2107__phi_v5, score_acc_inline2103__phi_v5 = pl.yield_(kv_acc_inline2107__ssa_v4, score_acc_inline2103__ssa_v4)
                    kv_acc_inline2107__rv_v2, score_acc_inline2103__rv_v2 = pl.yield_(kv_acc_inline2107__phi_v5, score_acc_inline2103__phi_v5)
                kv_proj_pad_inline2129__ssa_v1: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.assemble(kv_proj_pad_inline2129__ssa_v0, kv_acc_inline2107__rv_v2, [global_row0_inline2104__ssa_v0, o0_inline2117__ssa_v0])
                score_proj_pad_inline2143__ssa_v1: pl.Tensor[[512, 256], pl.FP32] = pl.tensor.assemble(score_proj_pad_inline2143__ssa_v0, score_acc_inline2103__rv_v2, [global_row0_inline2104__ssa_v0, o0_inline2117__ssa_v0])
            pooled_kv_inline2131__ssa_v0: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.create([512, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(b_dim_inline2158__ssa_v0, name_hint="scatter_softmax_pool_spmd", deps=[_kv_score_tid_inline2110__ssa_v0]) as pool_tid_inline2128__ssa_v0:
                c_idx_inline2094__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                first_pos_b_inline2144__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [c_idx_inline2094__ssa_v0 * s_dim_inline2127__ssa_v0])
                for s_idx_inline2147__idx_v0, (pooled_kv_inline2131__iter_v1,) in pl.range(s_dim_inline2127__ssa_v0, init_values=(pooled_kv_inline2131__ssa_v0,)):
                    token_inline2123__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2094__ssa_v0 * s_dim_inline2127__ssa_v0 + s_idx_inline2147__idx_v0
                    token_pos_inline2120__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [token_inline2123__ssa_v0])
                    pooled_kv_inline2131__ssa_v3: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(pooled_kv_inline2131__iter_v1, pl.tensor.full([1, 128], dtype=pl.FP32, value=0.0), [token_inline2123__ssa_v0, 0])
                    if (pl.cast(token_pos_inline2120__ssa_v0, pl.INDEX) + 1) % 4 == 0:
                        window_start_inline2142__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline2120__ssa_v0, pl.INDEX) - 8 + 1
                        for h0_inline2125__idx_v0, (pooled_kv_inline2131__iter_v4,) in pl.range(0, 128, 64, init_values=(pooled_kv_inline2131__ssa_v3,)):
                            last_ape_row_inline2113__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2120__ssa_v0, pl.INDEX) % 4, pl.INDEX)
                            mi_inline2145__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(score_proj_pad_inline2143__ssa_v1, [1, 64], [token_inline2123__ssa_v0, 128 + h0_inline2125__idx_v0]), pl.tensor.slice(inner_ape__ssa_v0, [1, 64], [last_ape_row_inline2113__ssa_v0, 128 + h0_inline2125__idx_v0]))
                            li_inline2148__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline2145__ssa_v0, mi_inline2145__ssa_v0))
                            oi_inline2132__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(kv_proj_pad_inline2129__ssa_v1, [1, 64], [token_inline2123__ssa_v0, 128 + h0_inline2125__idx_v0])
                            for state_idx_inline2149__idx_v0, (li_inline2148__iter_v1, mi_inline2145__iter_v1, oi_inline2132__iter_v1) in pl.range(7, init_values=(li_inline2148__ssa_v0, mi_inline2145__ssa_v0, oi_inline2132__ssa_v0)):
                                logical_pos_inline2150__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline2142__ssa_v0 + state_idx_inline2149__idx_v0
                                value_inline2163__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0)
                                score_inline2156__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                                state_half_inline2160__ssa_v0: pl.Scalar[pl.INDEX] = 0
                                if state_idx_inline2149__idx_v0 >= 4:
                                    state_half_inline2160__ssa_v1: pl.Scalar[pl.INDEX] = 128
                                    state_half_inline2160__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline2160__ssa_v1)
                                else:
                                    state_half_inline2160__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline2160__ssa_v0)
                                if logical_pos_inline2150__ssa_v0 >= 0 and logical_pos_inline2150__ssa_v0 < pl.cast(first_pos_b_inline2144__ssa_v0, pl.INDEX):
                                    ring_row_inline2159__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline2150__ssa_v0 % 8
                                    state_page_off_inline2154__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline2159__ssa_v0 // 2
                                    state_blk_id_i32_inline2130__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(inner_state_table_inline1324__ssa_v0, [c_idx_inline2094__ssa_v0, state_page_off_inline2154__ssa_v0])
                                    if pl.cast(state_blk_id_i32_inline2130__ssa_v0, pl.INDEX) >= 0:
                                        state_blk_id_inline2168__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline2130__ssa_v0, pl.INDEX)
                                        state_row_inline2166__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline2168__ssa_v0 * 2 + ring_row_inline2159__ssa_v0 % 2
                                        value_inline2163__ssa_v1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat_inline2139__ssa_v0, [1, 64], [state_row_inline2166__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0])
                                        score_inline2156__ssa_v1: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(compress_state_flat_inline2139__ssa_v0, [1, 64], [state_row_inline2166__ssa_v0, 256 + state_half_inline2160__phi_v2 + h0_inline2125__idx_v0])
                                        score_inline2156__phi_v2, value_inline2163__phi_v2 = pl.yield_(score_inline2156__ssa_v1, value_inline2163__ssa_v1)
                                    else:
                                        score_inline2156__phi_v2, value_inline2163__phi_v2 = pl.yield_(score_inline2156__ssa_v0, value_inline2163__ssa_v0)
                                    score_inline2156__phi_v3, value_inline2163__phi_v3 = pl.yield_(score_inline2156__phi_v2, value_inline2163__phi_v2)
                                else:
                                    score_inline2156__phi_v3, value_inline2163__phi_v3 = pl.yield_(score_inline2156__ssa_v0, value_inline2163__ssa_v0)
                                if logical_pos_inline2150__ssa_v0 >= pl.cast(first_pos_b_inline2144__ssa_v0, pl.INDEX):
                                    if logical_pos_inline2150__ssa_v0 <= pl.cast(token_pos_inline2120__ssa_v0, pl.INDEX):
                                        overlay_token_inline2169__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2094__ssa_v0 * s_dim_inline2127__ssa_v0 + logical_pos_inline2150__ssa_v0 - pl.cast(first_pos_b_inline2144__ssa_v0, pl.INDEX)
                                        ape_row_inline2136__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline2150__ssa_v0 % 4, pl.INDEX)
                                        value_inline2163__ssa_v4: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(kv_proj_pad_inline2129__ssa_v1, [1, 64], [overlay_token_inline2169__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0])
                                        score_inline2156__ssa_v4: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(score_proj_pad_inline2143__ssa_v1, [1, 64], [overlay_token_inline2169__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0]), pl.tensor.slice(inner_ape__ssa_v0, [1, 64], [ape_row_inline2136__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0]))
                                        score_inline2156__phi_v5, value_inline2163__phi_v5 = pl.yield_(score_inline2156__ssa_v4, value_inline2163__ssa_v4)
                                    else:
                                        score_inline2156__phi_v5, value_inline2163__phi_v5 = pl.yield_(score_inline2156__phi_v3, value_inline2163__phi_v3)
                                    score_inline2156__phi_v6, value_inline2163__phi_v6 = pl.yield_(score_inline2156__phi_v5, value_inline2163__phi_v5)
                                else:
                                    score_inline2156__phi_v6, value_inline2163__phi_v6 = pl.yield_(score_inline2156__phi_v3, value_inline2163__phi_v3)
                                mi_next_inline2126__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(mi_inline2145__iter_v1, score_inline2156__phi_v6)
                                alpha_inline2115__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_inline2145__iter_v1, mi_next_inline2126__ssa_v0))
                                beta_inline2170__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.exp(pl.tensor.sub(score_inline2156__phi_v6, mi_next_inline2126__ssa_v0))
                                li_inline2148__ssa_v3: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha_inline2115__ssa_v0, li_inline2148__iter_v1), beta_inline2170__ssa_v0)
                                oi_inline2132__ssa_v3: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(oi_inline2132__iter_v1, alpha_inline2115__ssa_v0), pl.tensor.mul(value_inline2163__phi_v6, beta_inline2170__ssa_v0))
                                mi_inline2145__ssa_v3: pl.Tensor[[1, 64], pl.FP32] = mi_next_inline2126__ssa_v0
                                li_inline2148__rv_v2, mi_inline2145__rv_v2, oi_inline2132__rv_v2 = pl.yield_(li_inline2148__ssa_v3, mi_inline2145__ssa_v3, oi_inline2132__ssa_v3)
                            pooled_kv_inline2131__ssa_v6: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(pooled_kv_inline2131__iter_v4, pl.tensor.div(oi_inline2132__rv_v2, li_inline2148__rv_v2), [token_inline2123__ssa_v0, h0_inline2125__idx_v0])
                            pooled_kv_inline2131__rv_v5: pl.Tensor[[512, 128], pl.FP32] = pl.yield_(pooled_kv_inline2131__ssa_v6)
                        pooled_kv_inline2131__phi_v7: pl.Tensor[[512, 128], pl.FP32] = pl.yield_(pooled_kv_inline2131__rv_v5)
                    else:
                        pooled_kv_inline2131__phi_v7: pl.Tensor[[512, 128], pl.FP32] = pl.yield_(pooled_kv_inline2131__ssa_v3)
                    pooled_kv_inline2131__rv_v2: pl.Tensor[[512, 128], pl.FP32] = pl.yield_(pooled_kv_inline2131__phi_v7)
            for c_idx_v1_inline2153__ssa_v0 in pl.spmd(b_dim_inline2158__ssa_v0, name_hint="compress_state_commit_spmd", deps=[pool_tid_inline2128__ssa_v0]):
                for s_idx_inline2121__idx_v0, (compress_state_flat_inline2139__iter_v1,) in pl.range(s_dim_inline2127__ssa_v0, init_values=(compress_state_flat_inline2139__ssa_v0,)):
                    token_inline2123__ssa_v1: pl.Scalar[pl.INDEX] = c_idx_v1_inline2153__ssa_v0 * s_dim_inline2127__ssa_v0 + s_idx_inline2121__idx_v0
                    state_row_i64_inline2137__ssa_v0: pl.Scalar[pl.INT64] = pl.tensor.read(inner_state_slots_inline1257__ssa_v0, [token_inline2123__ssa_v1])
                    if state_row_i64_inline2137__ssa_v0 >= 0:
                        state_row_inline2166__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline2137__ssa_v0, pl.INDEX)
                        token_pos_inline2120__ssa_v1: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [token_inline2123__ssa_v1])
                        ape_row_inline2136__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2120__ssa_v1, pl.INDEX) % 4, pl.INDEX)
                        compress_state_flat_inline2139__ssa_v3: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.assemble(compress_state_flat_inline2139__iter_v1, pl.tensor.slice(kv_proj_pad_inline2129__ssa_v1, [1, 256], [token_inline2123__ssa_v1, 0]), [state_row_inline2166__ssa_v1, 0])
                        compress_state_flat_inline2139__ssa_v4: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.tensor.assemble(compress_state_flat_inline2139__ssa_v3, pl.tensor.add(pl.tensor.slice(score_proj_pad_inline2143__ssa_v1, [1, 256], [token_inline2123__ssa_v1, 0]), pl.tensor.slice(inner_ape__ssa_v0, [1, 256], [ape_row_inline2136__ssa_v1, 0])), [state_row_inline2166__ssa_v1, 256])
                        compress_state_flat_inline2139__phi_v5: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.yield_(compress_state_flat_inline2139__ssa_v4)
                    else:
                        compress_state_flat_inline2139__phi_v5: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.yield_(compress_state_flat_inline2139__iter_v1)
                    compress_state_flat_inline2139__rv_v2: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32] = pl.yield_(compress_state_flat_inline2139__phi_v5)
            normed_kv_inline2164__ssa_v0: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.create([512, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            norm_w_2d_inline2173__ssa_v0: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.reshape(inner_norm_w__ssa_v0, [1, 128])
            with pl.spmd(rms_blocks_inline2124__ssa_v0, name_hint="rmsnorm_rope_spmd", deps=[pool_tid_inline2128__ssa_v0]) as rms_tid_inline2093__ssa_v0:
                rms_blk_inline2092__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                b0_inline2090__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline2092__ssa_v0 * 16
                rms_blk_rows_inline2141__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2140__ssa_v0 - b0_inline2090__ssa_v0)
                cos_b_inline2087__ssa_v0: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows_inline2141__ssa_v0, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cmp_cos_il_full_inline1249__rv_v2, [16, 64], [b0_inline2090__ssa_v0, 0], [rms_blk_rows_inline2141__ssa_v0, 64])
                sin_b_inline2085__ssa_v0: pl.Tensor[[16, 64], pl.FP32, pl.TensorView(valid_shape=[rms_blk_rows_inline2141__ssa_v0, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(cmp_sin_signed_full_inline1263__rv_v2, [16, 64], [b0_inline2090__ssa_v0, 0], [rms_blk_rows_inline2141__ssa_v0, 64])
                partial_sq_inline2155__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0)
                for k0_inline2152__idx_v0, (partial_sq_inline2155__iter_v1,) in pl.pipeline(0, 128, 64, stage=2, init_values=(partial_sq_inline2155__ssa_v0,)):
                    kv_rms_chunk_inline2084__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2131__rv_v2, [16, 64], [b0_inline2090__ssa_v0, k0_inline2152__idx_v0])
                    kv_rms_sq_inline2083__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.mul(kv_rms_chunk_inline2084__ssa_v0, kv_rms_chunk_inline2084__ssa_v0)
                    kv_rms_rowsum_inline2165__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_sum(kv_rms_sq_inline2083__ssa_v0), [1, 16])
                    partial_sq_inline2155__ssa_v3: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(partial_sq_inline2155__iter_v1, kv_rms_rowsum_inline2165__ssa_v0)
                    partial_sq_inline2155__rv_v2: pl.Tensor[[1, 16], pl.FP32] = pl.yield_(partial_sq_inline2155__ssa_v3)
                variance_inline2082__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.adds(pl.tensor.muls(partial_sq_inline2155__rv_v2, 0.0078125), 9.9999999999999995e-07), [16, 1])
                inv_rms_inline2122__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.recip(pl.tensor.sqrt(variance_inline2082__ssa_v0))
                for k0_inline2081__idx_v0, (normed_kv_inline2164__iter_v1,) in pl.pipeline(0, 64, 64, stage=2, init_values=(normed_kv_inline2164__ssa_v0,)):
                    kv_norm_chunk_inline2080__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2131__rv_v2, [16, 64], [b0_inline2090__ssa_v0, k0_inline2081__idx_v0])
                    gamma_inline2078__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d_inline2173__ssa_v0, [1, 64], [0, k0_inline2081__idx_v0]), target_type=pl.FP32, mode='round')
                    normed_chunk_inline2102__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_norm_chunk_inline2080__ssa_v0, inv_rms_inline2122__ssa_v0), gamma_inline2078__ssa_v0)
                    normed_kv_inline2164__ssa_v3: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.assemble(normed_kv_inline2164__iter_v1, pl.tensor.cast(normed_chunk_inline2102__ssa_v0, target_type=pl.BF16, mode='rint'), [b0_inline2090__ssa_v0, k0_inline2081__idx_v0])
                    normed_kv_inline2164__rv_v2: pl.Tensor[[512, 128], pl.BF16] = pl.yield_(normed_kv_inline2164__ssa_v3)
                kv_rope_norm_inline2135__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(pooled_kv_inline2131__rv_v2, [16, 64], [b0_inline2090__ssa_v0, 64])
                gamma_rope_inline2077__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(norm_w_2d_inline2173__ssa_v0, [1, 64], [0, 64]), target_type=pl.FP32, mode='round')
                rope_normed_inline2138__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(kv_rope_norm_inline2135__ssa_v0, inv_rms_inline2122__ssa_v0), gamma_rope_inline2077__ssa_v0)
                rope_ones_inline2089__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
                rope_col_inline2076__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(rope_ones_inline2089__ssa_v0, pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                rope_dup_f_inline2074__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(rope_col_inline2076__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                rope_lane_inline2073__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(rope_col_inline2076__ssa_v0, pl.tensor.muls(rope_dup_f_inline2074__ssa_v0, 2.0))
                rope_swap_idx_inline2079__ssa_v0: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(rope_col_inline2076__ssa_v0, 1.0), pl.tensor.muls(rope_lane_inline2073__ssa_v0, 2.0)), target_type=pl.INT32, mode='round')
                swapped_inline2071__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(rope_normed_inline2138__ssa_v0, rope_swap_idx_inline2079__ssa_v0, dim=-1)
                rope_rot_inline2070__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.mul(rope_normed_inline2138__ssa_v0, cos_b_inline2087__ssa_v0), pl.tensor.mul(swapped_inline2071__ssa_v0, sin_b_inline2085__ssa_v0))
                normed_kv_inline2164__ssa_v4: pl.Tensor[[512, 128], pl.BF16] = pl.tensor.assemble(normed_kv_inline2164__rv_v2, pl.tensor.cast(rope_rot_inline2070__ssa_v0, target_type=pl.BF16, mode='rint'), [b0_inline2090__ssa_v0, 64])
            kv_final_inline2118__ssa_v0: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.create([512, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(rms_blocks_inline2124__ssa_v0, name_hint="kv_hadamard_spmd", deps=[rms_tid_inline2093__ssa_v0]) as hadamard_tid_inline2075__ssa_v0:
                had_blk_inline2088__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                had_b0_inline2069__ssa_v0: pl.Scalar[pl.INDEX] = had_blk_inline2088__ssa_v0 * 16
                kv_proj_tile_inline2072__ssa_v0: pl.Tensor[[16, 128], pl.BF16] = pl.tensor.slice(normed_kv_inline2164__ssa_v4, [16, 128], [had_b0_inline2069__ssa_v0, 0])
                for o0_inline2068__idx_v0, (kv_final_inline2118__iter_v1,) in pl.range(0, 128, 64, init_values=(kv_final_inline2118__ssa_v0,)):
                    hadamard_tile_inline2134__ssa_v0: pl.Tensor[[128, 64], pl.BF16] = pl.tensor.slice(hadamard_idx__ssa_v0, [128, 64], [0, o0_inline2068__idx_v0])
                    kv_hadamard_acc_inline2151__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(kv_proj_tile_inline2072__ssa_v0, hadamard_tile_inline2134__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    kv_final_inline2118__ssa_v3: pl.Tensor[[512, 128], pl.FP32] = pl.tensor.assemble(kv_final_inline2118__iter_v1, kv_hadamard_acc_inline2151__ssa_v0, [had_b0_inline2069__ssa_v0, o0_inline2068__idx_v0])
                    kv_final_inline2118__rv_v2: pl.Tensor[[512, 128], pl.FP32] = pl.yield_(kv_final_inline2118__ssa_v3)
            with pl.spmd(rms_blocks_inline2124__ssa_v0, name_hint="kv_and_cache_write_spmd", deps=[hadamard_tid_inline2075__ssa_v0]) as _write_tid_inline2106__ssa_v0:
                wr_blk_inline2171__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                wr_b0_inline2157__ssa_v0: pl.Scalar[pl.INDEX] = wr_blk_inline2171__ssa_v0 * 16
                wr_blk_rows_inline2167__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2140__ssa_v0 - wr_b0_inline2157__ssa_v0)
                kv_blk_f32_inline2067__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.slice(kv_final_inline2118__rv_v2, [16, 128], [wr_b0_inline2157__ssa_v0, 0]), target_type=pl.BF16, mode='rint'), target_type=pl.FP32, mode='round')
                kv_amax_inline2091__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(pl.tensor.abs(kv_blk_f32_inline2067__ssa_v0)), [1, 16])
                kv_amax_v1_inline2066__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.maximum(kv_amax_inline2091__ssa_v0, pl.tensor.full([1, 16], dtype=pl.FP32, value=0.0001))
                kv_scale_q_row_inline2112__ssa_v0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 16], dtype=pl.FP32, value=127.0), kv_amax_v1_inline2066__ssa_v0)
                kv_scale_dq_col_inline2065__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(kv_scale_q_row_inline2112__ssa_v0), [16, 1])
                kv_scale_q_col_inline2086__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(kv_scale_q_row_inline2112__ssa_v0, [16, 1])
                kv_scaled_inline2111__ssa_v0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.row_expand_mul(kv_blk_f32_inline2067__ssa_v0, kv_scale_q_col_inline2086__ssa_v0)
                kv_i32_inline2133__ssa_v0: pl.Tensor[[16, 128], pl.INT32] = pl.tensor.cast(kv_scaled_inline2111__ssa_v0, target_type=pl.INT32, mode='rint')
                kv_half_inline2064__ssa_v0: pl.Tensor[[16, 128], pl.FP16] = pl.tensor.cast(kv_i32_inline2133__ssa_v0, target_type=pl.FP16, mode='round')
                kv_i8_blk_inline2095__ssa_v0: pl.Tensor[[16, 128], pl.INT8] = pl.tensor.cast(kv_half_inline2064__ssa_v0, target_type=pl.INT8, mode='trunc')
                for inner_inline2063__idx_v0, (idx_kv_cache_flat_inline2161__iter_v1, kv_flat_inline2098__iter_v1) in pl.range(wr_blk_rows_inline2167__ssa_v0, init_values=(idx_kv_cache_flat_inline2161__ssa_v0, kv_flat_inline2098__ssa_v0)):
                    token_inline2123__ssa_v2: pl.Scalar[pl.INDEX] = wr_b0_inline2157__ssa_v0 + inner_inline2063__idx_v0
                    cache_row_i64_inline2099__ssa_v0: pl.Scalar[pl.INT64] = pl.tensor.read(idx_slots_inline1322__ssa_v0, [token_inline2123__ssa_v2])
                    if cache_row_i64_inline2099__ssa_v0 >= 0:
                        cache_row_inline2146__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline2099__ssa_v0, pl.INDEX)
                        kv_flat_inline2098__ssa_v3: pl.Tensor[[KV_T_DYN, 128], pl.FP32] = pl.tensor.assemble(kv_flat_inline2098__iter_v1, pl.tensor.slice(kv_final_inline2118__rv_v2, [1, 128], [token_inline2123__ssa_v2, 0]), [token_inline2123__ssa_v2, 0])
                        idx_kv_cache_flat_inline2161__ssa_v3: pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.assemble(idx_kv_cache_flat_inline2161__iter_v1, pl.tensor.slice(kv_i8_blk_inline2095__ssa_v0, [1, 128], [inner_inline2063__idx_v0, 0]), [cache_row_inline2146__ssa_v0, 0])
                        pl.tensor.write(idx_kv_scale_flat_inline2116__ssa_v0, [cache_row_inline2146__ssa_v0, 0], pl.tensor.read(kv_scale_dq_col_inline2065__ssa_v0, [inner_inline2063__idx_v0, 0]))
                        idx_kv_cache_flat_inline2161__phi_v4, kv_flat_inline2098__phi_v4 = pl.yield_(idx_kv_cache_flat_inline2161__ssa_v3, kv_flat_inline2098__ssa_v3)
                    else:
                        idx_kv_cache_flat_inline2161__phi_v4, kv_flat_inline2098__phi_v4 = pl.yield_(idx_kv_cache_flat_inline2161__iter_v1, kv_flat_inline2098__iter_v1)
                    idx_kv_cache_flat_inline2161__rv_v2, kv_flat_inline2098__rv_v2 = pl.yield_(idx_kv_cache_flat_inline2161__phi_v4, kv_flat_inline2098__phi_v4)
            idx_cache_write_tid_inline1316__ssa_v0: pl.Scalar[pl.TASK_ID] = _write_tid_inline2106__ssa_v0
            bs_inline2301__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243__phi_v4, 0)
            bs_heads_inline2228__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2301__ssa_v0 * 64
            row_blocks_inline2258__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2301__ssa_v0 + 16 - 1) // 16
            qr_acc_pad_inline2225__ssa_v0: pl.Tensor[[256, 8192], pl.INT32] = pl.tensor.create([256, 8192], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            for qr_unit_inline2215__ssa_v0 in pl.spmd(8 * row_blocks_inline2258__ssa_v0, name_hint="idx_qr_proj_matmul_spmd", allow_early_resolve=True):
                qr_rb_inline2194__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline2215__ssa_v0 // 8
                ot_inline2200__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline2215__ssa_v0 - qr_rb_inline2194__ssa_v0 * 8
                qr_r0_inline2195__ssa_v0: pl.Scalar[pl.INDEX] = qr_rb_inline2194__ssa_v0 * 16
                qr_rows_inline2208__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2301__ssa_v0 - qr_r0_inline2195__ssa_v0)
                o_base_inline2220__ssa_v0: pl.Scalar[pl.INDEX] = ot_inline2200__ssa_v0 * 1024
                for ns_inline2239__idx_v0, (qr_acc_pad_inline2225__iter_v1,) in pl.range(0, 1024, 512, init_values=(qr_acc_pad_inline2225__ssa_v0,)):
                    qr_acc_inline2212__ssa_v0: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.create([16, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                    for kb_inline2210__idx_v0, (qr_acc_inline2212__iter_v1,) in pl.pipeline(4, stage=2, init_values=(qr_acc_inline2212__ssa_v0,)):
                        q0_inline2232__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2210__idx_v0 * 256
                        qr_tile_inline2201__ssa_v0: pl.Tensor[[16, 256], pl.INT8, pl.TensorView(valid_shape=[qr_rows_inline2208__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(qr_inline1255__ssa_v0, [16, 256], [qr_r0_inline2195__ssa_v0, q0_inline2232__ssa_v0], [qr_rows_inline2208__ssa_v0, 256])
                        wq_tile_inline2230__ssa_v0: pl.Tensor[[256, 512], pl.INT8] = pl.tensor.slice(idx_wq_b__ssa_v0, [256, 512], [q0_inline2232__ssa_v0, o_base_inline2220__ssa_v0 + ns_inline2239__idx_v0])
                        if q0_inline2232__ssa_v0 == 0:
                            qr_acc_inline2212__ssa_v3: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul(qr_tile_inline2201__ssa_v0, wq_tile_inline2230__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.INT32)
                            qr_acc_inline2212__phi_v5: pl.Tensor[[16, 512], pl.INT32] = pl.yield_(qr_acc_inline2212__ssa_v3)
                        else:
                            qr_acc_inline2212__ssa_v4: pl.Tensor[[16, 512], pl.INT32] = pl.tensor.matmul_acc(qr_acc_inline2212__iter_v1, qr_tile_inline2201__ssa_v0, wq_tile_inline2230__ssa_v0, a_trans=False, b_trans=False)
                            qr_acc_inline2212__phi_v5: pl.Tensor[[16, 512], pl.INT32] = pl.yield_(qr_acc_inline2212__ssa_v4)
                        qr_acc_inline2212__rv_v2: pl.Tensor[[16, 512], pl.INT32] = pl.yield_(qr_acc_inline2212__phi_v5)
                    qr_acc_pad_inline2225__ssa_v3: pl.Tensor[[256, 8192], pl.INT32] = pl.tensor.assemble(qr_acc_pad_inline2225__iter_v1, qr_acc_inline2212__rv_v2, [qr_r0_inline2195__ssa_v0, o_base_inline2220__ssa_v0 + ns_inline2239__idx_v0])
                    qr_acc_pad_inline2225__rv_v2: pl.Tensor[[256, 8192], pl.INT32] = pl.yield_(qr_acc_pad_inline2225__ssa_v3)
            qr_proj_inline2268__ssa_v0: pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32] = pl.tensor.create([bs_inline2301__ssa_v0, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for ot_inline2284__ssa_v0 in pl.spmd(8, name_hint="idx_qr_proj_dequant_spmd", allow_early_resolve=True):
                o_base_inline2220__ssa_v1: pl.Scalar[pl.INDEX] = ot_inline2284__ssa_v0 * 1024
                wq_scale_inline2262__ssa_v0: pl.Tensor[[1, 1024], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(idx_wq_b_scale__ssa_v0, [1024], [o_base_inline2220__ssa_v1]), [1, 1024])
                for dq_t0_inline2203__idx_v0, (qr_proj_inline2268__iter_v1,) in pl.range(0, bs_inline2301__ssa_v0, 8, init_values=(qr_proj_inline2268__ssa_v0,)):
                    acc_fp32_inline2257__ssa_v0: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.cast(pl.tensor.slice(qr_acc_pad_inline2225__rv_v2, [8, 1024], [dq_t0_inline2203__idx_v0, o_base_inline2220__ssa_v1]), target_type=pl.FP32, mode='none')
                    qr_scale_tile_inline2236__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.slice(qr_scale_inline1310__ssa_v0, [8, 1], [dq_t0_inline2203__idx_v0, 0])
                    qr_dequant_inline2240__ssa_v0: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(acc_fp32_inline2257__ssa_v0, qr_scale_tile_inline2236__ssa_v0), wq_scale_inline2262__ssa_v0)
                    qr_proj_inline2268__ssa_v3: pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32] = pl.tensor.assemble(qr_proj_inline2268__iter_v1, qr_dequant_inline2240__ssa_v0, [dq_t0_inline2203__idx_v0, o_base_inline2220__ssa_v1])
                    qr_proj_inline2268__rv_v2: pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32] = pl.yield_(qr_proj_inline2268__ssa_v3)
            qr_proj_flat_inline2295__ssa_v0: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32] = pl.tensor.reshape(qr_proj_inline2268__rv_v2, [bs_heads_inline2228__ssa_v0, 128])
            qr_bf16_inline2223__ssa_v0: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16] = pl.tensor.create([bs_heads_inline2228__ssa_v0, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            rope_swap_idx_t_inline2189__ssa_v0: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.create([32, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rope_swap_idx", allow_early_resolve=True):
                sw_col_inline2204__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.full([32, 64], dtype=pl.FP32, value=1.0), pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round'))
                sw_dup_f_inline2263__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.cast(pl.tensor.cast(pl.tensor.muls(sw_col_inline2204__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc'), target_type=pl.FP32, mode='round')
                sw_lane_inline2241__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.sub(sw_col_inline2204__ssa_v0, pl.tensor.muls(sw_dup_f_inline2263__ssa_v0, 2.0))
                rope_swap_idx_t_inline2189__ssa_v1: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_t_inline2189__ssa_v0, pl.tensor.cast(pl.tensor.sub(pl.tensor.adds(sw_col_inline2204__ssa_v0, 1.0), pl.tensor.muls(sw_lane_inline2241__ssa_v0, 2.0)), target_type=pl.INT32, mode='round'), [0, 0])
            for idx_inline2276__ssa_v0 in pl.spmd(bs_heads_inline2228__ssa_v0 // 32, name_hint="qr_rope_spmd", allow_early_resolve=True):
                o0_inline2221__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2276__ssa_v0 * 32
                token_idx_inline2272__ssa_v0: pl.Scalar[pl.INDEX] = o0_inline2221__ssa_v0 // 64
                rope_swap_idx_inline2190__ssa_v0: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.slice(rope_swap_idx_t_inline2189__ssa_v1, [32, 64], [0, 0])
                cos_row_inline2192__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(idx_cos_il_inline1282__rv_v2, [1, 64], [token_idx_inline2272__ssa_v0, 0])
                sin_row_inline2231__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(idx_sin_signed_inline1307__rv_v2, [1, 64], [token_idx_inline2272__ssa_v0, 0])
                qr_nope_slice_inline2206__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(qr_proj_flat_inline2295__ssa_v0, [32, 64], [o0_inline2221__ssa_v0, 0])
                qr_rope_slice_inline2222__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(qr_proj_flat_inline2295__ssa_v0, [32, 64], [o0_inline2221__ssa_v0, 64])
                qr_swapped_inline2205__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.gather(qr_rope_slice_inline2222__ssa_v0, rope_swap_idx_inline2190__ssa_v0, dim=-1)
                rope_rot_inline2185__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(qr_rope_slice_inline2222__ssa_v0, cos_row_inline2192__ssa_v0), pl.tensor.col_expand_mul(qr_swapped_inline2205__ssa_v0, sin_row_inline2231__ssa_v0))
                qr_vec_inline2181__ssa_v0: pl.Tensor[[32, 128], pl.BF16] = pl.tensor.concat(pl.tensor.cast(qr_nope_slice_inline2206__ssa_v0, target_type=pl.BF16, mode='rint'), pl.tensor.cast(rope_rot_inline2185__ssa_v0, target_type=pl.BF16, mode='rint'))
                qr_bf16_inline2223__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16] = pl.tensor.assemble(qr_bf16_inline2223__ssa_v0, qr_vec_inline2181__ssa_v0, [o0_inline2221__ssa_v0, 0])
            qh_acc_gm_inline2179__ssa_v0: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32] = pl.tensor.create([bs_heads_inline2228__ssa_v0, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            for idx_inline2227__ssa_v0 in pl.spmd(bs_heads_inline2228__ssa_v0 // 64, name_hint="qr_hadamard_matmul_spmd", allow_early_resolve=True):
                o0_inline2221__ssa_v1: pl.Scalar[pl.INDEX] = idx_inline2227__ssa_v0 * 64
                qh_acc_inline2178__ssa_v0: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.matmul(pl.tensor.slice(qr_bf16_inline2223__ssa_v1, [64, 128], [o0_inline2221__ssa_v1, 0]), hadamard_idx__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                qh_acc_gm_inline2179__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32] = pl.tensor.assemble(qh_acc_gm_inline2179__ssa_v0, qh_acc_inline2178__ssa_v0, [o0_inline2221__ssa_v1, 0])
            qr_hadamard_i8_inline2177__ssa_v0: pl.Tensor[[16384, 128], pl.INT8] = pl.tensor.create([16384, 128], dtype=pl.INT8, layout=pl.TensorLayout.ND)
            qr_hadamard_scale_dq_inline2234__ssa_v0: pl.Tensor[[16384, 1], pl.FP32] = pl.tensor.create([16384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(bs_heads_inline2228__ssa_v0 // 64, name_hint="qr_hadamard_quant_spmd", allow_early_resolve=True) as qh_quant_tid_inline2182__ssa_v0:
                idx_inline2227__ssa_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                o0_v1_inline2218__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2227__ssa_v1 * 64
                qh_amax_inline2278__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.full([1, 64], dtype=pl.FP32, value=0.0001)
                for h0_inline2198__idx_v0, (qh_amax_inline2278__iter_v1,) in pl.range(0, 128, 64, init_values=(qh_amax_inline2278__ssa_v0,)):
                    qh_a_f32_inline2175__ssa_v0: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(qh_acc_gm_inline2179__ssa_v1, [64, 64], [o0_v1_inline2218__ssa_v0, h0_inline2198__idx_v0])
                    qh_a_abs_inline2196__ssa_v0: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.maximum(qh_a_f32_inline2175__ssa_v0, pl.tensor.neg(qh_a_f32_inline2175__ssa_v0))
                    qh_a_max_inline2211__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(qh_a_abs_inline2196__ssa_v0), [1, 64])
                    qh_amax_inline2278__ssa_v3: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.maximum(qh_amax_inline2278__iter_v1, qh_a_max_inline2211__ssa_v0)
                    qh_amax_inline2278__rv_v2: pl.Tensor[[1, 64], pl.FP32] = pl.yield_(qh_amax_inline2278__ssa_v3)
                qh_scale_quant_row_inline2183__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.div(pl.tensor.full([1, 64], dtype=pl.FP32, value=127.0), qh_amax_inline2278__rv_v2)
                qh_scale_dq_inline2176__ssa_v0: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.recip(qh_scale_quant_row_inline2183__ssa_v0), [64, 1])
                qr_hadamard_scale_dq_inline2234__ssa_v1: pl.Tensor[[16384, 1], pl.FP32] = pl.tensor.assemble(qr_hadamard_scale_dq_inline2234__ssa_v0, qh_scale_dq_inline2176__ssa_v0, [o0_v1_inline2218__ssa_v0, 0])
                qh_scale_quant_inline2202__ssa_v0: pl.Tensor[[64, 1], pl.FP32] = pl.tensor.reshape(qh_scale_quant_row_inline2183__ssa_v0, [64, 1])
                for h1_inline2188__idx_v0, (qr_hadamard_i8_inline2177__iter_v1,) in pl.range(0, 128, 64, init_values=(qr_hadamard_i8_inline2177__ssa_v0,)):
                    qh_q_f32_inline2226__ssa_v0: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(qh_acc_gm_inline2179__ssa_v1, [64, 64], [o0_v1_inline2218__ssa_v0, h1_inline2188__idx_v0])
                    qh_q_scaled_inline2191__ssa_v0: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.row_expand_mul(qh_q_f32_inline2226__ssa_v0, qh_scale_quant_inline2202__ssa_v0)
                    qh_q_i32_inline2199__ssa_v0: pl.Tensor[[64, 64], pl.INT32] = pl.tensor.cast(qh_q_scaled_inline2191__ssa_v0, target_type=pl.INT32, mode='rint')
                    qh_q_half_inline2290__ssa_v0: pl.Tensor[[64, 64], pl.FP16] = pl.tensor.cast(qh_q_i32_inline2199__ssa_v0, target_type=pl.FP16, mode='round')
                    qh_i8_inline2233__ssa_v0: pl.Tensor[[64, 64], pl.INT8] = pl.tensor.cast(qh_q_half_inline2290__ssa_v0, target_type=pl.INT8, mode='trunc')
                    qr_hadamard_i8_inline2177__ssa_v3: pl.Tensor[[16384, 128], pl.INT8] = pl.tensor.assemble(qr_hadamard_i8_inline2177__iter_v1, qh_i8_inline2233__ssa_v0, [o0_v1_inline2218__ssa_v0, h1_inline2188__idx_v0])
                    qr_hadamard_i8_inline2177__rv_v2: pl.Tensor[[16384, 128], pl.INT8] = pl.yield_(qr_hadamard_i8_inline2177__ssa_v3)
            x_flat_inline2242__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16] = x_normed_t_inline1243__phi_v4
            weights_inline2244__ssa_v0: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            weights_partial_inline2245__ssa_v0: pl.Tensor[[1024, 64], pl.FP32] = pl.tensor.create([1024, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(4 * row_blocks_inline2258__ssa_v0, name_hint="weights_proj_spmd", deps=[late_dep_inline1297__ssa_v0]) as _weights_tid_inline2286__ssa_v0:
                w_unit_inline2246__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                w_rb_inline2237__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline2246__ssa_v0 // 4
                kb_inline2210__ssa_v1: pl.Scalar[pl.INDEX] = w_unit_inline2246__ssa_v0 - w_rb_inline2237__ssa_v0 * 4
                w_r0_inline2207__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_inline2237__ssa_v0 * 16
                w_rows_inline2229__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, bs_inline2301__ssa_v0 - w_r0_inline2207__ssa_v0)
                k_base_inline2248__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2210__ssa_v1 * 1024
                weights_acc_inline2217__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.create([16, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for db_inline2297__idx_v0, (weights_acc_inline2217__iter_v1,) in pl.range(2, init_values=(weights_acc_inline2217__ssa_v0,)):
                    d0_inline2250__ssa_v0: pl.Scalar[pl.INDEX] = k_base_inline2248__ssa_v0 + db_inline2297__idx_v0 * 512
                    x_tile_inline2253__ssa_v0: pl.Tensor[[16, 512], pl.BF16, pl.TensorView(valid_shape=[w_rows_inline2229__ssa_v0, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(x_flat_inline2242__ssa_v0, [16, 512], [w_r0_inline2207__ssa_v0, d0_inline2250__ssa_v0], [w_rows_inline2229__ssa_v0, 512])
                    weights_proj_tile_inline2255__ssa_v0: pl.Tensor[[512, 64], pl.BF16] = pl.tensor.slice(weights_proj__ssa_v0, [512, 64], [d0_inline2250__ssa_v0, 0])
                    if db_inline2297__idx_v0 == 0:
                        weights_acc_inline2217__ssa_v3: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul(x_tile_inline2253__ssa_v0, weights_proj_tile_inline2255__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        weights_acc_inline2217__phi_v5: pl.Tensor[[16, 64], pl.FP32] = pl.yield_(weights_acc_inline2217__ssa_v3)
                    else:
                        weights_acc_inline2217__ssa_v4: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.matmul_acc(weights_acc_inline2217__iter_v1, x_tile_inline2253__ssa_v0, weights_proj_tile_inline2255__ssa_v0, a_trans=False, b_trans=False)
                        weights_acc_inline2217__phi_v5: pl.Tensor[[16, 64], pl.FP32] = pl.yield_(weights_acc_inline2217__ssa_v4)
                    weights_acc_inline2217__rv_v2: pl.Tensor[[16, 64], pl.FP32] = pl.yield_(weights_acc_inline2217__phi_v5)
                weights_partial_inline2245__ssa_v1: pl.Tensor[[1024, 64], pl.FP32] = pl.tensor.assemble(weights_partial_inline2245__ssa_v0, weights_acc_inline2217__rv_v2, [kb_inline2210__ssa_v1 * 256 + w_r0_inline2207__ssa_v0, 0])
            with pl.spmd(row_blocks_inline2258__ssa_v0, name_hint="weights_proj_reduce_spmd", allow_early_resolve=True) as weights_tid_inline2256__ssa_v0:
                w_rb_v1_inline2235__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                w_r0_v1_inline2213__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_v1_inline2235__ssa_v0 * 16
                w_sum_inline2209__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(weights_partial_inline2245__ssa_v1, [16, 64], [w_r0_v1_inline2213__ssa_v0, 0])
                partial_r0_inline2260__ssa_v0: pl.Scalar[pl.INDEX] = pl.const(1, pl.INDEX) * pl.const(256, pl.INDEX) + w_r0_v1_inline2213__ssa_v0
                w_sum_inline2209__ssa_v1: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(w_sum_inline2209__ssa_v0, pl.tensor.slice(weights_partial_inline2245__ssa_v1, [16, 64], [partial_r0_inline2260__ssa_v0, 0]))
                partial_r0_inline2260__ssa_v1: pl.Scalar[pl.INDEX] = pl.const(2, pl.INDEX) * pl.const(256, pl.INDEX) + w_r0_v1_inline2213__ssa_v0
                w_sum_inline2209__ssa_v2: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(w_sum_inline2209__ssa_v1, pl.tensor.slice(weights_partial_inline2245__ssa_v1, [16, 64], [partial_r0_inline2260__ssa_v1, 0]))
                partial_r0_inline2260__ssa_v2: pl.Scalar[pl.INDEX] = pl.const(3, pl.INDEX) * pl.const(256, pl.INDEX) + w_r0_v1_inline2213__ssa_v0
                w_sum_inline2209__ssa_v3: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(w_sum_inline2209__ssa_v2, pl.tensor.slice(weights_partial_inline2245__ssa_v1, [16, 64], [partial_r0_inline2260__ssa_v2, 0]))
                weights_inline2244__ssa_v1: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(weights_inline2244__ssa_v0, pl.tensor.muls(w_sum_inline2209__ssa_v3, 0.011048543456039806), [w_r0_v1_inline2213__ssa_v0, 0])
            # Run exact Top-K with the score and pair arenas on separate rings.
            bs_inline61_inline2238__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_positions_inline1323__ssa_v0, 0)
            b_dim_inline57_inline2261__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table__ssa_v0, 0)
            idx_block_num_inline53_inline2264__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_kv_cache__ssa_v0, 0)
            idx_table_len_inline55_inline2193__ssa_v0: pl.Scalar[pl.INDEX] = b_dim_inline57_inline2261__ssa_v0 * 8192
            kv_cache_i8_flat_inline46_inline2265__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8] = pl.tensor.reshape(idx_kv_cache__ssa_v0, [idx_block_num_inline53_inline2264__ssa_v0 * 32, 128])
            kv_scale_flat_inline50_inline2214__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32] = pl.tensor.reshape(idx_kv_scale__ssa_v0, [idx_block_num_inline53_inline2264__ssa_v0 * 32, 1])
            idx_block_table_flat_inline47_inline2186__ssa_v0: pl.Tensor[[idx_table_len_inline55_inline2193__ssa_v0], pl.INT32] = pl.tensor.reshape(idx_block_table__ssa_v0, [idx_table_len_inline55_inline2193__ssa_v0])
            pair_arena_inline71_inline2266__ssa_v0: pl.Tensor[[4192, 1024], pl.FP32] = pl.tensor.create([4192, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.scope():
                score_arena_inline44_inline2267__ssa_v0: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32] = pl.tensor.create([bs_inline61_inline2238__ssa_v0, 262144], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                with pl.spmd(24, name_hint="indexer_score_leaf_wave_spmd", optimizations=[pl.cross_core_slot(slot_num=2)], deps=[qh_quant_tid_inline2182__ssa_v0, weights_tid_inline2256__ssa_v0, idx_cache_write_tid_inline1316__ssa_v0]) as score_tid_inline45_inline2269__ssa_v0:
                    worker_inline75_inline2270__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                    query_count_inline56_inline2271__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_positions_inline1323__ssa_v0, 0)
                    global_leaf_base_inline51_inline2273__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    for query_inline65_inline2274__idx_v0, (global_leaf_base_inline51_inline2273__iter_v1, score_arena_inline44_inline2267__iter_v1) in pl.range(query_count_inline56_inline2271__ssa_v0, init_values=(global_leaf_base_inline51_inline2273__ssa_v0, score_arena_inline44_inline2267__ssa_v0)):
                        batch_idx_inline60_inline2249__ssa_v0: pl.Scalar[pl.INDEX] = query_inline65_inline2274__idx_v0 // 8
                        position_inline54_inline2275__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(idx_positions_inline1323__ssa_v0, [query_inline65_inline2274__idx_v0])
                        cache_len_inline64_inline2279__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0]), pl.INDEX) // 4
                        visible_count_inline49_inline2280__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len_inline64_inline2279__ssa_v0, (pl.cast(position_inline54_inline2275__ssa_v0, pl.INDEX) + 1) // 4), 262144), 0)
                        leaf_count_inline66_inline2281__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count_inline49_inline2280__ssa_v0 + 8192 - 1) // 8192
                        base_mod_inline52_inline2283__ssa_v0: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273__iter_v1 % 24
                        first_leaf_inline67_inline2285__ssa_v0: pl.Scalar[pl.INDEX] = (worker_inline75_inline2270__ssa_v0 + base_mod_inline52_inline2283__ssa_v0) % 24
                        for leaf_inline48_inline2288__idx_v0, (score_arena_inline44_inline2267__iter_v3,) in pl.range(first_leaf_inline67_inline2285__ssa_v0, leaf_count_inline66_inline2281__ssa_v0, 24, init_values=(score_arena_inline44_inline2267__iter_v1,)):
                            logical_begin_inline63_inline2224__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline48_inline2288__idx_v0 * 8192
                            valid_count_inline68_inline2219__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8192, visible_count_inline49_inline2280__ssa_v0 - logical_begin_inline63_inline2224__ssa_v0)
                            query_head_begin_inline69_inline2289__ssa_v0: pl.Scalar[pl.INDEX] = query_inline65_inline2274__idx_v0 * 64
                            query_vector_inline70_inline2291__ssa_v0: pl.Tensor[[64, 128], pl.INT8] = pl.tensor.slice(qr_hadamard_i8_inline2177__rv_v2, [64, 128], [query_head_begin_inline69_inline2289__ssa_v0, 0])
                            query_scale_inline73_inline2197__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(qr_hadamard_scale_dq_inline2234__ssa_v1, [64, 1], [query_head_begin_inline69_inline2289__ssa_v0, 0]), [1, 64])
                            query_weight_inline76_inline2243__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(weights_inline2244__ssa_v1, [1, 64], [query_inline65_inline2274__idx_v0, 0])
                            page_count_inline72_inline2252__ssa_v0: pl.Scalar[pl.INDEX] = (valid_count_inline68_inline2219__ssa_v0 + 32 - 1) // 32
                            for page_inline43_inline2292__idx_v0, (score_arena_inline44_inline2267__iter_v5,) in pl.pipeline(page_count_inline72_inline2252__ssa_v0, stage=2, init_values=(score_arena_inline44_inline2267__iter_v3,)):
                                page_begin_inline42_inline2287__ssa_v0: pl.Scalar[pl.INDEX] = page_inline43_inline2292__idx_v0 * 32
                                logical_row_inline41_inline2282__ssa_v0: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0 + page_begin_inline42_inline2287__ssa_v0
                                logical_page_inline74_inline2293__ssa_v0: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0 // 32
                                physical_block_inline40_inline2298__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0 * 8192 + logical_page_inline74_inline2293__ssa_v0]), pl.INDEX)
                                physical_row_inline59_inline2299__ssa_v0: pl.Scalar[pl.INDEX] = physical_block_inline40_inline2298__ssa_v0 * 32
                                kv_i8_inline38_inline2300__ssa_v0: pl.Tensor[[32, 128], pl.INT8] = pl.tensor.slice(kv_cache_i8_flat_inline46_inline2265__ssa_v0, [32, 128], [physical_row_inline59_inline2299__ssa_v0, 0])
                                score_i32_inline37_inline2302__ssa_v0: pl.Tensor[[32, 64], pl.INT32] = pl.tensor.matmul(kv_i8_inline38_inline2300__ssa_v0, query_vector_inline70_inline2291__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.INT32)
                                score_fp32_inline35_inline2254__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.cast(score_i32_inline37_inline2302__ssa_v0, target_type=pl.FP32, mode='none')
                                score_fp32_v1_inline34_inline2251__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(score_fp32_inline35_inline2254__ssa_v0, query_scale_inline73_inline2197__ssa_v0)
                                score_fp32_v2_inline62_inline2294__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.maximum(score_fp32_v1_inline34_inline2251__ssa_v0, 0.0)
                                score_fp32_v3_inline33_inline2187__ssa_v0: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.col_expand_mul(score_fp32_v2_inline62_inline2294__ssa_v0, query_weight_inline76_inline2243__ssa_v0)
                                kv_scale_inline32_inline2184__ssa_v0: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.slice(kv_scale_flat_inline50_inline2214__ssa_v0, [32, 1], [physical_row_inline59_inline2299__ssa_v0, 0])
                                score_inline58_inline2180__ssa_v0: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.mul(pl.tensor.row_sum(score_fp32_v3_inline33_inline2187__ssa_v0), kv_scale_inline32_inline2184__ssa_v0)
                                score_row_inline31_inline2296__ssa_v0: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.reshape(score_inline58_inline2180__ssa_v0, [1, 32])
                                valid_rows_inline39_inline2216__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32, valid_count_inline68_inline2219__ssa_v0 - page_begin_inline42_inline2287__ssa_v0)
                                score_valid_inline30_inline2277__ssa_v0: pl.Tensor[[1, 32], pl.FP32] = pl.tensor.fillpad(pl.tensor.set_validshape(score_row_inline31_inline2296__ssa_v0, 1, valid_rows_inline39_inline2216__ssa_v0), pad_value=pl.PadValue.min)
                                score_arena_inline44_inline2267__ssa_v7: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32] = pl.tensor.assemble(score_arena_inline44_inline2267__iter_v5, score_valid_inline30_inline2277__ssa_v0, [query_inline65_inline2274__idx_v0, logical_row_inline41_inline2282__ssa_v0])
                                score_arena_inline44_inline2267__rv_v6: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32] = pl.yield_(score_arena_inline44_inline2267__ssa_v7)
                            score_arena_inline44_inline2267__rv_v4: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32] = pl.yield_(score_arena_inline44_inline2267__rv_v6)
                        global_leaf_base_inline51_inline2273__ssa_v3: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273__iter_v1 + leaf_count_inline66_inline2281__ssa_v0
                        global_leaf_base_inline51_inline2273__rv_v2, score_arena_inline44_inline2267__rv_v2 = pl.yield_(global_leaf_base_inline51_inline2273__ssa_v3, score_arena_inline44_inline2267__rv_v4)
                with pl.spmd(48, name_hint="indexer_topk_group_wave", deps=[score_tid_inline45_inline2269__ssa_v0]) as topk_tid_inline36_inline2174__ssa_v0:
                    self.indexer_topk_group_wave(idx_positions_inline1323__ssa_v0, kv_seq_lens__ssa_v0, score_arena_inline44_inline2267__rv_v2, pair_arena_inline71_inline2266__ssa_v0)
                with pl.spmd(bs_inline61_inline2238__ssa_v0, name_hint="indexer_topk_query_merge", deps=[topk_tid_inline36_inline2174__ssa_v0]) as _score_tid_inline29_inline2247__ssa_v0:
                    self.indexer_topk_query_merge(idx_positions_inline1323__ssa_v0, kv_seq_lens__ssa_v0, pair_arena_inline71_inline2266__ssa_v0, idx_topk_scores_inline1271__ssa_v0, idx_topk_inline1280__ssa_v0)
            idx_topk_scores_inline1271__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.FP32] = idx_topk_scores_inline1271__ssa_v0
            idx_topk_inline1280__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.INT32] = idx_topk_inline1280__ssa_v0
            idx_topk_scores_inline1271__ssa_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.FP32] = idx_topk_scores_inline1271__ssa_v1
            idx_topk_inline1280__ssa_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.INT32] = idx_topk_inline1280__ssa_v1
            # Plan and run CSA QK/PV over sparse blocks, and build inverse-RoPE metadata.
            ori_block_num_inline2362__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache__ssa_v0, 0)
            t_dim_inline2369__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_inline1246__ssa_v0, 0)
            t_heads_inline2364__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2369__ssa_v0 * 64
            t_blk_inline2373__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2369__ssa_v0 * 4 * 5 * 16
            qk_items_inline2347__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2369__ssa_v0 * 5
            rope_cs_blocks_inline2380__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2369__ssa_v0 // 8
            ori_kv_flat_inline2344__ssa_v0: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(kv_cache__ssa_v0, [ori_block_num_inline2362__ssa_v0 * 32, 512])
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_touch", allow_early_resolve=True):
                ori_kv_flat_inline2344__ssa_v1: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.assemble(ori_kv_flat_inline2344__ssa_v0, pl.tensor.slice(ori_kv_flat_inline2344__ssa_v0, [1, 512], [0, 0]), [0, 0])
            sparse_bias_inline2381__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32] = pl.tensor.create([t_dim_inline2369__ssa_v0, 640], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            cmp_sparse_indices_inline2383__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32] = pl.tensor.create([t_dim_inline2369__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            valid_block_mask_inline2385__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 5], pl.INT32] = pl.tensor.create([t_dim_inline2369__ssa_v0, 5], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            qk_order_inline2351__ssa_v0: pl.Tensor[[1280], pl.INT32] = pl.tensor.create([1280], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            qk_wcur_inline2412__ssa_v0: pl.Tensor[[1], pl.INT32] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_slots_build_valid_qk_plan") as qk_plan_tid_inline2387__ssa_v0:
                for bias_t0_inline2361__idx_v0, (cmp_sparse_indices_inline2383__iter_v1, sparse_bias_inline2381__iter_v1) in pl.range(0, t_dim_inline2369__ssa_v0, 8, init_values=(cmp_sparse_indices_inline2383__ssa_v0, sparse_bias_inline2381__ssa_v0)):
                    c_raw_inline2382__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.cast(pl.tensor.slice(idx_topk_inline1280__ssa_v2, [8, 512], [bias_t0_inline2361__idx_v0, 0]), target_type=pl.FP32, mode='round')
                    c_pos_inline2359__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.cast(pl.tensor.slice(position_ids_t1_inline1288__ssa_v0, [8, 1], [bias_t0_inline2361__idx_v0, 0]), target_type=pl.FP32, mode='round')
                    c_pos_scaled_inline2417__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.muls(pl.tensor.adds(c_pos_inline2359__ssa_v0, 1.0), 0.25)
                    c_pos_i32_inline2358__ssa_v0: pl.Tensor[[8, 1], pl.INT32] = pl.tensor.cast(c_pos_scaled_inline2417__ssa_v0, target_type=pl.INT32, mode='trunc')
                    c_pos_q_inline2367__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.cast(c_pos_i32_inline2358__ssa_v0, target_type=pl.FP32, mode='round')
                    c_upper_b_inline2379__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.row_expand_mul(pl.tensor.full([8, 512], dtype=pl.FP32, value=1.0), c_pos_q_inline2367__ssa_v0)
                    c_ge_inline2421__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.adds(c_raw_inline2382__ssa_v0, 1.0), 0.0), 1.0)
                    c_lt_inline2354__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.sub(c_upper_b_inline2379__ssa_v0, c_raw_inline2382__ssa_v0), 0.0), 1.0)
                    c_mask_inline2370__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.mul(c_ge_inline2421__ssa_v0, c_lt_inline2354__ssa_v0)
                    c_out_inline2410__ssa_v0: pl.Tensor[[8, 512], pl.FP32] = pl.tensor.subs(pl.tensor.mul(c_mask_inline2370__ssa_v0, pl.tensor.adds(c_raw_inline2382__ssa_v0, 1.0)), 1.0)
                    cmp_sparse_indices_inline2383__ssa_v3: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32] = pl.tensor.assemble(cmp_sparse_indices_inline2383__iter_v1, pl.tensor.cast(c_out_inline2410__ssa_v0, target_type=pl.INT32, mode='round'), [bias_t0_inline2361__idx_v0, 0])
                    for c_t0_inline2340__idx_v0 in pl.range(8):
                        pl.tensor.write(valid_block_mask_inline2385__ssa_v0, [bias_t0_inline2361__idx_v0 + c_t0_inline2340__idx_v0, 0], pl.cast(1, pl.INT32))
                    for c_sb_inline2407__idx_v0 in pl.range(1, 5):
                        c_s0_inline2342__ssa_v0: pl.Scalar[pl.INDEX] = (c_sb_inline2407__idx_v0 - 1) * 128
                        c_blk_valid_inline2350__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.row_max(pl.tensor.slice(c_mask_inline2370__ssa_v0, [8, 128], [0, c_s0_inline2342__ssa_v0]))
                        for c_dt_inline2378__idx_v0 in pl.range(8):
                            c_valid_inline2389__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(pl.tensor.read(c_blk_valid_inline2350__ssa_v0, [c_dt_inline2378__idx_v0, 0]), pl.INT32)
                            pl.tensor.write(valid_block_mask_inline2385__ssa_v0, [bias_t0_inline2361__idx_v0 + c_dt_inline2378__idx_v0, c_sb_inline2407__idx_v0], c_valid_inline2389__ssa_v0)
                    v_win_f_inline2375__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(window_swa_indices__ssa_v0, [8, 128], [bias_t0_inline2361__idx_v0, 0]), target_type=pl.FP32, mode='round')
                    v_win_valid_inline2414__ssa_v0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.minimum(pl.tensor.maximum(pl.tensor.adds(v_win_f_inline2375__ssa_v0, 1.0), 0.0), 1.0)
                    sparse_bias_inline2381__ssa_v3: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32] = pl.tensor.assemble(sparse_bias_inline2381__iter_v1, pl.tensor.muls(pl.tensor.subs(v_win_valid_inline2414__ssa_v0, 1.0), 1e+20), [bias_t0_inline2361__idx_v0, 0])
                    sparse_bias_inline2381__ssa_v4: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32] = pl.tensor.assemble(sparse_bias_inline2381__ssa_v3, pl.tensor.muls(pl.tensor.minimum(c_out_inline2410__ssa_v0, 0.0), 1e+20), [bias_t0_inline2361__idx_v0, 128])
                    if pl.const(640, pl.INDEX) > pl.const(640, pl.INDEX):
                        sparse_bias_inline2381__ssa_v5: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32] = pl.tensor.assemble(sparse_bias_inline2381__ssa_v4, pl.tensor.full([8, 0], dtype=pl.FP32, value=-1e+20), [bias_t0_inline2361__idx_v0, 640])
                        sparse_bias_inline2381__phi_v6: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32] = pl.yield_(sparse_bias_inline2381__ssa_v5)
                    else:
                        sparse_bias_inline2381__phi_v6: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32] = pl.yield_(sparse_bias_inline2381__ssa_v4)
                    cmp_sparse_indices_inline2383__rv_v2, sparse_bias_inline2381__rv_v2 = pl.yield_(cmp_sparse_indices_inline2383__ssa_v3, sparse_bias_inline2381__phi_v6)
                pl.tensor.write(qk_wcur_inline2412__ssa_v0, [0], pl.cast(0, pl.INT32))
                for plan_t_inline2392__idx_v0 in pl.range(t_dim_inline2369__ssa_v0):
                    for plan_sb_inline2386__idx_v0 in pl.range(5):
                        if pl.cast(pl.tensor.read(valid_block_mask_inline2385__ssa_v0, [plan_t_inline2392__idx_v0, plan_sb_inline2386__idx_v0]), pl.INDEX) > 0:
                            plan_w_inline2348__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(qk_wcur_inline2412__ssa_v0, [0])
                            pl.tensor.write(qk_order_inline2351__ssa_v0, [plan_w_inline2348__ssa_v0], pl.cast(plan_t_inline2392__idx_v0 * 5 + plan_sb_inline2386__idx_v0, pl.INT32))
                            pl.tensor.write(qk_wcur_inline2412__ssa_v0, [0], pl.cast(pl.cast(plan_w_inline2348__ssa_v0, pl.INDEX) + 1, pl.INT32))
                for plan_t_inline2394__idx_v0 in pl.range(t_dim_inline2369__ssa_v0):
                    for plan_sb_inline2395__idx_v0 in pl.range(5):
                        if pl.cast(pl.tensor.read(valid_block_mask_inline2385__ssa_v0, [plan_t_inline2394__idx_v0, plan_sb_inline2395__idx_v0]), pl.INDEX) <= 0:
                            plan_w_v1_inline2399__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(qk_wcur_inline2412__ssa_v0, [0])
                            pl.tensor.write(qk_order_inline2351__ssa_v0, [plan_w_v1_inline2399__ssa_v0], pl.cast(plan_t_inline2394__idx_v0 * 5 + plan_sb_inline2395__idx_v0, pl.INT32))
                            pl.tensor.write(qk_wcur_inline2412__ssa_v0, [0], pl.cast(pl.cast(plan_w_v1_inline2399__ssa_v0, pl.INDEX) + 1, pl.INT32))
            cmp_block_num_inline2376__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
            cmp_kv_flat_inline2401__ssa_v0: pl.Tensor[[cmp_block_num_inline2376__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16] = pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline2376__ssa_v0 * 32, 512])
            q_flat_inline2355__ssa_v0: pl.Tensor[[t_heads_inline2364__ssa_v0, 512], pl.BF16] = pl.tensor.reshape(q_inline1246__ssa_v0, [t_heads_inline2364__ssa_v0, 512])
            sparse_blk_mi_inline2404__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32] = pl.tensor.create([t_blk_inline2373__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            sparse_blk_li_inline2405__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32] = pl.tensor.create([t_blk_inline2373__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            sparse_blk_oi_inline2398__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32] = pl.tensor.create([t_blk_inline2373__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            with pl.spmd(24, name_hint="qk_pv_spmd", deps=[qk_plan_tid_inline2387__ssa_v0, cache_ready_dep_inline1304__ssa_v0], allow_early_resolve=True) as qk_tid_inline2349__ssa_v0:
                qk_core_inline2368__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                qk_lane_iters_inline2408__ssa_v0: pl.Scalar[pl.INDEX] = (qk_items_inline2347__ssa_v0 - qk_core_inline2368__ssa_v0 + 24 - 1) // 24
                for qk_it_inline2413__idx_v0, (sparse_blk_li_inline2405__iter_v1, sparse_blk_mi_inline2404__iter_v1, sparse_blk_oi_inline2398__iter_v1) in pl.range(qk_lane_iters_inline2408__ssa_v0, init_values=(sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0)):
                    qk_flat_inline2365__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2368__ssa_v0 + qk_it_inline2413__idx_v0 * 24
                    qk_item_inline2403__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(qk_order_inline2351__ssa_v0, [qk_flat_inline2365__ssa_v0]), pl.INDEX)
                    qk_t_inline2411__ssa_v0: pl.Scalar[pl.INDEX] = qk_item_inline2403__ssa_v0 // 5
                    qk_sb_inline2374__ssa_v0: pl.Scalar[pl.INDEX] = qk_item_inline2403__ssa_v0 - qk_t_inline2411__ssa_v0 * 5
                    qk_b_inline2371__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2411__ssa_v0 // 8
                    qk_token_base_inline2391__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2411__ssa_v0 * 4 * 5 * 16
                    qk_s0_inline2418__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline2374__ssa_v0 * 128
                    qk_bias_row_inline2419__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(sparse_bias_inline2381__rv_v2, [1, 128], [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0])
                    qk_block_valid_inline2422__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2385__ssa_v0, [qk_t_inline2411__ssa_v0, qk_sb_inline2374__ssa_v0])
                    if pl.cast(qk_block_valid_inline2422__ssa_v0, pl.INDEX) > 0:
                        qk_kv_inline2415__ssa_v0: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.create_l1([128, 512], dtype=pl.BF16, transpose=False)
                        qk_win_rows_inline2406__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(128 - qk_s0_inline2418__ssa_v0, 0), 128)
                        if qk_win_rows_inline2406__ssa_v0 > 0:
                            qk_pos_inline2339__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(position_ids_t1_inline1288__ssa_v0, [qk_t_inline2411__ssa_v0, 0]), pl.INDEX)
                            qk_win_len_inline2337__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(qk_pos_inline2339__ssa_v0 + 1, 128)
                            qk_win_start_inline2333__ssa_v0: pl.Scalar[pl.INDEX] = qk_pos_inline2339__ssa_v0 - qk_win_len_inline2337__ssa_v0 + 1
                            qk_run_rows_inline2332__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len_inline2337__ssa_v0 - qk_s0_inline2418__ssa_v0, 0), qk_win_rows_inline2406__ssa_v0)
                            qk_head_inline2353__ssa_v0: pl.Scalar[pl.INDEX] = (qk_win_start_inline2333__ssa_v0 + qk_s0_inline2418__ssa_v0) % 32
                            qk_run_lo_inline2372__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.const(0, pl.INDEX) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, 0)
                            qk_run_hi_inline2330__ssa_v0: pl.Scalar[pl.INDEX] = pl.min((pl.const(0, pl.INDEX) + pl.const(1, pl.INDEX)) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                            if qk_run_hi_inline2330__ssa_v0 > qk_run_lo_inline2372__ssa_v0:
                                qk_run_raw_inline2357__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v0])
                                qk_run_src_inline2384__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__ssa_v0, pl.INDEX), 0), pl.INDEX)
                                qk_kv_inline2415__ssa_v1: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__ssa_v0, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v0, 0], [qk_run_src_inline2384__ssa_v0, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v0 - qk_run_lo_inline2372__ssa_v0, 512], transpose=False)
                                qk_kv_inline2415__phi_v2: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v1)
                            else:
                                qk_kv_inline2415__phi_v2: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v0)
                            qk_run_lo_inline2372__ssa_v1: pl.Scalar[pl.INDEX] = pl.max(pl.const(1, pl.INDEX) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, 0)
                            qk_run_hi_inline2330__ssa_v1: pl.Scalar[pl.INDEX] = pl.min((pl.const(1, pl.INDEX) + pl.const(1, pl.INDEX)) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                            if qk_run_hi_inline2330__ssa_v1 > qk_run_lo_inline2372__ssa_v1:
                                qk_run_raw_inline2357__ssa_v1: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v1])
                                qk_run_src_inline2384__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__ssa_v1, pl.INDEX), 0), pl.INDEX)
                                qk_kv_inline2415__ssa_v3: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__phi_v2, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v1, 0], [qk_run_src_inline2384__ssa_v1, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v1 - qk_run_lo_inline2372__ssa_v1, 512], transpose=False)
                                qk_kv_inline2415__phi_v4: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v3)
                            else:
                                qk_kv_inline2415__phi_v4: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__phi_v2)
                            qk_run_lo_inline2372__ssa_v2: pl.Scalar[pl.INDEX] = pl.max(pl.const(2, pl.INDEX) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, 0)
                            qk_run_hi_inline2330__ssa_v2: pl.Scalar[pl.INDEX] = pl.min((pl.const(2, pl.INDEX) + pl.const(1, pl.INDEX)) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                            if qk_run_hi_inline2330__ssa_v2 > qk_run_lo_inline2372__ssa_v2:
                                qk_run_raw_inline2357__ssa_v2: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v2])
                                qk_run_src_inline2384__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__ssa_v2, pl.INDEX), 0), pl.INDEX)
                                qk_kv_inline2415__ssa_v5: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__phi_v4, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v2, 0], [qk_run_src_inline2384__ssa_v2, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v2 - qk_run_lo_inline2372__ssa_v2, 512], transpose=False)
                                qk_kv_inline2415__phi_v6: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v5)
                            else:
                                qk_kv_inline2415__phi_v6: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__phi_v4)
                            qk_run_lo_inline2372__ssa_v3: pl.Scalar[pl.INDEX] = pl.max(pl.const(3, pl.INDEX) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, 0)
                            qk_run_hi_inline2330__ssa_v3: pl.Scalar[pl.INDEX] = pl.min((pl.const(3, pl.INDEX) + pl.const(1, pl.INDEX)) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                            if qk_run_hi_inline2330__ssa_v3 > qk_run_lo_inline2372__ssa_v3:
                                qk_run_raw_inline2357__ssa_v3: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v3])
                                qk_run_src_inline2384__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__ssa_v3, pl.INDEX), 0), pl.INDEX)
                                qk_kv_inline2415__ssa_v7: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__phi_v6, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v3, 0], [qk_run_src_inline2384__ssa_v3, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v3 - qk_run_lo_inline2372__ssa_v3, 512], transpose=False)
                                qk_kv_inline2415__phi_v8: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v7)
                            else:
                                qk_kv_inline2415__phi_v8: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__phi_v6)
                            qk_run_lo_inline2372__ssa_v4: pl.Scalar[pl.INDEX] = pl.max(pl.const(4, pl.INDEX) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, 0)
                            qk_run_hi_inline2330__ssa_v4: pl.Scalar[pl.INDEX] = pl.min((pl.const(4, pl.INDEX) + pl.const(1, pl.INDEX)) * pl.const(32, pl.INDEX) - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                            if qk_run_hi_inline2330__ssa_v4 > qk_run_lo_inline2372__ssa_v4:
                                qk_run_raw_inline2357__ssa_v4: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v4])
                                qk_run_src_inline2384__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__ssa_v4, pl.INDEX), 0), pl.INDEX)
                                qk_kv_inline2415__ssa_v9: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__phi_v8, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v4, 0], [qk_run_src_inline2384__ssa_v4, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v4 - qk_run_lo_inline2372__ssa_v4, 512], transpose=False)
                                qk_kv_inline2415__phi_v10: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v9)
                            else:
                                qk_kv_inline2415__phi_v10: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__phi_v8)
                            qk_tail_n_inline2329__ssa_v0: pl.Scalar[pl.INDEX] = qk_win_rows_inline2406__ssa_v0 - qk_run_rows_inline2332__ssa_v0
                            if qk_tail_n_inline2329__ssa_v0 > 0:
                                qk_kv_inline2415__ssa_v11: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__phi_v10, ori_kv_flat_inline2344__ssa_v1, [qk_run_rows_inline2332__ssa_v0, 0], [0, 0], [128, 512], valid_shape=[qk_tail_n_inline2329__ssa_v0, 512], transpose=False)
                                qk_kv_inline2415__phi_v12: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v11)
                            else:
                                qk_kv_inline2415__phi_v12: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__phi_v10)
                            qk_kv_inline2415__phi_v13: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__phi_v12)
                        else:
                            qk_kv_inline2415__phi_v13: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v0)
                        for qk_r_inline2390__idx_v0, (qk_kv_inline2415__iter_v14,) in pl.range(qk_win_rows_inline2406__ssa_v0, 128, init_values=(qk_kv_inline2415__phi_v13,)):
                            qk_cmp_k_inline2328__ssa_v0: pl.Scalar[pl.INDEX] = qk_s0_inline2418__ssa_v0 + qk_r_inline2390__idx_v0 - 128
                            if qk_cmp_k_inline2328__ssa_v0 < 512:
                                qk_ridx_inline2377__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices_inline2383__rv_v2, [qk_t_inline2411__ssa_v0, qk_cmp_k_inline2328__ssa_v0])
                                if pl.cast(qk_ridx_inline2377__ssa_v0, pl.INDEX) >= 0:
                                    qk_slot_inline2388__ssa_v0: pl.Scalar[pl.INT32] = qk_ridx_inline2377__ssa_v0
                                    qk_cblk_inline2327__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.tensor.read(cmp_block_table__ssa_v0, [qk_b_inline2371__ssa_v0, pl.cast(qk_slot_inline2388__ssa_v0, pl.INDEX) // 32]), pl.INDEX)
                                    qk_csrc_inline2400__ssa_v0: pl.Scalar[pl.INDEX] = qk_cblk_inline2327__ssa_v0 * 32 + pl.cast(qk_slot_inline2388__ssa_v0, pl.INDEX) % 32
                                    qk_kv_inline2415__ssa_v16: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__iter_v14, cmp_kv_flat_inline2401__ssa_v0, [qk_r_inline2390__idx_v0, 0], [qk_csrc_inline2400__ssa_v0, 0], [1, 512], transpose=False)
                                    qk_kv_inline2415__phi_v18: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v16)
                                else:
                                    qk_kv_inline2415__ssa_v17: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__iter_v14, ori_kv_flat_inline2344__ssa_v1, [qk_r_inline2390__idx_v0, 0], [0, 0], [1, 512], transpose=False)
                                    qk_kv_inline2415__phi_v18: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v17)
                                qk_kv_inline2415__phi_v20: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__phi_v18)
                            else:
                                qk_kv_inline2415__ssa_v19: pl.Tensor[[128, 512], pl.BF16] = pl.tensor.gather_row(qk_kv_inline2415__iter_v14, ori_kv_flat_inline2344__ssa_v1, [qk_r_inline2390__idx_v0, 0], [0, 0], [1, 512], transpose=False)
                                qk_kv_inline2415__phi_v20: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__ssa_v19)
                            qk_kv_inline2415__rv_v15: pl.Tensor[[128, 512], pl.BF16] = pl.yield_(qk_kv_inline2415__phi_v20)
                        for qk_hb_inline2336__idx_v0, (sparse_blk_li_inline2405__iter_v3, sparse_blk_mi_inline2404__iter_v3, sparse_blk_oi_inline2398__iter_v3) in pl.pipeline(2, stage=2, init_values=(sparse_blk_li_inline2405__iter_v1, sparse_blk_mi_inline2404__iter_v1, sparse_blk_oi_inline2398__iter_v1)):
                            qk_h0_inline2393__ssa_v0: pl.Scalar[pl.INDEX] = qk_hb_inline2336__idx_v0 * 32
                            qk_head_row_inline2326__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2411__ssa_v0 * 64 + qk_h0_inline2393__ssa_v0
                            qk_q_tile_inline2338__ssa_v0: pl.Tensor[[32, 512], pl.BF16] = pl.tensor.slice(q_flat_inline2355__ssa_v0, [32, 512], [qk_head_row_inline2326__ssa_v0, 0])
                            qk_raw_inline2397__ssa_v0: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.matmul(qk_q_tile_inline2338__ssa_v0, qk_kv_inline2415__rv_v15, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                            qk_scaled_inline2416__ssa_v0: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.muls(qk_raw_inline2397__ssa_v0, 0.044194173824159223)
                            qk_scores_inline2325__ssa_v0: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.col_expand_add(qk_scaled_inline2416__ssa_v0, qk_bias_row_inline2419__ssa_v0)
                            qk_mi_inline2360__ssa_v0: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.row_max(qk_scores_inline2325__ssa_v0)
                            qk_exp_inline2324__ssa_v0: pl.Tensor[[32, 128], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(qk_scores_inline2325__ssa_v0, qk_mi_inline2360__ssa_v0))
                            qk_li_inline2323__ssa_v0: pl.Tensor[[32, 1], pl.FP32] = pl.tensor.row_sum(qk_exp_inline2324__ssa_v0)
                            qk_exp_bf16_inline2322__ssa_v0: pl.Tensor[[32, 128], pl.BF16] = pl.tensor.cast(qk_exp_inline2324__ssa_v0, target_type=pl.BF16, mode='rint')
                            qk_oi_inline2335__ssa_v0: pl.Tensor[[32, 512], pl.FP32] = pl.tensor.matmul(qk_exp_bf16_inline2322__ssa_v0, qk_kv_inline2415__rv_v15, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                            qk_h_idx_inline2320__ssa_v0: pl.Scalar[pl.INDEX] = qk_hb_inline2336__idx_v0 * 2 + 0
                            qk_r0_inline2343__ssa_v0: pl.Scalar[pl.INDEX] = pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX)
                            qk_blk_base_inline2319__ssa_v0: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2320__ssa_v0 * 5 * 16
                            qk_row_inline2356__ssa_v0: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v0 + qk_sb_inline2374__ssa_v0 * 16
                            sparse_blk_mi_inline2404__ssa_v5: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32] = pl.tensor.assemble(sparse_blk_mi_inline2404__iter_v3, pl.tensor.slice(qk_mi_inline2360__ssa_v0, [16, 1], [qk_r0_inline2343__ssa_v0, 0]), [qk_row_inline2356__ssa_v0, 0])
                            sparse_blk_li_inline2405__ssa_v5: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32] = pl.tensor.assemble(sparse_blk_li_inline2405__iter_v3, pl.tensor.slice(qk_li_inline2323__ssa_v0, [16, 1], [qk_r0_inline2343__ssa_v0, 0]), [qk_row_inline2356__ssa_v0, 0])
                            sparse_blk_oi_inline2398__ssa_v5: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32] = pl.tensor.assemble(sparse_blk_oi_inline2398__iter_v3, pl.tensor.slice(qk_oi_inline2335__ssa_v0, [16, 512], [qk_r0_inline2343__ssa_v0, 0]), [qk_row_inline2356__ssa_v0, 0])
                            qk_h_idx_inline2320__ssa_v1: pl.Scalar[pl.INDEX] = qk_hb_inline2336__idx_v0 * 2 + 1
                            qk_r0_inline2343__ssa_v1: pl.Scalar[pl.INDEX] = pl.const(1, pl.INDEX) * pl.const(16, pl.INDEX)
                            qk_blk_base_inline2319__ssa_v1: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2320__ssa_v1 * 5 * 16
                            qk_row_inline2356__ssa_v1: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v1 + qk_sb_inline2374__ssa_v0 * 16
                            sparse_blk_mi_inline2404__ssa_v6: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32] = pl.tensor.assemble(sparse_blk_mi_inline2404__ssa_v5, pl.tensor.slice(qk_mi_inline2360__ssa_v0, [16, 1], [qk_r0_inline2343__ssa_v1, 0]), [qk_row_inline2356__ssa_v1, 0])
                            sparse_blk_li_inline2405__ssa_v6: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32] = pl.tensor.assemble(sparse_blk_li_inline2405__ssa_v5, pl.tensor.slice(qk_li_inline2323__ssa_v0, [16, 1], [qk_r0_inline2343__ssa_v1, 0]), [qk_row_inline2356__ssa_v1, 0])
                            sparse_blk_oi_inline2398__ssa_v6: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32] = pl.tensor.assemble(sparse_blk_oi_inline2398__ssa_v5, pl.tensor.slice(qk_oi_inline2335__ssa_v0, [16, 512], [qk_r0_inline2343__ssa_v1, 0]), [qk_row_inline2356__ssa_v1, 0])
                            sparse_blk_li_inline2405__rv_v4, sparse_blk_mi_inline2404__rv_v4, sparse_blk_oi_inline2398__rv_v4 = pl.yield_(sparse_blk_li_inline2405__ssa_v6, sparse_blk_mi_inline2404__ssa_v6, sparse_blk_oi_inline2398__ssa_v6)
                        sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7, sparse_blk_oi_inline2398__phi_v10 = pl.yield_(sparse_blk_li_inline2405__rv_v4, sparse_blk_mi_inline2404__rv_v4, sparse_blk_oi_inline2398__rv_v4)
                    else:
                        qk_oi_zero_inline2318__ssa_v0: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                        for qk_h_idx_inline2317__idx_v0, (sparse_blk_oi_inline2398__iter_v7,) in pl.range(4, init_values=(sparse_blk_oi_inline2398__iter_v1,)):
                            qk_blk_base_inline2319__ssa_v2: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2317__idx_v0 * 5 * 16
                            qk_row_inline2356__ssa_v2: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v2 + qk_sb_inline2374__ssa_v0 * 16
                            for qk_hr_inline2346__idx_v0 in pl.range(16):
                                pl.tensor.write(sparse_blk_mi_inline2404__iter_v1, [qk_row_inline2356__ssa_v2 + qk_hr_inline2346__idx_v0, 0], -3.0000000000000001e+38)
                                pl.tensor.write(sparse_blk_li_inline2405__iter_v1, [qk_row_inline2356__ssa_v2 + qk_hr_inline2346__idx_v0, 0], 0.0)
                            sparse_blk_oi_inline2398__ssa_v9: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32] = pl.tensor.assemble(sparse_blk_oi_inline2398__iter_v7, qk_oi_zero_inline2318__ssa_v0, [qk_row_inline2356__ssa_v2, 0])
                            sparse_blk_oi_inline2398__rv_v8: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32] = pl.yield_(sparse_blk_oi_inline2398__ssa_v9)
                        sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7, sparse_blk_oi_inline2398__phi_v10 = pl.yield_(sparse_blk_li_inline2405__iter_v1, sparse_blk_mi_inline2404__iter_v1, sparse_blk_oi_inline2398__rv_v8)
                    sparse_blk_li_inline2405__rv_v2, sparse_blk_mi_inline2404__rv_v2, sparse_blk_oi_inline2398__rv_v2 = pl.yield_(sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7, sparse_blk_oi_inline2398__phi_v10)
            rope_cos_il_inline2316__ssa_v0: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            rope_sin_signed_inline2315__ssa_v0: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            rope_swap_idx_inline2314__ssa_v0: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.create([16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True) as rope_tid_inline2402__ssa_v0:
                sw_ones_inline2420__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.full([16, 64], dtype=pl.FP32, value=1.0)
                sw_idx_f_inline2366__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round')
                sw_col_inline2313__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.col_expand_mul(sw_ones_inline2420__ssa_v0, sw_idx_f_inline2366__ssa_v0)
                sw_dup_i32_inline2312__ssa_v0: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.cast(pl.tensor.muls(sw_col_inline2313__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc')
                sw_dup_f_inline2311__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.cast(sw_dup_i32_inline2312__ssa_v0, target_type=pl.FP32, mode='round')
                sw_lane_inline2310__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(sw_col_inline2313__ssa_v0, pl.tensor.muls(sw_dup_f_inline2311__ssa_v0, 2.0))
                sw_swap_f_inline2363__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.sub(pl.tensor.adds(sw_col_inline2313__ssa_v0, 1.0), pl.tensor.muls(sw_lane_inline2310__ssa_v0, 2.0))
                rope_swap_idx_inline2314__ssa_v1: pl.Tensor[[16, 64], pl.INT32] = pl.tensor.assemble(rope_swap_idx_inline2314__ssa_v0, pl.tensor.cast(sw_swap_f_inline2363__ssa_v0, target_type=pl.INT32, mode='round'), [0, 0])
                cs_ones_inline2309__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.full([8, 64], dtype=pl.FP32, value=1.0)
                cs_idx_f_inline2345__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(pl.tensor.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False), target_type=pl.FP32, mode='round')
                cs_col_inline2308__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.col_expand_mul(cs_ones_inline2309__ssa_v0, cs_idx_f_inline2345__ssa_v0)
                cs_dup_i32_inline2334__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(pl.tensor.muls(cs_col_inline2308__ssa_v0, 0.5), target_type=pl.INT32, mode='trunc')
                cs_dup_f_inline2307__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.cast(cs_dup_i32_inline2334__ssa_v0, target_type=pl.FP32, mode='round')
                cs_dup_idx_inline2396__ssa_v0: pl.Tensor[[8, 64], pl.INT32] = pl.tensor.cast(cs_dup_f_inline2307__ssa_v0, target_type=pl.INT32, mode='round')
                cs_lane_inline2352__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.sub(cs_col_inline2308__ssa_v0, pl.tensor.muls(cs_dup_f_inline2307__ssa_v0, 2.0))
                cs_sign_inline2306__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.neg(pl.tensor.subs(pl.tensor.muls(cs_lane_inline2352__ssa_v0, 2.0), 1.0))
                for cs_rb_inline2341__idx_v0, (rope_cos_il_inline2316__iter_v1, rope_sin_signed_inline2315__iter_v1) in pl.range(rope_cs_blocks_inline2380__ssa_v0, init_values=(rope_cos_il_inline2316__ssa_v0, rope_sin_signed_inline2315__ssa_v0)):
                    cs_t0_inline2305__ssa_v0: pl.Scalar[pl.INDEX] = cs_rb_inline2341__idx_v0 * 8
                    cs_cos_inline2304__ssa_v0: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.cast(pl.tensor.slice(freqs_cos_local__ssa_v0, [8, 32], [cs_t0_inline2305__ssa_v0, 0]), target_type=pl.FP32, mode='round')
                    cs_sin_inline2303__ssa_v0: pl.Tensor[[8, 32], pl.FP32] = pl.tensor.cast(pl.tensor.slice(freqs_sin_local__ssa_v0, [8, 32], [cs_t0_inline2305__ssa_v0, 0]), target_type=pl.FP32, mode='round')
                    rope_cos_il_inline2316__ssa_v3: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(rope_cos_il_inline2316__iter_v1, pl.tensor.gather(cs_cos_inline2304__ssa_v0, cs_dup_idx_inline2396__ssa_v0, dim=-1), [cs_t0_inline2305__ssa_v0, 0])
                    cs_sin_il_inline2409__ssa_v0: pl.Tensor[[8, 64], pl.FP32] = pl.tensor.gather(cs_sin_inline2303__ssa_v0, cs_dup_idx_inline2396__ssa_v0, dim=-1)
                    rope_sin_signed_inline2315__ssa_v3: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(rope_sin_signed_inline2315__iter_v1, pl.tensor.mul(cs_sin_il_inline2409__ssa_v0, cs_sign_inline2306__ssa_v0), [cs_t0_inline2305__ssa_v0, 0])
                    rope_cos_il_inline2316__rv_v2, rope_sin_signed_inline2315__rv_v2 = pl.yield_(rope_cos_il_inline2316__ssa_v3, rope_sin_signed_inline2315__ssa_v3)
            sparse_blk_mi_inline1234__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32] = sparse_blk_mi_inline2404__rv_v2
            sparse_blk_li_inline1283__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32] = sparse_blk_li_inline2405__rv_v2
            sparse_blk_oi_inline1233__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32] = sparse_blk_oi_inline2398__rv_v2
            rope_cos_il_inline1232__ssa_v0: pl.Tensor[[256, 64], pl.FP32] = rope_cos_il_inline2316__rv_v2
            rope_sin_signed_inline1231__ssa_v0: pl.Tensor[[256, 64], pl.FP32] = rope_sin_signed_inline2315__rv_v2
            rope_swap_idx_inline1230__ssa_v0: pl.Tensor[[16, 64], pl.INT32] = rope_swap_idx_inline2314__ssa_v1
            qk_tid_inline1229__ssa_v0: pl.Scalar[pl.TASK_ID] = qk_tid_inline2349__ssa_v0
            attn_rope_tid_inline1294__ssa_v0: pl.Scalar[pl.TASK_ID] = rope_tid_inline2402__ssa_v0
            attention_grouped_inline1276__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.create([2048, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            pack_work_count_inline1228__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline1251__ssa_v0 // 8 * 4
            with pl.spmd(48, name_hint="csa_merge_pack_publish_spmd", deps=[qk_tid_inline1229__ssa_v0, attn_rope_tid_inline1294__ssa_v0]) as publish_tid_inline1260__ssa_v0:
                worker_inline1227__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for pack_work_inline1270__idx_v0, (attention_grouped_inline1276__iter_v1,) in pl.range(worker_inline1227__ssa_v0, pack_work_count_inline1228__ssa_v0, 48, init_values=(attention_grouped_inline1276__ssa_v0,)):
                    token_block_inline1226__ssa_v0: pl.Scalar[pl.INDEX] = pack_work_inline1270__idx_v0 // 4
                    m_h_idx_inline1303__ssa_v0: pl.Scalar[pl.INDEX] = pack_work_inline1270__idx_v0 - token_block_inline1226__ssa_v0 * 4
                    m_t0_inline1278__ssa_v0: pl.Scalar[pl.INDEX] = token_block_inline1226__ssa_v0 * 8
                    m_h0_inline1224__ssa_v0: pl.Scalar[pl.INDEX] = m_h_idx_inline1303__ssa_v0 * 16
                    global_group0_inline1314__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 // 8
                    destination_rank_inline1223__ssa_v0: pl.Scalar[pl.INDEX] = global_group0_inline1314__ssa_v0 // 4
                    local_group0_inline1222__ssa_v0: pl.Scalar[pl.INDEX] = global_group0_inline1314__ssa_v0 - destination_rank_inline1223__ssa_v0 * 4
                    for m_dt_inline1287__idx_v0, (attention_grouped_inline1276__iter_v3,) in pl.range(8, init_values=(attention_grouped_inline1276__iter_v1,)):
                        m_t_inline1290__ssa_v0: pl.Scalar[pl.INDEX] = m_t0_inline1278__ssa_v0 + m_dt_inline1287__idx_v0
                        m_idx_inline1289__ssa_v0: pl.Scalar[pl.INDEX] = m_t_inline1290__ssa_v0 * 4 + m_h_idx_inline1303__ssa_v0
                        m_blk_base_inline1306__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline1289__ssa_v0 * 5 * 16
                        m_mi_inline1221__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi_inline1234__ssa_v0, [16, 1], [m_blk_base_inline1306__ssa_v0, 0])
                        m_li_inline1302__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li_inline1283__ssa_v0, [16, 1], [m_blk_base_inline1306__ssa_v0, 0])
                        m_oi_inline1220__ssa_v0: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi_inline1233__ssa_v0, [16, 512], [m_blk_base_inline1306__ssa_v0, 0])
                        for m_sb_inline1219__idx_v0, (m_li_inline1302__iter_v1, m_mi_inline1221__iter_v1, m_oi_inline1220__iter_v1) in pl.pipeline(1, 5, stage=2, init_values=(m_li_inline1302__ssa_v0, m_mi_inline1221__ssa_v0, m_oi_inline1220__ssa_v0)):
                            m_row_inline1218__ssa_v0: pl.Scalar[pl.INDEX] = m_blk_base_inline1306__ssa_v0 + m_sb_inline1219__idx_v0 * 16
                            m_cur_mi_inline1217__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_mi_inline1234__ssa_v0, [16, 1], [m_row_inline1218__ssa_v0, 0])
                            m_cur_li_inline1293__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.slice(sparse_blk_li_inline1283__ssa_v0, [16, 1], [m_row_inline1218__ssa_v0, 0])
                            m_cur_oi_inline1216__ssa_v0: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(sparse_blk_oi_inline1233__ssa_v0, [16, 512], [m_row_inline1218__ssa_v0, 0])
                            m_mi_new_inline1215__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.maximum(m_mi_inline1221__iter_v1, m_cur_mi_inline1217__ssa_v0)
                            m_alpha_inline1214__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_mi_inline1221__iter_v1, m_mi_new_inline1215__ssa_v0))
                            m_beta_inline1213__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(m_cur_mi_inline1217__ssa_v0, m_mi_new_inline1215__ssa_v0))
                            m_li_inline1302__ssa_v3: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(m_alpha_inline1214__ssa_v0, m_li_inline1302__iter_v1), pl.tensor.mul(m_beta_inline1213__ssa_v0, m_cur_li_inline1293__ssa_v0))
                            m_oi_inline1220__ssa_v3: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(m_oi_inline1220__iter_v1, m_alpha_inline1214__ssa_v0), pl.tensor.row_expand_mul(m_cur_oi_inline1216__ssa_v0, m_beta_inline1213__ssa_v0))
                            m_mi_inline1221__ssa_v3: pl.Tensor[[16, 1], pl.FP32] = m_mi_new_inline1215__ssa_v0
                            m_li_inline1302__rv_v2, m_mi_inline1221__rv_v2, m_oi_inline1220__rv_v2 = pl.yield_(m_li_inline1302__ssa_v3, m_mi_inline1221__ssa_v3, m_oi_inline1220__ssa_v3)
                        n_sink_bias_inline1266__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(attn_sink__ssa_v0, [16], [m_h0_inline1224__ssa_v0]), [16, 1])
                        n_sink_tile_inline1212__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(pl.tensor.sub(m_mi_inline1221__rv_v2, m_mi_inline1221__rv_v2), n_sink_bias_inline1266__ssa_v0)
                        n_denom_inline1254__ssa_v0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(m_li_inline1302__rv_v2, pl.tensor.exp(pl.tensor.sub(n_sink_tile_inline1212__ssa_v0, m_mi_inline1221__rv_v2)))
                        n_full_inline1211__ssa_v0: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.slice(pl.tensor.row_expand_div(m_oi_inline1220__rv_v2, n_denom_inline1254__ssa_v0), [16, 512], [0, 0])
                        n_bf16_inline1210__ssa_v0: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.cast(n_full_inline1211__ssa_v0, target_type=pl.BF16, mode='rint')
                        m_rope_inline1208__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(n_full_inline1211__ssa_v0, [16, 64], [0, 448])
                        m_cos_il_inline1286__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_cos_il_inline1232__ssa_v0, [1, 64], [m_t_inline1290__ssa_v0, 0])
                        m_sin_signed_inline1225__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(rope_sin_signed_inline1231__ssa_v0, [1, 64], [m_t_inline1290__ssa_v0, 0])
                        m_swapped_inline1207__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.gather(m_rope_inline1208__ssa_v0, pl.tensor.slice(rope_swap_idx_inline1230__ssa_v0, [16, 64], [0, 0]), dim=-1)
                        m_rot_inline1206__ssa_v0: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.add(pl.tensor.col_expand_mul(m_rope_inline1208__ssa_v0, m_cos_il_inline1286__ssa_v0), pl.tensor.col_expand_mul(m_swapped_inline1207__ssa_v0, m_sin_signed_inline1225__ssa_v0))
                        n_rope_bf16_inline1205__ssa_v0: pl.Tensor[[16, 64], pl.BF16] = pl.tensor.cast(m_rot_inline1206__ssa_v0, target_type=pl.BF16, mode='rint')
                        n_full_bf16_inline1285__ssa_v0: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.concat(pl.tensor.slice(n_bf16_inline1210__ssa_v0, [16, 448], [0, 0]), n_rope_bf16_inline1205__ssa_v0)
                        n_head_inline1244__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 0
                        source_row_inline1238__ssa_v0: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v0 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v0: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v0 % 8 * 512
                        attention_grouped_inline1276__ssa_v5: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__iter_v3, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [0, 0]), [source_row_inline1238__ssa_v0, source_col_inline1204__ssa_v0])
                        n_head_inline1244__ssa_v1: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 1
                        source_row_inline1238__ssa_v1: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v1 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v1: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v1 % 8 * 512
                        attention_grouped_inline1276__ssa_v6: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v5, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [1, 0]), [source_row_inline1238__ssa_v1, source_col_inline1204__ssa_v1])
                        n_head_inline1244__ssa_v2: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 2
                        source_row_inline1238__ssa_v2: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v2 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v2: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v2 % 8 * 512
                        attention_grouped_inline1276__ssa_v7: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v6, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [2, 0]), [source_row_inline1238__ssa_v2, source_col_inline1204__ssa_v2])
                        n_head_inline1244__ssa_v3: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 3
                        source_row_inline1238__ssa_v3: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v3 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v3: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v3 % 8 * 512
                        attention_grouped_inline1276__ssa_v8: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v7, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [3, 0]), [source_row_inline1238__ssa_v3, source_col_inline1204__ssa_v3])
                        n_head_inline1244__ssa_v4: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 4
                        source_row_inline1238__ssa_v4: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v4 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v4: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v4 % 8 * 512
                        attention_grouped_inline1276__ssa_v9: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v8, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [4, 0]), [source_row_inline1238__ssa_v4, source_col_inline1204__ssa_v4])
                        n_head_inline1244__ssa_v5: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 5
                        source_row_inline1238__ssa_v5: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v5 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v5: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v5 % 8 * 512
                        attention_grouped_inline1276__ssa_v10: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v9, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [5, 0]), [source_row_inline1238__ssa_v5, source_col_inline1204__ssa_v5])
                        n_head_inline1244__ssa_v6: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 6
                        source_row_inline1238__ssa_v6: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v6 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v6: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v6 % 8 * 512
                        attention_grouped_inline1276__ssa_v11: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v10, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [6, 0]), [source_row_inline1238__ssa_v6, source_col_inline1204__ssa_v6])
                        n_head_inline1244__ssa_v7: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 7
                        source_row_inline1238__ssa_v7: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v7 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v7: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v7 % 8 * 512
                        attention_grouped_inline1276__ssa_v12: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v11, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [7, 0]), [source_row_inline1238__ssa_v7, source_col_inline1204__ssa_v7])
                        n_head_inline1244__ssa_v8: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 8
                        source_row_inline1238__ssa_v8: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v8 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v8: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v8 % 8 * 512
                        attention_grouped_inline1276__ssa_v13: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v12, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [8, 0]), [source_row_inline1238__ssa_v8, source_col_inline1204__ssa_v8])
                        n_head_inline1244__ssa_v9: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 9
                        source_row_inline1238__ssa_v9: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v9 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v9: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v9 % 8 * 512
                        attention_grouped_inline1276__ssa_v14: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v13, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [9, 0]), [source_row_inline1238__ssa_v9, source_col_inline1204__ssa_v9])
                        n_head_inline1244__ssa_v10: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 10
                        source_row_inline1238__ssa_v10: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v10 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v10: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v10 % 8 * 512
                        attention_grouped_inline1276__ssa_v15: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v14, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [10, 0]), [source_row_inline1238__ssa_v10, source_col_inline1204__ssa_v10])
                        n_head_inline1244__ssa_v11: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 11
                        source_row_inline1238__ssa_v11: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v11 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v11: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v11 % 8 * 512
                        attention_grouped_inline1276__ssa_v16: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v15, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [11, 0]), [source_row_inline1238__ssa_v11, source_col_inline1204__ssa_v11])
                        n_head_inline1244__ssa_v12: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 12
                        source_row_inline1238__ssa_v12: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v12 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v12: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v12 % 8 * 512
                        attention_grouped_inline1276__ssa_v17: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v16, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [12, 0]), [source_row_inline1238__ssa_v12, source_col_inline1204__ssa_v12])
                        n_head_inline1244__ssa_v13: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 13
                        source_row_inline1238__ssa_v13: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v13 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v13: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v13 % 8 * 512
                        attention_grouped_inline1276__ssa_v18: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v17, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [13, 0]), [source_row_inline1238__ssa_v13, source_col_inline1204__ssa_v13])
                        n_head_inline1244__ssa_v14: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 14
                        source_row_inline1238__ssa_v14: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v14 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v14: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v14 % 8 * 512
                        attention_grouped_inline1276__ssa_v19: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v18, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [14, 0]), [source_row_inline1238__ssa_v14, source_col_inline1204__ssa_v14])
                        n_head_inline1244__ssa_v15: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 15
                        source_row_inline1238__ssa_v15: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v15 // 8 * 256 + m_t_inline1290__ssa_v0
                        source_col_inline1204__ssa_v15: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v15 % 8 * 512
                        attention_grouped_inline1276__ssa_v20: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_grouped_inline1276__ssa_v19, pl.tensor.slice(n_full_bf16_inline1285__ssa_v0, [1, 512], [15, 0]), [source_row_inline1238__ssa_v15, source_col_inline1204__ssa_v15])
                        attention_grouped_inline1276__rv_v4: pl.Tensor[[2048, 4096], pl.BF16] = pl.yield_(attention_grouped_inline1276__ssa_v20)
                    source_row_inline1238__ssa_v16: pl.Scalar[pl.INDEX] = (global_group0_inline1314__ssa_v0 + 0) * 256 + m_t0_inline1278__ssa_v0
                    target_row_inline1203__ssa_v0: pl.Scalar[pl.INDEX] = (local_group0_inline1222__ssa_v0 + 0) * 512 + pl.cast(tp_rank__ssa_v0, pl.INDEX) * 256 + m_t0_inline1278__ssa_v0
                    pld.tensor.put(attention_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + destination_rank_inline1223__ssa_v0, attention_grouped_inline1276__rv_v4, [target_row_inline1203__ssa_v0, 0], [source_row_inline1238__ssa_v16, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                    source_row_inline1238__ssa_v17: pl.Scalar[pl.INDEX] = (global_group0_inline1314__ssa_v0 + 1) * 256 + m_t0_inline1278__ssa_v0
                    target_row_inline1203__ssa_v1: pl.Scalar[pl.INDEX] = (local_group0_inline1222__ssa_v0 + 1) * 512 + pl.cast(tp_rank__ssa_v0, pl.INDEX) * 256 + m_t0_inline1278__ssa_v0
                    pld.tensor.put(attention_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + destination_rank_inline1223__ssa_v0, attention_grouped_inline1276__rv_v4, [target_row_inline1203__ssa_v1, 0], [source_row_inline1238__ssa_v17, 0], [8, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                    attention_grouped_inline1276__rv_v2: pl.Tensor[[2048, 4096], pl.BF16] = pl.yield_(attention_grouped_inline1276__rv_v4)
                for peer_tp_inline1273__idx_v0 in pl.range(2):
                    if peer_tp_inline1273__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(attention_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1273__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
            # Finish a non-overlapping producer-fused exchange and release its window.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_group_a2a_wait", deps=[publish_tid_inline1260__ssa_v0]) as wait_tid_inline2436__ssa_v0:
                expected_inline2426__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(48, pl.INT32)
                for source_tp_inline2427__idx_v0 in pl.range(2):
                    if source_tp_inline2427__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.wait(attention_signal__ssa_v0, [source_tp_inline2427__idx_v0, 0], expected_inline2426__ssa_v0, cmp=1)
            group_t_inline2428__ssa_v0: pl.Scalar[pl.INDEX] = 512
            with pl.spmd(48, name_hint="o_group_a2a_gather_spmd", deps=[wait_tid_inline2436__ssa_v0]) as gather_tid_inline2431__ssa_v0:
                worker_inline2433__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for local_group_inline2435__idx_v0, (attention_local_flat_inline1292__iter_v1,) in pl.range(4, init_values=(attention_local_flat_inline1292__ssa_v0,)):
                    group_base_row_inline2437__ssa_v0: pl.Scalar[pl.INDEX] = local_group_inline2435__idx_v0 * 512
                    for group_row_inline2432__idx_v0, (attention_local_flat_inline1292__iter_v3,) in pl.range(worker_inline2433__ssa_v0, group_t_inline2428__ssa_v0, 48, init_values=(attention_local_flat_inline1292__iter_v1,)):
                        copy_row_inline2430__ssa_v0: pl.Scalar[pl.INDEX] = group_base_row_inline2437__ssa_v0 + group_row_inline2432__idx_v0
                        attention_local_flat_inline1292__ssa_v5: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.assemble(attention_local_flat_inline1292__iter_v3, pl.tensor.slice(attention_window__ssa_v0, [1, 4096], [copy_row_inline2430__ssa_v0, 0]), [copy_row_inline2430__ssa_v0, 0])
                        attention_local_flat_inline1292__rv_v4: pl.Tensor[[2048, 4096], pl.BF16] = pl.yield_(attention_local_flat_inline1292__ssa_v5)
                    attention_local_flat_inline1292__rv_v2: pl.Tensor[[2048, 4096], pl.BF16] = pl.yield_(attention_local_flat_inline1292__rv_v4)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_group_a2a_complete", deps=[gather_tid_inline2431__ssa_v0]):
                completion_anchor_inline2439__ssa_v0: pl.Scalar[pl.BF16] = pl.tensor.read(attention_local_flat_inline1292__rv_v2, [0, 0])
                for peer_tp_inline2438__idx_v0 in pl.range(2):
                    if peer_tp_inline2438__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(attention_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline2438__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
                completion_expected_inline2425__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(pl.cast(48, pl.INDEX) + 1, pl.INT32)
                for source_tp_inline2424__idx_v0 in pl.range(2):
                    if source_tp_inline2424__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.wait(attention_signal__ssa_v0, [source_tp_inline2424__idx_v0, 0], completion_expected_inline2425__ssa_v0, cmp=1)
                reset_value_inline2434__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(-completion_expected_inline2425__ssa_v0, pl.INT32)
                self_rank_inline2423__ssa_v0: pl.Scalar[pl.INT32] = group_base__ssa_v0 + tp_rank__ssa_v0
                for source_tp_inline2429__idx_v0 in pl.range(2):
                    if source_tp_inline2429__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(attention_signal__ssa_v0, self_rank_inline2423__ssa_v0, [source_tp_inline2429__idx_v0, 0], reset_value_inline2434__ssa_v0, op=0)
                pl.tensor.write(attention_local_flat_inline1292__rv_v2, [0, 0], completion_anchor_inline2439__ssa_v0)
            attention_local_flat_inline1292__ssa_v6: pl.Tensor[[2048, 4096], pl.BF16] = attention_local_flat_inline1292__rv_v2
            attention_signal__ssa_v1: pld.DistributedTensor[[2, 1], pl.INT32] = attention_signal__ssa_v0
            attention_local_groups_inline1321__ssa_v0: pl.Tensor[[4, 512, 4096], pl.BF16] = pl.tensor.reshape(attention_local_flat_inline1292__ssa_v6, [4, 512, 4096])
            # Project O-B tiles directly into their ReduceScatter owner windows.
            group_t_inline2530__ssa_v0: pl.Scalar[pl.INDEX] = 512
            o_a_rows_inline2513__ssa_v0: pl.Scalar[pl.INDEX] = (group_t_inline2530__ssa_v0 + 128 - 1) // 128
            o_b_rows_inline2534__ssa_v0: pl.Scalar[pl.INDEX] = (group_t_inline2530__ssa_v0 + 128 - 1) // 128
            o_b_group_t_inline2487__ssa_v0: pl.Scalar[pl.INDEX] = o_b_rows_inline2534__ssa_v0 * 128
            owner_rows_inline2509__ssa_v0: pl.Scalar[pl.INDEX] = 16
            attn_2d_inline2548__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16] = pl.tensor.reshape(attention_local_groups_inline1321__ssa_v0, [2048, 4096])
            wo_a_flat_inline2521__ssa_v0: pl.Tensor[[4096, 4096], pl.BF16] = pl.tensor.reshape(wo_a__ssa_v0, [4096, 4096])
            publish_all_inline2525__ssa_v0: pl.Tensor[[512, 4096], pl.BF16] = pl.tensor.create([512, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            put_rows_inline2508__ssa_v0: pl.Scalar[pl.INDEX] = 32
            own_a_rows_inline2516__ssa_v0: pl.Scalar[pl.INDEX] = 2
            own_b_rows_inline2529__ssa_v0: pl.Scalar[pl.INDEX] = 2
            own_b_t_inline2531__ssa_v0: pl.Scalar[pl.INDEX] = own_b_rows_inline2529__ssa_v0 * 128
            own_quant_blocks_inline2490__ssa_v0: pl.Scalar[pl.INDEX] = 32
            own_pad_blocks_inline2511__ssa_v0: pl.Scalar[pl.INDEX] = (own_b_t_inline2531__ssa_v0 + 8 - 1) // 8
            own_act_rows_inline2505__ssa_v0: pl.Scalar[pl.INDEX] = 16
            for owner_inline2504__idx_v0, (publish_all_inline2525__iter_v1,) in pl.parallel(2, init_values=(publish_all_inline2525__ssa_v0,)):
                own_base_inline2502__ssa_v0: pl.Scalar[pl.INDEX] = owner_inline2504__idx_v0 * 256
                own_a_fp32_inline2500__ssa_v0: pl.Tensor[[256, 4096], pl.FP32] = pl.tensor.create([256, 4096], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                own_a_i8_inline2497__ssa_v0: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.create([256, 4096], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                own_scale_inline2494__ssa_v0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.create([4, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                own_b_i32_inline2528__ssa_v0: pl.Tensor[[256, 16384], pl.INT32] = pl.tensor.create([256, 16384], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                for local_group_inline2514__idx_v0, (own_a_fp32_inline2500__iter_v1, own_a_i8_inline2497__iter_v1, own_b_i32_inline2528__iter_v1, own_scale_inline2494__iter_v1) in pl.parallel(4, init_values=(own_a_fp32_inline2500__ssa_v0, own_a_i8_inline2497__ssa_v0, own_b_i32_inline2528__ssa_v0, own_scale_inline2494__ssa_v0)):
                    attention_row_inline2526__ssa_v0: pl.Scalar[pl.INDEX] = local_group_inline2514__idx_v0 * 512 + own_base_inline2502__ssa_v0
                    o_a_col_inline2496__ssa_v0: pl.Scalar[pl.INDEX] = local_group_inline2514__idx_v0 * 1024
                    for pa_unit_inline2493__ssa_v0 in pl.spmd(own_a_rows_inline2516__ssa_v0 * 8, name_hint="tp_o_a_spmd"):
                        pa_rb_inline2491__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline2493__ssa_v0 // 8
                        pa_nb_inline2499__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline2493__ssa_v0 - pa_rb_inline2491__ssa_v0 * 8
                        pa_t0_inline2489__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline2491__ssa_v0 * 128
                        pa_n0_inline2485__ssa_v0: pl.Scalar[pl.INDEX] = pa_nb_inline2499__ssa_v0 * 128
                        pa_rows_inline2532__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(128, 256 - pa_t0_inline2489__ssa_v0)
                        pa_src_inline2495__ssa_v0: pl.Scalar[pl.INDEX] = attention_row_inline2526__ssa_v0 + pa_t0_inline2489__ssa_v0
                        pa_wrow_inline2533__ssa_v0: pl.Scalar[pl.INDEX] = o_a_col_inline2496__ssa_v0 + pa_n0_inline2485__ssa_v0
                        pa_x0_inline2535__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline2532__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(attn_2d_inline2548__ssa_v0, [128, 256], [pa_src_inline2495__ssa_v0, 0], [pa_rows_inline2532__ssa_v0, 256])
                        pa_w0_inline2523__ssa_v0: pl.Tensor[[128, 256], pl.BF16] = pl.tensor.slice(wo_a_flat_inline2521__ssa_v0, [128, 256], [pa_wrow_inline2533__ssa_v0, 0])
                        pa_acc_inline2537__ssa_v0: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul(pa_x0_inline2535__ssa_v0, pa_w0_inline2523__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        for pa_k0_inline2561__idx_v0, (pa_acc_inline2537__iter_v1,) in pl.pipeline(256, 4096, 256, stage=2, init_values=(pa_acc_inline2537__ssa_v0,)):
                            pa_xk_inline2539__ssa_v0: pl.Tensor[[128, 256], pl.BF16, pl.TensorView(valid_shape=[pa_rows_inline2532__ssa_v0, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(attn_2d_inline2548__ssa_v0, [128, 256], [pa_src_inline2495__ssa_v0, pa_k0_inline2561__idx_v0], [pa_rows_inline2532__ssa_v0, 256])
                            pa_wk_inline2540__ssa_v0: pl.Tensor[[128, 256], pl.BF16] = pl.tensor.slice(wo_a_flat_inline2521__ssa_v0, [128, 256], [pa_wrow_inline2533__ssa_v0, pa_k0_inline2561__idx_v0])
                            pa_acc_inline2537__ssa_v3: pl.Tensor[[128, 128], pl.FP32] = pl.tensor.matmul_acc(pa_acc_inline2537__iter_v1, pa_xk_inline2539__ssa_v0, pa_wk_inline2540__ssa_v0, a_trans=False, b_trans=True)
                            pa_acc_inline2537__rv_v2: pl.Tensor[[128, 128], pl.FP32] = pl.yield_(pa_acc_inline2537__ssa_v3)
                        pa_valid_inline2542__ssa_v0: pl.Tensor[[128, 128], pl.FP32, pl.TensorView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.set_validshape(pa_acc_inline2537__rv_v2, pa_rows_inline2532__ssa_v0, 128)
                        own_a_fp32_inline2500__ssa_v3: pl.Tensor[[256, 4096], pl.FP32] = pl.tensor.assemble(own_a_fp32_inline2500__iter_v1, pa_valid_inline2542__ssa_v0, [pa_t0_inline2489__ssa_v0, pa_wrow_inline2533__ssa_v0])
                    for qz_worker_inline2544__ssa_v0 in pl.spmd(6, name_hint="tp_o_a_quant_spmd"):
                        for qz_blk_inline2546__idx_v0, (own_a_i8_inline2497__iter_v3, own_scale_inline2494__iter_v3) in pl.range(qz_worker_inline2544__ssa_v0, own_quant_blocks_inline2490__ssa_v0, 6, init_values=(own_a_i8_inline2497__iter_v1, own_scale_inline2494__iter_v1)):
                            qz_t_inline2547__ssa_v0: pl.Scalar[pl.INDEX] = qz_blk_inline2546__idx_v0 * 8
                            qz_rows_inline2501__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, 256 - qz_t_inline2547__ssa_v0)
                            qz_tile_inline2560__ssa_v0: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.slice(own_a_fp32_inline2500__ssa_v3, [8, 1024], [qz_t_inline2547__ssa_v0, o_a_col_inline2496__ssa_v0])
                            qz_amax_inline2524__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.row_max(pl.tensor.abs(qz_tile_inline2560__ssa_v0)), [1, 8])
                            qz_floor_inline2536__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=0.0001)
                            qz_amax_v1_inline2549__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.maximum(qz_floor_inline2536__ssa_v0, qz_amax_inline2524__ssa_v0)
                            qz_max_inline2512__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.full([1, 8], dtype=pl.FP32, value=127.0)
                            qz_sq_inline2527__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.div(qz_max_inline2512__ssa_v0, qz_amax_v1_inline2549__ssa_v0)
                            qz_sdq_inline2550__ssa_v0: pl.Tensor[[1, 8], pl.FP32] = pl.tensor.recip(qz_sq_inline2527__ssa_v0)
                            own_scale_inline2494__ssa_v5: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.assemble(own_scale_inline2494__iter_v3, pl.tensor.set_validshape(qz_sdq_inline2550__ssa_v0, 1, qz_rows_inline2501__ssa_v0), [local_group_inline2514__idx_v0, qz_t_inline2547__ssa_v0])
                            qz_sq_col_inline2519__ssa_v0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.reshape(qz_sq_inline2527__ssa_v0, [8, 1])
                            qz_scaled_inline2538__ssa_v0: pl.Tensor[[8, 1024], pl.FP32] = pl.tensor.row_expand_mul(qz_tile_inline2560__ssa_v0, qz_sq_col_inline2519__ssa_v0)
                            qz_i32_inline2551__ssa_v0: pl.Tensor[[8, 1024], pl.INT32] = pl.tensor.cast(qz_scaled_inline2538__ssa_v0, target_type=pl.INT32, mode='rint')
                            qz_f16_inline2552__ssa_v0: pl.Tensor[[8, 1024], pl.FP16] = pl.tensor.cast(qz_i32_inline2551__ssa_v0, target_type=pl.FP16, mode='round')
                            qz_i8_inline2503__ssa_v0: pl.Tensor[[8, 1024], pl.INT8] = pl.tensor.cast(qz_f16_inline2552__ssa_v0, target_type=pl.INT8, mode='trunc')
                            own_a_i8_inline2497__ssa_v5: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.assemble(own_a_i8_inline2497__iter_v3, pl.tensor.set_validshape(qz_i8_inline2503__ssa_v0, qz_rows_inline2501__ssa_v0, 1024), [qz_t_inline2547__ssa_v0, o_a_col_inline2496__ssa_v0])
                            own_a_i8_inline2497__rv_v4, own_scale_inline2494__rv_v4 = pl.yield_(own_a_i8_inline2497__ssa_v5, own_scale_inline2494__ssa_v5)
                        for qz_pad_inline2554__idx_v0, (own_a_i8_inline2497__iter_v6,) in pl.range(own_quant_blocks_inline2490__ssa_v0 + qz_worker_inline2544__ssa_v0, own_pad_blocks_inline2511__ssa_v0, 6, init_values=(own_a_i8_inline2497__rv_v4,)):
                            qz_pt_inline2558__ssa_v0: pl.Scalar[pl.INDEX] = qz_pad_inline2554__idx_v0 * 8
                            qz_prows_inline2562__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, own_b_t_inline2531__ssa_v0 - qz_pt_inline2558__ssa_v0)
                            qz_zero_inline2563__ssa_v0: pl.Tensor[[8, 1024], pl.FP16] = pl.tensor.full([8, 1024], dtype=pl.FP16, value=0.0)
                            qz_zero_i8_inline2492__ssa_v0: pl.Tensor[[8, 1024], pl.INT8] = pl.tensor.cast(qz_zero_inline2563__ssa_v0, target_type=pl.INT8, mode='trunc')
                            own_a_i8_inline2497__ssa_v8: pl.Tensor[[256, 4096], pl.INT8] = pl.tensor.assemble(own_a_i8_inline2497__iter_v6, pl.tensor.set_validshape(qz_zero_i8_inline2492__ssa_v0, qz_prows_inline2562__ssa_v0, 1024), [qz_pt_inline2558__ssa_v0, o_a_col_inline2496__ssa_v0])
                            own_a_i8_inline2497__rv_v7: pl.Tensor[[256, 4096], pl.INT8] = pl.yield_(own_a_i8_inline2497__ssa_v8)
                    for pb_unit_inline2483__ssa_v0 in pl.spmd(own_b_rows_inline2529__ssa_v0 * 8, name_hint="tp_o_b_spmd"):
                        pb_tb_inline2510__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline2483__ssa_v0 // 8
                        pb_db_inline2480__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline2483__ssa_v0 - pb_tb_inline2510__ssa_v0 * 8
                        pb_t0_inline2557__ssa_v0: pl.Scalar[pl.INDEX] = pb_tb_inline2510__ssa_v0 * 128
                        pb_d0_inline2479__ssa_v0: pl.Scalar[pl.INDEX] = pb_db_inline2480__ssa_v0 * 512
                        for pb_n0_inline2478__idx_v0, (own_b_i32_inline2528__iter_v3,) in pl.range(pb_d0_inline2479__ssa_v0, pb_d0_inline2479__ssa_v0 + 512, 256, init_values=(own_b_i32_inline2528__iter_v1,)):
                            pb_x0_inline2498__ssa_v0: pl.Tensor[[128, 256], pl.INT8] = pl.tensor.slice(own_a_i8_inline2497__rv_v7, [128, 256], [pb_t0_inline2557__ssa_v0, o_a_col_inline2496__ssa_v0])
                            pb_w0_inline2522__ssa_v0: pl.Tensor[[256, 256], pl.INT8] = pl.tensor.slice(wo_b__ssa_v0, [256, 256], [pb_n0_inline2478__idx_v0, o_a_col_inline2496__ssa_v0])
                            pb_acc_inline2507__ssa_v0: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul(pb_x0_inline2498__ssa_v0, pb_w0_inline2522__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.INT32)
                            for pb_k0_inline2477__idx_v0, (pb_acc_inline2507__iter_v1,) in pl.pipeline(256, 1024, 256, stage=2, init_values=(pb_acc_inline2507__ssa_v0,)):
                                pb_bk_inline2515__ssa_v0: pl.Scalar[pl.INDEX] = o_a_col_inline2496__ssa_v0 + pb_k0_inline2477__idx_v0
                                pb_xk_inline2520__ssa_v0: pl.Tensor[[128, 256], pl.INT8] = pl.tensor.slice(own_a_i8_inline2497__rv_v7, [128, 256], [pb_t0_inline2557__ssa_v0, pb_bk_inline2515__ssa_v0])
                                pb_wk_inline2476__ssa_v0: pl.Tensor[[256, 256], pl.INT8] = pl.tensor.slice(wo_b__ssa_v0, [256, 256], [pb_n0_inline2478__idx_v0, pb_bk_inline2515__ssa_v0])
                                pb_acc_inline2507__ssa_v3: pl.Tensor[[128, 256], pl.INT32] = pl.tensor.matmul_acc(pb_acc_inline2507__iter_v1, pb_xk_inline2520__ssa_v0, pb_wk_inline2476__ssa_v0, a_trans=False, b_trans=True)
                                pb_acc_inline2507__rv_v2: pl.Tensor[[128, 256], pl.INT32] = pl.yield_(pb_acc_inline2507__ssa_v3)
                            pb_col_inline2474__ssa_v0: pl.Scalar[pl.INDEX] = local_group_inline2514__idx_v0 * 4096 + pb_n0_inline2478__idx_v0
                            own_b_i32_inline2528__ssa_v5: pl.Tensor[[256, 16384], pl.INT32] = pl.tensor.assemble(own_b_i32_inline2528__iter_v3, pb_acc_inline2507__rv_v2, [pb_t0_inline2557__ssa_v0, pb_col_inline2474__ssa_v0])
                            own_b_i32_inline2528__rv_v4: pl.Tensor[[256, 16384], pl.INT32] = pl.yield_(own_b_i32_inline2528__ssa_v5)
                    own_a_fp32_inline2500__rv_v2, own_a_i8_inline2497__rv_v2, own_b_i32_inline2528__rv_v2, own_scale_inline2494__rv_v2 = pl.yield_(own_a_fp32_inline2500__ssa_v3, own_a_i8_inline2497__rv_v7, own_b_i32_inline2528__rv_v4, own_scale_inline2494__rv_v4)
                for dq_worker_inline2469__ssa_v0 in pl.spmd(12, name_hint="tp_o_b_dequant_spmd", optimizations=[pl.cross_core_slot(slot_num=2)]):
                    for dq_blk_inline2556__idx_v0, (publish_all_inline2525__iter_v3,) in pl.range(dq_worker_inline2469__ssa_v0, own_act_rows_inline2505__ssa_v0 * 8, 12, init_values=(publish_all_inline2525__iter_v1,)):
                        dq_rb_inline2484__ssa_v0: pl.Scalar[pl.INDEX] = dq_blk_inline2556__idx_v0 // 8
                        dq_nb_inline2468__ssa_v0: pl.Scalar[pl.INDEX] = dq_blk_inline2556__idx_v0 - dq_rb_inline2484__ssa_v0 * 8
                        dq_row_inline2467__ssa_v0: pl.Scalar[pl.INDEX] = dq_rb_inline2484__ssa_v0 * 16
                        dq_n0_inline2545__ssa_v0: pl.Scalar[pl.INDEX] = dq_nb_inline2468__ssa_v0 * 512
                        dq_rows_inline2466__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(16, 256 - dq_row_inline2467__ssa_v0)
                        dq_acc_inline2464__ssa_v0: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.full([16, 512], dtype=pl.FP32, value=0.0)
                        for dq_group_inline2463__idx_v0, (dq_acc_inline2464__iter_v1,) in pl.pipeline(4, stage=2, init_values=(dq_acc_inline2464__ssa_v0,)):
                            dq_col_inline2488__ssa_v0: pl.Scalar[pl.INDEX] = dq_group_inline2463__idx_v0 * 4096 + dq_n0_inline2545__ssa_v0
                            dq_i32_inline2461__ssa_v0: pl.Tensor[[16, 512], pl.INT32, pl.TensorView(valid_shape=[dq_rows_inline2466__ssa_v0, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(own_b_i32_inline2528__rv_v2, [16, 512], [dq_row_inline2467__ssa_v0, dq_col_inline2488__ssa_v0], [dq_rows_inline2466__ssa_v0, 512])
                            dq_fp32_inline2459__ssa_v0: pl.Tensor[[16, 512], pl.FP32, pl.TensorView(valid_shape=[dq_rows_inline2466__ssa_v0, 512], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.cast(dq_i32_inline2461__ssa_v0, target_type=pl.FP32, mode='none')
                            dq_srow_inline2518__ssa_v0: pl.Tensor[[1, 16], pl.FP32, pl.TensorView(valid_shape=[1, dq_rows_inline2466__ssa_v0], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(own_scale_inline2494__rv_v2, [1, 16], [dq_group_inline2463__idx_v0, dq_row_inline2467__ssa_v0], [1, dq_rows_inline2466__ssa_v0])
                            dq_scol_inline2462__ssa_v0: pl.Tensor[[16, 1], pl.FP32, pl.TensorView(valid_shape=[dq_rows_inline2466__ssa_v0, 1], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.reshape(dq_srow_inline2518__ssa_v0, [16, 1])
                            dq_acc_inline2464__ssa_v3: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.add(dq_acc_inline2464__iter_v1, pl.tensor.row_expand_mul(dq_fp32_inline2459__ssa_v0, dq_scol_inline2462__ssa_v0))
                            dq_acc_inline2464__rv_v2: pl.Tensor[[16, 512], pl.FP32] = pl.yield_(dq_acc_inline2464__ssa_v3)
                        dq_wscale_inline2458__ssa_v0: pl.Tensor[[1, 512], pl.FP32] = pl.tensor.reshape(pl.tensor.slice(wo_b_scale__ssa_v0, [512], [dq_n0_inline2545__ssa_v0]), [1, 512])
                        dq_bf16_inline2457__ssa_v0: pl.Tensor[[16, 512], pl.BF16] = pl.tensor.cast(pl.tensor.col_expand_mul(dq_acc_inline2464__rv_v2, dq_wscale_inline2458__ssa_v0), target_type=pl.BF16, mode='rint')
                        dq_stage_inline2456__ssa_v0: pl.Scalar[pl.INDEX] = owner_inline2504__idx_v0 * 256 + dq_row_inline2467__ssa_v0
                        publish_all_inline2525__ssa_v5: pl.Tensor[[512, 4096], pl.BF16] = pl.tensor.assemble(publish_all_inline2525__iter_v3, pl.tensor.set_validshape(dq_bf16_inline2457__ssa_v0, dq_rows_inline2466__ssa_v0, 512), [dq_stage_inline2456__ssa_v0, dq_n0_inline2545__ssa_v0])
                        publish_all_inline2525__rv_v4: pl.Tensor[[512, 4096], pl.BF16] = pl.yield_(publish_all_inline2525__ssa_v5)
                publish_all_inline2525__rv_v2: pl.Tensor[[512, 4096], pl.BF16] = pl.yield_(publish_all_inline2525__rv_v4)
            with pl.spmd(24, name_hint="tp_o_b_publish_spmd") as publish_tid_inline2454__ssa_v0:
                pub_worker_inline2453__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for pub_blk_inline2559__idx_v0 in pl.range(pub_worker_inline2453__ssa_v0, 2 * put_rows_inline2508__ssa_v0, 24):
                    pub_owner_inline2541__ssa_v0: pl.Scalar[pl.INDEX] = pub_blk_inline2559__idx_v0 // put_rows_inline2508__ssa_v0
                    pub_row_block_inline2460__ssa_v0: pl.Scalar[pl.INDEX] = pub_blk_inline2559__idx_v0 - pub_owner_inline2541__ssa_v0 * put_rows_inline2508__ssa_v0
                    pub_owner_row_inline2482__ssa_v0: pl.Scalar[pl.INDEX] = pub_row_block_inline2460__ssa_v0 * 8
                    pub_rows_inline2452__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(8, 256 - pub_owner_row_inline2482__ssa_v0)
                    pub_src_row_inline2486__ssa_v0: pl.Scalar[pl.INDEX] = pub_owner_inline2541__ssa_v0 * 256 + pub_owner_row_inline2482__ssa_v0
                    pub_dst_row_inline2451__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(tp_rank__ssa_v0, pl.INDEX) * 256 + pub_owner_row_inline2482__ssa_v0
                    pld.tensor.put(o_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + pub_owner_inline2541__ssa_v0, publish_all_inline2525__rv_v2, [pub_dst_row_inline2451__ssa_v0, 0], [pub_src_row_inline2486__ssa_v0, 0], [pub_rows_inline2452__ssa_v0, 4096], atomic=pl.AtomicType.None_, chunk_rows=8, chunk_cols=4096)
                for notify_owner_inline2517__idx_v0 in pl.range(2):
                    if notify_owner_inline2517__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(o_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + notify_owner_inline2517__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_rs_wait", deps=[publish_tid_inline2454__ssa_v0]) as wait_tid_inline2450__ssa_v0:
                expected_inline2475__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(24, pl.INT32)
                for source_tp_inline2449__idx_v0 in pl.range(2):
                    if source_tp_inline2449__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.wait(o_signal__ssa_v0, [source_tp_inline2449__idx_v0, 0], expected_inline2475__ssa_v0, cmp=1)
            with pl.spmd(48, name_hint="tp_o_rs_reduce_spmd", deps=[wait_tid_inline2450__ssa_v0]) as reduce_tid_inline2543__ssa_v0:
                worker_inline2455__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
                for block_inline2473__idx_v0 in pl.range(worker_inline2455__ssa_v0, 256, 48):
                    local_row_inline2555__ssa_v0: pl.Scalar[pl.INDEX] = block_inline2473__idx_v0 // 1
                    d_block_inline2465__ssa_v0: pl.Scalar[pl.INDEX] = block_inline2473__idx_v0 - local_row_inline2555__ssa_v0 * 1
                    d0_inline2447__ssa_v0: pl.Scalar[pl.INDEX] = d_block_inline2465__ssa_v0 * 4096
                    own_partial_inline2446__ssa_v0: pl.Tile[[1, 4096], pl.BF16] = pl.tile.load(o_window__ssa_v0, [local_row_inline2555__ssa_v0, d0_inline2447__ssa_v0], [1, 4096], [1, 4096])
                    reduce_acc_inline2445__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(own_partial_inline2446__ssa_v0, target_type=pl.FP32, mode='none')
                    for source_tp_inline2444__idx_v0, (reduce_acc_inline2445__iter_v1,) in pl.range(1, 2, init_values=(reduce_acc_inline2445__ssa_v0,)):
                        source_row_inline2481__ssa_v0: pl.Scalar[pl.INDEX] = source_tp_inline2444__idx_v0 * 256 + local_row_inline2555__ssa_v0
                        source_partial_inline2443__ssa_v0: pl.Tile[[1, 4096], pl.BF16] = pl.tile.load(o_window__ssa_v0, [source_row_inline2481__ssa_v0, d0_inline2447__ssa_v0], [1, 4096], [1, 4096])
                        source_fp32_inline2506__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.cast(source_partial_inline2443__ssa_v0, target_type=pl.FP32, mode='none')
                        reduce_acc_inline2445__ssa_v3: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.tile.add(reduce_acc_inline2445__iter_v1, source_fp32_inline2506__ssa_v0)
                        reduce_acc_inline2445__rv_v2: pl.Tile[[1, 4096], pl.FP32, pl.Mem.Vec] = pl.yield_(reduce_acc_inline2445__ssa_v3)
                    reduced_inline2553__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.Mem.Vec] = pl.tile.cast(reduce_acc_inline2445__rv_v2, target_type=pl.BF16, mode='rint')
                    pl.tile.store(reduced_inline2553__ssa_v0, [local_row_inline2555__ssa_v0, d0_inline2447__ssa_v0], attn_out_inline1284__ssa_v0)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_rs_complete", deps=[reduce_tid_inline2543__ssa_v0]):
                completion_anchor_inline2442__ssa_v0: pl.Scalar[pl.BF16] = pl.tensor.read(attn_out_inline1284__ssa_v0, [0, 0])
                for peer_tp_inline2472__idx_v0 in pl.range(2):
                    if peer_tp_inline2472__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(o_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline2472__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
                completion_expected_inline2441__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(25, pl.INT32)
                for source_tp_inline2448__idx_v0 in pl.range(2):
                    if source_tp_inline2448__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.wait(o_signal__ssa_v0, [source_tp_inline2448__idx_v0, 0], completion_expected_inline2441__ssa_v0, cmp=1)
                reset_value_inline2471__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(-25, pl.INT32)
                self_rank_inline2440__ssa_v0: pl.Scalar[pl.INT32] = group_base__ssa_v0 + tp_rank__ssa_v0
                for source_tp_inline2470__idx_v0 in pl.range(2):
                    if source_tp_inline2470__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                        pld.system.notify(o_signal__ssa_v0, self_rank_inline2440__ssa_v0, [source_tp_inline2470__idx_v0, 0], reset_value_inline2471__ssa_v0, op=0)
                pl.tensor.write(attn_out_inline1284__ssa_v0, [0, 0], completion_anchor_inline2442__ssa_v0)
            _o_reduced_inline1269__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16] = attn_out_inline1284__ssa_v0
            o_signal__ssa_v1: pld.DistributedTensor[[2, 1], pl.INT32] = o_signal__ssa_v0
        with pl.scope():
            t_dim_inline2576__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(attn_out_inline1284__ssa_v0, 0)
            residual_flat_inline2567__ssa_v0: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.tensor.reshape(x_hc__ssa_v0, [t_dim_inline2576__ssa_v0, 16384])
            y_flat_inline2568__ssa_v0: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.tensor.reshape(x_out__ssa_v0, [t_dim_inline2576__ssa_v0, 16384])
            token_tiles_inline2571__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline2576__ssa_v0 + 4 - 1) // 4
            for token_block_inline2572__ssa_v0 in pl.spmd(token_tiles_inline2571__ssa_v0, name_hint="hc_post_spmd"):
                t0_inline2573__ssa_v0: pl.Scalar[pl.INDEX] = token_block_inline2572__ssa_v0 * 4
                for t_inline2574__idx_v0, (y_flat_inline2568__iter_v1,) in pl.pipeline(t0_inline2573__ssa_v0, t0_inline2573__ssa_v0 + 4, stage=2, init_values=(y_flat_inline2568__ssa_v0,)):
                    if t_inline2574__idx_v0 < t_dim_inline2576__ssa_v0:
                        x_row_inline2578__ssa_v0: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.cast(pl.tensor.slice(attn_out_inline1284__ssa_v0, [1, 4096], [t_inline2574__idx_v0, 0]), target_type=pl.FP32, mode='round')
                        post_w_inline2577__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0, 0])
                        y_row_inline2569__ssa_v0: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(x_row_inline2578__ssa_v0, post_w_inline2577__ssa_v0)
                        for in_h_inline2570__idx_v0, (y_row_inline2569__iter_v1,) in pl.pipeline(4, stage=4, init_values=(y_row_inline2569__ssa_v0,)):
                            comb_w_inline2579__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, in_h_inline2570__idx_v0 * 4 + 0])
                            res_d_inline2566__ssa_v0: pl.Scalar[pl.INDEX] = in_h_inline2570__idx_v0 * 4096
                            res_row_inline2565__ssa_v0: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.slice(residual_flat_inline2567__ssa_v0, [1, 4096], [t_inline2574__idx_v0, res_d_inline2566__ssa_v0])
                            weighted_inline2564__ssa_v0: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(res_row_inline2565__ssa_v0, comb_w_inline2579__ssa_v0)
                            y_row_inline2569__ssa_v3: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(y_row_inline2569__iter_v1, weighted_inline2564__ssa_v0)
                            y_row_inline2569__rv_v2: pl.Tensor[[1, 4096], pl.FP32] = pl.yield_(y_row_inline2569__ssa_v3)
                        y_flat_inline2568__ssa_v3: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.tensor.assemble(y_flat_inline2568__iter_v1, y_row_inline2569__rv_v2, [t_inline2574__idx_v0, pl.const(0, pl.INDEX) * pl.const(4096, pl.INDEX)])
                        post_w_inline2577__ssa_v1: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0, 1])
                        y_row_inline2569__ssa_v4: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(x_row_inline2578__ssa_v0, post_w_inline2577__ssa_v1)
                        for in_h_inline2570__idx_v1, (y_row_inline2569__iter_v5,) in pl.pipeline(4, stage=4, init_values=(y_row_inline2569__ssa_v4,)):
                            comb_w_inline2579__ssa_v1: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, in_h_inline2570__idx_v1 * 4 + 1])
                            res_d_inline2566__ssa_v1: pl.Scalar[pl.INDEX] = in_h_inline2570__idx_v1 * 4096
                            res_row_inline2565__ssa_v1: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.slice(residual_flat_inline2567__ssa_v0, [1, 4096], [t_inline2574__idx_v0, res_d_inline2566__ssa_v1])
                            weighted_inline2564__ssa_v1: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(res_row_inline2565__ssa_v1, comb_w_inline2579__ssa_v1)
                            y_row_inline2569__ssa_v7: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(y_row_inline2569__iter_v5, weighted_inline2564__ssa_v1)
                            y_row_inline2569__rv_v6: pl.Tensor[[1, 4096], pl.FP32] = pl.yield_(y_row_inline2569__ssa_v7)
                        y_flat_inline2568__ssa_v4: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.tensor.assemble(y_flat_inline2568__ssa_v3, y_row_inline2569__rv_v6, [t_inline2574__idx_v0, pl.const(1, pl.INDEX) * pl.const(4096, pl.INDEX)])
                        post_w_inline2577__ssa_v2: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0, 2])
                        y_row_inline2569__ssa_v8: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(x_row_inline2578__ssa_v0, post_w_inline2577__ssa_v2)
                        for in_h_inline2570__idx_v2, (y_row_inline2569__iter_v9,) in pl.pipeline(4, stage=4, init_values=(y_row_inline2569__ssa_v8,)):
                            comb_w_inline2579__ssa_v2: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, in_h_inline2570__idx_v2 * 4 + 2])
                            res_d_inline2566__ssa_v2: pl.Scalar[pl.INDEX] = in_h_inline2570__idx_v2 * 4096
                            res_row_inline2565__ssa_v2: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.slice(residual_flat_inline2567__ssa_v0, [1, 4096], [t_inline2574__idx_v0, res_d_inline2566__ssa_v2])
                            weighted_inline2564__ssa_v2: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(res_row_inline2565__ssa_v2, comb_w_inline2579__ssa_v2)
                            y_row_inline2569__ssa_v11: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(y_row_inline2569__iter_v9, weighted_inline2564__ssa_v2)
                            y_row_inline2569__rv_v10: pl.Tensor[[1, 4096], pl.FP32] = pl.yield_(y_row_inline2569__ssa_v11)
                        y_flat_inline2568__ssa_v5: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.tensor.assemble(y_flat_inline2568__ssa_v4, y_row_inline2569__rv_v10, [t_inline2574__idx_v0, pl.const(2, pl.INDEX) * pl.const(4096, pl.INDEX)])
                        post_w_inline2577__ssa_v3: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0, 3])
                        y_row_inline2569__ssa_v12: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(x_row_inline2578__ssa_v0, post_w_inline2577__ssa_v3)
                        for in_h_inline2570__idx_v3, (y_row_inline2569__iter_v13,) in pl.pipeline(4, stage=4, init_values=(y_row_inline2569__ssa_v12,)):
                            comb_w_inline2579__ssa_v3: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, in_h_inline2570__idx_v3 * 4 + 3])
                            res_d_inline2566__ssa_v3: pl.Scalar[pl.INDEX] = in_h_inline2570__idx_v3 * 4096
                            res_row_inline2565__ssa_v3: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.slice(residual_flat_inline2567__ssa_v0, [1, 4096], [t_inline2574__idx_v0, res_d_inline2566__ssa_v3])
                            weighted_inline2564__ssa_v3: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.muls(res_row_inline2565__ssa_v3, comb_w_inline2579__ssa_v3)
                            y_row_inline2569__ssa_v15: pl.Tensor[[1, 4096], pl.FP32] = pl.tensor.add(y_row_inline2569__iter_v13, weighted_inline2564__ssa_v3)
                            y_row_inline2569__rv_v14: pl.Tensor[[1, 4096], pl.FP32] = pl.yield_(y_row_inline2569__ssa_v15)
                        y_flat_inline2568__ssa_v6: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.tensor.assemble(y_flat_inline2568__ssa_v5, y_row_inline2569__rv_v14, [t_inline2574__idx_v0, pl.const(3, pl.INDEX) * pl.const(4096, pl.INDEX)])
                        y_flat_inline2568__phi_v7: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.yield_(y_flat_inline2568__ssa_v6)
                    else:
                        y_flat_inline2568__phi_v7: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.yield_(y_flat_inline2568__iter_v1)
                    y_flat_inline2568__rv_v2: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32] = pl.yield_(y_flat_inline2568__phi_v7)
        return x_out__ssa_v0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def l3_decode_csa(self, x_hc__ssa_v0: pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32], hc_attn_fn__ssa_v0: pl.Tensor[[2, 24, 16384], pl.FP32], hc_attn_scale__ssa_v0: pl.Tensor[[2, 3], pl.FP32], hc_attn_base__ssa_v0: pl.Tensor[[2, 24], pl.FP32], attn_norm_w__ssa_v0: pl.Tensor[[2, 4096], pl.BF16], wq_a__ssa_v0: pl.Tensor[[2, 4096, 1024], pl.BF16], wq_b__ssa_v0: pl.Tensor[[2, 1024, 32768], pl.INT8], wq_b_scale__ssa_v0: pl.Tensor[[2, 32768], pl.FP32], wkv__ssa_v0: pl.Tensor[[2, 4096, 512], pl.BF16], gamma_cq__ssa_v0: pl.Tensor[[2, 1024], pl.BF16], gamma_ckv__ssa_v0: pl.Tensor[[2, 512], pl.BF16], freqs_cos_local__ssa_v0: pl.Tensor[[2, T_DYN, 64], pl.BF16], freqs_cos__ssa_v0: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16], freqs_sin_local__ssa_v0: pl.Tensor[[2, T_DYN, 64], pl.BF16], freqs_sin__ssa_v0: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16], cmp_freqs_cos__ssa_v0: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16], cmp_freqs_sin__ssa_v0: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16], cmp_wkv__ssa_v0: pl.Tensor[[2, 1024, 4096], pl.BF16], cmp_wgate__ssa_v0: pl.Tensor[[2, 1024, 4096], pl.BF16], cmp_ape__ssa_v0: pl.Tensor[[2, 4, 1024], pl.FP32], cmp_norm_w__ssa_v0: pl.Tensor[[2, 512], pl.BF16], compress_state__ssa_v0: pl.InOut[pl.Tensor[[2, MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32]], compress_state_block_table__ssa_v0: pl.Tensor[[2, KV_B_DYN, 4], pl.INT32], idx_wq_b__ssa_v0: pl.Tensor[[2, 1024, 8192], pl.INT8], idx_wq_b_scale__ssa_v0: pl.Tensor[[2, 8192], pl.FP32], weights_proj__ssa_v0: pl.Tensor[[2, 4096, 64], pl.BF16], hadamard_idx__ssa_v0: pl.Tensor[[2, 128, 128], pl.BF16], inner_wkv__ssa_v0: pl.Tensor[[2, 256, 4096], pl.BF16], inner_wgate__ssa_v0: pl.Tensor[[2, 256, 4096], pl.BF16], inner_ape__ssa_v0: pl.Tensor[[2, 4, 256], pl.FP32], inner_norm_w__ssa_v0: pl.Tensor[[2, 128], pl.BF16], inner_compress_state__ssa_v0: pl.InOut[pl.Tensor[[2, INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32]], inner_compress_state_block_table__ssa_v0: pl.Tensor[[2, KV_B_DYN, 4], pl.INT32], kv_cache__ssa_v0: pl.InOut[pl.Tensor[[2, ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_kv__ssa_v0: pl.InOut[pl.Tensor[[2, CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16]], cmp_block_table__ssa_v0: pl.Tensor[[2, B_DYN, 8192], pl.INT32], idx_kv_cache__ssa_v0: pl.InOut[pl.Tensor[[2, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8]], idx_kv_scale__ssa_v0: pl.InOut[pl.Tensor[[2, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32]], idx_block_table__ssa_v0: pl.Tensor[[2, B_DYN, 8192], pl.INT32], ori_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64], window_swa_indices__ssa_v0: pl.Tensor[[2, T_DYN, 128], pl.INT32], window_swa_lens__ssa_v0: pl.Tensor[[2, T_DYN], pl.INT32], cmp_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64], idx_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64], state_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64], inner_state_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64], position_ids_local__ssa_v0: pl.Tensor[[2, T_DYN], pl.INT32], position_ids__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT32], kv_seq_lens__ssa_v0: pl.Tensor[[2, B_DYN], pl.INT32], attn_sink__ssa_v0: pl.Tensor[[2, 64], pl.FP32], wo_a__ssa_v0: pl.Tensor[[2, 4, 1024, 4096], pl.BF16], wo_b__ssa_v0: pl.Tensor[[2, 4096, 4096], pl.INT8], wo_b_scale__ssa_v0: pl.Tensor[[2, 4096], pl.FP32], x_out__ssa_v0: pl.Out[pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32]], local_t__ssa_v0: pl.Scalar[pl.INT32]) -> pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32]:
        # Launch the complete CSA layer on one physical TP group.
        gather_window_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(4194304, pl.INT64))
        gather_signal_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        attention_window_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(16777216, pl.INT64))
        attention_signal_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        o_window_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(4194304, pl.INT64))
        o_signal_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        for rank__idx_v0 in pl.range(pld.system.world_size()):
            gather_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16] = pld.tensor.window(gather_window_buf__ssa_v0, [512, 4096], dtype=pl.BF16)
            gather_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32] = pld.tensor.window(gather_signal_buf__ssa_v0, [2, 1], dtype=pl.INT32)
            attention_window__ssa_v0: pld.DistributedTensor[[2048, 4096], pl.BF16] = pld.tensor.window(attention_window_buf__ssa_v0, [2048, 4096], dtype=pl.BF16)
            attention_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32] = pld.tensor.window(attention_signal_buf__ssa_v0, [2, 1], dtype=pl.INT32)
            o_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16] = pld.tensor.window(o_window_buf__ssa_v0, [512, 4096], dtype=pl.BF16)
            o_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32] = pld.tensor.window(o_signal_buf__ssa_v0, [2, 1], dtype=pl.INT32)
            self.decode_csa_test(pl.tensor.slice(x_hc__ssa_v0, [1, T_DYN, 4, 4096], [rank__idx_v0, 0, 0, 0], [], [0]), pl.tensor.slice(hc_attn_fn__ssa_v0, [1, 24, 16384], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(hc_attn_scale__ssa_v0, [1, 3], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(hc_attn_base__ssa_v0, [1, 24], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(attn_norm_w__ssa_v0, [1, 4096], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(wq_a__ssa_v0, [1, 4096, 1024], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(wq_b__ssa_v0, [1, 1024, 32768], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(wq_b_scale__ssa_v0, [1, 32768], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(wkv__ssa_v0, [1, 4096, 512], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(gamma_cq__ssa_v0, [1, 1024], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(gamma_ckv__ssa_v0, [1, 512], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(freqs_cos_local__ssa_v0, [1, T_DYN, 64], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(freqs_sin_local__ssa_v0, [1, T_DYN, 64], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(freqs_cos__ssa_v0, [1, KV_T_DYN, 64], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(freqs_sin__ssa_v0, [1, KV_T_DYN, 64], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(cmp_freqs_cos__ssa_v0, [1, KV_T_DYN, 64], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(cmp_freqs_sin__ssa_v0, [1, KV_T_DYN, 64], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(cmp_wkv__ssa_v0, [1, 1024, 4096], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(cmp_wgate__ssa_v0, [1, 1024, 4096], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(cmp_ape__ssa_v0, [1, 4, 1024], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(cmp_norm_w__ssa_v0, [1, 512], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(compress_state__ssa_v0, [1, MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], [rank__idx_v0, 0, 0, 0], [], [0]), pl.tensor.slice(compress_state_block_table__ssa_v0, [1, KV_B_DYN, 4], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(idx_wq_b__ssa_v0, [1, 1024, 8192], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(idx_wq_b_scale__ssa_v0, [1, 8192], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(weights_proj__ssa_v0, [1, 4096, 64], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(hadamard_idx__ssa_v0, [1, 128, 128], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(inner_wkv__ssa_v0, [1, 256, 4096], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(inner_wgate__ssa_v0, [1, 256, 4096], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(inner_ape__ssa_v0, [1, 4, 256], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(inner_norm_w__ssa_v0, [1, 128], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(inner_compress_state__ssa_v0, [1, INNER_STATE_BLOCK_NUM_DYN, 2, 512], [rank__idx_v0, 0, 0, 0], [], [0]), pl.tensor.slice(inner_compress_state_block_table__ssa_v0, [1, KV_B_DYN, 4], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(kv_cache__ssa_v0, [1, ORI_BLOCK_NUM_DYN, 32, 1, 512], [rank__idx_v0, 0, 0, 0, 0], [], [0]), pl.tensor.slice(cmp_kv__ssa_v0, [1, CMP_BLOCK_NUM_DYN, 32, 1, 512], [rank__idx_v0, 0, 0, 0, 0], [], [0]), pl.tensor.slice(cmp_block_table__ssa_v0, [1, B_DYN, 8192], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(idx_kv_cache__ssa_v0, [1, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], [rank__idx_v0, 0, 0, 0, 0], [], [0]), pl.tensor.slice(idx_kv_scale__ssa_v0, [1, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], [rank__idx_v0, 0, 0, 0, 0], [], [0]), pl.tensor.slice(idx_block_table__ssa_v0, [1, B_DYN, 8192], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(ori_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(window_swa_indices__ssa_v0, [1, T_DYN, 128], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(window_swa_lens__ssa_v0, [1, T_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(cmp_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(idx_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(state_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(inner_state_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(position_ids_local__ssa_v0, [1, T_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(position_ids__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(kv_seq_lens__ssa_v0, [1, B_DYN], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(attn_sink__ssa_v0, [1, 64], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(wo_a__ssa_v0, [1, 4, 1024, 4096], [rank__idx_v0, 0, 0, 0], [], [0]), pl.tensor.slice(wo_b__ssa_v0, [1, 4096, 4096], [rank__idx_v0, 0, 0], [], [0]), pl.tensor.slice(wo_b_scale__ssa_v0, [1, 4096], [rank__idx_v0, 0], [], [0]), pl.tensor.slice(x_out__ssa_v0, [1, T_DYN, 4, 4096], [rank__idx_v0, 0, 0, 0], [], [0]), gather_window__ssa_v0, gather_signal__ssa_v0, attention_window__ssa_v0, attention_signal__ssa_v0, o_window__ssa_v0, o_signal__ssa_v0, 0, rank__idx_v0, 256, device=rank__idx_v0)
        return x_out__ssa_v0
