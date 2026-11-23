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
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def comb_sinkhorn(t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], inv_rms_inline1463__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], mixes_raw_inline1505__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], hc_base_2d_inline1467__ssa_v0: pl.Tensor[[1, 24], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 96)], scale2_inline1480__ssa_v0: pl.Scalar[pl.FP32], comb_t_inline1267__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], comb_tail_store_inline1523__ssa_v0: pl.InOut[pl.Tensor[[8, 32], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1024)]]) -> tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32]]:
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_35: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_66: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_82: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_83: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_84: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_85: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_86: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_101: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_102: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_103: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_104: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_105: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_106: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        ob_inline1522__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t0_inline1476__ssa_v3: pl.Scalar[pl.INDEX] = ob_inline1522__ssa_v0 * 8
        valid_rows_inline1507__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v3, 8)
        inv_col_t_inline1560__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 1])] = pl.tile.load(inv_rms_inline1463__ssa_v1, [t0_inline1476__ssa_v3, 0], [8, 1], [valid_rows_inline1507__ssa_v2, 1], target_memory=pl.Mem.Vec)
        mix_g0_inline1525__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v3, 8], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
        mix_g1_inline1528__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v3, 12], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
        mix_g2_inline1531__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v3, 16], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
        mix_g3_inline1552__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v3, 20], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
        cb0_inline1458__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467__ssa_v0, [0, 8], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        cb1_inline1485__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467__ssa_v0, [0, 12], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        cb2_inline1532__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467__ssa_v0, [0, 16], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        cb3_inline1533__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_84, pl.const(3584, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, 4])] = pl.tile.load(hc_base_2d_inline1467__ssa_v0, [0, 20], [1, 8], [1, 4], target_memory=pl.Mem.Vec)
        t__tmp_v19: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.row_expand_mul(mix_g0_inline1525__ssa_v0, inv_col_t_inline1560__ssa_v0)
        t__tmp_v20: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.muls(t__tmp_v19, scale2_inline1480__ssa_v0)
        t__tmp_v21: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.col_expand(mix_g0_inline1525__ssa_v0, cb0_inline1458__ssa_v0)
        row0_inline1534__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.add(t__tmp_v20, t__tmp_v21)
        t__tmp_v22: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.row_expand_mul(mix_g1_inline1528__ssa_v0, inv_col_t_inline1560__ssa_v0)
        t__tmp_v23: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.muls(t__tmp_v22, scale2_inline1480__ssa_v0)
        t__tmp_v24: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.col_expand(mix_g1_inline1528__ssa_v0, cb1_inline1485__ssa_v0)
        row1_inline1575__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.add(t__tmp_v23, t__tmp_v24)
        t__tmp_v25: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.row_expand_mul(mix_g2_inline1531__ssa_v0, inv_col_t_inline1560__ssa_v0)
        t__tmp_v26: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.muls(t__tmp_v25, scale2_inline1480__ssa_v0)
        t__tmp_v27: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.col_expand(mix_g2_inline1531__ssa_v0, cb2_inline1532__ssa_v0)
        row2_inline1468__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.add(t__tmp_v26, t__tmp_v27)
        t__tmp_v28: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.row_expand_mul(mix_g3_inline1552__ssa_v0, inv_col_t_inline1560__ssa_v0)
        t__tmp_v29: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.muls(t__tmp_v28, scale2_inline1480__ssa_v0)
        t__tmp_v30: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.col_expand(mix_g3_inline1552__ssa_v0, cb3_inline1533__ssa_v0)
        row3_inline1484__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.add(t__tmp_v29, t__tmp_v30)
        row0_p_inline1478__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row0_inline1534__ssa_v0, pad_value=pl.PadValue.min)
        row1_p_inline1495__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row1_inline1575__ssa_v0, pad_value=pl.PadValue.min)
        row2_p_inline1535__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row2_inline1468__ssa_v0, pad_value=pl.PadValue.min)
        row3_p_inline1445__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(row3_inline1484__ssa_v0, pad_value=pl.PadValue.min)
        row_max_tmp_inline1487__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        row_sum_tmp_inline1498__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        row0_max_inline1538__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(row0_p_inline1478__ssa_v0, row_max_tmp_inline1487__ssa_v0)
        row1_max_inline1473__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(row1_p_inline1495__ssa_v0, row_max_tmp_inline1487__ssa_v0)
        row2_max_inline1539__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(row2_p_inline1535__ssa_v0, row_max_tmp_inline1487__ssa_v0)
        row3_max_inline1540__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_84, pl.const(3584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(row3_p_inline1445__ssa_v0, row_max_tmp_inline1487__ssa_v0)
        t__tmp_v31: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(row0_p_inline1478__ssa_v0, row0_max_inline1538__ssa_v0)
        row0_exp_inline1527__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(t__tmp_v31)
        t__tmp_v32: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(row1_p_inline1495__ssa_v0, row1_max_inline1473__ssa_v0)
        row1_exp_inline1543__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(t__tmp_v32)
        t__tmp_v33: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(row2_p_inline1535__ssa_v0, row2_max_inline1539__ssa_v0)
        row2_exp_inline1545__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(t__tmp_v33)
        t__tmp_v34: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_sub(row3_p_inline1445__ssa_v0, row3_max_inline1540__ssa_v0)
        row3_exp_inline1547__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.exp(t__tmp_v34)
        row0_sum_inline1512__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row0_exp_inline1527__ssa_v0, row_sum_tmp_inline1498__ssa_v0)
        row1_sum_inline1482__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row1_exp_inline1543__ssa_v0, row_sum_tmp_inline1498__ssa_v0)
        row2_sum_inline1546__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row2_exp_inline1545__ssa_v0, row_sum_tmp_inline1498__ssa_v0)
        row3_sum_inline1453__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row3_exp_inline1547__ssa_v0, row_sum_tmp_inline1498__ssa_v0)
        t__tmp_v35: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(row0_exp_inline1527__ssa_v0, row0_sum_inline1512__ssa_v0)
        row0_soft_inline1548__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(t__tmp_v35, 9.9999999999999995e-07)
        t__tmp_v36: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(row1_exp_inline1543__ssa_v0, row1_sum_inline1482__ssa_v0)
        row1_soft_inline1515__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(t__tmp_v36, 9.9999999999999995e-07)
        t__tmp_v37: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(row2_exp_inline1545__ssa_v0, row2_sum_inline1546__ssa_v0)
        row2_soft_inline1477__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(t__tmp_v37, 9.9999999999999995e-07)
        t__tmp_v38: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.row_expand_div(row3_exp_inline1547__ssa_v0, row3_sum_inline1453__ssa_v0)
        row3_soft_inline1549__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.adds(t__tmp_v38, 9.9999999999999995e-07)
        row0_valid_inline1508__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row0_soft_inline1548__ssa_v0, 8, 4)
        row1_valid_inline1551__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row1_soft_inline1515__ssa_v0, 8, 4)
        row2_valid_inline1517__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row2_soft_inline1477__ssa_v0, 8, 4)
        row3_valid_inline1553__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.min)] = pl.tile.set_validshape(row3_soft_inline1549__ssa_v0, 8, 4)
        row0_eff_inline1555__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row0_valid_inline1508__ssa_v0, pad_value=pl.PadValue.zero)
        row1_eff_inline1557__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row1_valid_inline1551__ssa_v0, pad_value=pl.PadValue.zero)
        row2_eff_inline1479__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row2_valid_inline1517__ssa_v0, pad_value=pl.PadValue.zero)
        row3_eff_inline1559__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.fillpad(row3_valid_inline1553__ssa_v0, pad_value=pl.PadValue.zero)
        row_sum_tmp_iter_inline1562__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tmp_v39: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(row0_eff_inline1555__ssa_v0, row1_eff_inline1557__ssa_v0)
        t__tmp_v40: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(row2_eff_inline1479__ssa_v0, row3_eff_inline1559__ssa_v0)
        col_sum_inline1563__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(t__tmp_v39, t__tmp_v40)
        col_sum_v1_inline1564__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_inline1563__ssa_v0, 9.9999999999999995e-07)
        row0_cur_inline1565__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_eff_inline1555__ssa_v0, col_sum_v1_inline1564__ssa_v0)
        row1_cur_inline1449__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_eff_inline1557__ssa_v0, col_sum_v1_inline1564__ssa_v0)
        row2_cur_inline1566__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_eff_inline1479__ssa_v0, col_sum_v1_inline1564__ssa_v0)
        row3_cur_inline1569__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_eff_inline1559__ssa_v0, col_sum_v1_inline1564__ssa_v0)
        for _sk_it_inline1460__idx_v0, (col_sum_v1_inline1564__iter_v1, row0_cur_inline1565__iter_v1, row1_cur_inline1449__iter_v1, row2_cur_inline1566__iter_v1, row3_cur_inline1569__iter_v1) in pl.range(0, 18, 2, init_values=(col_sum_v1_inline1564__ssa_v0, row0_cur_inline1565__ssa_v0, row1_cur_inline1449__ssa_v0, row2_cur_inline1566__ssa_v0, row3_cur_inline1569__ssa_v0)):
            t__tmp_v41: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row0_cur_inline1565__iter_v1, row_sum_tmp_iter_inline1562__ssa_v0)
            row0_rowsum_inline1571__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v41, [1, 8])
            row0_rowsum_inline1571__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row0_rowsum_inline1571__rm_a0_tmp_v0, 9.9999999999999995e-07)
            row0_rowsum_inline1571__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row0_rowsum_inline1571__row_major_tmp_v1, [8, 1])
            t__tmp_v42: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row1_cur_inline1449__iter_v1, row_sum_tmp_iter_inline1562__ssa_v0)
            row1_rowsum_inline1469__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v42, [1, 8])
            row1_rowsum_inline1469__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row1_rowsum_inline1469__rm_a0_tmp_v2, 9.9999999999999995e-07)
            row1_rowsum_inline1469__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row1_rowsum_inline1469__row_major_tmp_v3, [8, 1])
            t__tmp_v43: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row2_cur_inline1566__iter_v1, row_sum_tmp_iter_inline1562__ssa_v0)
            row2_rowsum_inline1573__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v43, [1, 8])
            row2_rowsum_inline1573__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_84, pl.const(3584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row2_rowsum_inline1573__rm_a0_tmp_v4, 9.9999999999999995e-07)
            row2_rowsum_inline1573__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_84, pl.const(3584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row2_rowsum_inline1573__row_major_tmp_v5, [8, 1])
            t__tmp_v44: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row3_cur_inline1569__iter_v1, row_sum_tmp_iter_inline1562__ssa_v0)
            row3_rowsum_inline1574__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v44, [1, 8])
            row3_rowsum_inline1574__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_85, pl.const(3840, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row3_rowsum_inline1574__rm_a0_tmp_v6, 9.9999999999999995e-07)
            row3_rowsum_inline1574__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_85, pl.const(3840, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row3_rowsum_inline1574__row_major_tmp_v7, [8, 1])
            row0_norm_inline1576__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row0_cur_inline1565__iter_v1, row0_rowsum_inline1571__ssa_v0)
            row1_norm_inline1504__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row1_cur_inline1449__iter_v1, row1_rowsum_inline1469__ssa_v0)
            row2_norm_inline1466__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row2_cur_inline1566__iter_v1, row2_rowsum_inline1573__ssa_v0)
            row3_norm_inline1577__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_84, pl.const(3584, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row3_cur_inline1569__iter_v1, row3_rowsum_inline1574__ssa_v0)
            t__tmp_v45: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_85, pl.const(3840, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(row0_norm_inline1576__ssa_v0, row1_norm_inline1504__ssa_v0)
            t__tmp_v46: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_86, pl.const(4096, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(row2_norm_inline1466__ssa_v0, row3_norm_inline1577__ssa_v0)
            col_sum_v1_inline1564__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_85, pl.const(3840, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(t__tmp_v45, t__tmp_v46)
            col_sum_v1_inline1564__ssa_v4: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_85, pl.const(3840, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_v1_inline1564__ssa_v3, 9.9999999999999995e-07)
            row0_cur_inline1565__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_norm_inline1576__ssa_v0, col_sum_v1_inline1564__ssa_v4)
            row1_cur_inline1449__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_norm_inline1504__ssa_v0, col_sum_v1_inline1564__ssa_v4)
            row2_cur_inline1566__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_norm_inline1466__ssa_v0, col_sum_v1_inline1564__ssa_v4)
            row3_cur_inline1569__ssa_v3: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_84, pl.const(3584, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_norm_inline1577__ssa_v0, col_sum_v1_inline1564__ssa_v4)
            t__tmp_v41_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row0_cur_inline1565__ssa_v3, row_sum_tmp_iter_inline1562__ssa_v0)
            row0_rowsum_inline1571__rm_a0_tmp_v0_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v41_1, [1, 8])
            row0_rowsum_inline1571__row_major_tmp_v1_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_102, pl.const(256, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row0_rowsum_inline1571__rm_a0_tmp_v0_1, 9.9999999999999995e-07)
            row0_rowsum_inline1571__ssa_v0_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_102, pl.const(256, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row0_rowsum_inline1571__row_major_tmp_v1_1, [8, 1])
            t__tmp_v42_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row1_cur_inline1449__ssa_v3, row_sum_tmp_iter_inline1562__ssa_v0)
            row1_rowsum_inline1469__rm_a0_tmp_v2_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v42_1, [1, 8])
            row1_rowsum_inline1469__row_major_tmp_v3_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_103, pl.const(512, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row1_rowsum_inline1469__rm_a0_tmp_v2_1, 9.9999999999999995e-07)
            row1_rowsum_inline1469__ssa_v0_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_103, pl.const(512, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row1_rowsum_inline1469__row_major_tmp_v3_1, [8, 1])
            t__tmp_v43_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row2_cur_inline1566__ssa_v3, row_sum_tmp_iter_inline1562__ssa_v0)
            row2_rowsum_inline1573__rm_a0_tmp_v4_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v43_1, [1, 8])
            row2_rowsum_inline1573__row_major_tmp_v5_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_104, pl.const(768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row2_rowsum_inline1573__rm_a0_tmp_v4_1, 9.9999999999999995e-07)
            row2_rowsum_inline1573__ssa_v0_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_104, pl.const(768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row2_rowsum_inline1573__row_major_tmp_v5_1, [8, 1])
            t__tmp_v44_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row3_cur_inline1569__ssa_v3, row_sum_tmp_iter_inline1562__ssa_v0)
            row3_rowsum_inline1574__rm_a0_tmp_v6_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v44_1, [1, 8])
            row3_rowsum_inline1574__row_major_tmp_v7_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_105, pl.const(1024, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row3_rowsum_inline1574__rm_a0_tmp_v6_1, 9.9999999999999995e-07)
            row3_rowsum_inline1574__ssa_v0_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_105, pl.const(1024, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row3_rowsum_inline1574__row_major_tmp_v7_1, [8, 1])
            row0_norm_inline1576__ssa_v0_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_101, pl.const(0, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row0_cur_inline1565__ssa_v3, row0_rowsum_inline1571__ssa_v0_1)
            row1_norm_inline1504__ssa_v0_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_102, pl.const(256, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row1_cur_inline1449__ssa_v3, row1_rowsum_inline1469__ssa_v0_1)
            row2_norm_inline1466__ssa_v0_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_103, pl.const(512, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row2_cur_inline1566__ssa_v3, row2_rowsum_inline1573__ssa_v0_1)
            row3_norm_inline1577__ssa_v0_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_104, pl.const(768, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row3_cur_inline1569__ssa_v3, row3_rowsum_inline1574__ssa_v0_1)
            t__tmp_v45_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_105, pl.const(1024, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(row0_norm_inline1576__ssa_v0_1, row1_norm_inline1504__ssa_v0_1)
            t__tmp_v46_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_106, pl.const(1280, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(row2_norm_inline1466__ssa_v0_1, row3_norm_inline1577__ssa_v0_1)
            col_sum_v1_inline1564__ssa_v3_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_105, pl.const(1024, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(t__tmp_v45_1, t__tmp_v46_1)
            col_sum_v1_inline1564__ssa_v4_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_v1_inline1564__ssa_v3_1, 9.9999999999999995e-07)
            row0_cur_inline1565__ssa_v3_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_norm_inline1576__ssa_v0_1, col_sum_v1_inline1564__ssa_v4_1)
            row1_cur_inline1449__ssa_v3_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_norm_inline1504__ssa_v0_1, col_sum_v1_inline1564__ssa_v4_1)
            row2_cur_inline1566__ssa_v3_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_norm_inline1466__ssa_v0_1, col_sum_v1_inline1564__ssa_v4_1)
            row3_cur_inline1569__ssa_v3_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_norm_inline1577__ssa_v0_1, col_sum_v1_inline1564__ssa_v4_1)
            col_sum_v1_inline1564__rv_v2_main, row0_cur_inline1565__rv_v2_main, row1_cur_inline1449__rv_v2_main, row2_cur_inline1566__rv_v2_main, row3_cur_inline1569__rv_v2_main = pl.yield_(col_sum_v1_inline1564__ssa_v4_1, row0_cur_inline1565__ssa_v3_1, row1_cur_inline1449__ssa_v3_1, row2_cur_inline1566__ssa_v3_1, row3_cur_inline1569__ssa_v3_1)
        t__tmp_v41_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row0_cur_inline1565__rv_v2_main, row_sum_tmp_iter_inline1562__ssa_v0)
        row0_rowsum_inline1571__rm_a0_tmp_v0_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v41_2, [1, 8])
        row0_rowsum_inline1571__row_major_tmp_v1_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row0_rowsum_inline1571__rm_a0_tmp_v0_2, 9.9999999999999995e-07)
        row0_rowsum_inline1571__ssa_v0_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row0_rowsum_inline1571__row_major_tmp_v1_2, [8, 1])
        t__tmp_v42_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row1_cur_inline1449__rv_v2_main, row_sum_tmp_iter_inline1562__ssa_v0)
        row1_rowsum_inline1469__rm_a0_tmp_v2_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v42_2, [1, 8])
        row1_rowsum_inline1469__row_major_tmp_v3_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row1_rowsum_inline1469__rm_a0_tmp_v2_2, 9.9999999999999995e-07)
        row1_rowsum_inline1469__ssa_v0_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_66, pl.const(2304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row1_rowsum_inline1469__row_major_tmp_v3_2, [8, 1])
        t__tmp_v43_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row2_cur_inline1566__rv_v2_main, row_sum_tmp_iter_inline1562__ssa_v0)
        row2_rowsum_inline1573__rm_a0_tmp_v4_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v43_2, [1, 8])
        row2_rowsum_inline1573__row_major_tmp_v5_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row2_rowsum_inline1573__rm_a0_tmp_v4_2, 9.9999999999999995e-07)
        row2_rowsum_inline1573__ssa_v0_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_82, pl.const(3072, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row2_rowsum_inline1573__row_major_tmp_v5_2, [8, 1])
        t__tmp_v44_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(row3_cur_inline1569__rv_v2_main, row_sum_tmp_iter_inline1562__ssa_v0)
        row3_rowsum_inline1574__rm_a0_tmp_v6_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_83, pl.const(3328, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v44_2, [1, 8])
        row3_rowsum_inline1574__row_major_tmp_v7_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(row3_rowsum_inline1574__rm_a0_tmp_v6_2, 9.9999999999999995e-07)
        row3_rowsum_inline1574__ssa_v0_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(row3_rowsum_inline1574__row_major_tmp_v7_2, [8, 1])
        row0_norm_inline1576__ssa_v0_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row0_cur_inline1565__rv_v2_main, row0_rowsum_inline1571__ssa_v0_2)
        row1_norm_inline1504__ssa_v0_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row1_cur_inline1449__rv_v2_main, row1_rowsum_inline1469__ssa_v0_2)
        row2_norm_inline1466__ssa_v0_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row2_cur_inline1566__rv_v2_main, row2_rowsum_inline1573__ssa_v0_2)
        row3_norm_inline1577__ssa_v0_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.row_expand_div(row3_cur_inline1569__rv_v2_main, row3_rowsum_inline1574__ssa_v0_2)
        t__tmp_v45_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(row0_norm_inline1576__ssa_v0_2, row1_norm_inline1504__ssa_v0_2)
        t__tmp_v46_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_35, pl.const(1792, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(row2_norm_inline1466__ssa_v0_2, row3_norm_inline1577__ssa_v0_2)
        col_sum_v1_inline1564__ssa_v3_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.add(t__tmp_v45_2, t__tmp_v46_2)
        col_sum_v1_inline1564__ssa_v4_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.adds(col_sum_v1_inline1564__ssa_v3_2, 9.9999999999999995e-07)
        row0_cur_inline1565__ssa_v3_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row0_norm_inline1576__ssa_v0_2, col_sum_v1_inline1564__ssa_v4_2)
        row1_cur_inline1449__ssa_v3_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row1_norm_inline1504__ssa_v0_2, col_sum_v1_inline1564__ssa_v4_2)
        row2_cur_inline1566__ssa_v3_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row2_norm_inline1466__ssa_v0_2, col_sum_v1_inline1564__ssa_v4_2)
        row3_cur_inline1569__ssa_v3_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = pl.tile.div(row3_norm_inline1577__ssa_v0_2, col_sum_v1_inline1564__ssa_v4_2)
        col_sum_v1_inline1564__rv_v2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(1536, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = col_sum_v1_inline1564__ssa_v4_2
        row0_cur_inline1565__rv_v2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = row0_cur_inline1565__ssa_v3_2
        row1_cur_inline1449__rv_v2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = row1_cur_inline1449__ssa_v3_2
        row2_cur_inline1566__rv_v2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = row2_cur_inline1566__ssa_v3_2
        row3_cur_inline1569__rv_v2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.zero)] = row3_cur_inline1569__ssa_v3_2
        if valid_rows_inline1507__ssa_v2 == 8:
            row0_out_inline1541__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row0_cur_inline1565__rv_v2, 8, 4)
            row1_out_inline1536__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row1_cur_inline1449__rv_v2, 8, 4)
            row2_out_inline1558__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row2_cur_inline1566__rv_v2, 8, 4)
            row3_out_inline1474__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4], pad=pl.PadValue.zero)] = pl.tile.set_validshape(row3_cur_inline1569__rv_v2, 8, 4)
            comb_t_inline1267__store: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(row0_out_inline1541__ssa_v0, [t0_inline1476__ssa_v3, 0], comb_t_inline1267__ssa_v0)
            comb_t_inline1267__store_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(row1_out_inline1536__ssa_v0, [t0_inline1476__ssa_v3, 4], comb_t_inline1267__ssa_v0)
            comb_t_inline1267__store_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(row2_out_inline1558__ssa_v0, [t0_inline1476__ssa_v3, 8], comb_t_inline1267__ssa_v0)
            comb_t_inline1267__store_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(row3_out_inline1474__ssa_v0, [t0_inline1476__ssa_v3, 12], comb_t_inline1267__ssa_v0)
        else:
            comb_tail_store_inline1523__store: pl.Tensor[[8, 32], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1024)] = pl.tile.store(row0_cur_inline1565__rv_v2, [0, 0], comb_tail_store_inline1523__ssa_v0)
            comb_tail_store_inline1523__store_v0: pl.Tensor[[8, 32], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1024)] = pl.tile.store(row1_cur_inline1449__rv_v2, [0, 8], comb_tail_store_inline1523__ssa_v0)
            comb_tail_store_inline1523__store_v1: pl.Tensor[[8, 32], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1024)] = pl.tile.store(row2_cur_inline1566__rv_v2, [0, 16], comb_tail_store_inline1523__ssa_v0)
            comb_tail_store_inline1523__store_v2: pl.Tensor[[8, 32], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1024)] = pl.tile.store(row3_cur_inline1569__rv_v2, [0, 24], comb_tail_store_inline1523__ssa_v0)
            row0_tail_inline1556__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(2048, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(comb_tail_store_inline1523__ssa_v0, [0, 0], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
            row1_tail_inline1578__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(2560, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(comb_tail_store_inline1523__ssa_v0, [0, 8], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
            row2_tail_inline1500__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(2816, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(comb_tail_store_inline1523__ssa_v0, [0, 16], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
            row3_tail_inline1451__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(4352, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v2, 4])] = pl.tile.load(comb_tail_store_inline1523__ssa_v0, [0, 24], [8, 8], [valid_rows_inline1507__ssa_v2, 4], target_memory=pl.Mem.Vec)
            comb_t_inline1267__store_v3: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(row0_tail_inline1556__ssa_v0, [t0_inline1476__ssa_v3, 0], comb_t_inline1267__ssa_v0)
            comb_t_inline1267__store_v4: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(row1_tail_inline1578__ssa_v0, [t0_inline1476__ssa_v3, 4], comb_t_inline1267__ssa_v0)
            comb_t_inline1267__store_v5: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(row2_tail_inline1500__ssa_v0, [t0_inline1476__ssa_v3, 8], comb_t_inline1267__ssa_v0)
            comb_t_inline1267__store_v6: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(row3_tail_inline1451__ssa_v0, [t0_inline1476__ssa_v3, 12], comb_t_inline1267__ssa_v0)
        return comb_t_inline1267__ssa_v0, comb_tail_store_inline1523__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def comb_sinkhorn_spmd(self, t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], inv_rms_inline1463__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], mixes_raw_inline1505__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], hc_base_2d_inline1467__ssa_v0: pl.Tensor[[1, 24], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 96)], scale2_inline1480__ssa_v0: pl.Scalar[pl.FP32], comb_t_inline1267__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], comb_tail_store_inline1523__ssa_v0: pl.InOut[pl.Tensor[[8, 32], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1024)]]) -> pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32]:
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32], pl.Tensor[[8, 32], pl.FP32]] = self.comb_sinkhorn(t_dim_inline1568__ssa_v0, inv_rms_inline1463__ssa_v1, mixes_raw_inline1505__ssa_v1, hc_base_2d_inline1467__ssa_v0, scale2_inline1480__ssa_v0, comb_t_inline1267__ssa_v0, comb_tail_store_inline1523__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.inout]})
        comb_t_inline1267__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        comb_tail_store_inline1523__ssa_v1: pl.Tensor[[8, 32], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 1024)] = ret__tmp_v0[1]
        return comb_t_inline1267__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit(compress_state_flat_inline2023__ssa_v0: pl.Out[pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], s_dim_inline2020__ssa_v0: pl.Scalar[pl.INDEX], cmp_state_slots_inline1247__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], cmp4_kv_proj_pad_inline2031__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)], cmp4_score_proj_pad_inline2019__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)]):
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        c_idx_v1_inline2030__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for s_idx_inline2009__idx_v0, (compress_state_flat_inline2023__iter_v1,) in pl.range(s_dim_inline2020__ssa_v0, init_values=(compress_state_flat_inline2023__ssa_v0,)):
            token_inline2040__ssa_v1: pl.Scalar[pl.INDEX] = c_idx_v1_inline2030__ssa_v0 * s_dim_inline2020__ssa_v0 + s_idx_inline2009__idx_v0
            state_row_i64_inline2006__tile: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_state_slots_inline1247__ssa_v0, [token_inline2040__ssa_v1])
            if 0 <= state_row_i64_inline2006__tile:
                state_row_inline2058__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline2006__tile, pl.INDEX)
                token_pos_inline2041__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [token_inline2040__ssa_v1])
                ape_row_inline2034__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2041__tile, pl.INDEX) % 4, pl.INDEX)
                t__tile: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(cmp4_kv_proj_pad_inline2031__ssa_v1, [token_inline2040__ssa_v1, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                compress_state_flat_inline2023__tile: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile, [state_row_inline2058__ssa_v1, 0], compress_state_flat_inline2023__iter_v1)
                t__tile_1: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(cmp4_score_proj_pad_inline2019__ssa_v1, [token_inline2040__ssa_v1, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                t__tile_2: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(cmp_ape__ssa_v0, [ape_row_inline2034__ssa_v1, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                t__tile_3: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                compress_state_flat_inline2023__tile_1: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_3, [state_row_inline2058__ssa_v1, 1024], compress_state_flat_inline2023__tile)
                compress_state_flat_inline2023__phi_v5: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(compress_state_flat_inline2023__tile_1)
            else:
                compress_state_flat_inline2023__phi_v5: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(compress_state_flat_inline2023__iter_v1)
            compress_state_flat_inline2023__rv_v2: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.yield_(compress_state_flat_inline2023__phi_v5)
        return
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def compress_state_commit_0(compress_state_flat_inline2139__ssa_v0: pl.Out[pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], s_dim_inline2127__ssa_v0: pl.Scalar[pl.INDEX], inner_state_slots_inline1257__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], kv_proj_pad_inline2129__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 524288)], score_proj_pad_inline2143__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)]):
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        c_idx_v1_inline2153__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for s_idx_inline2121__idx_v0, (compress_state_flat_inline2139__iter_v1,) in pl.range(s_dim_inline2127__ssa_v0, init_values=(compress_state_flat_inline2139__ssa_v0,)):
            token_inline2123__ssa_v1: pl.Scalar[pl.INDEX] = c_idx_v1_inline2153__ssa_v0 * s_dim_inline2127__ssa_v0 + s_idx_inline2121__idx_v0
            state_row_i64_inline2137__tile: pl.Scalar[pl.INT64] = pl.tensor.read(inner_state_slots_inline1257__ssa_v0, [token_inline2123__ssa_v1])
            if 0 <= state_row_i64_inline2137__tile:
                state_row_inline2166__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(state_row_i64_inline2137__tile, pl.INDEX)
                token_pos_inline2120__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [token_inline2123__ssa_v1])
                ape_row_inline2136__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2120__tile, pl.INDEX) % 4, pl.INDEX)
                t__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(kv_proj_pad_inline2129__ssa_v1, [token_inline2123__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                compress_state_flat_inline2139__tile: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile, [state_row_inline2166__ssa_v1, 0], compress_state_flat_inline2139__iter_v1)
                t__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(score_proj_pad_inline2143__ssa_v1, [token_inline2123__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                t__tile_2: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(inner_ape__ssa_v0, [ape_row_inline2136__ssa_v1, 0], [1, 256], [1, 256], target_memory=pl.Mem.Vec)
                t__tile_3: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_6, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                compress_state_flat_inline2139__tile_1: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_3, [state_row_inline2166__ssa_v1, 256], compress_state_flat_inline2139__tile)
                compress_state_flat_inline2139__phi_v5: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(compress_state_flat_inline2139__tile_1)
            else:
                compress_state_flat_inline2139__phi_v5: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(compress_state_flat_inline2139__iter_v1)
            compress_state_flat_inline2139__rv_v2: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.yield_(compress_state_flat_inline2139__phi_v5)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd(self, compress_state_flat_inline2023__ssa_v0: pl.Out[pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], s_dim_inline2020__ssa_v0: pl.Scalar[pl.INDEX], cmp_state_slots_inline1247__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], cmp4_kv_proj_pad_inline2031__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)], cmp4_score_proj_pad_inline2019__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16384)]):
        self.compress_state_commit(compress_state_flat_inline2023__ssa_v0, s_dim_inline2020__ssa_v0, cmp_state_slots_inline1247__ssa_v0, cmp_positions_inline1320__ssa_v0, cmp4_kv_proj_pad_inline2031__ssa_v1, cmp4_score_proj_pad_inline2019__ssa_v1, cmp_ape__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
    @pl.function(type=pl.FunctionType.Spmd)
    def compress_state_commit_spmd_0(self, compress_state_flat_inline2139__ssa_v0: pl.Out[pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], s_dim_inline2127__ssa_v0: pl.Scalar[pl.INDEX], inner_state_slots_inline1257__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], kv_proj_pad_inline2129__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 524288)], score_proj_pad_inline2143__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 4096)]):
        self.compress_state_commit_0(compress_state_flat_inline2139__ssa_v0, s_dim_inline2127__ssa_v0, inner_state_slots_inline1257__ssa_v0, cmp_positions_inline1320__ssa_v0, kv_proj_pad_inline2129__ssa_v1, score_proj_pad_inline2143__ssa_v1, inner_ape__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def cp_token_allgather_payload_wait(tp_rank__ssa_v0: pl.Scalar[pl.INT32], gather_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8)], gather_signal_ctx: pld.CommCtx):
        for source_tp_inline1637__idx_v0 in pl.range(2):
            if source_tp_inline1637__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.wait(gather_signal__ssa_v0, [source_tp_inline1637__idx_v0, 0], pl.cast(16, pl.INT32), cmp=1)
        pl.system.cacheinvalid()
        return
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def cp_token_allgather_push(full_local_inline1644__ssa_v0: pl.Scalar[pl.INDEX], gather_window__ssa_v0: pl.Out[pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)]], group_base__ssa_v0: pl.Scalar[pl.INT32], x_normed_t_inline1243__phi_v4: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], target_row_inline1642__ssa_v0: pl.Scalar[pl.INT32], local_t_inline1640__ssa_v0: pl.Scalar[pl.INT32], tp_rank__ssa_v0: pl.Scalar[pl.INT32], gather_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8)]], gather_window_ctx: pld.CommCtx, gather_signal_ctx: pld.CommCtx):
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        worker_inline1647__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for peer_tp_inline1635__idx_v0 in pl.range(2):
            for band_row_inline1648__idx_v0 in pl.range(worker_inline1647__ssa_v0 * 8, full_local_inline1644__ssa_v0, 128):
                tput_stage: pl.Tile[[8, 4096], pl.BF16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.create([8, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                pld.tile.put(gather_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1635__idx_v0, x_normed_t_inline1243__phi_v4, tput_stage, [pl.cast(target_row_inline1642__ssa_v0, pl.INDEX) + band_row_inline1648__idx_v0, 0], [band_row_inline1648__idx_v0, 0], [8, 4096], atomic=pl.AtomicType.None_)
                pl.system.fence()
            for tail_row_inline1651__idx_v0 in pl.range(full_local_inline1644__ssa_v0 + worker_inline1647__ssa_v0, local_t_inline1640__ssa_v0, 16):
                tput_stage_1: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([1, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
                pld.tile.put(gather_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1635__idx_v0, x_normed_t_inline1243__phi_v4, tput_stage_1, [pl.cast(target_row_inline1642__ssa_v0, pl.INDEX) + tail_row_inline1651__idx_v0, 0], [tail_row_inline1651__idx_v0, 0], [1, 4096], atomic=pl.AtomicType.None_)
                pl.system.fence()
        for peer_tp_inline1636__idx_v0 in pl.range(2):
            if peer_tp_inline1636__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(gather_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1636__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def cp_token_allgather_push_spmd(self, full_local_inline1644__ssa_v0: pl.Scalar[pl.INDEX], gather_window__ssa_v0: pl.Out[pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)]], group_base__ssa_v0: pl.Scalar[pl.INT32], x_normed_t_inline1243__phi_v4: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], target_row_inline1642__ssa_v0: pl.Scalar[pl.INT32], local_t_inline1640__ssa_v0: pl.Scalar[pl.INT32], tp_rank__ssa_v0: pl.Scalar[pl.INT32], gather_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8)]], gather_window_ctx: pld.CommCtx, gather_signal_ctx: pld.CommCtx):
        self.cp_token_allgather_push(full_local_inline1644__ssa_v0, gather_window__ssa_v0, group_base__ssa_v0, x_normed_t_inline1243__phi_v4, target_row_inline1642__ssa_v0, local_t_inline1640__ssa_v0, tp_rank__ssa_v0, gather_signal__ssa_v0, gather_window_ctx, gather_signal_ctx, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def cp_token_allgather_readback(x_normed_full_inline1240__ssa_v0: pl.Out[pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], full_rows_inline1641__ssa_v0: pl.Scalar[pl.INDEX], gather_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 4194304)], group_rows_inline1639__ssa_v0: pl.Scalar[pl.INDEX], tp_rank__ssa_v0: pl.Scalar[pl.INT32], gather_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8)]], group_base__ssa_v0: pl.Scalar[pl.INT32], gather_window_ctx: pld.CommCtx, gather_signal_ctx: pld.CommCtx) -> pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16]:
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 131072)
        worker_v1_inline1638__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for tile_row_inline1632__idx_v0, (x_normed_full_inline1240__iter_v1,) in pl.range(worker_v1_inline1638__ssa_v0 * 16, full_rows_inline1641__ssa_v0, 256, init_values=(x_normed_full_inline1240__ssa_v0,)):
            window_tile_inline1650__tile: pl.Tile[[16, 4096], pl.BF16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 131072), pl.Mem.Vec] = pl.tile.load(gather_window__ssa_v0, [tile_row_inline1632__idx_v0, 0], [16, 4096], [16, 4096], target_memory=pl.Mem.Vec)
            x_normed_full_inline1240__tile: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(window_tile_inline1650__tile, [tile_row_inline1632__idx_v0, 0], x_normed_full_inline1240__iter_v1)
            x_normed_full_inline1240__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.yield_(x_normed_full_inline1240__tile)
        for tail_row_inline1629__idx_v0, (x_normed_full_inline1240__iter_v4,) in pl.range(full_rows_inline1641__ssa_v0 + worker_v1_inline1638__ssa_v0, group_rows_inline1639__ssa_v0, 16, init_values=(x_normed_full_inline1240__rv_v2,)):
            window_row_inline1631__tile: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(gather_window__ssa_v0, [tail_row_inline1629__idx_v0, 0], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
            x_normed_full_inline1240__tile_1: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(window_row_inline1631__tile, [tail_row_inline1629__idx_v0, 0], x_normed_full_inline1240__iter_v4)
            x_normed_full_inline1240__rv_v5: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = pl.yield_(x_normed_full_inline1240__tile_1)
        for peer_tp_inline1628__idx_v0 in pl.range(2):
            if peer_tp_inline1628__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(gather_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1628__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
        return x_normed_full_inline1240__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def cp_token_allgather_readback_spmd(self, x_normed_full_inline1240__ssa_v0: pl.Out[pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], full_rows_inline1641__ssa_v0: pl.Scalar[pl.INDEX], gather_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 4194304)], group_rows_inline1639__ssa_v0: pl.Scalar[pl.INDEX], tp_rank__ssa_v0: pl.Scalar[pl.INT32], gather_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8)]], group_base__ssa_v0: pl.Scalar[pl.INT32], gather_window_ctx: pld.CommCtx, gather_signal_ctx: pld.CommCtx) -> pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16]:
        x_normed_full_inline1240__rv_v5: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.cp_token_allgather_readback(x_normed_full_inline1240__ssa_v0, full_rows_inline1641__ssa_v0, gather_window__ssa_v0, group_rows_inline1639__ssa_v0, tp_rank__ssa_v0, gather_signal__ssa_v0, group_base__ssa_v0, gather_window_ctx, gather_signal_ctx, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar]})
        return x_normed_full_inline1240__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def cp_token_allgather_readback_wait(tp_rank__ssa_v0: pl.Scalar[pl.INT32], gather_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8)], gather_signal_ctx: pld.CommCtx):
        for source_tp_inline1627__idx_v0 in pl.range(2):
            if source_tp_inline1627__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.wait(gather_signal__ssa_v0, [source_tp_inline1627__idx_v0, 0], pl.cast(32, pl.INT32), cmp=1)
        pl.system.cacheinvalid()
        return
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def cp_token_allgather_retire(x_normed_full_inline1240__rv_v5: pl.InOut[pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], group_base__ssa_v0: pl.Scalar[pl.INT32], tp_rank__ssa_v0: pl.Scalar[pl.INT32], gather_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 8)]], gather_signal_ctx: pld.CommCtx):
        completion_anchor_inline1626__tile: pl.Scalar[pl.BF16] = pl.tensor.read(x_normed_full_inline1240__rv_v5, [0, 0])
        reset_value_inline1649__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(-32, pl.INT32)
        self_rank_inline1625__ssa_v0: pl.Scalar[pl.INT32] = group_base__ssa_v0 + tp_rank__ssa_v0
        for source_tp_inline1645__idx_v0 in pl.range(2):
            if source_tp_inline1645__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(gather_signal__ssa_v0, self_rank_inline1625__ssa_v0, [source_tp_inline1645__idx_v0, 0], reset_value_inline1649__ssa_v0, op=0)
        pl.tensor.write(x_normed_full_inline1240__rv_v5, [0, 0], completion_anchor_inline1626__tile)
        return
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_cache_writeback(kv_cache_flat_inline1312__ssa_v0: pl.Out[pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], ori_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], kv_full_inline1265__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]):
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        wb_blk_inline1315__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        wb_t0_inline1317__ssa_v0: pl.Scalar[pl.INDEX] = wb_blk_inline1315__ssa_v0 * 8
        for write_dt_inline1318__idx_v0, (kv_cache_flat_inline1312__iter_v1,) in pl.range(8, init_values=(kv_cache_flat_inline1312__ssa_v0,)):
            write_t_inline1308__ssa_v0: pl.Scalar[pl.INDEX] = wb_t0_inline1317__ssa_v0 + write_dt_inline1318__idx_v0
            write_row_i64_inline1319__tile: pl.Scalar[pl.INT64] = pl.tensor.read(ori_slot_mapping__ssa_v0, [write_t_inline1308__ssa_v0])
            if 0 <= write_row_i64_inline1319__tile:
                write_row_inline1268__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(write_row_i64_inline1319__tile, pl.INDEX)
                t__tile: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(kv_full_inline1265__ssa_v0, [write_t_inline1308__ssa_v0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                kv_cache_flat_inline1312__tile: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile, [write_row_inline1268__ssa_v0, 0], kv_cache_flat_inline1312__iter_v1)
                kv_cache_flat_inline1312__phi_v4: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_cache_flat_inline1312__tile)
            else:
                kv_cache_flat_inline1312__phi_v4: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_cache_flat_inline1312__iter_v1)
            kv_cache_flat_inline1312__rv_v2: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_cache_flat_inline1312__phi_v4)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def csa_cache_writeback_spmd(self, kv_cache_flat_inline1312__ssa_v0: pl.Out[pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], ori_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], kv_full_inline1265__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]):
        self.csa_cache_writeback(kv_cache_flat_inline1312__ssa_v0, ori_slot_mapping__ssa_v0, kv_full_inline1265__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.input]})
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_merge_pack_publish(attention_grouped_inline1276__ssa_v0: pl.InOut[pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], pack_work_count_inline1228__ssa_v0: pl.Scalar[pl.INDEX], sparse_blk_mi_inline1234__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], sparse_blk_li_inline1283__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], sparse_blk_oi_inline1233__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)], rope_cos_il_inline1232__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 65536)], rope_sin_signed_inline1231__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 65536)], rope_swap_idx_inline1230__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 4096)], tp_rank__ssa_v0: pl.Scalar[pl.INT32], attention_window__ssa_v0: pl.Out[pld.DistributedTensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 16777216)]], group_base__ssa_v0: pl.Scalar[pl.INT32], attention_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8)]], attention_window_ctx: pld.CommCtx, attention_signal_ctx: pld.CommCtx):
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_20: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_52: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_53: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_56: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_57: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_66: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
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
                m_blk_base_inline1306__ssa_v0: pl.Scalar[pl.INDEX] = m_idx_inline1289__ssa_v0 * 80
                m_mi_inline1221__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_mi_inline1234__ssa_v0, [m_blk_base_inline1306__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                m_li_inline1302__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_li_inline1283__ssa_v0, [m_blk_base_inline1306__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                m_oi_inline1220__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(sparse_blk_oi_inline1233__ssa_v0, [m_blk_base_inline1306__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec)
                for m_sb_inline1219__idx_v0, (m_li_inline1302__iter_v1, m_mi_inline1221__iter_v1, m_oi_inline1220__iter_v1) in pl.range(1, 5, 2, init_values=(m_li_inline1302__tile, m_mi_inline1221__tile, m_oi_inline1220__tile)):
                    m_row_inline1218__ssa_v0: pl.Scalar[pl.INDEX] = m_blk_base_inline1306__ssa_v0 + m_sb_inline1219__idx_v0 * 16
                    m_row_inline1218__ssa_v0_1: pl.Scalar[pl.INDEX] = m_blk_base_inline1306__ssa_v0 + (m_sb_inline1219__idx_v0 * 16 + 16)
                    m_cur_mi_inline1217__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_mi_inline1234__ssa_v0, [m_row_inline1218__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    m_cur_li_inline1293__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_li_inline1283__ssa_v0, [m_row_inline1218__ssa_v0, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    m_cur_oi_inline1216__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(sparse_blk_oi_inline1233__ssa_v0, [m_row_inline1218__ssa_v0, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec)
                    m_cur_mi_inline1217__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_mi_inline1234__ssa_v0, [m_row_inline1218__ssa_v0_1, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    m_cur_li_inline1293__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(sparse_blk_li_inline1283__ssa_v0, [m_row_inline1218__ssa_v0_1, 0], [16, 1], [16, 1], target_memory=pl.Mem.Vec)
                    m_cur_oi_inline1216__tile_1: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(sparse_blk_oi_inline1233__ssa_v0, [m_row_inline1218__ssa_v0_1, 0], [16, 512], [16, 512], target_memory=pl.Mem.Vec)
                    m_mi_new_inline1215__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__iter_v1, [1, 16])
                    m_mi_new_inline1215__rm_a1_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_mi_inline1217__tile, [1, 16])
                    m_mi_new_inline1215__row_major_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(m_mi_new_inline1215__rm_a0_tmp_v0, m_mi_new_inline1215__rm_a1_tmp_v1)
                    m_mi_new_inline1215__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__row_major_tmp_v2, [16, 1])
                    m_mi_inline1221__ssa_v3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = m_mi_new_inline1215__tile
                    t__rm_a0_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__iter_v1, [1, 16])
                    t__rm_a1_tmp_v4: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__tile, [1, 16])
                    t__row_major_tmp_v5: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v3, t__rm_a1_tmp_v4)
                    t__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [16, 1])
                    m_alpha_inline1214__rm_a0_tmp_v6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 16])
                    m_alpha_inline1214__row_major_tmp_v7: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(m_alpha_inline1214__rm_a0_tmp_v6)
                    m_alpha_inline1214__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_alpha_inline1214__row_major_tmp_v7, [16, 1])
                    t__rm_a0_tmp_v8: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_mi_inline1217__tile, [1, 16])
                    t__rm_a1_tmp_v9: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__tile, [1, 16])
                    t__row_major_tmp_v10: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v8, t__rm_a1_tmp_v9)
                    t__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v10, [16, 1])
                    m_beta_inline1213__rm_a0_tmp_v11: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 16])
                    m_beta_inline1213__row_major_tmp_v12: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_23, pl.const(65728, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(m_beta_inline1213__rm_a0_tmp_v11)
                    m_beta_inline1213__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(65728, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_beta_inline1213__row_major_tmp_v12, [16, 1])
                    t__rm_a0_tmp_v13: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_20, pl.const(65664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_alpha_inline1214__tile, [1, 16])
                    t__rm_a1_tmp_v14: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__iter_v1, [1, 16])
                    t__row_major_tmp_v15: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v13, t__rm_a1_tmp_v14)
                    t__tile_2: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v15, [16, 1])
                    t__rm_a0_tmp_v16: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_23, pl.const(65728, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_beta_inline1213__tile, [1, 16])
                    t__rm_a1_tmp_v17: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_li_inline1293__tile, [1, 16])
                    t__row_major_tmp_v18: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v16, t__rm_a1_tmp_v17)
                    t__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v18, [16, 1])
                    m_li_inline1302__rm_a0_tmp_v19: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 16])
                    m_li_inline1302__rm_a1_tmp_v20: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 16])
                    m_li_inline1302__row_major_tmp_v21: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(m_li_inline1302__rm_a0_tmp_v19, m_li_inline1302__rm_a1_tmp_v20)
                    m_li_inline1302__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__row_major_tmp_v21, [16, 1])
                    t__tile_4: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(m_oi_inline1220__iter_v1, m_alpha_inline1214__tile)
                    t__tile_5: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(m_cur_oi_inline1216__tile, m_beta_inline1213__tile)
                    m_oi_inline1220__tile_1: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                    m_mi_new_inline1215__rm_a0_tmp_v0_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__ssa_v3, [1, 16])
                    m_mi_new_inline1215__rm_a1_tmp_v1_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_mi_inline1217__tile_1, [1, 16])
                    m_mi_new_inline1215__row_major_tmp_v2_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(m_mi_new_inline1215__rm_a0_tmp_v0_1, m_mi_new_inline1215__rm_a1_tmp_v1_1)
                    m_mi_new_inline1215__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__row_major_tmp_v2_1, [16, 1])
                    m_mi_inline1221__ssa_v3_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = m_mi_new_inline1215__tile_1
                    t__rm_a0_tmp_v3_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__ssa_v3, [1, 16])
                    t__rm_a1_tmp_v4_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__tile_1, [1, 16])
                    t__row_major_tmp_v5_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v3_1, t__rm_a1_tmp_v4_1)
                    t__tile_6: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5_1, [16, 1])
                    m_alpha_inline1214__rm_a0_tmp_v6_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 16])
                    m_alpha_inline1214__row_major_tmp_v7_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(m_alpha_inline1214__rm_a0_tmp_v6_1)
                    m_alpha_inline1214__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_alpha_inline1214__row_major_tmp_v7_1, [16, 1])
                    t__rm_a0_tmp_v8_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_mi_inline1217__tile_1, [1, 16])
                    t__rm_a1_tmp_v9_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_new_inline1215__tile_1, [1, 16])
                    t__row_major_tmp_v10_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v8_1, t__rm_a1_tmp_v9_1)
                    t__tile_7: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v10_1, [16, 1])
                    m_beta_inline1213__rm_a0_tmp_v11_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_7, [1, 16])
                    m_beta_inline1213__row_major_tmp_v12_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(m_beta_inline1213__rm_a0_tmp_v11_1)
                    m_beta_inline1213__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_beta_inline1213__row_major_tmp_v12_1, [16, 1])
                    t__rm_a0_tmp_v13_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_19, pl.const(65600, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_alpha_inline1214__tile_1, [1, 16])
                    t__rm_a1_tmp_v14_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__tile_1, [1, 16])
                    t__row_major_tmp_v15_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v13_1, t__rm_a1_tmp_v14_1)
                    t__tile_8: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v15_1, [16, 1])
                    t__rm_a0_tmp_v16_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_beta_inline1213__tile_1, [1, 16])
                    t__rm_a1_tmp_v17_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_cur_li_inline1293__tile_1, [1, 16])
                    t__row_major_tmp_v18_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.mul(t__rm_a0_tmp_v16_1, t__rm_a1_tmp_v17_1)
                    t__tile_9: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v18_1, [16, 1])
                    m_li_inline1302__rm_a0_tmp_v19_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_8, [1, 16])
                    m_li_inline1302__rm_a1_tmp_v20_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32768, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [1, 16])
                    m_li_inline1302__row_major_tmp_v21_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(m_li_inline1302__rm_a0_tmp_v19_1, m_li_inline1302__rm_a1_tmp_v20_1)
                    m_li_inline1302__tile_2: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__row_major_tmp_v21_1, [16, 1])
                    t__tile_10: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(m_oi_inline1220__tile_1, m_alpha_inline1214__tile_1)
                    t__tile_11: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(m_cur_oi_inline1216__tile_1, m_beta_inline1213__tile_1)
                    m_oi_inline1220__tile_2: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(t__tile_10, t__tile_11)
                    m_mi_inline1221__ssa_v3_mv: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.move(m_mi_inline1221__ssa_v3_1, target_memory=pl.Mem.Vec)
                    m_li_inline1302__tile_mv: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 64), pl.Mem.Vec] = pl.tile.move(m_li_inline1302__tile_2, target_memory=pl.Mem.Vec)
                    m_li_inline1302__rv_v2, m_mi_inline1221__rv_v2, m_oi_inline1220__rv_v2 = pl.yield_(m_li_inline1302__tile_mv, m_mi_inline1221__ssa_v3_mv, m_oi_inline1220__tile_2)
                t__tile_12: pl.Tile[[16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.load(attn_sink__ssa_v0, [m_h0_inline1224__ssa_v0], [16], [16], target_memory=pl.Mem.Vec)
                n_sink_bias_inline1266__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_12, [16, 1])
                t__rm_a0_tmp_v22: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__rv_v2, [1, 16])
                t__rm_a1_tmp_v23: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__rv_v2, [1, 16])
                t__row_major_tmp_v24: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v22, t__rm_a1_tmp_v23)
                t__tile_13: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v24, [16, 1])
                n_sink_tile_inline1212__rm_a0_tmp_v25: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_13, [1, 16])
                n_sink_tile_inline1212__rm_a1_tmp_v26: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_bias_inline1266__tile, [1, 16])
                n_sink_tile_inline1212__row_major_tmp_v27: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(n_sink_tile_inline1212__rm_a0_tmp_v25, n_sink_tile_inline1212__rm_a1_tmp_v26)
                n_sink_tile_inline1212__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_tile_inline1212__row_major_tmp_v27, [16, 1])
                t__rm_a0_tmp_v28: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_sink_tile_inline1212__tile, [1, 16])
                t__rm_a1_tmp_v29: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_mi_inline1221__rv_v2, [1, 16])
                t__row_major_tmp_v30: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sub(t__rm_a0_tmp_v28, t__rm_a1_tmp_v29)
                t__tile_14: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v30, [16, 1])
                t__rm_a0_tmp_v31: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_14, [1, 16])
                t__row_major_tmp_v32: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.exp(t__rm_a0_tmp_v31)
                t__tile_15: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v32, [16, 1])
                n_denom_inline1254__rm_a0_tmp_v33: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(m_li_inline1302__rv_v2, [1, 16])
                n_denom_inline1254__rm_a1_tmp_v34: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_15, [1, 16])
                n_denom_inline1254__row_major_tmp_v35: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(n_denom_inline1254__rm_a0_tmp_v33, n_denom_inline1254__rm_a1_tmp_v34)
                n_denom_inline1254__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(n_denom_inline1254__row_major_tmp_v35, [16, 1])
                t__tile_16: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_div(m_oi_inline1220__rv_v2, n_denom_inline1254__tile)
                n_full_inline1211__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.slice(t__tile_16, [16, 512], [0, 0])
                n_bf16_inline1210__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(n_full_inline1211__tile, target_type=pl.BF16, mode='rint')
                m_rope_inline1208__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_66, pl.const(101376, pl.INT64), 30976), pl.Mem.Vec] = pl.tile.slice(n_full_inline1211__tile, [16, 64], [0, 448])
                m_cos_il_inline1286__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_52, pl.const(98560, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(rope_cos_il_inline1232__ssa_v0, [m_t_inline1290__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                m_sin_signed_inline1225__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_53, pl.const(98816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(rope_sin_signed_inline1231__ssa_v0, [m_t_inline1290__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                t__tile_17: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(rope_swap_idx_inline1230__ssa_v0, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
                gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
                    gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(m_rope_inline1208__tile, [1, 64], [gather_lv, 0], [1, 64])
                    gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(t__tile_17, [1, 64], [gather_lv, 0], [1, 64])
                    gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_56, pl.const(99072, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                    gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_57, pl.const(99328, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                    gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                    gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                m_swapped_inline1207__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_27, pl.const(65792, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
                m_rope_inline1208__tile_textract: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.extract(t__tile_16, 0, 448, [16, 64], target_memory=pl.Mem.Vec)
                t__tile_18: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(m_rope_inline1208__tile_textract, m_cos_il_inline1286__tile)
                t__tile_19: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(m_swapped_inline1207__tile, m_sin_signed_inline1225__tile)
                m_rot_inline1206__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_18, t__tile_19)
                n_rope_bf16_inline1205__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_18, pl.const(32832, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(m_rot_inline1206__tile, target_type=pl.BF16, mode='rint')
                t__tile_20: pl.Tile[[16, 448], pl.BF16, pl.MemRef(mem_vec_15, pl.const(0, pl.INT64), 16256), pl.Mem.Vec] = pl.tile.slice(n_bf16_inline1210__tile, [16, 448], [0, 0])
                n_full_bf16_inline1285__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.concat(t__tile_20, n_rope_bf16_inline1205__tile)
                n_head_inline1244__ssa_v0: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0
                source_row_inline1238__ssa_v0: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v0 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v0: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v0 % 8 * 512
                t__tile_21: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [0, 0])
                attention_grouped_inline1276__tile: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_21, [source_row_inline1238__ssa_v0, source_col_inline1204__ssa_v0], attention_grouped_inline1276__iter_v3)
                n_head_inline1244__ssa_v1: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 1
                source_row_inline1238__ssa_v1: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v1 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v1: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v1 % 8 * 512
                t__tile_22: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(100608, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [1, 0])
                attention_grouped_inline1276__tile_1: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_22, [source_row_inline1238__ssa_v1, source_col_inline1204__ssa_v1], attention_grouped_inline1276__tile)
                n_head_inline1244__ssa_v2: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 2
                source_row_inline1238__ssa_v2: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v2 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v2: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v2 % 8 * 512
                t__tile_23: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(101632, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [2, 0])
                attention_grouped_inline1276__tile_2: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_23, [source_row_inline1238__ssa_v2, source_col_inline1204__ssa_v2], attention_grouped_inline1276__tile_1)
                n_head_inline1244__ssa_v3: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 3
                source_row_inline1238__ssa_v3: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v3 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v3: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v3 % 8 * 512
                t__tile_24: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(102656, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [3, 0])
                attention_grouped_inline1276__tile_3: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_24, [source_row_inline1238__ssa_v3, source_col_inline1204__ssa_v3], attention_grouped_inline1276__tile_2)
                n_head_inline1244__ssa_v4: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 4
                source_row_inline1238__ssa_v4: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v4 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v4: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v4 % 8 * 512
                t__tile_25: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(103680, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [4, 0])
                attention_grouped_inline1276__tile_4: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_25, [source_row_inline1238__ssa_v4, source_col_inline1204__ssa_v4], attention_grouped_inline1276__tile_3)
                n_head_inline1244__ssa_v5: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 5
                source_row_inline1238__ssa_v5: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v5 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v5: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v5 % 8 * 512
                t__tile_26: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(104704, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [5, 0])
                attention_grouped_inline1276__tile_5: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_26, [source_row_inline1238__ssa_v5, source_col_inline1204__ssa_v5], attention_grouped_inline1276__tile_4)
                n_head_inline1244__ssa_v6: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 6
                source_row_inline1238__ssa_v6: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v6 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v6: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v6 % 8 * 512
                t__tile_27: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(105728, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [6, 0])
                attention_grouped_inline1276__tile_6: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_27, [source_row_inline1238__ssa_v6, source_col_inline1204__ssa_v6], attention_grouped_inline1276__tile_5)
                n_head_inline1244__ssa_v7: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 7
                source_row_inline1238__ssa_v7: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v7 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v7: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v7 % 8 * 512
                t__tile_28: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(106752, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [7, 0])
                attention_grouped_inline1276__tile_7: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_28, [source_row_inline1238__ssa_v7, source_col_inline1204__ssa_v7], attention_grouped_inline1276__tile_6)
                n_head_inline1244__ssa_v8: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 8
                source_row_inline1238__ssa_v8: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v8 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v8: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v8 % 8 * 512
                t__tile_29: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(107776, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [8, 0])
                attention_grouped_inline1276__tile_8: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_29, [source_row_inline1238__ssa_v8, source_col_inline1204__ssa_v8], attention_grouped_inline1276__tile_7)
                n_head_inline1244__ssa_v9: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 9
                source_row_inline1238__ssa_v9: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v9 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v9: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v9 % 8 * 512
                t__tile_30: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(108800, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [9, 0])
                attention_grouped_inline1276__tile_9: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_30, [source_row_inline1238__ssa_v9, source_col_inline1204__ssa_v9], attention_grouped_inline1276__tile_8)
                n_head_inline1244__ssa_v10: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 10
                source_row_inline1238__ssa_v10: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v10 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v10: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v10 % 8 * 512
                t__tile_31: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(109824, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [10, 0])
                attention_grouped_inline1276__tile_10: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_31, [source_row_inline1238__ssa_v10, source_col_inline1204__ssa_v10], attention_grouped_inline1276__tile_9)
                n_head_inline1244__ssa_v11: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 11
                source_row_inline1238__ssa_v11: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v11 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v11: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v11 % 8 * 512
                t__tile_32: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(110848, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [11, 0])
                attention_grouped_inline1276__tile_11: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_32, [source_row_inline1238__ssa_v11, source_col_inline1204__ssa_v11], attention_grouped_inline1276__tile_10)
                n_head_inline1244__ssa_v12: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 12
                source_row_inline1238__ssa_v12: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v12 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v12: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v12 % 8 * 512
                t__tile_33: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(111872, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [12, 0])
                attention_grouped_inline1276__tile_12: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_33, [source_row_inline1238__ssa_v12, source_col_inline1204__ssa_v12], attention_grouped_inline1276__tile_11)
                n_head_inline1244__ssa_v13: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 13
                source_row_inline1238__ssa_v13: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v13 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v13: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v13 % 8 * 512
                t__tile_34: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(112896, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [13, 0])
                attention_grouped_inline1276__tile_13: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_34, [source_row_inline1238__ssa_v13, source_col_inline1204__ssa_v13], attention_grouped_inline1276__tile_12)
                n_head_inline1244__ssa_v14: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 14
                source_row_inline1238__ssa_v14: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v14 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v14: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v14 % 8 * 512
                t__tile_35: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(113920, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [14, 0])
                attention_grouped_inline1276__tile_14: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_35, [source_row_inline1238__ssa_v14, source_col_inline1204__ssa_v14], attention_grouped_inline1276__tile_13)
                n_head_inline1244__ssa_v15: pl.Scalar[pl.INDEX] = m_h0_inline1224__ssa_v0 + 15
                source_row_inline1238__ssa_v15: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v15 // 8 * 256 + m_t_inline1290__ssa_v0
                source_col_inline1204__ssa_v15: pl.Scalar[pl.INDEX] = n_head_inline1244__ssa_v15 % 8 * 512
                t__tile_36: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_66, pl.const(114944, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.slice(n_full_bf16_inline1285__tile, [1, 512], [15, 0])
                attention_grouped_inline1276__tile_15: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile_36, [source_row_inline1238__ssa_v15, source_col_inline1204__ssa_v15], attention_grouped_inline1276__tile_14)
                attention_grouped_inline1276__rv_v4: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 16777216)] = pl.yield_(attention_grouped_inline1276__tile_15)
            source_row_inline1238__ssa_v16: pl.Scalar[pl.INDEX] = global_group0_inline1314__ssa_v0 * 256 + m_t0_inline1278__ssa_v0
            target_row_inline1203__ssa_v0: pl.Scalar[pl.INDEX] = local_group0_inline1222__ssa_v0 * 512 + pl.cast(tp_rank__ssa_v0, pl.INDEX) * 256 + m_t0_inline1278__ssa_v0
            tput_stage: pl.Tile[[8, 4096], pl.BF16, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.create([8, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pld.tile.put(attention_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + destination_rank_inline1223__ssa_v0, attention_grouped_inline1276__rv_v4, tput_stage, [target_row_inline1203__ssa_v0, 0], [source_row_inline1238__ssa_v16, 0], [8, 4096], atomic=pl.AtomicType.None_)
            pl.system.fence()
            source_row_inline1238__ssa_v17: pl.Scalar[pl.INDEX] = global_group0_inline1314__ssa_v0 * 256 + m_t0_inline1278__ssa_v0 + 256
            target_row_inline1203__ssa_v1: pl.Scalar[pl.INDEX] = local_group0_inline1222__ssa_v0 * 512 + pl.cast(tp_rank__ssa_v0, pl.INDEX) * 256 + m_t0_inline1278__ssa_v0 + 512
            tput_stage_1: pl.Tile[[8, 4096], pl.BF16, pl.MemRef(mem_vec_66, pl.const(99584, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.create([8, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pld.tile.put(attention_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + destination_rank_inline1223__ssa_v0, attention_grouped_inline1276__rv_v4, tput_stage_1, [target_row_inline1203__ssa_v1, 0], [source_row_inline1238__ssa_v17, 0], [8, 4096], atomic=pl.AtomicType.None_)
            pl.system.fence()
            attention_grouped_inline1276__rv_v2: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_68", pl.const(0, pl.INT64), 16777216)] = pl.yield_(attention_grouped_inline1276__rv_v4)
        for peer_tp_inline1273__idx_v0 in pl.range(2):
            if peer_tp_inline1273__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(attention_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline1273__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def csa_merge_pack_publish_spmd(self, attention_grouped_inline1276__ssa_v0: pl.InOut[pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], pack_work_count_inline1228__ssa_v0: pl.Scalar[pl.INDEX], sparse_blk_mi_inline1234__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], sparse_blk_li_inline1283__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], sparse_blk_oi_inline1233__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)], rope_cos_il_inline1232__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 65536)], rope_sin_signed_inline1231__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 65536)], rope_swap_idx_inline1230__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 4096)], tp_rank__ssa_v0: pl.Scalar[pl.INT32], attention_window__ssa_v0: pl.Out[pld.DistributedTensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 16777216)]], group_base__ssa_v0: pl.Scalar[pl.INT32], attention_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 8)]], attention_window_ctx: pld.CommCtx, attention_signal_ctx: pld.CommCtx):
        self.csa_merge_pack_publish(attention_grouped_inline1276__ssa_v0, pack_work_count_inline1228__ssa_v0, sparse_blk_mi_inline1234__ssa_v0, sparse_blk_li_inline1283__ssa_v0, sparse_blk_oi_inline1233__ssa_v0, attn_sink__ssa_v0, rope_cos_il_inline1232__ssa_v0, rope_sin_signed_inline1231__ssa_v0, rope_swap_idx_inline1230__ssa_v0, tp_rank__ssa_v0, attention_window__ssa_v0, group_base__ssa_v0, attention_signal__ssa_v0, attention_window_ctx, attention_signal_ctx, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_rope_interleave(idx_cos_il_inline1282__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], idx_sin_signed_inline1307__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], t_dim_inline1251__ssa_v0: pl.Scalar[pl.INDEX], freqs_cos_local__ssa_v0: pl.Tensor[[T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], freqs_sin_local__ssa_v0: pl.Tensor[[T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], cmp_cos_il_full_inline1249__ssa_v0: pl.Out[pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]], cmp_sin_signed_full_inline1263__ssa_v0: pl.Out[pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]], kv_dim_inline1261__ssa_v0: pl.Scalar[pl.INDEX], cmp_freqs_cos__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], cmp_freqs_sin__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)]) -> tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32], pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32], pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32]]:
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_21: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        il_ones_inline1242__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(3072, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.full([4, 64], dtype=pl.FP32, value=1.0)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
        il_col_inline1252__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(3072, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.col_expand_mul(il_ones_inline1242__tile, t__tile_1)
        t__tile_2: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_col_inline1252__tile, 0.5)
        t__tile_3: pl.Tile[[4, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.INT32, mode='trunc')
        il_dup_f_inline1250__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.FP32, mode='round')
        il_dup_idx_inline1272__tile: pl.Tile[[4, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(1024, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(il_dup_f_inline1250__tile, target_type=pl.INT32, mode='round')
        t__tile_4: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_dup_f_inline1250__tile, 2.0)
        il_lane_inline1300__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(3072, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.sub(il_col_inline1252__tile, t__tile_4)
        t__tile_5: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(3072, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.muls(il_lane_inline1300__tile, 2.0)
        il_sign_inline1298__tile: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(3072, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.subs(t__tile_5, 1.0)
        for rope_t0_inline1256__idx_v0, (idx_cos_il_inline1282__iter_v1, idx_sin_signed_inline1307__iter_v1) in pl.range(0, t_dim_inline1251__ssa_v0, 4, init_values=(idx_cos_il_inline1282__ssa_v0, idx_sin_signed_inline1307__ssa_v0)):
            t__tile_6: pl.Tile[[4, 32], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(freqs_cos_local__ssa_v0, [rope_t0_inline1256__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec)
            t__tile_7: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode='round')
            gather_acc_init: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(4, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2048, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(t__tile_7, [1, 32], [gather_lv, 0], [1, 32])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx_inline1272__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_23, pl.const(2560, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            t__tile_8: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = gather_rv
            idx_cos_il_inline1282__tile: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_8, [rope_t0_inline1256__idx_v0, 0], idx_cos_il_inline1282__iter_v1)
            t__tile_9: pl.Tile[[4, 32], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(freqs_sin_local__ssa_v0, [rope_t0_inline1256__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec)
            t__tile_10: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tile_9, target_type=pl.FP32, mode='round')
            gather_acc_init_1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(4, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2048, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(t__tile_10, [1, 32], [gather_lv_1, 0], [1, 32])
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx_inline1272__tile, [1, 64], [gather_lv_1, 0], [1, 64])
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_23, pl.const(2560, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            t__tile_11: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = gather_rv_1
            t__tile_12: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.mul(t__tile_11, il_sign_inline1298__tile)
            idx_sin_signed_inline1307__tile: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_12, [rope_t0_inline1256__idx_v0, 0], idx_sin_signed_inline1307__iter_v1)
            idx_cos_il_inline1282__rv_v2, idx_sin_signed_inline1307__rv_v2 = pl.yield_(idx_cos_il_inline1282__tile, idx_sin_signed_inline1307__tile)
        for cmp_t0_inline1248__idx_v0, (cmp_cos_il_full_inline1249__iter_v1, cmp_sin_signed_full_inline1263__iter_v1) in pl.range(0, kv_dim_inline1261__ssa_v0, 4, init_values=(cmp_cos_il_full_inline1249__ssa_v0, cmp_sin_signed_full_inline1263__ssa_v0)):
            t__tile_13: pl.Tile[[4, 32], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(cmp_freqs_cos__ssa_v0, [cmp_t0_inline1248__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec)
            t__tile_14: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tile_13, target_type=pl.FP32, mode='round')
            gather_acc_init_2: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_2, (gather_ia_2,) in pl.range(4, init_values=(gather_acc_init_2,)):
                gather_inp_row_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2048, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(t__tile_14, [1, 32], [gather_lv_2, 0], [1, 32])
                gather_idx_row_2: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx_inline1272__tile, [1, 64], [gather_lv_2, 0], [1, 64])
                gather_row_tmp_2: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_23, pl.const(2560, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_2, gather_idx_row_2, gather_row_tmp_2)
                gather_asmbl_2: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia_2, gather_row_2, [gather_lv_2, 0])
                gather_rv_2: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl_2)
            t__tile_15: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = gather_rv_2
            cmp_cos_il_full_inline1249__tile: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_15, [cmp_t0_inline1248__idx_v0, 0], cmp_cos_il_full_inline1249__iter_v1)
            t__tile_16: pl.Tile[[4, 32], pl.BF16, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(cmp_freqs_sin__ssa_v0, [cmp_t0_inline1248__idx_v0, 0], [4, 32], [4, 32], target_memory=pl.Mem.Vec)
            t__tile_17: pl.Tile[[4, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2048, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tile_16, target_type=pl.FP32, mode='round')
            gather_acc_init_3: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_3, (gather_ia_3,) in pl.range(4, init_values=(gather_acc_init_3,)):
                gather_inp_row_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_21, pl.const(2048, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(t__tile_17, [1, 32], [gather_lv_3, 0], [1, 32])
                gather_idx_row_3: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(il_dup_idx_inline1272__tile, [1, 64], [gather_lv_3, 0], [1, 64])
                gather_row_tmp_3: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_23, pl.const(2560, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(2816, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_3, gather_idx_row_3, gather_row_tmp_3)
                gather_asmbl_3: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.assemble(gather_ia_3, gather_row_3, [gather_lv_3, 0])
                gather_rv_3: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.yield_(gather_asmbl_3)
            t__tile_18: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = gather_rv_3
            t__tile_19: pl.Tile[[4, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.mul(t__tile_18, il_sign_inline1298__tile)
            cmp_sin_signed_full_inline1263__tile: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_19, [cmp_t0_inline1248__idx_v0, 0], cmp_sin_signed_full_inline1263__iter_v1)
            cmp_cos_il_full_inline1249__rv_v2, cmp_sin_signed_full_inline1263__rv_v2 = pl.yield_(cmp_cos_il_full_inline1249__tile, cmp_sin_signed_full_inline1263__tile)
        return idx_cos_il_inline1282__ssa_v0, idx_sin_signed_inline1307__ssa_v0, cmp_cos_il_full_inline1249__ssa_v0, cmp_sin_signed_full_inline1263__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def csa_slots_build_valid_qk_plan(cmp_sparse_indices_inline2383__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], sparse_bias_inline2381__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], t_dim_inline2369__ssa_v0: pl.Scalar[pl.INDEX], idx_topk_inline1280__ssa_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], position_ids_t1_inline1288__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], valid_block_mask_inline2385__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline2369__ssa_v0, 5], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]], window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], qk_wcur_inline2412__ssa_v0: pl.InOut[pl.Tensor[[1], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 4)]], qk_order_inline2351__ssa_v0: pl.Out[pl.Tensor[[1280], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 5120)]]) -> tuple[pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32]]:
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        for bias_t0_inline2361__idx_v0, (cmp_sparse_indices_inline2383__iter_v1, sparse_bias_inline2381__iter_v1) in pl.range(0, t_dim_inline2369__ssa_v0, 8, init_values=(cmp_sparse_indices_inline2383__ssa_v0, sparse_bias_inline2381__ssa_v0)):
            t__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_8, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(idx_topk_inline1280__ssa_v2, [bias_t0_inline2361__idx_v0, 0], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
            c_raw_inline2382__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
            t__tile_1: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(position_ids_t1_inline1288__ssa_v0, [bias_t0_inline2361__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
            c_pos_inline2359__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 8])
            c_pos_inline2359__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(c_pos_inline2359__rm_a0_tmp_v0, target_type=pl.FP32, mode='round')
            c_pos_inline2359__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline2359__row_major_tmp_v1, [8, 1])
            t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_inline2359__tile, [1, 8])
            t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 1.0)
            t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
            c_pos_scaled_inline2417__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
            c_pos_scaled_inline2417__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(c_pos_scaled_inline2417__rm_a0_tmp_v4, 0.25)
            c_pos_scaled_inline2417__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline2417__row_major_tmp_v5, [8, 1])
            c_pos_i32_inline2358__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_scaled_inline2417__tile, [1, 8])
            c_pos_i32_inline2358__row_major_tmp_v7: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(c_pos_i32_inline2358__rm_a0_tmp_v6, target_type=pl.INT32, mode='trunc')
            c_pos_i32_inline2358__tile: pl.Tile[[8, 1], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline2358__row_major_tmp_v7, [8, 1])
            c_pos_q_inline2367__rm_a0_tmp_v8: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_i32_inline2358__tile, [1, 8])
            c_pos_q_inline2367__row_major_tmp_v9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(c_pos_q_inline2367__rm_a0_tmp_v8, target_type=pl.FP32, mode='round')
            c_pos_q_inline2367__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(c_pos_q_inline2367__row_major_tmp_v9, [8, 1])
            t__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.full([8, 512], dtype=pl.FP32, value=1.0)
            c_upper_b_inline2379__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_3, c_pos_q_inline2367__tile)
            t__tile_4: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline2382__tile, 1.0)
            t__tile_5: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_4, 0.0)
            c_ge_inline2421__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_5, 1.0)
            t__tile_6: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.sub(c_upper_b_inline2379__tile, c_raw_inline2382__tile)
            t__tile_7: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximums(t__tile_6, 0.0)
            c_lt_inline2354__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(t__tile_7, 1.0)
            c_mask_inline2370__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_ge_inline2421__tile, c_lt_inline2354__tile)
            t__tile_8: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.adds(c_raw_inline2382__tile, 1.0)
            t__tile_9: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(c_mask_inline2370__tile, t__tile_8)
            c_out_inline2410__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
            t__tile_10: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(c_out_inline2410__tile, target_type=pl.INT32, mode='round')
            cmp_sparse_indices_inline2383__tile: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_10, [bias_t0_inline2361__idx_v0, 0], cmp_sparse_indices_inline2383__iter_v1)
            for c_t0_inline2340__idx_v0 in pl.range(8):
                pl.tensor.write(valid_block_mask_inline2385__ssa_v0, [bias_t0_inline2361__idx_v0 + c_t0_inline2340__idx_v0, 0], pl.cast(1, pl.INT32))
            for c_sb_inline2407__idx_v0 in pl.range(1, 5):
                c_s0_inline2342__ssa_v0: pl.Scalar[pl.INDEX] = (c_sb_inline2407__idx_v0 - 1) * 128
                t__tile_11: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 14848), pl.Mem.Vec] = pl.tile.slice(c_mask_inline2370__tile, [8, 128], [0, c_s0_inline2342__ssa_v0])
                tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_18, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                c_blk_valid_inline2350__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_30, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile_11, tmp_tile)
                for c_dt_inline2378__idx_v0 in pl.range(8):
                    t__tile_12: pl.Scalar[pl.FP32] = pl.tile.read(c_blk_valid_inline2350__tile, [c_dt_inline2378__idx_v0, 0])
                    c_valid_inline2389__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(t__tile_12, pl.INT32)
                    pl.tensor.write(valid_block_mask_inline2385__ssa_v0, [bias_t0_inline2361__idx_v0 + c_dt_inline2378__idx_v0, c_sb_inline2407__idx_v0], c_valid_inline2389__ssa_v0)
            t__tile_13: pl.Tile[[8, 128], pl.INT32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(window_swa_indices__ssa_v0, [bias_t0_inline2361__idx_v0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
            v_win_f_inline2375__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_13, target_type=pl.FP32, mode='round')
            t__tile_14: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(v_win_f_inline2375__tile, 1.0)
            t__tile_15: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.maximums(t__tile_14, 0.0)
            v_win_valid_inline2414__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.minimums(t__tile_15, 1.0)
            t__tile_16: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.subs(v_win_valid_inline2414__tile, 1.0)
            t__tile_17: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_16, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(t__tile_16, 1e+20)
            sparse_bias_inline2381__tile: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_17, [bias_t0_inline2361__idx_v0, 0], sparse_bias_inline2381__iter_v1)
            t__tile_18: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.minimums(c_out_inline2410__tile, 0.0)
            t__tile_19: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_8, pl.const(32800, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(t__tile_18, 1e+20)
            sparse_bias_inline2381__tile_1: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_19, [bias_t0_inline2361__idx_v0, 128], sparse_bias_inline2381__tile)
            cmp_sparse_indices_inline2383__rv_v2, sparse_bias_inline2381__rv_v2 = pl.yield_(cmp_sparse_indices_inline2383__tile, sparse_bias_inline2381__tile_1)
        pl.tensor.write(qk_wcur_inline2412__ssa_v0, [0], pl.cast(0, pl.INT32))
        for plan_t_inline2392__idx_v0 in pl.range(t_dim_inline2369__ssa_v0):
            for plan_sb_inline2386__idx_v0 in pl.range(5):
                t__tile_20: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2385__ssa_v0, [plan_t_inline2392__idx_v0, plan_sb_inline2386__idx_v0])
                if 0 < pl.cast(t__tile_20, pl.INDEX):
                    plan_w_inline2348__tile: pl.Scalar[pl.INT32] = pl.tensor.read(qk_wcur_inline2412__ssa_v0, [0])
                    pl.tensor.write(qk_order_inline2351__ssa_v0, [plan_w_inline2348__tile], pl.cast(plan_t_inline2392__idx_v0 * 5 + plan_sb_inline2386__idx_v0, pl.INT32))
                    pl.tensor.write(qk_wcur_inline2412__ssa_v0, [0], pl.cast(pl.cast(plan_w_inline2348__tile, pl.INDEX) + 1, pl.INT32))
        for plan_t_inline2394__idx_v0 in pl.range(t_dim_inline2369__ssa_v0):
            for plan_sb_inline2395__idx_v0 in pl.range(5):
                t__tile_21: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2385__ssa_v0, [plan_t_inline2394__idx_v0, plan_sb_inline2395__idx_v0])
                if pl.cast(t__tile_21, pl.INDEX) <= 0:
                    plan_w_v1_inline2399__tile: pl.Scalar[pl.INT32] = pl.tensor.read(qk_wcur_inline2412__ssa_v0, [0])
                    pl.tensor.write(qk_order_inline2351__ssa_v0, [plan_w_v1_inline2399__tile], pl.cast(plan_t_inline2394__idx_v0 * 5 + plan_sb_inline2395__idx_v0, pl.INT32))
                    pl.tensor.write(qk_wcur_inline2412__ssa_v0, [0], pl.cast(pl.cast(plan_w_v1_inline2399__tile, pl.INDEX) + 1, pl.INT32))
        return cmp_sparse_indices_inline2383__ssa_v0, sparse_bias_inline2381__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_topk_group_wave(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], score_arena__ssa_v0: pl.Tensor[[T_DYN, 262144], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], pair_arena__ssa_v0: pl.InOut[pl.Tensor[[4192, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 17170432)]]):
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        mem_vec_38: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(position_ids__ssa_v0, 0)
        for query__idx_v0, (global_group_base__iter_v1,) in pl.range(query_count__ssa_v0, init_values=(0,)):
            batch_idx__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 // 8
            position__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query__idx_v0])
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx__ssa_v0])
            cache_len__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            visible_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len__ssa_v0, (pl.cast(position__tile, pl.INDEX) + 1) // 4), 262144), 0)
            leaf_count__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count__ssa_v0 + 8191) // 8192
            group_count__ssa_v0: pl.Scalar[pl.INDEX] = (leaf_count__ssa_v0 + 1) // 2
            base_mod__ssa_v0: pl.Scalar[pl.INDEX] = global_group_base__iter_v1 % 48
            first_group__ssa_v0: pl.Scalar[pl.INDEX] = (worker__ssa_v0 + base_mod__ssa_v0) % 48
            for group__idx_v0 in pl.range(first_group__ssa_v0, group_count__ssa_v0, 48):
                leaf_begin__ssa_v0: pl.Scalar[pl.INDEX] = group__idx_v0 * 2
                group_leaf_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(leaf_count__ssa_v0 - leaf_begin__ssa_v0, 2)
                group_root_slot__ssa_v0: pl.Scalar[pl.INDEX] = query__idx_v0 * 16 + group__idx_v0
                if group_leaf_count__ssa_v0 == 1:
                    logical_begin__ssa_v0: pl.Scalar[pl.INDEX] = leaf_begin__ssa_v0 * 8192
                    valid_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count__ssa_v0 - logical_begin__ssa_v0, 8192)
                    logical_begin_i32_inline126__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(logical_begin__ssa_v0, pl.INT32)
                    t__tmp_v1: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False)
                    leaf_indices_inline125__ssa_v0: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.adds(t__tmp_v1, logical_begin_i32_inline126__ssa_v0)
                    leaf_scores_raw_inline124__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_count__ssa_v0])] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, logical_begin__ssa_v0], [1, 8192], [1, valid_count__ssa_v0], target_memory=pl.Mem.Vec)
                    leaf_scores_inline121__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw_inline124__ssa_v0, pad_value=pl.PadValue.min)
                    t__tmp_v2: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(135168, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    leaf_scores_v1_inline118__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(135168, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores_inline121__ssa_v0, t__tmp_v2)
                    t__tmp_v3: pl.Tile[[1, 8192], pl.UINT32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.reinterpret_view(leaf_indices_inline125__ssa_v0, dtype=pl.UINT32)
                    pairs_inline120__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1_inline118__ssa_v0, t__tmp_v3)
                    pairs_v1_inline119__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_inline120__ssa_v0, pl.const(64, pl.INT32))
                    pairs_v2_inline123__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1_inline119__ssa_v0, pl.const(256, pl.INT32))
                    pairs_v3_inline117__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2_inline123__ssa_v0, pl.const(1024, pl.INT32))
                    pairs_v4_inline122__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3_inline117__ssa_v0, pl.const(4096, pl.INT32))
                    t__tmp_v4: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(pairs_v4_inline122__ssa_v0, [1, 1024], [0, 0])
                    pl.tile.store(t__tmp_v4, [group_root_slot__ssa_v0, 0], pair_arena__ssa_v0)
                else:
                    scratch_base__ssa_v0: pl.Scalar[pl.INDEX] = worker__ssa_v0 * 2 + 4096
                    leaf__ssa_v0: pl.Scalar[pl.INDEX] = leaf_begin__ssa_v0
                    logical_begin__ssa_v1: pl.Scalar[pl.INDEX] = leaf__ssa_v0 * 8192
                    valid_count__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(visible_count__ssa_v0 - logical_begin__ssa_v1, 8192)
                    logical_begin_i32_inline136__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(logical_begin__ssa_v1, pl.INT32)
                    t__tmp_v5: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False)
                    leaf_indices_inline135__ssa_v0: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.adds(t__tmp_v5, logical_begin_i32_inline136__ssa_v0)
                    leaf_scores_raw_inline134__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_count__ssa_v1])] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, logical_begin__ssa_v1], [1, 8192], [1, valid_count__ssa_v1], target_memory=pl.Mem.Vec)
                    leaf_scores_inline131__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw_inline134__ssa_v0, pad_value=pl.PadValue.min)
                    t__tmp_v6: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(135168, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    leaf_scores_v1_inline128__ssa_v0: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(135168, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores_inline131__ssa_v0, t__tmp_v6)
                    t__tmp_v7: pl.Tile[[1, 8192], pl.UINT32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.reinterpret_view(leaf_indices_inline135__ssa_v0, dtype=pl.UINT32)
                    pairs_inline130__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1_inline128__ssa_v0, t__tmp_v7)
                    pairs_v1_inline129__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_inline130__ssa_v0, pl.const(64, pl.INT32))
                    pairs_v2_inline133__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1_inline129__ssa_v0, pl.const(256, pl.INT32))
                    pairs_v3_inline127__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2_inline133__ssa_v0, pl.const(1024, pl.INT32))
                    pairs_v4_inline132__ssa_v0: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3_inline127__ssa_v0, pl.const(4096, pl.INT32))
                    t__tmp_v8: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(pairs_v4_inline132__ssa_v0, [1, 1024], [0, 0])
                    pl.tile.store(t__tmp_v8, [scratch_base__ssa_v0, 0], pair_arena__ssa_v0)
                    leaf__ssa_v1: pl.Scalar[pl.INDEX] = leaf_begin__ssa_v0 + 1
                    logical_begin__ssa_v2: pl.Scalar[pl.INDEX] = leaf__ssa_v1 * 8192
                    valid_count__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(visible_count__ssa_v0 - logical_begin__ssa_v2, 8192)
                    logical_begin_i32_inline136__ssa_v1: pl.Scalar[pl.INT32] = pl.cast(logical_begin__ssa_v2, pl.INT32)
                    t__tmp_v9: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8192], dtype=pl.INT32, descending=False)
                    leaf_indices_inline135__ssa_v1: pl.Tile[[1, 8192], pl.INT32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.adds(t__tmp_v9, logical_begin_i32_inline136__ssa_v1)
                    leaf_scores_raw_inline134__ssa_v1: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_count__ssa_v2])] = pl.tile.load(score_arena__ssa_v0, [query__idx_v0, logical_begin__ssa_v2], [1, 8192], [1, valid_count__ssa_v2], target_memory=pl.Mem.Vec)
                    leaf_scores_inline131__ssa_v1: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(leaf_scores_raw_inline134__ssa_v1, pad_value=pl.PadValue.min)
                    t__tmp_v10: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(135168, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([1, 8192], dtype=pl.FP32, value=-3.4028234663852886e+38)
                    leaf_scores_v1_inline128__ssa_v1: pl.Tile[[1, 8192], pl.FP32, pl.MemRef(mem_vec_8, pl.const(135168, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.maximum(leaf_scores_inline131__ssa_v1, t__tmp_v10)
                    t__tmp_v11: pl.Tile[[1, 8192], pl.UINT32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.reinterpret_view(leaf_indices_inline135__ssa_v1, dtype=pl.UINT32)
                    pairs_inline130__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.sort32(leaf_scores_v1_inline128__ssa_v1, t__tmp_v11)
                    pairs_v1_inline129__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_inline130__ssa_v1, pl.const(64, pl.INT32))
                    pairs_v2_inline133__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v1_inline129__ssa_v1, pl.const(256, pl.INT32))
                    pairs_v3_inline127__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v2_inline133__ssa_v1, pl.const(1024, pl.INT32))
                    pairs_v4_inline132__ssa_v1: pl.Tile[[1, 16384], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 65536), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.mrgsort_format1(pairs_v3_inline127__ssa_v1, pl.const(4096, pl.INT32))
                    t__tmp_v12: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.slice(pairs_v4_inline132__ssa_v1, [1, 1024], [0, 0])
                    pl.tile.store(t__tmp_v12, [scratch_base__ssa_v0 + 1, 0], pair_arena__ssa_v0)
                    left_inline141__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(135168, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [scratch_base__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    right_inline140__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_38, pl.const(131072, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [scratch_base__ssa_v0 + 1, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                    merge_tmp_inline138__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    merged_all_inline139__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline141__ssa_v0, right_inline140__ssa_v0, merge_tmp_inline138__ssa_v0, exhausted=False)
                    merged_inline137__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_11, pl.const(65536, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(merged_all_inline139__ssa_v0, [1, 1024], [0, 0])
                    pl.tile.store(merged_inline137__ssa_v0, [group_root_slot__ssa_v0, 0], pair_arena__ssa_v0)
            global_group_base__ssa_v3: pl.Scalar[pl.INDEX] = global_group_base__iter_v1 + group_count__ssa_v0
            global_group_base__rv_v2: pl.Scalar[pl.INDEX] = pl.yield_(global_group_base__ssa_v3)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def decode_csa_test_indexer_topk_group_wave(self, idx_positions_inline1323__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], score_arena_inline44_inline2267__rv_v2: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], pair_arena_inline71_inline2266__ssa_v0: pl.InOut[pl.Tensor[[4192, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 17170432)]]):
        self.indexer_topk_group_wave(idx_positions_inline1323__ssa_v0, kv_seq_lens__ssa_v0, score_arena_inline44_inline2267__rv_v2, pair_arena_inline71_inline2266__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout]})
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_topk_query_merge(position_ids__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], pair_arena__ssa_v0: pl.InOut[pl.Tensor[[4192, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 17170432)]], topk_scores__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], topk_indices__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]):
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        query__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        batch_idx__ssa_v0: pl.Scalar[pl.INDEX] = query__ssa_v0 // 8
        position__tile: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids__ssa_v0, [query__ssa_v0])
        t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx__ssa_v0])
        cache_len__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
        visible_count__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.min(cache_len__ssa_v0, (pl.cast(position__tile, pl.INDEX) + 1) // 4), 262144)
        t__tmp_v1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=-3.4028234663852886e+38)
        pl.tile.store(t__tmp_v1, [query__ssa_v0, 0], topk_scores__ssa_v0)
        t__tmp_v2: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
        pl.tile.store(t__tmp_v2, [query__ssa_v0, 0], topk_indices__ssa_v0)
        if 0 < visible_count__ssa_v0:
            leaf_count__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count__ssa_v0 + 8191) // 8192
            group_count__ssa_v0: pl.Scalar[pl.INDEX] = (leaf_count__ssa_v0 + 1) // 2
            arena_base__ssa_v0: pl.Scalar[pl.INDEX] = query__ssa_v0 * 16
            if 1 < group_count__ssa_v0:
                level1_count__ssa_v0: pl.Scalar[pl.INDEX] = (group_count__ssa_v0 + 1) // 2
                output_count_inline10__ssa_v0: pl.Scalar[pl.INDEX] = (group_count__ssa_v0 + 1) // 2
                for output_inline9__idx_v0 in pl.range(output_count_inline10__ssa_v0):
                    left_slot_inline8__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + output_inline9__idx_v0 * 2
                    right_slot_inline6__ssa_v0: pl.Scalar[pl.INDEX] = left_slot_inline8__ssa_v0 + 1
                    output_slot_inline7__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + output_inline9__idx_v0
                    if right_slot_inline6__ssa_v0 < arena_base__ssa_v0 + group_count__ssa_v0:
                        left_inline1329__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline8__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                        right_inline1328__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [right_slot_inline6__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                        merge_tmp_inline1326__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        merged_all_inline1327__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1329__ssa_v0, right_inline1328__ssa_v0, merge_tmp_inline1326__ssa_v0, exhausted=False)
                        merged_inline1325__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(merged_all_inline1327__ssa_v0, [1, 1024], [0, 0])
                        pl.tile.store(merged_inline1325__ssa_v0, [output_slot_inline7__ssa_v0, 0], pair_arena__ssa_v0)
                    else:
                        forwarded_inline5__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline8__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                        pl.tile.store(forwarded_inline5__ssa_v0, [output_slot_inline7__ssa_v0, 0], pair_arena__ssa_v0)
                if 1 < level1_count__ssa_v0:
                    level2_count__ssa_v0: pl.Scalar[pl.INDEX] = (level1_count__ssa_v0 + 1) // 2
                    output_count_inline16__ssa_v0: pl.Scalar[pl.INDEX] = (level1_count__ssa_v0 + 1) // 2
                    for output_inline15__idx_v0 in pl.range(output_count_inline16__ssa_v0):
                        left_slot_inline14__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + output_inline15__idx_v0 * 2
                        right_slot_inline12__ssa_v0: pl.Scalar[pl.INDEX] = left_slot_inline14__ssa_v0 + 1
                        output_slot_inline13__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + output_inline15__idx_v0
                        if right_slot_inline12__ssa_v0 < arena_base__ssa_v0 + level1_count__ssa_v0:
                            left_inline1334__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline14__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                            right_inline1333__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [right_slot_inline12__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                            merge_tmp_inline1331__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            merged_all_inline1332__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1334__ssa_v0, right_inline1333__ssa_v0, merge_tmp_inline1331__ssa_v0, exhausted=False)
                            merged_inline1330__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(merged_all_inline1332__ssa_v0, [1, 1024], [0, 0])
                            pl.tile.store(merged_inline1330__ssa_v0, [output_slot_inline13__ssa_v0, 0], pair_arena__ssa_v0)
                        else:
                            forwarded_inline11__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline14__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                            pl.tile.store(forwarded_inline11__ssa_v0, [output_slot_inline13__ssa_v0, 0], pair_arena__ssa_v0)
                    if 1 < level2_count__ssa_v0:
                        level3_count__ssa_v0: pl.Scalar[pl.INDEX] = (level2_count__ssa_v0 + 1) // 2
                        output_count_inline22__ssa_v0: pl.Scalar[pl.INDEX] = (level2_count__ssa_v0 + 1) // 2
                        for output_inline21__idx_v0 in pl.range(output_count_inline22__ssa_v0):
                            left_slot_inline20__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + output_inline21__idx_v0 * 2
                            right_slot_inline18__ssa_v0: pl.Scalar[pl.INDEX] = left_slot_inline20__ssa_v0 + 1
                            output_slot_inline19__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0 + output_inline21__idx_v0
                            if right_slot_inline18__ssa_v0 < arena_base__ssa_v0 + level2_count__ssa_v0:
                                left_inline1339__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline20__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                                right_inline1338__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [right_slot_inline18__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                                merge_tmp_inline1336__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                merged_all_inline1337__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1339__ssa_v0, right_inline1338__ssa_v0, merge_tmp_inline1336__ssa_v0, exhausted=False)
                                merged_inline1335__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(merged_all_inline1337__ssa_v0, [1, 1024], [0, 0])
                                pl.tile.store(merged_inline1335__ssa_v0, [output_slot_inline19__ssa_v0, 0], pair_arena__ssa_v0)
                            else:
                                forwarded_inline17__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline20__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                                pl.tile.store(forwarded_inline17__ssa_v0, [output_slot_inline19__ssa_v0, 0], pair_arena__ssa_v0)
                        if 1 < level3_count__ssa_v0:
                            left_slot_inline26__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0
                            right_slot_inline24__ssa_v0: pl.Scalar[pl.INDEX] = left_slot_inline26__ssa_v0 + 1
                            output_slot_inline25__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0
                            if right_slot_inline24__ssa_v0 < arena_base__ssa_v0 + level3_count__ssa_v0:
                                left_inline1344__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_7, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline26__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                                right_inline1343__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_8, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [right_slot_inline24__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                                merge_tmp_inline1341__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([1, 2048], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                                merged_all_inline1342__ssa_v0: pl.Tile[[1, 2048], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mrgsort_format2(left_inline1344__ssa_v0, right_inline1343__ssa_v0, merge_tmp_inline1341__ssa_v0, exhausted=False)
                                merged_inline1340__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.slice(merged_all_inline1342__ssa_v0, [1, 1024], [0, 0])
                                pl.tile.store(merged_inline1340__ssa_v0, [output_slot_inline25__ssa_v0, 0], pair_arena__ssa_v0)
                            else:
                                forwarded_inline23__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [left_slot_inline26__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
                                pl.tile.store(forwarded_inline23__ssa_v0, [output_slot_inline25__ssa_v0, 0], pair_arena__ssa_v0)
            root_slot__ssa_v0: pl.Scalar[pl.INDEX] = arena_base__ssa_v0
            root_pairs__ssa_v0: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pair_arena__ssa_v0, [root_slot__ssa_v0, 0], [1, 1024], [1, 1024], target_memory=pl.Mem.Vec)
            t__tmp_v3: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather_mask(root_pairs__ssa_v0, mask_pattern=1, output_dtype=pl.FP32)
            pl.tile.store(t__tmp_v3, [query__ssa_v0, 0], topk_scores__ssa_v0)
            root_indices__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather_mask(root_pairs__ssa_v0, mask_pattern=2, output_dtype=pl.INT32)
            output_indices__ssa_v0: pl.Tile[[1, 512], pl.INT32, pl.MemRef(mem_vec_9, pl.const(16384, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.INT32, value=-1)
            valid_topk__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count__ssa_v0, 512)
            for lane__idx_v0 in pl.range(valid_topk__ssa_v0):
                t__tmp_v4: pl.Scalar[pl.INT32] = pl.tile.read(root_indices__ssa_v0, [0, lane__idx_v0])
                pl.tile.write(output_indices__ssa_v0, [0, lane__idx_v0], t__tmp_v4)
            pl.tile.store(output_indices__ssa_v0, [query__ssa_v0, 0], topk_indices__ssa_v0)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def decode_csa_test_indexer_topk_query_merge(self, idx_positions_inline1323__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], pair_arena_inline71_inline2266__ssa_v0: pl.InOut[pl.Tensor[[4192, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 17170432)]], idx_topk_scores_inline1271__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], idx_topk_inline1280__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]):
        self.indexer_topk_query_merge(idx_positions_inline1323__ssa_v0, kv_seq_lens__ssa_v0, pair_arena_inline71_inline2266__ssa_v0, idx_topk_scores_inline1271__ssa_v0, idx_topk_inline1280__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.output_existing]})
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def hc_post(y_flat_inline2568__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_dim_inline2576__ssa_v0: pl.Scalar[pl.INDEX], attn_out_inline1284__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], post_t_inline1277__phi_v2: pl.Tensor[[T_DYN, 4], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], comb_t_inline1267__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], residual_flat_inline2567__ssa_v0: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]):
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        token_block_inline2572__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t0_inline2573__ssa_v0: pl.Scalar[pl.INDEX] = token_block_inline2572__ssa_v0 * 4
        unroll_main_end: pl.Scalar[pl.INDEX] = t0_inline2573__ssa_v0 + 4
        for t_inline2574__idx_v0, (y_flat_inline2568__iter_v1,) in pl.range(t0_inline2573__ssa_v0, unroll_main_end, 2, init_values=(y_flat_inline2568__ssa_v0,)):
            if t_inline2574__idx_v0 < t_dim_inline2576__ssa_v0:
                post_w_inline2577__tile: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0, 0])
                post_w_inline2577__tile_1: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0, 1])
                post_w_inline2577__tile_2: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0, 2])
                post_w_inline2577__tile_3: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0, 3])
                t__tile: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(attn_out_inline1284__ssa_v0, [t_inline2574__idx_v0, 0], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                x_row_inline2578__tile: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
                y_row_inline2569__tile: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile, post_w_inline2577__tile)
                comb_w_inline2579__tile: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 0])
                res_d_inline2566__ssa_v0: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_1: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 4])
                res_d_inline2566__ssa_v0_1: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_2: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 8])
                res_d_inline2566__ssa_v0_2: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_3: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 12])
                res_d_inline2566__ssa_v0_3: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v0], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_1: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v0_1], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_2: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v0_2], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_3: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v0_3], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile, comb_w_inline2579__tile)
                y_row_inline2569__tile_1: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile, weighted_inline2564__tile)
                weighted_inline2564__tile_1: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_1, comb_w_inline2579__tile_1)
                y_row_inline2569__tile_2: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_1, weighted_inline2564__tile_1)
                weighted_inline2564__tile_2: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_2, comb_w_inline2579__tile_2)
                y_row_inline2569__tile_3: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_2, weighted_inline2564__tile_2)
                weighted_inline2564__tile_3: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_3, comb_w_inline2579__tile_3)
                y_row_inline2569__tile_4: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_3, weighted_inline2564__tile_3)
                y_flat_inline2568__tile: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_4, [t_inline2574__idx_v0, 0], y_flat_inline2568__iter_v1)
                y_row_inline2569__tile_5: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile, post_w_inline2577__tile_1)
                comb_w_inline2579__tile_4: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 1])
                res_d_inline2566__ssa_v1: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_5: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 5])
                res_d_inline2566__ssa_v1_1: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_6: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 9])
                res_d_inline2566__ssa_v1_2: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_7: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 13])
                res_d_inline2566__ssa_v1_3: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_4: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v1], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_5: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v1_1], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_6: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v1_2], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_7: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v1_3], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_4: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_4, comb_w_inline2579__tile_4)
                y_row_inline2569__tile_6: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_5, weighted_inline2564__tile_4)
                weighted_inline2564__tile_5: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_5, comb_w_inline2579__tile_5)
                y_row_inline2569__tile_7: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_6, weighted_inline2564__tile_5)
                weighted_inline2564__tile_6: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_6, comb_w_inline2579__tile_6)
                y_row_inline2569__tile_8: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_7, weighted_inline2564__tile_6)
                weighted_inline2564__tile_7: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_7, comb_w_inline2579__tile_7)
                y_row_inline2569__tile_9: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_8, weighted_inline2564__tile_7)
                y_flat_inline2568__tile_1: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_9, [t_inline2574__idx_v0, 4096], y_flat_inline2568__tile)
                y_row_inline2569__tile_10: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile, post_w_inline2577__tile_2)
                comb_w_inline2579__tile_8: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 2])
                res_d_inline2566__ssa_v2: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_9: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 6])
                res_d_inline2566__ssa_v2_1: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_10: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 10])
                res_d_inline2566__ssa_v2_2: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_11: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 14])
                res_d_inline2566__ssa_v2_3: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_8: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v2], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_9: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v2_1], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_10: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v2_2], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_11: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v2_3], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_8: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_8, comb_w_inline2579__tile_8)
                y_row_inline2569__tile_11: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_10, weighted_inline2564__tile_8)
                weighted_inline2564__tile_9: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_9, comb_w_inline2579__tile_9)
                y_row_inline2569__tile_12: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_11, weighted_inline2564__tile_9)
                weighted_inline2564__tile_10: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_10, comb_w_inline2579__tile_10)
                y_row_inline2569__tile_13: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_12, weighted_inline2564__tile_10)
                weighted_inline2564__tile_11: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_11, comb_w_inline2579__tile_11)
                y_row_inline2569__tile_14: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_13, weighted_inline2564__tile_11)
                y_flat_inline2568__tile_2: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_14, [t_inline2574__idx_v0, 8192], y_flat_inline2568__tile_1)
                y_row_inline2569__tile_15: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile, post_w_inline2577__tile_3)
                comb_w_inline2579__tile_12: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 3])
                res_d_inline2566__ssa_v3: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_13: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 7])
                res_d_inline2566__ssa_v3_1: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_14: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 11])
                res_d_inline2566__ssa_v3_2: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_15: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0, 15])
                res_d_inline2566__ssa_v3_3: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_12: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v3], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_13: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v3_1], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_14: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v3_2], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_15: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0, res_d_inline2566__ssa_v3_3], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_12: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_12, comb_w_inline2579__tile_12)
                y_row_inline2569__tile_16: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_15, weighted_inline2564__tile_12)
                weighted_inline2564__tile_13: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_13, comb_w_inline2579__tile_13)
                y_row_inline2569__tile_17: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_16, weighted_inline2564__tile_13)
                weighted_inline2564__tile_14: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_14, comb_w_inline2579__tile_14)
                y_row_inline2569__tile_18: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_17, weighted_inline2564__tile_14)
                weighted_inline2564__tile_15: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_15, comb_w_inline2579__tile_15)
                y_row_inline2569__tile_19: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_18, weighted_inline2564__tile_15)
                y_flat_inline2568__tile_3: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_19, [t_inline2574__idx_v0, 12288], y_flat_inline2568__tile_2)
                y_flat_inline2568__phi_v7: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.yield_(y_flat_inline2568__tile_3)
            else:
                y_flat_inline2568__phi_v7: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.yield_(y_flat_inline2568__iter_v1)
            if t_inline2574__idx_v0 + 1 < t_dim_inline2576__ssa_v0:
                post_w_inline2577__tile_4: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0 + 1, 0])
                post_w_inline2577__tile_5: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0 + 1, 1])
                post_w_inline2577__tile_6: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0 + 1, 2])
                post_w_inline2577__tile_7: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [t_inline2574__idx_v0 + 1, 3])
                t__tile_1: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(attn_out_inline1284__ssa_v0, [t_inline2574__idx_v0 + 1, 0], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                x_row_inline2578__tile_1: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode='round')
                y_row_inline2569__tile_20: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile_1, post_w_inline2577__tile_4)
                comb_w_inline2579__tile_16: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 0])
                res_d_inline2566__ssa_v0_4: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_17: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 4])
                res_d_inline2566__ssa_v0_5: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_18: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 8])
                res_d_inline2566__ssa_v0_6: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_19: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 12])
                res_d_inline2566__ssa_v0_7: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_16: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v0_4], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_17: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v0_5], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_18: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v0_6], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_19: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v0_7], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_16: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_16, comb_w_inline2579__tile_16)
                y_row_inline2569__tile_21: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_20, weighted_inline2564__tile_16)
                weighted_inline2564__tile_17: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_17, comb_w_inline2579__tile_17)
                y_row_inline2569__tile_22: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_21, weighted_inline2564__tile_17)
                weighted_inline2564__tile_18: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_18, comb_w_inline2579__tile_18)
                y_row_inline2569__tile_23: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_22, weighted_inline2564__tile_18)
                weighted_inline2564__tile_19: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_19, comb_w_inline2579__tile_19)
                y_row_inline2569__tile_24: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_23, weighted_inline2564__tile_19)
                y_flat_inline2568__tile_4: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_24, [t_inline2574__idx_v0 + 1, 0], y_flat_inline2568__phi_v7)
                y_row_inline2569__tile_25: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile_1, post_w_inline2577__tile_5)
                comb_w_inline2579__tile_20: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 1])
                res_d_inline2566__ssa_v1_4: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_21: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 5])
                res_d_inline2566__ssa_v1_5: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_22: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 9])
                res_d_inline2566__ssa_v1_6: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_23: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 13])
                res_d_inline2566__ssa_v1_7: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_20: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v1_4], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_21: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v1_5], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_22: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v1_6], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_23: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v1_7], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_20: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_20, comb_w_inline2579__tile_20)
                y_row_inline2569__tile_26: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_25, weighted_inline2564__tile_20)
                weighted_inline2564__tile_21: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_21, comb_w_inline2579__tile_21)
                y_row_inline2569__tile_27: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_26, weighted_inline2564__tile_21)
                weighted_inline2564__tile_22: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_22, comb_w_inline2579__tile_22)
                y_row_inline2569__tile_28: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_27, weighted_inline2564__tile_22)
                weighted_inline2564__tile_23: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_23, comb_w_inline2579__tile_23)
                y_row_inline2569__tile_29: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_28, weighted_inline2564__tile_23)
                y_flat_inline2568__tile_5: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_29, [t_inline2574__idx_v0 + 1, 4096], y_flat_inline2568__tile_4)
                y_row_inline2569__tile_30: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile_1, post_w_inline2577__tile_6)
                comb_w_inline2579__tile_24: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 2])
                res_d_inline2566__ssa_v2_4: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_25: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 6])
                res_d_inline2566__ssa_v2_5: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_26: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 10])
                res_d_inline2566__ssa_v2_6: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_27: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 14])
                res_d_inline2566__ssa_v2_7: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_24: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v2_4], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_25: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v2_5], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_26: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v2_6], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_27: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v2_7], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_24: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_24, comb_w_inline2579__tile_24)
                y_row_inline2569__tile_31: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_30, weighted_inline2564__tile_24)
                weighted_inline2564__tile_25: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_25, comb_w_inline2579__tile_25)
                y_row_inline2569__tile_32: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_31, weighted_inline2564__tile_25)
                weighted_inline2564__tile_26: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_26, comb_w_inline2579__tile_26)
                y_row_inline2569__tile_33: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_32, weighted_inline2564__tile_26)
                weighted_inline2564__tile_27: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_27, comb_w_inline2579__tile_27)
                y_row_inline2569__tile_34: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_33, weighted_inline2564__tile_27)
                y_flat_inline2568__tile_6: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_34, [t_inline2574__idx_v0 + 1, 8192], y_flat_inline2568__tile_5)
                y_row_inline2569__tile_35: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile_1, post_w_inline2577__tile_7)
                comb_w_inline2579__tile_28: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 3])
                res_d_inline2566__ssa_v3_4: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_29: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 7])
                res_d_inline2566__ssa_v3_5: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_30: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 11])
                res_d_inline2566__ssa_v3_6: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_31: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [t_inline2574__idx_v0 + 1, 15])
                res_d_inline2566__ssa_v3_7: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_28: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v3_4], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_29: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v3_5], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_30: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v3_6], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_31: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [t_inline2574__idx_v0 + 1, res_d_inline2566__ssa_v3_7], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_28: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_28, comb_w_inline2579__tile_28)
                y_row_inline2569__tile_36: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_35, weighted_inline2564__tile_28)
                weighted_inline2564__tile_29: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_29, comb_w_inline2579__tile_29)
                y_row_inline2569__tile_37: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_36, weighted_inline2564__tile_29)
                weighted_inline2564__tile_30: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_30, comb_w_inline2579__tile_30)
                y_row_inline2569__tile_38: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_37, weighted_inline2564__tile_30)
                weighted_inline2564__tile_31: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_31, comb_w_inline2579__tile_31)
                y_row_inline2569__tile_39: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_38, weighted_inline2564__tile_31)
                y_flat_inline2568__tile_7: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_39, [t_inline2574__idx_v0 + 1, 12288], y_flat_inline2568__tile_6)
                y_flat_inline2568__phi_v7_1: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_122", pl.const(0, pl.INT64), 0)] = pl.yield_(y_flat_inline2568__tile_7)
            else:
                y_flat_inline2568__phi_v7_1: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_122", pl.const(0, pl.INT64), 0)] = pl.yield_(y_flat_inline2568__phi_v7)
            y_flat_inline2568__rv_v2_main: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = pl.yield_(y_flat_inline2568__phi_v7_1)
        unroll_rem: pl.Scalar[pl.INDEX] = t0_inline2573__ssa_v0 - unroll_main_end + 4
        if unroll_rem == 1:
            if unroll_main_end < t_dim_inline2576__ssa_v0:
                t__tile_2: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(attn_out_inline1284__ssa_v0, [unroll_main_end, 0], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                x_row_inline2578__tile_2: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode='round')
                post_w_inline2577__tile_8: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [unroll_main_end, 0])
                y_row_inline2569__tile_40: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile_2, post_w_inline2577__tile_8)
                comb_w_inline2579__tile_32: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 0])
                res_d_inline2566__ssa_v0_8: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_33: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 4])
                res_d_inline2566__ssa_v0_9: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_34: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 8])
                res_d_inline2566__ssa_v0_10: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_35: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 12])
                res_d_inline2566__ssa_v0_11: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_32: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v0_8], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_33: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v0_9], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_34: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v0_10], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_35: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v0_11], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_32: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_32, comb_w_inline2579__tile_32)
                y_row_inline2569__tile_41: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_40, weighted_inline2564__tile_32)
                weighted_inline2564__tile_33: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_33, comb_w_inline2579__tile_33)
                y_row_inline2569__tile_42: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_41, weighted_inline2564__tile_33)
                weighted_inline2564__tile_34: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_34, comb_w_inline2579__tile_34)
                y_row_inline2569__tile_43: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_42, weighted_inline2564__tile_34)
                weighted_inline2564__tile_35: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_35, comb_w_inline2579__tile_35)
                y_row_inline2569__tile_44: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_43, weighted_inline2564__tile_35)
                y_flat_inline2568__tile_8: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_44, [unroll_main_end, 0], y_flat_inline2568__rv_v2_main)
                post_w_inline2577__tile_9: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [unroll_main_end, 1])
                y_row_inline2569__tile_45: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile_2, post_w_inline2577__tile_9)
                comb_w_inline2579__tile_36: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 1])
                res_d_inline2566__ssa_v1_8: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_37: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 5])
                res_d_inline2566__ssa_v1_9: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_38: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 9])
                res_d_inline2566__ssa_v1_10: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_39: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 13])
                res_d_inline2566__ssa_v1_11: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_36: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v1_8], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_37: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v1_9], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_38: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v1_10], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_39: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v1_11], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_36: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_36, comb_w_inline2579__tile_36)
                y_row_inline2569__tile_46: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_45, weighted_inline2564__tile_36)
                weighted_inline2564__tile_37: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_37, comb_w_inline2579__tile_37)
                y_row_inline2569__tile_47: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_46, weighted_inline2564__tile_37)
                weighted_inline2564__tile_38: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_38, comb_w_inline2579__tile_38)
                y_row_inline2569__tile_48: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_47, weighted_inline2564__tile_38)
                weighted_inline2564__tile_39: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_39, comb_w_inline2579__tile_39)
                y_row_inline2569__tile_49: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_48, weighted_inline2564__tile_39)
                y_flat_inline2568__tile_9: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_49, [unroll_main_end, 4096], y_flat_inline2568__tile_8)
                post_w_inline2577__tile_10: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [unroll_main_end, 2])
                y_row_inline2569__tile_50: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile_2, post_w_inline2577__tile_10)
                comb_w_inline2579__tile_40: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 2])
                res_d_inline2566__ssa_v2_8: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_41: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 6])
                res_d_inline2566__ssa_v2_9: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_42: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 10])
                res_d_inline2566__ssa_v2_10: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_43: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 14])
                res_d_inline2566__ssa_v2_11: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_40: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v2_8], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_41: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v2_9], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_42: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v2_10], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_43: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16384, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v2_11], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_40: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_40, comb_w_inline2579__tile_40)
                y_row_inline2569__tile_51: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_50, weighted_inline2564__tile_40)
                weighted_inline2564__tile_41: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_41, comb_w_inline2579__tile_41)
                y_row_inline2569__tile_52: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_51, weighted_inline2564__tile_41)
                weighted_inline2564__tile_42: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_42, comb_w_inline2579__tile_42)
                y_row_inline2569__tile_53: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_52, weighted_inline2564__tile_42)
                weighted_inline2564__tile_43: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_43, comb_w_inline2579__tile_43)
                y_row_inline2569__tile_54: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_53, weighted_inline2564__tile_43)
                y_flat_inline2568__tile_10: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_54, [unroll_main_end, 8192], y_flat_inline2568__tile_9)
                post_w_inline2577__tile_11: pl.Scalar[pl.FP32] = pl.tensor.read(post_t_inline1277__phi_v2, [unroll_main_end, 3])
                y_row_inline2569__tile_55: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(x_row_inline2578__tile_2, post_w_inline2577__tile_11)
                comb_w_inline2579__tile_44: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 3])
                res_d_inline2566__ssa_v3_8: pl.Scalar[pl.INDEX] = 0
                comb_w_inline2579__tile_45: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 7])
                res_d_inline2566__ssa_v3_9: pl.Scalar[pl.INDEX] = 4096
                comb_w_inline2579__tile_46: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 11])
                res_d_inline2566__ssa_v3_10: pl.Scalar[pl.INDEX] = 8192
                comb_w_inline2579__tile_47: pl.Scalar[pl.FP32] = pl.tensor.read(comb_t_inline1267__ssa_v0, [unroll_main_end, 15])
                res_d_inline2566__ssa_v3_11: pl.Scalar[pl.INDEX] = 12288
                res_row_inline2565__tile_44: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v3_8], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_45: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v3_9], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_46: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_9, pl.const(81920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v3_10], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                res_row_inline2565__tile_47: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(residual_flat_inline2567__ssa_v0, [unroll_main_end, res_d_inline2566__ssa_v3_11], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                weighted_inline2564__tile_44: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_44, comb_w_inline2579__tile_44)
                y_row_inline2569__tile_56: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_55, weighted_inline2564__tile_44)
                weighted_inline2564__tile_45: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_45, comb_w_inline2579__tile_45)
                y_row_inline2569__tile_57: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_56, weighted_inline2564__tile_45)
                weighted_inline2564__tile_46: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_46, comb_w_inline2579__tile_46)
                y_row_inline2569__tile_58: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_7, pl.const(49152, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_57, weighted_inline2564__tile_46)
                weighted_inline2564__tile_47: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_8, pl.const(65536, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(res_row_inline2565__tile_47, comb_w_inline2579__tile_47)
                y_row_inline2569__tile_59: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_6, pl.const(32768, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(y_row_inline2569__tile_58, weighted_inline2564__tile_47)
                y_flat_inline2568__tile_11: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_row_inline2569__tile_59, [unroll_main_end, 12288], y_flat_inline2568__tile_10)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def hc_post_spmd(self, y_flat_inline2568__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_dim_inline2576__ssa_v0: pl.Scalar[pl.INDEX], attn_out_inline1284__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], post_t_inline1277__phi_v2: pl.Tensor[[T_DYN, 4], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], comb_t_inline1267__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], residual_flat_inline2567__ssa_v0: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]):
        self.hc_post(y_flat_inline2568__ssa_v0, t_dim_inline2576__ssa_v0, attn_out_inline1284__ssa_v1, post_t_inline1277__phi_v2, comb_t_inline1267__ssa_v1, residual_flat_inline2567__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def hc_pre_linear_reduce(mixes_partials_inline1475__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], t_linear_inline1486__ssa_v0: pl.Scalar[pl.INDEX], mixes_raw_inline1505__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32]:
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        linear_block_inline1567__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        linear_t0_inline1457__ssa_v0: pl.Scalar[pl.INDEX] = linear_block_inline1567__ssa_v0 * 16
        mixes_total_inline1570__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(mixes_partials_inline1475__ssa_v1, [linear_t0_inline1457__ssa_v0, 0], [16, 32], [16, 32], target_memory=pl.Mem.Vec)
        for linear_split_inline1456__idx_v0, (mixes_total_inline1570__iter_v1,) in pl.range(1, 4, init_values=(mixes_total_inline1570__tile,)):
            partial_t0_inline1464__ssa_v0: pl.Scalar[pl.INDEX] = linear_split_inline1456__idx_v0 * t_linear_inline1486__ssa_v0 + linear_t0_inline1457__ssa_v0
            partial_tile_inline1561__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_vec_3, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(mixes_partials_inline1475__ssa_v1, [partial_t0_inline1464__ssa_v0, 0], [16, 32], [16, 32], target_memory=pl.Mem.Vec)
            mixes_total_inline1570__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(mixes_total_inline1570__iter_v1, partial_tile_inline1561__tile)
            mixes_total_inline1570__rv_v2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(mixes_total_inline1570__tile_1)
        mixes_raw_inline1505__tile: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(mixes_total_inline1570__rv_v2, [linear_t0_inline1457__ssa_v0, 0], mixes_raw_inline1505__ssa_v0)
        return mixes_raw_inline1505__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def hc_pre_linear_reduce_spmd(self, mixes_partials_inline1475__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], t_linear_inline1486__ssa_v0: pl.Scalar[pl.INDEX], mixes_raw_inline1505__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32]:
        mixes_raw_inline1505__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = self.hc_pre_linear_reduce(mixes_partials_inline1475__ssa_v1, t_linear_inline1486__ssa_v0, mixes_raw_inline1505__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.output_existing]})
        return mixes_raw_inline1505__ssa_v0
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def hc_pre_linear(t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline1497__ssa_v0: pl.Tensor[[t_dim_inline1568__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], hc_attn_fn__ssa_v0: pl.Tensor[[24, 16384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)], t_linear_inline1486__ssa_v0: pl.Scalar[pl.INDEX], mixes_partials_inline1475__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32]:
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 2048)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_14: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_15: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        task_inline1516__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t0_inline1476__ssa_v1: pl.Scalar[pl.INDEX] = task_inline1516__ssa_v0 // 4 * 16
        linear_split_inline1509__ssa_v0: pl.Scalar[pl.INDEX] = task_inline1516__ssa_v0 % 4
        k_base_inline1494__ssa_v0: pl.Scalar[pl.INDEX] = linear_split_inline1509__ssa_v0 * 4096
        t_rows_inline1520__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v1, 16)
        acc_inline1524__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        for kb_inline1488__idx_v0, (acc_inline1524__iter_v1,) in pl.range(0, 16, 2, init_values=(acc_inline1524__tile,)):
            k0_inline1501__ssa_v1: pl.Scalar[pl.INDEX] = k_base_inline1494__ssa_v0 + kb_inline1488__idx_v0 * 256
            k0_inline1501__ssa_v1_1: pl.Scalar[pl.INDEX] = k_base_inline1494__ssa_v0 + (kb_inline1488__idx_v0 * 256 + 256)
            x_linear_chunk_inline1459__tile: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[t_rows_inline1520__ssa_v0, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v1, k0_inline1501__ssa_v1], [16, 256], [t_rows_inline1520__ssa_v0, 256], target_memory=pl.Mem.Mat)
            w_chunk_inline1519__tile: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(valid_shape=[24, 256])] = pl.tile.load(hc_attn_fn__ssa_v0, [0, k0_inline1501__ssa_v1], [32, 256], [24, 256], target_memory=pl.Mem.Mat)
            x_linear_chunk_inline1459__tile_1: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_mat_6, pl.const(49152, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[t_rows_inline1520__ssa_v0, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v1, k0_inline1501__ssa_v1_1], [16, 256], [t_rows_inline1520__ssa_v0, 256], target_memory=pl.Mem.Mat)
            w_chunk_inline1519__tile_1: pl.Tile[[32, 256], pl.FP32, pl.MemRef(mem_mat_7, pl.const(65536, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(valid_shape=[24, 256])] = pl.tile.load(hc_attn_fn__ssa_v0, [0, k0_inline1501__ssa_v1_1], [32, 256], [24, 256], target_memory=pl.Mem.Mat)
            if kb_inline1488__idx_v0 == 0:
                w_chunk_inline1519__tile_t: pl.Tile[[256, 32], pl.FP32, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(valid_shape=[256, 24], blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(w_chunk_inline1519__tile)
                x_linear_chunk_inline1459__tile_Left: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_left_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[t_rows_inline1520__ssa_v0, 256])] = pl.tile.move(x_linear_chunk_inline1459__tile, target_memory=pl.Mem.Left)
                w_chunk_inline1519__tile_t_Right: pl.Tile[[256, 32], pl.FP32, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right, pl.TileView(valid_shape=[256, 24])] = pl.tile.move(w_chunk_inline1519__tile_t, target_memory=pl.Mem.Right)
                acc_inline1524__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc, pl.TileView(valid_shape=[t_rows_inline1520__ssa_v0, 24], compact=pl.CompactMode.normal)] = pl.tile.matmul(x_linear_chunk_inline1459__tile_Left, w_chunk_inline1519__tile_t_Right)
                acc_inline1524__phi_v5: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc, pl.TileView(valid_shape=[t_rows_inline1520__ssa_v0, 24], compact=pl.CompactMode.normal)] = pl.yield_(acc_inline1524__tile_1)
            else:
                w_chunk_inline1519__tile_t_1: pl.Tile[[256, 32], pl.FP32, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(valid_shape=[256, 24], blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(w_chunk_inline1519__tile)
                x_linear_chunk_inline1459__tile_Left_1: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_left_8, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[t_rows_inline1520__ssa_v0, 256])] = pl.tile.move(x_linear_chunk_inline1459__tile, target_memory=pl.Mem.Left)
                w_chunk_inline1519__tile_t_Right_1: pl.Tile[[256, 32], pl.FP32, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right, pl.TileView(valid_shape=[256, 24])] = pl.tile.move(w_chunk_inline1519__tile_t_1, target_memory=pl.Mem.Right)
                acc_inline1524__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(acc_inline1524__iter_v1, x_linear_chunk_inline1459__tile_Left_1, w_chunk_inline1519__tile_t_Right_1)
                acc_inline1524__phi_v5: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc, pl.TileView(valid_shape=[t_rows_inline1520__ssa_v0, 24], compact=pl.CompactMode.normal)] = pl.yield_(acc_inline1524__tile_2)
            w_chunk_inline1519__tile_t_2: pl.Tile[[256, 32], pl.FP32, pl.MemRef(mem_mat_7, pl.const(65536, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(valid_shape=[256, 24], blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(w_chunk_inline1519__tile_1)
            x_linear_chunk_inline1459__tile_Left_2: pl.Tile[[16, 256], pl.FP32, pl.MemRef(mem_left_14, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[t_rows_inline1520__ssa_v0, 256])] = pl.tile.move(x_linear_chunk_inline1459__tile_1, target_memory=pl.Mem.Left)
            w_chunk_inline1519__tile_t_Right_2: pl.Tile[[256, 32], pl.FP32, pl.MemRef(mem_right_15, pl.const(0, pl.INT64), 32768), pl.Mem.Right, pl.TileView(valid_shape=[256, 24])] = pl.tile.move(w_chunk_inline1519__tile_t_2, target_memory=pl.Mem.Right)
            acc_inline1524__tile_3: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(acc_inline1524__phi_v5, x_linear_chunk_inline1459__tile_Left_2, w_chunk_inline1519__tile_t_Right_2)
            acc_inline1524__rv_v2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.yield_(acc_inline1524__tile_3)
        partial_row0_inline1471__ssa_v0: pl.Scalar[pl.INDEX] = linear_split_inline1509__ssa_v0 * t_linear_inline1486__ssa_v0 + t0_inline1476__ssa_v1
        mixes_partials_inline1475__tile: pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(acc_inline1524__rv_v2, [partial_row0_inline1471__ssa_v0, 0], mixes_partials_inline1475__ssa_v0)
        return mixes_partials_inline1475__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def hc_pre_linear_spmd(self, t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline1497__ssa_v0: pl.Tensor[[t_dim_inline1568__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], hc_attn_fn__ssa_v0: pl.Tensor[[24, 16384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)], t_linear_inline1486__ssa_v0: pl.Scalar[pl.INDEX], mixes_partials_inline1475__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32]:
        mixes_partials_inline1475__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.hc_pre_linear(t_dim_inline1568__ssa_v0, x_flat_inline1497__ssa_v0, hc_attn_fn__ssa_v0, t_linear_inline1486__ssa_v0, mixes_partials_inline1475__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing]})
        return mixes_partials_inline1475__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def hc_pre_rms(t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline1497__ssa_v0: pl.Tensor[[t_dim_inline1568__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], inv_rms_inline1463__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32]:
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_28: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_33: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_38: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_39: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        t_inline1518__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t0_inline1476__ssa_v0: pl.Scalar[pl.INDEX] = t_inline1518__ssa_v0 * 8
        valid_rows_inline1507__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v0, 8)
        sq_sum_inline1490__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(32832, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for kb_inline1493__idx_v0, (sq_sum_inline1490__iter_v1,) in pl.range(0, 32, 4, init_values=(sq_sum_inline1490__tile,)):
            k0_inline1501__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline1493__idx_v0 * 512
            k0_inline1501__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline1493__idx_v0 * 512 + 512
            k0_inline1501__ssa_v0_2: pl.Scalar[pl.INDEX] = kb_inline1493__idx_v0 * 512 + 1024
            k0_inline1501__ssa_v0_3: pl.Scalar[pl.INDEX] = kb_inline1493__idx_v0 * 512 + 1536
            if valid_rows_inline1507__ssa_v0 == 8:
                x_chunk_full_inline1502__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65696, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                x_sq_full_inline1510__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65696, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(x_chunk_full_inline1502__tile, x_chunk_full_inline1502__tile)
                tmp_tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(114912, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(131296, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(x_sq_full_inline1510__tile, tmp_tile)
                x_sq_row_full_inline1511__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(131296, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 8])
                sq_sum_inline1490__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(131296, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(sq_sum_inline1490__iter_v1, x_sq_row_full_inline1511__tile)
                sq_sum_inline1490__phi_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(131296, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__tile_1)
            else:
                x_chunk_tail_inline1513__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65696, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 512])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0], [8, 512], [valid_rows_inline1507__ssa_v0, 512], target_memory=pl.Mem.Vec)
                x_sq_tail_inline1503__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65696, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 512])] = pl.tile.mul(x_chunk_tail_inline1513__tile, x_chunk_tail_inline1513__tile)
                tmp_tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(114912, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 1])] = pl.tile.row_sum(x_sq_tail_inline1503__tile, tmp_tile_1)
                x_sq_row_tail_inline1537__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1507__ssa_v0])] = pl.tile.reshape(t__tile_1, [1, 8])
                sq_sum_inline1490__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65696, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(sq_sum_inline1490__iter_v1, x_sq_row_tail_inline1537__tile)
                sq_sum_inline1490__tile_mv: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(131296, pl.INT64), 32), pl.Mem.Vec] = pl.tile.move(sq_sum_inline1490__tile_2, target_memory=pl.Mem.Vec)
                sq_sum_inline1490__phi_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(131296, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__tile_mv)
            if valid_rows_inline1507__ssa_v0 == 8:
                x_chunk_full_inline1502__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                x_sq_full_inline1510__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(x_chunk_full_inline1502__tile_1, x_chunk_full_inline1502__tile_1)
                tmp_tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(x_sq_full_inline1510__tile_1, tmp_tile_2)
                x_sq_row_full_inline1511__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
                sq_sum_inline1490__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(sq_sum_inline1490__phi_v5, x_sq_row_full_inline1511__tile_1)
                sq_sum_inline1490__phi_v5_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__tile_3)
            else:
                x_chunk_tail_inline1513__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 512])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0_1], [8, 512], [valid_rows_inline1507__ssa_v0, 512], target_memory=pl.Mem.Vec)
                x_sq_tail_inline1503__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 512])] = pl.tile.mul(x_chunk_tail_inline1513__tile_1, x_chunk_tail_inline1513__tile_1)
                tmp_tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_16, pl.const(16416, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_3: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(32864, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 1])] = pl.tile.row_sum(x_sq_tail_inline1503__tile_1, tmp_tile_3)
                x_sq_row_tail_inline1537__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_22, pl.const(32864, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1507__ssa_v0])] = pl.tile.reshape(t__tile_3, [1, 8])
                sq_sum_inline1490__tile_4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(sq_sum_inline1490__phi_v5, x_sq_row_tail_inline1537__tile_1)
                sq_sum_inline1490__tile_mv_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.tile.move(sq_sum_inline1490__tile_4, target_memory=pl.Mem.Vec)
                sq_sum_inline1490__phi_v5_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_17, pl.const(32800, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__tile_mv_1)
            if valid_rows_inline1507__ssa_v0 == 8:
                x_chunk_full_inline1502__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_25, pl.const(32896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0_2], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                x_sq_full_inline1510__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_25, pl.const(32896, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(x_chunk_full_inline1502__tile_2, x_chunk_full_inline1502__tile_2)
                tmp_tile_4: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_4: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_28, pl.const(65664, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(x_sq_full_inline1510__tile_2, tmp_tile_4)
                x_sq_row_full_inline1511__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_28, pl.const(65664, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_4, [1, 8])
                sq_sum_inline1490__tile_5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_28, pl.const(65664, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(sq_sum_inline1490__phi_v5_1, x_sq_row_full_inline1511__tile_2)
                sq_sum_inline1490__phi_v5_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_28, pl.const(65664, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__tile_5)
            else:
                x_chunk_tail_inline1513__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_25, pl.const(32896, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 512])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0_2], [8, 512], [valid_rows_inline1507__ssa_v0, 512], target_memory=pl.Mem.Vec)
                x_sq_tail_inline1503__tile_2: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_25, pl.const(32896, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 512])] = pl.tile.mul(x_chunk_tail_inline1513__tile_2, x_chunk_tail_inline1513__tile_2)
                tmp_tile_5: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49280, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_5: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_33, pl.const(82080, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 1])] = pl.tile.row_sum(x_sq_tail_inline1503__tile_2, tmp_tile_5)
                x_sq_row_tail_inline1537__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_33, pl.const(82080, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1507__ssa_v0])] = pl.tile.reshape(t__tile_5, [1, 8])
                sq_sum_inline1490__tile_6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_25, pl.const(32896, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(sq_sum_inline1490__phi_v5_1, x_sq_row_tail_inline1537__tile_2)
                sq_sum_inline1490__tile_mv_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_28, pl.const(65664, pl.INT64), 32), pl.Mem.Vec] = pl.tile.move(sq_sum_inline1490__tile_6, target_memory=pl.Mem.Vec)
                sq_sum_inline1490__phi_v5_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_28, pl.const(65664, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__tile_mv_2)
            if valid_rows_inline1507__ssa_v0 == 8:
                x_chunk_full_inline1502__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(82112, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0_3], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                x_sq_full_inline1510__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(82112, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(x_chunk_full_inline1502__tile_3, x_chunk_full_inline1502__tile_3)
                tmp_tile_6: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_38, pl.const(98496, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_6: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_39, pl.const(114880, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(x_sq_full_inline1510__tile_3, tmp_tile_6)
                x_sq_row_full_inline1511__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(114880, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 8])
                sq_sum_inline1490__tile_7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(32832, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(sq_sum_inline1490__phi_v5_2, x_sq_row_full_inline1511__tile_3)
                sq_sum_inline1490__phi_v5_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(32832, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__tile_7)
            else:
                x_chunk_tail_inline1513__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(82112, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 512])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v0, k0_inline1501__ssa_v0_3], [8, 512], [valid_rows_inline1507__ssa_v0, 512], target_memory=pl.Mem.Vec)
                x_sq_tail_inline1503__tile_3: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_36, pl.const(82112, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 512])] = pl.tile.mul(x_chunk_tail_inline1513__tile_3, x_chunk_tail_inline1513__tile_3)
                tmp_tile_7: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_38, pl.const(98496, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_7: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_39, pl.const(114880, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v0, 1])] = pl.tile.row_sum(x_sq_tail_inline1503__tile_3, tmp_tile_7)
                x_sq_row_tail_inline1537__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(114880, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1507__ssa_v0])] = pl.tile.reshape(t__tile_7, [1, 8])
                sq_sum_inline1490__tile_8: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(32832, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(sq_sum_inline1490__phi_v5_2, x_sq_row_tail_inline1537__tile_3)
                sq_sum_inline1490__phi_v5_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(32832, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__tile_8)
            sq_sum_inline1490__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_2, pl.const(32832, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(sq_sum_inline1490__phi_v5_3)
        t__tile_8: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65696, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(sq_sum_inline1490__rv_v2, 6.103515625e-05)
        sq_mean_inline1514__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_3, pl.const(65696, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__tile_8, 9.9999999999999995e-07)
        rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_5, pl.const(114912, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(sq_mean_inline1514__tile, rsqrt_tmp)
        inv_inline1481__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [8, 1])
        inv_rms_inline1463__tile: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(inv_inline1481__tile, [t0_inline1476__ssa_v0, 0], inv_rms_inline1463__ssa_v0)
        return inv_rms_inline1463__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def hc_pre_rms_spmd(self, t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline1497__ssa_v0: pl.Tensor[[t_dim_inline1568__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], inv_rms_inline1463__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32]:
        inv_rms_inline1463__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = self.hc_pre_rms(t_dim_inline1568__ssa_v0, x_flat_inline1497__ssa_v0, inv_rms_inline1463__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing]})
        return inv_rms_inline1463__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def idx_qr_proj_dequant(idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 32768)], qr_proj_inline2268__ssa_v0: pl.Out[pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], bs_inline2301__ssa_v0: pl.Scalar[pl.INDEX], qr_acc_pad_inline2225__rv_v2: pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)], qr_scale_inline1310__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32]:
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        ot_inline2284__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        o_base_inline2220__ssa_v1: pl.Scalar[pl.INDEX] = ot_inline2284__ssa_v0 * 1024
        t__tile: pl.Tile[[1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(idx_wq_b_scale__ssa_v0, [o_base_inline2220__ssa_v1], [1024], [1024], target_memory=pl.Mem.Vec)
        wq_scale_inline2262__tile: pl.Tile[[1, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = t__tile
        for dq_t0_inline2203__idx_v0, (qr_proj_inline2268__iter_v1,) in pl.range(0, bs_inline2301__ssa_v0, 8, init_values=(qr_proj_inline2268__ssa_v0,)):
            t__tile_1: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_5, pl.const(4096, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(qr_acc_pad_inline2225__rv_v2, [dq_t0_inline2203__idx_v0, o_base_inline2220__ssa_v1], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            acc_fp32_inline2257__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4096, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode='none')
            qr_scale_tile_inline2236__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(36864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(qr_scale_inline1310__ssa_v0, [dq_t0_inline2203__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
            t__tile_2: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4096, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(acc_fp32_inline2257__tile, qr_scale_tile_inline2236__tile)
            qr_dequant_inline2240__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4096, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_2, wq_scale_inline2262__tile)
            qr_proj_inline2268__tile: pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_dequant_inline2240__tile, [dq_t0_inline2203__idx_v0, o_base_inline2220__ssa_v1], qr_proj_inline2268__iter_v1)
            qr_proj_inline2268__rv_v2: pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_proj_inline2268__tile)
        return qr_proj_inline2268__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def idx_qr_proj_dequant_spmd(self, idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 32768)], qr_proj_inline2268__ssa_v0: pl.Out[pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], bs_inline2301__ssa_v0: pl.Scalar[pl.INDEX], qr_acc_pad_inline2225__rv_v2: pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)], qr_scale_inline1310__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32]:
        qr_proj_inline2268__rv_v2: pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = self.idx_qr_proj_dequant(idx_wq_b_scale__ssa_v0, qr_proj_inline2268__ssa_v0, bs_inline2301__ssa_v0, qr_acc_pad_inline2225__rv_v2, qr_scale_inline1310__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]})
        return qr_proj_inline2268__ssa_v0
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def idx_qr_proj_matmul(bs_inline2301__ssa_v0: pl.Scalar[pl.INDEX], qr_acc_pad_inline2225__ssa_v0: pl.Out[pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8388608)]], qr_inline1255__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)]) -> pl.Tensor[[256, 8192], pl.INT32]:
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_left_9: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
        mem_right_10: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_11: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
        mem_right_12: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_21: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
        mem_left_23: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 1024)
        qr_unit_inline2215__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qr_rb_inline2194__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline2215__ssa_v0 // 8
        ot_inline2200__ssa_v0: pl.Scalar[pl.INDEX] = qr_unit_inline2215__ssa_v0 - qr_rb_inline2194__ssa_v0 * 8
        qr_r0_inline2195__ssa_v0: pl.Scalar[pl.INDEX] = qr_rb_inline2194__ssa_v0 * 16
        qr_rows_inline2208__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2301__ssa_v0 - qr_r0_inline2195__ssa_v0, 16)
        o_base_inline2220__ssa_v0: pl.Scalar[pl.INDEX] = ot_inline2200__ssa_v0 * 1024
        for ns_inline2239__idx_v0, (qr_acc_pad_inline2225__iter_v1,) in pl.range(0, 1024, 512, init_values=(qr_acc_pad_inline2225__ssa_v0,)):
            qr_acc_inline2212__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create([16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc)
            for kb_inline2210__idx_v0, (qr_acc_inline2212__iter_v1,) in pl.range(0, 4, 2, init_values=(qr_acc_inline2212__tile,)):
                q0_inline2232__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2210__idx_v0 * 256
                q0_inline2232__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline2210__idx_v0 * 256 + 256
                qr_tile_inline2201__tile: pl.Tile[[16, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 256])] = pl.tile.load(qr_inline1255__ssa_v0, [qr_r0_inline2195__ssa_v0, q0_inline2232__ssa_v0], [16, 256], [qr_rows_inline2208__ssa_v0, 256], target_memory=pl.Mem.Mat)
                wq_tile_inline2230__tile: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(4096, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(idx_wq_b__ssa_v0, [q0_inline2232__ssa_v0, o_base_inline2220__ssa_v0 + ns_inline2239__idx_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat)
                qr_tile_inline2201__tile_1: pl.Tile[[16, 256], pl.INT8, pl.MemRef(mem_mat_6, pl.const(135168, pl.INT64), 4096), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 256])] = pl.tile.load(qr_inline1255__ssa_v0, [qr_r0_inline2195__ssa_v0, q0_inline2232__ssa_v0_1], [16, 256], [qr_rows_inline2208__ssa_v0, 256], target_memory=pl.Mem.Mat)
                wq_tile_inline2230__tile_1: pl.Tile[[256, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(139264, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.load(idx_wq_b__ssa_v0, [q0_inline2232__ssa_v0_1, o_base_inline2220__ssa_v0 + ns_inline2239__idx_v0], [256, 512], [256, 512], target_memory=pl.Mem.Mat)
                if q0_inline2232__ssa_v0 == 0:
                    qr_acc_inline2212__tile_l0_init_storage: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc, compact=True)
                    qr_acc_inline2212__tile_l0_init: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(qr_acc_inline2212__tile_l0_init_storage, qr_rows_inline2208__ssa_v0, 512)
                    for qr_acc_inline2212__tile_l0_ko, (qr_acc_inline2212__tile_l0_c,) in pl.range(0, 256, 128, init_values=(qr_acc_inline2212__tile_l0_init,)):
                        qr_acc_inline2212__tile_l0_a: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_9, pl.const(3072, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_tile_inline2201__tile, 0, qr_acc_inline2212__tile_l0_ko, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline2212__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tile_inline2230__tile, qr_acc_inline2212__tile_l0_ko, 0, [64, 512], target_memory=pl.Mem.Right)
                        qr_acc_inline2212__tile_l0_a_1: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_tile_inline2201__tile, 0, qr_acc_inline2212__tile_l0_ko + 64, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline2212__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tile_inline2230__tile, qr_acc_inline2212__tile_l0_ko + 64, 0, [64, 512], target_memory=pl.Mem.Right)
                        qr_acc_inline2212__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(qr_acc_inline2212__tile_l0_c, qr_acc_inline2212__tile_l0_a, qr_acc_inline2212__tile_l0_b, qr_acc_inline2212__tile_l0_ko == 0)
                        qr_acc_inline2212__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(qr_acc_inline2212__tile_l0_c_acc, qr_acc_inline2212__tile_l0_a_1, qr_acc_inline2212__tile_l0_b_1, qr_acc_inline2212__tile_l0_ko == -64)
                        qr_acc_inline2212__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.yield_(qr_acc_inline2212__tile_l0_c_acc_1)
                    qr_acc_inline2212__phi_v5: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.yield_(qr_acc_inline2212__tile_1)
                else:
                    for qr_acc_inline2212__tile_l0_ko_1, (qr_acc_inline2212__tile_l0_c_1,) in pl.range(0, 256, 128, init_values=(qr_acc_inline2212__iter_v1,)):
                        qr_acc_inline2212__tile_l0_a_2: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_9, pl.const(3072, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_tile_inline2201__tile, 0, qr_acc_inline2212__tile_l0_ko_1, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline2212__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tile_inline2230__tile, qr_acc_inline2212__tile_l0_ko_1, 0, [64, 512], target_memory=pl.Mem.Right)
                        qr_acc_inline2212__tile_l0_a_3: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_tile_inline2201__tile, 0, qr_acc_inline2212__tile_l0_ko_1 + 64, [16, 64], target_memory=pl.Mem.Left)
                        qr_acc_inline2212__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tile_inline2230__tile, qr_acc_inline2212__tile_l0_ko_1 + 64, 0, [64, 512], target_memory=pl.Mem.Right)
                        qr_acc_inline2212__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(qr_acc_inline2212__tile_l0_c_1, qr_acc_inline2212__tile_l0_a_2, qr_acc_inline2212__tile_l0_b_2)
                        qr_acc_inline2212__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(qr_acc_inline2212__tile_l0_c_acc_2, qr_acc_inline2212__tile_l0_a_3, qr_acc_inline2212__tile_l0_b_3)
                        qr_acc_inline2212__tile_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(qr_acc_inline2212__tile_l0_c_acc_3)
                    qr_acc_inline2212__phi_v5: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.yield_(qr_acc_inline2212__tile_2)
                for qr_acc_inline2212__tile_l0_ko_2, (qr_acc_inline2212__tile_l0_c_2,) in pl.range(0, 256, 128, init_values=(qr_acc_inline2212__phi_v5,)):
                    qr_acc_inline2212__tile_l0_a_4: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_21, pl.const(1024, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_tile_inline2201__tile_1, 0, qr_acc_inline2212__tile_l0_ko_2, [16, 64], target_memory=pl.Mem.Left)
                    qr_acc_inline2212__tile_l0_b_4: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tile_inline2230__tile_1, qr_acc_inline2212__tile_l0_ko_2, 0, [64, 512], target_memory=pl.Mem.Right)
                    qr_acc_inline2212__tile_l0_a_5: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_23, pl.const(2048, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline2208__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_tile_inline2201__tile_1, 0, qr_acc_inline2212__tile_l0_ko_2 + 64, [16, 64], target_memory=pl.Mem.Left)
                    qr_acc_inline2212__tile_l0_b_5: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tile_inline2230__tile_1, qr_acc_inline2212__tile_l0_ko_2 + 64, 0, [64, 512], target_memory=pl.Mem.Right)
                    qr_acc_inline2212__tile_l0_c_acc_4: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(qr_acc_inline2212__tile_l0_c_2, qr_acc_inline2212__tile_l0_a_4, qr_acc_inline2212__tile_l0_b_4)
                    qr_acc_inline2212__tile_l0_c_acc_5: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(qr_acc_inline2212__tile_l0_c_acc_4, qr_acc_inline2212__tile_l0_a_5, qr_acc_inline2212__tile_l0_b_5)
                    qr_acc_inline2212__tile_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(qr_acc_inline2212__tile_l0_c_acc_5)
                qr_acc_inline2212__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(qr_acc_inline2212__tile_3)
            qr_acc_pad_inline2225__tile: pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8388608)] = pl.tile.store(qr_acc_inline2212__rv_v2, [qr_r0_inline2195__ssa_v0, o_base_inline2220__ssa_v0 + ns_inline2239__idx_v0], qr_acc_pad_inline2225__iter_v1)
            qr_acc_pad_inline2225__rv_v2: pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 8388608)] = pl.yield_(qr_acc_pad_inline2225__tile)
        return qr_acc_pad_inline2225__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def idx_qr_proj_matmul_spmd(self, bs_inline2301__ssa_v0: pl.Scalar[pl.INDEX], qr_acc_pad_inline2225__ssa_v0: pl.Out[pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8388608)]], qr_inline1255__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)]) -> pl.Tensor[[256, 8192], pl.INT32]:
        qr_acc_pad_inline2225__rv_v2: pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 8388608)] = self.idx_qr_proj_matmul(bs_inline2301__ssa_v0, qr_acc_pad_inline2225__ssa_v0, qr_inline1255__ssa_v0, idx_wq_b__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.input, pl.adir.input]})
        return qr_acc_pad_inline2225__ssa_v0
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def indexer_score_leaf_wave_aic(idx_positions_inline1323__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], score_arena_inline44_inline2267__ssa_v0: pl.Out[pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], qr_hadamard_i8_inline2177__rv_v2: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)], qr_hadamard_scale_dq_inline2234__ssa_v1: pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 65536)], weights_inline2244__ssa_v1: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 65536)], idx_block_table_flat_inline47_inline2186__ssa_v0: pl.Tensor[[idx_table_len_inline55_inline2193__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], kv_cache_i8_flat_inline46_inline2265__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)], kv_scale_flat_inline50_inline2214__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)], __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4)]]):
        pl.func_attr({"slot_num": 2})
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_12: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_left_13: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_14: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        mem_acc_15: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        mem_left_16: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_17: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 8192)
        indexer_score_leaf_wave_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name='indexer_score_leaf_wave_c2v_slot_buffer', peer_func='indexer_score_leaf_wave_aiv')
        pl.system.aic_initialize_pipe(indexer_score_leaf_wave_c2v_slot_buffer_import, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, slot_num=2)
        worker_inline75_inline2270__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        query_count_inline56_inline2271__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_positions_inline1323__ssa_v0, 0)
        for query_inline65_inline2274__idx_v0, (global_leaf_base_inline51_inline2273__iter_v1, score_arena_inline44_inline2267__iter_v1) in pl.range(query_count_inline56_inline2271__ssa_v0, init_values=(0, score_arena_inline44_inline2267__ssa_v0)):
            batch_idx_inline60_inline2249__ssa_v0: pl.Scalar[pl.INDEX] = query_inline65_inline2274__idx_v0 // 8
            position_inline54_inline2275__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_positions_inline1323__ssa_v0, [query_inline65_inline2274__idx_v0])
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0])
            cache_len_inline64_inline2279__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
            visible_count_inline49_inline2280__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len_inline64_inline2279__ssa_v0, (pl.cast(position_inline54_inline2275__tile, pl.INDEX) + 1) // 4), 262144), 0)
            leaf_count_inline66_inline2281__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count_inline49_inline2280__ssa_v0 + 8191) // 8192
            base_mod_inline52_inline2283__ssa_v0: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273__iter_v1 % 24
            first_leaf_inline67_inline2285__ssa_v0: pl.Scalar[pl.INDEX] = (worker_inline75_inline2270__ssa_v0 + base_mod_inline52_inline2283__ssa_v0) % 24
            for leaf_inline48_inline2288__idx_v0, (score_arena_inline44_inline2267__iter_v3,) in pl.range(first_leaf_inline67_inline2285__ssa_v0, leaf_count_inline66_inline2281__ssa_v0, 24, init_values=(score_arena_inline44_inline2267__iter_v1,)):
                logical_begin_inline63_inline2224__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline48_inline2288__idx_v0 * 8192
                valid_count_inline68_inline2219__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline49_inline2280__ssa_v0 - logical_begin_inline63_inline2224__ssa_v0, 8192)
                query_head_begin_inline69_inline2289__ssa_v0: pl.Scalar[pl.INDEX] = query_inline65_inline2274__idx_v0 * 64
                query_vector_inline70_inline2291__tile: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(qr_hadamard_i8_inline2177__rv_v2, [query_head_begin_inline69_inline2289__ssa_v0, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
                page_count_inline72_inline2252__ssa_v0: pl.Scalar[pl.INDEX] = (valid_count_inline68_inline2219__ssa_v0 + 31) // 32
                unroll_main_end: pl.Scalar[pl.INDEX] = page_count_inline72_inline2252__ssa_v0 // 2 * 2
                for page_inline43_inline2292__idx_v0, (score_arena_inline44_inline2267__iter_v5,) in pl.range(0, unroll_main_end, 2, init_values=(score_arena_inline44_inline2267__iter_v3,)):
                    page_begin_inline42_inline2287__ssa_v0: pl.Scalar[pl.INDEX] = page_inline43_inline2292__idx_v0 * 32
                    logical_row_inline41_inline2282__ssa_v0: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0 + page_begin_inline42_inline2287__ssa_v0
                    logical_page_inline74_inline2293__ssa_v0: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0 // 32
                    t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0 * 8192 + logical_page_inline74_inline2293__ssa_v0])
                    physical_block_inline40_inline2298__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                    physical_row_inline59_inline2299__ssa_v0: pl.Scalar[pl.INDEX] = physical_block_inline40_inline2298__ssa_v0 * 32
                    page_begin_inline42_inline2287__ssa_v0_1: pl.Scalar[pl.INDEX] = page_inline43_inline2292__idx_v0 * 32 + 32
                    logical_row_inline41_inline2282__ssa_v0_1: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0 + page_begin_inline42_inline2287__ssa_v0_1
                    logical_page_inline74_inline2293__ssa_v0_1: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0_1 // 32
                    t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0 * 8192 + logical_page_inline74_inline2293__ssa_v0_1])
                    physical_block_inline40_inline2298__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                    physical_row_inline59_inline2299__ssa_v0_1: pl.Scalar[pl.INDEX] = physical_block_inline40_inline2298__ssa_v0_1 * 32
                    kv_i8_inline38_inline2300__tile: pl.Tile[[32, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(kv_cache_i8_flat_inline46_inline2265__ssa_v0, [physical_row_inline59_inline2299__ssa_v0, 0], [32, 128], [32, 128], target_memory=pl.Mem.Mat)
                    kv_i8_inline38_inline2300__tile_1: pl.Tile[[32, 128], pl.INT8, pl.MemRef(mem_mat_12, pl.const(12288, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(kv_cache_i8_flat_inline46_inline2265__ssa_v0, [physical_row_inline59_inline2299__ssa_v0_1, 0], [32, 128], [32, 128], target_memory=pl.Mem.Mat)
                    query_vector_inline70_inline2291__tile_t: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(query_vector_inline70_inline2291__tile)
                    kv_i8_inline38_inline2300__tile_Left: pl.Tile[[32, 128], pl.INT8, pl.MemRef(mem_left_13, pl.const(0, pl.INT64), 4096), pl.Mem.Left] = pl.tile.move(kv_i8_inline38_inline2300__tile, target_memory=pl.Mem.Left)
                    query_vector_inline70_inline2291__tile_t_Right: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(query_vector_inline70_inline2291__tile_t, target_memory=pl.Mem.Right)
                    score_i32_inline37_inline2302__tile: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_acc_15, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul(kv_i8_inline38_inline2300__tile_Left, query_vector_inline70_inline2291__tile_t_Right)
                    pl.tile.tpush_to_aiv(score_i32_inline37_inline2302__tile, split=0)
                    query_vector_inline70_inline2291__tile_t_1: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(query_vector_inline70_inline2291__tile)
                    kv_i8_inline38_inline2300__tile_Left_1: pl.Tile[[32, 128], pl.INT8, pl.MemRef(mem_left_16, pl.const(4096, pl.INT64), 4096), pl.Mem.Left] = pl.tile.move(kv_i8_inline38_inline2300__tile_1, target_memory=pl.Mem.Left)
                    query_vector_inline70_inline2291__tile_t_Right_1: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_17, pl.const(8192, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(query_vector_inline70_inline2291__tile_t_1, target_memory=pl.Mem.Right)
                    score_i32_inline37_inline2302__tile_1: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_acc_15, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul(kv_i8_inline38_inline2300__tile_Left_1, query_vector_inline70_inline2291__tile_t_Right_1)
                    pl.tile.tpush_to_aiv(score_i32_inline37_inline2302__tile_1, split=0)
                    score_arena_inline44_inline2267__rv_v6_main: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__iter_v5)
                unroll_rem: pl.Scalar[pl.INDEX] = page_count_inline72_inline2252__ssa_v0 - unroll_main_end
                if unroll_rem == 1:
                    page_begin_inline42_inline2287__ssa_v0_2: pl.Scalar[pl.INDEX] = unroll_main_end * 32
                    logical_row_inline41_inline2282__ssa_v0_2: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0 + page_begin_inline42_inline2287__ssa_v0_2
                    logical_page_inline74_inline2293__ssa_v0_2: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0_2 // 32
                    t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0 * 8192 + logical_page_inline74_inline2293__ssa_v0_2])
                    physical_block_inline40_inline2298__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_3, pl.INDEX)
                    physical_row_inline59_inline2299__ssa_v0_2: pl.Scalar[pl.INDEX] = physical_block_inline40_inline2298__ssa_v0_2 * 32
                    kv_i8_inline38_inline2300__tile_2: pl.Tile[[32, 128], pl.INT8, pl.MemRef(mem_mat_11, pl.const(8192, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(kv_cache_i8_flat_inline46_inline2265__ssa_v0, [physical_row_inline59_inline2299__ssa_v0_2, 0], [32, 128], [32, 128], target_memory=pl.Mem.Mat)
                    query_vector_inline70_inline2291__tile_t_2: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(query_vector_inline70_inline2291__tile)
                    kv_i8_inline38_inline2300__tile_Left_2: pl.Tile[[32, 128], pl.INT8, pl.MemRef(mem_left_13, pl.const(0, pl.INT64), 4096), pl.Mem.Left] = pl.tile.move(kv_i8_inline38_inline2300__tile_2, target_memory=pl.Mem.Left)
                    query_vector_inline70_inline2291__tile_t_Right_2: pl.Tile[[128, 64], pl.INT8, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 8192), pl.Mem.Right] = pl.tile.move(query_vector_inline70_inline2291__tile_t_2, target_memory=pl.Mem.Right)
                    score_i32_inline37_inline2302__tile_2: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_acc_15, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul(kv_i8_inline38_inline2300__tile_Left_2, query_vector_inline70_inline2291__tile_t_Right_2)
                    pl.tile.tpush_to_aiv(score_i32_inline37_inline2302__tile_2, split=0)
                    score_arena_inline44_inline2267__rv_v6: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__rv_v6_main)
                else:
                    score_arena_inline44_inline2267__rv_v6: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__rv_v6_main)
                score_arena_inline44_inline2267__rv_v4: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__rv_v6)
            global_leaf_base_inline51_inline2273__ssa_v3: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273__iter_v1 + leaf_count_inline66_inline2281__ssa_v0
            global_leaf_base_inline51_inline2273__rv_v2, score_arena_inline44_inline2267__rv_v2 = pl.yield_(global_leaf_base_inline51_inline2273__ssa_v3, score_arena_inline44_inline2267__rv_v4)
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def indexer_score_leaf_wave_aiv(idx_positions_inline1323__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], score_arena_inline44_inline2267__ssa_v0: pl.Out[pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], qr_hadamard_i8_inline2177__rv_v2: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)], qr_hadamard_scale_dq_inline2234__ssa_v1: pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 65536)], weights_inline2244__ssa_v1: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 65536)], idx_block_table_flat_inline47_inline2186__ssa_v0: pl.Tensor[[idx_table_len_inline55_inline2193__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], kv_cache_i8_flat_inline46_inline2265__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)], kv_scale_flat_inline50_inline2214__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)], __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4)]]) -> pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32]:
        pl.func_attr({"slot_num": 2, "dual_aiv_dispatch": True})
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_26: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        indexer_score_leaf_wave_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name='indexer_score_leaf_wave_c2v_slot_buffer', size=16384, base=0)
        pl.system.aiv_initialize_pipe(indexer_score_leaf_wave_c2v_slot_buffer, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, slot_num=2)
        if subblock_idx == 0:
            worker_inline75_inline2270__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            query_count_inline56_inline2271__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_positions_inline1323__ssa_v0, 0)
            global_leaf_base_inline51_inline2273__ssa_v0: pl.Scalar[pl.INDEX] = 0
            for query_inline65_inline2274__idx_v0, (global_leaf_base_inline51_inline2273__iter_v1, score_arena_inline44_inline2267__iter_v1) in pl.range(query_count_inline56_inline2271__ssa_v0, init_values=(global_leaf_base_inline51_inline2273__ssa_v0, score_arena_inline44_inline2267__ssa_v0)):
                batch_idx_inline60_inline2249__ssa_v0: pl.Scalar[pl.INDEX] = query_inline65_inline2274__idx_v0 // 8
                position_inline54_inline2275__tile: pl.Scalar[pl.INT32] = pl.tensor.read(idx_positions_inline1323__ssa_v0, [query_inline65_inline2274__idx_v0])
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0])
                cache_len_inline64_inline2279__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX) // 4
                visible_count_inline49_inline2280__ssa_v0: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len_inline64_inline2279__ssa_v0, (pl.cast(position_inline54_inline2275__tile, pl.INDEX) + 1) // 4), 262144), 0)
                leaf_count_inline66_inline2281__ssa_v0: pl.Scalar[pl.INDEX] = (visible_count_inline49_inline2280__ssa_v0 + 8191) // 8192
                base_mod_inline52_inline2283__ssa_v0: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273__iter_v1 % 24
                first_leaf_inline67_inline2285__ssa_v0: pl.Scalar[pl.INDEX] = (worker_inline75_inline2270__ssa_v0 + base_mod_inline52_inline2283__ssa_v0) % 24
                for leaf_inline48_inline2288__idx_v0, (score_arena_inline44_inline2267__iter_v3,) in pl.range(first_leaf_inline67_inline2285__ssa_v0, leaf_count_inline66_inline2281__ssa_v0, 24, init_values=(score_arena_inline44_inline2267__iter_v1,)):
                    logical_begin_inline63_inline2224__ssa_v0: pl.Scalar[pl.INDEX] = leaf_inline48_inline2288__idx_v0 * 8192
                    valid_count_inline68_inline2219__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline49_inline2280__ssa_v0 - logical_begin_inline63_inline2224__ssa_v0, 8192)
                    query_head_begin_inline69_inline2289__ssa_v0: pl.Scalar[pl.INDEX] = query_inline65_inline2274__idx_v0 * 64
                    t__tile_1: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(qr_hadamard_scale_dq_inline2234__ssa_v1, [query_head_begin_inline69_inline2289__ssa_v0, 0], [64, 1], [64, 1], target_memory=pl.Mem.Vec)
                    query_scale_inline73_inline2197__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 64])
                    query_weight_inline76_inline2243__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(weights_inline2244__ssa_v1, [query_inline65_inline2274__idx_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    page_count_inline72_inline2252__ssa_v0: pl.Scalar[pl.INDEX] = (valid_count_inline68_inline2219__ssa_v0 + 31) // 32
                    unroll_main_end: pl.Scalar[pl.INDEX] = page_count_inline72_inline2252__ssa_v0 // 2 * 2
                    for page_inline43_inline2292__idx_v0, (score_arena_inline44_inline2267__iter_v5,) in pl.range(0, unroll_main_end, 2, init_values=(score_arena_inline44_inline2267__iter_v3,)):
                        page_begin_inline42_inline2287__ssa_v0: pl.Scalar[pl.INDEX] = page_inline43_inline2292__idx_v0 * 32
                        logical_row_inline41_inline2282__ssa_v0: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0 + page_begin_inline42_inline2287__ssa_v0
                        logical_page_inline74_inline2293__ssa_v0: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0 // 32
                        t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0 * 8192 + logical_page_inline74_inline2293__ssa_v0])
                        physical_block_inline40_inline2298__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                        physical_row_inline59_inline2299__ssa_v0: pl.Scalar[pl.INDEX] = physical_block_inline40_inline2298__ssa_v0 * 32
                        valid_rows_inline39_inline2216__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(valid_count_inline68_inline2219__ssa_v0 - page_begin_inline42_inline2287__ssa_v0, 32)
                        page_begin_inline42_inline2287__ssa_v0_1: pl.Scalar[pl.INDEX] = page_inline43_inline2292__idx_v0 * 32 + 32
                        logical_row_inline41_inline2282__ssa_v0_1: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0 + page_begin_inline42_inline2287__ssa_v0_1
                        logical_page_inline74_inline2293__ssa_v0_1: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0_1 // 32
                        t__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0 * 8192 + logical_page_inline74_inline2293__ssa_v0_1])
                        physical_block_inline40_inline2298__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_3, pl.INDEX)
                        physical_row_inline59_inline2299__ssa_v0_1: pl.Scalar[pl.INDEX] = physical_block_inline40_inline2298__ssa_v0_1 * 32
                        valid_rows_inline39_inline2216__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(valid_count_inline68_inline2219__ssa_v0 - page_begin_inline42_inline2287__ssa_v0_1, 32)
                        kv_scale_inline32_inline2184__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(16896, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(kv_scale_flat_inline50_inline2214__ssa_v0, [physical_row_inline59_inline2299__ssa_v0, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                        kv_scale_inline32_inline2184__tile_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(17024, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(kv_scale_flat_inline50_inline2214__ssa_v0, [physical_row_inline59_inline2299__ssa_v0_1, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                        score_i32_inline37_inline2302__tile_Vec: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
                        score_fp32_inline35_inline2254__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(score_i32_inline37_inline2302__tile_Vec, target_type=pl.FP32, mode='none')
                        pl.system.tfree_to_aic(score_i32_inline37_inline2302__tile_Vec, split=0)
                        score_fp32_v1_inline34_inline2251__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(score_fp32_inline35_inline2254__tile, query_scale_inline73_inline2197__tile)
                        score_fp32_v2_inline62_inline2294__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.maximums(score_fp32_v1_inline34_inline2251__tile, 0.0)
                        score_fp32_v3_inline33_inline2187__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(17152, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(score_fp32_v2_inline62_inline2294__tile, query_weight_inline76_inline2243__tile)
                        tmp_tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        t__tile_4: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(41728, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(score_fp32_v3_inline33_inline2187__tile, tmp_tile)
                        score_inline58_inline2180__rm_a0_tmp_v0: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_19, pl.const(41728, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_4, [1, 32])
                        score_inline58_inline2180__rm_a1_tmp_v1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(16896, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(kv_scale_inline32_inline2184__tile, [1, 32])
                        score_inline58_inline2180__row_major_tmp_v2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(score_inline58_inline2180__rm_a0_tmp_v0, score_inline58_inline2180__rm_a1_tmp_v1)
                        score_inline58_inline2180__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(score_inline58_inline2180__row_major_tmp_v2, [32, 1])
                        score_row_inline31_inline2296__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(score_inline58_inline2180__tile, [1, 32])
                        t__tile_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline39_inline2216__ssa_v0])] = pl.tile.set_validshape(score_row_inline31_inline2296__tile, 1, valid_rows_inline39_inline2216__ssa_v0)
                        score_valid_inline30_inline2277__tile: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(t__tile_5, pad_value=pl.PadValue.min)
                        score_arena_inline44_inline2267__tile: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(score_valid_inline30_inline2277__tile, [query_inline65_inline2274__idx_v0, logical_row_inline41_inline2282__ssa_v0], score_arena_inline44_inline2267__iter_v5)
                        score_i32_inline37_inline2302__tile_Vec_1: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
                        score_fp32_inline35_inline2254__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(score_i32_inline37_inline2302__tile_Vec_1, target_type=pl.FP32, mode='none')
                        pl.system.tfree_to_aic(score_i32_inline37_inline2302__tile_Vec_1, split=0)
                        score_fp32_v1_inline34_inline2251__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(score_fp32_inline35_inline2254__tile_1, query_scale_inline73_inline2197__tile)
                        score_fp32_v2_inline62_inline2294__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.maximums(score_fp32_v1_inline34_inline2251__tile_1, 0.0)
                        score_fp32_v3_inline33_inline2187__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(41856, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(score_fp32_v2_inline62_inline2294__tile_1, query_weight_inline76_inline2243__tile)
                        tmp_tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        t__tile_6: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(66432, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(score_fp32_v3_inline33_inline2187__tile_1, tmp_tile_1)
                        score_inline58_inline2180__rm_a0_tmp_v0_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_27, pl.const(66432, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [1, 32])
                        score_inline58_inline2180__rm_a1_tmp_v1_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_13, pl.const(17024, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(kv_scale_inline32_inline2184__tile_1, [1, 32])
                        score_inline58_inline2180__row_major_tmp_v2_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(score_inline58_inline2180__rm_a0_tmp_v0_1, score_inline58_inline2180__rm_a1_tmp_v1_1)
                        score_inline58_inline2180__tile_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(score_inline58_inline2180__row_major_tmp_v2_1, [32, 1])
                        score_row_inline31_inline2296__tile_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(score_inline58_inline2180__tile_1, [1, 32])
                        t__tile_7: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline39_inline2216__ssa_v0_1])] = pl.tile.set_validshape(score_row_inline31_inline2296__tile_1, 1, valid_rows_inline39_inline2216__ssa_v0_1)
                        score_valid_inline30_inline2277__tile_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(t__tile_7, pad_value=pl.PadValue.min)
                        score_arena_inline44_inline2267__tile_1: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(score_valid_inline30_inline2277__tile_1, [query_inline65_inline2274__idx_v0, logical_row_inline41_inline2282__ssa_v0_1], score_arena_inline44_inline2267__tile)
                        score_arena_inline44_inline2267__rv_v6_main: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__tile_1)
                    unroll_rem: pl.Scalar[pl.INDEX] = page_count_inline72_inline2252__ssa_v0 - unroll_main_end
                    if unroll_rem == 1:
                        page_begin_inline42_inline2287__ssa_v0_2: pl.Scalar[pl.INDEX] = unroll_main_end * 32
                        logical_row_inline41_inline2282__ssa_v0_2: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0 + page_begin_inline42_inline2287__ssa_v0_2
                        logical_page_inline74_inline2293__ssa_v0_2: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0_2 // 32
                        t__tile_8: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0 * 8192 + logical_page_inline74_inline2293__ssa_v0_2])
                        physical_block_inline40_inline2298__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.cast(t__tile_8, pl.INDEX)
                        physical_row_inline59_inline2299__ssa_v0_2: pl.Scalar[pl.INDEX] = physical_block_inline40_inline2298__ssa_v0_2 * 32
                        score_i32_inline37_inline2302__tile_Vec_2: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
                        score_fp32_inline35_inline2254__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(score_i32_inline37_inline2302__tile_Vec_2, target_type=pl.FP32, mode='none')
                        pl.system.tfree_to_aic(score_i32_inline37_inline2302__tile_Vec_2, split=0)
                        score_fp32_v1_inline34_inline2251__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(score_fp32_inline35_inline2254__tile_2, query_scale_inline73_inline2197__tile)
                        score_fp32_v2_inline62_inline2294__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.maximums(score_fp32_v1_inline34_inline2251__tile_2, 0.0)
                        score_fp32_v3_inline33_inline2187__tile_2: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(score_fp32_v2_inline62_inline2294__tile_2, query_weight_inline76_inline2243__tile)
                        kv_scale_inline32_inline2184__tile_2: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(17152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(kv_scale_flat_inline50_inline2214__ssa_v0, [physical_row_inline59_inline2299__ssa_v0_2, 0], [32, 1], [32, 1], target_memory=pl.Mem.Vec)
                        tmp_tile_2: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        t__tile_9: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(41856, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(score_fp32_v3_inline33_inline2187__tile_2, tmp_tile_2)
                        score_inline58_inline2180__rm_a0_tmp_v0_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(41856, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [1, 32])
                        score_inline58_inline2180__rm_a1_tmp_v1_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_17, pl.const(17152, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(kv_scale_inline32_inline2184__tile_2, [1, 32])
                        score_inline58_inline2180__row_major_tmp_v2_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.mul(score_inline58_inline2180__rm_a0_tmp_v0_2, score_inline58_inline2180__rm_a1_tmp_v1_2)
                        score_inline58_inline2180__tile_2: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(score_inline58_inline2180__row_major_tmp_v2_2, [32, 1])
                        score_row_inline31_inline2296__tile_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec] = pl.tile.reshape(score_inline58_inline2180__tile_2, [1, 32])
                        valid_rows_inline39_inline2216__ssa_v0_2: pl.Scalar[pl.INDEX] = pl.min(valid_count_inline68_inline2219__ssa_v0 - page_begin_inline42_inline2287__ssa_v0_2, 32)
                        t__tile_10: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline39_inline2216__ssa_v0_2])] = pl.tile.set_validshape(score_row_inline31_inline2296__tile_2, 1, valid_rows_inline39_inline2216__ssa_v0_2)
                        score_valid_inline30_inline2277__tile_2: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(pad=pl.PadValue.min)] = pl.tile.fillpad(t__tile_10, pad_value=pl.PadValue.min)
                        score_arena_inline44_inline2267__tile_2: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = pl.tile.store(score_valid_inline30_inline2277__tile_2, [query_inline65_inline2274__idx_v0, logical_row_inline41_inline2282__ssa_v0_2], score_arena_inline44_inline2267__rv_v6_main)
                        score_arena_inline44_inline2267__rv_v6: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__tile_2)
                    else:
                        score_arena_inline44_inline2267__rv_v6: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__rv_v6_main)
                    score_arena_inline44_inline2267__rv_v4: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__rv_v6)
                global_leaf_base_inline51_inline2273__ssa_v3: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273__iter_v1 + leaf_count_inline66_inline2281__ssa_v0
                global_leaf_base_inline51_inline2273__rv_v2, score_arena_inline44_inline2267__rv_v2 = pl.yield_(global_leaf_base_inline51_inline2273__ssa_v3, score_arena_inline44_inline2267__rv_v4)
            return score_arena_inline44_inline2267__ssa_v0
        else:
            worker_inline75_inline2270__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            query_count_inline56_inline2271__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_positions_inline1323__ssa_v0, 0)
            global_leaf_base_inline51_inline2273__ssa_v0_1: pl.Scalar[pl.INDEX] = 0
            for query_inline65_inline2274__idx_v0_1, (global_leaf_base_inline51_inline2273__iter_v1_1, score_arena_inline44_inline2267__iter_v1_1) in pl.range(query_count_inline56_inline2271__ssa_v0_1, init_values=(global_leaf_base_inline51_inline2273__ssa_v0_1, score_arena_inline44_inline2267__ssa_v0)):
                batch_idx_inline60_inline2249__ssa_v0_1: pl.Scalar[pl.INDEX] = query_inline65_inline2274__idx_v0_1 // 8
                position_inline54_inline2275__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(idx_positions_inline1323__ssa_v0, [query_inline65_inline2274__idx_v0_1])
                t__tile_11: pl.Scalar[pl.INT32] = pl.tensor.read(kv_seq_lens__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0_1])
                cache_len_inline64_inline2279__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_11, pl.INDEX) // 4
                visible_count_inline49_inline2280__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.max(pl.min(pl.min(cache_len_inline64_inline2279__ssa_v0_1, (pl.cast(position_inline54_inline2275__tile_1, pl.INDEX) + 1) // 4), 262144), 0)
                leaf_count_inline66_inline2281__ssa_v0_1: pl.Scalar[pl.INDEX] = (visible_count_inline49_inline2280__ssa_v0_1 + 8191) // 8192
                base_mod_inline52_inline2283__ssa_v0_1: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273__iter_v1_1 % 24
                first_leaf_inline67_inline2285__ssa_v0_1: pl.Scalar[pl.INDEX] = (worker_inline75_inline2270__ssa_v0_1 + base_mod_inline52_inline2283__ssa_v0_1) % 24
                for leaf_inline48_inline2288__idx_v0_1, (score_arena_inline44_inline2267__iter_v3_1,) in pl.range(first_leaf_inline67_inline2285__ssa_v0_1, leaf_count_inline66_inline2281__ssa_v0_1, 24, init_values=(score_arena_inline44_inline2267__iter_v1_1,)):
                    logical_begin_inline63_inline2224__ssa_v0_1: pl.Scalar[pl.INDEX] = leaf_inline48_inline2288__idx_v0_1 * 8192
                    valid_count_inline68_inline2219__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(visible_count_inline49_inline2280__ssa_v0_1 - logical_begin_inline63_inline2224__ssa_v0_1, 8192)
                    t__tile_12: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(16384, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([64, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    query_scale_inline73_inline2197__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(16384, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(t__tile_12, [1, 64])
                    query_weight_inline76_inline2243__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(16640, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    page_count_inline72_inline2252__ssa_v0_1: pl.Scalar[pl.INDEX] = (valid_count_inline68_inline2219__ssa_v0_1 + 31) // 32
                    unroll_main_end_1: pl.Scalar[pl.INDEX] = page_count_inline72_inline2252__ssa_v0_1 // 2 * 2
                    for page_inline43_inline2292__idx_v0_1, (score_arena_inline44_inline2267__iter_v5_1,) in pl.range(0, unroll_main_end_1, 2, init_values=(score_arena_inline44_inline2267__iter_v3_1,)):
                        page_begin_inline42_inline2287__ssa_v0_3: pl.Scalar[pl.INDEX] = page_inline43_inline2292__idx_v0_1 * 32
                        logical_row_inline41_inline2282__ssa_v0_3: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0_1 + page_begin_inline42_inline2287__ssa_v0_3
                        logical_page_inline74_inline2293__ssa_v0_3: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0_3 // 32
                        t__tile_13: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0_1 * 8192 + logical_page_inline74_inline2293__ssa_v0_3])
                        page_begin_inline42_inline2287__ssa_v0_4: pl.Scalar[pl.INDEX] = page_inline43_inline2292__idx_v0_1 * 32 + 32
                        logical_row_inline41_inline2282__ssa_v0_4: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0_1 + page_begin_inline42_inline2287__ssa_v0_4
                        logical_page_inline74_inline2293__ssa_v0_4: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0_4 // 32
                        t__tile_14: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0_1 * 8192 + logical_page_inline74_inline2293__ssa_v0_4])
                        score_arena_inline44_inline2267__tile_3: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 0)] = score_arena_inline44_inline2267__iter_v5_1
                        score_arena_inline44_inline2267__tile_4: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 0)] = score_arena_inline44_inline2267__iter_v5_1
                        score_i32_inline37_inline2302__tile_Vec_3: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.tpop_from_aic(split=0)
                        score_fp32_inline35_inline2254__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.cast(score_i32_inline37_inline2302__tile_Vec_3, target_type=pl.FP32, mode='none')
                        pl.system.tfree_to_aic(score_i32_inline37_inline2302__tile_Vec_3, split=0)
                        score_fp32_v1_inline34_inline2251__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.col_expand_mul(score_fp32_inline35_inline2254__tile_3, query_scale_inline73_inline2197__tile_1)
                        score_fp32_v2_inline62_inline2294__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.maximums(score_fp32_v1_inline34_inline2251__tile_3, 0.0)
                        score_fp32_v3_inline33_inline2187__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(17152, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.col_expand_mul(score_fp32_v2_inline62_inline2294__tile_3, query_weight_inline76_inline2243__tile_1)
                        kv_scale_inline32_inline2184__tile_3: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(16896, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([32, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        tmp_tile_3: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        t__tile_15: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(17024, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.row_sum(score_fp32_v3_inline33_inline2187__tile_3, tmp_tile_3)
                        score_inline58_inline2180__rm_a0_tmp_v0_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_13, pl.const(17024, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(t__tile_15, [1, 32])
                        score_inline58_inline2180__rm_a1_tmp_v1_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_12, pl.const(16896, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(kv_scale_inline32_inline2184__tile_3, [1, 32])
                        score_inline58_inline2180__row_major_tmp_v2_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.mul(score_inline58_inline2180__rm_a0_tmp_v0_3, score_inline58_inline2180__rm_a1_tmp_v1_3)
                        score_inline58_inline2180__tile_3: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(score_inline58_inline2180__row_major_tmp_v2_3, [32, 1])
                        score_row_inline31_inline2296__tile_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(score_inline58_inline2180__tile_3, [1, 32])
                        t__tile_16: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.set_validshape(score_row_inline31_inline2296__tile_3, 0, 0)
                        score_valid_inline30_inline2277__tile_3: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0], pad=pl.PadValue.min)] = pl.tile.fillpad(t__tile_16, pad_value=pl.PadValue.min)
                        score_i32_inline37_inline2302__tile_Vec_4: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.tpop_from_aic(split=0)
                        score_fp32_inline35_inline2254__tile_4: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.cast(score_i32_inline37_inline2302__tile_Vec_4, target_type=pl.FP32, mode='none')
                        pl.system.tfree_to_aic(score_i32_inline37_inline2302__tile_Vec_4, split=0)
                        score_fp32_v1_inline34_inline2251__tile_4: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.col_expand_mul(score_fp32_inline35_inline2254__tile_4, query_scale_inline73_inline2197__tile_1)
                        score_fp32_v2_inline62_inline2294__tile_4: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.maximums(score_fp32_v1_inline34_inline2251__tile_4, 0.0)
                        score_fp32_v3_inline33_inline2187__tile_4: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(41856, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.col_expand_mul(score_fp32_v2_inline62_inline2294__tile_4, query_weight_inline76_inline2243__tile_1)
                        kv_scale_inline32_inline2184__tile_4: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_19, pl.const(41728, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([32, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        tmp_tile_4: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        t__tile_17: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(66432, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.row_sum(score_fp32_v3_inline33_inline2187__tile_4, tmp_tile_4)
                        score_inline58_inline2180__rm_a0_tmp_v0_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_27, pl.const(66432, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(t__tile_17, [1, 32])
                        score_inline58_inline2180__rm_a1_tmp_v1_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_19, pl.const(41728, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(kv_scale_inline32_inline2184__tile_4, [1, 32])
                        score_inline58_inline2180__row_major_tmp_v2_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.mul(score_inline58_inline2180__rm_a0_tmp_v0_4, score_inline58_inline2180__rm_a1_tmp_v1_4)
                        score_inline58_inline2180__tile_4: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(score_inline58_inline2180__row_major_tmp_v2_4, [32, 1])
                        score_row_inline31_inline2296__tile_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(score_inline58_inline2180__tile_4, [1, 32])
                        t__tile_18: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.set_validshape(score_row_inline31_inline2296__tile_4, 0, 0)
                        score_valid_inline30_inline2277__tile_4: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0], pad=pl.PadValue.min)] = pl.tile.fillpad(t__tile_18, pad_value=pl.PadValue.min)
                        score_arena_inline44_inline2267__rv_v6_main_1: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__iter_v5_1)
                    unroll_rem_1: pl.Scalar[pl.INDEX] = page_count_inline72_inline2252__ssa_v0_1 - unroll_main_end_1
                    if unroll_rem_1 == 1:
                        page_begin_inline42_inline2287__ssa_v0_5: pl.Scalar[pl.INDEX] = unroll_main_end_1 * 32
                        logical_row_inline41_inline2282__ssa_v0_5: pl.Scalar[pl.INDEX] = logical_begin_inline63_inline2224__ssa_v0_1 + page_begin_inline42_inline2287__ssa_v0_5
                        logical_page_inline74_inline2293__ssa_v0_5: pl.Scalar[pl.INDEX] = logical_row_inline41_inline2282__ssa_v0_5 // 32
                        t__tile_19: pl.Scalar[pl.INT32] = pl.tensor.read(idx_block_table_flat_inline47_inline2186__ssa_v0, [batch_idx_inline60_inline2249__ssa_v0_1 * 8192 + logical_page_inline74_inline2293__ssa_v0_5])
                        score_i32_inline37_inline2302__tile_Vec_5: pl.Tile[[32, 64], pl.INT32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.tpop_from_aic(split=0)
                        score_fp32_inline35_inline2254__tile_5: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.cast(score_i32_inline37_inline2302__tile_Vec_5, target_type=pl.FP32, mode='none')
                        pl.system.tfree_to_aic(score_i32_inline37_inline2302__tile_Vec_5, split=0)
                        score_fp32_v1_inline34_inline2251__tile_5: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.col_expand_mul(score_fp32_inline35_inline2254__tile_5, query_scale_inline73_inline2197__tile_1)
                        score_fp32_v2_inline62_inline2294__tile_5: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.maximums(score_fp32_v1_inline34_inline2251__tile_5, 0.0)
                        score_fp32_v3_inline33_inline2187__tile_5: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_26, pl.const(50048, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.col_expand_mul(score_fp32_v2_inline62_inline2294__tile_5, query_weight_inline76_inline2243__tile_1)
                        kv_scale_inline32_inline2184__tile_5: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(17152, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([32, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        tmp_tile_5: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        t__tile_20: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_25, pl.const(41856, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.row_sum(score_fp32_v3_inline33_inline2187__tile_5, tmp_tile_5)
                        score_inline58_inline2180__rm_a0_tmp_v0_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_25, pl.const(41856, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(t__tile_20, [1, 32])
                        score_inline58_inline2180__rm_a1_tmp_v1_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_17, pl.const(17152, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(kv_scale_inline32_inline2184__tile_5, [1, 32])
                        score_inline58_inline2180__row_major_tmp_v2_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.mul(score_inline58_inline2180__rm_a0_tmp_v0_5, score_inline58_inline2180__rm_a1_tmp_v1_5)
                        score_inline58_inline2180__tile_5: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(score_inline58_inline2180__row_major_tmp_v2_5, [32, 1])
                        score_row_inline31_inline2296__tile_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(score_inline58_inline2180__tile_5, [1, 32])
                        t__tile_21: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.set_validshape(score_row_inline31_inline2296__tile_5, 0, 0)
                        score_valid_inline30_inline2277__tile_5: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_18, pl.const(25344, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0], pad=pl.PadValue.min)] = pl.tile.fillpad(t__tile_21, pad_value=pl.PadValue.min)
                        score_arena_inline44_inline2267__tile_5: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_75", pl.const(0, pl.INT64), 0)] = score_arena_inline44_inline2267__rv_v6_main_1
                        score_arena_inline44_inline2267__rv_v6_1: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_76", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__rv_v6_main_1)
                    else:
                        score_arena_inline44_inline2267__rv_v6_1: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_76", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__rv_v6_main_1)
                    score_arena_inline44_inline2267__rv_v4_1: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_77", pl.const(0, pl.INT64), 0)] = pl.yield_(score_arena_inline44_inline2267__rv_v6_1)
                global_leaf_base_inline51_inline2273__ssa_v3_1: pl.Scalar[pl.INDEX] = global_leaf_base_inline51_inline2273__iter_v1_1 + leaf_count_inline66_inline2281__ssa_v0_1
                global_leaf_base_inline51_inline2273__rv_v2_1, score_arena_inline44_inline2267__rv_v2_1 = pl.yield_(global_leaf_base_inline51_inline2273__ssa_v3_1, score_arena_inline44_inline2267__rv_v4_1)
            return score_arena_inline44_inline2267__ssa_v0
    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def indexer_score_leaf_wave(idx_positions_inline1323__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], score_arena_inline44_inline2267__ssa_v0: pl.Out[pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], qr_hadamard_i8_inline2177__rv_v2: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)], qr_hadamard_scale_dq_inline2234__ssa_v1: pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 65536)], weights_inline2244__ssa_v1: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 65536)], idx_block_table_flat_inline47_inline2186__ssa_v0: pl.Tensor[[idx_table_len_inline55_inline2193__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], kv_cache_i8_flat_inline46_inline2265__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)], kv_scale_flat_inline50_inline2214__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)], __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4)]]) -> pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32]:
        pl.func_attr({"slot_num": 2})
        self.indexer_score_leaf_wave_aic(idx_positions_inline1323__ssa_v0, score_arena_inline44_inline2267__ssa_v0, kv_seq_lens__ssa_v0, qr_hadamard_i8_inline2177__rv_v2, qr_hadamard_scale_dq_inline2234__ssa_v1, weights_inline2244__ssa_v1, idx_block_table_flat_inline47_inline2186__ssa_v0, kv_cache_i8_flat_inline46_inline2265__ssa_v0, kv_scale_flat_inline50_inline2214__ssa_v0, __gm_pipe_buffer, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
        self.indexer_score_leaf_wave_aiv(idx_positions_inline1323__ssa_v0, score_arena_inline44_inline2267__ssa_v0, kv_seq_lens__ssa_v0, qr_hadamard_i8_inline2177__rv_v2, qr_hadamard_scale_dq_inline2234__ssa_v1, weights_inline2244__ssa_v1, idx_block_table_flat_inline47_inline2186__ssa_v0, kv_cache_i8_flat_inline46_inline2265__ssa_v0, kv_scale_flat_inline50_inline2214__ssa_v0, __gm_pipe_buffer, attrs={"arg_directions": [pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout]})
        return score_arena_inline44_inline2267__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def indexer_score_leaf_wave_spmd(self, idx_positions_inline1323__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], score_arena_inline44_inline2267__ssa_v0: pl.Out[pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], qr_hadamard_i8_inline2177__rv_v2: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)], qr_hadamard_scale_dq_inline2234__ssa_v1: pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 65536)], weights_inline2244__ssa_v1: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 65536)], idx_block_table_flat_inline47_inline2186__ssa_v0: pl.Tensor[[idx_table_len_inline55_inline2193__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], kv_cache_i8_flat_inline46_inline2265__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)], kv_scale_flat_inline50_inline2214__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)], __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4)]]) -> pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32]:
        score_arena_inline44_inline2267__rv_v2: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)] = self.indexer_score_leaf_wave(idx_positions_inline1323__ssa_v0, score_arena_inline44_inline2267__ssa_v0, kv_seq_lens__ssa_v0, qr_hadamard_i8_inline2177__rv_v2, qr_hadamard_scale_dq_inline2234__ssa_v1, weights_inline2244__ssa_v1, idx_block_table_flat_inline47_inline2186__ssa_v0, kv_cache_i8_flat_inline46_inline2265__ssa_v0, kv_scale_flat_inline50_inline2214__ssa_v0, __gm_pipe_buffer, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
        return score_arena_inline44_inline2267__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_and_cache_write(bs_inline2140__ssa_v0: pl.Scalar[pl.INDEX], kv_final_inline2118__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 262144)], idx_kv_cache_flat_inline2161__ssa_v0: pl.Out[pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], kv_flat_inline2098__ssa_v0: pl.Out[pl.Tensor[[kv_dim_inline1261__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], idx_slots_inline1322__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], idx_kv_scale_flat_inline2116__ssa_v0: pl.Out[pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]):
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        wr_blk_inline2171__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        wr_b0_inline2157__ssa_v0: pl.Scalar[pl.INDEX] = wr_blk_inline2171__ssa_v0 * 16
        wr_blk_rows_inline2167__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2140__ssa_v0 - wr_b0_inline2157__ssa_v0, 16)
        t__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_5, pl.const(64, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(kv_final_inline2118__rv_v2, [wr_b0_inline2157__ssa_v0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec)
        t__tile_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_vec_8, pl.const(8256, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='rint')
        kv_blk_f32_inline2067__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_5, pl.const(64, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode='round')
        t__tile_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(8256, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(kv_blk_f32_inline2067__tile)
        tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_max(t__tile_2, tmp_tile)
        kv_amax_inline2091__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 16])
        t__tile_4: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_8, pl.const(8256, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0001)
        kv_amax_v1_inline2066__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_8, pl.const(8256, pl.INT64), 64), pl.Mem.Vec] = pl.tile.maximum(kv_amax_inline2091__tile, t__tile_4)
        t__tile_5: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16448, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=127.0)
        kv_scale_q_row_inline2112__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_8, pl.const(8256, pl.INT64), 64), pl.Mem.Vec] = pl.tile.div(t__tile_5, kv_amax_v1_inline2066__tile)
        t__tile_6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16448, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(kv_scale_q_row_inline2112__tile)
        kv_scale_dq_col_inline2065__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(16448, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_6, [16, 1])
        kv_scale_q_col_inline2086__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(8256, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(kv_scale_q_row_inline2112__tile, [16, 1])
        kv_scaled_inline2111__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_5, pl.const(64, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_blk_f32_inline2067__tile, kv_scale_q_col_inline2086__tile)
        kv_i32_inline2133__tile: pl.Tile[[16, 128], pl.INT32, pl.MemRef(mem_vec_5, pl.const(64, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(kv_scaled_inline2111__tile, target_type=pl.INT32, mode='rint')
        kv_half_inline2064__tile: pl.Tile[[16, 128], pl.FP16, pl.MemRef(mem_vec_5, pl.const(64, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(kv_i32_inline2133__tile, target_type=pl.FP16, mode='round')
        kv_i8_blk_inline2095__tile: pl.Tile[[16, 128], pl.INT8, pl.MemRef(mem_vec_5, pl.const(64, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(kv_half_inline2064__tile, target_type=pl.INT8, mode='trunc')
        for inner_inline2063__idx_v0, (idx_kv_cache_flat_inline2161__iter_v1, kv_flat_inline2098__iter_v1) in pl.range(wr_blk_rows_inline2167__ssa_v0, init_values=(idx_kv_cache_flat_inline2161__ssa_v0, kv_flat_inline2098__ssa_v0)):
            token_inline2123__ssa_v2: pl.Scalar[pl.INDEX] = wr_b0_inline2157__ssa_v0 + inner_inline2063__idx_v0
            cache_row_i64_inline2099__tile: pl.Scalar[pl.INT64] = pl.tensor.read(idx_slots_inline1322__ssa_v0, [token_inline2123__ssa_v2])
            if 0 <= cache_row_i64_inline2099__tile:
                cache_row_inline2146__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline2099__tile, pl.INDEX)
                t__tile_7: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(8256, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(kv_final_inline2118__rv_v2, [token_inline2123__ssa_v2, 0], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                kv_flat_inline2098__tile: pl.Tensor[[kv_dim_inline1261__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_7, [token_inline2123__ssa_v2, 0], kv_flat_inline2098__iter_v1)
                t__tile_8: pl.Tile[[1, 128], pl.INT8, pl.MemRef(mem_vec_5, pl.const(64, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(kv_i8_blk_inline2095__tile, [1, 128], [inner_inline2063__idx_v0, 0])
                idx_kv_cache_flat_inline2161__tile: pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_8, [cache_row_inline2146__ssa_v0, 0], idx_kv_cache_flat_inline2161__iter_v1)
                t__tile_9: pl.Scalar[pl.FP32] = pl.tile.read(kv_scale_dq_col_inline2065__tile, [inner_inline2063__idx_v0, 0])
                pl.tensor.write(idx_kv_scale_flat_inline2116__ssa_v0, [cache_row_inline2146__ssa_v0, 0], t__tile_9)
                idx_kv_cache_flat_inline2161__phi_v4, kv_flat_inline2098__phi_v4 = pl.yield_(idx_kv_cache_flat_inline2161__tile, kv_flat_inline2098__tile)
            else:
                idx_kv_cache_flat_inline2161__phi_v4, kv_flat_inline2098__phi_v4 = pl.yield_(idx_kv_cache_flat_inline2161__iter_v1, kv_flat_inline2098__iter_v1)
            idx_kv_cache_flat_inline2161__rv_v2, kv_flat_inline2098__rv_v2 = pl.yield_(idx_kv_cache_flat_inline2161__phi_v4, kv_flat_inline2098__phi_v4)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def kv_and_cache_write_spmd(self, bs_inline2140__ssa_v0: pl.Scalar[pl.INDEX], kv_final_inline2118__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 262144)], idx_kv_cache_flat_inline2161__ssa_v0: pl.Out[pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], kv_flat_inline2098__ssa_v0: pl.Out[pl.Tensor[[kv_dim_inline1261__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], idx_slots_inline1322__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], idx_kv_scale_flat_inline2116__ssa_v0: pl.Out[pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]):
        self.kv_and_cache_write(bs_inline2140__ssa_v0, kv_final_inline2118__rv_v2, idx_kv_cache_flat_inline2161__ssa_v0, kv_flat_inline2098__ssa_v0, idx_slots_inline1322__ssa_v0, idx_kv_scale_flat_inline2116__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.output_existing]})
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_hadamard(normed_kv_inline2164__ssa_v4: pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 131072)], kv_final_inline2118__ssa_v0: pl.Out[pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 262144)]], hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 32768)]) -> pl.Tensor[[512, 128], pl.FP32]:
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 4096)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 16384)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        had_blk_inline2088__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        had_b0_inline2069__ssa_v0: pl.Scalar[pl.INDEX] = had_blk_inline2088__ssa_v0 * 16
        kv_proj_tile_inline2072__tile: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 4096), pl.Mem.Mat] = pl.tile.load(normed_kv_inline2164__ssa_v4, [had_b0_inline2069__ssa_v0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Mat)
        for o0_inline2068__idx_v0, (kv_final_inline2118__iter_v1,) in pl.range(0, 128, 64, init_values=(kv_final_inline2118__ssa_v0,)):
            hadamard_tile_inline2134__tile: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_mat_4, pl.const(4096, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(hadamard_idx__ssa_v0, [0, o0_inline2068__idx_v0], [128, 64], [128, 64], target_memory=pl.Mem.Mat)
            kv_proj_tile_inline2072__tile_Left: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 4096), pl.Mem.Left] = pl.tile.move(kv_proj_tile_inline2072__tile, target_memory=pl.Mem.Left)
            hadamard_tile_inline2134__tile_Right: pl.Tile[[128, 64], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 16384), pl.Mem.Right] = pl.tile.move(hadamard_tile_inline2134__tile, target_memory=pl.Mem.Right)
            kv_hadamard_acc_inline2151__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul(kv_proj_tile_inline2072__tile_Left, hadamard_tile_inline2134__tile_Right)
            kv_final_inline2118__tile: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 262144)] = pl.tile.store(kv_hadamard_acc_inline2151__tile, [had_b0_inline2069__ssa_v0, o0_inline2068__idx_v0], kv_final_inline2118__iter_v1)
            kv_final_inline2118__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 262144)] = pl.yield_(kv_final_inline2118__tile)
        return kv_final_inline2118__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def kv_hadamard_spmd(self, normed_kv_inline2164__ssa_v4: pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 131072)], kv_final_inline2118__ssa_v0: pl.Out[pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 262144)]], hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 32768)]) -> pl.Tensor[[512, 128], pl.FP32]:
        kv_final_inline2118__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 262144)] = self.kv_hadamard(normed_kv_inline2164__ssa_v4, kv_final_inline2118__ssa_v0, hadamard_idx__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input]})
        return kv_final_inline2118__ssa_v0
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_proj_matmul(kv_fp32_inline1920__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline1928__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1954__idx_v0: pl.Scalar[pl.INDEX], x_view_inline1914__ssa_v0: pl.Tensor[[t_dim_inline1923__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)]) -> pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32]:
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_9: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_10: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_11: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_12: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_21: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_left_23: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        kbg_inline1940__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        kv_col0_inline1937__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1940__ssa_v0 // 8 * 128
        kv_k_base_inline1962__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1940__ssa_v0 // 4 % 2 * 2048
        kv_m_group_inline1922__ssa_v0: pl.Scalar[pl.INDEX] = kbg_inline1940__ssa_v0 % 4
        for t0_inline1943__idx_v0, (kv_fp32_inline1920__iter_v6,) in pl.range(kv_m_group_inline1922__ssa_v0 * 16, t_matmul_inline1930__ssa_v0, 64, init_values=(kv_fp32_inline1920__rv_v2,)):
            kv_acc_inline1938__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline1932__idx_v0, (kv_acc_inline1938__iter_v1,) in pl.range(0, 8, 2, init_values=(kv_acc_inline1938__tile,)):
                d0_inline1913__ssa_v0: pl.Scalar[pl.INDEX] = kv_k_base_inline1962__ssa_v0 + db_inline1932__idx_v0 * 256
                kv_rows_inline1911__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1928__ssa_v0 - t0_inline1943__idx_v0, 16)
                x_t0_inline1918__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1954__idx_v0 + t0_inline1943__idx_v0
                d0_inline1913__ssa_v0_1: pl.Scalar[pl.INDEX] = kv_k_base_inline1962__ssa_v0 + (db_inline1932__idx_v0 * 256 + 256)
                kv_rows_inline1911__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1928__ssa_v0 - t0_inline1943__idx_v0, 16)
                x_t0_inline1918__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline1954__idx_v0 + t0_inline1943__idx_v0
                kv_x_chunk_bf16_inline1910__tile: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 256])] = pl.tile.load(x_view_inline1914__ssa_v0, [x_t0_inline1918__ssa_v0, d0_inline1913__ssa_v0], [16, 256], [kv_rows_inline1911__ssa_v0, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline1933__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(8192, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wkv__ssa_v0, [d0_inline1913__ssa_v0, kv_col0_inline1937__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat)
                kv_x_chunk_bf16_inline1910__tile_1: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(73728, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0_1, 256])] = pl.tile.load(x_view_inline1914__ssa_v0, [x_t0_inline1918__ssa_v0_1, d0_inline1913__ssa_v0_1], [16, 256], [kv_rows_inline1911__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                wkv_chunk_inline1933__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wkv__ssa_v0, [d0_inline1913__ssa_v0_1, kv_col0_inline1937__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat)
                if db_inline1932__idx_v0 == 0:
                    kv_acc_inline1938__tile_l0_init_storage: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    kv_acc_inline1938__tile_l0_init: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(kv_acc_inline1938__tile_l0_init_storage, kv_rows_inline1911__ssa_v0, 128)
                    kv_acc_inline1938__tile_l0_a: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_9, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(kv_x_chunk_bf16_inline1910__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                    kv_acc_inline1938__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_chunk_inline1933__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    kv_acc_inline1938__tile_l0_a_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(kv_x_chunk_bf16_inline1910__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                    kv_acc_inline1938__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_chunk_inline1933__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                    kv_acc_inline1938__tile_l0_c_acc: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(kv_acc_inline1938__tile_l0_init, kv_acc_inline1938__tile_l0_a, kv_acc_inline1938__tile_l0_b, True)
                    kv_acc_inline1938__tile_l0_c_acc_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(kv_acc_inline1938__tile_l0_c_acc, kv_acc_inline1938__tile_l0_a_1, kv_acc_inline1938__tile_l0_b_1, False)
                    kv_acc_inline1938__phi_v5: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.yield_(kv_acc_inline1938__tile_l0_c_acc_1)
                else:
                    kv_acc_inline1938__tile_l0_a_2: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_9, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(kv_x_chunk_bf16_inline1910__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                    kv_acc_inline1938__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_chunk_inline1933__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    kv_acc_inline1938__tile_l0_a_3: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(kv_x_chunk_bf16_inline1910__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                    kv_acc_inline1938__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_chunk_inline1933__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                    kv_acc_inline1938__tile_l0_c_acc_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline1938__iter_v1, kv_acc_inline1938__tile_l0_a_2, kv_acc_inline1938__tile_l0_b_2)
                    kv_acc_inline1938__tile_l0_c_acc_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline1938__tile_l0_c_acc_2, kv_acc_inline1938__tile_l0_a_3, kv_acc_inline1938__tile_l0_b_3)
                    kv_acc_inline1938__phi_v5: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.yield_(kv_acc_inline1938__tile_l0_c_acc_3)
                kv_acc_inline1938__tile_l0_a_4: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_21, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(kv_x_chunk_bf16_inline1910__tile_1, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1938__tile_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_chunk_inline1933__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                kv_acc_inline1938__tile_l0_a_5: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_23, pl.const(8192, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[kv_rows_inline1911__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(kv_x_chunk_bf16_inline1910__tile_1, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                kv_acc_inline1938__tile_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_chunk_inline1933__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                kv_acc_inline1938__tile_l0_c_acc_4: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline1938__phi_v5, kv_acc_inline1938__tile_l0_a_4, kv_acc_inline1938__tile_l0_b_4)
                kv_acc_inline1938__tile_l0_c_acc_5: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline1938__tile_l0_c_acc_4, kv_acc_inline1938__tile_l0_a_5, kv_acc_inline1938__tile_l0_b_5)
                kv_acc_inline1938__rv_v2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(kv_acc_inline1938__tile_l0_c_acc_5)
            kv_fp32_inline1920__tile: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_acc_inline1938__rv_v2, [t0_inline1943__idx_v0, kv_col0_inline1937__ssa_v0], kv_fp32_inline1920__iter_v6, atomic=pl.AtomicType.Add)
            kv_fp32_inline1920__rv_v7: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1920__tile)
        return kv_fp32_inline1920__rv_v2
    @pl.function(type=pl.FunctionType.Spmd)
    def kv_proj_matmul_spmd(self, kv_fp32_inline1920__rv_v2: pl.InOut[pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline1928__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1954__idx_v0: pl.Scalar[pl.INDEX], x_view_inline1914__ssa_v0: pl.Tensor[[t_dim_inline1923__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)]) -> pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32]:
        kv_fp32_inline1920__rv_v7: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.kv_proj_matmul(kv_fp32_inline1920__rv_v2, t_matmul_inline1930__ssa_v0, tile_rows_inline1928__ssa_v0, tile_base_inline1954__idx_v0, x_view_inline1914__ssa_v0, wkv__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]})
        return kv_fp32_inline1920__rv_v2
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_proj_seed(kv_fp32_inline1920__ssa_v0: pl.Out[pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32]:
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        for kts0_inline1934__idx_v0, (kv_fp32_inline1920__iter_v1,) in pl.range(0, t_matmul_inline1930__ssa_v0, 16, init_values=(kv_fp32_inline1920__ssa_v0,)):
            for kvseed0_inline1936__idx_v0, (kv_fp32_inline1920__iter_v3,) in pl.range(0, 512, 128, init_values=(kv_fp32_inline1920__iter_v1,)):
                kv_seed_inline1926__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                kv_fp32_inline1920__tile: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_seed_inline1926__tile, [kts0_inline1934__idx_v0, kvseed0_inline1936__idx_v0], kv_fp32_inline1920__iter_v3)
                kv_fp32_inline1920__rv_v4: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1920__tile)
            kv_fp32_inline1920__rv_v2: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_fp32_inline1920__rv_v4)
        return kv_fp32_inline1920__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_rms_norm_rope(tile_rows_inline1928__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1954__idx_v0: pl.Scalar[pl.INDEX], kv_fp32_inline1920__rv_v7: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], kv_view_inline1909__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)], kv_cos_il_inline1258__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], kv_sin_signed_inline1301__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)], kv_swap_idx_inline1305__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16]:
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_50: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_115: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_116: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        tg_idx_inline1916__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline1915__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1916__ssa_v0 * 16
        valid_rows_inline1931__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1928__ssa_v0 - tg_inline1915__ssa_v0, 16)
        out_tg_inline1924__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1954__idx_v0 + tg_inline1915__ssa_v0
        if valid_rows_inline1931__ssa_v0 == 16:
            kv_sq_sum_inline1961__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_inline1944__idx_v0, (kv_sq_sum_inline1961__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_sum_inline1961__tile,)):
                kv_chunk_inline1941__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, kv_sq_col0_inline1944__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
                kv_chunk_inline1941__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, kv_sq_col0_inline1944__idx_v0 + 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
                kv_sq_inline1968__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(kv_chunk_inline1941__tile, kv_chunk_inline1941__tile)
                tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_inline1968__tile, tmp_tile)
                kv_row_sum_inline1946__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 16])
                kv_sq_sum_inline1961__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(kv_sq_sum_inline1961__iter_v1, kv_row_sum_inline1946__tile)
                kv_sq_inline1968__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(kv_chunk_inline1941__tile_1, kv_chunk_inline1941__tile_1)
                tmp_tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_116, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_sq_inline1968__tile_1, tmp_tile_1)
                kv_row_sum_inline1946__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_116, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 16])
                kv_sq_sum_inline1961__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(kv_sq_sum_inline1961__tile_1, kv_row_sum_inline1946__tile_1)
                kv_sq_sum_inline1961__rv_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 64), pl.Mem.Vec] = pl.yield_(kv_sq_sum_inline1961__tile_2)
            t__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_inline1961__rv_v2, 0.001953125)
            t__tile_3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tile_2, 9.9999999999999995e-07)
            rsqrt_tmp: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 64), pl.Mem.Vec] = pl.tile.create([1, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_inv_rms_inline1942__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_116, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.rsqrt(t__tile_3, rsqrt_tmp)
            kv_inv_rms_t_inline1949__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_116, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_inline1942__tile, [16, 1])
            for n0_inline1950__idx_v0, (kv_view_inline1909__iter_v1,) in pl.range(0, 384, 128, init_values=(kv_view_inline1909__ssa_v0,)):
                kv_chunk_inline1941__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, n0_inline1950__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
                t__tile_4: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [n0_inline1950__idx_v0], [64], [64], target_memory=pl.Mem.Vec)
                kv_chunk_inline1941__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, n0_inline1950__idx_v0 + 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
                t__tile_5: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [n0_inline1950__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec)
                gamma_kv_cast_inline1948__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_4, target_type=pl.FP32, mode='round')
                gamma_kv_chunk_inline1951__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1948__tile
                t__tile_6: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_chunk_inline1941__tile_2, kv_inv_rms_t_inline1949__tile)
                kv_normed_inline1925__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_6, gamma_kv_chunk_inline1951__tile)
                kv_normed_bf16_inline1908__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(kv_normed_inline1925__tile, target_type=pl.BF16, mode='rint')
                kv_view_inline1909__tile: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_normed_bf16_inline1908__tile, [out_tg_inline1924__ssa_v0, n0_inline1950__idx_v0], kv_view_inline1909__iter_v1)
                gamma_kv_cast_inline1948__tile_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.FP32, mode='round')
                gamma_kv_chunk_inline1951__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1948__tile_1
                t__tile_7: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_chunk_inline1941__tile_3, kv_inv_rms_t_inline1949__tile)
                kv_normed_inline1925__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_7, gamma_kv_chunk_inline1951__tile_1)
                kv_normed_bf16_inline1908__tile_1: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(kv_normed_inline1925__tile_1, target_type=pl.BF16, mode='rint')
                kv_view_inline1909__tile_1: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_normed_bf16_inline1908__tile_1, [out_tg_inline1924__ssa_v0, n0_inline1950__idx_v0 + 64], kv_view_inline1909__tile)
                kv_view_inline1909__rv_v2_main: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)] = pl.yield_(kv_view_inline1909__tile_1)
            kv_chunk_inline1941__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, 384], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            t__tile_8: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec)
            gamma_kv_cast_inline1948__tile_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_8, target_type=pl.FP32, mode='round')
            gamma_kv_chunk_inline1951__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_inline1948__tile_2
            t__tile_9: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_chunk_inline1941__tile_4, kv_inv_rms_t_inline1949__tile)
            kv_normed_inline1925__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_9, gamma_kv_chunk_inline1951__tile_2)
            kv_normed_bf16_inline1908__tile_2: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(kv_normed_inline1925__tile_2, target_type=pl.BF16, mode='rint')
            kv_view_inline1909__tile_2: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_normed_bf16_inline1908__tile_2, [out_tg_inline1924__ssa_v0, 384], kv_view_inline1909__rv_v2_main)
            kv_view_inline1909__rv_v2: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)] = kv_view_inline1909__tile_2
            t__tile_10: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec)
            gamma_rope_cast_inline1955__tile: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.FP32, mode='round')
            gamma_rope_inline1957__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_inline1955__tile
            kv_rope_chunk_inline1959__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, 448], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            t__tile_11: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_rope_chunk_inline1959__tile, kv_inv_rms_t_inline1949__tile)
            kv_rope_norm_chunk_inline1945__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_11, gamma_rope_inline1957__tile)
            kv_cos_il_full_inline1963__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_cos_il_inline1258__ssa_v0, [out_tg_inline1924__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            kv_sin_signed_full_inline1956__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_sin_signed_inline1301__ssa_v0, [out_tg_inline1924__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            kv_swap_idx_full_inline1953__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(kv_swap_idx_inline1305__ssa_v0, [out_tg_inline1924__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(kv_rope_norm_chunk_inline1945__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(kv_swap_idx_full_inline1953__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_116, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            kv_swapped_full_inline1912__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
            t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(kv_rope_norm_chunk_inline1945__tile, kv_cos_il_full_inline1963__tile)
            t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(kv_swapped_full_inline1912__tile, kv_sin_signed_full_inline1956__tile)
            kv_rope_rot_full_inline1921__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_12, t__tile_13)
            kv_rope_i16_full_inline1965__tile: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(kv_rope_rot_full_inline1921__tile, target_type=pl.BF16, mode='rint')
            kv_view_inline1909__tile_3: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_rope_i16_full_inline1965__tile, [out_tg_inline1924__ssa_v0, 448], kv_view_inline1909__rv_v2)
        else:
            kv_reduce_tmp_inline1967__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_sq_sum_tail_inline1970__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_tail_inline1969__idx_v0, (kv_sq_sum_tail_inline1970__iter_v1,) in pl.range(0, 512, 128, init_values=(kv_sq_sum_tail_inline1970__ssa_v0,)):
                kv_chunk_tail_inline1939__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, kv_sq_col0_tail_inline1969__idx_v0], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                kv_chunk_tail_inline1939__ssa_v0_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, kv_sq_col0_tail_inline1969__idx_v0 + 64], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                kv_sq_tail_inline1971__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.mul(kv_chunk_tail_inline1939__ssa_v0, kv_chunk_tail_inline1939__ssa_v0)
                t__tmp_v165: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 1])] = pl.tile.row_sum(kv_sq_tail_inline1971__ssa_v0, kv_reduce_tmp_inline1967__ssa_v0)
                kv_row_sum_tail_inline1973__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1931__ssa_v0])] = pl.tile.reshape(t__tmp_v165, [1, 16])
                kv_sq_sum_tail_inline1970__ssa_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(kv_sq_sum_tail_inline1970__iter_v1, kv_row_sum_tail_inline1973__ssa_v0)
                kv_sq_tail_inline1971__ssa_v0_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.mul(kv_chunk_tail_inline1939__ssa_v0_1, kv_chunk_tail_inline1939__ssa_v0_1)
                t__tmp_v165_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 1])] = pl.tile.row_sum(kv_sq_tail_inline1971__ssa_v0_1, kv_reduce_tmp_inline1967__ssa_v0)
                kv_row_sum_tail_inline1973__ssa_v0_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline1931__ssa_v0])] = pl.tile.reshape(t__tmp_v165_1, [1, 16])
                kv_sq_sum_tail_inline1970__ssa_v3_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(kv_sq_sum_tail_inline1970__ssa_v3, kv_row_sum_tail_inline1973__ssa_v0_1)
                kv_sq_sum_tail_inline1970__rv_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.yield_(kv_sq_sum_tail_inline1970__ssa_v3_1)
            t__tmp_v166: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(kv_sq_sum_tail_inline1970__rv_v2, 0.001953125)
            t__tmp_v167: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tmp_v166, 9.9999999999999995e-07)
            t__tmp_v168: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(t__tmp_v167)
            kv_inv_rms_tail_inline1958__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_116, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(t__tmp_v168)
            kv_inv_rms_t_tail_inline1947__ssa_v0: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_116, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(kv_inv_rms_tail_inline1958__ssa_v0, [16, 1])
            for n0_tail_inline1964__idx_v0 in pl.range(0, 384, 128):
                kv_chunk_tail_inline1939__ssa_v1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, n0_tail_inline1964__idx_v0], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline1960__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [n0_tail_inline1964__idx_v0], [64], [64], target_memory=pl.Mem.Vec)
                kv_chunk_tail_inline1939__ssa_v1_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, n0_tail_inline1964__idx_v0 + 64], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
                gamma_kv_input_tail_inline1960__ssa_v0_1: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [n0_tail_inline1964__idx_v0 + 64], [64], [64], target_memory=pl.Mem.Vec)
                gamma_kv_cast_tail_inline1905__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(gamma_kv_input_tail_inline1960__ssa_v0, target_type=pl.FP32, mode='round')
                gamma_kv_chunk_tail_inline1919__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1905__ssa_v0
                t__tmp_v169: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.row_expand_mul(kv_chunk_tail_inline1939__ssa_v1, kv_inv_rms_t_tail_inline1947__ssa_v0)
                kv_normed_tail_inline1904__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.col_expand_mul(t__tmp_v169, gamma_kv_chunk_tail_inline1919__ssa_v0)
                kv_normed_bf16_tail_inline1903__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.cast(kv_normed_tail_inline1904__ssa_v0, target_type=pl.BF16, mode='rint')
                kv_normed_valid_inline1902__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1903__ssa_v0, valid_rows_inline1931__ssa_v0, 64)
                kv_view_inline1909__store: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_normed_valid_inline1902__ssa_v0, [out_tg_inline1924__ssa_v0, n0_tail_inline1964__idx_v0], kv_view_inline1909__ssa_v0)
                gamma_kv_cast_tail_inline1905__ssa_v0_1: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(gamma_kv_input_tail_inline1960__ssa_v0_1, target_type=pl.FP32, mode='round')
                gamma_kv_chunk_tail_inline1919__ssa_v0_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1905__ssa_v0_1
                t__tmp_v169_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.row_expand_mul(kv_chunk_tail_inline1939__ssa_v1_1, kv_inv_rms_t_tail_inline1947__ssa_v0)
                kv_normed_tail_inline1904__ssa_v0_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.col_expand_mul(t__tmp_v169_1, gamma_kv_chunk_tail_inline1919__ssa_v0_1)
                kv_normed_bf16_tail_inline1903__ssa_v0_1: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.cast(kv_normed_tail_inline1904__ssa_v0_1, target_type=pl.BF16, mode='rint')
                kv_normed_valid_inline1902__ssa_v0_1: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1903__ssa_v0_1, valid_rows_inline1931__ssa_v0, 64)
                kv_view_inline1909__store_1: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_normed_valid_inline1902__ssa_v0_1, [out_tg_inline1924__ssa_v0, n0_tail_inline1964__idx_v0 + 64], kv_view_inline1909__ssa_v0)
            kv_chunk_tail_inline1939__ssa_v1_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, 384], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
            gamma_kv_input_tail_inline1960__ssa_v0_2: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [384], [64], [64], target_memory=pl.Mem.Vec)
            gamma_kv_cast_tail_inline1905__ssa_v0_2: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(gamma_kv_input_tail_inline1960__ssa_v0_2, target_type=pl.FP32, mode='round')
            gamma_kv_chunk_tail_inline1919__ssa_v0_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = gamma_kv_cast_tail_inline1905__ssa_v0_2
            t__tmp_v169_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.row_expand_mul(kv_chunk_tail_inline1939__ssa_v1_2, kv_inv_rms_t_tail_inline1947__ssa_v0)
            kv_normed_tail_inline1904__ssa_v0_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.col_expand_mul(t__tmp_v169_2, gamma_kv_chunk_tail_inline1919__ssa_v0_2)
            kv_normed_bf16_tail_inline1903__ssa_v0_2: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.cast(kv_normed_tail_inline1904__ssa_v0_2, target_type=pl.BF16, mode='rint')
            kv_normed_valid_inline1902__ssa_v0_2: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.set_validshape(kv_normed_bf16_tail_inline1903__ssa_v0_2, valid_rows_inline1931__ssa_v0, 64)
            kv_view_inline1909__store_2: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_normed_valid_inline1902__ssa_v0_2, [out_tg_inline1924__ssa_v0, 384], kv_view_inline1909__ssa_v0)
            gamma_rope_input_tail_inline1901__ssa_v0: pl.Tile[[64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(gamma_ckv__ssa_v0, [448], [64], [64], target_memory=pl.Mem.Vec)
            gamma_rope_cast_tail_inline1900__ssa_v0: pl.Tile[[64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(gamma_rope_input_tail_inline1901__ssa_v0, target_type=pl.FP32, mode='round')
            gamma_rope_tail_inline1899__ssa_v0: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 256), pl.Mem.Vec] = gamma_rope_cast_tail_inline1900__ssa_v0
            kv_rope_chunk_tail_inline1907__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_fp32_inline1920__rv_v7, [tg_inline1915__ssa_v0, 448], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v170: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.row_expand_mul(kv_rope_chunk_tail_inline1907__ssa_v0, kv_inv_rms_t_tail_inline1947__ssa_v0)
            kv_rope_norm_tail_inline1898__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.col_expand_mul(t__tmp_v170, gamma_rope_tail_inline1899__ssa_v0)
            kv_cos_il_tail_inline1929__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_cos_il_inline1258__ssa_v0, [out_tg_inline1924__ssa_v0, 0], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
            kv_sin_signed_tail_inline1917__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(32768, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.load(kv_sin_signed_inline1301__ssa_v0, [out_tg_inline1924__ssa_v0, 0], [16, 64], [valid_rows_inline1931__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v171: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
            t__tmp_v172: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            t__tmp_v173: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v172, target_type=pl.FP32, mode='round')
            kv_col_inline1897__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v171, t__tmp_v173)
            t__tmp_v174: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(kv_col_inline1897__ssa_v0, 0.5)
            t__tmp_v175: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tmp_v174, target_type=pl.INT32, mode='trunc')
            kv_dup_f_inline1952__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tmp_v175, target_type=pl.FP32, mode='round')
            t__tmp_v176: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(kv_dup_f_inline1952__ssa_v0, 2.0)
            kv_lane_inline1896__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(kv_col_inline1897__ssa_v0, t__tmp_v176)
            t__tmp_v177: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(kv_col_inline1897__ssa_v0, 1.0)
            t__tmp_v178: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(kv_lane_inline1896__ssa_v0, 2.0)
            kv_swap_f_inline1895__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tmp_v177, t__tmp_v178)
            t__tmp_v179: pl.Tile[[1, 16], pl.INT32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 64), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 16], dtype=pl.INT32, descending=False)
            t__tmp_v180: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 64), pl.Mem.Vec] = pl.tile.cast(t__tmp_v179, target_type=pl.FP32, mode='round')
            kv_row_seed_inline1966__ssa_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(t__tmp_v180, 64.0)
            t__tmp_v181: pl.Tile[[64, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([64, 16], dtype=pl.FP32, value=1.0)
            kv_row_grid_inline1893__ssa_v0: pl.Tile[[64, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v181, kv_row_seed_inline1966__ssa_v0)
            transpose_tmp: pl.Tile[[64, 16], pl.FP32, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([64, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            kv_row_offset_inline1892__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_116, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.transpose(kv_row_grid_inline1893__ssa_v0, 0, 1, transpose_tmp)
            t__tmp_v182: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(kv_swap_f_inline1895__ssa_v0, kv_row_offset_inline1892__ssa_v0)
            kv_swap_idx_tail_inline1891__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(16384, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tmp_v182, target_type=pl.INT32, mode='round')
            kv_gather_tmp_inline1890__ssa_v0: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(28672, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            kv_swapped_tail_inline1894__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_115, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.gather(kv_rope_norm_tail_inline1898__ssa_v0, kv_swap_idx_tail_inline1891__ssa_v0, kv_gather_tmp_inline1890__ssa_v0)
            t__tmp_v183: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.mul(kv_rope_norm_tail_inline1898__ssa_v0, kv_cos_il_tail_inline1929__ssa_v0)
            t__tmp_v184: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(20480, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(kv_swapped_tail_inline1894__ssa_v0, kv_sin_signed_tail_inline1917__ssa_v0)
            kv_rope_rot_tail_inline1927__ssa_v0: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.add(t__tmp_v183, t__tmp_v184)
            kv_rope_i16_tail_inline1889__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.cast(kv_rope_rot_tail_inline1927__ssa_v0, target_type=pl.BF16, mode='rint')
            kv_rope_valid_inline1906__ssa_v0: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1931__ssa_v0, 64])] = pl.tile.set_validshape(kv_rope_i16_tail_inline1889__ssa_v0, valid_rows_inline1931__ssa_v0, 64)
            kv_view_inline1909__store_v0: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_rope_valid_inline1906__ssa_v0, [out_tg_inline1924__ssa_v0, 448], kv_view_inline1909__ssa_v0)
        return kv_view_inline1909__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def kv_rms_norm_rope_spmd(self, tile_rows_inline1928__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1954__idx_v0: pl.Scalar[pl.INDEX], kv_fp32_inline1920__rv_v7: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], kv_view_inline1909__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1024)], kv_cos_il_inline1258__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], kv_sin_signed_inline1301__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)], kv_swap_idx_inline1305__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]):
        kv_view_inline1909__ssa_v1: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = self.kv_rms_norm_rope(tile_rows_inline1928__ssa_v0, tile_base_inline1954__idx_v0, kv_fp32_inline1920__rv_v7, kv_view_inline1909__ssa_v0, gamma_ckv__ssa_v0, kv_cos_il_inline1258__ssa_v0, kv_sin_signed_inline1301__ssa_v0, kv_swap_idx_inline1305__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj(bs_inline2038__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline2021__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 8388608)], cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)], cmp4_kv_proj_pad_inline2031__ssa_v0: pl.Out[pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)]], cmp4_score_proj_pad_inline2019__ssa_v0: pl.Out[pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)]]) -> tuple[pl.Tensor[[512, 1024], pl.FP32], pl.Tensor[[512, 1024], pl.FP32]]:
        mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_acc_6: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_8: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_12: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_14: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_15: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_16: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_17: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_38: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_left_40: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        idx_inline2012__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        global_row0_inline2011__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2012__ssa_v0 // 16 * 16
        o0_inline2017__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2012__ssa_v0 % 16 * 64
        kv_acc_inline2010__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        score_acc_inline2014__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        for kb_inline2001__idx_v0, (kv_acc_inline2010__iter_v1, score_acc_inline2014__iter_v1) in pl.range(0, 8, 2, init_values=(kv_acc_inline2010__tile, score_acc_inline2014__tile)):
            k0_inline2035__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2001__idx_v0 * 512
            x_rows_inline2000__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2038__ssa_v0 - global_row0_inline2011__ssa_v0, 16)
            k0_inline2035__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline2001__idx_v0 * 512 + 512
            x_rows_inline2000__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(bs_inline2038__ssa_v0 - global_row0_inline2011__ssa_v0, 16)
            x_tile_inline2033__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_7, pl.const(147456, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 512])] = pl.tile.load(x_flat_inline2021__ssa_v0, [global_row0_inline2011__ssa_v0, k0_inline2035__ssa_v0], [16, 512], [x_rows_inline2000__ssa_v0, 512], target_memory=pl.Mem.Mat)
            wkv_tile_inline2007__tile: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_8, pl.const(163840, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(cmp_wkv__ssa_v0, [o0_inline2017__ssa_v0, k0_inline2035__ssa_v0], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
            wgate_tile_inline1998__tile: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_9, pl.const(229376, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(cmp_wgate__ssa_v0, [o0_inline2017__ssa_v0, k0_inline2035__ssa_v0], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
            x_tile_inline2033__tile_1: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0_1, 512])] = pl.tile.load(x_flat_inline2021__ssa_v0, [global_row0_inline2011__ssa_v0, k0_inline2035__ssa_v0_1], [16, 512], [x_rows_inline2000__ssa_v0_1, 512], target_memory=pl.Mem.Mat)
            wkv_tile_inline2007__tile_1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(cmp_wkv__ssa_v0, [o0_inline2017__ssa_v0, k0_inline2035__ssa_v0_1], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
            wgate_tile_inline1998__tile_1: pl.Tile[[64, 512], pl.BF16, pl.MemRef(mem_mat_12, pl.const(81920, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(cmp_wgate__ssa_v0, [o0_inline2017__ssa_v0, k0_inline2035__ssa_v0_1], [64, 512], [64, 512], target_memory=pl.Mem.Mat)
            if k0_inline2035__ssa_v0 == 0:
                wkv_tile_inline2007__tile_t: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_8, pl.const(163840, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wkv_tile_inline2007__tile)
                kv_acc_inline2010__tile_l0_init_storage: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                kv_acc_inline2010__tile_l0_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(kv_acc_inline2010__tile_l0_init_storage, x_rows_inline2000__ssa_v0, 64)
                kv_acc_inline2010__tile_l0_a: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_14, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                kv_acc_inline2010__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_15, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_tile_inline2007__tile_t, 0, 0, [256, 64], target_memory=pl.Mem.Right)
                kv_acc_inline2010__tile_l0_a_1: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_16, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                kv_acc_inline2010__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_tile_inline2007__tile_t, 256, 0, [256, 64], target_memory=pl.Mem.Right)
                kv_acc_inline2010__tile_l0_c_acc: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(kv_acc_inline2010__tile_l0_init, kv_acc_inline2010__tile_l0_a, kv_acc_inline2010__tile_l0_b, True)
                kv_acc_inline2010__tile_l0_c_acc_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(kv_acc_inline2010__tile_l0_c_acc, kv_acc_inline2010__tile_l0_a_1, kv_acc_inline2010__tile_l0_b_1, False)
                wgate_tile_inline1998__tile_t: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_9, pl.const(229376, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wgate_tile_inline1998__tile)
                score_acc_inline2014__tile_l0_init_storage: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                score_acc_inline2014__tile_l0_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(score_acc_inline2014__tile_l0_init_storage, x_rows_inline2000__ssa_v0, 64)
                score_acc_inline2014__tile_l0_a: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_14, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                score_acc_inline2014__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_15, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wgate_tile_inline1998__tile_t, 0, 0, [256, 64], target_memory=pl.Mem.Right)
                score_acc_inline2014__tile_l0_a_1: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_16, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                score_acc_inline2014__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wgate_tile_inline1998__tile_t, 256, 0, [256, 64], target_memory=pl.Mem.Right)
                score_acc_inline2014__tile_l0_c_acc: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(score_acc_inline2014__tile_l0_init, score_acc_inline2014__tile_l0_a, score_acc_inline2014__tile_l0_b, True)
                score_acc_inline2014__tile_l0_c_acc_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(score_acc_inline2014__tile_l0_c_acc, score_acc_inline2014__tile_l0_a_1, score_acc_inline2014__tile_l0_b_1, False)
                kv_acc_inline2010__phi_v5, score_acc_inline2014__phi_v5 = pl.yield_(kv_acc_inline2010__tile_l0_c_acc_1, score_acc_inline2014__tile_l0_c_acc_1)
            else:
                wkv_tile_inline2007__tile_t_1: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_8, pl.const(163840, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wkv_tile_inline2007__tile)
                kv_acc_inline2010__tile_l0_a_2: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_14, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                kv_acc_inline2010__tile_l0_b_2: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_15, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_tile_inline2007__tile_t_1, 0, 0, [256, 64], target_memory=pl.Mem.Right)
                kv_acc_inline2010__tile_l0_a_3: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_16, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                kv_acc_inline2010__tile_l0_b_3: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_tile_inline2007__tile_t_1, 256, 0, [256, 64], target_memory=pl.Mem.Right)
                kv_acc_inline2010__tile_l0_c_acc_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline2010__iter_v1, kv_acc_inline2010__tile_l0_a_2, kv_acc_inline2010__tile_l0_b_2)
                kv_acc_inline2010__tile_l0_c_acc_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline2010__tile_l0_c_acc_2, kv_acc_inline2010__tile_l0_a_3, kv_acc_inline2010__tile_l0_b_3)
                wgate_tile_inline1998__tile_t_1: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_9, pl.const(229376, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wgate_tile_inline1998__tile)
                score_acc_inline2014__tile_l0_a_2: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_14, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                score_acc_inline2014__tile_l0_b_2: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_15, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wgate_tile_inline1998__tile_t_1, 0, 0, [256, 64], target_memory=pl.Mem.Right)
                score_acc_inline2014__tile_l0_a_3: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_16, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                score_acc_inline2014__tile_l0_b_3: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wgate_tile_inline1998__tile_t_1, 256, 0, [256, 64], target_memory=pl.Mem.Right)
                score_acc_inline2014__tile_l0_c_acc_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(score_acc_inline2014__iter_v1, score_acc_inline2014__tile_l0_a_2, score_acc_inline2014__tile_l0_b_2)
                score_acc_inline2014__tile_l0_c_acc_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(score_acc_inline2014__tile_l0_c_acc_2, score_acc_inline2014__tile_l0_a_3, score_acc_inline2014__tile_l0_b_3)
                kv_acc_inline2010__phi_v5, score_acc_inline2014__phi_v5 = pl.yield_(kv_acc_inline2010__tile_l0_c_acc_3, score_acc_inline2014__tile_l0_c_acc_3)
            wkv_tile_inline2007__tile_t_2: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wkv_tile_inline2007__tile_1)
            kv_acc_inline2010__tile_l0_a_4: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_38, pl.const(16384, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0_1, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile_1, 0, 0, [16, 256], target_memory=pl.Mem.Left)
            kv_acc_inline2010__tile_l0_b_4: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_15, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_tile_inline2007__tile_t_2, 0, 0, [256, 64], target_memory=pl.Mem.Right)
            kv_acc_inline2010__tile_l0_a_5: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_40, pl.const(24576, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0_1, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile_1, 0, 256, [16, 256], target_memory=pl.Mem.Left)
            kv_acc_inline2010__tile_l0_b_5: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wkv_tile_inline2007__tile_t_2, 256, 0, [256, 64], target_memory=pl.Mem.Right)
            kv_acc_inline2010__tile_l0_c_acc_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline2010__phi_v5, kv_acc_inline2010__tile_l0_a_4, kv_acc_inline2010__tile_l0_b_4)
            kv_acc_inline2010__tile_l0_c_acc_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline2010__tile_l0_c_acc_4, kv_acc_inline2010__tile_l0_a_5, kv_acc_inline2010__tile_l0_b_5)
            wgate_tile_inline1998__tile_t_2: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_12, pl.const(81920, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wgate_tile_inline1998__tile_1)
            score_acc_inline2014__tile_l0_a_4: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_38, pl.const(16384, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0_1, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile_1, 0, 0, [16, 256], target_memory=pl.Mem.Left)
            score_acc_inline2014__tile_l0_b_4: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_15, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wgate_tile_inline1998__tile_t_2, 0, 0, [256, 64], target_memory=pl.Mem.Right)
            score_acc_inline2014__tile_l0_a_5: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_40, pl.const(24576, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2000__ssa_v0_1, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2033__tile_1, 0, 256, [16, 256], target_memory=pl.Mem.Left)
            score_acc_inline2014__tile_l0_b_5: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_17, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wgate_tile_inline1998__tile_t_2, 256, 0, [256, 64], target_memory=pl.Mem.Right)
            score_acc_inline2014__tile_l0_c_acc_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(score_acc_inline2014__phi_v5, score_acc_inline2014__tile_l0_a_4, score_acc_inline2014__tile_l0_b_4)
            score_acc_inline2014__tile_l0_c_acc_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_6, pl.const(4096, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(score_acc_inline2014__tile_l0_c_acc_4, score_acc_inline2014__tile_l0_a_5, score_acc_inline2014__tile_l0_b_5)
            kv_acc_inline2010__rv_v2, score_acc_inline2014__rv_v2 = pl.yield_(kv_acc_inline2010__tile_l0_c_acc_5, score_acc_inline2014__tile_l0_c_acc_5)
        cmp4_kv_proj_pad_inline2031__tile: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)] = pl.tile.store(kv_acc_inline2010__rv_v2, [global_row0_inline2011__ssa_v0, o0_inline2017__ssa_v0], cmp4_kv_proj_pad_inline2031__ssa_v0)
        cmp4_score_proj_pad_inline2019__tile: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)] = pl.tile.store(score_acc_inline2014__rv_v2, [global_row0_inline2011__ssa_v0, o0_inline2017__ssa_v0], cmp4_score_proj_pad_inline2019__ssa_v0)
        return cmp4_kv_proj_pad_inline2031__ssa_v0, cmp4_score_proj_pad_inline2019__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd(self, bs_inline2038__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline2021__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 8388608)], cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)], cmp4_kv_proj_pad_inline2031__ssa_v0: pl.Out[pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 2097152)]], cmp4_score_proj_pad_inline2019__ssa_v0: pl.Out[pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)]]) -> tuple[pl.Tensor[[512, 1024], pl.FP32], pl.Tensor[[512, 1024], pl.FP32]]:
        ret__tmp_v0: pl.Tuple[pl.Tensor[[512, 1024], pl.FP32], pl.Tensor[[512, 1024], pl.FP32]] = self.kv_score_proj(bs_inline2038__ssa_v0, x_flat_inline2021__ssa_v0, cmp_wkv__ssa_v0, cmp_wgate__ssa_v0, cmp4_kv_proj_pad_inline2031__ssa_v0, cmp4_score_proj_pad_inline2019__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]})
        cmp4_kv_proj_pad_inline2031__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 2097152)] = ret__tmp_v0[0]
        cmp4_score_proj_pad_inline2019__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 2097152)] = ret__tmp_v0[1]
        return cmp4_kv_proj_pad_inline2031__ssa_v0, cmp4_score_proj_pad_inline2019__ssa_v0
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def kv_score_proj_0(bs_inline2140__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline2162__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2097152)], inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2097152)], kv_proj_pad_inline2129__ssa_v0: pl.Out[pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 524288)]], score_proj_pad_inline2143__ssa_v0: pl.Out[pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]]) -> tuple[pl.Tensor[[512, 256], pl.FP32], pl.Tensor[[512, 256], pl.FP32]]:
        mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 2048)
        mem_acc_6: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 2048)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_8: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_9: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_12: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_left_13: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_14: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_23: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_24: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        idx_inline2108__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        global_row0_inline2104__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2108__ssa_v0 // 8 * 16
        o0_inline2117__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2108__ssa_v0 % 8 * 32
        kv_acc_inline2107__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        score_acc_inline2103__tile: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        for kb_inline2105__idx_v0, (kv_acc_inline2107__iter_v1, score_acc_inline2103__iter_v1) in pl.range(0, 8, 2, init_values=(kv_acc_inline2107__tile, score_acc_inline2103__tile)):
            k0_inline2172__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2105__idx_v0 * 512
            x_rows_inline2097__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2140__ssa_v0 - global_row0_inline2104__ssa_v0, 16)
            k0_inline2172__ssa_v0_1: pl.Scalar[pl.INDEX] = kb_inline2105__idx_v0 * 512 + 512
            x_rows_inline2097__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(bs_inline2140__ssa_v0 - global_row0_inline2104__ssa_v0, 16)
            x_tile_inline2096__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline2097__ssa_v0, 512])] = pl.tile.load(x_flat_inline2162__ssa_v0, [global_row0_inline2104__ssa_v0, k0_inline2172__ssa_v0], [16, 512], [x_rows_inline2097__ssa_v0, 512], target_memory=pl.Mem.Mat)
            wkv_tile_inline2114__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(inner_wkv__ssa_v0, [o0_inline2117__ssa_v0, k0_inline2172__ssa_v0], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
            wgate_tile_inline2100__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(inner_wgate__ssa_v0, [o0_inline2117__ssa_v0, k0_inline2172__ssa_v0], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
            x_tile_inline2096__tile_1: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[x_rows_inline2097__ssa_v0_1, 512])] = pl.tile.load(x_flat_inline2162__ssa_v0, [global_row0_inline2104__ssa_v0, k0_inline2172__ssa_v0_1], [16, 512], [x_rows_inline2097__ssa_v0_1, 512], target_memory=pl.Mem.Mat)
            wkv_tile_inline2114__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(inner_wkv__ssa_v0, [o0_inline2117__ssa_v0, k0_inline2172__ssa_v0_1], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
            wgate_tile_inline2100__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_12, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(inner_wgate__ssa_v0, [o0_inline2117__ssa_v0, k0_inline2172__ssa_v0_1], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
            if k0_inline2172__ssa_v0 == 0:
                wkv_tile_inline2114__tile_t: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wkv_tile_inline2114__tile)
                x_tile_inline2096__tile_Left: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_left_13, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2097__ssa_v0, 512])] = pl.tile.move(x_tile_inline2096__tile, target_memory=pl.Mem.Left)
                wkv_tile_inline2114__tile_t_Right: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(wkv_tile_inline2114__tile_t, target_memory=pl.Mem.Right)
                kv_acc_inline2107__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline2097__ssa_v0, 32], compact=pl.CompactMode.normal)] = pl.tile.matmul(x_tile_inline2096__tile_Left, wkv_tile_inline2114__tile_t_Right)
                wgate_tile_inline2100__tile_t: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wgate_tile_inline2100__tile)
                wgate_tile_inline2100__tile_t_Right: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(wgate_tile_inline2100__tile_t, target_memory=pl.Mem.Right)
                score_acc_inline2103__tile_1: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc, pl.TileView(valid_shape=[x_rows_inline2097__ssa_v0, 32], compact=pl.CompactMode.normal)] = pl.tile.matmul(x_tile_inline2096__tile_Left, wgate_tile_inline2100__tile_t_Right)
                kv_acc_inline2107__phi_v5, score_acc_inline2103__phi_v5 = pl.yield_(kv_acc_inline2107__tile_1, score_acc_inline2103__tile_1)
            else:
                wkv_tile_inline2114__tile_t_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_mat_8, pl.const(98304, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wkv_tile_inline2114__tile)
                x_tile_inline2096__tile_Left_1: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_left_13, pl.const(0, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2097__ssa_v0, 512])] = pl.tile.move(x_tile_inline2096__tile, target_memory=pl.Mem.Left)
                wkv_tile_inline2114__tile_t_Right_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(wkv_tile_inline2114__tile_t_1, target_memory=pl.Mem.Right)
                kv_acc_inline2107__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline2107__iter_v1, x_tile_inline2096__tile_Left_1, wkv_tile_inline2114__tile_t_Right_1)
                wgate_tile_inline2100__tile_t_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_mat_9, pl.const(131072, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wgate_tile_inline2100__tile)
                wgate_tile_inline2100__tile_t_Right_1: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_14, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(wgate_tile_inline2100__tile_t_1, target_memory=pl.Mem.Right)
                score_acc_inline2103__tile_2: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(score_acc_inline2103__iter_v1, x_tile_inline2096__tile_Left_1, wgate_tile_inline2100__tile_t_Right_1)
                kv_acc_inline2107__phi_v5, score_acc_inline2103__phi_v5 = pl.yield_(kv_acc_inline2107__tile_2, score_acc_inline2103__tile_2)
            wkv_tile_inline2114__tile_t_2: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_mat_11, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wkv_tile_inline2114__tile_1)
            x_tile_inline2096__tile_Left_2: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_left_23, pl.const(16384, pl.INT64), 16384), pl.Mem.Left, pl.TileView(valid_shape=[x_rows_inline2097__ssa_v0_1, 512])] = pl.tile.move(x_tile_inline2096__tile_1, target_memory=pl.Mem.Left)
            wkv_tile_inline2114__tile_t_Right_2: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_24, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(wkv_tile_inline2114__tile_t_2, target_memory=pl.Mem.Right)
            kv_acc_inline2107__tile_3: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(kv_acc_inline2107__phi_v5, x_tile_inline2096__tile_Left_2, wkv_tile_inline2114__tile_t_Right_2)
            wgate_tile_inline2100__tile_t_2: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_mat_12, pl.const(49152, pl.INT64), 32768), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(wgate_tile_inline2100__tile_1)
            wgate_tile_inline2100__tile_t_Right_2: pl.Tile[[512, 32], pl.BF16, pl.MemRef(mem_right_24, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(wgate_tile_inline2100__tile_t_2, target_memory=pl.Mem.Right)
            score_acc_inline2103__tile_3: pl.Tile[[16, 32], pl.FP32, pl.MemRef(mem_acc_6, pl.const(2048, pl.INT64), 2048), pl.Mem.Acc] = pl.tile.matmul_acc(score_acc_inline2103__phi_v5, x_tile_inline2096__tile_Left_2, wgate_tile_inline2100__tile_t_Right_2)
            kv_acc_inline2107__rv_v2, score_acc_inline2103__rv_v2 = pl.yield_(kv_acc_inline2107__tile_3, score_acc_inline2103__tile_3)
        kv_proj_pad_inline2129__tile: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 524288)] = pl.tile.store(kv_acc_inline2107__rv_v2, [global_row0_inline2104__ssa_v0, o0_inline2117__ssa_v0], kv_proj_pad_inline2129__ssa_v0)
        score_proj_pad_inline2143__tile: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)] = pl.tile.store(score_acc_inline2103__rv_v2, [global_row0_inline2104__ssa_v0, o0_inline2117__ssa_v0], score_proj_pad_inline2143__ssa_v0)
        return kv_proj_pad_inline2129__ssa_v0, score_proj_pad_inline2143__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def kv_score_proj_spmd_0(self, bs_inline2140__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline2162__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2097152)], inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2097152)], kv_proj_pad_inline2129__ssa_v0: pl.Out[pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 524288)]], score_proj_pad_inline2143__ssa_v0: pl.Out[pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)]]) -> tuple[pl.Tensor[[512, 256], pl.FP32], pl.Tensor[[512, 256], pl.FP32]]:
        ret__tmp_v0: pl.Tuple[pl.Tensor[[512, 256], pl.FP32], pl.Tensor[[512, 256], pl.FP32]] = self.kv_score_proj_0(bs_inline2140__ssa_v0, x_flat_inline2162__ssa_v0, inner_wkv__ssa_v0, inner_wgate__ssa_v0, kv_proj_pad_inline2129__ssa_v0, score_proj_pad_inline2143__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]})
        kv_proj_pad_inline2129__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0[0]
        score_proj_pad_inline2143__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0[1]
        return kv_proj_pad_inline2129__ssa_v0, score_proj_pad_inline2143__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def kv_touch(ori_kv_flat_inline2344__ssa_v0: pl.InOut[pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16]:
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        t__tile: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(ori_kv_flat_inline2344__ssa_v0, [0, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
        ori_kv_flat_inline2344__tile: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile, [0, 0], ori_kv_flat_inline2344__ssa_v0)
        return ori_kv_flat_inline2344__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def mix_x(t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], pre_val_store_inline1529__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], x_mixed_inline1253__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], x_mixed_tail_store_inline1462__ssa_v0: pl.InOut[pl.Tensor[[8, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 65536)]], x_flat_inline1497__ssa_v0: pl.Tensor[[t_dim_inline1568__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16]:
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        blk_inline1491__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t0_inline1476__ssa_v4: pl.Scalar[pl.INDEX] = blk_inline1491__ssa_v0 * 8
        valid_rows_inline1507__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v4, 8)
        t__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41216, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(pre_val_store_inline1529__ssa_v1, [t0_inline1476__ssa_v4, 0], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
        transpose_tmp: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(49408, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([8, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        pre_tile_t_inline1443__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(40960, pl.INT64), 256), pl.Mem.Vec] = pl.tile.transpose(t__tile, 0, 1, transpose_tmp)
        t__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(40960, pl.INT64), 32), pl.Mem.Vec] = pl.tile.slice(pre_tile_t_inline1443__tile, [1, 8], [0, 0])
        pre0_inline1526__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(40960, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [8, 1])
        t__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(40992, pl.INT64), 32), pl.Mem.Vec] = pl.tile.slice(pre_tile_t_inline1443__tile, [1, 8], [1, 0])
        pre1_inline1442__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(40992, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [8, 1])
        t__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(41024, pl.INT64), 32), pl.Mem.Vec] = pl.tile.slice(pre_tile_t_inline1443__tile, [1, 8], [2, 0])
        pre2_inline1441__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(41024, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [8, 1])
        t__tile_4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(41056, pl.INT64), 32), pl.Mem.Vec] = pl.tile.slice(pre_tile_t_inline1443__tile, [1, 8], [3, 0])
        pre3_inline1440__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(41056, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_4, [8, 1])
        for db_inline1465__idx_v0, (x_mixed_inline1253__iter_v1, x_mixed_tail_store_inline1462__iter_v1) in pl.range(0, 16, 2, init_values=(x_mixed_inline1253__ssa_v0, x_mixed_tail_store_inline1462__ssa_v0)):
            d0_inline1496__ssa_v0: pl.Scalar[pl.INDEX] = db_inline1465__idx_v0 * 256
            d0_inline1496__ssa_v0_1: pl.Scalar[pl.INDEX] = db_inline1465__idx_v0 * 256 + 256
            x0_inline1439__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41216, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
            x1_inline1446__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(49408, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0 + 4096], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
            x2_inline1542__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_9, pl.const(57600, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0 + 8192], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
            x3_inline1438__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0 + 12288], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
            x0_inline1439__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0_1], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
            x1_inline1446__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0_1 + 4096], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
            x2_inline1542__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_13, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0_1 + 8192], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
            x3_inline1438__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_flat_inline1497__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0_1 + 12288], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
            y0_inline1437__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41216, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.row_expand_mul(x0_inline1439__tile, pre0_inline1526__tile)
            y1_inline1550__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(49408, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.row_expand_mul(x1_inline1446__tile, pre1_inline1442__tile)
            y2_inline1436__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_9, pl.const(57600, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.row_expand_mul(x2_inline1542__tile, pre2_inline1441__tile)
            y3_inline1435__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.row_expand_mul(x3_inline1438__tile, pre3_inline1440__tile)
            t__tile_5: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41216, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.add(y0_inline1437__tile, y1_inline1550__tile)
            t__tile_6: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(49408, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.add(y2_inline1436__tile, y3_inline1435__tile)
            y_tile_inline1434__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41216, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.add(t__tile_5, t__tile_6)
            y_bf16_inline1433__tile: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_7, pl.const(41216, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.cast(y_tile_inline1434__tile, target_type=pl.BF16, mode='rint')
            if valid_rows_inline1507__ssa_v3 == 8:
                x_mixed_inline1253__tile: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_bf16_inline1433__tile, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0], x_mixed_inline1253__iter_v1)
                x_mixed_inline1253__phi_v4, x_mixed_tail_store_inline1462__phi_v4 = pl.yield_(x_mixed_inline1253__tile, x_mixed_tail_store_inline1462__iter_v1)
            else:
                x_mixed_tail_store_inline1462__tile: pl.Tensor[[8, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 65536)] = pl.tile.store(y_bf16_inline1433__tile, [0, d0_inline1496__ssa_v0], x_mixed_tail_store_inline1462__iter_v1)
                y_out_inline1521__ssa_v0: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_7, pl.const(41216, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_mixed_tail_store_inline1462__tile, [0, d0_inline1496__ssa_v0], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(y_out_inline1521__ssa_v0, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0], x_mixed_inline1253__iter_v1)
                x_mixed_inline1253__phi_v4, x_mixed_tail_store_inline1462__phi_v4 = pl.yield_(x_mixed_inline1253__iter_v1, x_mixed_tail_store_inline1462__tile)
            y0_inline1437__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.row_expand_mul(x0_inline1439__tile_1, pre0_inline1526__tile)
            y1_inline1550__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.row_expand_mul(x1_inline1446__tile_1, pre1_inline1442__tile)
            y2_inline1436__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_13, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.row_expand_mul(x2_inline1542__tile_1, pre2_inline1441__tile)
            y3_inline1435__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32768, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.row_expand_mul(x3_inline1438__tile_1, pre3_inline1440__tile)
            t__tile_7: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.add(y0_inline1437__tile_1, y1_inline1550__tile_1)
            t__tile_8: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.add(y2_inline1436__tile_1, y3_inline1435__tile_1)
            y_tile_inline1434__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.add(t__tile_7, t__tile_8)
            y_bf16_inline1433__tile_1: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.cast(y_tile_inline1434__tile_1, target_type=pl.BF16, mode='rint')
            if valid_rows_inline1507__ssa_v3 == 8:
                x_mixed_inline1253__tile_1: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 0)] = pl.tile.store(y_bf16_inline1433__tile_1, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0_1], x_mixed_inline1253__phi_v4)
                x_mixed_inline1253__phi_v4_1, x_mixed_tail_store_inline1462__phi_v4_1 = pl.yield_(x_mixed_inline1253__tile_1, x_mixed_tail_store_inline1462__phi_v4)
            else:
                x_mixed_tail_store_inline1462__tile_1: pl.Tensor[[8, 4096], pl.BF16, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 65536)] = pl.tile.store(y_bf16_inline1433__tile_1, [0, d0_inline1496__ssa_v0_1], x_mixed_tail_store_inline1462__phi_v4)
                y_out_inline1521__ssa_v0_1: pl.Tile[[8, 256], pl.BF16, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v3, 256])] = pl.tile.load(x_mixed_tail_store_inline1462__tile_1, [0, d0_inline1496__ssa_v0_1], [8, 256], [valid_rows_inline1507__ssa_v3, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(y_out_inline1521__ssa_v0_1, [t0_inline1476__ssa_v4, d0_inline1496__ssa_v0_1], x_mixed_inline1253__phi_v4)
                x_mixed_inline1253__phi_v4_1, x_mixed_tail_store_inline1462__phi_v4_1 = pl.yield_(x_mixed_inline1253__phi_v4, x_mixed_tail_store_inline1462__tile_1)
            x_mixed_inline1253__rv_v2, x_mixed_tail_store_inline1462__rv_v2 = pl.yield_(x_mixed_inline1253__phi_v4_1, x_mixed_tail_store_inline1462__phi_v4_1)
        return x_mixed_inline1253__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def mix_x_spmd(self, t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], pre_val_store_inline1529__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], x_mixed_inline1253__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], x_mixed_tail_store_inline1462__ssa_v0: pl.InOut[pl.Tensor[[8, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 65536)]], x_flat_inline1497__ssa_v0: pl.Tensor[[t_dim_inline1568__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16]:
        x_mixed_inline1253__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = self.mix_x(t_dim_inline1568__ssa_v0, pre_val_store_inline1529__ssa_v1, x_mixed_inline1253__ssa_v0, x_mixed_tail_store_inline1462__ssa_v0, x_flat_inline1497__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.inout, pl.adir.input]})
        return x_mixed_inline1253__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def o_group_a2a_complete(attention_local_flat_inline1292__rv_v2: pl.InOut[pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], tp_rank__ssa_v0: pl.Scalar[pl.INT32], attention_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 8)]], group_base__ssa_v0: pl.Scalar[pl.INT32], attention_signal_ctx: pld.CommCtx):
        completion_anchor_inline2439__tile: pl.Scalar[pl.BF16] = pl.tensor.read(attention_local_flat_inline1292__rv_v2, [0, 0])
        for peer_tp_inline2438__idx_v0 in pl.range(2):
            if peer_tp_inline2438__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(attention_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline2438__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
        completion_expected_inline2425__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(pl.cast(48, pl.INDEX) + 1, pl.INT32)
        for source_tp_inline2424__idx_v0 in pl.range(2):
            if source_tp_inline2424__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.wait(attention_signal__ssa_v0, [source_tp_inline2424__idx_v0, 0], completion_expected_inline2425__ssa_v0, cmp=1)
        pl.system.cacheinvalid()
        reset_value_inline2434__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(-completion_expected_inline2425__ssa_v0, pl.INT32)
        self_rank_inline2423__ssa_v0: pl.Scalar[pl.INT32] = group_base__ssa_v0 + tp_rank__ssa_v0
        for source_tp_inline2429__idx_v0 in pl.range(2):
            if source_tp_inline2429__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(attention_signal__ssa_v0, self_rank_inline2423__ssa_v0, [source_tp_inline2429__idx_v0, 0], reset_value_inline2434__ssa_v0, op=0)
        pl.tensor.write(attention_local_flat_inline1292__rv_v2, [0, 0], completion_anchor_inline2439__tile)
        return
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def o_group_a2a_gather(attention_local_flat_inline1292__ssa_v0: pl.Out[pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], attention_window__ssa_v0: pld.DistributedTensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16777216)], attention_window_ctx: pld.CommCtx) -> pl.Tensor[[2048, 4096], pl.BF16]:
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        worker_inline2433__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for local_group_inline2435__idx_v0, (attention_local_flat_inline1292__iter_v1,) in pl.range(4, init_values=(attention_local_flat_inline1292__ssa_v0,)):
            group_base_row_inline2437__ssa_v0: pl.Scalar[pl.INDEX] = local_group_inline2435__idx_v0 * 512
            for group_row_inline2432__idx_v0, (attention_local_flat_inline1292__iter_v3,) in pl.range(worker_inline2433__ssa_v0, 512, 48, init_values=(attention_local_flat_inline1292__iter_v1,)):
                copy_row_inline2430__ssa_v0: pl.Scalar[pl.INDEX] = group_base_row_inline2437__ssa_v0 + group_row_inline2432__idx_v0
                t__tile: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(attention_window__ssa_v0, [copy_row_inline2430__ssa_v0, 0], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
                attention_local_flat_inline1292__tile: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(t__tile, [copy_row_inline2430__ssa_v0, 0], attention_local_flat_inline1292__iter_v3)
                attention_local_flat_inline1292__rv_v4: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16777216)] = pl.yield_(attention_local_flat_inline1292__tile)
            attention_local_flat_inline1292__rv_v2: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 16777216)] = pl.yield_(attention_local_flat_inline1292__rv_v4)
        return attention_local_flat_inline1292__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def o_group_a2a_gather_spmd(self, attention_local_flat_inline1292__ssa_v0: pl.Out[pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], attention_window__ssa_v0: pld.DistributedTensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16777216)], attention_window_ctx: pld.CommCtx) -> pl.Tensor[[2048, 4096], pl.BF16]:
        attention_local_flat_inline1292__rv_v2: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16777216)] = self.o_group_a2a_gather(attention_local_flat_inline1292__ssa_v0, attention_window__ssa_v0, attention_window_ctx, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar]})
        return attention_local_flat_inline1292__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def o_group_a2a_wait(tp_rank__ssa_v0: pl.Scalar[pl.INT32], attention_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8)], attention_signal_ctx: pld.CommCtx):
        expected_inline2426__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(48, pl.INT32)
        for source_tp_inline2427__idx_v0 in pl.range(2):
            if source_tp_inline2427__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.wait(attention_signal__ssa_v0, [source_tp_inline2427__idx_v0, 0], expected_inline2426__ssa_v0, cmp=1)
        pl.system.cacheinvalid()
        return
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def q_rope_prepare(t_dim_inline1682__ssa_v0: pl.Scalar[pl.INDEX], rope_cos_view_inline1679__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], rope_sin_view_inline1674__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], rope_cos_il_view_inline1670__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], rope_sin_signed_view_inline1668__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], rope_swap_idx_view_inline1694__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]) -> tuple[pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32]]:
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_35: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_61: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        qrp_idx_inline1681__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qrp_t0_inline1672__ssa_v0: pl.Scalar[pl.INDEX] = qrp_idx_inline1681__ssa_v0 * 8
        qrp_valid_rows_inline1667__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1682__ssa_v0 - qrp_t0_inline1672__ssa_v0, 8)
        qrp_ones_inline1686__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        qrp_idx_i32_inline1666__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        qrp_idx_fp32_inline1661__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(qrp_idx_i32_inline1666__tile, target_type=pl.FP32, mode='round')
        qrp_col_inline1675__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(qrp_ones_inline1686__tile, qrp_idx_fp32_inline1661__tile)
        qrp_half_inline1683__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_col_inline1675__tile, 0.5)
        qrp_dup_i32_inline1685__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_half_inline1683__tile, target_type=pl.INT32, mode='trunc')
        qrp_dup_f_inline1663__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_dup_i32_inline1685__tile, target_type=pl.FP32, mode='round')
        qrp_dup_idx_inline1680__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_dup_f_inline1663__tile, target_type=pl.INT32, mode='round')
        t__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_dup_f_inline1663__tile, 2.0)
        qrp_lane_inline1678__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_col_inline1675__tile, t__tile)
        qrp_next_col_inline1689__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_col_inline1675__tile, 1.0)
        qrp_lane_offset_inline1692__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1678__tile, 2.0)
        qrp_swap_f_inline1690__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_next_col_inline1689__tile, qrp_lane_offset_inline1692__tile)
        qrp_swap_idx_inline1695__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_swap_f_inline1690__tile, target_type=pl.INT32, mode='round')
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1678__tile, 2.0)
        qrp_sign_inline1687__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_1, 1.0)
        if qrp_valid_rows_inline1667__ssa_v0 == 8:
            qrp_cos_rows_full_inline1693__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(rope_cos_view_inline1679__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
            qrp_sin_rows_full_inline1696__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(rope_sin_view_inline1674__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
            qrp_cos_full_inline1676__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_cos_rows_full_inline1693__tile, target_type=pl.FP32, mode='round')
            qrp_sin_full_inline1669__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_sin_rows_full_inline1696__tile, target_type=pl.FP32, mode='round')
            gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qrp_cos_full_inline1676__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qrp_dup_idx_inline1680__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_61, pl.const(11264, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            qrp_cos_il_full_inline1697__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
            gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qrp_sin_full_inline1669__tile, [1, 64], [gather_lv_1, 0], [1, 64])
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qrp_dup_idx_inline1680__tile, [1, 64], [gather_lv_1, 0], [1, 64])
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_61, pl.const(11264, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            qrp_sin_il_full_inline1671__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
            qrp_sin_signed_full_inline1691__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_full_inline1671__tile, qrp_sign_inline1687__tile)
            rope_cos_il_view_inline1670__tile: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(qrp_cos_il_full_inline1697__tile, [qrp_t0_inline1672__ssa_v0, 0], rope_cos_il_view_inline1670__ssa_v0)
            rope_sin_signed_view_inline1668__tile: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(qrp_sin_signed_full_inline1691__tile, [qrp_t0_inline1672__ssa_v0, 0], rope_sin_signed_view_inline1668__ssa_v0)
            rope_swap_idx_view_inline1694__tile: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(qrp_swap_idx_inline1695__tile, [qrp_t0_inline1672__ssa_v0, 0], rope_swap_idx_view_inline1694__ssa_v0)
        else:
            qrp_cos_rows_tail_inline1684__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_61, pl.const(11264, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.load(rope_cos_view_inline1679__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1667__ssa_v0, 64], target_memory=pl.Mem.Vec)
            qrp_sin_rows_tail_inline1660__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_35, pl.const(8192, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.load(rope_sin_view_inline1674__ssa_v0, [qrp_t0_inline1672__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1667__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v89: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
            t__tmp_v90: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            t__tmp_v91: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v90, target_type=pl.FP32, mode='round')
            qrp_tail_col_inline1659__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v89, t__tmp_v91)
            t__tmp_v92: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_col_inline1659__ssa_v0, 0.5)
            t__tmp_v93: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v92, target_type=pl.INT32, mode='trunc')
            qrp_tail_dup_f_inline1657__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v93, target_type=pl.FP32, mode='round')
            t__tmp_v94: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_dup_f_inline1657__ssa_v0, 2.0)
            qrp_tail_lane_inline1662__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_tail_col_inline1659__ssa_v0, t__tmp_v94)
            t__tmp_v95: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_tail_col_inline1659__ssa_v0, 1.0)
            t__tmp_v96: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1662__ssa_v0, 2.0)
            qrp_tail_swap_f_inline1665__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v95, t__tmp_v96)
            t__tmp_v97: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
            t__tmp_v98: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v97, target_type=pl.FP32, mode='round')
            qrp_row_seed_inline1664__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v98, 64.0)
            t__tmp_v99: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
            qrp_row_grid_inline1656__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v99, qrp_row_seed_inline1664__ssa_v0)
            transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            qrp_row_offset_inline1655__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(qrp_row_grid_inline1656__ssa_v0, 0, 1, transpose_tmp)
            t__tmp_v100: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(qrp_tail_dup_f_inline1657__ssa_v0, qrp_row_offset_inline1655__ssa_v0)
            qrp_dup_idx_tail_inline1658__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v100, target_type=pl.INT32, mode='round')
            qrp_gather_tmp_inline1653__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            t__tmp_v101: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.cast(qrp_cos_rows_tail_inline1684__ssa_v0, target_type=pl.FP32, mode='round')
            qrp_cos_il_tail_inline1652__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(t__tmp_v101, qrp_dup_idx_tail_inline1658__ssa_v0, qrp_gather_tmp_inline1653__ssa_v0)
            t__tmp_v102: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.cast(qrp_sin_rows_tail_inline1660__ssa_v0, target_type=pl.FP32, mode='round')
            qrp_sin_il_tail_inline1654__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_61, pl.const(11264, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(t__tmp_v102, qrp_dup_idx_tail_inline1658__ssa_v0, qrp_gather_tmp_inline1653__ssa_v0)
            t__tmp_v103: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1662__ssa_v0, 2.0)
            qrp_tail_sign_inline1677__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tmp_v103, 1.0)
            qrp_sin_signed_tail_inline1688__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_tail_inline1654__ssa_v0, qrp_tail_sign_inline1677__ssa_v0)
            t__tmp_v104: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.set_validshape(qrp_cos_il_tail_inline1652__ssa_v0, qrp_valid_rows_inline1667__ssa_v0, 64)
            rope_cos_il_view_inline1670__store: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tmp_v104, [qrp_t0_inline1672__ssa_v0, 0], rope_cos_il_view_inline1670__ssa_v0)
            t__tmp_v105: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.set_validshape(qrp_sin_signed_tail_inline1688__ssa_v0, qrp_valid_rows_inline1667__ssa_v0, 64)
            rope_sin_signed_view_inline1668__store: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tmp_v105, [qrp_t0_inline1672__ssa_v0, 0], rope_sin_signed_view_inline1668__ssa_v0)
            t__tmp_v106: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_tail_swap_f_inline1665__ssa_v0, target_type=pl.INT32, mode='round')
            t__tmp_v107: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1667__ssa_v0, 64])] = pl.tile.set_validshape(t__tmp_v106, qrp_valid_rows_inline1667__ssa_v0, 64)
            rope_swap_idx_view_inline1694__store: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tmp_v107, [qrp_t0_inline1672__ssa_v0, 0], rope_swap_idx_view_inline1694__ssa_v0)
        return rope_cos_il_view_inline1670__ssa_v0, rope_sin_signed_view_inline1668__ssa_v0, rope_swap_idx_view_inline1694__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def q_rope_prepare_spmd(self, t_dim_inline1682__ssa_v0: pl.Scalar[pl.INDEX], rope_cos_view_inline1679__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], rope_sin_view_inline1674__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], rope_cos_il_view_inline1670__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], rope_sin_signed_view_inline1668__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], rope_swap_idx_view_inline1694__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]):
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32]] = self.q_rope_prepare(t_dim_inline1682__ssa_v0, rope_cos_view_inline1679__ssa_v0, rope_sin_view_inline1674__ssa_v0, rope_cos_il_view_inline1670__ssa_v0, rope_sin_signed_view_inline1668__ssa_v0, rope_swap_idx_view_inline1694__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing]})
        rope_cos_il_view_inline1670__ssa_v2: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        rope_sin_signed_view_inline1668__ssa_v2: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        rope_swap_idx_view_inline1694__ssa_v2: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def q_rope_prepare_0(t_dim_inline1728__ssa_v0: pl.Scalar[pl.INDEX], rope_cos_view_inline1725__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], rope_sin_view_inline1720__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], rope_cos_il_view_inline1716__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], rope_sin_signed_view_inline1714__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], rope_swap_idx_view_inline1740__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]) -> tuple[pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32]]:
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_35: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_61: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        qrp_idx_inline1727__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qrp_t0_inline1718__ssa_v0: pl.Scalar[pl.INDEX] = qrp_idx_inline1727__ssa_v0 * 8
        qrp_valid_rows_inline1713__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1728__ssa_v0 - qrp_t0_inline1718__ssa_v0, 8)
        qrp_ones_inline1732__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        qrp_idx_i32_inline1712__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        qrp_idx_fp32_inline1707__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(qrp_idx_i32_inline1712__tile, target_type=pl.FP32, mode='round')
        qrp_col_inline1721__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(qrp_ones_inline1732__tile, qrp_idx_fp32_inline1707__tile)
        qrp_half_inline1729__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_col_inline1721__tile, 0.5)
        qrp_dup_i32_inline1731__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_half_inline1729__tile, target_type=pl.INT32, mode='trunc')
        qrp_dup_f_inline1709__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_dup_i32_inline1731__tile, target_type=pl.FP32, mode='round')
        qrp_dup_idx_inline1726__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_dup_f_inline1709__tile, target_type=pl.INT32, mode='round')
        t__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_dup_f_inline1709__tile, 2.0)
        qrp_lane_inline1724__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_col_inline1721__tile, t__tile)
        qrp_next_col_inline1735__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_col_inline1721__tile, 1.0)
        qrp_lane_offset_inline1738__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1724__tile, 2.0)
        qrp_swap_f_inline1736__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_next_col_inline1735__tile, qrp_lane_offset_inline1738__tile)
        qrp_swap_idx_inline1741__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_swap_f_inline1736__tile, target_type=pl.INT32, mode='round')
        t__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_lane_inline1724__tile, 2.0)
        qrp_sign_inline1733__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_1, 1.0)
        if qrp_valid_rows_inline1713__ssa_v0 == 8:
            qrp_cos_rows_full_inline1739__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(rope_cos_view_inline1725__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
            qrp_sin_rows_full_inline1742__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.load(rope_sin_view_inline1720__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
            qrp_cos_full_inline1722__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_cos_rows_full_inline1739__tile, target_type=pl.FP32, mode='round')
            qrp_sin_full_inline1715__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_sin_rows_full_inline1742__tile, target_type=pl.FP32, mode='round')
            gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qrp_cos_full_inline1722__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qrp_dup_idx_inline1726__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_61, pl.const(11264, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            qrp_cos_il_full_inline1743__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
            gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qrp_sin_full_inline1715__tile, [1, 64], [gather_lv_1, 0], [1, 64])
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qrp_dup_idx_inline1726__tile, [1, 64], [gather_lv_1, 0], [1, 64])
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_61, pl.const(11264, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_35, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            qrp_sin_il_full_inline1717__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
            qrp_sin_signed_full_inline1737__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_full_inline1717__tile, qrp_sign_inline1733__tile)
            rope_cos_il_view_inline1716__tile: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(qrp_cos_il_full_inline1743__tile, [qrp_t0_inline1718__ssa_v0, 0], rope_cos_il_view_inline1716__ssa_v0)
            rope_sin_signed_view_inline1714__tile: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(qrp_sin_signed_full_inline1737__tile, [qrp_t0_inline1718__ssa_v0, 0], rope_sin_signed_view_inline1714__ssa_v0)
            rope_swap_idx_view_inline1740__tile: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(qrp_swap_idx_inline1741__tile, [qrp_t0_inline1718__ssa_v0, 0], rope_swap_idx_view_inline1740__ssa_v0)
        else:
            qrp_cos_rows_tail_inline1730__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_61, pl.const(11264, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.load(rope_cos_view_inline1725__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1713__ssa_v0, 64], target_memory=pl.Mem.Vec)
            qrp_sin_rows_tail_inline1706__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_35, pl.const(8192, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.load(rope_sin_view_inline1720__ssa_v0, [qrp_t0_inline1718__ssa_v0, 0], [8, 64], [qrp_valid_rows_inline1713__ssa_v0, 64], target_memory=pl.Mem.Vec)
            t__tmp_v110: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
            t__tmp_v111: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
            t__tmp_v112: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v111, target_type=pl.FP32, mode='round')
            qrp_tail_col_inline1705__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v110, t__tmp_v112)
            t__tmp_v113: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_col_inline1705__ssa_v0, 0.5)
            t__tmp_v114: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v113, target_type=pl.INT32, mode='trunc')
            qrp_tail_dup_f_inline1703__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v114, target_type=pl.FP32, mode='round')
            t__tmp_v115: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_dup_f_inline1703__ssa_v0, 2.0)
            qrp_tail_lane_inline1708__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(qrp_tail_col_inline1705__ssa_v0, t__tmp_v115)
            t__tmp_v116: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(qrp_tail_col_inline1705__ssa_v0, 1.0)
            t__tmp_v117: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1708__ssa_v0, 2.0)
            qrp_tail_swap_f_inline1711__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v116, t__tmp_v117)
            t__tmp_v118: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
            t__tmp_v119: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v118, target_type=pl.FP32, mode='round')
            qrp_row_seed_inline1710__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v119, 64.0)
            t__tmp_v120: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
            qrp_row_grid_inline1702__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v120, qrp_row_seed_inline1710__ssa_v0)
            transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            qrp_row_offset_inline1701__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(qrp_row_grid_inline1702__ssa_v0, 0, 1, transpose_tmp)
            t__tmp_v121: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(qrp_tail_dup_f_inline1703__ssa_v0, qrp_row_offset_inline1701__ssa_v0)
            qrp_dup_idx_tail_inline1704__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v121, target_type=pl.INT32, mode='round')
            qrp_gather_tmp_inline1699__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_16, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            t__tmp_v122: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.cast(qrp_cos_rows_tail_inline1730__ssa_v0, target_type=pl.FP32, mode='round')
            qrp_cos_il_tail_inline1698__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(t__tmp_v122, qrp_dup_idx_tail_inline1704__ssa_v0, qrp_gather_tmp_inline1699__ssa_v0)
            t__tmp_v123: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.cast(qrp_sin_rows_tail_inline1706__ssa_v0, target_type=pl.FP32, mode='round')
            qrp_sin_il_tail_inline1700__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_61, pl.const(11264, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(t__tmp_v123, qrp_dup_idx_tail_inline1704__ssa_v0, qrp_gather_tmp_inline1699__ssa_v0)
            t__tmp_v124: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(qrp_tail_lane_inline1708__ssa_v0, 2.0)
            qrp_tail_sign_inline1723__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tmp_v124, 1.0)
            qrp_sin_signed_tail_inline1734__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(qrp_sin_il_tail_inline1700__ssa_v0, qrp_tail_sign_inline1723__ssa_v0)
            t__tmp_v125: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_25, pl.const(6144, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.set_validshape(qrp_cos_il_tail_inline1698__ssa_v0, qrp_valid_rows_inline1713__ssa_v0, 64)
            rope_cos_il_view_inline1716__store: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tmp_v125, [qrp_t0_inline1718__ssa_v0, 0], rope_cos_il_view_inline1716__ssa_v0)
            t__tmp_v126: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(13312, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.set_validshape(qrp_sin_signed_tail_inline1734__ssa_v0, qrp_valid_rows_inline1713__ssa_v0, 64)
            rope_sin_signed_view_inline1714__store: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tmp_v126, [qrp_t0_inline1718__ssa_v0, 0], rope_sin_signed_view_inline1714__ssa_v0)
            t__tmp_v127: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qrp_tail_swap_f_inline1711__ssa_v0, target_type=pl.INT32, mode='round')
            t__tmp_v128: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(9216, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[qrp_valid_rows_inline1713__ssa_v0, 64])] = pl.tile.set_validshape(t__tmp_v127, qrp_valid_rows_inline1713__ssa_v0, 64)
            rope_swap_idx_view_inline1740__store: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tmp_v128, [qrp_t0_inline1718__ssa_v0, 0], rope_swap_idx_view_inline1740__ssa_v0)
        return rope_cos_il_view_inline1716__ssa_v0, rope_sin_signed_view_inline1714__ssa_v0, rope_swap_idx_view_inline1740__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def q_rope_prepare_spmd_0(self, t_dim_inline1728__ssa_v0: pl.Scalar[pl.INDEX], rope_cos_view_inline1725__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], rope_sin_view_inline1720__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], rope_cos_il_view_inline1716__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], rope_sin_signed_view_inline1714__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], rope_swap_idx_view_inline1740__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]):
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32]] = self.q_rope_prepare_0(t_dim_inline1728__ssa_v0, rope_cos_view_inline1725__ssa_v0, rope_sin_view_inline1720__ssa_v0, rope_cos_il_view_inline1716__ssa_v0, rope_sin_signed_view_inline1714__ssa_v0, rope_swap_idx_view_inline1740__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing]})
        rope_cos_il_view_inline1716__ssa_v2: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        rope_sin_signed_view_inline1714__ssa_v2: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        rope_swap_idx_view_inline1740__ssa_v2: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qk_pv_aic(qk_items_inline2347__ssa_v0: pl.Scalar[pl.INDEX], sparse_blk_li_inline2405__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], sparse_blk_mi_inline2404__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], sparse_blk_oi_inline2398__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], qk_order_inline2351__ssa_v0: pl.Tensor[[1280], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 5120)], sparse_bias_inline2381__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)], valid_block_mask_inline2385__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 5], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], position_ids_t1_inline1288__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)], ori_kv_flat_inline2344__ssa_v1: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)], cmp_sparse_indices_inline2383__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)], cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)], cmp_kv_flat_inline2401__ssa_v0: pl.Tensor[[cmp_block_num_inline2376__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)], q_flat_inline2355__ssa_v0: pl.Tensor[[t_heads_inline2364__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)], __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 4)]]):
        mem_mat_14: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 131072)
        mem_mat_25: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_left_27: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_28: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_29: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_30: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_acc_39: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        qk_pv_v2c_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name='qk_pv_v2c_slot_buffer', size=32768, base=0)
        qk_pv_c2v_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name='qk_pv_c2v_slot_buffer', peer_func='qk_pv_aiv')
        pl.system.aic_initialize_pipe(qk_pv_c2v_slot_buffer_import, qk_pv_v2c_slot_buffer, dir_mask=3, slot_size=16384, slot_num=2)
        qk_core_inline2368__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        qk_lane_iters_inline2408__ssa_v0: pl.Scalar[pl.INDEX] = (qk_items_inline2347__ssa_v0 - qk_core_inline2368__ssa_v0 + 23) // 24
        for qk_it_inline2413__idx_v0, (sparse_blk_li_inline2405__iter_v1, sparse_blk_mi_inline2404__iter_v1, sparse_blk_oi_inline2398__iter_v1) in pl.range(qk_lane_iters_inline2408__ssa_v0, init_values=(sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0)):
            qk_flat_inline2365__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2368__ssa_v0 + qk_it_inline2413__idx_v0 * 24
            t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(qk_order_inline2351__ssa_v0, [qk_flat_inline2365__ssa_v0])
            qk_item_inline2403__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX)
            qk_t_inline2411__ssa_v0: pl.Scalar[pl.INDEX] = qk_item_inline2403__ssa_v0 // 5
            qk_sb_inline2374__ssa_v0: pl.Scalar[pl.INDEX] = qk_item_inline2403__ssa_v0 - qk_t_inline2411__ssa_v0 * 5
            qk_b_inline2371__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2411__ssa_v0 // 8
            qk_token_base_inline2391__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2411__ssa_v0 * 320
            qk_s0_inline2418__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline2374__ssa_v0 * 128
            qk_block_valid_inline2422__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2385__ssa_v0, [qk_t_inline2411__ssa_v0, qk_sb_inline2374__ssa_v0])
            if 0 < pl.cast(qk_block_valid_inline2422__tile, pl.INDEX):
                qk_kv_inline2415__tile: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.create([128, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat, transpose=False)
                qk_win_rows_inline2406__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(128 - qk_s0_inline2418__ssa_v0, 0), 128)
                if 0 < qk_win_rows_inline2406__ssa_v0:
                    t__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(position_ids_t1_inline1288__ssa_v0, [qk_t_inline2411__ssa_v0, 0])
                    qk_pos_inline2339__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_1, pl.INDEX)
                    qk_win_len_inline2337__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(qk_pos_inline2339__ssa_v0 + 1, 128)
                    qk_win_start_inline2333__ssa_v0: pl.Scalar[pl.INDEX] = qk_pos_inline2339__ssa_v0 - qk_win_len_inline2337__ssa_v0 + 1
                    qk_run_rows_inline2332__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(pl.max(qk_win_len_inline2337__ssa_v0 - qk_s0_inline2418__ssa_v0, 0), qk_win_rows_inline2406__ssa_v0)
                    qk_head_inline2353__ssa_v0: pl.Scalar[pl.INDEX] = (qk_win_start_inline2333__ssa_v0 + qk_s0_inline2418__ssa_v0) % 32
                    qk_run_lo_inline2372__ssa_v0: pl.Scalar[pl.INDEX] = 0
                    qk_run_hi_inline2330__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(32 - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                    if qk_run_lo_inline2372__ssa_v0 < qk_run_hi_inline2330__ssa_v0:
                        qk_run_raw_inline2357__tile: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v0])
                        qk_run_src_inline2384__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__tile, pl.INDEX), 0), pl.INDEX)
                        qk_kv_inline2415__tile_1: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__tile, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v0, 0], [qk_run_src_inline2384__ssa_v0, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v0 - qk_run_lo_inline2372__ssa_v0, 512], transpose=False)
                        qk_kv_inline2415__phi_v2: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_1)
                    else:
                        qk_kv_inline2415__phi_v2: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile)
                    qk_run_lo_inline2372__ssa_v1: pl.Scalar[pl.INDEX] = 32 - qk_head_inline2353__ssa_v0
                    qk_run_hi_inline2330__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(64 - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                    if qk_run_lo_inline2372__ssa_v1 < qk_run_hi_inline2330__ssa_v1:
                        qk_run_raw_inline2357__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v1])
                        qk_run_src_inline2384__ssa_v1: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__tile_1, pl.INDEX), 0), pl.INDEX)
                        qk_kv_inline2415__tile_2: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__phi_v2, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v1, 0], [qk_run_src_inline2384__ssa_v1, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v1 - qk_run_lo_inline2372__ssa_v1, 512], transpose=False)
                        qk_kv_inline2415__phi_v4: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_2)
                    else:
                        qk_kv_inline2415__phi_v4: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__phi_v2)
                    qk_run_lo_inline2372__ssa_v2: pl.Scalar[pl.INDEX] = 64 - qk_head_inline2353__ssa_v0
                    qk_run_hi_inline2330__ssa_v2: pl.Scalar[pl.INDEX] = pl.min(96 - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                    if qk_run_lo_inline2372__ssa_v2 < qk_run_hi_inline2330__ssa_v2:
                        qk_run_raw_inline2357__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v2])
                        qk_run_src_inline2384__ssa_v2: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__tile_2, pl.INDEX), 0), pl.INDEX)
                        qk_kv_inline2415__tile_3: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__phi_v4, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v2, 0], [qk_run_src_inline2384__ssa_v2, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v2 - qk_run_lo_inline2372__ssa_v2, 512], transpose=False)
                        qk_kv_inline2415__phi_v6: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_3)
                    else:
                        qk_kv_inline2415__phi_v6: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__phi_v4)
                    qk_run_lo_inline2372__ssa_v3: pl.Scalar[pl.INDEX] = 96 - qk_head_inline2353__ssa_v0
                    qk_run_hi_inline2330__ssa_v3: pl.Scalar[pl.INDEX] = pl.min(128 - qk_head_inline2353__ssa_v0, qk_run_rows_inline2332__ssa_v0)
                    if qk_run_lo_inline2372__ssa_v3 < qk_run_hi_inline2330__ssa_v3:
                        qk_run_raw_inline2357__tile_3: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v3])
                        qk_run_src_inline2384__ssa_v3: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__tile_3, pl.INDEX), 0), pl.INDEX)
                        qk_kv_inline2415__tile_4: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__phi_v6, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v3, 0], [qk_run_src_inline2384__ssa_v3, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v3 - qk_run_lo_inline2372__ssa_v3, 512], transpose=False)
                        qk_kv_inline2415__phi_v8: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_4)
                    else:
                        qk_kv_inline2415__phi_v8: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__phi_v6)
                    qk_run_lo_inline2372__ssa_v4: pl.Scalar[pl.INDEX] = 128 - qk_head_inline2353__ssa_v0
                    qk_run_hi_inline2330__ssa_v4: pl.Scalar[pl.INDEX] = qk_run_rows_inline2332__ssa_v0
                    if qk_run_lo_inline2372__ssa_v4 < qk_run_hi_inline2330__ssa_v4:
                        qk_run_raw_inline2357__tile_4: pl.Scalar[pl.INT32] = pl.tensor.read(window_swa_indices__ssa_v0, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0 + qk_run_lo_inline2372__ssa_v4])
                        qk_run_src_inline2384__ssa_v4: pl.Scalar[pl.INDEX] = pl.cast(pl.max(pl.cast(qk_run_raw_inline2357__tile_4, pl.INDEX), 0), pl.INDEX)
                        qk_kv_inline2415__tile_5: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__phi_v8, ori_kv_flat_inline2344__ssa_v1, [qk_run_lo_inline2372__ssa_v4, 0], [qk_run_src_inline2384__ssa_v4, 0], [128, 512], valid_shape=[qk_run_hi_inline2330__ssa_v4 - qk_run_lo_inline2372__ssa_v4, 512], transpose=False)
                        qk_kv_inline2415__phi_v10: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_5)
                    else:
                        qk_kv_inline2415__phi_v10: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__phi_v8)
                    qk_tail_n_inline2329__ssa_v0: pl.Scalar[pl.INDEX] = qk_win_rows_inline2406__ssa_v0 - qk_run_rows_inline2332__ssa_v0
                    if 0 < qk_tail_n_inline2329__ssa_v0:
                        qk_kv_inline2415__tile_6: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__phi_v10, ori_kv_flat_inline2344__ssa_v1, [qk_run_rows_inline2332__ssa_v0, 0], [0, 0], [128, 512], valid_shape=[qk_tail_n_inline2329__ssa_v0, 512], transpose=False)
                        qk_kv_inline2415__phi_v12: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_6)
                    else:
                        qk_kv_inline2415__phi_v12: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__phi_v10)
                    qk_kv_inline2415__phi_v13: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__phi_v12)
                else:
                    qk_kv_inline2415__phi_v13: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile)
                for qk_r_inline2390__idx_v0, (qk_kv_inline2415__iter_v14,) in pl.range(qk_win_rows_inline2406__ssa_v0, 128, init_values=(qk_kv_inline2415__phi_v13,)):
                    qk_cmp_k_inline2328__ssa_v0: pl.Scalar[pl.INDEX] = qk_s0_inline2418__ssa_v0 + qk_r_inline2390__idx_v0 - 128
                    if qk_cmp_k_inline2328__ssa_v0 < 512:
                        qk_ridx_inline2377__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_sparse_indices_inline2383__rv_v2, [qk_t_inline2411__ssa_v0, qk_cmp_k_inline2328__ssa_v0])
                        if 0 <= pl.cast(qk_ridx_inline2377__tile, pl.INDEX):
                            qk_slot_inline2388__ssa_v0: pl.Scalar[pl.INT32] = qk_ridx_inline2377__tile
                            t__tile_2: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_block_table__ssa_v0, [qk_b_inline2371__ssa_v0, pl.cast(qk_slot_inline2388__ssa_v0, pl.INDEX) // 32])
                            qk_cblk_inline2327__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile_2, pl.INDEX)
                            qk_csrc_inline2400__ssa_v0: pl.Scalar[pl.INDEX] = qk_cblk_inline2327__ssa_v0 * 32 + pl.cast(qk_slot_inline2388__ssa_v0, pl.INDEX) % 32
                            qk_kv_inline2415__tile_7: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__iter_v14, cmp_kv_flat_inline2401__ssa_v0, [qk_r_inline2390__idx_v0, 0], [qk_csrc_inline2400__ssa_v0, 0], [1, 512], transpose=False)
                            qk_kv_inline2415__phi_v18: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_7)
                        else:
                            qk_kv_inline2415__tile_8: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__iter_v14, ori_kv_flat_inline2344__ssa_v1, [qk_r_inline2390__idx_v0, 0], [0, 0], [1, 512], transpose=False)
                            qk_kv_inline2415__phi_v18: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_8)
                        qk_kv_inline2415__phi_v20: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__phi_v18)
                    else:
                        qk_kv_inline2415__tile_9: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.tile.gather_row(qk_kv_inline2415__iter_v14, ori_kv_flat_inline2344__ssa_v1, [qk_r_inline2390__idx_v0, 0], [0, 0], [1, 512], transpose=False)
                        qk_kv_inline2415__phi_v20: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__tile_9)
                    qk_kv_inline2415__rv_v15: pl.Tile[[128, 512], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat] = pl.yield_(qk_kv_inline2415__phi_v20)
                qk_h0_inline2393__ssa_v0: pl.Scalar[pl.INDEX] = 0
                qk_head_row_inline2326__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2411__ssa_v0 * 64 + qk_h0_inline2393__ssa_v0
                qk_q_tile_inline2338__tile: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_25, pl.const(163840, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(q_flat_inline2355__ssa_v0, [qk_head_row_inline2326__ssa_v0, 0], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                qk_kv_inline2415__rv_v15_t: pl.Tile[[512, 128], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(qk_kv_inline2415__rv_v15)
                qk_raw_inline2397__tile_l0_init: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 16384), pl.Mem.Acc] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                for qk_raw_inline2397__tile_l0_ko, (qk_raw_inline2397__tile_l0_c,) in pl.range(0, 512, 256, init_values=(qk_raw_inline2397__tile_l0_init,)):
                    qk_raw_inline2397__tile_l0_a: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_left_27, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qk_q_tile_inline2338__tile, 0, qk_raw_inline2397__tile_l0_ko, [32, 128], target_memory=pl.Mem.Left)
                    qk_raw_inline2397__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_28, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(qk_kv_inline2415__rv_v15_t, qk_raw_inline2397__tile_l0_ko, 0, [128, 128], target_memory=pl.Mem.Right)
                    qk_raw_inline2397__tile_l0_a_1: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_left_29, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qk_q_tile_inline2338__tile, 0, qk_raw_inline2397__tile_l0_ko + 128, [32, 128], target_memory=pl.Mem.Left)
                    qk_raw_inline2397__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_30, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(qk_kv_inline2415__rv_v15_t, qk_raw_inline2397__tile_l0_ko + 128, 0, [128, 128], target_memory=pl.Mem.Right)
                    qk_raw_inline2397__tile_l0_c_acc: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 16384), pl.Mem.Acc] = pl.tile.matmul_acc(qk_raw_inline2397__tile_l0_c, qk_raw_inline2397__tile_l0_a, qk_raw_inline2397__tile_l0_b, qk_raw_inline2397__tile_l0_ko == 0)
                    qk_raw_inline2397__tile_l0_c_acc_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 16384), pl.Mem.Acc] = pl.tile.matmul_acc(qk_raw_inline2397__tile_l0_c_acc, qk_raw_inline2397__tile_l0_a_1, qk_raw_inline2397__tile_l0_b_1, qk_raw_inline2397__tile_l0_ko == -128)
                    qk_raw_inline2397__tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 16384), pl.Mem.Acc] = pl.yield_(qk_raw_inline2397__tile_l0_c_acc_1)
                pl.tile.tpush_to_aiv(qk_raw_inline2397__tile, split=0)
                qk_h0_inline2393__ssa_v0_1: pl.Scalar[pl.INDEX] = 32
                qk_head_row_inline2326__ssa_v0_1: pl.Scalar[pl.INDEX] = qk_t_inline2411__ssa_v0 * 64 + qk_h0_inline2393__ssa_v0_1
                qk_q_tile_inline2338__tile_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_mat_25, pl.const(163840, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(q_flat_inline2355__ssa_v0, [qk_head_row_inline2326__ssa_v0_1, 0], [32, 512], [32, 512], target_memory=pl.Mem.Mat)
                qk_kv_inline2415__rv_v15_t_1: pl.Tile[[512, 128], pl.BF16, pl.MemRef(mem_mat_14, pl.const(32768, pl.INT64), 131072), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(qk_kv_inline2415__rv_v15)
                qk_raw_inline2397__tile_l0_init_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 16384), pl.Mem.Acc] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                for qk_raw_inline2397__tile_l0_ko_1, (qk_raw_inline2397__tile_l0_c_1,) in pl.range(0, 512, 256, init_values=(qk_raw_inline2397__tile_l0_init_1,)):
                    qk_raw_inline2397__tile_l0_a_2: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_left_27, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qk_q_tile_inline2338__tile_1, 0, qk_raw_inline2397__tile_l0_ko_1, [32, 128], target_memory=pl.Mem.Left)
                    qk_raw_inline2397__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_28, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(qk_kv_inline2415__rv_v15_t_1, qk_raw_inline2397__tile_l0_ko_1, 0, [128, 128], target_memory=pl.Mem.Right)
                    qk_raw_inline2397__tile_l0_a_3: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_left_29, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qk_q_tile_inline2338__tile_1, 0, qk_raw_inline2397__tile_l0_ko_1 + 128, [32, 128], target_memory=pl.Mem.Left)
                    qk_raw_inline2397__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_30, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(qk_kv_inline2415__rv_v15_t_1, qk_raw_inline2397__tile_l0_ko_1 + 128, 0, [128, 128], target_memory=pl.Mem.Right)
                    qk_raw_inline2397__tile_l0_c_acc_2: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 16384), pl.Mem.Acc] = pl.tile.matmul_acc(qk_raw_inline2397__tile_l0_c_1, qk_raw_inline2397__tile_l0_a_2, qk_raw_inline2397__tile_l0_b_2, qk_raw_inline2397__tile_l0_ko_1 == 0)
                    qk_raw_inline2397__tile_l0_c_acc_3: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 16384), pl.Mem.Acc] = pl.tile.matmul_acc(qk_raw_inline2397__tile_l0_c_acc_2, qk_raw_inline2397__tile_l0_a_3, qk_raw_inline2397__tile_l0_b_3, qk_raw_inline2397__tile_l0_ko_1 == -128)
                    qk_raw_inline2397__tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 16384), pl.Mem.Acc] = pl.yield_(qk_raw_inline2397__tile_l0_c_acc_3)
                pl.tile.tpush_to_aiv(qk_raw_inline2397__tile_1, split=0)
                qk_oi_inline2335__tile_l0_init: pl.Tile[[32, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create([32, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                qk_oi_inline2335__tile_l0_lmat: pl.Tile[[32, 128], pl.BF16, pl.Mem.Mat] = pl.tile.tpop_from_aiv(split=0)
                for qk_oi_inline2335__tile_l0_ko, (qk_oi_inline2335__tile_l0_c,) in pl.range(0, 128, 64, init_values=(qk_oi_inline2335__tile_l0_init,)):
                    qk_oi_inline2335__tile_l0_a: pl.Tile[[32, 32], pl.BF16, pl.MemRef(mem_left_27, pl.const(0, pl.INT64), 2048), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qk_oi_inline2335__tile_l0_lmat, 0, qk_oi_inline2335__tile_l0_ko, [32, 32], target_memory=pl.Mem.Left)
                    qk_oi_inline2335__tile_l0_b: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_28, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(qk_kv_inline2415__rv_v15, qk_oi_inline2335__tile_l0_ko, 0, [32, 512], target_memory=pl.Mem.Right)
                    qk_oi_inline2335__tile_l0_a_1: pl.Tile[[32, 32], pl.BF16, pl.MemRef(mem_left_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qk_oi_inline2335__tile_l0_lmat, 0, qk_oi_inline2335__tile_l0_ko + 32, [32, 32], target_memory=pl.Mem.Left)
                    qk_oi_inline2335__tile_l0_b_1: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_30, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(qk_kv_inline2415__rv_v15, qk_oi_inline2335__tile_l0_ko + 32, 0, [32, 512], target_memory=pl.Mem.Right)
                    qk_oi_inline2335__tile_l0_c_acc: pl.Tile[[32, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(qk_oi_inline2335__tile_l0_c, qk_oi_inline2335__tile_l0_a, qk_oi_inline2335__tile_l0_b, qk_oi_inline2335__tile_l0_ko == 0)
                    qk_oi_inline2335__tile_l0_c_acc_1: pl.Tile[[32, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(qk_oi_inline2335__tile_l0_c_acc, qk_oi_inline2335__tile_l0_a_1, qk_oi_inline2335__tile_l0_b_1, qk_oi_inline2335__tile_l0_ko == -32)
                    qk_oi_inline2335__tile: pl.Tile[[32, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(qk_oi_inline2335__tile_l0_c_acc_1)
                pl.system.tfree_to_aiv(qk_oi_inline2335__tile_l0_lmat, split=0)
                qk_h_idx_inline2320__ssa_v0: pl.Scalar[pl.INDEX] = 0
                qk_r0_inline2343__ssa_v0: pl.Scalar[pl.INDEX] = 0
                qk_blk_base_inline2319__ssa_v0: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2320__ssa_v0 * 80
                qk_row_inline2356__ssa_v0: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v0 + qk_sb_inline2374__ssa_v0 * 16
                t__tile_3: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.slice(qk_oi_inline2335__tile, [16, 512], [qk_r0_inline2343__ssa_v0, 0])
                sparse_blk_oi_inline2398__tile: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_3, [qk_row_inline2356__ssa_v0, 0], sparse_blk_oi_inline2398__iter_v1)
                qk_h_idx_inline2320__ssa_v1: pl.Scalar[pl.INDEX] = 1
                qk_r0_inline2343__ssa_v1: pl.Scalar[pl.INDEX] = 16
                qk_blk_base_inline2319__ssa_v1: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2320__ssa_v1 * 80
                qk_row_inline2356__ssa_v1: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v1 + qk_sb_inline2374__ssa_v0 * 16
                t__tile_4: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.slice(qk_oi_inline2335__tile, [16, 512], [qk_r0_inline2343__ssa_v1, 0])
                sparse_blk_oi_inline2398__tile_1: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_4, [qk_row_inline2356__ssa_v1, 0], sparse_blk_oi_inline2398__tile)
                qk_oi_inline2335__tile_l0_init_1: pl.Tile[[32, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create([32, 512], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                qk_oi_inline2335__tile_l0_lmat_1: pl.Tile[[32, 128], pl.BF16, pl.Mem.Mat] = pl.tile.tpop_from_aiv(split=0)
                for qk_oi_inline2335__tile_l0_ko_1, (qk_oi_inline2335__tile_l0_c_1,) in pl.range(0, 128, 64, init_values=(qk_oi_inline2335__tile_l0_init_1,)):
                    qk_oi_inline2335__tile_l0_a_2: pl.Tile[[32, 32], pl.BF16, pl.MemRef(mem_left_27, pl.const(0, pl.INT64), 2048), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qk_oi_inline2335__tile_l0_lmat_1, 0, qk_oi_inline2335__tile_l0_ko_1, [32, 32], target_memory=pl.Mem.Left)
                    qk_oi_inline2335__tile_l0_b_2: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_28, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(qk_kv_inline2415__rv_v15, qk_oi_inline2335__tile_l0_ko_1, 0, [32, 512], target_memory=pl.Mem.Right)
                    qk_oi_inline2335__tile_l0_a_3: pl.Tile[[32, 32], pl.BF16, pl.MemRef(mem_left_29, pl.const(8192, pl.INT64), 2048), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qk_oi_inline2335__tile_l0_lmat_1, 0, qk_oi_inline2335__tile_l0_ko_1 + 32, [32, 32], target_memory=pl.Mem.Left)
                    qk_oi_inline2335__tile_l0_b_3: pl.Tile[[32, 512], pl.BF16, pl.MemRef(mem_right_30, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(qk_kv_inline2415__rv_v15, qk_oi_inline2335__tile_l0_ko_1 + 32, 0, [32, 512], target_memory=pl.Mem.Right)
                    qk_oi_inline2335__tile_l0_c_acc_2: pl.Tile[[32, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(qk_oi_inline2335__tile_l0_c_1, qk_oi_inline2335__tile_l0_a_2, qk_oi_inline2335__tile_l0_b_2, qk_oi_inline2335__tile_l0_ko_1 == 0)
                    qk_oi_inline2335__tile_l0_c_acc_3: pl.Tile[[32, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(qk_oi_inline2335__tile_l0_c_acc_2, qk_oi_inline2335__tile_l0_a_3, qk_oi_inline2335__tile_l0_b_3, qk_oi_inline2335__tile_l0_ko_1 == -32)
                    qk_oi_inline2335__tile_1: pl.Tile[[32, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(qk_oi_inline2335__tile_l0_c_acc_3)
                pl.system.tfree_to_aiv(qk_oi_inline2335__tile_l0_lmat_1, split=0)
                qk_h_idx_inline2320__ssa_v0_1: pl.Scalar[pl.INDEX] = 2
                qk_r0_inline2343__ssa_v0_1: pl.Scalar[pl.INDEX] = 0
                qk_blk_base_inline2319__ssa_v0_1: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2320__ssa_v0_1 * 80
                qk_row_inline2356__ssa_v0_1: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v0_1 + qk_sb_inline2374__ssa_v0 * 16
                t__tile_5: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.slice(qk_oi_inline2335__tile_1, [16, 512], [qk_r0_inline2343__ssa_v0_1, 0])
                sparse_blk_oi_inline2398__tile_2: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_5, [qk_row_inline2356__ssa_v0_1, 0], sparse_blk_oi_inline2398__tile_1)
                qk_h_idx_inline2320__ssa_v1_1: pl.Scalar[pl.INDEX] = 3
                qk_r0_inline2343__ssa_v1_1: pl.Scalar[pl.INDEX] = 16
                qk_blk_base_inline2319__ssa_v1_1: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2320__ssa_v1_1 * 80
                qk_row_inline2356__ssa_v1_1: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v1_1 + qk_sb_inline2374__ssa_v0 * 16
                t__tile_6: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_acc_39, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.slice(qk_oi_inline2335__tile_1, [16, 512], [qk_r0_inline2343__ssa_v1_1, 0])
                sparse_blk_oi_inline2398__tile_3: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_6, [qk_row_inline2356__ssa_v1_1, 0], sparse_blk_oi_inline2398__tile_2)
                sparse_blk_li_inline2405__rv_v4: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_54", pl.const(0, pl.INT64), 0)] = sparse_blk_li_inline2405__iter_v1
                sparse_blk_mi_inline2404__rv_v4: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 0)] = sparse_blk_mi_inline2404__iter_v1
                sparse_blk_oi_inline2398__rv_v4: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 0)] = sparse_blk_oi_inline2398__tile_3
                sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7 = pl.yield_(sparse_blk_li_inline2405__rv_v4, sparse_blk_mi_inline2404__rv_v4)
            else:
                sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7 = pl.yield_(sparse_blk_li_inline2405__iter_v1, sparse_blk_mi_inline2404__iter_v1)
            sparse_blk_li_inline2405__rv_v2, sparse_blk_mi_inline2404__rv_v2, sparse_blk_oi_inline2398__rv_v2 = pl.yield_(sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7, sparse_blk_oi_inline2398__ssa_v0)
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qk_pv_aiv(qk_items_inline2347__ssa_v0: pl.Scalar[pl.INDEX], sparse_blk_li_inline2405__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], sparse_blk_mi_inline2404__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], sparse_blk_oi_inline2398__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], qk_order_inline2351__ssa_v0: pl.Tensor[[1280], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 5120)], sparse_bias_inline2381__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)], valid_block_mask_inline2385__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 5], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], position_ids_t1_inline1288__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)], ori_kv_flat_inline2344__ssa_v1: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)], cmp_sparse_indices_inline2383__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)], cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)], cmp_kv_flat_inline2401__ssa_v0: pl.Tensor[[cmp_block_num_inline2376__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)], q_flat_inline2355__ssa_v0: pl.Tensor[[t_heads_inline2364__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)], __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 4)]]) -> tuple[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32]]:
        pl.func_attr({"dual_aiv_dispatch": True})
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 128)
        mem_vec_27: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        subblock_idx: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        qk_pv_v2c_slot_buffer_import: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name='qk_pv_v2c_slot_buffer', peer_func='qk_pv_aic')
        qk_pv_c2v_slot_buffer: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name='qk_pv_c2v_slot_buffer', size=32768, base=0)
        pl.system.aiv_initialize_pipe(qk_pv_c2v_slot_buffer, qk_pv_v2c_slot_buffer_import, dir_mask=3, slot_size=16384, slot_num=2)
        if subblock_idx == 0:
            qk_core_inline2368__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            qk_lane_iters_inline2408__ssa_v0: pl.Scalar[pl.INDEX] = (qk_items_inline2347__ssa_v0 - qk_core_inline2368__ssa_v0 + 23) // 24
            for qk_it_inline2413__idx_v0, (sparse_blk_li_inline2405__iter_v1, sparse_blk_mi_inline2404__iter_v1, sparse_blk_oi_inline2398__iter_v1) in pl.range(qk_lane_iters_inline2408__ssa_v0, init_values=(sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0)):
                qk_flat_inline2365__ssa_v0: pl.Scalar[pl.INDEX] = qk_core_inline2368__ssa_v0 + qk_it_inline2413__idx_v0 * 24
                t__tile: pl.Scalar[pl.INT32] = pl.tensor.read(qk_order_inline2351__ssa_v0, [qk_flat_inline2365__ssa_v0])
                qk_item_inline2403__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(t__tile, pl.INDEX)
                qk_t_inline2411__ssa_v0: pl.Scalar[pl.INDEX] = qk_item_inline2403__ssa_v0 // 5
                qk_sb_inline2374__ssa_v0: pl.Scalar[pl.INDEX] = qk_item_inline2403__ssa_v0 - qk_t_inline2411__ssa_v0 * 5
                qk_token_base_inline2391__ssa_v0: pl.Scalar[pl.INDEX] = qk_t_inline2411__ssa_v0 * 320
                qk_s0_inline2418__ssa_v0: pl.Scalar[pl.INDEX] = qk_sb_inline2374__ssa_v0 * 128
                qk_bias_row_inline2419__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32768, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(sparse_bias_inline2381__rv_v2, [qk_t_inline2411__ssa_v0, qk_s0_inline2418__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec)
                qk_block_valid_inline2422__tile: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2385__ssa_v0, [qk_t_inline2411__ssa_v0, qk_sb_inline2374__ssa_v0])
                if 0 < pl.cast(qk_block_valid_inline2422__tile, pl.INDEX):
                    for qk_hb_inline2336__idx_v0, (sparse_blk_li_inline2405__iter_v3, sparse_blk_mi_inline2404__iter_v3, sparse_blk_oi_inline2398__iter_v3) in pl.range(2, init_values=(sparse_blk_li_inline2405__iter_v1, sparse_blk_mi_inline2404__iter_v1, sparse_blk_oi_inline2398__iter_v1)):
                        qk_raw_inline2397__tile_Vec: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
                        qk_scaled_inline2416__tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.muls(qk_raw_inline2397__tile_Vec, 0.044194173824159223)
                        pl.system.tfree_to_aic(qk_raw_inline2397__tile_Vec, split=0)
                        qk_scores_inline2325__tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_add(qk_scaled_inline2416__tile, qk_bias_row_inline2419__tile)
                        tmp_tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_17, pl.const(33280, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        qk_mi_inline2360__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49664, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_max(qk_scores_inline2325__tile, tmp_tile)
                        t__tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_sub(qk_scores_inline2325__tile, qk_mi_inline2360__tile)
                        qk_exp_inline2324__tile: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.exp(t__tile_1)
                        tmp_tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_17, pl.const(33280, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        qk_li_inline2323__tile: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(49792, pl.INT64), 128), pl.Mem.Vec] = pl.tile.row_sum(qk_exp_inline2324__tile, tmp_tile_1)
                        qk_exp_bf16_inline2322__tile: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(qk_exp_inline2324__tile, target_type=pl.BF16, mode='rint')
                        pl.tile.tpush_to_aic(qk_exp_bf16_inline2322__tile, split=0)
                        qk_h_idx_inline2320__ssa_v0: pl.Scalar[pl.INDEX] = qk_hb_inline2336__idx_v0 * 2
                        qk_r0_inline2343__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        qk_blk_base_inline2319__ssa_v0: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2320__ssa_v0 * 80
                        qk_row_inline2356__ssa_v0: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v0 + qk_sb_inline2374__ssa_v0 * 16
                        t__tile_2: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.slice(qk_mi_inline2360__tile, [16, 1], [qk_r0_inline2343__ssa_v0, 0])
                        sparse_blk_mi_inline2404__tile: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_2, [qk_row_inline2356__ssa_v0, 0], sparse_blk_mi_inline2404__iter_v3)
                        t__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(49792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.slice(qk_li_inline2323__tile, [16, 1], [qk_r0_inline2343__ssa_v0, 0])
                        sparse_blk_li_inline2405__tile: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_3, [qk_row_inline2356__ssa_v0, 0], sparse_blk_li_inline2405__iter_v3)
                        qk_h_idx_inline2320__ssa_v1: pl.Scalar[pl.INDEX] = qk_hb_inline2336__idx_v0 * 2 + 1
                        qk_r0_inline2343__ssa_v1: pl.Scalar[pl.INDEX] = 16
                        qk_blk_base_inline2319__ssa_v1: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2320__ssa_v1 * 80
                        qk_row_inline2356__ssa_v1: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v1 + qk_sb_inline2374__ssa_v0 * 16
                        t__tile_4: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49664, pl.INT64), 64), pl.Mem.Vec] = pl.tile.slice(qk_mi_inline2360__tile, [16, 1], [qk_r0_inline2343__ssa_v1, 0])
                        sparse_blk_mi_inline2404__tile_1: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_4, [qk_row_inline2356__ssa_v1, 0], sparse_blk_mi_inline2404__tile)
                        t__tile_5: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_22, pl.const(49792, pl.INT64), 64), pl.Mem.Vec] = pl.tile.slice(qk_li_inline2323__tile, [16, 1], [qk_r0_inline2343__ssa_v1, 0])
                        sparse_blk_li_inline2405__tile_1: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_5, [qk_row_inline2356__ssa_v1, 0], sparse_blk_li_inline2405__tile)
                        sparse_blk_li_inline2405__rv_v4, sparse_blk_mi_inline2404__rv_v4, sparse_blk_oi_inline2398__rv_v4 = pl.yield_(sparse_blk_li_inline2405__tile_1, sparse_blk_mi_inline2404__tile_1, sparse_blk_oi_inline2398__iter_v3)
                    sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7, sparse_blk_oi_inline2398__phi_v10 = pl.yield_(sparse_blk_li_inline2405__rv_v4, sparse_blk_mi_inline2404__rv_v4, sparse_blk_oi_inline2398__rv_v4)
                else:
                    qk_oi_zero_inline2318__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([16, 512], dtype=pl.FP32, value=0.0)
                    for qk_h_idx_inline2317__idx_v0, (sparse_blk_oi_inline2398__iter_v7,) in pl.range(4, init_values=(sparse_blk_oi_inline2398__iter_v1,)):
                        qk_blk_base_inline2319__ssa_v2: pl.Scalar[pl.INDEX] = qk_token_base_inline2391__ssa_v0 + qk_h_idx_inline2317__idx_v0 * 80
                        qk_row_inline2356__ssa_v2: pl.Scalar[pl.INDEX] = qk_blk_base_inline2319__ssa_v2 + qk_sb_inline2374__ssa_v0 * 16
                        sparse_blk_mi_inline2404__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_17, pl.const(33280, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=-3.0000000000000001e+38)
                        sparse_blk_mi_inline2404__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(33280, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(sparse_blk_mi_inline2404__tile_2, [16, 1])
                        pl.tile.store(sparse_blk_mi_inline2404__tile_3, [qk_row_inline2356__ssa_v2, 0], sparse_blk_mi_inline2404__iter_v1)
                        sparse_blk_li_inline2405__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_17, pl.const(33280, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                        sparse_blk_li_inline2405__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_17, pl.const(33280, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(sparse_blk_li_inline2405__tile_2, [16, 1])
                        pl.tile.store(sparse_blk_li_inline2405__tile_3, [qk_row_inline2356__ssa_v2, 0], sparse_blk_li_inline2405__iter_v1)
                        sparse_blk_oi_inline2398__tile: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(qk_oi_zero_inline2318__tile, [qk_row_inline2356__ssa_v2, 0], sparse_blk_oi_inline2398__iter_v7)
                        sparse_blk_oi_inline2398__rv_v8: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = pl.yield_(sparse_blk_oi_inline2398__tile)
                    sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7, sparse_blk_oi_inline2398__phi_v10 = pl.yield_(sparse_blk_li_inline2405__iter_v1, sparse_blk_mi_inline2404__iter_v1, sparse_blk_oi_inline2398__rv_v8)
                sparse_blk_li_inline2405__rv_v2, sparse_blk_mi_inline2404__rv_v2, sparse_blk_oi_inline2398__rv_v2 = pl.yield_(sparse_blk_li_inline2405__phi_v7, sparse_blk_mi_inline2404__phi_v7, sparse_blk_oi_inline2398__phi_v10)
            return sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0
        else:
            qk_core_inline2368__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            qk_lane_iters_inline2408__ssa_v0_1: pl.Scalar[pl.INDEX] = (qk_items_inline2347__ssa_v0 - qk_core_inline2368__ssa_v0_1 + 23) // 24
            for qk_it_inline2413__idx_v0_1, (sparse_blk_li_inline2405__iter_v1_1, sparse_blk_mi_inline2404__iter_v1_1, sparse_blk_oi_inline2398__iter_v1_1) in pl.range(qk_lane_iters_inline2408__ssa_v0_1, init_values=(sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0)):
                qk_flat_inline2365__ssa_v0_1: pl.Scalar[pl.INDEX] = qk_core_inline2368__ssa_v0_1 + qk_it_inline2413__idx_v0_1 * 24
                t__tile_6: pl.Scalar[pl.INT32] = pl.tensor.read(qk_order_inline2351__ssa_v0, [qk_flat_inline2365__ssa_v0_1])
                qk_item_inline2403__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.cast(t__tile_6, pl.INDEX)
                qk_t_inline2411__ssa_v0_1: pl.Scalar[pl.INDEX] = qk_item_inline2403__ssa_v0_1 // 5
                qk_sb_inline2374__ssa_v0_1: pl.Scalar[pl.INDEX] = qk_item_inline2403__ssa_v0_1 - qk_t_inline2411__ssa_v0_1 * 5
                qk_bias_row_inline2419__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_14, pl.const(32768, pl.INT64), 512), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([1, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                qk_block_valid_inline2422__tile_1: pl.Scalar[pl.INT32] = pl.tensor.read(valid_block_mask_inline2385__ssa_v0, [qk_t_inline2411__ssa_v0_1, qk_sb_inline2374__ssa_v0_1])
                if 0 < pl.cast(qk_block_valid_inline2422__tile_1, pl.INDEX):
                    for qk_hb_inline2336__idx_v0_1, (sparse_blk_li_inline2405__iter_v3_1, sparse_blk_mi_inline2404__iter_v3_1, sparse_blk_oi_inline2398__iter_v3_1) in pl.range(2, init_values=(sparse_blk_li_inline2405__iter_v1_1, sparse_blk_mi_inline2404__iter_v1_1, sparse_blk_oi_inline2398__iter_v1_1)):
                        qk_raw_inline2397__tile_Vec_1: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.tpop_from_aic(split=0)
                        qk_scaled_inline2416__tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.muls(qk_raw_inline2397__tile_Vec_1, 0.044194173824159223)
                        pl.system.tfree_to_aic(qk_raw_inline2397__tile_Vec_1, split=0)
                        qk_scores_inline2325__tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.col_expand_add(qk_scaled_inline2416__tile_1, qk_bias_row_inline2419__tile_1)
                        tmp_tile_2: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_17, pl.const(33280, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        qk_mi_inline2360__tile_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49664, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.row_max(qk_scores_inline2325__tile_1, tmp_tile_2)
                        t__tile_7: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.row_expand_sub(qk_scores_inline2325__tile_1, qk_mi_inline2360__tile_1)
                        qk_exp_inline2324__tile_1: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.exp(t__tile_7)
                        tmp_tile_3: pl.Tile[[32, 128], pl.FP32, pl.MemRef(mem_vec_17, pl.const(33280, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([32, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        qk_li_inline2323__tile_1: pl.Tile[[32, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(49664, pl.INT64), 128), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.row_sum(qk_exp_inline2324__tile_1, tmp_tile_3)
                        qk_exp_bf16_inline2322__tile_1: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.cast(qk_exp_inline2324__tile_1, target_type=pl.BF16, mode='rint')
                        pl.tile.tpush_to_aic(qk_exp_bf16_inline2322__tile_1, split=0)
                        t__tile_8: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([16, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        sparse_blk_mi_inline2404__tile_4: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_48", pl.const(0, pl.INT64), 0)] = sparse_blk_mi_inline2404__iter_v3_1
                        t__tile_9: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([16, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        sparse_blk_li_inline2405__tile_4: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_50", pl.const(0, pl.INT64), 0)] = sparse_blk_li_inline2405__iter_v3_1
                        t__tile_10: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([16, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        sparse_blk_mi_inline2404__tile_5: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_52", pl.const(0, pl.INT64), 0)] = sparse_blk_mi_inline2404__iter_v3_1
                        t__tile_11: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.create([16, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                        sparse_blk_li_inline2405__tile_5: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_54", pl.const(0, pl.INT64), 0)] = sparse_blk_li_inline2405__iter_v3_1
                        sparse_blk_li_inline2405__rv_v4_1, sparse_blk_mi_inline2404__rv_v4_1, sparse_blk_oi_inline2398__rv_v4_1 = pl.yield_(sparse_blk_li_inline2405__iter_v3_1, sparse_blk_mi_inline2404__iter_v3_1, sparse_blk_oi_inline2398__iter_v3_1)
                    sparse_blk_li_inline2405__phi_v7_1, sparse_blk_mi_inline2404__phi_v7_1, sparse_blk_oi_inline2398__phi_v10_1 = pl.yield_(sparse_blk_li_inline2405__rv_v4_1, sparse_blk_mi_inline2404__rv_v4_1, sparse_blk_oi_inline2398__rv_v4_1)
                else:
                    qk_oi_zero_inline2318__tile_1: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.full([16, 512], dtype=pl.FP32, value=0.0)
                    for qk_h_idx_inline2317__idx_v0_1, (sparse_blk_oi_inline2398__iter_v7_1,) in pl.range(4, init_values=(sparse_blk_oi_inline2398__iter_v1_1,)):
                        sparse_blk_mi_inline2404__tile_6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.full([1, 16], dtype=pl.FP32, value=-3.0000000000000001e+38)
                        sparse_blk_mi_inline2404__tile_7: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(sparse_blk_mi_inline2404__tile_6, [16, 1])
                        sparse_blk_li_inline2405__tile_6: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
                        sparse_blk_li_inline2405__tile_7: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_27, pl.const(49920, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[0, 0])] = pl.tile.reshape(sparse_blk_li_inline2405__tile_6, [16, 1])
                        sparse_blk_oi_inline2398__tile_1: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_61", pl.const(0, pl.INT64), 0)] = sparse_blk_oi_inline2398__iter_v7_1
                        sparse_blk_oi_inline2398__rv_v8_1: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.yield_(sparse_blk_oi_inline2398__iter_v7_1)
                    sparse_blk_li_inline2405__phi_v7_1, sparse_blk_mi_inline2404__phi_v7_1, sparse_blk_oi_inline2398__phi_v10_1 = pl.yield_(sparse_blk_li_inline2405__iter_v1_1, sparse_blk_mi_inline2404__iter_v1_1, sparse_blk_oi_inline2398__rv_v8_1)
                sparse_blk_li_inline2405__rv_v2_1, sparse_blk_mi_inline2404__rv_v2_1, sparse_blk_oi_inline2398__rv_v2_1 = pl.yield_(sparse_blk_li_inline2405__phi_v7_1, sparse_blk_mi_inline2404__phi_v7_1, sparse_blk_oi_inline2398__phi_v10_1)
            return sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0
    @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
    def qk_pv(qk_items_inline2347__ssa_v0: pl.Scalar[pl.INDEX], sparse_blk_li_inline2405__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], sparse_blk_mi_inline2404__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], sparse_blk_oi_inline2398__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], qk_order_inline2351__ssa_v0: pl.Tensor[[1280], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 5120)], sparse_bias_inline2381__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)], valid_block_mask_inline2385__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 5], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], position_ids_t1_inline1288__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)], ori_kv_flat_inline2344__ssa_v1: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)], cmp_sparse_indices_inline2383__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)], cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)], cmp_kv_flat_inline2401__ssa_v0: pl.Tensor[[cmp_block_num_inline2376__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)], q_flat_inline2355__ssa_v0: pl.Tensor[[t_heads_inline2364__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)], __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 4)]]) -> tuple[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32]]:
        self.qk_pv_aic(qk_items_inline2347__ssa_v0, sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0, qk_order_inline2351__ssa_v0, sparse_bias_inline2381__rv_v2, valid_block_mask_inline2385__ssa_v0, position_ids_t1_inline1288__ssa_v0, window_swa_indices__ssa_v0, ori_kv_flat_inline2344__ssa_v1, cmp_sparse_indices_inline2383__rv_v2, cmp_block_table__ssa_v0, cmp_kv_flat_inline2401__ssa_v0, q_flat_inline2355__ssa_v0, __gm_pipe_buffer, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
        self.qk_pv_aiv(qk_items_inline2347__ssa_v0, sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0, qk_order_inline2351__ssa_v0, sparse_bias_inline2381__rv_v2, valid_block_mask_inline2385__ssa_v0, position_ids_t1_inline1288__ssa_v0, window_swa_indices__ssa_v0, ori_kv_flat_inline2344__ssa_v1, cmp_sparse_indices_inline2383__rv_v2, cmp_block_table__ssa_v0, cmp_kv_flat_inline2401__ssa_v0, q_flat_inline2355__ssa_v0, __gm_pipe_buffer, attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout]})
        return sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def qk_pv_spmd(self, qk_items_inline2347__ssa_v0: pl.Scalar[pl.INDEX], sparse_blk_li_inline2405__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], sparse_blk_mi_inline2404__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], sparse_blk_oi_inline2398__ssa_v0: pl.Out[pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], qk_order_inline2351__ssa_v0: pl.Tensor[[1280], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 5120)], sparse_bias_inline2381__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)], valid_block_mask_inline2385__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 5], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], position_ids_t1_inline1288__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.INT32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)], window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)], ori_kv_flat_inline2344__ssa_v1: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)], cmp_sparse_indices_inline2383__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 0)], cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 0)], cmp_kv_flat_inline2401__ssa_v0: pl.Tensor[[cmp_block_num_inline2376__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)], q_flat_inline2355__ssa_v0: pl.Tensor[[t_heads_inline2364__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)], __gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 4)]]) -> tuple[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32]]:
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32]] = self.qk_pv(qk_items_inline2347__ssa_v0, sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0, qk_order_inline2351__ssa_v0, sparse_bias_inline2381__rv_v2, valid_block_mask_inline2385__ssa_v0, position_ids_t1_inline1288__ssa_v0, window_swa_indices__ssa_v0, ori_kv_flat_inline2344__ssa_v1, cmp_sparse_indices_inline2383__rv_v2, cmp_block_table__ssa_v0, cmp_kv_flat_inline2401__ssa_v0, q_flat_inline2355__ssa_v0, __gm_pipe_buffer, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
        sparse_blk_li_inline2405__rv_v2: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        sparse_blk_mi_inline2404__rv_v2: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        sparse_blk_oi_inline2398__rv_v2: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
        return sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qproj_dequant_rms_nope_rope(out_tg_inline1826__ssa_v0: pl.Scalar[pl.INDEX], q_flat_inline1856__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1799__idx_v0: pl.Scalar[pl.INDEX], qr_scale_pad_store_inline1814__ssa_v1: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], q_cos_il_inline1311__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], q_sin_signed_inline1295__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], q_swap_idx_inline1313__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)], q_proj_i32_inline1835__rv_v5: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)]):
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_18: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_19: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_29: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_30: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_39: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_40: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_50: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_51: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        hg_idx_inline1857__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        hg_inline1858__ssa_v0: pl.Scalar[pl.INDEX] = hg_idx_inline1857__ssa_v0 * 4
        for tg_inline1844__idx_v0, (out_tg_inline1826__iter_v1, q_flat_inline1856__iter_v1) in pl.range(0, tile_rows_inline1798__ssa_v0, 8, init_values=(out_tg_inline1826__ssa_v0, q_flat_inline1856__ssa_v0)):
            out_tg_inline1826__ssa_v3: pl.Scalar[pl.INDEX] = tile_base_inline1799__idx_v0 + tg_inline1844__idx_v0
            if tg_inline1844__idx_v0 + 8 <= tile_rows_inline1798__ssa_v0:
                qr_scale_dq_t_inline1860__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(101376, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(qr_scale_pad_store_inline1814__ssa_v1, [tg_inline1844__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                q_cos_il_inline1849__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(q_cos_il_inline1311__ssa_v0, [out_tg_inline1826__ssa_v3, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                q_sin_signed_inline1862__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(q_sin_signed_inline1295__ssa_v0, [out_tg_inline1826__ssa_v3, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                q_swap_idx_inline1788__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(q_swap_idx_inline1313__ssa_v0, [out_tg_inline1826__ssa_v3, 0], [8, 64], [8, 64], target_memory=pl.Mem.Vec)
                for h_inner_inline1863__idx_v0, (q_flat_inline1856__iter_v3,) in pl.range(0, 4, 2, init_values=(q_flat_inline1856__iter_v1,)):
                    h_inline1768__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline1858__ssa_v0 + h_inner_inline1863__idx_v0
                    h0_inline1800__ssa_v0: pl.Scalar[pl.INDEX] = h_inline1768__ssa_v0 * 512
                    h_inline1768__ssa_v0_1: pl.Scalar[pl.INDEX] = hg_inline1858__ssa_v0 + (h_inner_inline1863__idx_v0 + 1)
                    h0_inline1800__ssa_v0_1: pl.Scalar[pl.INDEX] = h_inline1768__ssa_v0_1 * 512
                    q_head_acc_inline1865__tile: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(q_proj_i32_inline1835__rv_v5, [tg_inline1844__idx_v0, h0_inline1800__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    t__tile: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(wq_b_scale__ssa_v0, [h0_inline1800__ssa_v0], [512], [512], target_memory=pl.Mem.Vec)
                    q_head_acc_inline1865__tile_1: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(q_proj_i32_inline1835__rv_v5, [tg_inline1844__idx_v0, h0_inline1800__ssa_v0_1], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    t__tile_1: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(wq_b_scale__ssa_v0, [h0_inline1800__ssa_v0_1], [512], [512], target_memory=pl.Mem.Vec)
                    q_head_scale_inline1843__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = t__tile
                    q_head_acc_fp32_inline1866__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(q_head_acc_inline1865__tile, target_type=pl.FP32, mode='none')
                    q_head_row_scaled_inline1867__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(q_head_acc_fp32_inline1866__tile, qr_scale_dq_t_inline1860__tile)
                    q_head_dq_inline1825__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(q_head_row_scaled_inline1867__tile, q_head_scale_inline1843__tile)
                    q_head_sq_inline1868__tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(q_head_dq_inline1825__tile, q_head_dq_inline1825__tile)
                    tmp_tile: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline1887__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_inline1868__tile, tmp_tile)
                    q_head_sq_sum_inline1870__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_row_inline1887__tile, [1, 8])
                    q_head_sq_mean_inline1846__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(q_head_sq_sum_inline1870__tile, 0.001953125)
                    q_head_var_inline1864__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(q_head_sq_mean_inline1846__tile, 9.9999999999999995e-07)
                    rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline1871__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(q_head_var_inline1864__tile, rsqrt_tmp)
                    q_head_inv_rms_t_inline1771__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_inv_rms_inline1871__tile, [8, 1])
                    t__tile_2: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_inline1825__tile, [8, 448], [0, 0])
                    q_nope_normed_inline1872__tile: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_2, q_head_inv_rms_t_inline1771__tile)
                    q_nope_bf16_inline1873__tile: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(q_nope_normed_inline1872__tile, target_type=pl.BF16, mode='rint')
                    q_rope_chunk_raw_inline1837__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(3840, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(q_head_dq_inline1825__tile, [8, 64], [0, 448])
                    q_rope_chunk_inline1811__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(q_rope_chunk_raw_inline1837__tile, q_head_inv_rms_t_inline1771__tile)
                    gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                        gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(q_rope_chunk_inline1811__tile, [1, 64], [gather_lv, 0], [1, 64])
                        gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(q_swap_idx_inline1788__tile, [1, 64], [gather_lv, 0], [1, 64])
                        gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_29, pl.const(67584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                        gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_30, pl.const(67840, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                        gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                        gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
                    q_rope_swapped_inline1778__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
                    q_rope_base_inline1838__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(q_rope_chunk_inline1811__tile, q_cos_il_inline1849__tile)
                    q_rope_delta_inline1874__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(q_rope_swapped_inline1778__tile, q_sin_signed_inline1862__tile)
                    q_rope_rot_inline1875__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(q_rope_base_inline1838__tile, q_rope_delta_inline1874__tile)
                    q_rope_bf16_inline1833__tile: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(q_rope_rot_inline1875__tile, target_type=pl.BF16, mode='rint')
                    q_flat_inline1856__tile: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(q_nope_bf16_inline1873__tile, [out_tg_inline1826__ssa_v3, h0_inline1800__ssa_v0], q_flat_inline1856__iter_v3)
                    q_flat_inline1856__tile_1: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(q_rope_bf16_inline1833__tile, [out_tg_inline1826__ssa_v3, h0_inline1800__ssa_v0 + 448], q_flat_inline1856__tile)
                    q_head_scale_inline1843__tile_1: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec] = t__tile_1
                    q_head_acc_fp32_inline1866__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(q_head_acc_inline1865__tile_1, target_type=pl.FP32, mode='none')
                    q_head_row_scaled_inline1867__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(q_head_acc_fp32_inline1866__tile_1, qr_scale_dq_t_inline1860__tile)
                    q_head_dq_inline1825__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(q_head_row_scaled_inline1867__tile_1, q_head_scale_inline1843__tile_1)
                    q_head_sq_inline1868__tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(q_head_dq_inline1825__tile_1, q_head_dq_inline1825__tile_1)
                    tmp_tile_1: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_sq_row_inline1887__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_inline1868__tile_1, tmp_tile_1)
                    q_head_sq_sum_inline1870__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_row_inline1887__tile_1, [1, 8])
                    q_head_sq_mean_inline1846__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(q_head_sq_sum_inline1870__tile_1, 0.001953125)
                    q_head_var_inline1864__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(q_head_sq_mean_inline1846__tile_1, 9.9999999999999995e-07)
                    rsqrt_tmp_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    q_head_inv_rms_inline1871__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(q_head_var_inline1864__tile_1, rsqrt_tmp_1)
                    q_head_inv_rms_t_inline1771__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_inv_rms_inline1871__tile_1, [8, 1])
                    t__tile_3: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_inline1825__tile_1, [8, 448], [0, 0])
                    q_nope_normed_inline1872__tile_1: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_3, q_head_inv_rms_t_inline1771__tile_1)
                    q_nope_bf16_inline1873__tile_1: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(q_nope_normed_inline1872__tile_1, target_type=pl.BF16, mode='rint')
                    q_rope_chunk_raw_inline1837__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(q_head_dq_inline1825__tile_1, [8, 64], [0, 448])
                    q_rope_chunk_inline1811__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(q_rope_chunk_raw_inline1837__tile_1, q_head_inv_rms_t_inline1771__tile_1)
                    gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                    for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                        gather_inp_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(q_rope_chunk_inline1811__tile_1, [1, 64], [gather_lv_1, 0], [1, 64])
                        gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(q_swap_idx_inline1788__tile, [1, 64], [gather_lv_1, 0], [1, 64])
                        gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_50, pl.const(100864, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                        gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_51, pl.const(101120, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                        gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                        gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
                    q_rope_swapped_inline1778__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
                    q_rope_base_inline1838__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(q_rope_chunk_inline1811__tile_1, q_cos_il_inline1849__tile)
                    q_rope_delta_inline1874__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(q_rope_swapped_inline1778__tile_1, q_sin_signed_inline1862__tile)
                    q_rope_rot_inline1875__tile_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(q_rope_base_inline1838__tile_1, q_rope_delta_inline1874__tile_1)
                    q_rope_bf16_inline1833__tile_1: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(q_rope_rot_inline1875__tile_1, target_type=pl.BF16, mode='rint')
                    q_flat_inline1856__tile_2: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(q_nope_bf16_inline1873__tile_1, [out_tg_inline1826__ssa_v3, h0_inline1800__ssa_v0_1], q_flat_inline1856__tile_1)
                    q_flat_inline1856__tile_3: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(q_rope_bf16_inline1833__tile_1, [out_tg_inline1826__ssa_v3, h0_inline1800__ssa_v0_1 + 448], q_flat_inline1856__tile_2)
                    q_flat_inline1856__rv_v4: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline1856__tile_3)
                q_flat_inline1856__phi_v7: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_103", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline1856__rv_v4)
            else:
                valid_tail_rows_inline1859__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline1798__ssa_v0 - tg_inline1844__idx_v0
                qr_scale_dq_tail_inline1876__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(103456, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(qr_scale_pad_store_inline1814__ssa_v1, [tg_inline1844__idx_v0, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
                q_cos_il_tail_inline1880__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_19, pl.const(51200, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1859__ssa_v0, 64])] = pl.tile.load(q_cos_il_inline1311__ssa_v0, [out_tg_inline1826__ssa_v3, 0], [8, 64], [valid_tail_rows_inline1859__ssa_v0, 64], target_memory=pl.Mem.Vec)
                q_sin_signed_tail_inline1882__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_39, pl.const(68096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1859__ssa_v0, 64])] = pl.tile.load(q_sin_signed_inline1295__ssa_v0, [out_tg_inline1826__ssa_v3, 0], [8, 64], [valid_tail_rows_inline1859__ssa_v0, 64], target_memory=pl.Mem.Vec)
                t__tmp_v140: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
                t__tmp_v141: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
                t__tmp_v142: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tmp_v141, target_type=pl.FP32, mode='round')
                q_col_inline1883__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v140, t__tmp_v142)
                t__tmp_v143: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_col_inline1883__ssa_v0, 0.5)
                t__tmp_v144: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v143, target_type=pl.INT32, mode='trunc')
                q_dup_f_inline1884__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v144, target_type=pl.FP32, mode='round')
                t__tmp_v145: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_dup_f_inline1884__ssa_v0, 2.0)
                q_lane_inline1869__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(q_col_inline1883__ssa_v0, t__tmp_v145)
                t__tmp_v146: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.adds(q_col_inline1883__ssa_v0, 1.0)
                t__tmp_v147: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(q_lane_inline1869__ssa_v0, 2.0)
                q_swap_f_inline1803__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(t__tmp_v146, t__tmp_v147)
                t__tmp_v148: pl.Tile[[1, 8], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 8], dtype=pl.INT32, descending=False)
                t__tmp_v149: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 32), pl.Mem.Vec] = pl.tile.cast(t__tmp_v148, target_type=pl.FP32, mode='round')
                q_row_seed_inline1759__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__tmp_v149, 64.0)
                t__tmp_v150: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([64, 8], dtype=pl.FP32, value=1.0)
                q_row_grid_inline1820__ssa_v0: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tmp_v150, q_row_seed_inline1759__ssa_v0)
                transpose_tmp: pl.Tile[[64, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([64, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                q_row_offset_inline1885__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.transpose(q_row_grid_inline1820__ssa_v0, 0, 1, transpose_tmp)
                t__tmp_v151: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(q_swap_f_inline1803__ssa_v0, q_row_offset_inline1885__ssa_v0)
                q_swap_idx_tail_inline1877__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_40, pl.const(84480, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tmp_v151, target_type=pl.INT32, mode='round')
                q_head_reduce_tmp_inline1760__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_11, pl.const(2048, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.create([8, 512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                q_gather_tmp_inline1888__ssa_v0: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_8, pl.const(101408, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for h_inner_tail_inline1773__idx_v0 in pl.range(4):
                    h_tail_inline1789__ssa_v0: pl.Scalar[pl.INDEX] = hg_inline1858__ssa_v0 + h_inner_tail_inline1773__idx_v0
                    h0_tail_inline1881__ssa_v0: pl.Scalar[pl.INDEX] = h_tail_inline1789__ssa_v0 * 512
                    q_head_acc_tail_inline1839__ssa_v0: pl.Tile[[8, 512], pl.INT32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(q_proj_i32_inline1835__rv_v5, [tg_inline1844__idx_v0, h0_tail_inline1881__ssa_v0], [8, 512], [8, 512], target_memory=pl.Mem.Vec)
                    q_head_scale_input_tail_inline1840__ssa_v0: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(wq_b_scale__ssa_v0, [h0_tail_inline1881__ssa_v0], [512], [512], target_memory=pl.Mem.Vec)
                    q_head_scale_tail_inline1816__ssa_v0: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = q_head_scale_input_tail_inline1840__ssa_v0
                    q_head_acc_fp32_tail_inline1755__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(q_head_acc_tail_inline1839__ssa_v0, target_type=pl.FP32, mode='none')
                    q_head_row_scaled_tail_inline1754__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(q_head_acc_fp32_tail_inline1755__ssa_v0, qr_scale_dq_tail_inline1876__ssa_v0)
                    q_head_dq_tail_inline1806__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.col_expand_mul(q_head_row_scaled_tail_inline1754__ssa_v0, q_head_scale_tail_inline1816__ssa_v0)
                    q_head_sq_tail_inline1753__ssa_v0: pl.Tile[[8, 512], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.mul(q_head_dq_tail_inline1806__ssa_v0, q_head_dq_tail_inline1806__ssa_v0)
                    q_head_sq_sum_tail_inline1752__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(q_head_sq_tail_inline1753__ssa_v0, q_head_reduce_tmp_inline1760__ssa_v0)
                    t__rm_a0_tmp_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_sq_sum_tail_inline1752__ssa_v0, [1, 8])
                    t__row_major_tmp_v1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(t__rm_a0_tmp_v0, 0.001953125)
                    t__tmp_v152: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [8, 1])
                    t__rm_a0_tmp_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v152, [1, 8])
                    t__row_major_tmp_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__rm_a0_tmp_v2, 9.9999999999999995e-07)
                    t__tmp_v153: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v3, [8, 1])
                    t__rm_a0_tmp_v4: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v153, [1, 8])
                    t__row_major_tmp_v5: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v4)
                    t__tmp_v154: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v5, [8, 1])
                    q_head_inv_rms_tail_inline1783__rm_a0_tmp_v6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tmp_v154, [1, 8])
                    q_head_inv_rms_tail_inline1783__row_major_tmp_v7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(q_head_inv_rms_tail_inline1783__rm_a0_tmp_v6)
                    q_head_inv_rms_tail_inline1783__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(q_head_inv_rms_tail_inline1783__row_major_tmp_v7, [8, 1])
                    t__tmp_v155: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 16128), pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail_inline1806__ssa_v0, [8, 448], [0, 0])
                    q_nope_normed_tail_inline1751__ssa_v0: pl.Tile[[8, 448], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 14336), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tmp_v155, q_head_inv_rms_tail_inline1783__ssa_v0)
                    q_nope_bf16_tail_inline1750__ssa_v0: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec] = pl.tile.cast(q_nope_normed_tail_inline1751__ssa_v0, target_type=pl.BF16, mode='rint')
                    q_nope_valid_inline1749__ssa_v0: pl.Tile[[8, 448], pl.BF16, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 7168), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1859__ssa_v0, 448])] = pl.tile.set_validshape(q_nope_bf16_tail_inline1750__ssa_v0, valid_tail_rows_inline1859__ssa_v0, 448)
                    pl.tile.store(q_nope_valid_inline1749__ssa_v0, [out_tg_inline1826__ssa_v3, h0_tail_inline1881__ssa_v0], q_flat_inline1856__iter_v1)
                    q_rope_chunk_raw_tail_inline1748__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(20224, pl.INT64), 14592), pl.Mem.Vec] = pl.tile.slice(q_head_dq_tail_inline1806__ssa_v0, [8, 64], [0, 448])
                    q_rope_chunk_tail_inline1812__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.row_expand_mul(q_rope_chunk_raw_tail_inline1748__ssa_v0, q_head_inv_rms_tail_inline1783__ssa_v0)
                    q_rope_swapped_tail_inline1747__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.gather(q_rope_chunk_tail_inline1812__ssa_v0, q_swap_idx_tail_inline1877__ssa_v0, q_gather_tmp_inline1888__ssa_v0)
                    q_rope_base_tail_inline1746__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(q_rope_chunk_tail_inline1812__ssa_v0, q_cos_il_tail_inline1880__ssa_v0)
                    q_rope_delta_tail_inline1745__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_18, pl.const(34816, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(q_rope_swapped_tail_inline1747__ssa_v0, q_sin_signed_tail_inline1882__ssa_v0)
                    q_rope_rot_tail_inline1854__ssa_v0: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.add(q_rope_base_tail_inline1746__ssa_v0, q_rope_delta_tail_inline1745__ssa_v0)
                    q_rope_bf16_tail_inline1744__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(q_rope_rot_tail_inline1854__ssa_v0, target_type=pl.BF16, mode='rint')
                    q_rope_valid_inline1830__ssa_v0: pl.Tile[[8, 64], pl.BF16, pl.MemRef(mem_vec_13, pl.const(18432, pl.INT64), 1024), pl.Mem.Vec, pl.TileView(valid_shape=[valid_tail_rows_inline1859__ssa_v0, 64])] = pl.tile.set_validshape(q_rope_bf16_tail_inline1744__ssa_v0, valid_tail_rows_inline1859__ssa_v0, 64)
                    pl.tile.store(q_rope_valid_inline1830__ssa_v0, [out_tg_inline1826__ssa_v3, h0_tail_inline1881__ssa_v0 + 448], q_flat_inline1856__iter_v1)
                q_flat_inline1856__phi_v7: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_103", pl.const(0, pl.INT64), 0)] = pl.yield_(q_flat_inline1856__iter_v1)
            out_tg_inline1826__rv_v2, q_flat_inline1856__rv_v2 = pl.yield_(out_tg_inline1826__ssa_v3, q_flat_inline1856__phi_v7)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_dequant_rms_nope_rope_spmd(self, out_tg_inline1826__ssa_v0: pl.Scalar[pl.INDEX], q_flat_inline1856__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1799__idx_v0: pl.Scalar[pl.INDEX], qr_scale_pad_store_inline1814__ssa_v1: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], q_cos_il_inline1311__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], q_sin_signed_inline1295__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], q_swap_idx_inline1313__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)], q_proj_i32_inline1835__rv_v5: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 131072)]):
        self.qproj_dequant_rms_nope_rope(out_tg_inline1826__ssa_v0, q_flat_inline1856__ssa_v0, tile_rows_inline1798__ssa_v0, tile_base_inline1799__idx_v0, qr_scale_pad_store_inline1814__ssa_v1, q_cos_il_inline1311__ssa_v0, q_sin_signed_inline1295__ssa_v0, q_swap_idx_inline1313__ssa_v0, q_proj_i32_inline1835__rv_v5, wq_b_scale__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qproj_matmul(q_proj_i32_inline1835__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], qproj_full_rows_inline1804__ssa_v0: pl.Scalar[pl.INDEX], qr_i8_matmul_inline1787__rv_v2: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)], qproj_t_matmul_inline1791__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32]:
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_9: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_10: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_11: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_12: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_21: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_left_23: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        qproj_n_idx_inline1841__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        w_col0_inline1842__ssa_v0: pl.Scalar[pl.INDEX] = qproj_n_idx_inline1841__ssa_v0 * 512
        for t0_inline1861__idx_v0, (q_proj_i32_inline1835__iter_v1,) in pl.range(0, qproj_full_rows_inline1804__ssa_v0, 64, init_values=(q_proj_i32_inline1835__ssa_v0,)):
            col_acc_inline1847__tile: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create([64, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc)
            for qr_proj_col0_inline1848__idx_v0, (col_acc_inline1847__iter_v1,) in pl.range(0, 1024, 256, init_values=(col_acc_inline1847__tile,)):
                qr_i8_chunk_inline1879__tile: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(qr_i8_matmul_inline1787__rv_v2, [t0_inline1861__idx_v0, qr_proj_col0_inline1848__idx_v0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
                wq_chunk_inline1845__tile: pl.Tile[[128, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(8192, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wq_b__ssa_v0, [qr_proj_col0_inline1848__idx_v0, w_col0_inline1842__ssa_v0], [128, 512], [128, 512], target_memory=pl.Mem.Mat)
                qr_i8_chunk_inline1879__tile_1: pl.Tile[[64, 128], pl.INT8, pl.MemRef(mem_mat_6, pl.const(73728, pl.INT64), 8192), pl.Mem.Mat] = pl.tile.load(qr_i8_matmul_inline1787__rv_v2, [t0_inline1861__idx_v0, qr_proj_col0_inline1848__idx_v0 + 128], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
                wq_chunk_inline1845__tile_1: pl.Tile[[128, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wq_b__ssa_v0, [qr_proj_col0_inline1848__idx_v0 + 128, w_col0_inline1842__ssa_v0], [128, 512], [128, 512], target_memory=pl.Mem.Mat)
                if qr_proj_col0_inline1848__idx_v0 == 0:
                    col_acc_inline1847__tile_l0_init: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.create([64, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc)
                    col_acc_inline1847__tile_l0_a: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_9, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qr_i8_chunk_inline1879__tile, 0, 0, [64, 64], target_memory=pl.Mem.Left)
                    col_acc_inline1847__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_chunk_inline1845__tile, 0, 0, [64, 512], target_memory=pl.Mem.Right)
                    col_acc_inline1847__tile_l0_a_1: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qr_i8_chunk_inline1879__tile, 0, 64, [64, 64], target_memory=pl.Mem.Left)
                    col_acc_inline1847__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_chunk_inline1845__tile, 64, 0, [64, 512], target_memory=pl.Mem.Right)
                    col_acc_inline1847__tile_l0_c_acc: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(col_acc_inline1847__tile_l0_init, col_acc_inline1847__tile_l0_a, col_acc_inline1847__tile_l0_b, True)
                    col_acc_inline1847__tile_l0_c_acc_1: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(col_acc_inline1847__tile_l0_c_acc, col_acc_inline1847__tile_l0_a_1, col_acc_inline1847__tile_l0_b_1, False)
                    col_acc_inline1847__phi_v5: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline1847__tile_l0_c_acc_1)
                else:
                    col_acc_inline1847__tile_l0_a_2: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_9, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qr_i8_chunk_inline1879__tile, 0, 0, [64, 64], target_memory=pl.Mem.Left)
                    col_acc_inline1847__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_chunk_inline1845__tile, 0, 0, [64, 512], target_memory=pl.Mem.Right)
                    col_acc_inline1847__tile_l0_a_3: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qr_i8_chunk_inline1879__tile, 0, 64, [64, 64], target_memory=pl.Mem.Left)
                    col_acc_inline1847__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_chunk_inline1845__tile, 64, 0, [64, 512], target_memory=pl.Mem.Right)
                    col_acc_inline1847__tile_l0_c_acc_2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(col_acc_inline1847__iter_v1, col_acc_inline1847__tile_l0_a_2, col_acc_inline1847__tile_l0_b_2)
                    col_acc_inline1847__tile_l0_c_acc_3: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(col_acc_inline1847__tile_l0_c_acc_2, col_acc_inline1847__tile_l0_a_3, col_acc_inline1847__tile_l0_b_3)
                    col_acc_inline1847__phi_v5: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline1847__tile_l0_c_acc_3)
                col_acc_inline1847__tile_l0_a_4: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_21, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qr_i8_chunk_inline1879__tile_1, 0, 0, [64, 64], target_memory=pl.Mem.Left)
                col_acc_inline1847__tile_l0_b_4: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_chunk_inline1845__tile_1, 0, 0, [64, 512], target_memory=pl.Mem.Right)
                col_acc_inline1847__tile_l0_a_5: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_left_23, pl.const(8192, pl.INT64), 4096), pl.Mem.Left, pl.TileView(blayout=pl.TileLayout.row_major)] = pl.tile.extract(qr_i8_chunk_inline1879__tile_1, 0, 64, [64, 64], target_memory=pl.Mem.Left)
                col_acc_inline1847__tile_l0_b_5: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_chunk_inline1845__tile_1, 64, 0, [64, 512], target_memory=pl.Mem.Right)
                col_acc_inline1847__tile_l0_c_acc_4: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(col_acc_inline1847__phi_v5, col_acc_inline1847__tile_l0_a_4, col_acc_inline1847__tile_l0_b_4)
                col_acc_inline1847__tile_l0_c_acc_5: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(col_acc_inline1847__tile_l0_c_acc_4, col_acc_inline1847__tile_l0_a_5, col_acc_inline1847__tile_l0_b_5)
                col_acc_inline1847__rv_v2: pl.Tile[[64, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.yield_(col_acc_inline1847__tile_l0_c_acc_5)
            q_proj_i32_inline1835__tile: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(col_acc_inline1847__rv_v2, [t0_inline1861__idx_v0, w_col0_inline1842__ssa_v0], q_proj_i32_inline1835__iter_v1)
            q_proj_i32_inline1835__rv_v2: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = pl.yield_(q_proj_i32_inline1835__tile)
        tail_w_col0_inline1809__ssa_v0: pl.Scalar[pl.INDEX] = w_col0_inline1842__ssa_v0
        for tail_t0_inline1850__idx_v0, (q_proj_i32_inline1835__iter_v4,) in pl.range(qproj_full_rows_inline1804__ssa_v0, qproj_t_matmul_inline1791__ssa_v0, 16, init_values=(q_proj_i32_inline1835__rv_v2,)):
            qproj_tail_rows_inline1851__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1798__ssa_v0 - tail_t0_inline1850__idx_v0, 16)
            tail_acc_inline1801__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.create([16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc)
            for tail_qr_col0_inline1852__idx_v0, (tail_acc_inline1801__iter_v1,) in pl.range(0, 1024, 256, init_values=(tail_acc_inline1801__tile,)):
                qr_i8_tail_inline1763__tile: pl.Tile[[16, 128], pl.INT8, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 2048), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 128])] = pl.tile.load(qr_i8_matmul_inline1787__rv_v2, [tail_t0_inline1850__idx_v0, tail_qr_col0_inline1852__idx_v0], [16, 128], [qproj_tail_rows_inline1851__ssa_v0, 128], target_memory=pl.Mem.Mat)
                wq_tail_inline1853__tile: pl.Tile[[128, 512], pl.INT8, pl.MemRef(mem_mat_5, pl.const(8192, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wq_b__ssa_v0, [tail_qr_col0_inline1852__idx_v0, tail_w_col0_inline1809__ssa_v0], [128, 512], [128, 512], target_memory=pl.Mem.Mat)
                qr_i8_tail_inline1763__tile_1: pl.Tile[[16, 128], pl.INT8, pl.MemRef(mem_mat_6, pl.const(73728, pl.INT64), 2048), pl.Mem.Mat, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 128])] = pl.tile.load(qr_i8_matmul_inline1787__rv_v2, [tail_t0_inline1850__idx_v0, tail_qr_col0_inline1852__idx_v0 + 128], [16, 128], [qproj_tail_rows_inline1851__ssa_v0, 128], target_memory=pl.Mem.Mat)
                wq_tail_inline1853__tile_1: pl.Tile[[128, 512], pl.INT8, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wq_b__ssa_v0, [tail_qr_col0_inline1852__idx_v0 + 128, tail_w_col0_inline1809__ssa_v0], [128, 512], [128, 512], target_memory=pl.Mem.Mat)
                if tail_qr_col0_inline1852__idx_v0 == 0:
                    tail_acc_inline1801__tile_l0_init_storage: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([16, 512], dtype=pl.INT32, target_memory=pl.Mem.Acc, compact=True)
                    tail_acc_inline1801__tile_l0_init: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(tail_acc_inline1801__tile_l0_init_storage, qproj_tail_rows_inline1851__ssa_v0, 512)
                    tail_acc_inline1801__tile_l0_a: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_9, pl.const(12288, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_i8_tail_inline1763__tile, 0, 0, [16, 64], target_memory=pl.Mem.Left)
                    tail_acc_inline1801__tile_l0_b: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tail_inline1853__tile, 0, 0, [64, 512], target_memory=pl.Mem.Right)
                    tail_acc_inline1801__tile_l0_a_1: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_i8_tail_inline1763__tile, 0, 64, [16, 64], target_memory=pl.Mem.Left)
                    tail_acc_inline1801__tile_l0_b_1: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tail_inline1853__tile, 64, 0, [64, 512], target_memory=pl.Mem.Right)
                    tail_acc_inline1801__tile_l0_c_acc: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(tail_acc_inline1801__tile_l0_init, tail_acc_inline1801__tile_l0_a, tail_acc_inline1801__tile_l0_b, True)
                    tail_acc_inline1801__tile_l0_c_acc_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(tail_acc_inline1801__tile_l0_c_acc, tail_acc_inline1801__tile_l0_a_1, tail_acc_inline1801__tile_l0_b_1, False)
                    tail_acc_inline1801__phi_v5: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.yield_(tail_acc_inline1801__tile_l0_c_acc_1)
                else:
                    tail_acc_inline1801__tile_l0_a_2: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_9, pl.const(12288, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_i8_tail_inline1763__tile, 0, 0, [16, 64], target_memory=pl.Mem.Left)
                    tail_acc_inline1801__tile_l0_b_2: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tail_inline1853__tile, 0, 0, [64, 512], target_memory=pl.Mem.Right)
                    tail_acc_inline1801__tile_l0_a_3: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_i8_tail_inline1763__tile, 0, 64, [16, 64], target_memory=pl.Mem.Left)
                    tail_acc_inline1801__tile_l0_b_3: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tail_inline1853__tile, 64, 0, [64, 512], target_memory=pl.Mem.Right)
                    tail_acc_inline1801__tile_l0_c_acc_2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(tail_acc_inline1801__iter_v1, tail_acc_inline1801__tile_l0_a_2, tail_acc_inline1801__tile_l0_b_2)
                    tail_acc_inline1801__tile_l0_c_acc_3: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(tail_acc_inline1801__tile_l0_c_acc_2, tail_acc_inline1801__tile_l0_a_3, tail_acc_inline1801__tile_l0_b_3)
                    tail_acc_inline1801__phi_v5: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 512], compact=pl.CompactMode.normal)] = pl.yield_(tail_acc_inline1801__tile_l0_c_acc_3)
                tail_acc_inline1801__tile_l0_a_4: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_21, pl.const(4096, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_i8_tail_inline1763__tile_1, 0, 0, [16, 64], target_memory=pl.Mem.Left)
                tail_acc_inline1801__tile_l0_b_4: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tail_inline1853__tile_1, 0, 0, [64, 512], target_memory=pl.Mem.Right)
                tail_acc_inline1801__tile_l0_a_5: pl.Tile[[16, 64], pl.INT8, pl.MemRef(mem_left_23, pl.const(8192, pl.INT64), 1024), pl.Mem.Left, pl.TileView(valid_shape=[qproj_tail_rows_inline1851__ssa_v0, 64], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(qr_i8_tail_inline1763__tile_1, 0, 64, [16, 64], target_memory=pl.Mem.Left)
                tail_acc_inline1801__tile_l0_b_5: pl.Tile[[64, 512], pl.INT8, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(wq_tail_inline1853__tile_1, 64, 0, [64, 512], target_memory=pl.Mem.Right)
                tail_acc_inline1801__tile_l0_c_acc_4: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(tail_acc_inline1801__phi_v5, tail_acc_inline1801__tile_l0_a_4, tail_acc_inline1801__tile_l0_b_4)
                tail_acc_inline1801__tile_l0_c_acc_5: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.tile.matmul_acc(tail_acc_inline1801__tile_l0_c_acc_4, tail_acc_inline1801__tile_l0_a_5, tail_acc_inline1801__tile_l0_b_5)
                tail_acc_inline1801__rv_v2: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 65536), pl.Mem.Acc] = pl.yield_(tail_acc_inline1801__tile_l0_c_acc_5)
            q_proj_i32_inline1835__tile_1: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = pl.tile.store(tail_acc_inline1801__rv_v2, [tail_t0_inline1850__idx_v0, tail_w_col0_inline1809__ssa_v0], q_proj_i32_inline1835__iter_v4)
            q_proj_i32_inline1835__rv_v5: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_64", pl.const(0, pl.INT64), 0)] = pl.yield_(q_proj_i32_inline1835__tile_1)
        return q_proj_i32_inline1835__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def qproj_matmul_spmd(self, q_proj_i32_inline1835__ssa_v0: pl.Out[pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], qproj_full_rows_inline1804__ssa_v0: pl.Scalar[pl.INDEX], qr_i8_matmul_inline1787__rv_v2: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 33554432)], qproj_t_matmul_inline1791__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32]:
        q_proj_i32_inline1835__rv_v5: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qproj_matmul(q_proj_i32_inline1835__ssa_v0, qproj_full_rows_inline1804__ssa_v0, qr_i8_matmul_inline1787__rv_v2, wq_b__ssa_v0, qproj_t_matmul_inline1791__ssa_v0, tile_rows_inline1798__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar]})
        return q_proj_i32_inline1835__ssa_v0
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_hadamard_matmul(qr_bf16_inline2223__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 32768)], qh_acc_gm_inline2179__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32]:
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 16384)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 32768)
        idx_inline2227__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        o0_inline2221__ssa_v1: pl.Scalar[pl.INDEX] = idx_inline2227__ssa_v0 * 64
        t__tile: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_mat_3, pl.const(0, pl.INT64), 16384), pl.Mem.Mat] = pl.tile.load(qr_bf16_inline2223__ssa_v1, [o0_inline2221__ssa_v1, 0], [64, 128], [64, 128], target_memory=pl.Mem.Mat)
        hadamard_idx__ssa_v0_mat: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(16384, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(hadamard_idx__ssa_v0, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Mat)
        t__tile_Left: pl.Tile[[64, 128], pl.BF16, pl.MemRef(mem_left_5, pl.const(0, pl.INT64), 16384), pl.Mem.Left] = pl.tile.move(t__tile, target_memory=pl.Mem.Left)
        hadamard_idx__ssa_v0_mat_Right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.move(hadamard_idx__ssa_v0_mat, target_memory=pl.Mem.Right)
        qh_acc_inline2178__tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 32768), pl.Mem.Acc] = pl.tile.matmul(t__tile_Left, hadamard_idx__ssa_v0_mat_Right)
        qh_acc_gm_inline2179__tile: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(qh_acc_inline2178__tile, [o0_inline2221__ssa_v1, 0], qh_acc_gm_inline2179__ssa_v0)
        return qh_acc_gm_inline2179__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def qr_hadamard_matmul_spmd(self, qr_bf16_inline2223__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 32768)], qh_acc_gm_inline2179__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32]:
        qh_acc_gm_inline2179__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qr_hadamard_matmul(qr_bf16_inline2223__ssa_v1, hadamard_idx__ssa_v0, qh_acc_gm_inline2179__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing]})
        return qh_acc_gm_inline2179__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_hadamard_quant(qh_acc_gm_inline2179__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], qr_hadamard_scale_dq_inline2234__ssa_v0: pl.Out[pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 65536)]], qr_hadamard_i8_inline2177__ssa_v0: pl.Out[pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2097152)]]) -> tuple[pl.Tensor[[16384, 1], pl.FP32], pl.Tensor[[16384, 128], pl.INT8]]:
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        idx_inline2227__ssa_v1: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        o0_v1_inline2218__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2227__ssa_v1 * 64
        qh_amax_inline2278__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0001)
        for h0_inline2198__idx_v0, (qh_amax_inline2278__iter_v1,) in pl.range(0, 128, 64, init_values=(qh_amax_inline2278__tile,)):
            qh_a_f32_inline2175__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(qh_acc_gm_inline2179__ssa_v1, [o0_v1_inline2218__ssa_v0, h0_inline2198__idx_v0], [64, 64], [64, 64], target_memory=pl.Mem.Vec)
            t__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(256, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.neg(qh_a_f32_inline2175__tile)
            qh_a_abs_inline2196__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(256, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.maximum(qh_a_f32_inline2175__tile, t__tile)
            tmp_tile: pl.Tile[[64, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([64, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_1: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(49408, pl.INT64), 256), pl.Mem.Vec] = pl.tile.row_max(qh_a_abs_inline2196__tile, tmp_tile)
            qh_a_max_inline2211__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(49408, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 64])
            qh_amax_inline2278__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(qh_amax_inline2278__iter_v1, qh_a_max_inline2211__tile)
            qh_amax_inline2278__rv_v2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.yield_(qh_amax_inline2278__tile_1)
        t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=127.0)
        qh_scale_quant_row_inline2183__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(t__tile_2, qh_amax_inline2278__rv_v2)
        t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.recip(qh_scale_quant_row_inline2183__tile)
        qh_scale_dq_inline2176__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [64, 1])
        qr_hadamard_scale_dq_inline2234__tile: pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 65536)] = pl.tile.store(qh_scale_dq_inline2176__tile, [o0_v1_inline2218__ssa_v0, 0], qr_hadamard_scale_dq_inline2234__ssa_v0)
        qh_scale_quant_inline2202__tile: pl.Tile[[64, 1], pl.FP32, pl.MemRef(mem_vec_5, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.reshape(qh_scale_quant_row_inline2183__tile, [64, 1])
        for h1_inline2188__idx_v0, (qr_hadamard_i8_inline2177__iter_v1,) in pl.range(0, 128, 64, init_values=(qr_hadamard_i8_inline2177__ssa_v0,)):
            qh_q_f32_inline2226__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.load(qh_acc_gm_inline2179__ssa_v1, [o0_v1_inline2218__ssa_v0, h1_inline2188__idx_v0], [64, 64], [64, 64], target_memory=pl.Mem.Vec)
            qh_q_scaled_inline2191__tile: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.row_expand_mul(qh_q_f32_inline2226__tile, qh_scale_quant_inline2202__tile)
            qh_q_i32_inline2199__tile: pl.Tile[[64, 64], pl.INT32, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(qh_q_scaled_inline2191__tile, target_type=pl.INT32, mode='rint')
            qh_q_half_inline2290__tile: pl.Tile[[64, 64], pl.FP16, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(qh_q_i32_inline2199__tile, target_type=pl.FP16, mode='round')
            qh_i8_inline2233__tile: pl.Tile[[64, 64], pl.INT8, pl.MemRef(mem_vec_7, pl.const(16640, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(qh_q_half_inline2290__tile, target_type=pl.INT8, mode='trunc')
            qr_hadamard_i8_inline2177__tile: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2097152)] = pl.tile.store(qh_i8_inline2233__tile, [o0_v1_inline2218__ssa_v0, h1_inline2188__idx_v0], qr_hadamard_i8_inline2177__iter_v1)
            qr_hadamard_i8_inline2177__rv_v2: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 2097152)] = pl.yield_(qr_hadamard_i8_inline2177__tile)
        return qr_hadamard_scale_dq_inline2234__ssa_v0, qr_hadamard_i8_inline2177__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def qr_hadamard_quant_spmd(self, qh_acc_gm_inline2179__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], qr_hadamard_scale_dq_inline2234__ssa_v0: pl.Out[pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 65536)]], qr_hadamard_i8_inline2177__ssa_v0: pl.Out[pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2097152)]]) -> tuple[pl.Tensor[[16384, 1], pl.FP32], pl.Tensor[[16384, 128], pl.INT8]]:
        ret__tmp_v0: pl.Tuple[pl.Tensor[[16384, 1], pl.FP32], pl.Tensor[[16384, 128], pl.INT8]] = self.qr_hadamard_quant(qh_acc_gm_inline2179__ssa_v1, qr_hadamard_scale_dq_inline2234__ssa_v0, qr_hadamard_i8_inline2177__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]})
        qr_hadamard_scale_dq_inline2234__ssa_v1: pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 65536)] = ret__tmp_v0[0]
        qr_hadamard_i8_inline2177__rv_v2: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)] = ret__tmp_v0[1]
        return qr_hadamard_scale_dq_inline2234__ssa_v0, qr_hadamard_i8_inline2177__ssa_v0
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def qr_proj_matmul(qr_fp32_inline1834__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], qr_t_matmul_inline1793__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1799__idx_v0: pl.Scalar[pl.INDEX], x_view_inline1797__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)]) -> pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32]:
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 8192)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_6: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 8192)
        mem_mat_7: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_9: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_10: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_11: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_right_12: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_21: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        mem_left_23: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 4096)
        qbg_idx_inline1807__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        q_a_col0_inline1795__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline1807__ssa_v0 // 2 * 128
        qr_k_base_inline1819__ssa_v0: pl.Scalar[pl.INDEX] = qbg_idx_inline1807__ssa_v0 % 2 * 2048
        for t0_inline1823__idx_v0, (qr_fp32_inline1834__iter_v6,) in pl.range(0, qr_t_matmul_inline1793__ssa_v0, 16, init_values=(qr_fp32_inline1834__rv_v2,)):
            q_acc_inline1824__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            for db_inline1815__idx_v0, (q_acc_inline1824__iter_v1,) in pl.range(0, 8, 2, init_values=(q_acc_inline1824__tile,)):
                qr_d0_inline1822__ssa_v0: pl.Scalar[pl.INDEX] = qr_k_base_inline1819__ssa_v0 + db_inline1815__idx_v0 * 256
                qr_rows_inline1808__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1798__ssa_v0 - t0_inline1823__idx_v0, 16)
                x_t0_inline1766__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1799__idx_v0 + t0_inline1823__idx_v0
                qr_d0_inline1822__ssa_v0_1: pl.Scalar[pl.INDEX] = qr_k_base_inline1819__ssa_v0 + (db_inline1815__idx_v0 * 256 + 256)
                qr_rows_inline1808__ssa_v0_1: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1798__ssa_v0 - t0_inline1823__idx_v0, 16)
                x_t0_inline1766__ssa_v0_1: pl.Scalar[pl.INDEX] = tile_base_inline1799__idx_v0 + t0_inline1823__idx_v0
                q_x_chunk_bf16_inline1785__tile: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 256])] = pl.tile.load(x_view_inline1797__ssa_v0, [x_t0_inline1766__ssa_v0, qr_d0_inline1822__ssa_v0], [16, 256], [qr_rows_inline1808__ssa_v0, 256], target_memory=pl.Mem.Mat)
                w_chunk_inline1827__tile: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_5, pl.const(8192, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wq_a__ssa_v0, [qr_d0_inline1822__ssa_v0, q_a_col0_inline1795__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat)
                q_x_chunk_bf16_inline1785__tile_1: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_mat_6, pl.const(73728, pl.INT64), 8192), pl.Mem.Mat, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0_1, 256])] = pl.tile.load(x_view_inline1797__ssa_v0, [x_t0_inline1766__ssa_v0_1, qr_d0_inline1822__ssa_v0_1], [16, 256], [qr_rows_inline1808__ssa_v0_1, 256], target_memory=pl.Mem.Mat)
                w_chunk_inline1827__tile_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_7, pl.const(81920, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wq_a__ssa_v0, [qr_d0_inline1822__ssa_v0_1, q_a_col0_inline1795__ssa_v0], [256, 128], [256, 128], target_memory=pl.Mem.Mat)
                if db_inline1815__idx_v0 == 0:
                    q_acc_inline1824__tile_l0_init_storage: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                    q_acc_inline1824__tile_l0_init: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(q_acc_inline1824__tile_l0_init_storage, qr_rows_inline1808__ssa_v0, 128)
                    q_acc_inline1824__tile_l0_a: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_9, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(q_x_chunk_bf16_inline1785__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                    q_acc_inline1824__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(w_chunk_inline1827__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    q_acc_inline1824__tile_l0_a_1: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(q_x_chunk_bf16_inline1785__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                    q_acc_inline1824__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(w_chunk_inline1827__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                    q_acc_inline1824__tile_l0_c_acc: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(q_acc_inline1824__tile_l0_init, q_acc_inline1824__tile_l0_a, q_acc_inline1824__tile_l0_b, True)
                    q_acc_inline1824__tile_l0_c_acc_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(q_acc_inline1824__tile_l0_c_acc, q_acc_inline1824__tile_l0_a_1, q_acc_inline1824__tile_l0_b_1, False)
                    q_acc_inline1824__phi_v5: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.yield_(q_acc_inline1824__tile_l0_c_acc_1)
                else:
                    q_acc_inline1824__tile_l0_a_2: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_9, pl.const(12288, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(q_x_chunk_bf16_inline1785__tile, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                    q_acc_inline1824__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(w_chunk_inline1827__tile, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                    q_acc_inline1824__tile_l0_a_3: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_11, pl.const(0, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(q_x_chunk_bf16_inline1785__tile, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                    q_acc_inline1824__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(w_chunk_inline1827__tile, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                    q_acc_inline1824__tile_l0_c_acc_2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(q_acc_inline1824__iter_v1, q_acc_inline1824__tile_l0_a_2, q_acc_inline1824__tile_l0_b_2)
                    q_acc_inline1824__tile_l0_c_acc_3: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(q_acc_inline1824__tile_l0_c_acc_2, q_acc_inline1824__tile_l0_a_3, q_acc_inline1824__tile_l0_b_3)
                    q_acc_inline1824__phi_v5: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.yield_(q_acc_inline1824__tile_l0_c_acc_3)
                q_acc_inline1824__tile_l0_a_4: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_21, pl.const(4096, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(q_x_chunk_bf16_inline1785__tile_1, 0, 0, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline1824__tile_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(w_chunk_inline1827__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Right)
                q_acc_inline1824__tile_l0_a_5: pl.Tile[[16, 128], pl.BF16, pl.MemRef(mem_left_23, pl.const(8192, pl.INT64), 4096), pl.Mem.Left, pl.TileView(valid_shape=[qr_rows_inline1808__ssa_v0_1, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(q_x_chunk_bf16_inline1785__tile_1, 0, 128, [16, 128], target_memory=pl.Mem.Left)
                q_acc_inline1824__tile_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_12, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(w_chunk_inline1827__tile_1, 128, 0, [128, 128], target_memory=pl.Mem.Right)
                q_acc_inline1824__tile_l0_c_acc_4: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(q_acc_inline1824__phi_v5, q_acc_inline1824__tile_l0_a_4, q_acc_inline1824__tile_l0_b_4)
                q_acc_inline1824__tile_l0_c_acc_5: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.tile.matmul_acc(q_acc_inline1824__tile_l0_c_acc_4, q_acc_inline1824__tile_l0_a_5, q_acc_inline1824__tile_l0_b_5)
                q_acc_inline1824__rv_v2: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 8192), pl.Mem.Acc] = pl.yield_(q_acc_inline1824__tile_l0_c_acc_5)
            qr_fp32_inline1834__tile: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(q_acc_inline1824__rv_v2, [t0_inline1823__idx_v0, q_a_col0_inline1795__ssa_v0], qr_fp32_inline1834__iter_v6, atomic=pl.AtomicType.Add)
            qr_fp32_inline1834__rv_v7: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_fp32_inline1834__tile)
        return qr_fp32_inline1834__rv_v2
    @pl.function(type=pl.FunctionType.Spmd)
    def qr_proj_matmul_spmd(self, qr_fp32_inline1834__rv_v2: pl.InOut[pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], qr_t_matmul_inline1793__ssa_v0: pl.Scalar[pl.INDEX], tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1799__idx_v0: pl.Scalar[pl.INDEX], x_view_inline1797__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8388608)]) -> pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32]:
        qr_fp32_inline1834__rv_v7: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = self.qr_proj_matmul(qr_fp32_inline1834__rv_v2, qr_t_matmul_inline1793__ssa_v0, tile_rows_inline1798__ssa_v0, tile_base_inline1799__idx_v0, x_view_inline1797__ssa_v0, wq_a__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]})
        return qr_fp32_inline1834__rv_v2
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_proj_seed(qr_fp32_inline1834__ssa_v0: pl.Out[pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], qr_t_matmul_inline1793__ssa_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32]:
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        for ts0_inline1784__idx_v0, (qr_fp32_inline1834__iter_v1,) in pl.range(0, qr_t_matmul_inline1793__ssa_v0, 16, init_values=(qr_fp32_inline1834__ssa_v0,)):
            for nseed0_inline1855__idx_v0, (qr_fp32_inline1834__iter_v3,) in pl.range(0, 1024, 128, init_values=(qr_fp32_inline1834__iter_v1,)):
                qr_seed_inline1790__tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([16, 128], dtype=pl.FP32, value=0.0)
                qr_fp32_inline1834__tile: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_seed_inline1790__tile, [ts0_inline1784__idx_v0, nseed0_inline1855__idx_v0], qr_fp32_inline1834__iter_v3)
                qr_fp32_inline1834__rv_v4: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_fp32_inline1834__tile)
            qr_fp32_inline1834__rv_v2: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_fp32_inline1834__rv_v4)
        return qr_fp32_inline1834__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_rms_norm_quant(tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1799__idx_v0: pl.Scalar[pl.INDEX], qr_fp32_inline1834__rv_v7: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)], qr_scale_pad_store_inline1814__ssa_v0: pl.InOut[pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], qr_scale_view_inline1796__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], qr_i8_matmul_inline1787__ssa_v0: pl.InOut[pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]], qr_view_inline1775__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]]) -> tuple[pl.Scalar[pl.INDEX], pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32], pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8], pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32]]:
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_22: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_23: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        tg_idx_inline1757__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline1758__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1757__ssa_v0 * 8
        valid_rows_inline1818__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(tile_rows_inline1798__ssa_v0 - tg_inline1758__ssa_v0, 8)
        out_tg_inline1826__ssa_v0: pl.Scalar[pl.INDEX] = tile_base_inline1799__idx_v0 + tg_inline1758__ssa_v0
        qr_sq_sum_inline1817__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(41536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        qr_amax_g_inline1782__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
        for qr_rms_col0_inline1886__idx_v0, (qr_amax_g_inline1782__iter_v1, qr_sq_sum_inline1817__iter_v1) in pl.range(0, 1024, 512, init_values=(qr_amax_g_inline1782__tile, qr_sq_sum_inline1817__tile)):
            qr_rms_chunk_inline1781__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(qr_fp32_inline1834__rv_v7, [tg_inline1758__ssa_v0, qr_rms_col0_inline1886__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec)
            t__tile: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(gamma_cq__ssa_v0, [qr_rms_col0_inline1886__idx_v0], [256], [256], target_memory=pl.Mem.Vec)
            qr_rms_chunk_inline1781__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(qr_fp32_inline1834__rv_v7, [tg_inline1758__ssa_v0, qr_rms_col0_inline1886__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec)
            t__tile_1: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(gamma_cq__ssa_v0, [qr_rms_col0_inline1886__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec)
            qr_rms_sq_inline1779__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8704, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(qr_rms_chunk_inline1781__tile, qr_rms_chunk_inline1781__tile)
            tmp_tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16896, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_2: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(25088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(qr_rms_sq_inline1779__tile, tmp_tile)
            qr_rms_row_sum_inline1821__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(25088, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [1, 8])
            qr_sq_sum_inline1817__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16896, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(qr_sq_sum_inline1817__iter_v1, qr_rms_row_sum_inline1821__tile)
            gamma_rms_cast_inline1774__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8704, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
            gamma_rms_chunk_inline1772__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8704, pl.INT64), 1024), pl.Mem.Vec] = gamma_rms_cast_inline1774__tile
            qr_g_inline1786__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(qr_rms_chunk_inline1781__tile, gamma_rms_chunk_inline1772__tile)
            qr_g_abs_inline1836__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_g_inline1786__tile)
            tmp_tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8704, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_3: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(qr_g_abs_inline1836__tile, tmp_tile_1)
            qr_g_row_max_inline1770__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 8])
            qr_amax_g_inline1782__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(qr_amax_g_inline1782__iter_v1, qr_g_row_max_inline1770__tile)
            qr_rms_sq_inline1779__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_22, pl.const(25120, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.mul(qr_rms_chunk_inline1781__tile_1, qr_rms_chunk_inline1781__tile_1)
            tmp_tile_2: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_23, pl.const(33312, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_4: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_24, pl.const(41504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(qr_rms_sq_inline1779__tile_1, tmp_tile_2)
            qr_rms_row_sum_inline1821__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(41504, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_4, [1, 8])
            qr_sq_sum_inline1817__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(41536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(qr_sq_sum_inline1817__tile_1, qr_rms_row_sum_inline1821__tile_1)
            gamma_rms_cast_inline1774__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_22, pl.const(25120, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode='round')
            gamma_rms_chunk_inline1772__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_22, pl.const(25120, pl.INT64), 1024), pl.Mem.Vec] = gamma_rms_cast_inline1774__tile_1
            qr_g_inline1786__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(qr_rms_chunk_inline1781__tile_1, gamma_rms_chunk_inline1772__tile_1)
            qr_g_abs_inline1836__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.abs(qr_g_inline1786__tile_1)
            tmp_tile_3: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_22, pl.const(25120, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([8, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_5: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_23, pl.const(33312, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(qr_g_abs_inline1836__tile_1, tmp_tile_3)
            qr_g_row_max_inline1770__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_23, pl.const(33312, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_5, [1, 8])
            qr_amax_g_inline1782__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(41568, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(qr_amax_g_inline1782__tile_1, qr_g_row_max_inline1770__tile_1)
            qr_amax_g_inline1782__rv_v2, qr_sq_sum_inline1817__rv_v2 = pl.yield_(qr_amax_g_inline1782__tile_2, qr_sq_sum_inline1817__tile_2)
        t__tile_6: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(qr_sq_sum_inline1817__rv_v2, 0.0009765625)
        t__tile_7: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__tile_6, 9.9999999999999995e-07)
        rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        qr_inv_rms_inline1769__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(t__tile_7, rsqrt_tmp)
        qr_inv_rms_t_inline1802__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_9, pl.const(49792, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_inv_rms_inline1769__tile, [8, 1])
        qr_amax_floor_inline1829__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
        qr_amax_normed_inline1792__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.mul(qr_inv_rms_inline1769__tile, qr_amax_g_inline1782__rv_v2)
        qr_tile_amax_inline1828__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(qr_amax_floor_inline1829__tile, qr_amax_normed_inline1792__tile)
        t__tile_8: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
        qr_scale_quant_row_inline1767__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(t__tile_8, qr_tile_amax_inline1828__tile)
        qr_scale_quant_t_inline1765__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qr_scale_quant_row_inline1767__tile, [8, 1])
        t__tile_9: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(qr_scale_quant_row_inline1767__tile)
        qr_tile_scale_dq_inline1764__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_9, [8, 1])
        qr_scale_pad_store_inline1814__tile: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_tile_scale_dq_inline1764__tile, [tg_inline1758__ssa_v0, 0], qr_scale_pad_store_inline1814__ssa_v0)
        if valid_rows_inline1818__ssa_v0 == 8:
            qr_scale_view_inline1796__tile: pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_tile_scale_dq_inline1764__tile, [out_tg_inline1826__ssa_v0, 0], qr_scale_view_inline1796__ssa_v0)
        else:
            qr_scale_tail_inline1777__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1818__ssa_v0, 1])] = pl.tile.load(qr_scale_pad_store_inline1814__tile, [tg_inline1758__ssa_v0, 0], [8, 1], [valid_rows_inline1818__ssa_v0, 1], target_memory=pl.Mem.Vec)
            qr_scale_view_inline1796__store: pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_scale_tail_inline1777__ssa_v0, [out_tg_inline1826__ssa_v0, 0], qr_scale_view_inline1796__ssa_v0)
        for qa_inline1761__idx_v0, (qr_i8_matmul_inline1787__iter_v1, qr_view_inline1775__iter_v1) in pl.range(0, 1024, 512, init_values=(qr_i8_matmul_inline1787__ssa_v0, qr_view_inline1775__ssa_v0)):
            qr_chunk_inline1780__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(qr_fp32_inline1834__rv_v7, [tg_inline1758__ssa_v0, qa_inline1761__idx_v0], [8, 256], [8, 256], target_memory=pl.Mem.Vec)
            t__tile_10: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_22, pl.const(25120, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(gamma_cq__ssa_v0, [qa_inline1761__idx_v0], [256], [256], target_memory=pl.Mem.Vec)
            qr_chunk_inline1780__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(qr_fp32_inline1834__rv_v7, [tg_inline1758__ssa_v0, qa_inline1761__idx_v0 + 256], [8, 256], [8, 256], target_memory=pl.Mem.Vec)
            t__tile_11: pl.Tile[[256], pl.BF16, pl.MemRef(mem_vec_23, pl.const(33312, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(gamma_cq__ssa_v0, [qa_inline1761__idx_v0 + 256], [256], [256], target_memory=pl.Mem.Vec)
            gamma_q_cast_inline1776__tile: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8704, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.FP32, mode='round')
            gamma_q_chunk_inline1794__tile: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8704, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline1776__tile
            t__tile_12: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(qr_chunk_inline1780__tile, qr_inv_rms_t_inline1802__tile)
            qr_q_normed_inline1810__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_12, gamma_q_chunk_inline1794__tile)
            qr_q_scaled_inline1805__tile: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(qr_q_normed_inline1810__tile, qr_scale_quant_t_inline1765__tile)
            qr_q_i32_inline1762__tile: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(qr_q_scaled_inline1805__tile, target_type=pl.INT32, mode='rint')
            qr_q_half_inline1756__tile: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(qr_q_i32_inline1762__tile, target_type=pl.FP16, mode='round')
            qr_q_i8_inline1831__tile: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qr_q_half_inline1756__tile, target_type=pl.INT8, mode='trunc')
            qr_i8_matmul_inline1787__tile: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_q_i8_inline1831__tile, [tg_inline1758__ssa_v0, qa_inline1761__idx_v0], qr_i8_matmul_inline1787__iter_v1)
            if valid_rows_inline1818__ssa_v0 == 8:
                qr_view_inline1775__tile: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_q_i8_inline1831__tile, [out_tg_inline1826__ssa_v0, qa_inline1761__idx_v0], qr_view_inline1775__iter_v1)
                qr_view_inline1775__phi_v4: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline1775__tile)
            else:
                qr_q_tail_inline1832__ssa_v0: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_8, pl.const(41600, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1818__ssa_v0, 256])] = pl.tile.load(qr_i8_matmul_inline1787__tile, [tg_inline1758__ssa_v0, qa_inline1761__idx_v0], [8, 256], [valid_rows_inline1818__ssa_v0, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(qr_q_tail_inline1832__ssa_v0, [out_tg_inline1826__ssa_v0, qa_inline1761__idx_v0], qr_view_inline1775__iter_v1)
                qr_view_inline1775__phi_v4: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline1775__iter_v1)
            gamma_q_cast_inline1776__tile_1: pl.Tile[[256], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16896, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.FP32, mode='round')
            gamma_q_chunk_inline1794__tile_1: pl.Tile[[1, 256], pl.FP32, pl.MemRef(mem_vec_13, pl.const(16896, pl.INT64), 1024), pl.Mem.Vec] = gamma_q_cast_inline1776__tile_1
            t__tile_13: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(qr_chunk_inline1780__tile_1, qr_inv_rms_t_inline1802__tile)
            qr_q_normed_inline1810__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_13, gamma_q_chunk_inline1794__tile_1)
            qr_q_scaled_inline1805__tile_1: pl.Tile[[8, 256], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.row_expand_mul(qr_q_normed_inline1810__tile_1, qr_scale_quant_t_inline1765__tile)
            qr_q_i32_inline1762__tile_1: pl.Tile[[8, 256], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(qr_q_scaled_inline1805__tile_1, target_type=pl.INT32, mode='rint')
            qr_q_half_inline1756__tile_1: pl.Tile[[8, 256], pl.FP16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(qr_q_i32_inline1762__tile_1, target_type=pl.FP16, mode='round')
            qr_q_i8_inline1831__tile_1: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(qr_q_half_inline1756__tile_1, target_type=pl.INT8, mode='trunc')
            qr_i8_matmul_inline1787__tile_1: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_q_i8_inline1831__tile_1, [tg_inline1758__ssa_v0, qa_inline1761__idx_v0 + 256], qr_i8_matmul_inline1787__tile)
            if valid_rows_inline1818__ssa_v0 == 8:
                qr_view_inline1775__tile_1: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_q_i8_inline1831__tile_1, [out_tg_inline1826__ssa_v0, qa_inline1761__idx_v0 + 256], qr_view_inline1775__phi_v4)
                qr_view_inline1775__phi_v4_1: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_66", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline1775__tile_1)
            else:
                qr_q_tail_inline1832__ssa_v0_1: pl.Tile[[8, 256], pl.INT8, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1818__ssa_v0, 256])] = pl.tile.load(qr_i8_matmul_inline1787__tile_1, [tg_inline1758__ssa_v0, qa_inline1761__idx_v0 + 256], [8, 256], [valid_rows_inline1818__ssa_v0, 256], target_memory=pl.Mem.Vec)
                pl.tile.store(qr_q_tail_inline1832__ssa_v0_1, [out_tg_inline1826__ssa_v0, qa_inline1761__idx_v0 + 256], qr_view_inline1775__phi_v4)
                qr_view_inline1775__phi_v4_1: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_66", pl.const(0, pl.INT64), 0)] = pl.yield_(qr_view_inline1775__phi_v4)
            qr_i8_matmul_inline1787__rv_v2, qr_view_inline1775__rv_v2 = pl.yield_(qr_i8_matmul_inline1787__tile_1, qr_view_inline1775__phi_v4_1)
        return out_tg_inline1826__ssa_v0, qr_scale_pad_store_inline1814__ssa_v0, qr_i8_matmul_inline1787__ssa_v0, qr_scale_view_inline1796__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def qr_rms_norm_quant_spmd(self, tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX], tile_base_inline1799__idx_v0: pl.Scalar[pl.INDEX], qr_fp32_inline1834__rv_v7: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 2048)], qr_scale_pad_store_inline1814__ssa_v0: pl.InOut[pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)]], qr_scale_view_inline1796__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], qr_i8_matmul_inline1787__ssa_v0: pl.InOut[pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]], qr_view_inline1775__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]]) -> tuple[pl.Scalar[pl.INDEX], pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32], pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8]]:
        ret__tmp_v0: pl.Tuple[pl.Scalar[pl.INDEX], pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32], pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8], pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32]] = self.qr_rms_norm_quant(tile_rows_inline1798__ssa_v0, tile_base_inline1799__idx_v0, qr_fp32_inline1834__rv_v7, gamma_cq__ssa_v0, qr_scale_pad_store_inline1814__ssa_v0, qr_scale_view_inline1796__ssa_v0, qr_i8_matmul_inline1787__ssa_v0, qr_view_inline1775__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.inout, pl.adir.output_existing]})
        out_tg_inline1826__ssa_v0: pl.Scalar[pl.INDEX] = ret__tmp_v0[0]
        qr_scale_pad_store_inline1814__ssa_v1: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        qr_i8_matmul_inline1787__rv_v2: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
        qr_scale_view_inline1796__ssa_v2: pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[3]
        return out_tg_inline1826__ssa_v0, qr_scale_pad_store_inline1814__ssa_v0, qr_i8_matmul_inline1787__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_rope(rope_swap_idx_t_inline2189__ssa_v1: pl.Tensor[[32, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8192)], idx_cos_il_inline1282__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], idx_sin_signed_inline1307__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], qr_proj_flat_inline2295__ssa_v0: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], qr_bf16_inline2223__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16]:
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        idx_inline2276__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        o0_inline2221__ssa_v0: pl.Scalar[pl.INDEX] = idx_inline2276__ssa_v0 * 32
        token_idx_inline2272__ssa_v0: pl.Scalar[pl.INDEX] = o0_inline2221__ssa_v0 // 64
        rope_swap_idx_inline2190__tile: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(rope_swap_idx_t_inline2189__ssa_v1, [0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
        cos_row_inline2192__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(16896, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(idx_cos_il_inline1282__rv_v2, [token_idx_inline2272__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        sin_row_inline2231__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(17152, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(idx_sin_signed_inline1307__rv_v2, [token_idx_inline2272__ssa_v0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        qr_nope_slice_inline2206__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(17408, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(qr_proj_flat_inline2295__ssa_v0, [o0_inline2221__ssa_v0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
        qr_rope_slice_inline2222__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(25600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(qr_proj_flat_inline2295__ssa_v0, [o0_inline2221__ssa_v0, 64], [32, 64], [32, 64], target_memory=pl.Mem.Vec)
        gather_acc_init: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([32, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for gather_lv, (gather_ia,) in pl.range(32, init_values=(gather_acc_init,)):
            gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(25600, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(qr_rope_slice_inline2222__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_swap_idx_inline2190__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_11, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(8448, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
            gather_asmbl: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
            gather_rv: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.yield_(gather_asmbl)
        qr_swapped_inline2205__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = gather_rv
        t__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(qr_rope_slice_inline2222__tile, cos_row_inline2192__tile)
        t__tile_1: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(25600, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(qr_swapped_inline2205__tile, sin_row_inline2231__tile)
        rope_rot_inline2185__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.add(t__tile, t__tile_1)
        t__tile_2: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_8, pl.const(17408, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(qr_nope_slice_inline2206__tile, target_type=pl.BF16, mode='rint')
        t__tile_3: pl.Tile[[32, 64], pl.BF16, pl.MemRef(mem_vec_9, pl.const(25600, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(rope_rot_inline2185__tile, target_type=pl.BF16, mode='rint')
        qr_vec_inline2181__tile: pl.Tile[[32, 128], pl.BF16, pl.MemRef(mem_vec_5, pl.const(8704, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.concat(t__tile_2, t__tile_3)
        qr_bf16_inline2223__tile: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(qr_vec_inline2181__tile, [o0_inline2221__ssa_v0, 0], qr_bf16_inline2223__ssa_v0)
        return qr_bf16_inline2223__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def qr_rope_spmd(self, rope_swap_idx_t_inline2189__ssa_v1: pl.Tensor[[32, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8192)], idx_cos_il_inline1282__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], idx_sin_signed_inline1307__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], qr_proj_flat_inline2295__ssa_v0: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], qr_bf16_inline2223__ssa_v0: pl.Out[pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]]) -> pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16]:
        qr_bf16_inline2223__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = self.qr_rope(rope_swap_idx_t_inline2189__ssa_v1, idx_cos_il_inline1282__rv_v2, idx_sin_signed_inline1307__rv_v2, qr_proj_flat_inline2295__ssa_v0, qr_bf16_inline2223__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
        return qr_bf16_inline2223__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def qr_rope_swap_idx(rope_swap_idx_t_inline2189__ssa_v0: pl.Out[pl.Tensor[[32, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8192)]]) -> pl.Tensor[[32, 64], pl.INT32]:
        mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        t__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.full([32, 64], dtype=pl.FP32, value=1.0)
        t__tile_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8192, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.FP32, mode='round')
        sw_col_inline2204__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile, t__tile_2)
        t__tile_3: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(sw_col_inline2204__tile, 0.5)
        t__tile_4: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.INT32, mode='trunc')
        sw_dup_f_inline2263__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_4, target_type=pl.FP32, mode='round')
        t__tile_5: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline2263__tile, 2.0)
        sw_lane_inline2241__tile: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(sw_col_inline2204__tile, t__tile_5)
        t__tile_6: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.adds(sw_col_inline2204__tile, 1.0)
        t__tile_7: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(8192, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline2241__tile, 2.0)
        t__tile_8: pl.Tile[[32, 64], pl.FP32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.sub(t__tile_6, t__tile_7)
        t__tile_9: pl.Tile[[32, 64], pl.INT32, pl.MemRef(mem_vec_1, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(t__tile_8, target_type=pl.INT32, mode='round')
        rope_swap_idx_t_inline2189__tile: pl.Tensor[[32, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8192)] = pl.tile.store(t__tile_9, [0, 0], rope_swap_idx_t_inline2189__ssa_v0)
        return rope_swap_idx_t_inline2189__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rms_norm(t_dim_inline1611__ssa_v0: pl.Scalar[pl.INDEX], x_mixed_inline1253__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], x_normed_t_inline1243__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], attn_norm_w__ssa_v0: pl.Tensor[[4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8192)]) -> tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16], pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16]]:
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_39: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        tg_idx_inline1603__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        tg_inline1598__ssa_v0: pl.Scalar[pl.INDEX] = tg_idx_inline1603__ssa_v0 * 8
        valid_rows_inline1600__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1611__ssa_v0 - tg_inline1598__ssa_v0, 8)
        if valid_rows_inline1600__ssa_v0 == 8:
            t__tmp_v74: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            tg_inline80_inline1597__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v74 * 8
            x_sq_sum_inline83_inline1612__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            for rms_db_inline82_inline1602__idx_v0, (x_sq_sum_inline83_inline1612__iter_v1,) in pl.range(0, 32, 2, init_values=(x_sq_sum_inline83_inline1612__tile,)):
                rms_d0_inline87_inline1595__ssa_v0: pl.Scalar[pl.INDEX] = rms_db_inline82_inline1602__idx_v0 * 128
                rms_d0_inline87_inline1595__ssa_v0_1: pl.Scalar[pl.INDEX] = rms_db_inline82_inline1602__idx_v0 * 128 + 128
                rms_x_input_inline84_inline1593__tile: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline80_inline1597__ssa_v0, rms_d0_inline87_inline1595__ssa_v0], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                rms_x_input_inline84_inline1593__tile_1: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline80_inline1597__ssa_v0, rms_d0_inline87_inline1595__ssa_v0_1], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                rms_x_chunk_inline88_inline1608__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(rms_x_input_inline84_inline1593__tile, target_type=pl.FP32, mode='round')
                rms_x_sq_inline89_inline1619__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(rms_x_chunk_inline88_inline1608__tile, rms_x_chunk_inline88_inline1608__tile)
                tmp_tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(rms_x_sq_inline89_inline1619__tile, tmp_tile)
                rms_x_row_sum_inline81_inline1609__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 8])
                x_sq_sum_inline83_inline1612__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(x_sq_sum_inline83_inline1612__iter_v1, rms_x_row_sum_inline81_inline1609__tile)
                rms_x_chunk_inline88_inline1608__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(rms_x_input_inline84_inline1593__tile_1, target_type=pl.FP32, mode='round')
                rms_x_sq_inline89_inline1619__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(rms_x_chunk_inline88_inline1608__tile_1, rms_x_chunk_inline88_inline1608__tile_1)
                tmp_tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_sum(rms_x_sq_inline89_inline1619__tile_1, tmp_tile_1)
                rms_x_row_sum_inline81_inline1609__tile_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 8])
                x_sq_sum_inline83_inline1612__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(x_sq_sum_inline83_inline1612__tile_1, rms_x_row_sum_inline81_inline1609__tile_1)
                x_sq_sum_inline83_inline1612__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(x_sq_sum_inline83_inline1612__tile_2)
            t__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(x_sq_sum_inline83_inline1612__rv_v2, 0.000244140625)
            t__tile_3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__tile_2, 9.9999999999999995e-07)
            rsqrt_tmp: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 32), pl.Mem.Vec] = pl.tile.create([1, 8], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            x_inv_rms_inline85_inline1610__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.rsqrt(t__tile_3, rsqrt_tmp)
            x_inv_rms_t_inline93_inline1594__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(x_inv_rms_inline85_inline1610__tile, [8, 1])
            for apply_db_inline86_inline1614__idx_v0, (x_normed_t_inline1243__iter_v1,) in pl.range(0, 32, 2, init_values=(x_normed_t_inline1243__ssa_v0,)):
                apply_d0_inline91_inline1615__ssa_v0: pl.Scalar[pl.INDEX] = apply_db_inline86_inline1614__idx_v0 * 128
                apply_d0_inline91_inline1615__ssa_v0_1: pl.Scalar[pl.INDEX] = apply_db_inline86_inline1614__idx_v0 * 128 + 128
                apply_x_input_inline94_inline1613__tile: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline80_inline1597__ssa_v0, apply_d0_inline91_inline1615__ssa_v0], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                norm_w_input_inline77_inline1617__tile: pl.Tile[[128], pl.BF16, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(attn_norm_w__ssa_v0, [apply_d0_inline91_inline1615__ssa_v0], [128], [128], target_memory=pl.Mem.Vec)
                apply_x_input_inline94_inline1613__tile_1: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline80_inline1597__ssa_v0, apply_d0_inline91_inline1615__ssa_v0_1], [8, 128], [8, 128], target_memory=pl.Mem.Vec)
                norm_w_input_inline77_inline1617__tile_1: pl.Tile[[128], pl.BF16, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(attn_norm_w__ssa_v0, [apply_d0_inline91_inline1615__ssa_v0_1], [128], [128], target_memory=pl.Mem.Vec)
                apply_x_chunk_inline78_inline1620__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(apply_x_input_inline94_inline1613__tile, target_type=pl.FP32, mode='round')
                t__tile_4: pl.Tile[[1, 128], pl.BF16, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 256), pl.Mem.Vec] = norm_w_input_inline77_inline1617__tile
                norm_w_chunk_inline92_inline1604__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tile_4, target_type=pl.FP32, mode='round')
                x_scaled_inline90_inline1622__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(apply_x_chunk_inline78_inline1620__tile, x_inv_rms_t_inline93_inline1594__tile)
                x_normed_chunk_inline79_inline1592__tile: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(x_scaled_inline90_inline1622__tile, norm_w_chunk_inline92_inline1604__tile)
                t__tile_5: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(x_normed_chunk_inline79_inline1592__tile, target_type=pl.BF16, mode='rint')
                x_normed_t_inline1243__tile: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_5, [tg_inline80_inline1597__ssa_v0, apply_d0_inline91_inline1615__ssa_v0], x_normed_t_inline1243__iter_v1)
                apply_x_chunk_inline78_inline1620__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(apply_x_input_inline94_inline1613__tile_1, target_type=pl.FP32, mode='round')
                t__tile_6: pl.Tile[[1, 128], pl.BF16, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 256), pl.Mem.Vec] = norm_w_input_inline77_inline1617__tile_1
                norm_w_chunk_inline92_inline1604__tile_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode='round')
                x_scaled_inline90_inline1622__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(apply_x_chunk_inline78_inline1620__tile_1, x_inv_rms_t_inline93_inline1594__tile)
                x_normed_chunk_inline79_inline1592__tile_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(x_scaled_inline90_inline1622__tile_1, norm_w_chunk_inline92_inline1604__tile_1)
                t__tile_7: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(x_normed_chunk_inline79_inline1592__tile_1, target_type=pl.BF16, mode='rint')
                x_normed_t_inline1243__tile_1: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_7, [tg_inline80_inline1597__ssa_v0, apply_d0_inline91_inline1615__ssa_v0_1], x_normed_t_inline1243__tile)
                x_normed_t_inline1243__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_35", pl.const(0, pl.INT64), 0)] = pl.yield_(x_normed_t_inline1243__tile_1)
        else:
            t__tmp_v80: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
            tg_inline103_inline1596__ssa_v0: pl.Scalar[pl.INDEX] = t__tmp_v80 * 8
            t__tmp_v81: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_mixed_inline1253__rv_v2, 0)
            valid_rows_inline113_inline1601__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t__tmp_v81 - tg_inline103_inline1596__ssa_v0, 8)
            row_reduce_tmp_inline101_inline1623__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([8, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            x_sq_sum_inline104_inline1624__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
            for rms_db_inline105_inline1591__idx_v0, (x_sq_sum_inline104_inline1624__iter_v1,) in pl.range(0, 32, 2, init_values=(x_sq_sum_inline104_inline1624__ssa_v0,)):
                rms_d0_inline106_inline1589__ssa_v0: pl.Scalar[pl.INDEX] = rms_db_inline105_inline1591__idx_v0 * 128
                rms_d0_inline106_inline1589__ssa_v0_1: pl.Scalar[pl.INDEX] = rms_db_inline105_inline1591__idx_v0 * 128 + 128
                rms_x_input_inline107_inline1587__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline103_inline1596__ssa_v0, rms_d0_inline106_inline1589__ssa_v0], [8, 128], [valid_rows_inline113_inline1601__ssa_v0, 128], target_memory=pl.Mem.Vec)
                rms_x_input_inline107_inline1587__ssa_v0_1: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline103_inline1596__ssa_v0, rms_d0_inline106_inline1589__ssa_v0_1], [8, 128], [valid_rows_inline113_inline1601__ssa_v0, 128], target_memory=pl.Mem.Vec)
                rms_x_chunk_inline108_inline1586__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(rms_x_input_inline107_inline1587__ssa_v0, target_type=pl.FP32, mode='round')
                rms_x_sq_inline111_inline1585__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.mul(rms_x_chunk_inline108_inline1586__ssa_v0, rms_x_chunk_inline108_inline1586__ssa_v0)
                t__tmp_v82: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 1])] = pl.tile.row_sum(rms_x_sq_inline111_inline1585__ssa_v0, row_reduce_tmp_inline101_inline1623__ssa_v0)
                rms_x_row_sum_inline114_inline1584__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline113_inline1601__ssa_v0])] = pl.tile.reshape(t__tmp_v82, [1, 8])
                x_sq_sum_inline104_inline1624__ssa_v3: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(x_sq_sum_inline104_inline1624__iter_v1, rms_x_row_sum_inline114_inline1584__ssa_v0)
                rms_x_chunk_inline108_inline1586__ssa_v0_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(rms_x_input_inline107_inline1587__ssa_v0_1, target_type=pl.FP32, mode='round')
                rms_x_sq_inline111_inline1585__ssa_v0_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.mul(rms_x_chunk_inline108_inline1586__ssa_v0_1, rms_x_chunk_inline108_inline1586__ssa_v0_1)
                t__tmp_v82_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 1])] = pl.tile.row_sum(rms_x_sq_inline111_inline1585__ssa_v0_1, row_reduce_tmp_inline101_inline1623__ssa_v0)
                rms_x_row_sum_inline114_inline1584__ssa_v0_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_rows_inline113_inline1601__ssa_v0])] = pl.tile.reshape(t__tmp_v82_1, [1, 8])
                x_sq_sum_inline104_inline1624__ssa_v3_1: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 32), pl.Mem.Vec] = pl.tile.add(x_sq_sum_inline104_inline1624__ssa_v3, rms_x_row_sum_inline114_inline1584__ssa_v0_1)
                x_sq_sum_inline104_inline1624__rv_v2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 32), pl.Mem.Vec] = pl.yield_(x_sq_sum_inline104_inline1624__ssa_v3_1)
            t__tmp_v83: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 32), pl.Mem.Vec] = pl.tile.muls(x_sq_sum_inline104_inline1624__rv_v2, 0.000244140625)
            t__tmp_v84: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 32), pl.Mem.Vec] = pl.tile.adds(t__tmp_v83, 9.9999999999999995e-07)
            t__tmp_v85: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 32), pl.Mem.Vec] = pl.tile.sqrt(t__tmp_v84)
            x_inv_rms_inline115_inline1618__ssa_v0: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(t__tmp_v85)
            x_inv_rms_t_inline116_inline1621__ssa_v0: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(8192, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(x_inv_rms_inline115_inline1618__ssa_v0, [8, 1])
            for apply_db_inline110_inline1583__idx_v0 in pl.range(0, 32, 2):
                apply_d0_inline100_inline1582__ssa_v0: pl.Scalar[pl.INDEX] = apply_db_inline110_inline1583__idx_v0 * 128
                apply_d0_inline100_inline1582__ssa_v0_1: pl.Scalar[pl.INDEX] = apply_db_inline110_inline1583__idx_v0 * 128 + 128
                apply_x_input_inline99_inline1616__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline103_inline1596__ssa_v0, apply_d0_inline100_inline1582__ssa_v0], [8, 128], [valid_rows_inline113_inline1601__ssa_v0, 128], target_memory=pl.Mem.Vec)
                norm_w_input_inline96_inline1588__ssa_v0: pl.Tile[[128], pl.BF16, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(attn_norm_w__ssa_v0, [apply_d0_inline100_inline1582__ssa_v0], [128], [128], target_memory=pl.Mem.Vec)
                apply_x_input_inline99_inline1616__ssa_v0_1: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.load(x_mixed_inline1253__rv_v2, [tg_inline103_inline1596__ssa_v0, apply_d0_inline100_inline1582__ssa_v0_1], [8, 128], [valid_rows_inline113_inline1601__ssa_v0, 128], target_memory=pl.Mem.Vec)
                norm_w_input_inline96_inline1588__ssa_v0_1: pl.Tile[[128], pl.BF16, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(attn_norm_w__ssa_v0, [apply_d0_inline100_inline1582__ssa_v0_1], [128], [128], target_memory=pl.Mem.Vec)
                apply_x_chunk_inline98_inline1599__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(apply_x_input_inline99_inline1616__ssa_v0, target_type=pl.FP32, mode='round')
                t__tmp_v86: pl.Tile[[1, 128], pl.BF16, pl.MemRef(mem_vec_39, pl.const(8480, pl.INT64), 256), pl.Mem.Vec] = norm_w_input_inline96_inline1588__ssa_v0
                norm_w_chunk_inline102_inline1581__ssa_v0: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tmp_v86, target_type=pl.FP32, mode='round')
                x_scaled_inline109_inline1606__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.row_expand_mul(apply_x_chunk_inline98_inline1599__ssa_v0, x_inv_rms_t_inline116_inline1621__ssa_v0)
                x_normed_chunk_inline112_inline1580__ssa_v0: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.col_expand_mul(x_scaled_inline109_inline1606__ssa_v0, norm_w_chunk_inline102_inline1581__ssa_v0)
                x_normed_bf16_inline95_inline1590__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(x_normed_chunk_inline112_inline1580__ssa_v0, target_type=pl.BF16, mode='rint')
                x_normed_valid_inline97_inline1579__ssa_v0: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_6, pl.const(10528, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.set_validshape(x_normed_bf16_inline95_inline1590__ssa_v0, valid_rows_inline113_inline1601__ssa_v0, 128)
                x_normed_t_inline1243__store: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(x_normed_valid_inline97_inline1579__ssa_v0, [tg_inline103_inline1596__ssa_v0, apply_d0_inline100_inline1582__ssa_v0], x_normed_t_inline1243__ssa_v0)
                apply_x_chunk_inline98_inline1599__ssa_v0_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(apply_x_input_inline99_inline1616__ssa_v0_1, target_type=pl.FP32, mode='round')
                t__tmp_v86_1: pl.Tile[[1, 128], pl.BF16, pl.MemRef(mem_vec_24, pl.const(8224, pl.INT64), 256), pl.Mem.Vec] = norm_w_input_inline96_inline1588__ssa_v0_1
                norm_w_chunk_inline102_inline1581__ssa_v0_1: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 512), pl.Mem.Vec] = pl.tile.cast(t__tmp_v86_1, target_type=pl.FP32, mode='round')
                x_scaled_inline109_inline1606__ssa_v0_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.row_expand_mul(apply_x_chunk_inline98_inline1599__ssa_v0_1, x_inv_rms_t_inline116_inline1621__ssa_v0)
                x_normed_chunk_inline112_inline1580__ssa_v0_1: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.col_expand_mul(x_scaled_inline109_inline1606__ssa_v0_1, norm_w_chunk_inline102_inline1581__ssa_v0_1)
                x_normed_bf16_inline95_inline1590__ssa_v0_1: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.cast(x_normed_chunk_inline112_inline1580__ssa_v0_1, target_type=pl.BF16, mode='rint')
                x_normed_valid_inline97_inline1579__ssa_v0_1: pl.Tile[[8, 128], pl.BF16, pl.MemRef(mem_vec_8, pl.const(14624, pl.INT64), 2048), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline113_inline1601__ssa_v0, 128])] = pl.tile.set_validshape(x_normed_bf16_inline95_inline1590__ssa_v0_1, valid_rows_inline113_inline1601__ssa_v0, 128)
                x_normed_t_inline1243__store_1: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(x_normed_valid_inline97_inline1579__ssa_v0_1, [tg_inline103_inline1596__ssa_v0, apply_d0_inline100_inline1582__ssa_v0_1], x_normed_t_inline1243__ssa_v0)
        return x_normed_t_inline1243__ssa_v0, x_normed_t_inline1243__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def rms_norm_spmd(self, t_dim_inline1611__ssa_v0: pl.Scalar[pl.INDEX], x_mixed_inline1253__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], x_normed_t_inline1243__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], attn_norm_w__ssa_v0: pl.Tensor[[4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8192)]) -> pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16]:
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16], pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16]] = self.rms_norm(t_dim_inline1611__ssa_v0, x_mixed_inline1253__rv_v2, x_normed_t_inline1243__ssa_v0, attn_norm_w__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input]})
        x_normed_t_inline1243__phi_v4: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        x_normed_t_inline1243__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        return x_normed_t_inline1243__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope_cache_write(bs_inline2038__ssa_v0: pl.Scalar[pl.INDEX], cmp_cos_il_full_inline1249__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], cmp_sin_signed_full_inline1263__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], pooled_kv_inline2008__rv_v2: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1048576)], normed_kv_inline2016__ssa_v0: pl.InOut[pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1048576)]], norm_w_2d_inline2060__ssa_v0: pl.Tensor[[1, 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1024)], cmp_kv_cache_flat_inline2036__ssa_v0: pl.Out[pl.Tensor[[cmp_block_num_inline2055__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]], kv_flat_inline2039__ssa_v0: pl.Out[pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]], cmp_slots_inline1296__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)]):
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_46: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_47: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        rms_blk_inline1993__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        b0_inline1992__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline1993__ssa_v0 * 16
        rms_blk_rows_inline1991__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2038__ssa_v0 - b0_inline1992__ssa_v0, 16)
        cos_b_inline2057__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16896, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline1991__ssa_v0, 64])] = pl.tile.load(cmp_cos_il_full_inline1249__rv_v2, [b0_inline1992__ssa_v0, 0], [16, 64], [rms_blk_rows_inline1991__ssa_v0, 64], target_memory=pl.Mem.Vec)
        sin_b_inline1989__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(20992, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline1991__ssa_v0, 64])] = pl.tile.load(cmp_sin_signed_full_inline1263__rv_v2, [b0_inline1992__ssa_v0, 0], [16, 64], [rms_blk_rows_inline1991__ssa_v0, 64], target_memory=pl.Mem.Vec)
        partial_sq_inline2025__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
        for k0_inline1987__idx_v0, (partial_sq_inline2025__iter_v1,) in pl.range(0, 512, 64, init_values=(partial_sq_inline2025__tile,)):
            kv_rms_chunk_inline1986__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pooled_kv_inline2008__rv_v2, [b0_inline1992__ssa_v0, k0_inline1987__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            kv_rms_sq_inline1988__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(kv_rms_chunk_inline1986__tile, kv_rms_chunk_inline1986__tile)
            tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_46, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline1988__tile, tmp_tile)
            kv_rms_rowsum_inline2053__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_46, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 16])
            partial_sq_inline2025__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(partial_sq_inline2025__iter_v1, kv_rms_rowsum_inline2053__tile)
            partial_sq_inline2025__rv_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.yield_(partial_sq_inline2025__tile_1)
        t__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(partial_sq_inline2025__rv_v2, 0.001953125)
        t__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tile_1, 9.9999999999999995e-07)
        variance_inline1990__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_2, [16, 1])
        t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline1990__tile, [1, 16])
        t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v0)
        t__tile_3: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
        inv_rms_inline1984__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [1, 16])
        inv_rms_inline1984__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_46, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(inv_rms_inline1984__rm_a0_tmp_v2)
        inv_rms_inline1984__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_46, pl.const(16384, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(inv_rms_inline1984__row_major_tmp_v3, [16, 1])
        for k0_inline1983__idx_v0, (normed_kv_inline2016__iter_v1,) in pl.range(0, 448, 64, init_values=(normed_kv_inline2016__ssa_v0,)):
            kv_norm_chunk_inline2037__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pooled_kv_inline2008__rv_v2, [b0_inline1992__ssa_v0, k0_inline1983__idx_v0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
            t__tile_4: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(norm_w_2d_inline2060__ssa_v0, [0, k0_inline1983__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
            gamma_inline2024__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_4, target_type=pl.FP32, mode='round')
            t__tile_5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_norm_chunk_inline2037__tile, inv_rms_inline1984__tile)
            normed_chunk_inline1982__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_5, gamma_inline2024__tile)
            normed_kv_inline2016__tile: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1048576)] = pl.tile.store(normed_chunk_inline1982__tile, [b0_inline1992__ssa_v0, k0_inline1983__idx_v0], normed_kv_inline2016__iter_v1)
            normed_kv_inline2016__rv_v2: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 1048576)] = pl.yield_(normed_kv_inline2016__tile)
        kv_rope_norm_inline1981__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pooled_kv_inline2008__rv_v2, [b0_inline1992__ssa_v0, 448], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        t__tile_6: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(norm_w_2d_inline2060__ssa_v0, [0, 448], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        gamma_rope_inline1980__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode='round')
        t__tile_7: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_rope_norm_inline1981__tile, inv_rms_inline1984__tile)
        rope_normed_inline1979__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_7, gamma_rope_inline1980__tile)
        rope_ones_inline1978__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        t__tile_8: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_8, target_type=pl.FP32, mode='round')
        rope_col_inline1977__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(rope_ones_inline1978__tile, t__tile_9)
        t__tile_10: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline1977__tile, 0.5)
        t__tile_11: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.INT32, mode='trunc')
        rope_dup_f_inline1976__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.FP32, mode='round')
        t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline1976__tile, 2.0)
        rope_lane_inline2048__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(rope_col_inline1977__tile, t__tile_12)
        t__tile_13: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline1977__tile, 1.0)
        t__tile_14: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline2048__tile, 2.0)
        t__tile_15: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_13, t__tile_14)
        rope_swap_idx_inline1997__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_15, target_type=pl.INT32, mode='round')
        gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
            gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_normed_inline1979__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_12, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_swap_idx_inline1997__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_46, pl.const(16384, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_47, pl.const(16640, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
            gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
            gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
        swapped_inline1975__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(12288, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
        t__tile_16: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(rope_normed_inline1979__tile, cos_b_inline2057__tile)
        t__tile_17: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_8, pl.const(16896, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(swapped_inline1975__tile, sin_b_inline1989__tile)
        rope_rot_inline2002__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_16, t__tile_17)
        normed_kv_inline2016__tile_1: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 1048576)] = pl.tile.store(rope_rot_inline2002__tile, [b0_inline1992__ssa_v0, 448], normed_kv_inline2016__rv_v2)
        for inner_inline1985__idx_v0, (cmp_kv_cache_flat_inline2036__iter_v1, kv_flat_inline2039__iter_v1) in pl.range(rms_blk_rows_inline1991__ssa_v0, init_values=(cmp_kv_cache_flat_inline2036__ssa_v0, kv_flat_inline2039__ssa_v0)):
            token_inline2040__ssa_v2: pl.Scalar[pl.INDEX] = b0_inline1992__ssa_v0 + inner_inline1985__idx_v0
            cache_row_i64_inline1974__tile: pl.Scalar[pl.INT64] = pl.tensor.read(cmp_slots_inline1296__ssa_v0, [token_inline2040__ssa_v2])
            if 0 <= cache_row_i64_inline1974__tile:
                cache_row_inline2047__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(cache_row_i64_inline1974__tile, pl.INDEX)
                kv_row_fp32_inline2043__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(normed_kv_inline2016__tile_1, [token_inline2040__ssa_v2, 0], [1, 512], [1, 512], target_memory=pl.Mem.Vec)
                kv_flat_inline2039__tile: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = pl.tile.store(kv_row_fp32_inline2043__tile, [token_inline2040__ssa_v2, 0], kv_flat_inline2039__iter_v1)
                t__tile_18: pl.Tile[[1, 512], pl.BF16, pl.MemRef(mem_vec_13, pl.const(4096, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(kv_row_fp32_inline2043__tile, target_type=pl.BF16, mode='rint')
                cmp_kv_cache_flat_inline2036__tile: pl.Tensor[[cmp_block_num_inline2055__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_18, [cache_row_inline2047__ssa_v0, 0], cmp_kv_cache_flat_inline2036__iter_v1)
                cmp_kv_cache_flat_inline2036__phi_v4, kv_flat_inline2039__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline2036__tile, kv_flat_inline2039__tile)
            else:
                cmp_kv_cache_flat_inline2036__phi_v4, kv_flat_inline2039__phi_v4 = pl.yield_(cmp_kv_cache_flat_inline2036__iter_v1, kv_flat_inline2039__iter_v1)
            cmp_kv_cache_flat_inline2036__rv_v2, kv_flat_inline2039__rv_v2 = pl.yield_(cmp_kv_cache_flat_inline2036__phi_v4, kv_flat_inline2039__phi_v4)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_cache_write_spmd(self, bs_inline2038__ssa_v0: pl.Scalar[pl.INDEX], cmp_cos_il_full_inline1249__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], cmp_sin_signed_full_inline1263__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], pooled_kv_inline2008__rv_v2: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 1048576)], normed_kv_inline2016__ssa_v0: pl.InOut[pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 1048576)]], norm_w_2d_inline2060__ssa_v0: pl.Tensor[[1, 512], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1024)], cmp_kv_cache_flat_inline2036__ssa_v0: pl.Out[pl.Tensor[[cmp_block_num_inline2055__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)]], kv_flat_inline2039__ssa_v0: pl.Out[pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]], cmp_slots_inline1296__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)]):
        self.rmsnorm_rope_cache_write(bs_inline2038__ssa_v0, cmp_cos_il_full_inline1249__rv_v2, cmp_sin_signed_full_inline1263__rv_v2, pooled_kv_inline2008__rv_v2, normed_kv_inline2016__ssa_v0, norm_w_2d_inline2060__ssa_v0, cmp_kv_cache_flat_inline2036__ssa_v0, kv_flat_inline2039__ssa_v0, cmp_slots_inline1296__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input]})
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rmsnorm_rope(bs_inline2140__ssa_v0: pl.Scalar[pl.INDEX], cmp_cos_il_full_inline1249__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], cmp_sin_signed_full_inline1263__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], pooled_kv_inline2131__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 262144)], normed_kv_inline2164__ssa_v0: pl.Out[pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 131072)]], norm_w_2d_inline2173__ssa_v0: pl.Tensor[[1, 128], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)]) -> pl.Tensor[[512, 128], pl.BF16]:
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_10: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_12: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_50: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        rms_blk_inline2092__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        b0_inline2090__ssa_v0: pl.Scalar[pl.INDEX] = rms_blk_inline2092__ssa_v0 * 16
        rms_blk_rows_inline2141__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2140__ssa_v0 - b0_inline2090__ssa_v0, 16)
        cos_b_inline2087__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(24704, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline2141__ssa_v0, 64])] = pl.tile.load(cmp_cos_il_full_inline1249__rv_v2, [b0_inline2090__ssa_v0, 0], [16, 64], [rms_blk_rows_inline2141__ssa_v0, 64], target_memory=pl.Mem.Vec)
        sin_b_inline2085__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_6, pl.const(29056, pl.INT64), 4096), pl.Mem.Vec, pl.TileView(valid_shape=[rms_blk_rows_inline2141__ssa_v0, 64])] = pl.tile.load(cmp_sin_signed_full_inline1263__rv_v2, [b0_inline2090__ssa_v0, 0], [16, 64], [rms_blk_rows_inline2141__ssa_v0, 64], target_memory=pl.Mem.Vec)
        partial_sq_inline2155__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28800, pl.INT64), 64), pl.Mem.Vec] = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
        kv_rms_chunk_inline2084__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pooled_kv_inline2131__rv_v2, [b0_inline2090__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        kv_rms_chunk_inline2084__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pooled_kv_inline2131__rv_v2, [b0_inline2090__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        kv_rms_sq_inline2083__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(kv_rms_chunk_inline2084__tile, kv_rms_chunk_inline2084__tile)
        tmp_tile: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_12, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline2083__tile, tmp_tile)
        kv_rms_rowsum_inline2165__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_12, pl.const(12288, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile, [1, 16])
        partial_sq_inline2155__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(partial_sq_inline2155__tile, kv_rms_rowsum_inline2165__tile)
        kv_rms_sq_inline2083__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(12352, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(kv_rms_chunk_inline2084__tile_1, kv_rms_chunk_inline2084__tile_1)
        tmp_tile_1: pl.Tile[[16, 128], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.create([16, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        t__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_16, pl.const(24640, pl.INT64), 64), pl.Mem.Vec] = pl.tile.row_sum(kv_rms_sq_inline2083__tile_1, tmp_tile_1)
        kv_rms_rowsum_inline2165__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_16, pl.const(24640, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 16])
        partial_sq_inline2155__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28800, pl.INT64), 64), pl.Mem.Vec] = pl.tile.add(partial_sq_inline2155__tile_1, kv_rms_rowsum_inline2165__tile_1)
        t__tile_2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.muls(partial_sq_inline2155__tile_2, 0.0078125)
        t__tile_3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.adds(t__tile_2, 9.9999999999999995e-07)
        variance_inline2082__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [16, 1])
        t__rm_a0_tmp_v0: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(variance_inline2082__tile, [1, 16])
        t__row_major_tmp_v1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.sqrt(t__rm_a0_tmp_v0)
        t__tile_4: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__row_major_tmp_v1, [16, 1])
        inv_rms_inline2122__rm_a0_tmp_v2: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(t__tile_4, [1, 16])
        inv_rms_inline2122__row_major_tmp_v3: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_14, pl.const(12352, pl.INT64), 64), pl.Mem.Vec] = pl.tile.recip(inv_rms_inline2122__rm_a0_tmp_v2)
        inv_rms_inline2122__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_14, pl.const(12352, pl.INT64), 64), pl.Mem.Vec] = pl.tile.reshape(inv_rms_inline2122__row_major_tmp_v3, [16, 1])
        kv_norm_chunk_inline2080__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pooled_kv_inline2131__rv_v2, [b0_inline2090__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        t__tile_5: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(norm_w_2d_inline2173__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        gamma_inline2078__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_5, target_type=pl.FP32, mode='round')
        t__tile_6: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_norm_chunk_inline2080__tile, inv_rms_inline2122__tile)
        normed_chunk_inline2102__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_6, gamma_inline2078__tile)
        t__tile_7: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(normed_chunk_inline2102__tile, target_type=pl.BF16, mode='rint')
        normed_kv_inline2164__tile: pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 131072)] = pl.tile.store(t__tile_7, [b0_inline2090__ssa_v0, 0], normed_kv_inline2164__ssa_v0)
        normed_kv_inline2164__rv_v2: pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 131072)] = normed_kv_inline2164__tile
        kv_rope_norm_inline2135__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(pooled_kv_inline2131__rv_v2, [b0_inline2090__ssa_v0, 64], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        t__tile_8: pl.Tile[[1, 64], pl.BF16, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 128), pl.Mem.Vec] = pl.tile.load(norm_w_2d_inline2173__ssa_v0, [0, 64], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
        gamma_rope_inline2077__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_8, target_type=pl.FP32, mode='round')
        t__tile_9: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.row_expand_mul(kv_rope_norm_inline2135__tile, inv_rms_inline2122__tile)
        rope_normed_inline2138__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_9, gamma_rope_inline2077__tile)
        rope_ones_inline2089__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        t__tile_10: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        t__tile_11: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_10, target_type=pl.FP32, mode='round')
        rope_col_inline2076__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(rope_ones_inline2089__tile, t__tile_11)
        t__tile_12: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_col_inline2076__tile, 0.5)
        t__tile_13: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_12, target_type=pl.INT32, mode='trunc')
        rope_dup_f_inline2074__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_13, target_type=pl.FP32, mode='round')
        t__tile_14: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_dup_f_inline2074__tile, 2.0)
        rope_lane_inline2073__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(rope_col_inline2076__tile, t__tile_14)
        t__tile_15: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(rope_col_inline2076__tile, 1.0)
        t__tile_16: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(rope_lane_inline2073__tile, 2.0)
        t__tile_17: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_15, t__tile_16)
        rope_swap_idx_inline2079__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_17, target_type=pl.INT32, mode='round')
        gather_acc_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for gather_lv, (gather_ia,) in pl.range(16, init_values=(gather_acc_init,)):
            gather_inp_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_normed_inline2138__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(rope_swap_idx_inline2079__tile, [1, 64], [gather_lv, 0], [1, 64])
            gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_14, pl.const(12352, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
            gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_50, pl.const(28800, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
            gather_asmbl: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
            gather_rv: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.yield_(gather_asmbl)
        swapped_inline2071__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_10, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = gather_rv
        t__tile_18: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(rope_normed_inline2138__tile, cos_b_inline2087__tile)
        t__tile_19: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(16448, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.mul(swapped_inline2071__tile, sin_b_inline2085__tile)
        rope_rot_inline2070__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(t__tile_18, t__tile_19)
        t__tile_20: pl.Tile[[16, 64], pl.BF16, pl.MemRef(mem_vec_11, pl.const(4096, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(rope_rot_inline2070__tile, target_type=pl.BF16, mode='rint')
        normed_kv_inline2164__tile_1: pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 131072)] = pl.tile.store(t__tile_20, [b0_inline2090__ssa_v0, 64], normed_kv_inline2164__rv_v2)
        return normed_kv_inline2164__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def rmsnorm_rope_spmd(self, bs_inline2140__ssa_v0: pl.Scalar[pl.INDEX], cmp_cos_il_full_inline1249__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], cmp_sin_signed_full_inline1263__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)], pooled_kv_inline2131__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 262144)], normed_kv_inline2164__ssa_v0: pl.Out[pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 131072)]], norm_w_2d_inline2173__ssa_v0: pl.Tensor[[1, 128], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 256)]) -> pl.Tensor[[512, 128], pl.BF16]:
        normed_kv_inline2164__ssa_v4: pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 131072)] = self.rmsnorm_rope(bs_inline2140__ssa_v0, cmp_cos_il_full_inline1249__rv_v2, cmp_sin_signed_full_inline1263__rv_v2, pooled_kv_inline2131__rv_v2, normed_kv_inline2164__ssa_v0, norm_w_2d_inline2173__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.input]})
        return normed_kv_inline2164__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def rope_cs(rope_swap_idx_inline2314__ssa_v0: pl.Out[pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)]], rope_cos_il_inline2316__ssa_v0: pl.Out[pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 65536)]], rope_sin_signed_inline2315__ssa_v0: pl.Out[pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 65536)]], rope_cs_blocks_inline2380__ssa_v0: pl.Scalar[pl.INDEX], freqs_cos_local__ssa_v0: pl.Tensor[[T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)], freqs_sin_local__ssa_v0: pl.Tensor[[T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]) -> tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_25: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_32: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_34: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)
        mem_vec_36: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_37: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        sw_ones_inline2420__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
        t__tile: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        sw_idx_f_inline2366__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
        sw_col_inline2313__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.col_expand_mul(sw_ones_inline2420__tile, sw_idx_f_inline2366__tile)
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_col_inline2313__tile, 0.5)
        sw_dup_i32_inline2312__tile: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(t__tile_1, target_type=pl.INT32, mode='trunc')
        sw_dup_f_inline2311__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(sw_dup_i32_inline2312__tile, target_type=pl.FP32, mode='round')
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_dup_f_inline2311__tile, 2.0)
        sw_lane_inline2310__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(sw_col_inline2313__tile, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.adds(sw_col_inline2313__tile, 1.0)
        t__tile_4: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(sw_lane_inline2310__tile, 2.0)
        sw_swap_f_inline2363__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.sub(t__tile_3, t__tile_4)
        t__tile_5: pl.Tile[[16, 64], pl.INT32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.cast(sw_swap_f_inline2363__tile, target_type=pl.INT32, mode='round')
        rope_swap_idx_inline2314__tile: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4096)] = pl.tile.store(t__tile_5, [0, 0], rope_swap_idx_inline2314__ssa_v0)
        cs_ones_inline2309__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([8, 64], dtype=pl.FP32, value=1.0)
        t__tile_6: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 256), pl.Mem.Vec] = pl.tile.ci(pl.const(0, pl.INT32), [1, 64], dtype=pl.INT32, descending=False)
        cs_idx_f_inline2345__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 256), pl.Mem.Vec] = pl.tile.cast(t__tile_6, target_type=pl.FP32, mode='round')
        cs_col_inline2308__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.col_expand_mul(cs_ones_inline2309__tile, cs_idx_f_inline2345__tile)
        t__tile_7: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_col_inline2308__tile, 0.5)
        cs_dup_i32_inline2334__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(t__tile_7, target_type=pl.INT32, mode='trunc')
        cs_dup_f_inline2307__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(cs_dup_i32_inline2334__tile, target_type=pl.FP32, mode='round')
        cs_dup_idx_inline2396__tile: pl.Tile[[8, 64], pl.INT32, pl.MemRef(mem_vec_25, pl.const(0, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.cast(cs_dup_f_inline2307__tile, target_type=pl.INT32, mode='round')
        t__tile_8: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_dup_f_inline2307__tile, 2.0)
        cs_lane_inline2352__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.sub(cs_col_inline2308__tile, t__tile_8)
        t__tile_9: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.muls(cs_lane_inline2352__tile, 2.0)
        t__tile_10: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.subs(t__tile_9, 1.0)
        cs_sign_inline2306__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_5, pl.const(4608, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.neg(t__tile_10)
        for cs_rb_inline2341__idx_v0, (rope_cos_il_inline2316__iter_v1, rope_sin_signed_inline2315__iter_v1) in pl.range(rope_cs_blocks_inline2380__ssa_v0, init_values=(rope_cos_il_inline2316__ssa_v0, rope_sin_signed_inline2315__ssa_v0)):
            cs_t0_inline2305__ssa_v0: pl.Scalar[pl.INDEX] = cs_rb_inline2341__idx_v0 * 8
            t__tile_11: pl.Tile[[8, 32], pl.BF16, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(freqs_cos_local__ssa_v0, [cs_t0_inline2305__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec)
            cs_cos_inline2304__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_32, pl.const(2048, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_11, target_type=pl.FP32, mode='round')
            t__tile_12: pl.Tile[[8, 32], pl.BF16, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 512), pl.Mem.Vec] = pl.tile.load(freqs_sin_local__ssa_v0, [cs_t0_inline2305__ssa_v0, 0], [8, 32], [8, 32], target_memory=pl.Mem.Vec)
            cs_sin_inline2303__tile: pl.Tile[[8, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(3072, pl.INT64), 1024), pl.Mem.Vec] = pl.tile.cast(t__tile_12, target_type=pl.FP32, mode='round')
            gather_acc_init: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv, (gather_ia,) in pl.range(8, init_values=(gather_acc_init,)):
                gather_inp_row: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_32, pl.const(2048, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(cs_cos_inline2304__tile, [1, 32], [gather_lv, 0], [1, 32])
                gather_idx_row: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_25, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(cs_dup_idx_inline2396__tile, [1, 64], [gather_lv, 0], [1, 64])
                gather_row_tmp: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_36, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_37, pl.const(4352, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row, gather_idx_row, gather_row_tmp)
                gather_asmbl: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia, gather_row, [gather_lv, 0])
                gather_rv: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl)
            t__tile_13: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = gather_rv
            rope_cos_il_inline2316__tile: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 65536)] = pl.tile.store(t__tile_13, [cs_t0_inline2305__ssa_v0, 0], rope_cos_il_inline2316__iter_v1)
            gather_acc_init_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for gather_lv_1, (gather_ia_1,) in pl.range(8, init_values=(gather_acc_init_1,)):
                gather_inp_row_1: pl.Tile[[1, 32], pl.FP32, pl.MemRef(mem_vec_34, pl.const(3072, pl.INT64), 128), pl.Mem.Vec] = pl.tile.slice(cs_sin_inline2303__tile, [1, 32], [gather_lv_1, 0], [1, 32])
                gather_idx_row_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_25, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.slice(cs_dup_idx_inline2396__tile, [1, 64], [gather_lv_1, 0], [1, 64])
                gather_row_tmp_1: pl.Tile[[1, 64], pl.INT32, pl.MemRef(mem_vec_32, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.create([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                gather_row_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_36, pl.const(4096, pl.INT64), 256), pl.Mem.Vec] = pl.tile.gather(gather_inp_row_1, gather_idx_row_1, gather_row_tmp_1)
                gather_asmbl_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.assemble(gather_ia_1, gather_row_1, [gather_lv_1, 0])
                gather_rv_1: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.yield_(gather_asmbl_1)
            cs_sin_il_inline2409__tile: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = gather_rv_1
            t__tile_14: pl.Tile[[8, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(8704, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.mul(cs_sin_il_inline2409__tile, cs_sign_inline2306__tile)
            rope_sin_signed_inline2315__tile: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 65536)] = pl.tile.store(t__tile_14, [cs_t0_inline2305__ssa_v0, 0], rope_sin_signed_inline2315__iter_v1)
            rope_cos_il_inline2316__rv_v2, rope_sin_signed_inline2315__rv_v2 = pl.yield_(rope_cos_il_inline2316__tile, rope_sin_signed_inline2315__tile)
        return rope_swap_idx_inline2314__ssa_v0, rope_cos_il_inline2316__ssa_v0, rope_sin_signed_inline2315__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool(cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], s_dim_inline2020__ssa_v0: pl.Scalar[pl.INDEX], pooled_kv_inline2008__ssa_v0: pl.Out[pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1048576)]], cmp4_score_proj_pad_inline2019__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2097152)], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)], cmp4_kv_proj_pad_inline2031__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)], cmp_state_table_inline1275__ssa_v0: pl.Tensor[[kv_b_dim_inline1264__ssa_v0, 4], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], compress_state_flat_inline2023__ssa_v0: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[512, 512], pl.FP32]:
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 2048)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        c_idx_inline2049__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        first_pos_b_inline2028__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [c_idx_inline2049__ssa_v0 * s_dim_inline2020__ssa_v0])
        for s_idx_inline2026__idx_v0, (pooled_kv_inline2008__iter_v1,) in pl.range(s_dim_inline2020__ssa_v0, init_values=(pooled_kv_inline2008__ssa_v0,)):
            token_inline2040__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2049__ssa_v0 * s_dim_inline2020__ssa_v0 + s_idx_inline2026__idx_v0
            token_pos_inline2041__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [token_inline2040__ssa_v0])
            t__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.full([1, 512], dtype=pl.FP32, value=0.0)
            pooled_kv_inline2008__tile: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1048576)] = pl.tile.store(t__tile, [token_inline2040__ssa_v0, 0], pooled_kv_inline2008__iter_v1)
            if (pl.cast(token_pos_inline2041__tile, pl.INDEX) + 1) % 4 == 0:
                window_start_inline1995__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline2041__tile, pl.INDEX) - 8 + 1
                for h0_inline2042__idx_v0, (pooled_kv_inline2008__iter_v4,) in pl.range(0, 512, 64, init_values=(pooled_kv_inline2008__tile,)):
                    last_ape_row_inline2044__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2041__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(cmp4_score_proj_pad_inline2019__ssa_v1, [token_inline2040__ssa_v0, h0_inline2042__idx_v0 + 512], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(cmp_ape__ssa_v0, [last_ape_row_inline2044__ssa_v0, h0_inline2042__idx_v0 + 512], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    mi_inline2045__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(mi_inline2045__tile, mi_inline2045__tile)
                    li_inline2029__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_3)
                    oi_inline2046__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(cmp4_kv_proj_pad_inline2031__ssa_v1, [token_inline2040__ssa_v0, h0_inline2042__idx_v0 + 512], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    for state_idx_inline2032__idx_v0, (li_inline2029__iter_v1, mi_inline2045__iter_v1, oi_inline2046__iter_v1) in pl.range(7, init_values=(li_inline2029__tile, mi_inline2045__tile, oi_inline2046__tile)):
                        logical_pos_inline2022__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline1995__ssa_v0 + state_idx_inline2032__idx_v0
                        value_inline2050__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                        score_inline2003__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                        state_half_inline2052__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        if 4 <= state_idx_inline2032__idx_v0:
                            state_half_inline2052__ssa_v1: pl.Scalar[pl.INDEX] = 512
                            state_half_inline2052__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline2052__ssa_v1)
                        else:
                            state_half_inline2052__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline2052__ssa_v0)
                        if 0 <= logical_pos_inline2022__ssa_v0 and logical_pos_inline2022__ssa_v0 < pl.cast(first_pos_b_inline2028__tile, pl.INDEX):
                            ring_row_inline2004__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline2022__ssa_v0 % 8
                            state_page_off_inline2054__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline2004__ssa_v0 // 2
                            state_blk_id_i32_inline1996__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_state_table_inline1275__ssa_v0, [c_idx_inline2049__ssa_v0, state_page_off_inline2054__ssa_v0])
                            if 0 <= pl.cast(state_blk_id_i32_inline1996__tile, pl.INDEX):
                                state_blk_id_inline2056__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline1996__tile, pl.INDEX)
                                state_row_inline2058__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline2056__ssa_v0 * 2 + ring_row_inline2004__ssa_v0 % 2
                                value_inline2050__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(compress_state_flat_inline2023__ssa_v0, [state_row_inline2058__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                score_inline2003__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(compress_state_flat_inline2023__ssa_v0, [state_row_inline2058__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0 + 1024], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                score_inline2003__phi_v2, value_inline2050__phi_v2 = pl.yield_(score_inline2003__tile_1, value_inline2050__tile_1)
                            else:
                                score_inline2003__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(score_inline2003__tile, target_memory=pl.Mem.Vec)
                                value_inline2050__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(value_inline2050__tile, target_memory=pl.Mem.Vec)
                                score_inline2003__phi_v2, value_inline2050__phi_v2 = pl.yield_(score_inline2003__tile_mv, value_inline2050__tile_mv)
                            score_inline2003__phi_v3, value_inline2050__phi_v3 = pl.yield_(score_inline2003__phi_v2, value_inline2050__phi_v2)
                        else:
                            score_inline2003__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(score_inline2003__tile, target_memory=pl.Mem.Vec)
                            value_inline2050__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(value_inline2050__tile, target_memory=pl.Mem.Vec)
                            score_inline2003__phi_v3, value_inline2050__phi_v3 = pl.yield_(score_inline2003__tile_mv_1, value_inline2050__tile_mv_1)
                        if pl.cast(first_pos_b_inline2028__tile, pl.INDEX) <= logical_pos_inline2022__ssa_v0:
                            if logical_pos_inline2022__ssa_v0 <= pl.cast(token_pos_inline2041__tile, pl.INDEX):
                                overlay_token_inline2005__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2049__ssa_v0 * s_dim_inline2020__ssa_v0 + logical_pos_inline2022__ssa_v0 - pl.cast(first_pos_b_inline2028__tile, pl.INDEX)
                                ape_row_inline2034__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline2022__ssa_v0 % 4, pl.INDEX)
                                value_inline2050__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(cmp4_kv_proj_pad_inline2031__ssa_v1, [overlay_token_inline2005__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(cmp4_score_proj_pad_inline2019__ssa_v1, [overlay_token_inline2005__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(cmp_ape__ssa_v0, [ape_row_inline2034__ssa_v0, state_half_inline2052__phi_v2 + h0_inline2042__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                score_inline2003__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                                score_inline2003__phi_v5, value_inline2050__phi_v5 = pl.yield_(score_inline2003__tile_2, value_inline2050__tile_2)
                            else:
                                score_inline2003__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(score_inline2003__phi_v3, target_memory=pl.Mem.Vec)
                                value_inline2050__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(value_inline2050__phi_v3, target_memory=pl.Mem.Vec)
                                score_inline2003__phi_v5, value_inline2050__phi_v5 = pl.yield_(score_inline2003__phi_v3_mv, value_inline2050__phi_v3_mv)
                            score_inline2003__phi_v6, value_inline2050__phi_v6 = pl.yield_(score_inline2003__phi_v5, value_inline2050__phi_v5)
                        else:
                            score_inline2003__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(score_inline2003__phi_v3, target_memory=pl.Mem.Vec)
                            value_inline2050__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(value_inline2050__phi_v3, target_memory=pl.Mem.Vec)
                            score_inline2003__phi_v6, value_inline2050__phi_v6 = pl.yield_(score_inline2003__phi_v3_mv_1, value_inline2050__phi_v3_mv_1)
                        mi_next_inline2059__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(mi_inline2045__iter_v1, score_inline2003__phi_v6)
                        t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(mi_inline2045__iter_v1, mi_next_inline2059__tile)
                        alpha_inline2027__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_6)
                        t__tile_7: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(score_inline2003__phi_v6, mi_next_inline2059__tile)
                        beta_inline1999__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_7)
                        t__tile_8: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(alpha_inline2027__tile, li_inline2029__iter_v1)
                        li_inline2029__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(3584, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_8, beta_inline1999__tile)
                        t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(oi_inline2046__iter_v1, alpha_inline2027__tile)
                        t__tile_10: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(value_inline2050__phi_v6, beta_inline1999__tile)
                        oi_inline2046__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_9, t__tile_10)
                        mi_inline2045__ssa_v3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = mi_next_inline2059__tile
                        mi_inline2045__ssa_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(mi_inline2045__ssa_v3, target_memory=pl.Mem.Vec)
                        li_inline2029__rv_v2, mi_inline2045__rv_v2, oi_inline2046__rv_v2 = pl.yield_(li_inline2029__tile_1, mi_inline2045__ssa_v3_mv, oi_inline2046__tile_1)
                    t__tile_11: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(oi_inline2046__rv_v2, li_inline2029__rv_v2)
                    pooled_kv_inline2008__tile_1: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1048576)] = pl.tile.store(t__tile_11, [token_inline2040__ssa_v0, h0_inline2042__idx_v0], pooled_kv_inline2008__iter_v4)
                    pooled_kv_inline2008__rv_v5: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 1048576)] = pl.yield_(pooled_kv_inline2008__tile_1)
                pooled_kv_inline2008__phi_v7: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 1048576)] = pl.yield_(pooled_kv_inline2008__rv_v5)
            else:
                pooled_kv_inline2008__phi_v7: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 1048576)] = pl.yield_(pooled_kv_inline2008__tile)
            pooled_kv_inline2008__rv_v2: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 1048576)] = pl.yield_(pooled_kv_inline2008__phi_v7)
        return pooled_kv_inline2008__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd(self, cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], s_dim_inline2020__ssa_v0: pl.Scalar[pl.INDEX], pooled_kv_inline2008__ssa_v0: pl.Out[pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1048576)]], cmp4_score_proj_pad_inline2019__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 2097152)], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)], cmp4_kv_proj_pad_inline2031__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 2097152)], cmp_state_table_inline1275__ssa_v0: pl.Tensor[[kv_b_dim_inline1264__ssa_v0, 4], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], compress_state_flat_inline2023__ssa_v0: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[512, 512], pl.FP32]:
        pooled_kv_inline2008__rv_v2: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 1048576)] = self.scatter_softmax_pool(cmp_positions_inline1320__ssa_v0, s_dim_inline2020__ssa_v0, pooled_kv_inline2008__ssa_v0, cmp4_score_proj_pad_inline2019__ssa_v1, cmp_ape__ssa_v0, cmp4_kv_proj_pad_inline2031__ssa_v1, cmp_state_table_inline1275__ssa_v0, compress_state_flat_inline2023__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
        return pooled_kv_inline2008__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def scatter_softmax_pool_0(cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], s_dim_inline2127__ssa_v0: pl.Scalar[pl.INDEX], pooled_kv_inline2131__ssa_v0: pl.Out[pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 262144)]], score_proj_pad_inline2143__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)], kv_proj_pad_inline2129__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)], inner_state_table_inline1324__ssa_v0: pl.Tensor[[kv_b_dim_inline1264__ssa_v0, 4], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], compress_state_flat_inline2139__ssa_v0: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[512, 128], pl.FP32]:
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 512)
        mem_vec_9: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_13: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_14: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_15: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_16: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_17: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_24: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        c_idx_inline2094__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        first_pos_b_inline2144__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [c_idx_inline2094__ssa_v0 * s_dim_inline2127__ssa_v0])
        for s_idx_inline2147__idx_v0, (pooled_kv_inline2131__iter_v1,) in pl.range(s_dim_inline2127__ssa_v0, init_values=(pooled_kv_inline2131__ssa_v0,)):
            token_inline2123__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2094__ssa_v0 * s_dim_inline2127__ssa_v0 + s_idx_inline2147__idx_v0
            token_pos_inline2120__tile: pl.Scalar[pl.INT32] = pl.tensor.read(cmp_positions_inline1320__ssa_v0, [token_inline2123__ssa_v0])
            t__tile: pl.Tile[[1, 128], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 512), pl.Mem.Vec] = pl.tile.full([1, 128], dtype=pl.FP32, value=0.0)
            pooled_kv_inline2131__tile: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 262144)] = pl.tile.store(t__tile, [token_inline2123__ssa_v0, 0], pooled_kv_inline2131__iter_v1)
            if (pl.cast(token_pos_inline2120__tile, pl.INDEX) + 1) % 4 == 0:
                window_start_inline2142__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(token_pos_inline2120__tile, pl.INDEX) - 8 + 1
                for h0_inline2125__idx_v0, (pooled_kv_inline2131__iter_v4,) in pl.range(0, 128, 64, init_values=(pooled_kv_inline2131__tile,)):
                    last_ape_row_inline2113__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(pl.cast(token_pos_inline2120__tile, pl.INDEX) % 4, pl.INDEX)
                    t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(score_proj_pad_inline2143__ssa_v1, [token_inline2123__ssa_v0, h0_inline2125__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(inner_ape__ssa_v0, [last_ape_row_inline2113__ssa_v0, h0_inline2125__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    mi_inline2145__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                    t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(mi_inline2145__tile, mi_inline2145__tile)
                    li_inline2148__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_3)
                    oi_inline2132__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(kv_proj_pad_inline2129__ssa_v1, [token_inline2123__ssa_v0, h0_inline2125__idx_v0 + 128], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                    for state_idx_inline2149__idx_v0, (li_inline2148__iter_v1, mi_inline2145__iter_v1, oi_inline2132__iter_v1) in pl.range(7, init_values=(li_inline2148__tile, mi_inline2145__tile, oi_inline2132__tile)):
                        logical_pos_inline2150__ssa_v0: pl.Scalar[pl.INDEX] = window_start_inline2142__ssa_v0 + state_idx_inline2149__idx_v0
                        value_inline2163__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                        score_inline2156__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.full([1, 64], dtype=pl.FP32, value=-3.4028234663852886e+38)
                        state_half_inline2160__ssa_v0: pl.Scalar[pl.INDEX] = 0
                        if 4 <= state_idx_inline2149__idx_v0:
                            state_half_inline2160__ssa_v1: pl.Scalar[pl.INDEX] = 128
                            state_half_inline2160__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline2160__ssa_v1)
                        else:
                            state_half_inline2160__phi_v2: pl.Scalar[pl.INDEX] = pl.yield_(state_half_inline2160__ssa_v0)
                        if 0 <= logical_pos_inline2150__ssa_v0 and logical_pos_inline2150__ssa_v0 < pl.cast(first_pos_b_inline2144__tile, pl.INDEX):
                            ring_row_inline2159__ssa_v0: pl.Scalar[pl.INDEX] = logical_pos_inline2150__ssa_v0 % 8
                            state_page_off_inline2154__ssa_v0: pl.Scalar[pl.INDEX] = ring_row_inline2159__ssa_v0 // 2
                            state_blk_id_i32_inline2130__tile: pl.Scalar[pl.INT32] = pl.tensor.read(inner_state_table_inline1324__ssa_v0, [c_idx_inline2094__ssa_v0, state_page_off_inline2154__ssa_v0])
                            if 0 <= pl.cast(state_blk_id_i32_inline2130__tile, pl.INDEX):
                                state_blk_id_inline2168__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(state_blk_id_i32_inline2130__tile, pl.INDEX)
                                state_row_inline2166__ssa_v0: pl.Scalar[pl.INDEX] = state_blk_id_inline2168__ssa_v0 * 2 + ring_row_inline2159__ssa_v0 % 2
                                value_inline2163__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(compress_state_flat_inline2139__ssa_v0, [state_row_inline2166__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                score_inline2156__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(compress_state_flat_inline2139__ssa_v0, [state_row_inline2166__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0 + 256], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                score_inline2156__phi_v2, value_inline2163__phi_v2 = pl.yield_(score_inline2156__tile_1, value_inline2163__tile_1)
                            else:
                                score_inline2156__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(score_inline2156__tile, target_memory=pl.Mem.Vec)
                                value_inline2163__tile_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(value_inline2163__tile, target_memory=pl.Mem.Vec)
                                score_inline2156__phi_v2, value_inline2163__phi_v2 = pl.yield_(score_inline2156__tile_mv, value_inline2163__tile_mv)
                            score_inline2156__phi_v3, value_inline2163__phi_v3 = pl.yield_(score_inline2156__phi_v2, value_inline2163__phi_v2)
                        else:
                            score_inline2156__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(score_inline2156__tile, target_memory=pl.Mem.Vec)
                            value_inline2163__tile_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(value_inline2163__tile, target_memory=pl.Mem.Vec)
                            score_inline2156__phi_v3, value_inline2163__phi_v3 = pl.yield_(score_inline2156__tile_mv_1, value_inline2163__tile_mv_1)
                        if pl.cast(first_pos_b_inline2144__tile, pl.INDEX) <= logical_pos_inline2150__ssa_v0:
                            if logical_pos_inline2150__ssa_v0 <= pl.cast(token_pos_inline2120__tile, pl.INDEX):
                                overlay_token_inline2169__ssa_v0: pl.Scalar[pl.INDEX] = c_idx_inline2094__ssa_v0 * s_dim_inline2127__ssa_v0 + logical_pos_inline2150__ssa_v0 - pl.cast(first_pos_b_inline2144__tile, pl.INDEX)
                                ape_row_inline2136__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(logical_pos_inline2150__ssa_v0 % 4, pl.INDEX)
                                value_inline2163__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(kv_proj_pad_inline2129__ssa_v1, [overlay_token_inline2169__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(score_proj_pad_inline2143__ssa_v1, [overlay_token_inline2169__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(inner_ape__ssa_v0, [ape_row_inline2136__ssa_v0, state_half_inline2160__phi_v2 + h0_inline2125__idx_v0], [1, 64], [1, 64], target_memory=pl.Mem.Vec)
                                score_inline2156__tile_2: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
                                score_inline2156__phi_v5, value_inline2163__phi_v5 = pl.yield_(score_inline2156__tile_2, value_inline2163__tile_2)
                            else:
                                score_inline2156__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(score_inline2156__phi_v3, target_memory=pl.Mem.Vec)
                                value_inline2163__phi_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(value_inline2163__phi_v3, target_memory=pl.Mem.Vec)
                                score_inline2156__phi_v5, value_inline2163__phi_v5 = pl.yield_(score_inline2156__phi_v3_mv, value_inline2163__phi_v3_mv)
                            score_inline2156__phi_v6, value_inline2163__phi_v6 = pl.yield_(score_inline2156__phi_v5, value_inline2163__phi_v5)
                        else:
                            score_inline2156__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(score_inline2156__phi_v3, target_memory=pl.Mem.Vec)
                            value_inline2163__phi_v3_mv_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(value_inline2163__phi_v3, target_memory=pl.Mem.Vec)
                            score_inline2156__phi_v6, value_inline2163__phi_v6 = pl.yield_(score_inline2156__phi_v3_mv_1, value_inline2163__phi_v3_mv_1)
                        mi_next_inline2126__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = pl.tile.maximum(mi_inline2145__iter_v1, score_inline2156__phi_v6)
                        t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(mi_inline2145__iter_v1, mi_next_inline2126__tile)
                        alpha_inline2115__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_6)
                        t__tile_7: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.sub(score_inline2156__phi_v6, mi_next_inline2126__tile)
                        beta_inline2170__tile: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_15, pl.const(512, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_7)
                        t__tile_8: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_24, pl.const(1280, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(alpha_inline2115__tile, li_inline2148__iter_v1)
                        li_inline2148__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_9, pl.const(2048, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_8, beta_inline2170__tile)
                        t__tile_9: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_17, pl.const(1024, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(oi_inline2132__iter_v1, alpha_inline2115__tile)
                        t__tile_10: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_14, pl.const(256, pl.INT64), 256), pl.Mem.Vec] = pl.tile.mul(value_inline2163__phi_v6, beta_inline2170__tile)
                        oi_inline2132__tile_1: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_13, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(t__tile_9, t__tile_10)
                        mi_inline2145__ssa_v3: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_16, pl.const(768, pl.INT64), 256), pl.Mem.Vec] = mi_next_inline2126__tile
                        mi_inline2145__ssa_v3_mv: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.move(mi_inline2145__ssa_v3, target_memory=pl.Mem.Vec)
                        li_inline2148__rv_v2, mi_inline2145__rv_v2, oi_inline2132__rv_v2 = pl.yield_(li_inline2148__tile_1, mi_inline2145__ssa_v3_mv, oi_inline2132__tile_1)
                    t__tile_11: pl.Tile[[1, 64], pl.FP32, pl.MemRef(mem_vec_7, pl.const(1536, pl.INT64), 256), pl.Mem.Vec] = pl.tile.div(oi_inline2132__rv_v2, li_inline2148__rv_v2)
                    pooled_kv_inline2131__tile_1: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 262144)] = pl.tile.store(t__tile_11, [token_inline2123__ssa_v0, h0_inline2125__idx_v0], pooled_kv_inline2131__iter_v4)
                    pooled_kv_inline2131__rv_v5: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 262144)] = pl.yield_(pooled_kv_inline2131__tile_1)
                pooled_kv_inline2131__phi_v7: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 262144)] = pl.yield_(pooled_kv_inline2131__rv_v5)
            else:
                pooled_kv_inline2131__phi_v7: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 262144)] = pl.yield_(pooled_kv_inline2131__tile)
            pooled_kv_inline2131__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 262144)] = pl.yield_(pooled_kv_inline2131__phi_v7)
        return pooled_kv_inline2131__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def scatter_softmax_pool_spmd_0(self, cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], s_dim_inline2127__ssa_v0: pl.Scalar[pl.INDEX], pooled_kv_inline2131__ssa_v0: pl.Out[pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 262144)]], score_proj_pad_inline2143__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 524288)], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)], kv_proj_pad_inline2129__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 524288)], inner_state_table_inline1324__ssa_v0: pl.Tensor[[kv_b_dim_inline1264__ssa_v0, 4], pl.INT32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 0)], compress_state_flat_inline2139__ssa_v0: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)]) -> pl.Tensor[[512, 128], pl.FP32]:
        pooled_kv_inline2131__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 262144)] = self.scatter_softmax_pool_0(cmp_positions_inline1320__ssa_v0, s_dim_inline2127__ssa_v0, pooled_kv_inline2131__ssa_v0, score_proj_pad_inline2143__ssa_v1, inner_ape__ssa_v0, kv_proj_pad_inline2129__ssa_v1, inner_state_table_inline1324__ssa_v0, compress_state_flat_inline2139__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
        return pooled_kv_inline2131__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def split_pre_post(t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], inv_rms_inline1463__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], hc_attn_base__ssa_v0: pl.Tensor[[24], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 96)], mixes_raw_inline1505__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], scale0_inline1499__ssa_v0: pl.Scalar[pl.FP32], pre_val_store_inline1529__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], scale1_inline1530__ssa_v0: pl.Scalar[pl.FP32], post_t_inline1277__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]], post_tail_store_inline1544__ssa_v0: pl.InOut[pl.Tensor[[8, 8], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 256)]]) -> tuple[pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32], pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32], pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32]]:
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        mem_vec_11: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 256)
        ob_inline1554__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        t0_inline1476__ssa_v2: pl.Scalar[pl.INDEX] = ob_inline1554__ssa_v0 * 8
        valid_rows_inline1507__ssa_v1: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1568__ssa_v0 - t0_inline1476__ssa_v2, 8)
        inv_col_inline1450__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(256, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(inv_rms_inline1463__ssa_v1, [t0_inline1476__ssa_v2, 0], [8, 1], [8, 1], target_memory=pl.Mem.Vec)
        t__tile: pl.Tile[[8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(288, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(hc_attn_base__ssa_v0, [0], [8], [8], target_memory=pl.Mem.Vec)
        pre_base_inline1470__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(288, pl.INT64), 32), pl.Mem.Vec] = t__tile
        t__tile_1: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v2, 0], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
        t__tile_2: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_1, inv_col_inline1450__tile)
        pre_scaled_inline1447__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(t__tile_2, scale0_inline1499__ssa_v0)
        t__tile_3: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.col_expand(pre_scaled_inline1447__tile, pre_base_inline1470__tile)
        pre_logits_inline1455__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(pre_scaled_inline1447__tile, t__tile_3)
        t__tile_4: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.neg(pre_logits_inline1455__tile)
        t__tile_5: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_4)
        t__tile_6: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.adds(t__tile_5, 1.0)
        pre_sig_inline1452__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.recip(t__tile_6)
        pre_val_inline1448__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.adds(pre_sig_inline1452__tile, 9.9999999999999995e-07)
        pre_val_store_inline1529__tile: pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)] = pl.tile.store(pre_val_inline1448__tile, [t0_inline1476__ssa_v2, 0], pre_val_store_inline1529__ssa_v0)
        t__tile_7: pl.Tile[[8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(288, pl.INT64), 32), pl.Mem.Vec] = pl.tile.load(hc_attn_base__ssa_v0, [4], [8], [8], target_memory=pl.Mem.Vec)
        post_base_inline1472__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_7, pl.const(288, pl.INT64), 32), pl.Mem.Vec] = t__tile_7
        t__tile_8: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.load(mixes_raw_inline1505__ssa_v1, [t0_inline1476__ssa_v2, 4], [8, 8], [8, 8], target_memory=pl.Mem.Vec)
        t__tile_9: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.row_expand_mul(t__tile_8, inv_col_inline1450__tile)
        post_scaled_inline1461__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(t__tile_9, scale1_inline1530__ssa_v0)
        t__tile_10: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.col_expand(post_scaled_inline1461__tile, post_base_inline1472__tile)
        post_logits_inline1506__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.add(post_scaled_inline1461__tile, t__tile_10)
        t__tile_11: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.neg(post_logits_inline1506__tile)
        t__tile_12: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.exp(t__tile_11)
        t__tile_13: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.adds(t__tile_12, 1.0)
        post_sig_inline1444__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_11, pl.const(0, pl.INT64), 256), pl.Mem.Vec] = pl.tile.recip(t__tile_13)
        post_pad_inline1489__tile: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec] = pl.tile.muls(post_sig_inline1444__tile, 2.0)
        if valid_rows_inline1507__ssa_v1 == 8:
            t__tile_14: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[8, 4])] = pl.tile.slice(post_pad_inline1489__tile, [8, 8], [0, 0], [8, 4])
            post_t_inline1277__tile: pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(t__tile_14, [t0_inline1476__ssa_v2, 0], post_t_inline1277__ssa_v0)
            post_t_inline1277__phi_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = pl.yield_(post_t_inline1277__tile)
        else:
            post_tail_store_inline1544__tile: pl.Tensor[[8, 8], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 256)] = pl.tile.store(post_pad_inline1489__tile, [0, 0], post_tail_store_inline1544__ssa_v0)
            post_tile_inline1483__ssa_v0: pl.Tile[[8, 8], pl.FP32, pl.MemRef(mem_vec_8, pl.const(320, pl.INT64), 256), pl.Mem.Vec, pl.TileView(valid_shape=[valid_rows_inline1507__ssa_v1, 4])] = pl.tile.load(post_tail_store_inline1544__tile, [0, 0], [8, 8], [valid_rows_inline1507__ssa_v1, 4], target_memory=pl.Mem.Vec)
            post_t_inline1277__store: pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)] = pl.tile.store(post_tile_inline1483__ssa_v0, [t0_inline1476__ssa_v2, 0], post_t_inline1277__ssa_v0)
            post_t_inline1277__phi_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 0)] = pl.yield_(post_t_inline1277__ssa_v0)
        return pre_val_store_inline1529__ssa_v0, post_t_inline1277__phi_v2, post_t_inline1277__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def split_pre_post_spmd(self, t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX], inv_rms_inline1463__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], hc_attn_base__ssa_v0: pl.Tensor[[24], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 96)], mixes_raw_inline1505__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)], scale0_inline1499__ssa_v0: pl.Scalar[pl.FP32], pre_val_store_inline1529__ssa_v0: pl.Out[pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 0)]], scale1_inline1530__ssa_v0: pl.Scalar[pl.FP32], post_t_inline1277__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 0)]], post_tail_store_inline1544__ssa_v0: pl.InOut[pl.Tensor[[8, 8], pl.FP32, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 256)]]) -> tuple[pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32], pl.Tensor[[T_DYN, 4], pl.FP32]]:
        ret__tmp_v0: pl.Tuple[pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32], pl.Tensor[[T_DYN, 4], pl.FP32], pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32]] = self.split_pre_post(t_dim_inline1568__ssa_v0, inv_rms_inline1463__ssa_v1, hc_attn_base__ssa_v0, mixes_raw_inline1505__ssa_v1, scale0_inline1499__ssa_v0, pre_val_store_inline1529__ssa_v0, scale1_inline1530__ssa_v0, post_t_inline1277__ssa_v0, post_tail_store_inline1544__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.inout, pl.adir.inout]})
        pre_val_store_inline1529__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
        post_t_inline1277__phi_v2: pl.Tensor[[T_DYN, 4], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[1]
        post_t_inline1277__ssa_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[2]
        return pre_val_store_inline1529__ssa_v0, post_t_inline1277__phi_v2
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def tp_o_a_quant(own_a_i8_inline2497__iter_v1: pl.Out[pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1048576)]], own_scale_inline2494__iter_v1: pl.Out[pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 4096)]], own_a_fp32_inline2500__ssa_v3: pl.Tensor[[256, 4096], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)], o_a_col_inline2496__ssa_v0: pl.Scalar[pl.INDEX], local_group_inline2514__idx_v0: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[256, 4096], pl.INT8], pl.Tensor[[4, 256], pl.FP32]]:
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32)
        qz_worker_inline2544__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for qz_blk_inline2546__idx_v0, (own_a_i8_inline2497__iter_v3, own_scale_inline2494__iter_v3) in pl.range(qz_worker_inline2544__ssa_v0, 32, 6, init_values=(own_a_i8_inline2497__iter_v1, own_scale_inline2494__iter_v1)):
            qz_t_inline2547__ssa_v0: pl.Scalar[pl.INDEX] = qz_blk_inline2546__idx_v0 * 8
            qz_rows_inline2501__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(256 - qz_t_inline2547__ssa_v0, 8)
            qz_tile_inline2560__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.load(own_a_fp32_inline2500__ssa_v3, [qz_t_inline2547__ssa_v0, o_a_col_inline2496__ssa_v0], [8, 1024], [8, 1024], target_memory=pl.Mem.Vec)
            t__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.abs(qz_tile_inline2560__tile)
            tmp_tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.create([8, 1024], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_1: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(98304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.row_max(t__tile, tmp_tile)
            qz_amax_inline2524__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_6, pl.const(98304, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [1, 8])
            qz_floor_inline2536__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0001)
            qz_amax_v1_inline2549__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.maximum(qz_floor_inline2536__tile, qz_amax_inline2524__tile)
            qz_max_inline2512__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.full([1, 8], dtype=pl.FP32, value=127.0)
            qz_sq_inline2527__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.div(qz_max_inline2512__tile, qz_amax_v1_inline2549__tile)
            qz_sdq_inline2550__tile: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32), pl.Mem.Vec] = pl.tile.recip(qz_sq_inline2527__tile)
            t__tile_2: pl.Tile[[1, 8], pl.FP32, pl.MemRef(mem_vec_5, pl.const(65536, pl.INT64), 32), pl.Mem.Vec, pl.TileView(valid_shape=[1, qz_rows_inline2501__ssa_v0])] = pl.tile.set_validshape(qz_sdq_inline2550__tile, 1, qz_rows_inline2501__ssa_v0)
            own_scale_inline2494__tile: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 4096)] = pl.tile.store(t__tile_2, [local_group_inline2514__idx_v0, qz_t_inline2547__ssa_v0], own_scale_inline2494__iter_v3)
            qz_sq_col_inline2519__tile: pl.Tile[[8, 1], pl.FP32, pl.MemRef(mem_vec_4, pl.const(32768, pl.INT64), 32), pl.Mem.Vec] = pl.tile.reshape(qz_sq_inline2527__tile, [8, 1])
            qz_scaled_inline2538__tile: pl.Tile[[8, 1024], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.row_expand_mul(qz_tile_inline2560__tile, qz_sq_col_inline2519__tile)
            qz_i32_inline2551__tile: pl.Tile[[8, 1024], pl.INT32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.cast(qz_scaled_inline2538__tile, target_type=pl.INT32, mode='rint')
            qz_f16_inline2552__tile: pl.Tile[[8, 1024], pl.FP16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(qz_i32_inline2551__tile, target_type=pl.FP16, mode='round')
            qz_i8_inline2503__tile: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(qz_f16_inline2552__tile, target_type=pl.INT8, mode='trunc')
            t__tile_3: pl.Tile[[8, 1024], pl.INT8, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec, pl.TileView(valid_shape=[qz_rows_inline2501__ssa_v0, 1024])] = pl.tile.set_validshape(qz_i8_inline2503__tile, qz_rows_inline2501__ssa_v0, 1024)
            own_a_i8_inline2497__tile: pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1048576)] = pl.tile.store(t__tile_3, [qz_t_inline2547__ssa_v0, o_a_col_inline2496__ssa_v0], own_a_i8_inline2497__iter_v3)
            own_a_i8_inline2497__rv_v4, own_scale_inline2494__rv_v4 = pl.yield_(own_a_i8_inline2497__tile, own_scale_inline2494__tile)
        own_a_i8_inline2497__rv_v7: pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 1048576)] = own_a_i8_inline2497__rv_v4
        return own_a_i8_inline2497__iter_v1, own_scale_inline2494__iter_v1
    @pl.function(type=pl.FunctionType.Spmd)
    def tp_o_a_quant_spmd(self, own_a_i8_inline2497__iter_v1: pl.Out[pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 1048576)]], own_scale_inline2494__iter_v1: pl.Out[pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 4096)]], own_a_fp32_inline2500__ssa_v3: pl.Tensor[[256, 4096], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)], o_a_col_inline2496__ssa_v0: pl.Scalar[pl.INDEX], local_group_inline2514__idx_v0: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[256, 4096], pl.INT8]]:
        ret__tmp_v0: pl.Tuple[pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[256, 4096], pl.INT8]] = self.tp_o_a_quant(own_a_i8_inline2497__iter_v1, own_scale_inline2494__iter_v1, own_a_fp32_inline2500__ssa_v3, o_a_col_inline2496__ssa_v0, local_group_inline2514__idx_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.scalar]})
        own_scale_inline2494__rv_v4: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4096)] = ret__tmp_v0[1]
        own_a_i8_inline2497__rv_v7: pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 1048576)] = ret__tmp_v0[0]
        return own_scale_inline2494__iter_v1, own_a_i8_inline2497__iter_v1
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def tp_o_a(attention_row_inline2526__ssa_v0: pl.Scalar[pl.INDEX], o_a_col_inline2496__ssa_v0: pl.Scalar[pl.INDEX], attn_2d_inline2548__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)], wo_a_flat_inline2521__ssa_v0: pl.Tensor[[4096, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 33554432)], own_a_fp32_inline2500__iter_v1: pl.Out[pl.Tensor[[256, 4096], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)]]) -> pl.Tensor[[256, 4096], pl.FP32]:
        mem_mat_3: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_acc_5: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 65536)
        mem_left_6: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_7: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_8: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_9: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_mat_13: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_mat_14: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        pa_unit_inline2493__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pa_rb_inline2491__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline2493__ssa_v0 // 8
        pa_nb_inline2499__ssa_v0: pl.Scalar[pl.INDEX] = pa_unit_inline2493__ssa_v0 - pa_rb_inline2491__ssa_v0 * 8
        pa_t0_inline2489__ssa_v0: pl.Scalar[pl.INDEX] = pa_rb_inline2491__ssa_v0 * 128
        pa_n0_inline2485__ssa_v0: pl.Scalar[pl.INDEX] = pa_nb_inline2499__ssa_v0 * 128
        pa_rows_inline2532__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(256 - pa_t0_inline2489__ssa_v0, 128)
        pa_src_inline2495__ssa_v0: pl.Scalar[pl.INDEX] = attention_row_inline2526__ssa_v0 + pa_t0_inline2489__ssa_v0
        pa_wrow_inline2533__ssa_v0: pl.Scalar[pl.INDEX] = o_a_col_inline2496__ssa_v0 + pa_n0_inline2485__ssa_v0
        pa_x0_inline2535__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 256])] = pl.tile.load(attn_2d_inline2548__ssa_v0, [pa_src_inline2495__ssa_v0, 0], [128, 256], [pa_rows_inline2532__ssa_v0, 256], target_memory=pl.Mem.Mat)
        pa_w0_inline2523__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wo_a_flat_inline2521__ssa_v0, [pa_wrow_inline2533__ssa_v0, 0], [128, 256], [128, 256], target_memory=pl.Mem.Mat)
        pa_w0_inline2523__tile_t: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(pa_w0_inline2523__tile)
        pa_acc_inline2537__tile_l0_init_storage: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        pa_acc_inline2537__tile_l0_init: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(pa_acc_inline2537__tile_l0_init_storage, pa_rows_inline2532__ssa_v0, 128)
        pa_acc_inline2537__tile_l0_a: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768), pl.Mem.Left, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(pa_x0_inline2535__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        pa_acc_inline2537__tile_l0_b: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(pa_w0_inline2523__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right)
        pa_acc_inline2537__tile_l0_a_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(pa_x0_inline2535__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        pa_acc_inline2537__tile_l0_b_1: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(pa_w0_inline2523__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right)
        pa_acc_inline2537__tile_l0_c_acc: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(pa_acc_inline2537__tile_l0_init, pa_acc_inline2537__tile_l0_a, pa_acc_inline2537__tile_l0_b, True)
        pa_acc_inline2537__tile_l0_c_acc_1: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(pa_acc_inline2537__tile_l0_c_acc, pa_acc_inline2537__tile_l0_a_1, pa_acc_inline2537__tile_l0_b_1, False)
        for pa_k0_inline2561__idx_v0, (pa_acc_inline2537__iter_v1,) in pl.range(256, 3840, 512, init_values=(pa_acc_inline2537__tile_l0_c_acc_1,)):
            pa_xk_inline2539__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 256])] = pl.tile.load(attn_2d_inline2548__ssa_v0, [pa_src_inline2495__ssa_v0, pa_k0_inline2561__idx_v0], [128, 256], [pa_rows_inline2532__ssa_v0, 256], target_memory=pl.Mem.Mat)
            pa_wk_inline2540__tile: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wo_a_flat_inline2521__ssa_v0, [pa_wrow_inline2533__ssa_v0, pa_k0_inline2561__idx_v0], [128, 256], [128, 256], target_memory=pl.Mem.Mat)
            pa_xk_inline2539__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_13, pl.const(0, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 256])] = pl.tile.load(attn_2d_inline2548__ssa_v0, [pa_src_inline2495__ssa_v0, pa_k0_inline2561__idx_v0 + 256], [128, 256], [pa_rows_inline2532__ssa_v0, 256], target_memory=pl.Mem.Mat)
            pa_wk_inline2540__tile_1: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wo_a_flat_inline2521__ssa_v0, [pa_wrow_inline2533__ssa_v0, pa_k0_inline2561__idx_v0 + 256], [128, 256], [128, 256], target_memory=pl.Mem.Mat)
            pa_wk_inline2540__tile_t: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(pa_wk_inline2540__tile)
            pa_acc_inline2537__tile_l0_a_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768), pl.Mem.Left, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(pa_xk_inline2539__tile, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            pa_acc_inline2537__tile_l0_b_2: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(pa_wk_inline2540__tile_t, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            pa_acc_inline2537__tile_l0_a_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(pa_xk_inline2539__tile, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            pa_acc_inline2537__tile_l0_b_3: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(pa_wk_inline2540__tile_t, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            pa_acc_inline2537__tile_l0_c_acc_2: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(pa_acc_inline2537__iter_v1, pa_acc_inline2537__tile_l0_a_2, pa_acc_inline2537__tile_l0_b_2)
            pa_acc_inline2537__tile_l0_c_acc_3: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(pa_acc_inline2537__tile_l0_c_acc_2, pa_acc_inline2537__tile_l0_a_3, pa_acc_inline2537__tile_l0_b_3)
            pa_wk_inline2540__tile_t_1: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_14, pl.const(65536, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(pa_wk_inline2540__tile_1)
            pa_acc_inline2537__tile_l0_a_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768), pl.Mem.Left, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(pa_xk_inline2539__tile_1, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            pa_acc_inline2537__tile_l0_b_4: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(pa_wk_inline2540__tile_t_1, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            pa_acc_inline2537__tile_l0_a_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(pa_xk_inline2539__tile_1, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            pa_acc_inline2537__tile_l0_b_5: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(pa_wk_inline2540__tile_t_1, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            pa_acc_inline2537__tile_l0_c_acc_4: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(pa_acc_inline2537__tile_l0_c_acc_3, pa_acc_inline2537__tile_l0_a_4, pa_acc_inline2537__tile_l0_b_4)
            pa_acc_inline2537__tile_l0_c_acc_5: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(pa_acc_inline2537__tile_l0_c_acc_4, pa_acc_inline2537__tile_l0_a_5, pa_acc_inline2537__tile_l0_b_5)
            pa_acc_inline2537__rv_v2_main: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.yield_(pa_acc_inline2537__tile_l0_c_acc_5)
        pa_xk_inline2539__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_3, pl.const(131072, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 256])] = pl.tile.load(attn_2d_inline2548__ssa_v0, [pa_src_inline2495__ssa_v0, 3840], [128, 256], [pa_rows_inline2532__ssa_v0, 256], target_memory=pl.Mem.Mat)
        pa_wk_inline2540__tile_2: pl.Tile[[128, 256], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wo_a_flat_inline2521__ssa_v0, [pa_wrow_inline2533__ssa_v0, 3840], [128, 256], [128, 256], target_memory=pl.Mem.Mat)
        pa_wk_inline2540__tile_t_2: pl.Tile[[256, 128], pl.BF16, pl.MemRef(mem_mat_4, pl.const(196608, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(pa_wk_inline2540__tile_2)
        pa_acc_inline2537__tile_l0_a_6: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_left_6, pl.const(0, pl.INT64), 32768), pl.Mem.Left, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(pa_xk_inline2539__tile_2, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        pa_acc_inline2537__tile_l0_b_6: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_7, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(pa_wk_inline2540__tile_t_2, 0, 0, [128, 128], target_memory=pl.Mem.Right)
        pa_acc_inline2537__tile_l0_a_7: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_left_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Left, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(pa_xk_inline2539__tile_2, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        pa_acc_inline2537__tile_l0_b_7: pl.Tile[[128, 128], pl.BF16, pl.MemRef(mem_right_9, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(pa_wk_inline2540__tile_t_2, 128, 0, [128, 128], target_memory=pl.Mem.Right)
        pa_acc_inline2537__tile_l0_c_acc_6: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(pa_acc_inline2537__rv_v2_main, pa_acc_inline2537__tile_l0_a_6, pa_acc_inline2537__tile_l0_b_6)
        pa_acc_inline2537__tile_l0_c_acc_7: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(pa_acc_inline2537__tile_l0_c_acc_6, pa_acc_inline2537__tile_l0_a_7, pa_acc_inline2537__tile_l0_b_7)
        pa_acc_inline2537__rv_v2: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pa_acc_inline2537__tile_l0_c_acc_7
        pa_valid_inline2542__tile: pl.Tile[[128, 128], pl.FP32, pl.MemRef(mem_acc_5, pl.const(0, pl.INT64), 65536), pl.Mem.Acc, pl.TileView(valid_shape=[pa_rows_inline2532__ssa_v0, 128], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(pa_acc_inline2537__rv_v2, pa_rows_inline2532__ssa_v0, 128)
        own_a_fp32_inline2500__tile: pl.Tensor[[256, 4096], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)] = pl.tile.store(pa_valid_inline2542__tile, [pa_t0_inline2489__ssa_v0, pa_wrow_inline2533__ssa_v0], own_a_fp32_inline2500__iter_v1)
        return own_a_fp32_inline2500__iter_v1
    @pl.function(type=pl.FunctionType.Spmd)
    def tp_o_a_spmd(self, attention_row_inline2526__ssa_v0: pl.Scalar[pl.INDEX], o_a_col_inline2496__ssa_v0: pl.Scalar[pl.INDEX], attn_2d_inline2548__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)], wo_a_flat_inline2521__ssa_v0: pl.Tensor[[4096, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 33554432)], own_a_fp32_inline2500__iter_v1: pl.Out[pl.Tensor[[256, 4096], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4194304)]]) -> pl.Tensor[[256, 4096], pl.FP32]:
        own_a_fp32_inline2500__ssa_v3: pl.Tensor[[256, 4096], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 4194304)] = self.tp_o_a(attention_row_inline2526__ssa_v0, o_a_col_inline2496__ssa_v0, attn_2d_inline2548__ssa_v0, wo_a_flat_inline2521__ssa_v0, own_a_fp32_inline2500__iter_v1, attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
        return own_a_fp32_inline2500__iter_v1
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def tp_o_b_dequant(publish_all_inline2525__iter_v1: pl.Out[pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)]], own_b_i32_inline2528__rv_v2: pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16777216)], own_scale_inline2494__rv_v2: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4096)], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)], owner_inline2504__idx_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[512, 4096], pl.BF16]:
        pl.func_attr({"slot_num": 2})
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_6: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        mem_vec_7: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 32768)
        mem_vec_8: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 64)
        dq_worker_inline2469__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for dq_blk_inline2556__idx_v0, (publish_all_inline2525__iter_v3,) in pl.range(dq_worker_inline2469__ssa_v0, 128, 12, init_values=(publish_all_inline2525__iter_v1,)):
            dq_rb_inline2484__ssa_v0: pl.Scalar[pl.INDEX] = dq_blk_inline2556__idx_v0 // 8
            dq_nb_inline2468__ssa_v0: pl.Scalar[pl.INDEX] = dq_blk_inline2556__idx_v0 - dq_rb_inline2484__ssa_v0 * 8
            dq_row_inline2467__ssa_v0: pl.Scalar[pl.INDEX] = dq_rb_inline2484__ssa_v0 * 16
            dq_n0_inline2545__ssa_v0: pl.Scalar[pl.INDEX] = dq_nb_inline2468__ssa_v0 * 512
            dq_rows_inline2466__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(256 - dq_row_inline2467__ssa_v0, 16)
            dq_acc_inline2464__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.full([16, 512], dtype=pl.FP32, value=0.0)
            for dq_group_inline2463__idx_v0, (dq_acc_inline2464__iter_v1,) in pl.range(0, 4, 2, init_values=(dq_acc_inline2464__tile,)):
                dq_col_inline2488__ssa_v0: pl.Scalar[pl.INDEX] = dq_group_inline2463__idx_v0 * 4096 + dq_n0_inline2545__ssa_v0
                dq_col_inline2488__ssa_v0_1: pl.Scalar[pl.INDEX] = dq_group_inline2463__idx_v0 * 4096 + dq_n0_inline2545__ssa_v0 + 4096
                dq_i32_inline2461__tile: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_vec_5, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 512])] = pl.tile.load(own_b_i32_inline2528__rv_v2, [dq_row_inline2467__ssa_v0, dq_col_inline2488__ssa_v0], [16, 512], [dq_rows_inline2466__ssa_v0, 512], target_memory=pl.Mem.Vec)
                dq_srow_inline2518__tile: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_6, pl.const(65536, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[1, dq_rows_inline2466__ssa_v0])] = pl.tile.load(own_scale_inline2494__rv_v2, [dq_group_inline2463__idx_v0, dq_row_inline2467__ssa_v0], [1, 16], [1, dq_rows_inline2466__ssa_v0], target_memory=pl.Mem.Vec)
                dq_i32_inline2461__tile_1: pl.Tile[[16, 512], pl.INT32, pl.MemRef(mem_vec_7, pl.const(65600, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 512])] = pl.tile.load(own_b_i32_inline2528__rv_v2, [dq_row_inline2467__ssa_v0, dq_col_inline2488__ssa_v0_1], [16, 512], [dq_rows_inline2466__ssa_v0, 512], target_memory=pl.Mem.Vec)
                dq_srow_inline2518__tile_1: pl.Tile[[1, 16], pl.FP32, pl.MemRef(mem_vec_8, pl.const(98368, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[1, dq_rows_inline2466__ssa_v0])] = pl.tile.load(own_scale_inline2494__rv_v2, [dq_group_inline2463__idx_v0 + 1, dq_row_inline2467__ssa_v0], [1, 16], [1, dq_rows_inline2466__ssa_v0], target_memory=pl.Mem.Vec)
                dq_fp32_inline2459__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 512])] = pl.tile.cast(dq_i32_inline2461__tile, target_type=pl.FP32, mode='none')
                dq_scol_inline2462__tile: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_6, pl.const(65536, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 1])] = pl.tile.reshape(dq_srow_inline2518__tile, [16, 1])
                t__tile: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 512])] = pl.tile.row_expand_mul(dq_fp32_inline2459__tile, dq_scol_inline2462__tile)
                dq_acc_inline2464__tile_1: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(32768, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(dq_acc_inline2464__iter_v1, t__tile)
                dq_fp32_inline2459__tile_1: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(65600, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 512])] = pl.tile.cast(dq_i32_inline2461__tile_1, target_type=pl.FP32, mode='none')
                dq_scol_inline2462__tile_1: pl.Tile[[16, 1], pl.FP32, pl.MemRef(mem_vec_8, pl.const(98368, pl.INT64), 64), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 1])] = pl.tile.reshape(dq_srow_inline2518__tile_1, [16, 1])
                t__tile_1: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_7, pl.const(65600, pl.INT64), 32768), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 512])] = pl.tile.row_expand_mul(dq_fp32_inline2459__tile_1, dq_scol_inline2462__tile_1)
                dq_acc_inline2464__tile_2: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.add(dq_acc_inline2464__tile_1, t__tile_1)
                dq_acc_inline2464__rv_v2: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.yield_(dq_acc_inline2464__tile_2)
            t__tile_2: pl.Tile[[512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = pl.tile.load(wo_b_scale__ssa_v0, [dq_n0_inline2545__ssa_v0], [512], [512], target_memory=pl.Mem.Vec)
            dq_wscale_inline2458__tile: pl.Tile[[1, 512], pl.FP32, pl.MemRef(mem_vec_5, pl.const(32768, pl.INT64), 2048), pl.Mem.Vec] = t__tile_2
            t__tile_3: pl.Tile[[16, 512], pl.FP32, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 32768), pl.Mem.Vec] = pl.tile.col_expand_mul(dq_acc_inline2464__rv_v2, dq_wscale_inline2458__tile)
            dq_bf16_inline2457__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(t__tile_3, target_type=pl.BF16, mode='rint')
            dq_stage_inline2456__ssa_v0: pl.Scalar[pl.INDEX] = owner_inline2504__idx_v0 * 256 + dq_row_inline2467__ssa_v0
            t__tile_4: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_vec_4, pl.const(0, pl.INT64), 16384), pl.Mem.Vec, pl.TileView(valid_shape=[dq_rows_inline2466__ssa_v0, 512])] = pl.tile.set_validshape(dq_bf16_inline2457__tile, dq_rows_inline2466__ssa_v0, 512)
            publish_all_inline2525__tile: pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)] = pl.tile.store(t__tile_4, [dq_stage_inline2456__ssa_v0, dq_n0_inline2545__ssa_v0], publish_all_inline2525__iter_v3)
            publish_all_inline2525__rv_v4: pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 4194304)] = pl.yield_(publish_all_inline2525__tile)
        return publish_all_inline2525__iter_v1
    @pl.function(type=pl.FunctionType.Spmd)
    def tp_o_b_dequant_spmd(self, publish_all_inline2525__iter_v1: pl.Out[pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)]], own_b_i32_inline2528__rv_v2: pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 16777216)], own_scale_inline2494__rv_v2: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 4096)], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16384)], owner_inline2504__idx_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[512, 4096], pl.BF16]:
        publish_all_inline2525__rv_v4: pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 4194304)] = self.tp_o_b_dequant(publish_all_inline2525__iter_v1, own_b_i32_inline2528__rv_v2, own_scale_inline2494__rv_v2, wo_b_scale__ssa_v0, owner_inline2504__idx_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar]})
        return publish_all_inline2525__iter_v1
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def tp_o_b_publish(tp_rank__ssa_v0: pl.Scalar[pl.INT32], o_window__ssa_v0: pl.Out[pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)]], group_base__ssa_v0: pl.Scalar[pl.INT32], publish_all_inline2525__rv_v2: pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 4194304)], o_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8)]], o_window_ctx: pld.CommCtx, o_signal_ctx: pld.CommCtx):
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
        pub_worker_inline2453__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for pub_blk_inline2559__idx_v0 in pl.range(pub_worker_inline2453__ssa_v0, 64, 24):
            pub_owner_inline2541__ssa_v0: pl.Scalar[pl.INDEX] = pub_blk_inline2559__idx_v0 // 32
            pub_row_block_inline2460__ssa_v0: pl.Scalar[pl.INDEX] = pub_blk_inline2559__idx_v0 - pub_owner_inline2541__ssa_v0 * 32
            pub_owner_row_inline2482__ssa_v0: pl.Scalar[pl.INDEX] = pub_row_block_inline2460__ssa_v0 * 8
            pub_rows_inline2452__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(256 - pub_owner_row_inline2482__ssa_v0, 8)
            pub_src_row_inline2486__ssa_v0: pl.Scalar[pl.INDEX] = pub_owner_inline2541__ssa_v0 * 256 + pub_owner_row_inline2482__ssa_v0
            pub_dst_row_inline2451__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(tp_rank__ssa_v0, pl.INDEX) * 256 + pub_owner_row_inline2482__ssa_v0
            tput_stage: pl.Tile[[8, 4096], pl.BF16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 65536), pl.Mem.Vec] = pl.tile.create([8, 4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
            pld.tile.put(o_window__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + pub_owner_inline2541__ssa_v0, publish_all_inline2525__rv_v2, tput_stage, [pub_dst_row_inline2451__ssa_v0, 0], [pub_src_row_inline2486__ssa_v0, 0], [pub_rows_inline2452__ssa_v0, 4096], atomic=pl.AtomicType.None_)
            pl.system.fence()
        for notify_owner_inline2517__idx_v0 in pl.range(2):
            if notify_owner_inline2517__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(o_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + notify_owner_inline2517__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
        return
    @pl.function(type=pl.FunctionType.Spmd)
    def tp_o_b_publish_spmd(self, tp_rank__ssa_v0: pl.Scalar[pl.INT32], o_window__ssa_v0: pl.Out[pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)]], group_base__ssa_v0: pl.Scalar[pl.INT32], publish_all_inline2525__rv_v2: pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 4194304)], o_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 8)]], o_window_ctx: pld.CommCtx, o_signal_ctx: pld.CommCtx):
        self.tp_o_b_publish(tp_rank__ssa_v0, o_window__ssa_v0, group_base__ssa_v0, publish_all_inline2525__rv_v2, o_signal__ssa_v0, o_window_ctx, o_signal_ctx, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def tp_o_b(own_b_i32_inline2528__iter_v1: pl.Out[pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], own_a_i8_inline2497__rv_v7: pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1048576)], o_a_col_inline2496__ssa_v0: pl.Scalar[pl.INDEX], wo_b__ssa_v0: pl.Tensor[[4096, 4096], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16777216)], local_group_inline2514__idx_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[256, 16384], pl.INT32]:
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_5: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        mem_right_6: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 65536)
        mem_acc_7: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 131072)
        mem_mat_8: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_10: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 32768)
        mem_mat_11: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_14: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 32768)
        pb_unit_inline2483__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        pb_tb_inline2510__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline2483__ssa_v0 // 8
        pb_db_inline2480__ssa_v0: pl.Scalar[pl.INDEX] = pb_unit_inline2483__ssa_v0 - pb_tb_inline2510__ssa_v0 * 8
        pb_t0_inline2557__ssa_v0: pl.Scalar[pl.INDEX] = pb_tb_inline2510__ssa_v0 * 128
        pb_d0_inline2479__ssa_v0: pl.Scalar[pl.INDEX] = pb_db_inline2480__ssa_v0 * 512
        for pb_n0_inline2478__idx_v0, (own_b_i32_inline2528__iter_v3,) in pl.range(pb_d0_inline2479__ssa_v0, pb_d0_inline2479__ssa_v0 + 512, 256, init_values=(own_b_i32_inline2528__iter_v1,)):
            pb_x0_inline2498__tile: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_11, pl.const(32768, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(own_a_i8_inline2497__rv_v7, [pb_t0_inline2557__ssa_v0, o_a_col_inline2496__ssa_v0], [128, 256], [128, 256], target_memory=pl.Mem.Mat)
            pb_w0_inline2522__tile: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(98304, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wo_b__ssa_v0, [pb_n0_inline2478__idx_v0, o_a_col_inline2496__ssa_v0], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
            pb_w0_inline2522__tile_t: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(98304, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(pb_w0_inline2522__tile)
            pb_x0_inline2498__tile_Left: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_5, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(pb_x0_inline2498__tile, target_memory=pl.Mem.Left)
            pb_w0_inline2522__tile_t_Right: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(pb_w0_inline2522__tile_t, target_memory=pl.Mem.Right)
            pb_acc_inline2507__tile: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul(pb_x0_inline2498__tile_Left, pb_w0_inline2522__tile_t_Right)
            pb_bk_inline2515__ssa_v0: pl.Scalar[pl.INDEX] = o_a_col_inline2496__ssa_v0 + 256
            pb_bk_inline2515__ssa_v0_1: pl.Scalar[pl.INDEX] = o_a_col_inline2496__ssa_v0 + 512
            pb_xk_inline2520__tile: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_8, pl.const(163840, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(own_a_i8_inline2497__rv_v7, [pb_t0_inline2557__ssa_v0, pb_bk_inline2515__ssa_v0], [128, 256], [128, 256], target_memory=pl.Mem.Mat)
            pb_wk_inline2476__tile: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(98304, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wo_b__ssa_v0, [pb_n0_inline2478__idx_v0, pb_bk_inline2515__ssa_v0], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
            pb_xk_inline2520__tile_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_10, pl.const(0, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(own_a_i8_inline2497__rv_v7, [pb_t0_inline2557__ssa_v0, pb_bk_inline2515__ssa_v0_1], [128, 256], [128, 256], target_memory=pl.Mem.Mat)
            pb_wk_inline2476__tile_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_11, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wo_b__ssa_v0, [pb_n0_inline2478__idx_v0, pb_bk_inline2515__ssa_v0_1], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
            pb_wk_inline2476__tile_t: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(98304, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(pb_wk_inline2476__tile)
            pb_xk_inline2520__tile_Left: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_5, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(pb_xk_inline2520__tile, target_memory=pl.Mem.Left)
            pb_wk_inline2476__tile_t_Right: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(pb_wk_inline2476__tile_t, target_memory=pl.Mem.Right)
            pb_acc_inline2507__tile_1: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(pb_acc_inline2507__tile, pb_xk_inline2520__tile_Left, pb_wk_inline2476__tile_t_Right)
            pb_wk_inline2476__tile_t_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_11, pl.const(32768, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(pb_wk_inline2476__tile_1)
            pb_xk_inline2520__tile_Left_1: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_14, pl.const(0, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(pb_xk_inline2520__tile_1, target_memory=pl.Mem.Left)
            pb_wk_inline2476__tile_t_Right_1: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(pb_wk_inline2476__tile_t_1, target_memory=pl.Mem.Right)
            pb_acc_inline2507__tile_2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(pb_acc_inline2507__tile_1, pb_xk_inline2520__tile_Left_1, pb_wk_inline2476__tile_t_Right_1)
            pb_bk_inline2515__ssa_v0_2: pl.Scalar[pl.INDEX] = o_a_col_inline2496__ssa_v0 + 768
            pb_xk_inline2520__tile_2: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_mat_11, pl.const(32768, pl.INT64), 32768), pl.Mem.Mat] = pl.tile.load(own_a_i8_inline2497__rv_v7, [pb_t0_inline2557__ssa_v0, pb_bk_inline2515__ssa_v0_2], [128, 256], [128, 256], target_memory=pl.Mem.Mat)
            pb_wk_inline2476__tile_2: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(98304, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(wo_b__ssa_v0, [pb_n0_inline2478__idx_v0, pb_bk_inline2515__ssa_v0_2], [256, 256], [256, 256], target_memory=pl.Mem.Mat)
            pb_wk_inline2476__tile_t_2: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_mat_4, pl.const(98304, pl.INT64), 65536), pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major)] = pl.tile.transpose_view(pb_wk_inline2476__tile_2)
            pb_xk_inline2520__tile_Left_2: pl.Tile[[128, 256], pl.INT8, pl.MemRef(mem_left_5, pl.const(32768, pl.INT64), 32768), pl.Mem.Left] = pl.tile.move(pb_xk_inline2520__tile_2, target_memory=pl.Mem.Left)
            pb_wk_inline2476__tile_t_Right_2: pl.Tile[[256, 256], pl.INT8, pl.MemRef(mem_right_6, pl.const(0, pl.INT64), 65536), pl.Mem.Right] = pl.tile.move(pb_wk_inline2476__tile_t_2, target_memory=pl.Mem.Right)
            pb_acc_inline2507__tile_3: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pl.tile.matmul_acc(pb_acc_inline2507__tile_2, pb_xk_inline2520__tile_Left_2, pb_wk_inline2476__tile_t_Right_2)
            pb_acc_inline2507__rv_v2: pl.Tile[[128, 256], pl.INT32, pl.MemRef(mem_acc_7, pl.const(0, pl.INT64), 131072), pl.Mem.Acc] = pb_acc_inline2507__tile_3
            pb_col_inline2474__ssa_v0: pl.Scalar[pl.INDEX] = local_group_inline2514__idx_v0 * 4096 + pb_n0_inline2478__idx_v0
            own_b_i32_inline2528__tile: pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)] = pl.tile.store(pb_acc_inline2507__rv_v2, [pb_t0_inline2557__ssa_v0, pb_col_inline2474__ssa_v0], own_b_i32_inline2528__iter_v3)
            own_b_i32_inline2528__rv_v4: pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 16777216)] = pl.yield_(own_b_i32_inline2528__tile)
        return own_b_i32_inline2528__iter_v1
    @pl.function(type=pl.FunctionType.Spmd)
    def tp_o_b_spmd(self, own_b_i32_inline2528__iter_v1: pl.Out[pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 16777216)]], own_a_i8_inline2497__rv_v7: pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1048576)], o_a_col_inline2496__ssa_v0: pl.Scalar[pl.INDEX], wo_b__ssa_v0: pl.Tensor[[4096, 4096], pl.INT8, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 16777216)], local_group_inline2514__idx_v0: pl.Scalar[pl.INDEX]) -> pl.Tensor[[256, 16384], pl.INT32]:
        own_b_i32_inline2528__rv_v4: pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 16777216)] = self.tp_o_b(own_b_i32_inline2528__iter_v1, own_a_i8_inline2497__rv_v7, o_a_col_inline2496__ssa_v0, wo_b__ssa_v0, local_group_inline2514__idx_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.scalar]})
        return own_b_i32_inline2528__iter_v1
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def tp_o_rs_complete(attn_out_inline1284__ssa_v0: pl.InOut[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]], tp_rank__ssa_v0: pl.Scalar[pl.INT32], o_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 8)]], group_base__ssa_v0: pl.Scalar[pl.INT32], o_signal_ctx: pld.CommCtx):
        completion_anchor_inline2442__tile: pl.Scalar[pl.BF16] = pl.tensor.read(attn_out_inline1284__ssa_v0, [0, 0])
        for peer_tp_inline2472__idx_v0 in pl.range(2):
            if peer_tp_inline2472__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(o_signal__ssa_v0, pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline2472__idx_v0, [tp_rank__ssa_v0, 0], pl.const(1, pl.INT32), op=0)
        completion_expected_inline2441__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(25, pl.INT32)
        for source_tp_inline2448__idx_v0 in pl.range(2):
            if source_tp_inline2448__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.wait(o_signal__ssa_v0, [source_tp_inline2448__idx_v0, 0], completion_expected_inline2441__ssa_v0, cmp=1)
        pl.system.cacheinvalid()
        reset_value_inline2471__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(-25, pl.INT32)
        self_rank_inline2440__ssa_v0: pl.Scalar[pl.INT32] = group_base__ssa_v0 + tp_rank__ssa_v0
        for source_tp_inline2470__idx_v0 in pl.range(2):
            if source_tp_inline2470__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.notify(o_signal__ssa_v0, self_rank_inline2440__ssa_v0, [source_tp_inline2470__idx_v0, 0], reset_value_inline2471__ssa_v0, op=0)
        pl.tensor.write(attn_out_inline1284__ssa_v0, [0, 0], completion_anchor_inline2442__tile)
        return
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def tp_o_rs_reduce(o_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)], attn_out_inline1284__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], o_window_ctx: pld.CommCtx) -> pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16]:
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        mem_vec_4: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 8192)
        mem_vec_5: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 16384)
        worker_inline2455__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        for block_inline2473__idx_v0 in pl.range(worker_inline2455__ssa_v0, 256, 48):
            local_row_inline2555__ssa_v0: pl.Scalar[pl.INDEX] = block_inline2473__idx_v0
            d_block_inline2465__ssa_v0: pl.Scalar[pl.INDEX] = block_inline2473__idx_v0 - local_row_inline2555__ssa_v0
            d0_inline2447__ssa_v0: pl.Scalar[pl.INDEX] = d_block_inline2465__ssa_v0 * 4096
            own_partial_inline2446__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_5, pl.const(24576, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(o_window__ssa_v0, [local_row_inline2555__ssa_v0, d0_inline2447__ssa_v0], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
            reduce_acc_inline2445__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(own_partial_inline2446__ssa_v0, target_type=pl.FP32, mode='none')
            source_row_inline2481__ssa_v0: pl.Scalar[pl.INDEX] = local_row_inline2555__ssa_v0 + 256
            source_partial_inline2443__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_4, pl.const(16384, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.load(o_window__ssa_v0, [source_row_inline2481__ssa_v0, d0_inline2447__ssa_v0], [1, 4096], [1, 4096], target_memory=pl.Mem.Vec)
            source_fp32_inline2506__ssa_v0: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_5, pl.const(24576, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.cast(source_partial_inline2443__ssa_v0, target_type=pl.FP32, mode='none')
            reduce_acc_inline2445__ssa_v3: pl.Tile[[1, 4096], pl.FP32, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 16384), pl.Mem.Vec] = pl.tile.add(reduce_acc_inline2445__ssa_v0, source_fp32_inline2506__ssa_v0)
            reduced_inline2553__ssa_v0: pl.Tile[[1, 4096], pl.BF16, pl.MemRef(mem_vec_3, pl.const(0, pl.INT64), 8192), pl.Mem.Vec] = pl.tile.cast(reduce_acc_inline2445__ssa_v3, target_type=pl.BF16, mode='rint')
            attn_out_inline1284__store: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)] = pl.tile.store(reduced_inline2553__ssa_v0, [local_row_inline2555__ssa_v0, d0_inline2447__ssa_v0], attn_out_inline1284__ssa_v0)
        return attn_out_inline1284__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def tp_o_rs_reduce_spmd(self, o_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 4194304)], attn_out_inline1284__ssa_v0: pl.Out[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 0)]], o_window_ctx: pld.CommCtx) -> pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16]:
        attn_out_inline1284__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 0)] = self.tp_o_rs_reduce(o_window__ssa_v0, attn_out_inline1284__ssa_v0, o_window_ctx, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar]})
        return attn_out_inline1284__ssa_v0
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def tp_o_rs_wait(tp_rank__ssa_v0: pl.Scalar[pl.INT32], o_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 8)], o_signal_ctx: pld.CommCtx):
        expected_inline2475__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(24, pl.INT32)
        for source_tp_inline2449__idx_v0 in pl.range(2):
            if source_tp_inline2449__idx_v0 != pl.cast(tp_rank__ssa_v0, pl.INDEX):
                pld.system.wait(o_signal__ssa_v0, [source_tp_inline2449__idx_v0, 0], expected_inline2475__ssa_v0, cmp=1)
        pl.system.cacheinvalid()
        return
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def weights_proj_reduce(weights_partial_inline2245__ssa_v1: pl.Tensor[[1024, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 262144)], weights_inline2244__ssa_v0: pl.Out[pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 65536)]]) -> pl.Tensor[[256, 64], pl.FP32]:
        mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        mem_vec_3: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
        w_rb_v1_inline2235__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        w_r0_v1_inline2213__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_v1_inline2235__ssa_v0 * 16
        w_sum_inline2209__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(weights_partial_inline2245__ssa_v1, [w_r0_v1_inline2213__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        partial_r0_inline2260__ssa_v0: pl.Scalar[pl.INDEX] = w_r0_v1_inline2213__ssa_v0 + 256
        t__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(weights_partial_inline2245__ssa_v1, [partial_r0_inline2260__ssa_v0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        w_sum_inline2209__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(w_sum_inline2209__tile, t__tile)
        partial_r0_inline2260__ssa_v1: pl.Scalar[pl.INDEX] = w_r0_v1_inline2213__ssa_v0 + 512
        t__tile_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(weights_partial_inline2245__ssa_v1, [partial_r0_inline2260__ssa_v1, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        w_sum_inline2209__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(w_sum_inline2209__tile_1, t__tile_1)
        partial_r0_inline2260__ssa_v2: pl.Scalar[pl.INDEX] = w_r0_v1_inline2213__ssa_v0 + 768
        t__tile_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_3, pl.const(4096, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.load(weights_partial_inline2245__ssa_v1, [partial_r0_inline2260__ssa_v2, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec)
        w_sum_inline2209__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.add(w_sum_inline2209__tile_2, t__tile_2)
        t__tile_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_vec_2, pl.const(0, pl.INT64), 4096), pl.Mem.Vec] = pl.tile.muls(w_sum_inline2209__tile_3, 0.011048543456039806)
        weights_inline2244__tile: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 65536)] = pl.tile.store(t__tile_3, [w_r0_v1_inline2213__ssa_v0, 0], weights_inline2244__ssa_v0)
        return weights_inline2244__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_reduce_spmd(self, weights_partial_inline2245__ssa_v1: pl.Tensor[[1024, 64], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 262144)], weights_inline2244__ssa_v0: pl.Out[pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 65536)]]) -> pl.Tensor[[256, 64], pl.FP32]:
        weights_inline2244__ssa_v1: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 65536)] = self.weights_proj_reduce(weights_partial_inline2245__ssa_v1, weights_inline2244__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
        return weights_inline2244__ssa_v0
    @pl.function(type=pl.FunctionType.AIC, level=pl.Level.AIC, role=pl.Role.SubWorker)
    def weights_proj(bs_inline2301__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline2242__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)], weights_partial_inline2245__ssa_v0: pl.Out[pl.Tensor[[1024, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 262144)]]) -> pl.Tensor[[1024, 64], pl.FP32]:
        mem_acc_3: pl.Ptr = pl.tile.alloc(pl.Mem.Acc, 4096)
        mem_mat_4: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 16384)
        mem_mat_5: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 65536)
        mem_left_7: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_8: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        mem_left_9: pl.Ptr = pl.tile.alloc(pl.Mem.Left, 8192)
        mem_right_10: pl.Ptr = pl.tile.alloc(pl.Mem.Right, 32768)
        w_unit_inline2246__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        w_rb_inline2237__ssa_v0: pl.Scalar[pl.INDEX] = w_unit_inline2246__ssa_v0 // 4
        kb_inline2210__ssa_v1: pl.Scalar[pl.INDEX] = w_unit_inline2246__ssa_v0 - w_rb_inline2237__ssa_v0 * 4
        w_r0_inline2207__ssa_v0: pl.Scalar[pl.INDEX] = w_rb_inline2237__ssa_v0 * 16
        w_rows_inline2229__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(bs_inline2301__ssa_v0 - w_r0_inline2207__ssa_v0, 16)
        k_base_inline2248__ssa_v0: pl.Scalar[pl.INDEX] = kb_inline2210__ssa_v1 * 1024
        weights_acc_inline2217__tile: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        for db_inline2297__idx_v0, (weights_acc_inline2217__iter_v1,) in pl.range(2, init_values=(weights_acc_inline2217__tile,)):
            d0_inline2250__ssa_v0: pl.Scalar[pl.INDEX] = k_base_inline2248__ssa_v0 + db_inline2297__idx_v0 * 512
            x_tile_inline2253__tile: pl.Tile[[16, 512], pl.BF16, pl.MemRef(mem_mat_4, pl.const(0, pl.INT64), 16384), pl.Mem.Mat, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 512])] = pl.tile.load(x_flat_inline2242__ssa_v0, [w_r0_inline2207__ssa_v0, d0_inline2250__ssa_v0], [16, 512], [w_rows_inline2229__ssa_v0, 512], target_memory=pl.Mem.Mat)
            weights_proj_tile_inline2255__tile: pl.Tile[[512, 64], pl.BF16, pl.MemRef(mem_mat_5, pl.const(16384, pl.INT64), 65536), pl.Mem.Mat] = pl.tile.load(weights_proj__ssa_v0, [d0_inline2250__ssa_v0, 0], [512, 64], [512, 64], target_memory=pl.Mem.Mat)
            if db_inline2297__idx_v0 == 0:
                weights_acc_inline2217__tile_l0_init_storage: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(compact=pl.CompactMode.normal)] = pl.tile.create([16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
                weights_acc_inline2217__tile_l0_init: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.set_validshape(weights_acc_inline2217__tile_l0_init_storage, w_rows_inline2229__ssa_v0, 64)
                weights_acc_inline2217__tile_l0_a: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_7, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2253__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline2217__tile_l0_b: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(weights_proj_tile_inline2255__tile, 0, 0, [256, 64], target_memory=pl.Mem.Right)
                weights_acc_inline2217__tile_l0_a_1: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_9, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2253__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline2217__tile_l0_b_1: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(weights_proj_tile_inline2255__tile, 256, 0, [256, 64], target_memory=pl.Mem.Right)
                weights_acc_inline2217__tile_l0_c_acc: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(weights_acc_inline2217__tile_l0_init, weights_acc_inline2217__tile_l0_a, weights_acc_inline2217__tile_l0_b, True)
                weights_acc_inline2217__tile_l0_c_acc_1: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.tile.matmul_acc(weights_acc_inline2217__tile_l0_c_acc, weights_acc_inline2217__tile_l0_a_1, weights_acc_inline2217__tile_l0_b_1, False)
                weights_acc_inline2217__phi_v5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.yield_(weights_acc_inline2217__tile_l0_c_acc_1)
            else:
                weights_acc_inline2217__tile_l0_a_2: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_7, pl.const(0, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2253__tile, 0, 0, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline2217__tile_l0_b_2: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_8, pl.const(32768, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(weights_proj_tile_inline2255__tile, 0, 0, [256, 64], target_memory=pl.Mem.Right)
                weights_acc_inline2217__tile_l0_a_3: pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_left_9, pl.const(8192, pl.INT64), 8192), pl.Mem.Left, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 256], blayout=pl.TileLayout.row_major, compact=pl.CompactMode.normal)] = pl.tile.extract(x_tile_inline2253__tile, 0, 256, [16, 256], target_memory=pl.Mem.Left)
                weights_acc_inline2217__tile_l0_b_3: pl.Tile[[256, 64], pl.BF16, pl.MemRef(mem_right_10, pl.const(0, pl.INT64), 32768), pl.Mem.Right] = pl.tile.extract(weights_proj_tile_inline2255__tile, 256, 0, [256, 64], target_memory=pl.Mem.Right)
                weights_acc_inline2217__tile_l0_c_acc_2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(weights_acc_inline2217__iter_v1, weights_acc_inline2217__tile_l0_a_2, weights_acc_inline2217__tile_l0_b_2)
                weights_acc_inline2217__tile_l0_c_acc_3: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.tile.matmul_acc(weights_acc_inline2217__tile_l0_c_acc_2, weights_acc_inline2217__tile_l0_a_3, weights_acc_inline2217__tile_l0_b_3)
                weights_acc_inline2217__phi_v5: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc, pl.TileView(valid_shape=[w_rows_inline2229__ssa_v0, 64], compact=pl.CompactMode.normal)] = pl.yield_(weights_acc_inline2217__tile_l0_c_acc_3)
            weights_acc_inline2217__rv_v2: pl.Tile[[16, 64], pl.FP32, pl.MemRef(mem_acc_3, pl.const(0, pl.INT64), 4096), pl.Mem.Acc] = pl.yield_(weights_acc_inline2217__phi_v5)
        weights_partial_inline2245__tile: pl.Tensor[[1024, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 262144)] = pl.tile.store(weights_acc_inline2217__rv_v2, [kb_inline2210__ssa_v1 * 256 + w_r0_inline2207__ssa_v0, 0], weights_partial_inline2245__ssa_v0)
        return weights_partial_inline2245__ssa_v0
    @pl.function(type=pl.FunctionType.Spmd)
    def weights_proj_spmd(self, bs_inline2301__ssa_v0: pl.Scalar[pl.INDEX], x_flat_inline2242__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 524288)], weights_partial_inline2245__ssa_v0: pl.Out[pl.Tensor[[1024, 64], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 262144)]]) -> pl.Tensor[[1024, 64], pl.FP32]:
        weights_partial_inline2245__ssa_v1: pl.Tensor[[1024, 64], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 262144)] = self.weights_proj(bs_inline2301__ssa_v0, x_flat_inline2242__ssa_v0, weights_proj__ssa_v0, weights_partial_inline2245__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
        return weights_partial_inline2245__ssa_v0
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator, auto_scope=False)
    def decode_csa_test(self, x_hc__ssa_v0: pl.Tensor[[T_DYN, 4, 4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], hc_attn_fn__ssa_v0: pl.Tensor[[24, 16384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 1572864)], hc_attn_scale__ssa_v0: pl.Tensor[[3], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 12)], hc_attn_base__ssa_v0: pl.Tensor[[24], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 96)], attn_norm_w__ssa_v0: pl.Tensor[[4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 8192)], wq_a__ssa_v0: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 8388608)], wq_b__ssa_v0: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 33554432)], wq_b_scale__ssa_v0: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 131072)], wkv__ssa_v0: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 4194304)], gamma_cq__ssa_v0: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 2048)], gamma_ckv__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 1024)], freqs_cos_local__ssa_v0: pl.Tensor[[T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)], freqs_sin_local__ssa_v0: pl.Tensor[[T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)], freqs_cos__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)], freqs_sin__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)], cmp_freqs_cos__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)], cmp_freqs_sin__ssa_v0: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)], cmp_wkv__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 8388608)], cmp_wgate__ssa_v0: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 8388608)], cmp_ape__ssa_v0: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 16384)], cmp_norm_w__ssa_v0: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 1024)], compress_state__ssa_v0: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 0)]], compress_state_block_table__ssa_v0: pl.Tensor[[KV_B_DYN, 4], pl.INT32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 0)], idx_wq_b__ssa_v0: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 8388608)], idx_wq_b_scale__ssa_v0: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 32768)], weights_proj__ssa_v0: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 524288)], hadamard_idx__ssa_v0: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 32768)], inner_wkv__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 2097152)], inner_wgate__ssa_v0: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 2097152)], inner_ape__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 4096)], inner_norm_w__ssa_v0: pl.Tensor[[128], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 256)], inner_compress_state__ssa_v0: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 0)]], inner_compress_state_block_table__ssa_v0: pl.Tensor[[KV_B_DYN, 4], pl.INT32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 0)], kv_cache__ssa_v0: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)]], cmp_kv__ssa_v0: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)]], cmp_block_table__ssa_v0: pl.Tensor[[B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_35", pl.const(0, pl.INT64), 0)], idx_kv_cache__ssa_v0: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)]], idx_kv_scale__ssa_v0: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)]], idx_block_table__ssa_v0: pl.Tensor[[B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 0)], ori_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_39", pl.const(0, pl.INT64), 0)], window_swa_indices__ssa_v0: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 0)], window_swa_lens__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)], cmp_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)], idx_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 0)], state_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 0)], inner_state_slot_mapping__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 0)], position_ids_local__ssa_v0: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 0)], position_ids__ssa_v0: pl.Tensor[[KV_T_DYN], pl.INT32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 0)], kv_seq_lens__ssa_v0: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_48", pl.const(0, pl.INT64), 0)], attn_sink__ssa_v0: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_49", pl.const(0, pl.INT64), 256)], wo_a__ssa_v0: pl.Tensor[[4, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_50", pl.const(0, pl.INT64), 33554432)], wo_b__ssa_v0: pl.Tensor[[4096, 4096], pl.INT8, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 16777216)], wo_b_scale__ssa_v0: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_52", pl.const(0, pl.INT64), 16384)], x_out__ssa_v0: pl.Out[pl.Tensor[[T_DYN, 4, 4096], pl.FP32, pl.MemRef("mem_ddr_53", pl.const(0, pl.INT64), 0)]], gather_window__ssa_v0: pl.Out[pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_54", pl.const(0, pl.INT64), 4194304)]], gather_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 8)]], attention_window__ssa_v0: pl.Out[pld.DistributedTensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 16777216)]], attention_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 8)]], o_window__ssa_v0: pl.Out[pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_58", pl.const(0, pl.INT64), 4194304)]], o_signal__ssa_v0: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_59", pl.const(0, pl.INT64), 8)]], group_base__ssa_v0: pl.Scalar[pl.INT32], tp_rank__ssa_v0: pl.Scalar[pl.INT32], local_t__ssa_v0: pl.Scalar[pl.INT32], gather_window_ctx: pld.CommCtx, gather_signal_ctx: pld.CommCtx, attention_window_ctx: pld.CommCtx, attention_signal_ctx: pld.CommCtx, o_window_ctx: pld.CommCtx, o_signal_ctx: pld.CommCtx):
        with pl.scope():
            t_dim_inline1251__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc__ssa_v0, 0)
            kv_dim_inline1261__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(ori_slot_mapping__ssa_v0, 0)
            kv_b_dim_inline1264__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state_block_table__ssa_v0, 0)
            q_inline1246__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64, 512], pl.BF16, pl.MemRef("mem_ddr_60", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            idx_topk_scores_inline1271__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_61", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            idx_topk_inline1280__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_62", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
            post_t_inline1277__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4], pl.FP32, pl.MemRef("mem_ddr_63", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 4], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            comb_t_inline1267__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_64", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            x_mixed_inline1253__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_65", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            with pl.scope():
                t_dim_inline1568__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_hc__ssa_v0, 0)
                token_tiles_inline1492__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1568__ssa_v0 + 7) // 8
                t_linear_inline1486__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1568__ssa_v0 + 15) // 16 * 16
                x_flat_inline1497__ssa_v0: pl.Tensor[[t_dim_inline1568__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(x_hc__ssa_v0, [t_dim_inline1568__ssa_v0, 16384])
                scale0_inline1499__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale__ssa_v0, [0])
                scale1_inline1530__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale__ssa_v0, [1])
                scale2_inline1480__ssa_v0: pl.Scalar[pl.FP32] = pl.tensor.read(hc_attn_scale__ssa_v0, [2])
                hc_base_2d_inline1467__ssa_v0: pl.Tensor[[1, 24], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 96)] = pl.tensor.reshape(hc_attn_base__ssa_v0, [1, 24])
                inv_rms_inline1463__ssa_v0: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_66", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_linear_inline1486__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0: pl.Tuple[pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.hc_pre_rms_spmd, t_dim_inline1568__ssa_v0, x_flat_inline1497__ssa_v0, inv_rms_inline1463__ssa_v0, core_num=token_tiles_inline1492__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing]})
                inv_rms_inline1463__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_67", pl.const(0, pl.INT64), 0)] = ret__tmp_v0[0]
                tid__ssa_v5: pl.Scalar[pl.TASK_ID] = ret__tmp_v0[1]
                mixes_partials_inline1475__ssa_v0: pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32, pl.MemRef("mem_ddr_68", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_linear_inline1486__ssa_v0 * 4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_1: pl.Tuple[pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.hc_pre_linear_spmd, t_dim_inline1568__ssa_v0, x_flat_inline1497__ssa_v0, hc_attn_fn__ssa_v0, t_linear_inline1486__ssa_v0, mixes_partials_inline1475__ssa_v0, core_num=t_linear_inline1486__ssa_v0 // 16 * 4, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing]})
                mixes_partials_inline1475__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0 * pl.const(4, pl.INDEX), 32], pl.FP32, pl.MemRef("mem_ddr_69", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_1[0]
                tid__ssa_v6: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_1[1]
                mixes_raw_inline1505__ssa_v0: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_70", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_linear_inline1486__ssa_v0, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_2: pl.Tuple[pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.hc_pre_linear_reduce_spmd, mixes_partials_inline1475__ssa_v1, t_linear_inline1486__ssa_v0, mixes_raw_inline1505__ssa_v0, core_num=t_linear_inline1486__ssa_v0 // 16, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.output_existing]})
                mixes_raw_inline1505__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 32], pl.FP32, pl.MemRef("mem_ddr_71", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_2[0]
                tid__ssa_v7: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_2[1]
                pre_val_store_inline1529__ssa_v0: pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32, pl.MemRef("mem_ddr_72", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_linear_inline1486__ssa_v0, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                post_tail_store_inline1544__ssa_v0: pl.Tensor[[8, 8], pl.FP32, pl.MemRef("mem_ddr_73", pl.const(0, pl.INT64), 256)] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_3: pl.Tuple[pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32], pl.Tensor[[T_DYN, 4], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.split_pre_post_spmd, t_dim_inline1568__ssa_v0, inv_rms_inline1463__ssa_v1, hc_attn_base__ssa_v0, mixes_raw_inline1505__ssa_v1, scale0_inline1499__ssa_v0, pre_val_store_inline1529__ssa_v0, scale1_inline1530__ssa_v0, post_t_inline1277__ssa_v0, post_tail_store_inline1544__ssa_v0, core_num=token_tiles_inline1492__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.inout, pl.adir.inout]})
                pre_val_store_inline1529__ssa_v1: pl.Tensor[[t_linear_inline1486__ssa_v0, 8], pl.FP32, pl.MemRef("mem_ddr_74", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_3[0]
                post_t_inline1277__phi_v2: pl.Tensor[[T_DYN, 4], pl.FP32, pl.MemRef("mem_ddr_75", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_3[1]
                tid__ssa_v8: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_3[2]
                comb_tail_store_inline1523__ssa_v0: pl.Tensor[[8, 32], pl.FP32, pl.MemRef("mem_ddr_76", pl.const(0, pl.INT64), 1024)] = pl.tensor.create([8, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_4: pl.Tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.comb_sinkhorn_spmd, t_dim_inline1568__ssa_v0, inv_rms_inline1463__ssa_v1, mixes_raw_inline1505__ssa_v1, hc_base_2d_inline1467__ssa_v0, scale2_inline1480__ssa_v0, comb_t_inline1267__ssa_v0, comb_tail_store_inline1523__ssa_v0, core_num=token_tiles_inline1492__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.inout]})
                comb_t_inline1267__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 16], pl.FP32, pl.MemRef("mem_ddr_77", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_4[0]
                tid__ssa_v9: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_4[1]
                x_mixed_tail_store_inline1462__ssa_v0: pl.Tensor[[8, 4096], pl.BF16, pl.MemRef("mem_ddr_78", pl.const(0, pl.INT64), 65536)] = pl.tensor.create([8, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                ret__tmp_v0_5: pl.Tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.mix_x_spmd, t_dim_inline1568__ssa_v0, pre_val_store_inline1529__ssa_v1, x_mixed_inline1253__ssa_v0, x_mixed_tail_store_inline1462__ssa_v0, x_flat_inline1497__ssa_v0, core_num=token_tiles_inline1492__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.inout, pl.adir.input]})
                x_mixed_inline1253__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_79", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_5[0]
                tid__ssa_v10: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_5[1]
            idx_cos_il_inline1282__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_80", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            idx_sin_signed_inline1307__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_81", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            cmp_cos_il_full_inline1249__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_82", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            cmp_sin_signed_full_inline1263__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_83", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            ret__tmp_v0_6: pl.Tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32], pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32], pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32], pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.csa_rope_interleave, idx_cos_il_inline1282__ssa_v0, idx_sin_signed_inline1307__ssa_v0, t_dim_inline1251__ssa_v0, freqs_cos_local__ssa_v0, freqs_sin_local__ssa_v0, cmp_cos_il_full_inline1249__ssa_v0, cmp_sin_signed_full_inline1263__ssa_v0, kv_dim_inline1261__ssa_v0, cmp_freqs_cos__ssa_v0, cmp_freqs_sin__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]})
            idx_cos_il_inline1282__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_84", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_6[0]
            idx_sin_signed_inline1307__rv_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_85", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_6[1]
            cmp_cos_il_full_inline1249__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_86", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_6[2]
            cmp_sin_signed_full_inline1263__rv_v2: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_87", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_6[3]
            rope_tid_inline1259__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_6[4]
            x_normed_t_inline1243__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_88", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            with pl.scope():
                t_dim_inline1611__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_mixed_inline1253__rv_v2, 0)
                token_tiles_inline1607__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1611__ssa_v0 + 7) // 8
                ret__tmp_v0_7: pl.Tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.rms_norm_spmd, t_dim_inline1611__ssa_v0, x_mixed_inline1253__rv_v2, x_normed_t_inline1243__ssa_v0, attn_norm_w__ssa_v0, core_num=token_tiles_inline1607__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input]})
                x_normed_t_inline1243__phi_v4: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_89", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_7[0]
                rms_tid_inline1605__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_7[1]
            kv_wb_blocks_inline1274__ssa_v0: pl.Scalar[pl.INDEX] = kv_dim_inline1261__ssa_v0 // 8
            x_normed_full_inline1240__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_90", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            with pl.scope():
                local_rows_inline1634__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243__phi_v4, 0)
                local_t_inline1640__ssa_v0: pl.Scalar[pl.INT32] = pl.cast(local_rows_inline1634__ssa_v0, pl.INT32)
                target_row_inline1642__ssa_v0: pl.Scalar[pl.INT32] = tp_rank__ssa_v0 * local_t_inline1640__ssa_v0
                full_local_inline1644__ssa_v0: pl.Scalar[pl.INDEX] = pl.cast(local_t_inline1640__ssa_v0, pl.INDEX) // 8 * 8
                ret__tmp_v0_8: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.cp_token_allgather_push_spmd, full_local_inline1644__ssa_v0, gather_window__ssa_v0, group_base__ssa_v0, x_normed_t_inline1243__phi_v4, target_row_inline1642__ssa_v0, local_t_inline1640__ssa_v0, tp_rank__ssa_v0, gather_signal__ssa_v0, gather_window_ctx, gather_signal_ctx, core_num=16, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
                _push_tid_inline1646__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_8[0]
                ret__tmp_v0_9: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(self.cp_token_allgather_payload_wait, tp_rank__ssa_v0, gather_signal__ssa_v0, gather_signal_ctx, deps=[_push_tid_inline1646__ssa_v0], attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.scalar]})
                _payload_wait_tid_inline1643__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_9[0]
                group_rows_inline1639__ssa_v0: pl.Scalar[pl.INDEX] = local_rows_inline1634__ssa_v0 * 2
                full_rows_inline1641__ssa_v0: pl.Scalar[pl.INDEX] = group_rows_inline1639__ssa_v0 // 16 * 16
                ret__tmp_v0_10: pl.Tuple[pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.cp_token_allgather_readback_spmd, x_normed_full_inline1240__ssa_v0, full_rows_inline1641__ssa_v0, gather_window__ssa_v0, group_rows_inline1639__ssa_v0, tp_rank__ssa_v0, gather_signal__ssa_v0, group_base__ssa_v0, gather_window_ctx, gather_signal_ctx, deps=[_push_tid_inline1646__ssa_v0, _payload_wait_tid_inline1643__ssa_v0], core_num=16, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.scalar, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar]})
                x_normed_full_inline1240__rv_v5: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_91", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_10[0]
                _readback_tid_inline1633__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_10[1]
                ret__tmp_v0_11: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(self.cp_token_allgather_readback_wait, tp_rank__ssa_v0, gather_signal__ssa_v0, gather_signal_ctx, deps=[_readback_tid_inline1633__ssa_v0], attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.scalar]})
                _readback_wait_tid_inline1630__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_11[0]
                ret__tmp_v0_12: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(self.cp_token_allgather_retire, x_normed_full_inline1240__rv_v5, group_base__ssa_v0, tp_rank__ssa_v0, gather_signal__ssa_v0, gather_signal_ctx, deps=[_readback_tid_inline1633__ssa_v0, _readback_wait_tid_inline1630__ssa_v0], attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.inout, pl.adir.scalar]})
                tid__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_12[0]
                _gathered_normed_inline1281__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_92", pl.const(0, pl.INT64), 0)] = x_normed_full_inline1240__rv_v5
                gather_signal__ssa_v1: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_93", pl.const(0, pl.INT64), 8)] = gather_signal__ssa_v0
            kv_full_inline1265__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_94", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 512], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            qr_inline1255__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_95", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
            qr_scale_inline1310__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_96", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            position_ids_t1_inline1288__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 1], pl.INT32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(position_ids_local__ssa_v0, [t_dim_inline1251__ssa_v0, 1])
            attention_local_flat_inline1292__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_97", pl.const(0, pl.INT64), 16777216)] = pl.tensor.create([2048, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            attn_out_inline1284__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_98", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
            with pl.scope():
                late_dep_inline1297__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[rope_tid_inline1259__ssa_v0])
                kv_cos_il_inline1258__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_99", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                kv_sin_signed_inline1301__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_100", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                kv_swap_idx_inline1305__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_101", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                t_dim_inline1682__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_cos__ssa_v0, 0)
                rope_cos_view_inline1679__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(freqs_cos__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
                rope_sin_view_inline1674__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(freqs_sin__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
                rope_cos_il_view_inline1670__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_99", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(kv_cos_il_inline1258__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
                rope_sin_signed_view_inline1668__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_100", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(kv_sin_signed_inline1301__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
                rope_swap_idx_view_inline1694__ssa_v0: pl.Tensor[[t_dim_inline1682__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_101", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(kv_swap_idx_inline1305__ssa_v0, [t_dim_inline1682__ssa_v0, 64])
                token_tiles_inline1673__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1682__ssa_v0 + 7) // 8
                ret__tmp_v0_13: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.q_rope_prepare_spmd, t_dim_inline1682__ssa_v0, rope_cos_view_inline1679__ssa_v0, rope_sin_view_inline1674__ssa_v0, rope_cos_il_view_inline1670__ssa_v0, rope_sin_signed_view_inline1668__ssa_v0, rope_swap_idx_view_inline1694__ssa_v0, core_num=token_tiles_inline1673__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing]})
                tid__ssa_v11: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_13[0]
                q_cos_il_inline1311__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_102", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                q_sin_signed_inline1295__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_103", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                q_swap_idx_inline1313__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_104", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline1251__ssa_v0, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                t_dim_inline1728__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(freqs_cos_local__ssa_v0, 0)
                rope_cos_view_inline1725__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(freqs_cos_local__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
                rope_sin_view_inline1720__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(freqs_sin_local__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
                rope_cos_il_view_inline1716__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_102", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(q_cos_il_inline1311__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
                rope_sin_signed_view_inline1714__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.FP32, pl.MemRef("mem_ddr_103", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(q_sin_signed_inline1295__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
                rope_swap_idx_view_inline1740__ssa_v0: pl.Tensor[[t_dim_inline1728__ssa_v0, 64], pl.INT32, pl.MemRef("mem_ddr_104", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(q_swap_idx_inline1313__ssa_v0, [t_dim_inline1728__ssa_v0, 64])
                token_tiles_inline1719__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline1728__ssa_v0 + 7) // 8
                ret__tmp_v0_14: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.q_rope_prepare_spmd_0, t_dim_inline1728__ssa_v0, rope_cos_view_inline1725__ssa_v0, rope_sin_view_inline1720__ssa_v0, rope_cos_il_view_inline1716__ssa_v0, rope_sin_signed_view_inline1714__ssa_v0, rope_swap_idx_view_inline1740__ssa_v0, core_num=token_tiles_inline1719__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing]})
                tid__ssa_v12: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_14[0]
                t_dim_inline1813__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243__phi_v4, 0)
                for tile_base_inline1799__idx_v0 in pl.range(0, t_dim_inline1813__ssa_v0, 512):
                    with pl.scope():
                        tile_rows_inline1798__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1813__ssa_v0 - tile_base_inline1799__idx_v0, 512)
                        with pl.scope():
                            x_view_inline1797__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_89", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(x_normed_t_inline1243__phi_v4, [t_dim_inline1813__ssa_v0, 4096])
                            qr_t_matmul_inline1793__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1798__ssa_v0 + 15) // 16 * 16
                            qproj_t_matmul_inline1791__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1798__ssa_v0 + 15) // 16 * 16
                            qproj_full_rows_inline1804__ssa_v0: pl.Scalar[pl.INDEX] = tile_rows_inline1798__ssa_v0 // 64 * 64
                            qr_fp32_inline1834__ssa_v0: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_105", pl.const(0, pl.INT64), 0)] = pl.tensor.create([qr_t_matmul_inline1793__ssa_v0, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            qr_fp32_inline1834__rv_v2: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_106", pl.const(0, pl.INT64), 0)] = self.qr_proj_seed(qr_fp32_inline1834__ssa_v0, qr_t_matmul_inline1793__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]})
                            ret__tmp_v0_15: pl.Tuple[pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.qr_proj_matmul_spmd, qr_fp32_inline1834__rv_v2, qr_t_matmul_inline1793__ssa_v0, tile_rows_inline1798__ssa_v0, tile_base_inline1799__idx_v0, x_view_inline1797__ssa_v0, wq_a__ssa_v0, core_num=16, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]})
                            qr_fp32_inline1834__rv_v7: pl.Tensor[[qr_t_matmul_inline1793__ssa_v0, 1024], pl.FP32, pl.MemRef("mem_ddr_107", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_15[0]
                            tid__ssa_v13: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_15[1]
                            qr_view_inline1775__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_95", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(qr_inline1255__ssa_v0, [t_dim_inline1813__ssa_v0, 1024])
                            qr_scale_view_inline1796__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_96", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(qr_scale_inline1310__ssa_v0, [t_dim_inline1813__ssa_v0, 1])
                            qr_i8_matmul_inline1787__ssa_v0: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_108", pl.const(0, pl.INT64), 0)] = pl.tensor.create([qproj_t_matmul_inline1791__ssa_v0, 1024], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                            qr_scale_pad_store_inline1814__ssa_v0: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_109", pl.const(0, pl.INT64), 0)] = pl.tensor.create([qproj_t_matmul_inline1791__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                            qr_token_tiles_inline1878__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1798__ssa_v0 + 7) // 8
                            ret__tmp_v0_16: pl.Tuple[pl.Scalar[pl.INDEX], pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32], pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.qr_rms_norm_quant_spmd, tile_rows_inline1798__ssa_v0, tile_base_inline1799__idx_v0, qr_fp32_inline1834__rv_v7, gamma_cq__ssa_v0, qr_scale_pad_store_inline1814__ssa_v0, qr_scale_view_inline1796__ssa_v0, qr_i8_matmul_inline1787__ssa_v0, qr_view_inline1775__ssa_v0, core_num=qr_token_tiles_inline1878__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout]})
                            out_tg_inline1826__ssa_v0: pl.Scalar[pl.INDEX] = ret__tmp_v0_16[0]
                            qr_scale_pad_store_inline1814__ssa_v1: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_110", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_16[1]
                            qr_i8_matmul_inline1787__rv_v2: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 1024], pl.INT8, pl.MemRef("mem_ddr_111", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_16[2]
                            tid__ssa_v14: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_16[3]
                            q_proj_i32_inline1835__ssa_v0: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_112", pl.const(0, pl.INT64), 0)] = pl.tensor.create([qproj_t_matmul_inline1791__ssa_v0, 32768], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                            q_proj_i32_inline1835__rv_v5: pl.Tensor[[qproj_t_matmul_inline1791__ssa_v0, 32768], pl.INT32, pl.MemRef("mem_ddr_113", pl.const(0, pl.INT64), 0)] = self.qproj_matmul_spmd(q_proj_i32_inline1835__ssa_v0, qproj_full_rows_inline1804__ssa_v0, qr_i8_matmul_inline1787__rv_v2, wq_b__ssa_v0, qproj_t_matmul_inline1791__ssa_v0, tile_rows_inline1798__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.scalar], "core_num": 64})
                            q_flat_inline1856__ssa_v0: pl.Tensor[[t_dim_inline1813__ssa_v0, 32768], pl.BF16, pl.MemRef("mem_ddr_60", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(q_inline1246__ssa_v0, [t_dim_inline1813__ssa_v0, 32768])
                            ret__tmp_v0_17: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.qproj_dequant_rms_nope_rope_spmd, out_tg_inline1826__ssa_v0, q_flat_inline1856__ssa_v0, tile_rows_inline1798__ssa_v0, tile_base_inline1799__idx_v0, qr_scale_pad_store_inline1814__ssa_v1, q_cos_il_inline1311__ssa_v0, q_sin_signed_inline1295__ssa_v0, q_swap_idx_inline1313__ssa_v0, q_proj_i32_inline1835__rv_v5, wq_b_scale__ssa_v0, core_num=16, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
                            tid__ssa_v15: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_17[0]
                t_dim_inline1923__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240__rv_v5, 0)
                for tile_base_inline1954__idx_v0 in pl.range(0, t_dim_inline1923__ssa_v0, 512):
                    with pl.scope():
                        tile_rows_inline1928__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(t_dim_inline1923__ssa_v0 - tile_base_inline1954__idx_v0, 512)
                        with pl.scope():
                            x_view_inline1914__ssa_v0: pl.Tensor[[t_dim_inline1923__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_91", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(x_normed_full_inline1240__rv_v5, [t_dim_inline1923__ssa_v0, 4096])
                            t_matmul_inline1930__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1928__ssa_v0 + 15) // 16 * 16
                            kv_fp32_inline1920__ssa_v0: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_114", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_matmul_inline1930__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            kv_fp32_inline1920__rv_v2: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_115", pl.const(0, pl.INT64), 0)] = self.kv_proj_seed(kv_fp32_inline1920__ssa_v0, t_matmul_inline1930__ssa_v0, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar]})
                            ret__tmp_v0_18: pl.Tuple[pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.kv_proj_matmul_spmd, kv_fp32_inline1920__rv_v2, t_matmul_inline1930__ssa_v0, tile_rows_inline1928__ssa_v0, tile_base_inline1954__idx_v0, x_view_inline1914__ssa_v0, wkv__ssa_v0, deps=[late_dep_inline1297__ssa_v0], core_num=32, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input]})
                            kv_fp32_inline1920__rv_v7: pl.Tensor[[t_matmul_inline1930__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_116", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_18[0]
                            _kv_tid_inline1935__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_18[1]
                            kv_view_inline1909__ssa_v0: pl.Tensor[[t_dim_inline1923__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_94", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(kv_full_inline1265__ssa_v0, [t_dim_inline1923__ssa_v0, 512])
                            kv_token_tiles_inline1972__ssa_v0: pl.Scalar[pl.INDEX] = (tile_rows_inline1928__ssa_v0 + 15) // 16
                            self.kv_rms_norm_rope_spmd(tile_rows_inline1928__ssa_v0, tile_base_inline1954__idx_v0, kv_fp32_inline1920__rv_v7, kv_view_inline1909__ssa_v0, gamma_ckv__ssa_v0, kv_cos_il_inline1258__ssa_v0, kv_sin_signed_inline1301__ssa_v0, kv_swap_idx_inline1305__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input], "core_num": kv_token_tiles_inline1972__ssa_v0})
                ori_block_num_inline1291__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache__ssa_v0, 0)
                kv_cache_flat_inline1312__ssa_v0: pl.Tensor[[ori_block_num_inline1291__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(kv_cache__ssa_v0, [ori_block_num_inline1291__ssa_v0 * 32, 512])
                ret__tmp_v0_19: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.csa_cache_writeback_spmd, kv_cache_flat_inline1312__ssa_v0, ori_slot_mapping__ssa_v0, kv_full_inline1265__ssa_v0, core_num=kv_wb_blocks_inline1274__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.input]})
                ori_cache_write_tid_inline1279__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_19[0]
                cmp_positions_inline1320__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(position_ids__ssa_v0, [kv_dim_inline1261__ssa_v0])
                cmp_slots_inline1296__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(cmp_slot_mapping__ssa_v0, [kv_dim_inline1261__ssa_v0])
                cmp_state_slots_inline1247__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(state_slot_mapping__ssa_v0, [kv_dim_inline1261__ssa_v0])
                idx_slots_inline1322__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(idx_slot_mapping__ssa_v0, [kv_dim_inline1261__ssa_v0])
                idx_positions_inline1323__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(position_ids_local__ssa_v0, [t_dim_inline1251__ssa_v0])
                inner_state_slots_inline1257__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0], pl.INT64, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(inner_state_slot_mapping__ssa_v0, [kv_dim_inline1261__ssa_v0])
                cmp_state_table_inline1275__ssa_v0: pl.Tensor[[kv_b_dim_inline1264__ssa_v0, 4], pl.INT32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(compress_state_block_table__ssa_v0, [kv_b_dim_inline1264__ssa_v0, 4])
                inner_state_table_inline1324__ssa_v0: pl.Tensor[[kv_b_dim_inline1264__ssa_v0, 4], pl.INT32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(inner_compress_state_block_table__ssa_v0, [kv_b_dim_inline1264__ssa_v0, 4])
                cmp_out_inline1299__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_117", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                b_dim_inline2018__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_state_table_inline1275__ssa_v0, 0)
                bs_inline2038__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240__rv_v5, 0)
                s_dim_inline2020__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2038__ssa_v0 // b_dim_inline2018__ssa_v0
                t_matmul_inline2013__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2038__ssa_v0 + 15) // 16 * 16
                rms_blocks_inline2061__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2038__ssa_v0 + 15) // 16
                x_flat_inline2021__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_118", pl.const(0, pl.INT64), 0)] = x_normed_full_inline1240__rv_v5
                cmp4_kv_proj_pad_inline2031__ssa_v0: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_119", pl.const(0, pl.INT64), 2097152)] = pl.tensor.create([512, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                cmp4_score_proj_pad_inline2019__ssa_v0: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_120", pl.const(0, pl.INT64), 2097152)] = pl.tensor.create([512, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                compress_state_block_num_inline2051__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(compress_state__ssa_v0, 0)
                cmp_block_num_inline2055__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
                compress_state_flat_inline2023__ssa_v0: pl.Tensor[[compress_state_block_num_inline2051__ssa_v0 * pl.const(2, pl.INDEX), 2048], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(compress_state__ssa_v0, [compress_state_block_num_inline2051__ssa_v0 * 2, 2048])
                kv_flat_inline2039__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_121", pl.const(0, pl.INT64), 0)] = cmp_out_inline1299__ssa_v0
                cmp_kv_cache_flat_inline2036__ssa_v0: pl.Tensor[[cmp_block_num_inline2055__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline2055__ssa_v0 * 32, 512])
                ret__tmp_v0_20: pl.Tuple[pl.Tensor[[512, 1024], pl.FP32], pl.Tensor[[512, 1024], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.kv_score_proj_spmd, bs_inline2038__ssa_v0, x_flat_inline2021__ssa_v0, cmp_wkv__ssa_v0, cmp_wgate__ssa_v0, cmp4_kv_proj_pad_inline2031__ssa_v0, cmp4_score_proj_pad_inline2019__ssa_v0, deps=[late_dep_inline1297__ssa_v0], core_num=t_matmul_inline2013__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]})
                cmp4_kv_proj_pad_inline2031__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_122", pl.const(0, pl.INT64), 2097152)] = ret__tmp_v0_20[0]
                cmp4_score_proj_pad_inline2019__ssa_v1: pl.Tensor[[512, 1024], pl.FP32, pl.MemRef("mem_ddr_123", pl.const(0, pl.INT64), 2097152)] = ret__tmp_v0_20[1]
                _kv_score_tid_inline2015__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_20[2]
                pooled_kv_inline2008__ssa_v0: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_124", pl.const(0, pl.INT64), 1048576)] = pl.tensor.create([512, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_21: pl.Tuple[pl.Tensor[[512, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.scatter_softmax_pool_spmd, cmp_positions_inline1320__ssa_v0, s_dim_inline2020__ssa_v0, pooled_kv_inline2008__ssa_v0, cmp4_score_proj_pad_inline2019__ssa_v1, cmp_ape__ssa_v0, cmp4_kv_proj_pad_inline2031__ssa_v1, cmp_state_table_inline1275__ssa_v0, compress_state_flat_inline2023__ssa_v0, deps=[_kv_score_tid_inline2015__ssa_v0], core_num=b_dim_inline2018__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
                pooled_kv_inline2008__rv_v2: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_125", pl.const(0, pl.INT64), 1048576)] = ret__tmp_v0_21[0]
                pool_tid_inline1994__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_21[1]
                ret__tmp_v0_22: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.compress_state_commit_spmd, compress_state_flat_inline2023__ssa_v0, s_dim_inline2020__ssa_v0, cmp_state_slots_inline1247__ssa_v0, cmp_positions_inline1320__ssa_v0, cmp4_kv_proj_pad_inline2031__ssa_v1, cmp4_score_proj_pad_inline2019__ssa_v1, cmp_ape__ssa_v0, deps=[pool_tid_inline1994__ssa_v0], core_num=b_dim_inline2018__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
                tid__ssa_v16: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_22[0]
                normed_kv_inline2016__ssa_v0: pl.Tensor[[512, 512], pl.FP32, pl.MemRef("mem_ddr_126", pl.const(0, pl.INT64), 1048576)] = pl.tensor.create([512, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                norm_w_2d_inline2060__ssa_v0: pl.Tensor[[1, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 1024)] = pl.tensor.reshape(cmp_norm_w__ssa_v0, [1, 512])
                ret__tmp_v0_23: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.rmsnorm_rope_cache_write_spmd, bs_inline2038__ssa_v0, cmp_cos_il_full_inline1249__rv_v2, cmp_sin_signed_full_inline1263__rv_v2, pooled_kv_inline2008__rv_v2, normed_kv_inline2016__ssa_v0, norm_w_2d_inline2060__ssa_v0, cmp_kv_cache_flat_inline2036__ssa_v0, kv_flat_inline2039__ssa_v0, cmp_slots_inline1296__ssa_v0, deps=[pool_tid_inline1994__ssa_v0], core_num=rms_blocks_inline2061__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input]})
                cache_write_tid_inline2062__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_23[0]
                cmp_out_inline1299__ssa_v1: pl.Tensor[[kv_dim_inline1261__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_127", pl.const(0, pl.INT64), 0)] = cmp_out_inline1299__ssa_v0
                cmp_cache_write_tid_inline1237__ssa_v0: pl.Scalar[pl.TASK_ID] = cache_write_tid_inline2062__ssa_v0
                cache_ready_dep_inline1304__ssa_v0: pl.Scalar[pl.TASK_ID] = pl.system.task_dummy(deps=[ori_cache_write_tid_inline1279__ssa_v0, cmp_cache_write_tid_inline1237__ssa_v0])
                idx_kv_unused_inline1241__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_128", pl.const(0, pl.INT64), 0)] = pl.tensor.create([kv_dim_inline1261__ssa_v0, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                b_dim_inline2158__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_state_table_inline1324__ssa_v0, 0)
                bs_inline2140__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_full_inline1240__rv_v5, 0)
                s_dim_inline2127__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2140__ssa_v0 // b_dim_inline2158__ssa_v0
                t_matmul_inline2119__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2140__ssa_v0 + 15) // 16 * 16
                rms_blocks_inline2124__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2140__ssa_v0 + 15) // 16
                x_flat_inline2162__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_129", pl.const(0, pl.INT64), 0)] = x_normed_full_inline1240__rv_v5
                kv_proj_pad_inline2129__ssa_v0: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_130", pl.const(0, pl.INT64), 524288)] = pl.tensor.create([512, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                score_proj_pad_inline2143__ssa_v0: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_131", pl.const(0, pl.INT64), 524288)] = pl.tensor.create([512, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                compress_state_block_num_inline2109__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(inner_compress_state__ssa_v0, 0)
                idx_block_num_inline2101__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_kv_cache__ssa_v0, 0)
                compress_state_flat_inline2139__ssa_v0: pl.Tensor[[compress_state_block_num_inline2109__ssa_v0 * pl.const(2, pl.INDEX), 512], pl.FP32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(inner_compress_state__ssa_v0, [compress_state_block_num_inline2109__ssa_v0 * 2, 512])
                kv_flat_inline2098__ssa_v0: pl.Tensor[[kv_dim_inline1261__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_132", pl.const(0, pl.INT64), 0)] = idx_kv_unused_inline1241__ssa_v0
                idx_kv_cache_flat_inline2161__ssa_v0: pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(idx_kv_cache__ssa_v0, [idx_block_num_inline2101__ssa_v0 * 32, 128])
                idx_kv_scale_flat_inline2116__ssa_v0: pl.Tensor[[idx_block_num_inline2101__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(idx_kv_scale__ssa_v0, [idx_block_num_inline2101__ssa_v0 * 32, 1])
                ret__tmp_v0_24: pl.Tuple[pl.Tensor[[512, 256], pl.FP32], pl.Tensor[[512, 256], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.kv_score_proj_spmd_0, bs_inline2140__ssa_v0, x_flat_inline2162__ssa_v0, inner_wkv__ssa_v0, inner_wgate__ssa_v0, kv_proj_pad_inline2129__ssa_v0, score_proj_pad_inline2143__ssa_v0, deps=[late_dep_inline1297__ssa_v0], core_num=t_matmul_inline2119__ssa_v0 // 2, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]})
                kv_proj_pad_inline2129__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_133", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0_24[0]
                score_proj_pad_inline2143__ssa_v1: pl.Tensor[[512, 256], pl.FP32, pl.MemRef("mem_ddr_134", pl.const(0, pl.INT64), 524288)] = ret__tmp_v0_24[1]
                _kv_score_tid_inline2110__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_24[2]
                pooled_kv_inline2131__ssa_v0: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_135", pl.const(0, pl.INT64), 262144)] = pl.tensor.create([512, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_25: pl.Tuple[pl.Tensor[[512, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.scatter_softmax_pool_spmd_0, cmp_positions_inline1320__ssa_v0, s_dim_inline2127__ssa_v0, pooled_kv_inline2131__ssa_v0, score_proj_pad_inline2143__ssa_v1, inner_ape__ssa_v0, kv_proj_pad_inline2129__ssa_v1, inner_state_table_inline1324__ssa_v0, compress_state_flat_inline2139__ssa_v0, deps=[_kv_score_tid_inline2110__ssa_v0], core_num=b_dim_inline2158__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
                pooled_kv_inline2131__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_136", pl.const(0, pl.INT64), 262144)] = ret__tmp_v0_25[0]
                pool_tid_inline2128__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_25[1]
                ret__tmp_v0_26: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.compress_state_commit_spmd_0, compress_state_flat_inline2139__ssa_v0, s_dim_inline2127__ssa_v0, inner_state_slots_inline1257__ssa_v0, cmp_positions_inline1320__ssa_v0, kv_proj_pad_inline2129__ssa_v1, score_proj_pad_inline2143__ssa_v1, inner_ape__ssa_v0, deps=[pool_tid_inline2128__ssa_v0], core_num=b_dim_inline2158__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input]})
                tid__ssa_v17: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_26[0]
                normed_kv_inline2164__ssa_v0: pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_137", pl.const(0, pl.INT64), 131072)] = pl.tensor.create([512, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                norm_w_2d_inline2173__ssa_v0: pl.Tensor[[1, 128], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 256)] = pl.tensor.reshape(inner_norm_w__ssa_v0, [1, 128])
                ret__tmp_v0_27: pl.Tuple[pl.Tensor[[512, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.rmsnorm_rope_spmd, bs_inline2140__ssa_v0, cmp_cos_il_full_inline1249__rv_v2, cmp_sin_signed_full_inline1263__rv_v2, pooled_kv_inline2131__rv_v2, normed_kv_inline2164__ssa_v0, norm_w_2d_inline2173__ssa_v0, deps=[pool_tid_inline2128__ssa_v0], core_num=rms_blocks_inline2124__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing, pl.adir.input]})
                normed_kv_inline2164__ssa_v4: pl.Tensor[[512, 128], pl.BF16, pl.MemRef("mem_ddr_138", pl.const(0, pl.INT64), 131072)] = ret__tmp_v0_27[0]
                rms_tid_inline2093__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_27[1]
                kv_final_inline2118__ssa_v0: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_139", pl.const(0, pl.INT64), 262144)] = pl.tensor.create([512, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_28: pl.Tuple[pl.Tensor[[512, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.kv_hadamard_spmd, normed_kv_inline2164__ssa_v4, kv_final_inline2118__ssa_v0, hadamard_idx__ssa_v0, deps=[rms_tid_inline2093__ssa_v0], core_num=rms_blocks_inline2124__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input]})
                kv_final_inline2118__rv_v2: pl.Tensor[[512, 128], pl.FP32, pl.MemRef("mem_ddr_140", pl.const(0, pl.INT64), 262144)] = ret__tmp_v0_28[0]
                hadamard_tid_inline2075__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_28[1]
                ret__tmp_v0_29: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.kv_and_cache_write_spmd, bs_inline2140__ssa_v0, kv_final_inline2118__rv_v2, idx_kv_cache_flat_inline2161__ssa_v0, kv_flat_inline2098__ssa_v0, idx_slots_inline1322__ssa_v0, idx_kv_scale_flat_inline2116__ssa_v0, deps=[hadamard_tid_inline2075__ssa_v0], core_num=rms_blocks_inline2124__ssa_v0, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.output_existing]})
                _write_tid_inline2106__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_29[0]
                idx_cache_write_tid_inline1316__ssa_v0: pl.Scalar[pl.TASK_ID] = _write_tid_inline2106__ssa_v0
                bs_inline2301__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(x_normed_t_inline1243__phi_v4, 0)
                bs_heads_inline2228__ssa_v0: pl.Scalar[pl.INDEX] = bs_inline2301__ssa_v0 * 64
                row_blocks_inline2258__ssa_v0: pl.Scalar[pl.INDEX] = (bs_inline2301__ssa_v0 + 15) // 16
                qr_acc_pad_inline2225__ssa_v0: pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_141", pl.const(0, pl.INT64), 8388608)] = pl.tensor.create([256, 8192], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_30: pl.Tuple[pl.Tensor[[256, 8192], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.idx_qr_proj_matmul_spmd, bs_inline2301__ssa_v0, qr_acc_pad_inline2225__ssa_v0, qr_inline1255__ssa_v0, idx_wq_b__ssa_v0, core_num=row_blocks_inline2258__ssa_v0 * 8, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.input, pl.adir.input]})
                qr_acc_pad_inline2225__rv_v2: pl.Tensor[[256, 8192], pl.INT32, pl.MemRef("mem_ddr_142", pl.const(0, pl.INT64), 8388608)] = ret__tmp_v0_30[0]
                tid__ssa_v18: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_30[1]
                qr_proj_inline2268__ssa_v0: pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32, pl.MemRef("mem_ddr_143", pl.const(0, pl.INT64), 0)] = pl.tensor.create([bs_inline2301__ssa_v0, 8192], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_31: pl.Tuple[pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.idx_qr_proj_dequant_spmd, idx_wq_b_scale__ssa_v0, qr_proj_inline2268__ssa_v0, bs_inline2301__ssa_v0, qr_acc_pad_inline2225__rv_v2, qr_scale_inline1310__ssa_v0, core_num=8, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]})
                qr_proj_inline2268__rv_v2: pl.Tensor[[bs_inline2301__ssa_v0, 8192], pl.FP32, pl.MemRef("mem_ddr_144", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_31[0]
                tid__ssa_v19: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_31[1]
                qr_proj_flat_inline2295__ssa_v0: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_144", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(qr_proj_inline2268__rv_v2, [bs_heads_inline2228__ssa_v0, 128])
                qr_bf16_inline2223__ssa_v0: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_145", pl.const(0, pl.INT64), 0)] = pl.tensor.create([bs_heads_inline2228__ssa_v0, 128], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                rope_swap_idx_t_inline2189__ssa_v0: pl.Tensor[[32, 64], pl.INT32, pl.MemRef("mem_ddr_146", pl.const(0, pl.INT64), 8192)] = pl.tensor.create([32, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_32: pl.Tuple[pl.Tensor[[32, 64], pl.INT32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.qr_rope_swap_idx, rope_swap_idx_t_inline2189__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.output_existing]})
                rope_swap_idx_t_inline2189__ssa_v1: pl.Tensor[[32, 64], pl.INT32, pl.MemRef("mem_ddr_147", pl.const(0, pl.INT64), 8192)] = ret__tmp_v0_32[0]
                tid__ssa_v1: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_32[1]
                ret__tmp_v0_33: pl.Tuple[pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.qr_rope_spmd, rope_swap_idx_t_inline2189__ssa_v1, idx_cos_il_inline1282__rv_v2, idx_sin_signed_inline1307__rv_v2, qr_proj_flat_inline2295__ssa_v0, qr_bf16_inline2223__ssa_v0, core_num=bs_heads_inline2228__ssa_v0 // 32, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
                qr_bf16_inline2223__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.BF16, pl.MemRef("mem_ddr_148", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_33[0]
                tid__ssa_v20: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_33[1]
                qh_acc_gm_inline2179__ssa_v0: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_149", pl.const(0, pl.INT64), 0)] = pl.tensor.create([bs_heads_inline2228__ssa_v0, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_34: pl.Tuple[pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.qr_hadamard_matmul_spmd, qr_bf16_inline2223__ssa_v1, hadamard_idx__ssa_v0, qh_acc_gm_inline2179__ssa_v0, core_num=bs_heads_inline2228__ssa_v0 // 64, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.output_existing]})
                qh_acc_gm_inline2179__ssa_v1: pl.Tensor[[bs_heads_inline2228__ssa_v0, 128], pl.FP32, pl.MemRef("mem_ddr_150", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_34[0]
                tid__ssa_v21: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_34[1]
                qr_hadamard_i8_inline2177__ssa_v0: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_151", pl.const(0, pl.INT64), 2097152)] = pl.tensor.create([16384, 128], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                qr_hadamard_scale_dq_inline2234__ssa_v0: pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_152", pl.const(0, pl.INT64), 65536)] = pl.tensor.create([16384, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_35: pl.Tuple[pl.Tensor[[16384, 1], pl.FP32], pl.Tensor[[16384, 128], pl.INT8], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.qr_hadamard_quant_spmd, qh_acc_gm_inline2179__ssa_v1, qr_hadamard_scale_dq_inline2234__ssa_v0, qr_hadamard_i8_inline2177__ssa_v0, core_num=bs_heads_inline2228__ssa_v0 // 64, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.output_existing]})
                qr_hadamard_scale_dq_inline2234__ssa_v1: pl.Tensor[[16384, 1], pl.FP32, pl.MemRef("mem_ddr_153", pl.const(0, pl.INT64), 65536)] = ret__tmp_v0_35[0]
                qr_hadamard_i8_inline2177__rv_v2: pl.Tensor[[16384, 128], pl.INT8, pl.MemRef("mem_ddr_154", pl.const(0, pl.INT64), 2097152)] = ret__tmp_v0_35[1]
                qh_quant_tid_inline2182__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_35[2]
                x_flat_inline2242__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_155", pl.const(0, pl.INT64), 0)] = x_normed_t_inline1243__phi_v4
                weights_inline2244__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_156", pl.const(0, pl.INT64), 65536)] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                weights_partial_inline2245__ssa_v0: pl.Tensor[[1024, 64], pl.FP32, pl.MemRef("mem_ddr_157", pl.const(0, pl.INT64), 262144)] = pl.tensor.create([1024, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_36: pl.Tuple[pl.Tensor[[1024, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.weights_proj_spmd, bs_inline2301__ssa_v0, x_flat_inline2242__ssa_v0, weights_proj__ssa_v0, weights_partial_inline2245__ssa_v0, deps=[late_dep_inline1297__ssa_v0], core_num=row_blocks_inline2258__ssa_v0 * 4, attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
                weights_partial_inline2245__ssa_v1: pl.Tensor[[1024, 64], pl.FP32, pl.MemRef("mem_ddr_158", pl.const(0, pl.INT64), 262144)] = ret__tmp_v0_36[0]
                _weights_tid_inline2286__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_36[1]
                ret__tmp_v0_37: pl.Tuple[pl.Tensor[[256, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.weights_proj_reduce_spmd, weights_partial_inline2245__ssa_v1, weights_inline2244__ssa_v0, core_num=row_blocks_inline2258__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                weights_inline2244__ssa_v1: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_159", pl.const(0, pl.INT64), 65536)] = ret__tmp_v0_37[0]
                weights_tid_inline2256__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_37[1]
                bs_inline61_inline2238__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_positions_inline1323__ssa_v0, 0)
                b_dim_inline57_inline2261__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_block_table__ssa_v0, 0)
                idx_block_num_inline53_inline2264__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(idx_kv_cache__ssa_v0, 0)
                idx_table_len_inline55_inline2193__ssa_v0: pl.Scalar[pl.INDEX] = b_dim_inline57_inline2261__ssa_v0 * 8192
                kv_cache_i8_flat_inline46_inline2265__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 128], pl.INT8, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(idx_kv_cache__ssa_v0, [idx_block_num_inline53_inline2264__ssa_v0 * 32, 128])
                kv_scale_flat_inline50_inline2214__ssa_v0: pl.Tensor[[idx_block_num_inline53_inline2264__ssa_v0 * pl.const(32, pl.INDEX), 1], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(idx_kv_scale__ssa_v0, [idx_block_num_inline53_inline2264__ssa_v0 * 32, 1])
                idx_block_table_flat_inline47_inline2186__ssa_v0: pl.Tensor[[idx_table_len_inline55_inline2193__ssa_v0], pl.INT32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(idx_block_table__ssa_v0, [idx_table_len_inline55_inline2193__ssa_v0])
                pair_arena_inline71_inline2266__ssa_v0: pl.Tensor[[4192, 1024], pl.FP32, pl.MemRef("mem_ddr_160", pl.const(0, pl.INT64), 17170432)] = pl.tensor.create([4192, 1024], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                with pl.scope():
                    score_arena_inline44_inline2267__ssa_v0: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_161", pl.const(0, pl.INT64), 0)] = pl.tensor.create([bs_inline61_inline2238__ssa_v0, 262144], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_162", pl.const(0, pl.INT64), 4)] = pl.tensor.create([1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                    ret__tmp_v0_38: pl.Tuple[pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.indexer_score_leaf_wave_spmd, idx_positions_inline1323__ssa_v0, score_arena_inline44_inline2267__ssa_v0, kv_seq_lens__ssa_v0, qr_hadamard_i8_inline2177__rv_v2, qr_hadamard_scale_dq_inline2234__ssa_v1, weights_inline2244__ssa_v1, idx_block_table_flat_inline47_inline2186__ssa_v0, kv_cache_i8_flat_inline46_inline2265__ssa_v0, kv_scale_flat_inline50_inline2214__ssa_v0, gm_pipe_buffer_0, deps=[qh_quant_tid_inline2182__ssa_v0, weights_tid_inline2256__ssa_v0, idx_cache_write_tid_inline1316__ssa_v0], core_num=24, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
                    score_arena_inline44_inline2267__rv_v2: pl.Tensor[[bs_inline61_inline2238__ssa_v0, 262144], pl.FP32, pl.MemRef("mem_ddr_163", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_38[0]
                    score_tid_inline45_inline2269__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_38[1]
                    ret__tmp_v0_39: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.decode_csa_test_indexer_topk_group_wave, idx_positions_inline1323__ssa_v0, kv_seq_lens__ssa_v0, score_arena_inline44_inline2267__rv_v2, pair_arena_inline71_inline2266__ssa_v0, deps=[score_tid_inline45_inline2269__ssa_v0], core_num=48, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout]})
                    topk_tid_inline36_inline2174__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_39[0]
                    ret__tmp_v0_40: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.decode_csa_test_indexer_topk_query_merge, idx_positions_inline1323__ssa_v0, kv_seq_lens__ssa_v0, pair_arena_inline71_inline2266__ssa_v0, idx_topk_scores_inline1271__ssa_v0, idx_topk_inline1280__ssa_v0, deps=[topk_tid_inline36_inline2174__ssa_v0], core_num=bs_inline61_inline2238__ssa_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.output_existing, pl.adir.output_existing]})
                    _score_tid_inline29_inline2247__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_40[0]
                idx_topk_scores_inline1271__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_164", pl.const(0, pl.INT64), 0)] = idx_topk_scores_inline1271__ssa_v0
                idx_topk_inline1280__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_165", pl.const(0, pl.INT64), 0)] = idx_topk_inline1280__ssa_v0
                idx_topk_scores_inline1271__ssa_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_166", pl.const(0, pl.INT64), 0)] = idx_topk_scores_inline1271__ssa_v1
                idx_topk_inline1280__ssa_v2: pl.Tensor[[t_dim_inline1251__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_167", pl.const(0, pl.INT64), 0)] = idx_topk_inline1280__ssa_v1
                ori_block_num_inline2362__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(kv_cache__ssa_v0, 0)
                t_dim_inline2369__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(q_inline1246__ssa_v0, 0)
                t_heads_inline2364__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2369__ssa_v0 * 64
                t_blk_inline2373__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2369__ssa_v0 * 320
                qk_items_inline2347__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2369__ssa_v0 * 5
                rope_cs_blocks_inline2380__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline2369__ssa_v0 // 8
                ori_kv_flat_inline2344__ssa_v0: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(kv_cache__ssa_v0, [ori_block_num_inline2362__ssa_v0 * 32, 512])
                ret__tmp_v0_41: pl.Tuple[pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.submit(self.kv_touch, ori_kv_flat_inline2344__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.inout]})
                ori_kv_flat_inline2344__ssa_v1: pl.Tensor[[ori_block_num_inline2362__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_168", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_41[0]
                tid__ssa_v2: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_41[1]
                sparse_bias_inline2381__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_169", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline2369__ssa_v0, 640], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                cmp_sparse_indices_inline2383__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_170", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline2369__ssa_v0, 512], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                valid_block_mask_inline2385__ssa_v0: pl.Tensor[[t_dim_inline2369__ssa_v0, 5], pl.INT32, pl.MemRef("mem_ddr_171", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_dim_inline2369__ssa_v0, 5], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                qk_order_inline2351__ssa_v0: pl.Tensor[[1280], pl.INT32, pl.MemRef("mem_ddr_172", pl.const(0, pl.INT64), 5120)] = pl.tensor.create([1280], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                qk_wcur_inline2412__ssa_v0: pl.Tensor[[1], pl.INT32, pl.MemRef("mem_ddr_173", pl.const(0, pl.INT64), 4)] = pl.tensor.create([1], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_42: pl.Tuple[pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32], pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.csa_slots_build_valid_qk_plan, cmp_sparse_indices_inline2383__ssa_v0, sparse_bias_inline2381__ssa_v0, t_dim_inline2369__ssa_v0, idx_topk_inline1280__ssa_v2, position_ids_t1_inline1288__ssa_v0, valid_block_mask_inline2385__ssa_v0, window_swa_indices__ssa_v0, qk_wcur_inline2412__ssa_v0, qk_order_inline2351__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.inout, pl.adir.output_existing]})
                cmp_sparse_indices_inline2383__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 512], pl.INT32, pl.MemRef("mem_ddr_174", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_42[0]
                sparse_bias_inline2381__rv_v2: pl.Tensor[[t_dim_inline2369__ssa_v0, 640], pl.FP32, pl.MemRef("mem_ddr_175", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_42[1]
                qk_plan_tid_inline2387__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_42[2]
                cmp_block_num_inline2376__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(cmp_kv__ssa_v0, 0)
                cmp_kv_flat_inline2401__ssa_v0: pl.Tensor[[cmp_block_num_inline2376__ssa_v0 * pl.const(32, pl.INDEX), 512], pl.BF16, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(cmp_kv__ssa_v0, [cmp_block_num_inline2376__ssa_v0 * 32, 512])
                q_flat_inline2355__ssa_v0: pl.Tensor[[t_heads_inline2364__ssa_v0, 512], pl.BF16, pl.MemRef("mem_ddr_60", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(q_inline1246__ssa_v0, [t_heads_inline2364__ssa_v0, 512])
                sparse_blk_mi_inline2404__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_176", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_blk_inline2373__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                sparse_blk_li_inline2405__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_177", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_blk_inline2373__ssa_v0, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                sparse_blk_oi_inline2398__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_178", pl.const(0, pl.INT64), 0)] = pl.tensor.create([t_blk_inline2373__ssa_v0, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                gm_pipe_buffer_1: pl.Tensor[[1], pl.FP32, pl.MemRef("mem_ddr_179", pl.const(0, pl.INT64), 4)] = pl.tensor.create([1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                ret__tmp_v0_43: pl.Tuple[pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32], pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.qk_pv_spmd, qk_items_inline2347__ssa_v0, sparse_blk_li_inline2405__ssa_v0, sparse_blk_mi_inline2404__ssa_v0, sparse_blk_oi_inline2398__ssa_v0, qk_order_inline2351__ssa_v0, sparse_bias_inline2381__rv_v2, valid_block_mask_inline2385__ssa_v0, position_ids_t1_inline1288__ssa_v0, window_swa_indices__ssa_v0, ori_kv_flat_inline2344__ssa_v1, cmp_sparse_indices_inline2383__rv_v2, cmp_block_table__ssa_v0, cmp_kv_flat_inline2401__ssa_v0, q_flat_inline2355__ssa_v0, gm_pipe_buffer_1, deps=[qk_plan_tid_inline2387__ssa_v0, cache_ready_dep_inline1304__ssa_v0], core_num=24, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.output_existing]})
                sparse_blk_li_inline2405__rv_v2: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_180", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_43[0]
                sparse_blk_mi_inline2404__rv_v2: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_181", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_43[1]
                sparse_blk_oi_inline2398__rv_v2: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_182", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_43[2]
                qk_tid_inline2349__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_43[3]
                rope_cos_il_inline2316__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_183", pl.const(0, pl.INT64), 65536)] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                rope_sin_signed_inline2315__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_184", pl.const(0, pl.INT64), 65536)] = pl.tensor.create([256, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                rope_swap_idx_inline2314__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_185", pl.const(0, pl.INT64), 4096)] = pl.tensor.create([16, 64], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_44: pl.Tuple[pl.Tensor[[16, 64], pl.INT32], pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32], pl.Scalar[pl.TASK_ID]] = pl.submit(self.rope_cs, rope_swap_idx_inline2314__ssa_v0, rope_cos_il_inline2316__ssa_v0, rope_sin_signed_inline2315__ssa_v0, rope_cs_blocks_inline2380__ssa_v0, freqs_cos_local__ssa_v0, freqs_sin_local__ssa_v0, allow_early_resolve=True, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input]})
                rope_swap_idx_inline2314__ssa_v1: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_186", pl.const(0, pl.INT64), 4096)] = ret__tmp_v0_44[0]
                rope_cos_il_inline2316__rv_v2: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_187", pl.const(0, pl.INT64), 65536)] = ret__tmp_v0_44[1]
                rope_sin_signed_inline2315__rv_v2: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_188", pl.const(0, pl.INT64), 65536)] = ret__tmp_v0_44[2]
                rope_tid_inline2402__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_44[3]
                sparse_blk_mi_inline1234__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_189", pl.const(0, pl.INT64), 0)] = sparse_blk_mi_inline2404__rv_v2
                sparse_blk_li_inline1283__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 1], pl.FP32, pl.MemRef("mem_ddr_190", pl.const(0, pl.INT64), 0)] = sparse_blk_li_inline2405__rv_v2
                sparse_blk_oi_inline1233__ssa_v0: pl.Tensor[[t_blk_inline2373__ssa_v0, 512], pl.FP32, pl.MemRef("mem_ddr_191", pl.const(0, pl.INT64), 0)] = sparse_blk_oi_inline2398__rv_v2
                rope_cos_il_inline1232__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_192", pl.const(0, pl.INT64), 65536)] = rope_cos_il_inline2316__rv_v2
                rope_sin_signed_inline1231__ssa_v0: pl.Tensor[[256, 64], pl.FP32, pl.MemRef("mem_ddr_193", pl.const(0, pl.INT64), 65536)] = rope_sin_signed_inline2315__rv_v2
                rope_swap_idx_inline1230__ssa_v0: pl.Tensor[[16, 64], pl.INT32, pl.MemRef("mem_ddr_194", pl.const(0, pl.INT64), 4096)] = rope_swap_idx_inline2314__ssa_v1
                qk_tid_inline1229__ssa_v0: pl.Scalar[pl.TASK_ID] = qk_tid_inline2349__ssa_v0
                attn_rope_tid_inline1294__ssa_v0: pl.Scalar[pl.TASK_ID] = rope_tid_inline2402__ssa_v0
                attention_grouped_inline1276__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_195", pl.const(0, pl.INT64), 16777216)] = pl.tensor.create([2048, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                pack_work_count_inline1228__ssa_v0: pl.Scalar[pl.INDEX] = t_dim_inline1251__ssa_v0 // 8 * 4
                ret__tmp_v0_45: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.csa_merge_pack_publish_spmd, attention_grouped_inline1276__ssa_v0, pack_work_count_inline1228__ssa_v0, sparse_blk_mi_inline1234__ssa_v0, sparse_blk_li_inline1283__ssa_v0, sparse_blk_oi_inline1233__ssa_v0, attn_sink__ssa_v0, rope_cos_il_inline1232__ssa_v0, rope_sin_signed_inline1231__ssa_v0, rope_swap_idx_inline1230__ssa_v0, tp_rank__ssa_v0, attention_window__ssa_v0, group_base__ssa_v0, attention_signal__ssa_v0, attention_window_ctx, attention_signal_ctx, deps=[qk_tid_inline1229__ssa_v0, attn_rope_tid_inline1294__ssa_v0], core_num=48, attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
                publish_tid_inline1260__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_45[0]
                ret__tmp_v0_46: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(self.o_group_a2a_wait, tp_rank__ssa_v0, attention_signal__ssa_v0, attention_signal_ctx, deps=[publish_tid_inline1260__ssa_v0], attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.scalar]})
                wait_tid_inline2436__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_46[0]
                ret__tmp_v0_47: pl.Tuple[pl.Tensor[[2048, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.o_group_a2a_gather_spmd, attention_local_flat_inline1292__ssa_v0, attention_window__ssa_v0, attention_window_ctx, deps=[wait_tid_inline2436__ssa_v0], core_num=48, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar]})
                attention_local_flat_inline1292__rv_v2: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_196", pl.const(0, pl.INT64), 16777216)] = ret__tmp_v0_47[0]
                gather_tid_inline2431__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_47[1]
                ret__tmp_v0_48: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(self.o_group_a2a_complete, attention_local_flat_inline1292__rv_v2, tp_rank__ssa_v0, attention_signal__ssa_v0, group_base__ssa_v0, attention_signal_ctx, deps=[gather_tid_inline2431__ssa_v0], attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
                tid__ssa_v3: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_48[0]
                attention_local_flat_inline1292__ssa_v6: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_197", pl.const(0, pl.INT64), 16777216)] = attention_local_flat_inline1292__rv_v2
                attention_signal__ssa_v1: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_198", pl.const(0, pl.INT64), 8)] = attention_signal__ssa_v0
                attention_local_groups_inline1321__ssa_v0: pl.Tensor[[4, 512, 4096], pl.BF16, pl.MemRef("mem_ddr_197", pl.const(0, pl.INT64), 16777216)] = pl.tensor.reshape(attention_local_flat_inline1292__ssa_v6, [4, 512, 4096])
                attn_2d_inline2548__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_197", pl.const(0, pl.INT64), 16777216)] = pl.tensor.reshape(attention_local_groups_inline1321__ssa_v0, [2048, 4096])
                wo_a_flat_inline2521__ssa_v0: pl.Tensor[[4096, 4096], pl.BF16, pl.MemRef("mem_ddr_50", pl.const(0, pl.INT64), 33554432)] = pl.tensor.reshape(wo_a__ssa_v0, [4096, 4096])
                publish_all_inline2525__ssa_v0: pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_199", pl.const(0, pl.INT64), 4194304)] = pl.tensor.create([512, 4096], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                for owner_inline2504__idx_v0, (publish_all_inline2525__iter_v1,) in pl.parallel(2, init_values=(publish_all_inline2525__ssa_v0,), attrs={"iter_arg_rebind_0": False}):
                    with pl.scope():
                        own_base_inline2502__ssa_v0: pl.Scalar[pl.INDEX] = owner_inline2504__idx_v0 * 256
                        own_a_fp32_inline2500__ssa_v0: pl.Tensor[[256, 4096], pl.FP32, pl.MemRef("mem_ddr_200", pl.const(0, pl.INT64), 4194304)] = pl.tensor.create([256, 4096], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        own_a_i8_inline2497__ssa_v0: pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_201", pl.const(0, pl.INT64), 1048576)] = pl.tensor.create([256, 4096], dtype=pl.INT8, layout=pl.TensorLayout.ND)
                        own_scale_inline2494__ssa_v0: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_202", pl.const(0, pl.INT64), 4096)] = pl.tensor.create([4, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)
                        own_b_i32_inline2528__ssa_v0: pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_203", pl.const(0, pl.INT64), 16777216)] = pl.tensor.create([256, 16384], dtype=pl.INT32, layout=pl.TensorLayout.ND)
                        for local_group_inline2514__idx_v0, (own_a_fp32_inline2500__iter_v1, own_a_i8_inline2497__iter_v1, own_b_i32_inline2528__iter_v1, own_scale_inline2494__iter_v1) in pl.parallel(4, init_values=(own_a_fp32_inline2500__ssa_v0, own_a_i8_inline2497__ssa_v0, own_b_i32_inline2528__ssa_v0, own_scale_inline2494__ssa_v0), attrs={"iter_arg_rebind_0": False, "iter_arg_rebind_1": False, "iter_arg_rebind_2": False, "iter_arg_rebind_3": False}):
                            with pl.scope():
                                attention_row_inline2526__ssa_v0: pl.Scalar[pl.INDEX] = local_group_inline2514__idx_v0 * 512 + own_base_inline2502__ssa_v0
                                o_a_col_inline2496__ssa_v0: pl.Scalar[pl.INDEX] = local_group_inline2514__idx_v0 * 1024
                                own_a_fp32_inline2500__ssa_v3: pl.Tensor[[256, 4096], pl.FP32, pl.MemRef("mem_ddr_204", pl.const(0, pl.INT64), 4194304)] = self.tp_o_a_spmd(attention_row_inline2526__ssa_v0, o_a_col_inline2496__ssa_v0, attn_2d_inline2548__ssa_v0, wo_a_flat_inline2521__ssa_v0, own_a_fp32_inline2500__iter_v1, attrs={"arg_directions": [pl.adir.scalar, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.output_existing], "core_num": 16})
                                ret__tmp_v0_49: pl.Tuple[pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[256, 4096], pl.INT8]] = self.tp_o_a_quant_spmd(own_a_i8_inline2497__iter_v1, own_scale_inline2494__iter_v1, own_a_fp32_inline2500__ssa_v3, o_a_col_inline2496__ssa_v0, local_group_inline2514__idx_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.scalar], "core_num": 6})
                                own_scale_inline2494__rv_v4: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_205", pl.const(0, pl.INT64), 4096)] = ret__tmp_v0_49[0]
                                own_a_i8_inline2497__rv_v7: pl.Tensor[[256, 4096], pl.INT8, pl.MemRef("mem_ddr_206", pl.const(0, pl.INT64), 1048576)] = ret__tmp_v0_49[1]
                                own_b_i32_inline2528__rv_v4: pl.Tensor[[256, 16384], pl.INT32, pl.MemRef("mem_ddr_207", pl.const(0, pl.INT64), 16777216)] = self.tp_o_b_spmd(own_b_i32_inline2528__iter_v1, own_a_i8_inline2497__rv_v7, o_a_col_inline2496__ssa_v0, wo_b__ssa_v0, local_group_inline2514__idx_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.scalar, pl.adir.input, pl.adir.scalar], "core_num": 16})
                                own_a_fp32_inline2500__rv_v2, own_a_i8_inline2497__rv_v2, own_b_i32_inline2528__rv_v2, own_scale_inline2494__rv_v2 = pl.yield_(own_a_fp32_inline2500__ssa_v3, own_a_i8_inline2497__rv_v7, own_b_i32_inline2528__rv_v4, own_scale_inline2494__rv_v4)
                        publish_all_inline2525__rv_v4: pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_212", pl.const(0, pl.INT64), 4194304)] = self.tp_o_b_dequant_spmd(publish_all_inline2525__iter_v1, own_b_i32_inline2528__rv_v2, own_scale_inline2494__rv_v2, wo_b_scale__ssa_v0, owner_inline2504__idx_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.scalar], "core_num": 12})
                        publish_all_inline2525__rv_v2: pl.Tensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_213", pl.const(0, pl.INT64), 4194304)] = pl.yield_(publish_all_inline2525__rv_v4)
                ret__tmp_v0_50: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.tp_o_b_publish_spmd, tp_rank__ssa_v0, o_window__ssa_v0, group_base__ssa_v0, publish_all_inline2525__rv_v2, o_signal__ssa_v0, o_window_ctx, o_signal_ctx, core_num=24, attrs={"arg_directions": [pl.adir.scalar, pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
                publish_tid_inline2454__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_50[0]
                ret__tmp_v0_51: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(self.tp_o_rs_wait, tp_rank__ssa_v0, o_signal__ssa_v0, o_signal_ctx, deps=[publish_tid_inline2454__ssa_v0], attrs={"arg_directions": [pl.adir.scalar, pl.adir.input, pl.adir.scalar]})
                wait_tid_inline2450__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_51[0]
                ret__tmp_v0_52: pl.Tuple[pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16], pl.Scalar[pl.TASK_ID]] = pl.spmd_submit(self.tp_o_rs_reduce_spmd, o_window__ssa_v0, attn_out_inline1284__ssa_v0, o_window_ctx, deps=[wait_tid_inline2450__ssa_v0], core_num=48, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing, pl.adir.scalar]})
                attn_out_inline1284__ssa_v1: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_214", pl.const(0, pl.INT64), 0)] = ret__tmp_v0_52[0]
                reduce_tid_inline2543__ssa_v0: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_52[1]
                ret__tmp_v0_53: pl.Tuple[pl.Scalar[pl.TASK_ID]] = pl.submit(self.tp_o_rs_complete, attn_out_inline1284__ssa_v1, tp_rank__ssa_v0, o_signal__ssa_v0, group_base__ssa_v0, o_signal_ctx, deps=[reduce_tid_inline2543__ssa_v0], attrs={"arg_directions": [pl.adir.inout, pl.adir.scalar, pl.adir.inout, pl.adir.scalar, pl.adir.scalar]})
                tid__ssa_v4: pl.Scalar[pl.TASK_ID] = ret__tmp_v0_53[0]
                _o_reduced_inline1269__ssa_v0: pl.Tensor[[t_dim_inline1251__ssa_v0, 4096], pl.BF16, pl.MemRef("mem_ddr_215", pl.const(0, pl.INT64), 0)] = attn_out_inline1284__ssa_v1
                o_signal__ssa_v1: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_216", pl.const(0, pl.INT64), 8)] = o_signal__ssa_v0
            with pl.scope():
                t_dim_inline2576__ssa_v0: pl.Scalar[pl.INDEX] = pl.tensor.dim(attn_out_inline1284__ssa_v1, 0)
                residual_flat_inline2567__ssa_v0: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(x_hc__ssa_v0, [t_dim_inline2576__ssa_v0, 16384])
                y_flat_inline2568__ssa_v0: pl.Tensor[[t_dim_inline2576__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_53", pl.const(0, pl.INT64), 0)] = pl.tensor.reshape(x_out__ssa_v0, [t_dim_inline2576__ssa_v0, 16384])
                token_tiles_inline2571__ssa_v0: pl.Scalar[pl.INDEX] = (t_dim_inline2576__ssa_v0 + 3) // 4
                self.hc_post_spmd(y_flat_inline2568__ssa_v0, t_dim_inline2576__ssa_v0, attn_out_inline1284__ssa_v1, post_t_inline1277__phi_v2, comb_t_inline1267__ssa_v1, residual_flat_inline2567__ssa_v0, attrs={"arg_directions": [pl.adir.output_existing, pl.adir.scalar, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input], "core_num": token_tiles_inline2571__ssa_v0})
            return x_out__ssa_v0
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def l3_decode_csa(self, x_hc__ssa_v0: pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)], hc_attn_fn__ssa_v0: pl.Tensor[[2, 24, 16384], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(0, pl.INT64), 3145728)], hc_attn_scale__ssa_v0: pl.Tensor[[2, 3], pl.FP32, pl.MemRef("mem_ddr_2", pl.const(0, pl.INT64), 24)], hc_attn_base__ssa_v0: pl.Tensor[[2, 24], pl.FP32, pl.MemRef("mem_ddr_3", pl.const(0, pl.INT64), 192)], attn_norm_w__ssa_v0: pl.Tensor[[2, 4096], pl.BF16, pl.MemRef("mem_ddr_4", pl.const(0, pl.INT64), 16384)], wq_a__ssa_v0: pl.Tensor[[2, 4096, 1024], pl.BF16, pl.MemRef("mem_ddr_5", pl.const(0, pl.INT64), 16777216)], wq_b__ssa_v0: pl.Tensor[[2, 1024, 32768], pl.INT8, pl.MemRef("mem_ddr_6", pl.const(0, pl.INT64), 67108864)], wq_b_scale__ssa_v0: pl.Tensor[[2, 32768], pl.FP32, pl.MemRef("mem_ddr_7", pl.const(0, pl.INT64), 262144)], wkv__ssa_v0: pl.Tensor[[2, 4096, 512], pl.BF16, pl.MemRef("mem_ddr_8", pl.const(0, pl.INT64), 8388608)], gamma_cq__ssa_v0: pl.Tensor[[2, 1024], pl.BF16, pl.MemRef("mem_ddr_9", pl.const(0, pl.INT64), 4096)], gamma_ckv__ssa_v0: pl.Tensor[[2, 512], pl.BF16, pl.MemRef("mem_ddr_10", pl.const(0, pl.INT64), 2048)], freqs_cos_local__ssa_v0: pl.Tensor[[2, T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_11", pl.const(0, pl.INT64), 0)], freqs_cos__ssa_v0: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_12", pl.const(0, pl.INT64), 0)], freqs_sin_local__ssa_v0: pl.Tensor[[2, T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_13", pl.const(0, pl.INT64), 0)], freqs_sin__ssa_v0: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_14", pl.const(0, pl.INT64), 0)], cmp_freqs_cos__ssa_v0: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_15", pl.const(0, pl.INT64), 0)], cmp_freqs_sin__ssa_v0: pl.Tensor[[2, KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_16", pl.const(0, pl.INT64), 0)], cmp_wkv__ssa_v0: pl.Tensor[[2, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_17", pl.const(0, pl.INT64), 16777216)], cmp_wgate__ssa_v0: pl.Tensor[[2, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_18", pl.const(0, pl.INT64), 16777216)], cmp_ape__ssa_v0: pl.Tensor[[2, 4, 1024], pl.FP32, pl.MemRef("mem_ddr_19", pl.const(0, pl.INT64), 32768)], cmp_norm_w__ssa_v0: pl.Tensor[[2, 512], pl.BF16, pl.MemRef("mem_ddr_20", pl.const(0, pl.INT64), 2048)], compress_state__ssa_v0: pl.InOut[pl.Tensor[[2, MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32, pl.MemRef("mem_ddr_21", pl.const(0, pl.INT64), 0)]], compress_state_block_table__ssa_v0: pl.Tensor[[2, KV_B_DYN, 4], pl.INT32, pl.MemRef("mem_ddr_22", pl.const(0, pl.INT64), 0)], idx_wq_b__ssa_v0: pl.Tensor[[2, 1024, 8192], pl.INT8, pl.MemRef("mem_ddr_23", pl.const(0, pl.INT64), 16777216)], idx_wq_b_scale__ssa_v0: pl.Tensor[[2, 8192], pl.FP32, pl.MemRef("mem_ddr_24", pl.const(0, pl.INT64), 65536)], weights_proj__ssa_v0: pl.Tensor[[2, 4096, 64], pl.BF16, pl.MemRef("mem_ddr_25", pl.const(0, pl.INT64), 1048576)], hadamard_idx__ssa_v0: pl.Tensor[[2, 128, 128], pl.BF16, pl.MemRef("mem_ddr_26", pl.const(0, pl.INT64), 65536)], inner_wkv__ssa_v0: pl.Tensor[[2, 256, 4096], pl.BF16, pl.MemRef("mem_ddr_27", pl.const(0, pl.INT64), 4194304)], inner_wgate__ssa_v0: pl.Tensor[[2, 256, 4096], pl.BF16, pl.MemRef("mem_ddr_28", pl.const(0, pl.INT64), 4194304)], inner_ape__ssa_v0: pl.Tensor[[2, 4, 256], pl.FP32, pl.MemRef("mem_ddr_29", pl.const(0, pl.INT64), 8192)], inner_norm_w__ssa_v0: pl.Tensor[[2, 128], pl.BF16, pl.MemRef("mem_ddr_30", pl.const(0, pl.INT64), 512)], inner_compress_state__ssa_v0: pl.InOut[pl.Tensor[[2, INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32, pl.MemRef("mem_ddr_31", pl.const(0, pl.INT64), 0)]], inner_compress_state_block_table__ssa_v0: pl.Tensor[[2, KV_B_DYN, 4], pl.INT32, pl.MemRef("mem_ddr_32", pl.const(0, pl.INT64), 0)], kv_cache__ssa_v0: pl.InOut[pl.Tensor[[2, ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_33", pl.const(0, pl.INT64), 0)]], cmp_kv__ssa_v0: pl.InOut[pl.Tensor[[2, CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_34", pl.const(0, pl.INT64), 0)]], cmp_block_table__ssa_v0: pl.Tensor[[2, B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_35", pl.const(0, pl.INT64), 0)], idx_kv_cache__ssa_v0: pl.InOut[pl.Tensor[[2, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8, pl.MemRef("mem_ddr_36", pl.const(0, pl.INT64), 0)]], idx_kv_scale__ssa_v0: pl.InOut[pl.Tensor[[2, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32, pl.MemRef("mem_ddr_37", pl.const(0, pl.INT64), 0)]], idx_block_table__ssa_v0: pl.Tensor[[2, B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_38", pl.const(0, pl.INT64), 0)], ori_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_39", pl.const(0, pl.INT64), 0)], window_swa_indices__ssa_v0: pl.Tensor[[2, T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_40", pl.const(0, pl.INT64), 0)], window_swa_lens__ssa_v0: pl.Tensor[[2, T_DYN], pl.INT32, pl.MemRef("mem_ddr_41", pl.const(0, pl.INT64), 0)], cmp_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_42", pl.const(0, pl.INT64), 0)], idx_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_43", pl.const(0, pl.INT64), 0)], state_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_44", pl.const(0, pl.INT64), 0)], inner_state_slot_mapping__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_45", pl.const(0, pl.INT64), 0)], position_ids_local__ssa_v0: pl.Tensor[[2, T_DYN], pl.INT32, pl.MemRef("mem_ddr_46", pl.const(0, pl.INT64), 0)], position_ids__ssa_v0: pl.Tensor[[2, KV_T_DYN], pl.INT32, pl.MemRef("mem_ddr_47", pl.const(0, pl.INT64), 0)], kv_seq_lens__ssa_v0: pl.Tensor[[2, B_DYN], pl.INT32, pl.MemRef("mem_ddr_48", pl.const(0, pl.INT64), 0)], attn_sink__ssa_v0: pl.Tensor[[2, 64], pl.FP32, pl.MemRef("mem_ddr_49", pl.const(0, pl.INT64), 512)], wo_a__ssa_v0: pl.Tensor[[2, 4, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_50", pl.const(0, pl.INT64), 67108864)], wo_b__ssa_v0: pl.Tensor[[2, 4096, 4096], pl.INT8, pl.MemRef("mem_ddr_51", pl.const(0, pl.INT64), 33554432)], wo_b_scale__ssa_v0: pl.Tensor[[2, 4096], pl.FP32, pl.MemRef("mem_ddr_52", pl.const(0, pl.INT64), 32768)], x_out__ssa_v0: pl.Out[pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32, pl.MemRef("mem_ddr_53", pl.const(0, pl.INT64), 0)]], local_t__ssa_v0: pl.Scalar[pl.INT32]) -> pl.Tensor[[2, T_DYN, 4, 4096], pl.FP32]:
        # pld.comm_domain: devices=all, slots=[gather_window_buf__ssa_v0, gather_signal_buf__ssa_v0, attention_window_buf__ssa_v0, attention_signal_buf__ssa_v0, o_window_buf__ssa_v0, o_signal_buf__ssa_v0]
        gather_window_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(4194304, pl.INT64))
        gather_signal_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        attention_window_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(16777216, pl.INT64))
        attention_signal_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        o_window_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(4194304, pl.INT64))
        o_signal_buf__ssa_v0: pl.Ptr = pld.tensor.alloc_window_buffer(pl.const(8, pl.INT64))
        t__tmp_v0: pl.Scalar[pl.INT64] = pld.system.world_size()
        for rank__idx_v0 in pl.range(t__tmp_v0):
            gather_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_54", pl.const(0, pl.INT64), 4194304)] = pld.tensor.window(gather_window_buf__ssa_v0, [512, 4096], dtype=pl.BF16)
            gather_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_55", pl.const(0, pl.INT64), 8)] = pld.tensor.window(gather_signal_buf__ssa_v0, [2, 1], dtype=pl.INT32)
            attention_window__ssa_v0: pld.DistributedTensor[[2048, 4096], pl.BF16, pl.MemRef("mem_ddr_56", pl.const(0, pl.INT64), 16777216)] = pld.tensor.window(attention_window_buf__ssa_v0, [2048, 4096], dtype=pl.BF16)
            attention_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_57", pl.const(0, pl.INT64), 8)] = pld.tensor.window(attention_signal_buf__ssa_v0, [2, 1], dtype=pl.INT32)
            o_window__ssa_v0: pld.DistributedTensor[[512, 4096], pl.BF16, pl.MemRef("mem_ddr_58", pl.const(0, pl.INT64), 4194304)] = pld.tensor.window(o_window_buf__ssa_v0, [512, 4096], dtype=pl.BF16)
            o_signal__ssa_v0: pld.DistributedTensor[[2, 1], pl.INT32, pl.MemRef("mem_ddr_59", pl.const(0, pl.INT64), 8)] = pld.tensor.window(o_signal_buf__ssa_v0, [2, 1], dtype=pl.INT32)
            t__tmp_v1: pl.Tensor[[T_DYN, 4, 4096], pl.FP32, pl.MemRef("mem_ddr_0", rank__idx_v0 * (T_DYN * 4 * 4096) * 4, 0)] = pl.tensor.slice(x_hc__ssa_v0, [1, T_DYN, 4, 4096], [rank__idx_v0, 0, 0, 0], [], [0])
            t__tmp_v2: pl.Tensor[[24, 16384], pl.FP32, pl.MemRef("mem_ddr_1", rank__idx_v0 * 393216 * 4, 1572864)] = pl.tensor.slice(hc_attn_fn__ssa_v0, [1, 24, 16384], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v3: pl.Tensor[[3], pl.FP32, pl.MemRef("mem_ddr_2", rank__idx_v0 * 3 * 4, 12)] = pl.tensor.slice(hc_attn_scale__ssa_v0, [1, 3], [rank__idx_v0, 0], [], [0])
            t__tmp_v4: pl.Tensor[[24], pl.FP32, pl.MemRef("mem_ddr_3", rank__idx_v0 * 24 * 4, 96)] = pl.tensor.slice(hc_attn_base__ssa_v0, [1, 24], [rank__idx_v0, 0], [], [0])
            t__tmp_v5: pl.Tensor[[4096], pl.BF16, pl.MemRef("mem_ddr_4", rank__idx_v0 * 4096 * 2, 8192)] = pl.tensor.slice(attn_norm_w__ssa_v0, [1, 4096], [rank__idx_v0, 0], [], [0])
            t__tmp_v6: pl.Tensor[[4096, 1024], pl.BF16, pl.MemRef("mem_ddr_5", rank__idx_v0 * 4194304 * 2, 8388608)] = pl.tensor.slice(wq_a__ssa_v0, [1, 4096, 1024], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v7: pl.Tensor[[1024, 32768], pl.INT8, pl.MemRef("mem_ddr_6", rank__idx_v0 * 33554432, 33554432)] = pl.tensor.slice(wq_b__ssa_v0, [1, 1024, 32768], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v8: pl.Tensor[[32768], pl.FP32, pl.MemRef("mem_ddr_7", rank__idx_v0 * 32768 * 4, 131072)] = pl.tensor.slice(wq_b_scale__ssa_v0, [1, 32768], [rank__idx_v0, 0], [], [0])
            t__tmp_v9: pl.Tensor[[4096, 512], pl.BF16, pl.MemRef("mem_ddr_8", rank__idx_v0 * 2097152 * 2, 4194304)] = pl.tensor.slice(wkv__ssa_v0, [1, 4096, 512], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v10: pl.Tensor[[1024], pl.BF16, pl.MemRef("mem_ddr_9", rank__idx_v0 * 1024 * 2, 2048)] = pl.tensor.slice(gamma_cq__ssa_v0, [1, 1024], [rank__idx_v0, 0], [], [0])
            t__tmp_v11: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_10", rank__idx_v0 * 512 * 2, 1024)] = pl.tensor.slice(gamma_ckv__ssa_v0, [1, 512], [rank__idx_v0, 0], [], [0])
            t__tmp_v12: pl.Tensor[[T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_11", rank__idx_v0 * (T_DYN * 64) * 2, 0)] = pl.tensor.slice(freqs_cos_local__ssa_v0, [1, T_DYN, 64], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v13: pl.Tensor[[T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_13", rank__idx_v0 * (T_DYN * 64) * 2, 0)] = pl.tensor.slice(freqs_sin_local__ssa_v0, [1, T_DYN, 64], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v14: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_12", rank__idx_v0 * (KV_T_DYN * 64) * 2, 0)] = pl.tensor.slice(freqs_cos__ssa_v0, [1, KV_T_DYN, 64], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v15: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_14", rank__idx_v0 * (KV_T_DYN * 64) * 2, 0)] = pl.tensor.slice(freqs_sin__ssa_v0, [1, KV_T_DYN, 64], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v16: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_15", rank__idx_v0 * (KV_T_DYN * 64) * 2, 0)] = pl.tensor.slice(cmp_freqs_cos__ssa_v0, [1, KV_T_DYN, 64], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v17: pl.Tensor[[KV_T_DYN, 64], pl.BF16, pl.MemRef("mem_ddr_16", rank__idx_v0 * (KV_T_DYN * 64) * 2, 0)] = pl.tensor.slice(cmp_freqs_sin__ssa_v0, [1, KV_T_DYN, 64], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v18: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_17", rank__idx_v0 * 4194304 * 2, 8388608)] = pl.tensor.slice(cmp_wkv__ssa_v0, [1, 1024, 4096], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v19: pl.Tensor[[1024, 4096], pl.BF16, pl.MemRef("mem_ddr_18", rank__idx_v0 * 4194304 * 2, 8388608)] = pl.tensor.slice(cmp_wgate__ssa_v0, [1, 1024, 4096], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v20: pl.Tensor[[4, 1024], pl.FP32, pl.MemRef("mem_ddr_19", rank__idx_v0 * 4096 * 4, 16384)] = pl.tensor.slice(cmp_ape__ssa_v0, [1, 4, 1024], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v21: pl.Tensor[[512], pl.BF16, pl.MemRef("mem_ddr_20", rank__idx_v0 * 512 * 2, 1024)] = pl.tensor.slice(cmp_norm_w__ssa_v0, [1, 512], [rank__idx_v0, 0], [], [0])
            t__tmp_v22: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], pl.FP32, pl.MemRef("mem_ddr_21", rank__idx_v0 * (MAIN_STATE_BLOCK_NUM_DYN * 2 * 2048) * 4, 0)] = pl.tensor.slice(compress_state__ssa_v0, [1, MAIN_STATE_BLOCK_NUM_DYN, 2, 2048], [rank__idx_v0, 0, 0, 0], [], [0])
            t__tmp_v23: pl.Tensor[[KV_B_DYN, 4], pl.INT32, pl.MemRef("mem_ddr_22", rank__idx_v0 * (KV_B_DYN * 4) * 4, 0)] = pl.tensor.slice(compress_state_block_table__ssa_v0, [1, KV_B_DYN, 4], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v24: pl.Tensor[[1024, 8192], pl.INT8, pl.MemRef("mem_ddr_23", rank__idx_v0 * 8388608, 8388608)] = pl.tensor.slice(idx_wq_b__ssa_v0, [1, 1024, 8192], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v25: pl.Tensor[[8192], pl.FP32, pl.MemRef("mem_ddr_24", rank__idx_v0 * 8192 * 4, 32768)] = pl.tensor.slice(idx_wq_b_scale__ssa_v0, [1, 8192], [rank__idx_v0, 0], [], [0])
            t__tmp_v26: pl.Tensor[[4096, 64], pl.BF16, pl.MemRef("mem_ddr_25", rank__idx_v0 * 262144 * 2, 524288)] = pl.tensor.slice(weights_proj__ssa_v0, [1, 4096, 64], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v27: pl.Tensor[[128, 128], pl.BF16, pl.MemRef("mem_ddr_26", rank__idx_v0 * 16384 * 2, 32768)] = pl.tensor.slice(hadamard_idx__ssa_v0, [1, 128, 128], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v28: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_27", rank__idx_v0 * 1048576 * 2, 2097152)] = pl.tensor.slice(inner_wkv__ssa_v0, [1, 256, 4096], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v29: pl.Tensor[[256, 4096], pl.BF16, pl.MemRef("mem_ddr_28", rank__idx_v0 * 1048576 * 2, 2097152)] = pl.tensor.slice(inner_wgate__ssa_v0, [1, 256, 4096], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v30: pl.Tensor[[4, 256], pl.FP32, pl.MemRef("mem_ddr_29", rank__idx_v0 * 1024 * 4, 4096)] = pl.tensor.slice(inner_ape__ssa_v0, [1, 4, 256], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v31: pl.Tensor[[128], pl.BF16, pl.MemRef("mem_ddr_30", rank__idx_v0 * 128 * 2, 256)] = pl.tensor.slice(inner_norm_w__ssa_v0, [1, 128], [rank__idx_v0, 0], [], [0])
            t__tmp_v32: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, 2, 512], pl.FP32, pl.MemRef("mem_ddr_31", rank__idx_v0 * (INNER_STATE_BLOCK_NUM_DYN * 2 * 512) * 4, 0)] = pl.tensor.slice(inner_compress_state__ssa_v0, [1, INNER_STATE_BLOCK_NUM_DYN, 2, 512], [rank__idx_v0, 0, 0, 0], [], [0])
            t__tmp_v33: pl.Tensor[[KV_B_DYN, 4], pl.INT32, pl.MemRef("mem_ddr_32", rank__idx_v0 * (KV_B_DYN * 4) * 4, 0)] = pl.tensor.slice(inner_compress_state_block_table__ssa_v0, [1, KV_B_DYN, 4], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v34: pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_33", rank__idx_v0 * (ORI_BLOCK_NUM_DYN * 32 * 512) * 2, 0)] = pl.tensor.slice(kv_cache__ssa_v0, [1, ORI_BLOCK_NUM_DYN, 32, 1, 512], [rank__idx_v0, 0, 0, 0, 0], [], [0])
            t__tmp_v35: pl.Tensor[[CMP_BLOCK_NUM_DYN, 32, 1, 512], pl.BF16, pl.MemRef("mem_ddr_34", rank__idx_v0 * (CMP_BLOCK_NUM_DYN * 32 * 512) * 2, 0)] = pl.tensor.slice(cmp_kv__ssa_v0, [1, CMP_BLOCK_NUM_DYN, 32, 1, 512], [rank__idx_v0, 0, 0, 0, 0], [], [0])
            t__tmp_v36: pl.Tensor[[B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_35", rank__idx_v0 * (B_DYN * 8192) * 4, 0)] = pl.tensor.slice(cmp_block_table__ssa_v0, [1, B_DYN, 8192], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v37: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], pl.INT8, pl.MemRef("mem_ddr_36", rank__idx_v0 * (IDX_CACHE_BLOCK_NUM_DYN * 32 * 128), 0)] = pl.tensor.slice(idx_kv_cache__ssa_v0, [1, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 128], [rank__idx_v0, 0, 0, 0, 0], [], [0])
            t__tmp_v38: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], pl.FP32, pl.MemRef("mem_ddr_37", rank__idx_v0 * (IDX_CACHE_BLOCK_NUM_DYN * 32) * 4, 0)] = pl.tensor.slice(idx_kv_scale__ssa_v0, [1, IDX_CACHE_BLOCK_NUM_DYN, 32, 1, 1], [rank__idx_v0, 0, 0, 0, 0], [], [0])
            t__tmp_v39: pl.Tensor[[B_DYN, 8192], pl.INT32, pl.MemRef("mem_ddr_38", rank__idx_v0 * (B_DYN * 8192) * 4, 0)] = pl.tensor.slice(idx_block_table__ssa_v0, [1, B_DYN, 8192], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v40: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_39", rank__idx_v0 * KV_T_DYN * 8, 0)] = pl.tensor.slice(ori_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v41: pl.Tensor[[T_DYN, 128], pl.INT32, pl.MemRef("mem_ddr_40", rank__idx_v0 * (T_DYN * 128) * 4, 0)] = pl.tensor.slice(window_swa_indices__ssa_v0, [1, T_DYN, 128], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v42: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_41", rank__idx_v0 * T_DYN * 4, 0)] = pl.tensor.slice(window_swa_lens__ssa_v0, [1, T_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v43: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_42", rank__idx_v0 * KV_T_DYN * 8, 0)] = pl.tensor.slice(cmp_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v44: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_43", rank__idx_v0 * KV_T_DYN * 8, 0)] = pl.tensor.slice(idx_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v45: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_44", rank__idx_v0 * KV_T_DYN * 8, 0)] = pl.tensor.slice(state_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v46: pl.Tensor[[KV_T_DYN], pl.INT64, pl.MemRef("mem_ddr_45", rank__idx_v0 * KV_T_DYN * 8, 0)] = pl.tensor.slice(inner_state_slot_mapping__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v47: pl.Tensor[[T_DYN], pl.INT32, pl.MemRef("mem_ddr_46", rank__idx_v0 * T_DYN * 4, 0)] = pl.tensor.slice(position_ids_local__ssa_v0, [1, T_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v48: pl.Tensor[[KV_T_DYN], pl.INT32, pl.MemRef("mem_ddr_47", rank__idx_v0 * KV_T_DYN * 4, 0)] = pl.tensor.slice(position_ids__ssa_v0, [1, KV_T_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v49: pl.Tensor[[B_DYN], pl.INT32, pl.MemRef("mem_ddr_48", rank__idx_v0 * B_DYN * 4, 0)] = pl.tensor.slice(kv_seq_lens__ssa_v0, [1, B_DYN], [rank__idx_v0, 0], [], [0])
            t__tmp_v50: pl.Tensor[[64], pl.FP32, pl.MemRef("mem_ddr_49", rank__idx_v0 * 64 * 4, 256)] = pl.tensor.slice(attn_sink__ssa_v0, [1, 64], [rank__idx_v0, 0], [], [0])
            t__tmp_v51: pl.Tensor[[4, 1024, 4096], pl.BF16, pl.MemRef("mem_ddr_50", rank__idx_v0 * 16777216 * 2, 33554432)] = pl.tensor.slice(wo_a__ssa_v0, [1, 4, 1024, 4096], [rank__idx_v0, 0, 0, 0], [], [0])
            t__tmp_v52: pl.Tensor[[4096, 4096], pl.INT8, pl.MemRef("mem_ddr_51", rank__idx_v0 * 16777216, 16777216)] = pl.tensor.slice(wo_b__ssa_v0, [1, 4096, 4096], [rank__idx_v0, 0, 0], [], [0])
            t__tmp_v53: pl.Tensor[[4096], pl.FP32, pl.MemRef("mem_ddr_52", rank__idx_v0 * 4096 * 4, 16384)] = pl.tensor.slice(wo_b_scale__ssa_v0, [1, 4096], [rank__idx_v0, 0], [], [0])
            t__tmp_v54: pl.Tensor[[T_DYN, 4, 4096], pl.FP32, pl.MemRef("mem_ddr_53", rank__idx_v0 * (T_DYN * 4 * 4096) * 4, 0)] = pl.tensor.slice(x_out__ssa_v0, [1, T_DYN, 4, 4096], [rank__idx_v0, 0, 0, 0], [], [0])
            gather_window_ctx: pld.CommCtx = pld.system.get_comm_ctx(gather_window__ssa_v0)
            gather_signal_ctx: pld.CommCtx = pld.system.get_comm_ctx(gather_signal__ssa_v0)
            attention_window_ctx: pld.CommCtx = pld.system.get_comm_ctx(attention_window__ssa_v0)
            attention_signal_ctx: pld.CommCtx = pld.system.get_comm_ctx(attention_signal__ssa_v0)
            o_window_ctx: pld.CommCtx = pld.system.get_comm_ctx(o_window__ssa_v0)
            o_signal_ctx: pld.CommCtx = pld.system.get_comm_ctx(o_signal__ssa_v0)
            self.decode_csa_test(t__tmp_v1, t__tmp_v2, t__tmp_v3, t__tmp_v4, t__tmp_v5, t__tmp_v6, t__tmp_v7, t__tmp_v8, t__tmp_v9, t__tmp_v10, t__tmp_v11, t__tmp_v12, t__tmp_v13, t__tmp_v14, t__tmp_v15, t__tmp_v16, t__tmp_v17, t__tmp_v18, t__tmp_v19, t__tmp_v20, t__tmp_v21, t__tmp_v22, t__tmp_v23, t__tmp_v24, t__tmp_v25, t__tmp_v26, t__tmp_v27, t__tmp_v28, t__tmp_v29, t__tmp_v30, t__tmp_v31, t__tmp_v32, t__tmp_v33, t__tmp_v34, t__tmp_v35, t__tmp_v36, t__tmp_v37, t__tmp_v38, t__tmp_v39, t__tmp_v40, t__tmp_v41, t__tmp_v42, t__tmp_v43, t__tmp_v44, t__tmp_v45, t__tmp_v46, t__tmp_v47, t__tmp_v48, t__tmp_v49, t__tmp_v50, t__tmp_v51, t__tmp_v52, t__tmp_v53, t__tmp_v54, gather_window__ssa_v0, gather_signal__ssa_v0, attention_window__ssa_v0, attention_signal__ssa_v0, o_window__ssa_v0, o_signal__ssa_v0, 0, rank__idx_v0, 256, gather_window_ctx, gather_signal_ctx, attention_window_ctx, attention_signal_ctx, o_window_ctx, o_signal_ctx, device=rank__idx_v0, attrs={"arg_directions": [pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.input, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.inout, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar, pl.adir.scalar]})
        return x_out__ssa_v0
