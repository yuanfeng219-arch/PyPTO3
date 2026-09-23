#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

AICORE void kv_and_cache_write(__gm__ float* v1, __gm__ int8_t* v2, __gm__ float* v3, __gm__ int64_t* v4, __gm__ float* v5, int64_t v6, int64_t v7, int64_t v8, int32_t v9, int32_t v10) {
  RoundMode v11 = RoundMode::CAST_TRUNC;
  RoundMode v12 = RoundMode::CAST_ROUND;
  SaturationMode v13 = SaturationMode::OFF;
  RoundMode v14 = RoundMode::CAST_RINT;
  const float v15 = 127.0f;
  const float v16 = 9.99999974E-5f;
  const int64_t v17 = 16;
  const int64_t v18 = 1;
  const int64_t v19 = 0;
  const int64_t v20 = 16448;
  const int64_t v21 = 8256;
  const int64_t v22 = 64;
  const int64_t v23 = 128;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %wr_blk_inline2171__ssa_v0, %8
  int64_t v24 = (int64_t) ((uint64_t) ((int64_t) v9) * (uint64_t) v17);
  // pto: %9
  int64_t v25 = (int64_t) ((uint64_t) v6 - (uint64_t) v24);
  // pto: %10
  // pto: %t__tile
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %t__tile
  uint64_t v27 = (uint64_t) v22;
  TASSIGN(v26, v27);
  // pto: %kv_final_inline2118__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 128> v28 = pto::Shape<1, 1, 1, 16, 128>();
  // pto: %kv_final_inline2118__rv_v2_pview
  pto::Stride<2048, 2048, 2048, 128, 1> v29 = pto::Stride<2048, 2048, 2048, 128, 1>();
  // pto: %11, %kv_final_inline2118__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v19 + (v24 < v19 ? v19 : v24) * v23), v28, v29);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  TLOAD(v26, v30);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %0
  Tile<TileType::Vec, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v31 = Tile<TileType::Vec, bfloat16_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %0
  uint64_t v32 = (uint64_t) v21;
  TASSIGN(v31, v32);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(v31, v26, v14, v13);
  // pto: %kv_blk_f32_inline2067__tile
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %kv_blk_f32_inline2067__tile
  uint64_t v34 = (uint64_t) v22;
  TASSIGN(v33, v34);
  pipe_barrier(PIPE_V);
  TCVT(v33, v31, v12, v13);
  // pto: %1
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %1
  uint64_t v36 = (uint64_t) v21;
  TASSIGN(v35, v36);
  pipe_barrier(PIPE_V);
  TABS(v35, v33);
  // pto: %tmp_tile
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %tmp_tile
  uint64_t v38 = (uint64_t) v20;
  TASSIGN(v37, v38);
  // pto: %2
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v39 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %2
  uint64_t v40 = (uint64_t) v19;
  TASSIGN(v39, v40);
  pipe_barrier(PIPE_V);
  TROWMAX(v39, v35, v37);
  // pto: %kv_amax_inline2091__tile
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  // pto: %kv_amax_inline2091__tile
  uint64_t v42 = (uint64_t) v19;
  TASSIGN(v41, v42);
  // pto: %3
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  // pto: %3
  uint64_t v44 = (uint64_t) v21;
  TASSIGN(v43, v44);
  pipe_barrier(PIPE_V);
  TEXPANDS(v43, v16);
  // pto: %kv_amax_v1_inline2066__tile
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  // pto: %kv_amax_v1_inline2066__tile
  uint64_t v46 = (uint64_t) v21;
  TASSIGN(v45, v46);
  pipe_barrier(PIPE_V);
  TMAX(v45, v41, v43);
  // pto: %4
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  // pto: %4
  uint64_t v48 = (uint64_t) v20;
  TASSIGN(v47, v48);
  TEXPANDS(v47, v15);
  // pto: %kv_scale_q_row_inline2112__tile
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  // pto: %kv_scale_q_row_inline2112__tile
  uint64_t v50 = (uint64_t) v21;
  TASSIGN(v49, v50);
  pipe_barrier(PIPE_V);
  TDIV(v49, v47, v45);
  // pto: %5
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v17);
  // pto: %5
  uint64_t v52 = (uint64_t) v20;
  TASSIGN(v51, v52);
  pipe_barrier(PIPE_V);
  TRECIP(v51, v49);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  // pto: %kv_scale_dq_col_inline2065__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %kv_scale_dq_col_inline2065__tile
  uint64_t v54 = (uint64_t) v20;
  TASSIGN(v53, v54);
  // pto: %kv_scale_q_col_inline2086__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v18);
  // pto: %kv_scale_q_col_inline2086__tile
  uint64_t v56 = (uint64_t) v21;
  TASSIGN(v55, v56);
  // pto: %kv_scaled_inline2111__tile
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %kv_scaled_inline2111__tile
  uint64_t v58 = (uint64_t) v22;
  TASSIGN(v57, v58);
  TROWEXPANDMUL(v57, v33, v55);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %kv_i32_inline2133__tile
  Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Vec, int32_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %kv_i32_inline2133__tile
  uint64_t v60 = (uint64_t) v22;
  TASSIGN(v59, v60);
  pipe_barrier(PIPE_V);
  TCVT(v59, v57, v14, v13);
  // pto: %kv_half_inline2064__tile
  Tile<TileType::Vec, half, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Vec, half, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %kv_half_inline2064__tile
  uint64_t v62 = (uint64_t) v22;
  TASSIGN(v61, v62);
  pipe_barrier(PIPE_V);
  TCVT(v61, v59, v12, v13);
  // pto: %kv_i8_blk_inline2095__tile
  Tile<TileType::Vec, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, int8_t, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v17, v23);
  // pto: %kv_i8_blk_inline2095__tile
  uint64_t v64 = (uint64_t) v22;
  TASSIGN(v63, v64);
  pipe_barrier(PIPE_V);
  TCVT(v63, v61, v11, v13);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (int64_t i65 = v19; i65 < (v25 < v17 ? v25 : v17); i65 += v18) {
    // pto: %12
    int64_t v66 = (int64_t) ((uint64_t) v24 + (uint64_t) i65);
    // pto: %cache_row_i64_inline2099__tile
    int64_t v67 = (v4)[v66];
    // pto: %14
    if (v67 >= v19) {
      // pto: %6
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v23);
      // pto: %6
      uint64_t v69 = (uint64_t) v21;
      TASSIGN(v68, v69);
      // pto: %16
      int64_t v70 = v66 < v19 ? v19 : v66;
      // pto: %17
      pto::Shape<1, 1, 1, 1, 128> v71 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %17
      pto::Stride<128, 128, 128, 128, 1> v72 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %17
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v73 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v19 + v70 * v23), v71, v72);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v68, v73);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %kv_flat_inline2098__iter_v1_pview
      pto::Shape<1, 1, 1, 1, 128> v74 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %kv_flat_inline2098__iter_v1_pview
      pto::Stride<128, 128, 128, 128, 1> v75 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %kv_flat_inline2098__iter_v1_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + (v19 + v70 * v23), v74, v75);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v76, v68);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      // pto: %7
      Tile<TileType::Vec, int8_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, int8_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v18, v23);
      // pto: %7
      uint64_t v78 = (uint64_t) v22;
      TASSIGN(v77, v78);
      // pto: %slice_view
      Tile<TileType::Vec, int8_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79;
      // pto: %slice_view
      Tile<TileType::Vec, int8_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = v79;
      // pto: %slice_view
      uint64_t v81 = (uint64_t) ((int64_t) ((uint64_t) ((int64_t) ((uint64_t) i65 * (uint64_t) v23)) + (uint64_t) v22));
      TASSIGN(v80, v81);
      // pto: %idx_kv_cache_flat_inline2161__iter_v1_pview
      pto::Shape<1, 1, 1, 1, 128> v82 = pto::Shape<1, 1, 1, 1, 128>();
      // pto: %idx_kv_cache_flat_inline2161__iter_v1_pview
      pto::Stride<128, 128, 128, 128, 1> v83 = pto::Stride<128, 128, 128, 128, 1>();
      // pto: %19, %idx_kv_cache_flat_inline2161__iter_v1_pview
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v84 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v19 + (v67 < v19 ? v19 : v67) * v23), v82, v83);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v84, v80);
      // pto: %20
      float v85 = v53.GetValue(i65);
      (v5)[v67] = v85;
    }
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  #endif // __DAV_VEC__

  pipe_barrier(PIPE_ALL);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  dsb((mem_dsb_t)0);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}