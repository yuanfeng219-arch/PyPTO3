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

AICORE void compress_state_commit_0(__gm__ float* v1, __gm__ int64_t* v2, __gm__ int32_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, int64_t v7, int64_t v8, int64_t v9, int32_t v10, int32_t v11) {
  const int64_t v12 = 512;
  const int64_t v13 = 4;
  const int64_t v14 = 256;
  const int64_t v15 = 1;
  const int64_t v16 = 1024;
  const int64_t v17 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int64_t i18 = v17; i18 < v7; i18 += v15) {
    // pto: %c_idx_v1_inline2153__ssa_v0, %3, %4
    int64_t v19 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) ((int64_t) v10) * (uint64_t) v7)) + (uint64_t) i18);
    // pto: %state_row_i64_inline2137__tile
    int64_t v20 = (v2)[v19];
    // pto: %6
    if (v20 >= v17) {
      // pto: %token_pos_inline2120__tile
      int32_t v21 = (v3)[v19];
      // pto: %8, %9
      int64_t v22 = (int64_t) v21 % v13;
      // pto: %t__tile
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v23 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
      // pto: %t__tile
      uint64_t v24 = (uint64_t) v17;
      TASSIGN(v23, v24);
      // pto: %11
      int64_t v25 = v19 < v17 ? v17 : v19;
      // pto: %kv_proj_pad_inline2129__ssa_v1_pview
      pto::Shape<1, 1, 1, 1, 256> v26 = pto::Shape<1, 1, 1, 1, 256>();
      // pto: %kv_proj_pad_inline2129__ssa_v1_pview
      pto::Stride<256, 256, 256, 256, 1> v27 = pto::Stride<256, 256, 256, 256, 1>();
      // pto: %kv_proj_pad_inline2129__ssa_v1_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v28 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v4 + (v17 + v25 * v14), v26, v27);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v23, v28);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %12
      int64_t v29 = v20 < v17 ? v17 : v20;
      // pto: %compress_state_flat_inline2139__iter_v1_pview
      pto::Shape<1, 1, 1, 1, 256> v30 = pto::Shape<1, 1, 1, 1, 256>();
      // pto: %compress_state_flat_inline2139__iter_v1_pview
      pto::Stride<512, 512, 512, 512, 1> v31 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %compress_state_flat_inline2139__iter_v1_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v1 + (v17 + v29 * v12), v30, v31);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v32, v23);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      // pto: %0
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
      // pto: %0
      uint64_t v34 = (uint64_t) v17;
      TASSIGN(v33, v34);
      // pto: %score_proj_pad_inline2143__ssa_v1_pview
      pto::Shape<1, 1, 1, 1, 256> v35 = pto::Shape<1, 1, 1, 1, 256>();
      // pto: %score_proj_pad_inline2143__ssa_v1_pview
      pto::Stride<256, 256, 256, 256, 1> v36 = pto::Stride<256, 256, 256, 256, 1>();
      // pto: %score_proj_pad_inline2143__ssa_v1_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v5 + (v17 + v25 * v14), v35, v36);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v33, v37);
      // pto: %1
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
      // pto: %1
      uint64_t v39 = (uint64_t) v16;
      TASSIGN(v38, v39);
      // pto: %inner_ape__ssa_v0_pview
      pto::Shape<1, 1, 1, 1, 256> v40 = pto::Shape<1, 1, 1, 1, 256>();
      // pto: %inner_ape__ssa_v0_pview
      pto::Stride<256, 256, 256, 256, 1> v41 = pto::Stride<256, 256, 256, 256, 1>();
      // pto: %14, %inner_ape__ssa_v0_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v6 + (v17 + (v22 < v17 ? v17 : v22) * v14), v40, v41);
      TLOAD(v38, v42);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %2
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
      // pto: %2
      uint64_t v44 = (uint64_t) v17;
      TASSIGN(v43, v44);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TADD(v43, v33, v38);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %compress_state_flat_inline2139__tile_pview
      pto::Shape<1, 1, 1, 1, 256> v45 = pto::Shape<1, 1, 1, 1, 256>();
      // pto: %compress_state_flat_inline2139__tile_pview
      pto::Stride<512, 512, 512, 512, 1> v46 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %compress_state_flat_inline2139__tile_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v1 + (v14 + v29 * v12), v45, v46);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(v47, v43);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}