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

AICORE void qr_hadamard_matmul(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ float* v3, int64_t v4, int32_t v5, int32_t v6) {
  const int64_t v7 = 64;
  const int64_t v8 = 128;
  const int64_t v9 = 16384;
  const int64_t v10 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %idx_inline2227__ssa_v0, %0
  int64_t v11 = (int64_t) ((uint64_t) ((int64_t) v5) * (uint64_t) v7);
  // pto: %t__tile
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v12 = Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v7, v8);
  // pto: %t__tile
  uint64_t v13 = (uint64_t) v10;
  TASSIGN(v12, v13);
  // pto: %1
  int64_t v14 = v11 < v10 ? v10 : v11;
  // pto: %qr_bf16_inline2223__ssa_v1_pview
  pto::Shape<1, 1, 1, 64, 128> v15 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %qr_bf16_inline2223__ssa_v1_pview
  pto::Stride<8192, 8192, 8192, 128, 1> v16 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %qr_bf16_inline2223__ssa_v1_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v17 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v1 + (v10 + v14 * v8), v15, v16);
  TLOAD(v12, v17);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  // pto: %hadamard_idx__ssa_v0_mat
  Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v18 = Tile<TileType::Mat, bfloat16_t, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
  // pto: %hadamard_idx__ssa_v0_mat
  uint64_t v19 = (uint64_t) v9;
  TASSIGN(v18, v19);
  // pto: %hadamard_idx__ssa_v0_pview
  pto::Shape<1, 1, 1, 128, 128> v20 = pto::Shape<1, 1, 1, 128, 128>();
  // pto: %hadamard_idx__ssa_v0_pview
  pto::Stride<16384, 16384, 16384, 128, 1> v21 = pto::Stride<16384, 16384, 16384, 128, 1>();
  // pto: %hadamard_idx__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v22 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2, v20, v21);
  TLOAD(v18, v22);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  // pto: %t__tile_Left
  Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v23 = Tile<TileType::Left, bfloat16_t, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v7, v8);
  // pto: %t__tile_Left
  uint64_t v24 = (uint64_t) v10;
  TASSIGN(v23, v24);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v23, v12);
  // pto: %hadamard_idx__ssa_v0_mat_Right
  Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Right, bfloat16_t, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
  // pto: %hadamard_idx__ssa_v0_mat_Right
  uint64_t v26 = (uint64_t) v10;
  TASSIGN(v25, v26);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v25, v18);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  // pto: %qh_acc_inline2178__tile
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v27 = Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v7, v8);
  // pto: %qh_acc_inline2178__tile
  uint64_t v28 = (uint64_t) v10;
  TASSIGN(v27, v28);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v27, v23, v25);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %qh_acc_gm_inline2179__ssa_v0_pview
  pto::Shape<1, 1, 1, 64, 128> v29 = pto::Shape<1, 1, 1, 64, 128>();
  // pto: %qh_acc_gm_inline2179__ssa_v0_pview
  pto::Stride<8192, 8192, 8192, 128, 1> v30 = pto::Stride<8192, 8192, 8192, 128, 1>();
  // pto: %qh_acc_gm_inline2179__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v31 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v10 + v14 * v8), v29, v30);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v31, v27);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}