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

AICORE void idx_qr_proj_dequant(__gm__ float* v1, __gm__ float* v2, __gm__ int32_t* v3, __gm__ float* v4, int64_t v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  SaturationMode v10 = SaturationMode::OFF;
  RoundMode v11 = RoundMode::CAST_NONE;
  const int64_t v12 = 8192;
  const int64_t v13 = 8;
  const int64_t v14 = 1024;
  const int64_t v15 = 1;
  const int64_t v16 = 36864;
  const int64_t v17 = 4096;
  const int64_t v18 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %qr_scale_inline1310__ssa_v0_view
  int64_t v19 = v7 * v15;
  // pto: %qr_scale_inline1310__ssa_v0_view
  int64_t v20 = v15 * v19;
  // pto: %qr_scale_inline1310__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v21 = pto::Shape<1, 1, 1, -1, -1>(v15, v15, v15, v7, v15);
  // pto: %qr_scale_inline1310__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v22 = pto::Stride<-1, -1, -1, -1, -1>(v15 * v20, v20, v19, v15, v7);
  // pto: %qr_scale_inline1310__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v4, v21, v22);
  // pto: %ot_inline2284__ssa_v0, %2
  int64_t v24 = (int64_t) ((uint64_t) ((int64_t) v8) * (uint64_t) v14);
  // pto: %t__tile
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %t__tile
  uint64_t v26 = (uint64_t) v18;
  TASSIGN(v25, v26);
  // pto: %3
  int64_t v27 = v24 < v18 ? v18 : v24;
  // pto: %idx_wq_b_scale__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 1024> v28 = pto::Shape<1, 1, 1, 1, 1024>();
  // pto: %idx_wq_b_scale__ssa_v0_pview
  pto::Stride<1024, 1024, 1024, 1024, 1> v29 = pto::Stride<1024, 1024, 1024, 1024, 1>();
  // pto: %idx_wq_b_scale__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v1 + (v18 + v27), v28, v29);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  TLOAD(v25, v30);
  // pto: %wq_scale_inline2262__tile
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v31 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %wq_scale_inline2262__tile
  uint64_t v32 = (uint64_t) v18;
  TASSIGN(v31, v32);
  for (int64_t i33 = v18; i33 < v6; i33 += v13) {
    // pto: %0
    Tile<TileType::Vec, int32_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, int32_t, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
    // pto: %0
    uint64_t v35 = (uint64_t) v17;
    TASSIGN(v34, v35);
    // pto: %4
    int64_t v36 = i33 < v18 ? v18 : i33;
    // pto: %qr_acc_pad_inline2225__rv_v2_pview
    pto::Shape<1, 1, 1, 8, 1024> v37 = pto::Shape<1, 1, 1, 8, 1024>();
    // pto: %qr_acc_pad_inline2225__rv_v2_pview
    pto::Stride<65536, 65536, 65536, 8192, 1> v38 = pto::Stride<65536, 65536, 65536, 8192, 1>();
    // pto: %qr_acc_pad_inline2225__rv_v2_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v39 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(v3 + ((v18 + v36 * v12) + v27), v37, v38);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v34, v39);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %acc_fp32_inline2257__tile
    Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
    // pto: %acc_fp32_inline2257__tile
    uint64_t v41 = (uint64_t) v17;
    TASSIGN(v40, v41);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v40, v34, v11, v10);
    // pto: %qr_scale_tile_inline2236__tile
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v15);
    // pto: %qr_scale_tile_inline2236__tile
    uint64_t v43 = (uint64_t) v16;
    TASSIGN(v42, v43);
    // pto: %qr_scale_inline1310__ssa_v0_pview
    __gm__ float* v44 = PTOAS__GLOBAL_TENSOR_DATA(v23);
    // pto: %qr_scale_inline1310__ssa_v0_pview
    int64_t v45 = v13 * v15;
    // pto: %qr_scale_inline1310__ssa_v0_pview
    int64_t v46 = v15 * v45;
    // pto: %qr_scale_inline1310__ssa_v0_pview
    pto::Shape<1, 1, 1, 8, 1> v47 = pto::Shape<1, 1, 1, 8, 1>(v15, v15, v15, v13, v15);
    // pto: %qr_scale_inline1310__ssa_v0_pview
    pto::Stride<-1, -1, -1, -1, -1> v48 = pto::Stride<-1, -1, -1, -1, -1>(v15 * v46, v46, v45, v15, v7);
    // pto: %qr_scale_inline1310__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v49 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v44 + ((v18 + v36 * v15) + v18 * v7), v47, v48);
    TLOAD(v42, v49);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %1
    Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v50 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
    // pto: %1
    uint64_t v51 = (uint64_t) v17;
    TASSIGN(v50, v51);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_V);
    TROWEXPANDMUL(v50, v40, v42);
    // pto: %qr_dequant_inline2240__tile
    Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v52 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
    // pto: %qr_dequant_inline2240__tile
    uint64_t v53 = (uint64_t) v17;
    TASSIGN(v52, v53);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v52, v50, v31);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %qr_proj_inline2268__iter_v1_pview
    pto::Shape<1, 1, 1, 8, 1024> v54 = pto::Shape<1, 1, 1, 8, 1024>();
    // pto: %qr_proj_inline2268__iter_v1_pview
    pto::Stride<65536, 65536, 65536, 8192, 1> v55 = pto::Stride<65536, 65536, 65536, 8192, 1>();
    // pto: %qr_proj_inline2268__iter_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND> v56 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1024>, pto::Stride<65536, 65536, 65536, 8192, 1>, pto::Layout::ND>(v2 + ((v18 + v36 * v12) + v27), v54, v55);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v56, v52);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}