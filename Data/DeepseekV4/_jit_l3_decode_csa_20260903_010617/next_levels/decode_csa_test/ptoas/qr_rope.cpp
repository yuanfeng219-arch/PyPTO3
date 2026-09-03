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

AICORE void qr_rope(__gm__ int32_t* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ bfloat16_t* v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  SaturationMode v10 = SaturationMode::OFF;
  RoundMode v11 = RoundMode::CAST_RINT;
  const int64_t v12 = 128;
  const int64_t v13 = 1;
  const int64_t v14 = 64;
  const int64_t v15 = 32;
  const int64_t v16 = 8448;
  const int64_t v17 = 8192;
  const int64_t v18 = 0;
  const int64_t v19 = 25600;
  const int64_t v20 = 17408;
  const int64_t v21 = 17152;
  const int64_t v22 = 16896;
  const int64_t v23 = 8704;
  const int64_t v24 = 256;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %idx_inline2276__ssa_v0, %3
  int64_t v25 = (int64_t) ((uint64_t) ((int64_t) v8) * (uint64_t) v15);
  // pto: %4
  int64_t v26 = v25 / v14;
  // pto: %rope_swap_idx_inline2190__tile
  Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %rope_swap_idx_inline2190__tile
  uint64_t v28 = (uint64_t) v23;
  TASSIGN(v27, v28);
  // pto: %rope_swap_idx_t_inline2189__ssa_v1_pview
  pto::Shape<1, 1, 1, 32, 64> v29 = pto::Shape<1, 1, 1, 32, 64>();
  // pto: %rope_swap_idx_t_inline2189__ssa_v1_pview
  pto::Stride<2048, 2048, 2048, 64, 1> v30 = pto::Stride<2048, 2048, 2048, 64, 1>();
  // pto: %rope_swap_idx_t_inline2189__ssa_v1_pview
  GlobalTensor<int32_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v31 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v1, v29, v30);
  TLOAD(v27, v31);
  // pto: %cos_row_inline2192__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
  // pto: %cos_row_inline2192__tile
  uint64_t v33 = (uint64_t) v22;
  TASSIGN(v32, v33);
  // pto: %5
  int64_t v34 = v26 < v18 ? v18 : v26;
  // pto: %idx_cos_il_inline1282__rv_v2_pview
  pto::Shape<1, 1, 1, 1, 64> v35 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %idx_cos_il_inline1282__rv_v2_pview
  pto::Stride<64, 64, 64, 64, 1> v36 = pto::Stride<64, 64, 64, 64, 1>();
  // pto: %idx_cos_il_inline1282__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v2 + (v18 + v34 * v14), v35, v36);
  TLOAD(v32, v37);
  // pto: %sin_row_inline2231__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
  // pto: %sin_row_inline2231__tile
  uint64_t v39 = (uint64_t) v21;
  TASSIGN(v38, v39);
  // pto: %idx_sin_signed_inline1307__rv_v2_pview
  pto::Shape<1, 1, 1, 1, 64> v40 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %idx_sin_signed_inline1307__rv_v2_pview
  pto::Stride<64, 64, 64, 64, 1> v41 = pto::Stride<64, 64, 64, 64, 1>();
  // pto: %idx_sin_signed_inline1307__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v18 + v34 * v14), v40, v41);
  TLOAD(v38, v42);
  // pto: %qr_nope_slice_inline2206__tile
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %qr_nope_slice_inline2206__tile
  uint64_t v44 = (uint64_t) v20;
  TASSIGN(v43, v44);
  // pto: %7
  int64_t v45 = v25 < v18 ? v18 : v25;
  // pto: %qr_proj_flat_inline2295__ssa_v0_pview
  pto::Shape<1, 1, 1, 32, 64> v46 = pto::Shape<1, 1, 1, 32, 64>();
  // pto: %qr_proj_flat_inline2295__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 128, 1> v47 = pto::Stride<4096, 4096, 4096, 128, 1>();
  // pto: %qr_proj_flat_inline2295__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v4 + (v18 + v45 * v12), v46, v47);
  TLOAD(v43, v48);
  // pto: %qr_rope_slice_inline2222__tile
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %qr_rope_slice_inline2222__tile
  uint64_t v50 = (uint64_t) v19;
  TASSIGN(v49, v50);
  // pto: %9
  pto::Shape<1, 1, 1, 32, 64> v51 = pto::Shape<1, 1, 1, 32, 64>();
  // pto: %9
  pto::Stride<4096, 4096, 4096, 128, 1> v52 = pto::Stride<4096, 4096, 4096, 128, 1>();
  // pto: %9
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v4 + (v14 + v45 * v12), v51, v52);
  TLOAD(v49, v53);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %gather_acc_init
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %gather_acc_init
  uint64_t v55 = (uint64_t) v18;
  TASSIGN(v54, v55);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  for (int64_t i56 = v18; i56 < v15; i56 += v13) {
    // pto: %gather_inp_row
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
    // pto: %gather_inp_row
    uint64_t v58 = (uint64_t) v19;
    TASSIGN(v57, v58);
    // pto: %slice_view
    int64_t v59 = (int64_t) ((uint64_t) i56 * (uint64_t) v24);
    // pto: %slice_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v60;
    // pto: %slice_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v61 = v60;
    // pto: %slice_view
    uint64_t v62 = (uint64_t) ((int64_t) ((uint64_t) v59 + (uint64_t) v19));
    TASSIGN(v61, v62);
    // pto: %gather_idx_row
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
    // pto: %gather_idx_row
    uint64_t v64 = (uint64_t) v23;
    TASSIGN(v63, v64);
    // pto: %10
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65;
    // pto: %10
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v66 = v65;
    // pto: %10
    uint64_t v67 = (uint64_t) ((int64_t) ((uint64_t) v59 + (uint64_t) v23));
    TASSIGN(v66, v67);
    // pto: %gather_row_tmp
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
    // pto: %gather_row_tmp
    uint64_t v69 = (uint64_t) v17;
    TASSIGN(v68, v69);
    // pto: %gather_row
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v14);
    // pto: %gather_row
    uint64_t v71 = (uint64_t) v16;
    TASSIGN(v70, v71);
    pipe_barrier(PIPE_V);
    TGATHER(v70, v61, v66, v68);
    // pto: %assemble_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v72;
    // pto: %assemble_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = v72;
    // pto: %assemble_view
    uint64_t v74 = (uint64_t) v59;
    TASSIGN(v73, v74);
    pipe_barrier(PIPE_V);
    TMOV(v73, v70);
  }
  // pto: %qr_swapped_inline2205__tile
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %qr_swapped_inline2205__tile
  uint64_t v76 = (uint64_t) v18;
  TASSIGN(v75, v76);
  // pto: %t__tile
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %t__tile
  uint64_t v78 = (uint64_t) v23;
  TASSIGN(v77, v78);
  TCOLEXPANDMUL(v77, v49, v32);
  // pto: %0
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %0
  uint64_t v80 = (uint64_t) v19;
  TASSIGN(v79, v80);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v79, v75, v38);
  // pto: %rope_rot_inline2185__tile
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %rope_rot_inline2185__tile
  uint64_t v82 = (uint64_t) v23;
  TASSIGN(v81, v82);
  pipe_barrier(PIPE_V);
  TADD(v81, v77, v79);
  // pto: %1
  Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v83 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %1
  uint64_t v84 = (uint64_t) v20;
  TASSIGN(v83, v84);
  TCVT(v83, v43, v11, v10);
  // pto: %2
  Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = Tile<TileType::Vec, bfloat16_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %2
  uint64_t v86 = (uint64_t) v19;
  TASSIGN(v85, v86);
  pipe_barrier(PIPE_V);
  TCVT(v85, v81, v11, v10);
  // pto: %qr_vec_inline2181__tile
  Tile<TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Vec, bfloat16_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
  // pto: %qr_vec_inline2181__tile
  uint64_t v88 = (uint64_t) v23;
  TASSIGN(v87, v88);
  pipe_barrier(PIPE_V);
  TCONCAT(v87, v83, v85);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %qr_bf16_inline2223__ssa_v0_pview
  pto::Shape<1, 1, 1, 32, 128> v89 = pto::Shape<1, 1, 1, 32, 128>();
  // pto: %qr_bf16_inline2223__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 128, 1> v90 = pto::Stride<4096, 4096, 4096, 128, 1>();
  // pto: %qr_bf16_inline2223__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v91 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v5 + (v18 + v45 * v12), v89, v90);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v91, v87);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}