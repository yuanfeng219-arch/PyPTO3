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

AICORE void rmsnorm_rope_cache_write(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ bfloat16_t* v5, __gm__ bfloat16_t* v6, __gm__ float* v7, __gm__ int64_t* v8, int64_t v9, int64_t v10, int64_t v11, int32_t v12, int32_t v13) {
  RoundMode v14 = RoundMode::CAST_RINT;
  RoundMode v15 = RoundMode::CAST_TRUNC;
  unsigned v16 = 448;
  SaturationMode v17 = SaturationMode::OFF;
  RoundMode v18 = RoundMode::CAST_ROUND;
  const float v19 = 2.0f;
  const float v20 = 0.5f;
  const int32_t v21 = 0;
  const float v22 = 1.0f;
  const int64_t v23 = 448;
  const float v24 = 9.99999997E-7f;
  const float v25 = 0.001953125f;
  const int64_t v26 = 128;
  const float v27 = 0.0f;
  const int64_t v28 = 16;
  const int64_t v29 = 512;
  const int64_t v30 = 1;
  const int64_t v31 = 64;
  const int64_t v32 = 16640;
  const int64_t v33 = 16384;
  const int64_t v34 = 0;
  const int64_t v35 = 4096;
  const int64_t v36 = 12288;
  const int64_t v37 = 20992;
  const int64_t v38 = 16896;
  const int64_t v39 = 256;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  int64_t v40 = v10 * v31;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  int64_t v41 = v30 * v40;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v42 = pto::Shape<1, 1, 1, -1, -1>(v30, v30, v30, v10, v31);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v43 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v41, v41, v40, v31, v30);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v42, v43);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  int64_t v45 = v10 * v31;
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  int64_t v46 = v30 * v45;
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v47 = pto::Shape<1, 1, 1, -1, -1>(v30, v30, v30, v10, v31);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v48 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v46, v46, v45, v31, v30);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v49 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v47, v48);
  // pto: %rms_blk_inline1993__ssa_v0, %19
  int64_t v50 = (int64_t) ((uint64_t) ((int64_t) v12) * (uint64_t) v28);
  // pto: %20
  int64_t v51 = (int64_t) ((uint64_t) v9 - (uint64_t) v50);
  // pto: %21
  int64_t v52 = v51 < v28 ? v51 : v28;
  // pto: %cos_b_inline2057__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v31);
  // pto: %cos_b_inline2057__tile
  uint64_t v54 = (uint64_t) v38;
  TASSIGN(v53, v54);
  // pto: %22
  int64_t v55 = v50 < v34 ? v34 : v50;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  __gm__ float* v56 = PTOAS__GLOBAL_TENSOR_DATA(v44);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  int64_t v57 = v52 * v31;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  int64_t v58 = v30 * v57;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  pto::Shape<1, 1, 1, -1, 64> v59 = pto::Shape<1, 1, 1, -1, 64>(v30, v30, v30, v52, v31);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  pto::Stride<-1, -1, -1, -1, -1> v60 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v58, v58, v57, v31, v30);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v56 + ((v34 + v55 * v31) + v34 * v30), v59, v60);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  TLOAD(v53, v61);
  // pto: %sin_b_inline1989__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v52, v31);
  // pto: %sin_b_inline1989__tile
  uint64_t v63 = (uint64_t) v37;
  TASSIGN(v62, v63);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  __gm__ float* v64 = PTOAS__GLOBAL_TENSOR_DATA(v49);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  int64_t v65 = v52 * v31;
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  int64_t v66 = v30 * v65;
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  pto::Shape<1, 1, 1, -1, 64> v67 = pto::Shape<1, 1, 1, -1, 64>(v30, v30, v30, v52, v31);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  pto::Stride<-1, -1, -1, -1, -1> v68 = pto::Stride<-1, -1, -1, -1, -1>(v30 * v66, v66, v65, v31, v30);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v69 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v64 + ((v34 + v55 * v31) + v34 * v30), v67, v68);
  TLOAD(v62, v69);
  // pto: %partial_sq_inline2025__tile
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
  // pto: %partial_sq_inline2025__tile
  uint64_t v71 = (uint64_t) v36;
  TASSIGN(v70, v71);
  TEXPANDS(v70, v27);
  for (int64_t i72 = v34; i72 < v29; i72 += v31) {
    // pto: %kv_rms_chunk_inline1986__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
    // pto: %kv_rms_chunk_inline1986__tile
    uint64_t v74 = (uint64_t) v35;
    TASSIGN(v73, v74);
    // pto: %pooled_kv_inline2008__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 64> v75 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %pooled_kv_inline2008__rv_v2_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v76 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %pooled_kv_inline2008__rv_v2_pview, %25
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v77 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v3 + ((v34 + v55 * v29) + (i72 < v34 ? v34 : i72)), v75, v76);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v73, v77);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %kv_rms_sq_inline1988__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
    // pto: %kv_rms_sq_inline1988__tile
    uint64_t v79 = (uint64_t) v34;
    TASSIGN(v78, v79);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(v78, v73, v73);
    // pto: %tmp_tile
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v26);
    // pto: %tmp_tile
    uint64_t v81 = (uint64_t) v35;
    TASSIGN(v80, v81);
    // pto: %t__tile
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v30);
    // pto: %t__tile
    uint64_t v83 = (uint64_t) v33;
    TASSIGN(v82, v83);
    pipe_barrier(PIPE_V);
    TROWSUM(v82, v78, v80);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    // pto: %kv_rms_rowsum_inline2053__tile
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v84 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
    // pto: %kv_rms_rowsum_inline2053__tile
    uint64_t v85 = (uint64_t) v33;
    TASSIGN(v84, v85);
    // pto: %0
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
    // pto: %0
    uint64_t v87 = (uint64_t) v36;
    TASSIGN(v86, v87);
    pipe_barrier(PIPE_V);
    TADD(v86, v70, v84);
  }
  // pto: %1
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
  // pto: %1
  uint64_t v89 = (uint64_t) v35;
  TASSIGN(v88, v89);
  pipe_barrier(PIPE_V);
  TMULS(v88, v70, v25);
  // pto: %2
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
  // pto: %2
  uint64_t v91 = (uint64_t) v35;
  TASSIGN(v90, v91);
  pipe_barrier(PIPE_V);
  TADDS(v90, v88, v24);
  // pto: %variance_inline1990__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v30);
  // pto: %variance_inline1990__tile
  uint64_t v93 = (uint64_t) v35;
  TASSIGN(v92, v93);
  // pto: %t__rm_a0_tmp_v0
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
  // pto: %t__rm_a0_tmp_v0
  uint64_t v95 = (uint64_t) v35;
  TASSIGN(v94, v95);
  // pto: %t__row_major_tmp_v1
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
  // pto: %t__row_major_tmp_v1
  uint64_t v97 = (uint64_t) v35;
  TASSIGN(v96, v97);
  pipe_barrier(PIPE_V);
  TSQRT(v96, v94);
  // pto: %3
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v30);
  // pto: %3
  uint64_t v99 = (uint64_t) v35;
  TASSIGN(v98, v99);
  // pto: %inv_rms_inline1984__rm_a0_tmp_v2
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
  // pto: %inv_rms_inline1984__rm_a0_tmp_v2
  uint64_t v101 = (uint64_t) v35;
  TASSIGN(v100, v101);
  // pto: %inv_rms_inline1984__row_major_tmp_v3
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v28);
  // pto: %inv_rms_inline1984__row_major_tmp_v3
  uint64_t v103 = (uint64_t) v33;
  TASSIGN(v102, v103);
  pipe_barrier(PIPE_V);
  TRECIP(v102, v100);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  // pto: %inv_rms_inline1984__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v30);
  // pto: %inv_rms_inline1984__tile
  uint64_t v105 = (uint64_t) v33;
  TASSIGN(v104, v105);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  for (int64_t i106 = v34; i106 < v23; i106 += v31) {
    // pto: %kv_norm_chunk_inline2037__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
    // pto: %kv_norm_chunk_inline2037__tile
    uint64_t v108 = (uint64_t) v35;
    TASSIGN(v107, v108);
    // pto: %27
    int64_t v109 = i106 < v34 ? v34 : i106;
    // pto: %28
    pto::Shape<1, 1, 1, 16, 64> v110 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %28
    pto::Stride<8192, 8192, 8192, 512, 1> v111 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %28
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v112 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v3 + ((v34 + v55 * v29) + v109), v110, v111);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v107, v112);
    // pto: %4
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
    // pto: %4
    uint64_t v114 = (uint64_t) v36;
    TASSIGN(v113, v114);
    // pto: %norm_w_2d_inline2060__ssa_v0_pview
    pto::Shape<1, 1, 1, 1, 64> v115 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %norm_w_2d_inline2060__ssa_v0_pview
    pto::Stride<512, 512, 512, 512, 1> v116 = pto::Stride<512, 512, 512, 512, 1>();
    // pto: %norm_w_2d_inline2060__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v117 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v5 + (v34 + v109), v115, v116);
    TLOAD(v113, v117);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %gamma_inline2024__tile
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
    // pto: %gamma_inline2024__tile
    uint64_t v119 = (uint64_t) v34;
    TASSIGN(v118, v119);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCVT(v118, v113, v18, v17);
    // pto: %5
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
    // pto: %5
    uint64_t v121 = (uint64_t) v35;
    TASSIGN(v120, v121);
    TROWEXPANDMUL(v120, v107, v104);
    // pto: %normed_chunk_inline1982__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v122 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
    // pto: %normed_chunk_inline1982__tile
    uint64_t v123 = (uint64_t) v35;
    TASSIGN(v122, v123);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v122, v120, v118);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // pto: %normed_kv_inline2016__iter_v1_pview
    pto::Shape<1, 1, 1, 16, 64> v124 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %normed_kv_inline2016__iter_v1_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v125 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %normed_kv_inline2016__iter_v1_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v126 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v4 + ((v34 + v55 * v29) + v109), v124, v125);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v126, v122);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  // pto: %kv_rope_norm_inline1981__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %kv_rope_norm_inline1981__tile
  uint64_t v128 = (uint64_t) v35;
  TASSIGN(v127, v128);
  // pto: %33
  pto::Shape<1, 1, 1, 16, 64> v129 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %33
  pto::Stride<8192, 8192, 8192, 512, 1> v130 = pto::Stride<8192, 8192, 8192, 512, 1>();
  // pto: %33
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v131 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v3 + (v23 + v55 * v29), v129, v130);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  TLOAD(v127, v131);
  // pto: %6
  Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
  // pto: %6
  uint64_t v133 = (uint64_t) v36;
  TASSIGN(v132, v133);
  // pto: %34
  pto::Shape<1, 1, 1, 1, 64> v134 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %34
  pto::Stride<512, 512, 512, 512, 1> v135 = pto::Stride<512, 512, 512, 512, 1>();
  // pto: %34
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v136 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v5 + v16, v134, v135);
  TLOAD(v132, v136);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  // pto: %gamma_rope_inline1980__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
  // pto: %gamma_rope_inline1980__tile
  uint64_t v138 = (uint64_t) v34;
  TASSIGN(v137, v138);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  TCVT(v137, v132, v18, v17);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  // pto: %7
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %7
  uint64_t v140 = (uint64_t) v35;
  TASSIGN(v139, v140);
  TROWEXPANDMUL(v139, v127, v104);
  // pto: %rope_normed_inline1979__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %rope_normed_inline1979__tile
  uint64_t v142 = (uint64_t) v35;
  TASSIGN(v141, v142);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v141, v139, v137);
  // pto: %rope_ones_inline1978__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %rope_ones_inline1978__tile
  uint64_t v144 = (uint64_t) v34;
  TASSIGN(v143, v144);
  pipe_barrier(PIPE_V);
  TEXPANDS(v143, v22);
  // pto: %8
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
  // pto: %8
  uint64_t v146 = (uint64_t) v36;
  TASSIGN(v145, v146);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v145, v21);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %9
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v147 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
  // pto: %9
  uint64_t v148 = (uint64_t) v36;
  TASSIGN(v147, v148);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  TCVT(v147, v145, v18, v17);
  // pto: %rope_col_inline1977__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %rope_col_inline1977__tile
  uint64_t v150 = (uint64_t) v34;
  TASSIGN(v149, v150);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v149, v143, v147);
  // pto: %10
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %10
  uint64_t v152 = (uint64_t) v36;
  TASSIGN(v151, v152);
  pipe_barrier(PIPE_V);
  TMULS(v151, v149, v20);
  // pto: %11
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %11
  uint64_t v154 = (uint64_t) v36;
  TASSIGN(v153, v154);
  pipe_barrier(PIPE_V);
  TCVT(v153, v151, v15, v17);
  // pto: %rope_dup_f_inline1976__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %rope_dup_f_inline1976__tile
  uint64_t v156 = (uint64_t) v36;
  TASSIGN(v155, v156);
  pipe_barrier(PIPE_V);
  TCVT(v155, v153, v18, v17);
  // pto: %12
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v157 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %12
  uint64_t v158 = (uint64_t) v36;
  TASSIGN(v157, v158);
  pipe_barrier(PIPE_V);
  TMULS(v157, v155, v19);
  // pto: %rope_lane_inline2048__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %rope_lane_inline2048__tile
  uint64_t v160 = (uint64_t) v36;
  TASSIGN(v159, v160);
  pipe_barrier(PIPE_V);
  TSUB(v159, v149, v157);
  // pto: %13
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v161 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %13
  uint64_t v162 = (uint64_t) v34;
  TASSIGN(v161, v162);
  pipe_barrier(PIPE_V);
  TADDS(v161, v149, v22);
  // pto: %14
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %14
  uint64_t v164 = (uint64_t) v36;
  TASSIGN(v163, v164);
  TMULS(v163, v159, v19);
  // pto: %15
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %15
  uint64_t v166 = (uint64_t) v34;
  TASSIGN(v165, v166);
  pipe_barrier(PIPE_V);
  TSUB(v165, v161, v163);
  // pto: %rope_swap_idx_inline1997__tile
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %rope_swap_idx_inline1997__tile
  uint64_t v168 = (uint64_t) v34;
  TASSIGN(v167, v168);
  pipe_barrier(PIPE_V);
  TCVT(v167, v165, v18, v17);
  // pto: %gather_acc_init
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %gather_acc_init
  uint64_t v170 = (uint64_t) v36;
  TASSIGN(v169, v170);
  for (int64_t i171 = v34; i171 < v28; i171 += v30) {
    // pto: %gather_inp_row
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v172 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
    // pto: %gather_inp_row
    uint64_t v173 = (uint64_t) v35;
    TASSIGN(v172, v173);
    // pto: %slice_view
    int64_t v174 = (int64_t) ((uint64_t) i171 * (uint64_t) v39);
    // pto: %slice_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v175;
    // pto: %slice_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v176 = v175;
    // pto: %slice_view
    uint64_t v177 = (uint64_t) ((int64_t) ((uint64_t) v174 + (uint64_t) v35));
    TASSIGN(v176, v177);
    // pto: %gather_idx_row
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v178 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
    // pto: %gather_idx_row
    uint64_t v179 = (uint64_t) v34;
    TASSIGN(v178, v179);
    // pto: %35
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180;
    // pto: %35
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v181 = v180;
    // pto: %35
    uint64_t v182 = (uint64_t) v174;
    TASSIGN(v181, v182);
    // pto: %gather_row_tmp
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
    // pto: %gather_row_tmp
    uint64_t v184 = (uint64_t) v33;
    TASSIGN(v183, v184);
    // pto: %gather_row
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v185 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v31);
    // pto: %gather_row
    uint64_t v186 = (uint64_t) v32;
    TASSIGN(v185, v186);
    pipe_barrier(PIPE_V);
    TGATHER(v185, v176, v181, v183);
    // pto: %assemble_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187;
    // pto: %assemble_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v188 = v187;
    // pto: %assemble_view
    uint64_t v189 = (uint64_t) ((int64_t) ((uint64_t) v174 + (uint64_t) v36));
    TASSIGN(v188, v189);
    pipe_barrier(PIPE_V);
    TMOV(v188, v185);
  }
  // pto: %swapped_inline1975__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v190 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %swapped_inline1975__tile
  uint64_t v191 = (uint64_t) v36;
  TASSIGN(v190, v191);
  // pto: %16
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %16
  uint64_t v193 = (uint64_t) v35;
  TASSIGN(v192, v193);
  TMUL(v192, v141, v53);
  // pto: %17
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %17
  uint64_t v195 = (uint64_t) v38;
  TASSIGN(v194, v195);
  pipe_barrier(PIPE_V);
  TMUL(v194, v190, v62);
  // pto: %rope_rot_inline2002__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v196 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28, v31);
  // pto: %rope_rot_inline2002__tile
  uint64_t v197 = (uint64_t) v35;
  TASSIGN(v196, v197);
  pipe_barrier(PIPE_V);
  TADD(v196, v192, v194);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  // pto: %normed_kv_inline2016__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 64> v198 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %normed_kv_inline2016__rv_v2_pview
  pto::Stride<8192, 8192, 8192, 512, 1> v199 = pto::Stride<8192, 8192, 8192, 512, 1>();
  // pto: %normed_kv_inline2016__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v200 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v4 + (v23 + v55 * v29), v198, v199);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TSTORE(v200, v196);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  for (int64_t i201 = v34; i201 < v52; i201 += v30) {
    // pto: %38
    int64_t v202 = (int64_t) ((uint64_t) v50 + (uint64_t) i201);
    // pto: %cache_row_i64_inline1974__tile
    int64_t v203 = (v8)[v202];
    // pto: %40
    if (v203 >= v34) {
      // pto: %kv_row_fp32_inline2043__tile
      Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v29);
      // pto: %kv_row_fp32_inline2043__tile
      uint64_t v205 = (uint64_t) v35;
      TASSIGN(v204, v205);
      // pto: %42
      int64_t v206 = v202 < v34 ? v34 : v202;
      // pto: %normed_kv_inline2016__tile_pview
      pto::Shape<1, 1, 1, 1, 512> v207 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %normed_kv_inline2016__tile_pview
      pto::Stride<512, 512, 512, 512, 1> v208 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %normed_kv_inline2016__tile_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v209 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v4 + (v34 + v206 * v29), v207, v208);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v204, v209);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      // pto: %kv_flat_inline2039__iter_v1_pview
      pto::Shape<1, 1, 1, 1, 512> v210 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %kv_flat_inline2039__iter_v1_pview
      pto::Stride<512, 512, 512, 512, 1> v211 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %kv_flat_inline2039__iter_v1_pview
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v212 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v7 + (v34 + v206 * v29), v210, v211);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v212, v204);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      // pto: %18
      Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v213 = Tile<TileType::Vec, bfloat16_t, 1, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v30, v29);
      // pto: %18
      uint64_t v214 = (uint64_t) v35;
      TASSIGN(v213, v214);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TCVT(v213, v204, v14, v17);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      // pto: %cmp_kv_cache_flat_inline2036__iter_v1_pview
      pto::Shape<1, 1, 1, 1, 512> v215 = pto::Shape<1, 1, 1, 1, 512>();
      // pto: %cmp_kv_cache_flat_inline2036__iter_v1_pview
      pto::Stride<512, 512, 512, 512, 1> v216 = pto::Stride<512, 512, 512, 512, 1>();
      // pto: %44, %cmp_kv_cache_flat_inline2036__iter_v1_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND> v217 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 512>, pto::Stride<512, 512, 512, 512, 1>, pto::Layout::ND>(v6 + (v34 + (v203 < v34 ? v34 : v203) * v29), v215, v216);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
      TSTORE(v217, v213);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    }
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}