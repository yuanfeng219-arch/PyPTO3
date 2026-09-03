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

AICORE void rmsnorm_rope(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ bfloat16_t* v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  RoundMode v10 = RoundMode::CAST_TRUNC;
  unsigned v11 = 64;
  RoundMode v12 = RoundMode::CAST_RINT;
  SaturationMode v13 = SaturationMode::OFF;
  RoundMode v14 = RoundMode::CAST_ROUND;
  const float v15 = 2.0f;
  const float v16 = 0.5f;
  const int32_t v17 = 0;
  const float v18 = 1.0f;
  const float v19 = 9.99999997E-7f;
  const float v20 = 0.0078125f;
  const float v21 = 0.0f;
  const int64_t v22 = 16;
  const int64_t v23 = 128;
  const int64_t v24 = 1;
  const int64_t v25 = 64;
  const int64_t v26 = 24640;
  const int64_t v27 = 12352;
  const int64_t v28 = 12288;
  const int64_t v29 = 0;
  const int64_t v30 = 16448;
  const int64_t v31 = 4096;
  const int64_t v32 = 28800;
  const int64_t v33 = 29056;
  const int64_t v34 = 24704;
  const int64_t v35 = 256;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  int64_t v36 = v7 * v25;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  int64_t v37 = v24 * v36;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v38 = pto::Shape<1, 1, 1, -1, -1>(v24, v24, v24, v7, v25);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v39 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v37, v37, v36, v25, v24);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v38, v39);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  int64_t v41 = v7 * v25;
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  int64_t v42 = v24 * v41;
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  pto::Shape<1, 1, 1, -1, -1> v43 = pto::Shape<1, 1, 1, -1, -1>(v24, v24, v24, v7, v25);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  pto::Stride<-1, -1, -1, -1, -1> v44 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v42, v42, v41, v25, v24);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v43, v44);
  // pto: %rms_blk_inline2092__ssa_v0, %26
  int64_t v46 = (int64_t) ((uint64_t) ((int64_t) v8) * (uint64_t) v22);
  // pto: %27
  int64_t v47 = (int64_t) ((uint64_t) v6 - (uint64_t) v46);
  // pto: %28
  int64_t v48 = v47 < v22 ? v47 : v22;
  // pto: %cos_b_inline2087__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v25);
  // pto: %cos_b_inline2087__tile
  uint64_t v50 = (uint64_t) v34;
  TASSIGN(v49, v50);
  // pto: %29
  int64_t v51 = v46 < v29 ? v29 : v46;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  __gm__ float* v52 = PTOAS__GLOBAL_TENSOR_DATA(v40);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  int64_t v53 = v48 * v25;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  int64_t v54 = v24 * v53;
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  pto::Shape<1, 1, 1, -1, 64> v55 = pto::Shape<1, 1, 1, -1, 64>(v24, v24, v24, v48, v25);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v54, v54, v53, v25, v24);
  // pto: %cmp_cos_il_full_inline1249__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v57 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v52 + ((v29 + v51 * v25) + v29 * v24), v55, v56);
  TLOAD(v49, v57);
  // pto: %sin_b_inline2085__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v48, v25);
  // pto: %sin_b_inline2085__tile
  uint64_t v59 = (uint64_t) v33;
  TASSIGN(v58, v59);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  __gm__ float* v60 = PTOAS__GLOBAL_TENSOR_DATA(v45);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  int64_t v61 = v48 * v25;
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  int64_t v62 = v24 * v61;
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  pto::Shape<1, 1, 1, -1, 64> v63 = pto::Shape<1, 1, 1, -1, 64>(v24, v24, v24, v48, v25);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  pto::Stride<-1, -1, -1, -1, -1> v64 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v62, v62, v61, v25, v24);
  // pto: %cmp_sin_signed_full_inline1263__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v65 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v60 + ((v29 + v51 * v25) + v29 * v24), v63, v64);
  TLOAD(v58, v65);
  // pto: %partial_sq_inline2155__tile
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %partial_sq_inline2155__tile
  uint64_t v67 = (uint64_t) v32;
  TASSIGN(v66, v67);
  TEXPANDS(v66, v21);
  // pto: %kv_rms_chunk_inline2084__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %kv_rms_chunk_inline2084__tile
  uint64_t v69 = (uint64_t) v31;
  TASSIGN(v68, v69);
  // pto: %pooled_kv_inline2131__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 64> v70 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %pooled_kv_inline2131__rv_v2_pview
  pto::Stride<2048, 2048, 2048, 128, 1> v71 = pto::Stride<2048, 2048, 2048, 128, 1>();
  // pto: %pooled_kv_inline2131__rv_v2_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v72 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v3 + (v29 + v51 * v23), v70, v71);
  TLOAD(v68, v72);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %0
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %0
  uint64_t v74 = (uint64_t) v30;
  TASSIGN(v73, v74);
  // pto: %33
  pto::Shape<1, 1, 1, 16, 64> v75 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %33
  pto::Stride<2048, 2048, 2048, 128, 1> v76 = pto::Stride<2048, 2048, 2048, 128, 1>();
  // pto: %33
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v77 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v3 + (v25 + v51 * v23), v75, v76);
  TLOAD(v73, v77);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  // pto: %kv_rms_sq_inline2083__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %kv_rms_sq_inline2083__tile
  uint64_t v79 = (uint64_t) v29;
  TASSIGN(v78, v79);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TMUL(v78, v68, v68);
  // pto: %tmp_tile
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v23);
  // pto: %tmp_tile
  uint64_t v81 = (uint64_t) v31;
  TASSIGN(v80, v81);
  // pto: %t__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v24);
  // pto: %t__tile
  uint64_t v83 = (uint64_t) v28;
  TASSIGN(v82, v83);
  pipe_barrier(PIPE_V);
  TROWSUM(v82, v78, v80);
  // pto: %kv_rms_rowsum_inline2165__tile
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v84 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %kv_rms_rowsum_inline2165__tile
  uint64_t v85 = (uint64_t) v28;
  TASSIGN(v84, v85);
  // pto: %1
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %1
  uint64_t v87 = (uint64_t) v31;
  TASSIGN(v86, v87);
  pipe_barrier(PIPE_V);
  TADD(v86, v66, v84);
  // pto: %2
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %2
  uint64_t v89 = (uint64_t) v27;
  TASSIGN(v88, v89);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  TMUL(v88, v73, v73);
  // pto: %3
  Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v23);
  // pto: %3
  uint64_t v91 = (uint64_t) v30;
  TASSIGN(v90, v91);
  // pto: %4
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v24);
  // pto: %4
  uint64_t v93 = (uint64_t) v26;
  TASSIGN(v92, v93);
  pipe_barrier(PIPE_V);
  TROWSUM(v92, v88, v90);
  // pto: %5
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %5
  uint64_t v95 = (uint64_t) v26;
  TASSIGN(v94, v95);
  // pto: %6
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %6
  uint64_t v97 = (uint64_t) v32;
  TASSIGN(v96, v97);
  pipe_barrier(PIPE_V);
  TADD(v96, v86, v94);
  // pto: %7
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %7
  uint64_t v99 = (uint64_t) v31;
  TASSIGN(v98, v99);
  pipe_barrier(PIPE_V);
  TMULS(v98, v96, v20);
  // pto: %8
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %8
  uint64_t v101 = (uint64_t) v31;
  TASSIGN(v100, v101);
  pipe_barrier(PIPE_V);
  TADDS(v100, v98, v19);
  // pto: %variance_inline2082__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v24);
  // pto: %variance_inline2082__tile
  uint64_t v103 = (uint64_t) v31;
  TASSIGN(v102, v103);
  // pto: %t__rm_a0_tmp_v0
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %t__rm_a0_tmp_v0
  uint64_t v105 = (uint64_t) v31;
  TASSIGN(v104, v105);
  // pto: %t__row_major_tmp_v1
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %t__row_major_tmp_v1
  uint64_t v107 = (uint64_t) v31;
  TASSIGN(v106, v107);
  pipe_barrier(PIPE_V);
  TSQRT(v106, v104);
  // pto: %9
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v24);
  // pto: %9
  uint64_t v109 = (uint64_t) v31;
  TASSIGN(v108, v109);
  // pto: %inv_rms_inline2122__rm_a0_tmp_v2
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %inv_rms_inline2122__rm_a0_tmp_v2
  uint64_t v111 = (uint64_t) v31;
  TASSIGN(v110, v111);
  // pto: %inv_rms_inline2122__row_major_tmp_v3
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v22);
  // pto: %inv_rms_inline2122__row_major_tmp_v3
  uint64_t v113 = (uint64_t) v27;
  TASSIGN(v112, v113);
  pipe_barrier(PIPE_V);
  TRECIP(v112, v110);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  // pto: %inv_rms_inline2122__tile
  Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v24);
  // pto: %inv_rms_inline2122__tile
  uint64_t v115 = (uint64_t) v27;
  TASSIGN(v114, v115);
  // pto: %kv_norm_chunk_inline2080__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %kv_norm_chunk_inline2080__tile
  uint64_t v117 = (uint64_t) v31;
  TASSIGN(v116, v117);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TLOAD(v116, v72);
  // pto: %10
  Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
  // pto: %10
  uint64_t v119 = (uint64_t) v29;
  TASSIGN(v118, v119);
  // pto: %norm_w_2d_inline2173__ssa_v0_pview
  pto::Shape<1, 1, 1, 1, 64> v120 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %norm_w_2d_inline2173__ssa_v0_pview
  pto::Stride<128, 128, 128, 128, 1> v121 = pto::Stride<128, 128, 128, 128, 1>();
  // pto: %norm_w_2d_inline2173__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v122 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v5, v120, v121);
  TLOAD(v118, v122);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  // pto: %gamma_inline2078__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
  // pto: %gamma_inline2078__tile
  uint64_t v124 = (uint64_t) v30;
  TASSIGN(v123, v124);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  TCVT(v123, v118, v14, v13);
  // pto: %11
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %11
  uint64_t v126 = (uint64_t) v31;
  TASSIGN(v125, v126);
  TROWEXPANDMUL(v125, v116, v114);
  // pto: %normed_chunk_inline2102__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %normed_chunk_inline2102__tile
  uint64_t v128 = (uint64_t) v31;
  TASSIGN(v127, v128);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v127, v125, v123);
  // pto: %12
  Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %12
  uint64_t v130 = (uint64_t) v31;
  TASSIGN(v129, v130);
  pipe_barrier(PIPE_V);
  TCVT(v129, v127, v12, v13);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %normed_kv_inline2164__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 64> v131 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %normed_kv_inline2164__ssa_v0_pview
  pto::Stride<2048, 2048, 2048, 128, 1> v132 = pto::Stride<2048, 2048, 2048, 128, 1>();
  // pto: %normed_kv_inline2164__ssa_v0_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v133 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v4 + (v29 + v51 * v23), v131, v132);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v133, v129);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  // pto: %kv_rope_norm_inline2135__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %kv_rope_norm_inline2135__tile
  uint64_t v135 = (uint64_t) v31;
  TASSIGN(v134, v135);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  TLOAD(v134, v77);
  // pto: %13
  Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v136 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
  // pto: %13
  uint64_t v137 = (uint64_t) v29;
  TASSIGN(v136, v137);
  // pto: %39
  pto::Shape<1, 1, 1, 1, 64> v138 = pto::Shape<1, 1, 1, 1, 64>();
  // pto: %39
  pto::Stride<128, 128, 128, 128, 1> v139 = pto::Stride<128, 128, 128, 128, 1>();
  // pto: %39
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v140 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v5 + v11, v138, v139);
  TLOAD(v136, v140);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
  // pto: %gamma_rope_inline2077__tile
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
  // pto: %gamma_rope_inline2077__tile
  uint64_t v142 = (uint64_t) v30;
  TASSIGN(v141, v142);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
  TCVT(v141, v136, v14, v13);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  // pto: %14
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %14
  uint64_t v144 = (uint64_t) v31;
  TASSIGN(v143, v144);
  TROWEXPANDMUL(v143, v134, v114);
  // pto: %rope_normed_inline2138__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %rope_normed_inline2138__tile
  uint64_t v146 = (uint64_t) v31;
  TASSIGN(v145, v146);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v145, v143, v141);
  // pto: %rope_ones_inline2089__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v147 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %rope_ones_inline2089__tile
  uint64_t v148 = (uint64_t) v30;
  TASSIGN(v147, v148);
  pipe_barrier(PIPE_V);
  TEXPANDS(v147, v18);
  // pto: %15
  Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
  // pto: %15
  uint64_t v150 = (uint64_t) v29;
  TASSIGN(v149, v150);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v149, v17);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  // pto: %16
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
  // pto: %16
  uint64_t v152 = (uint64_t) v29;
  TASSIGN(v151, v152);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  TCVT(v151, v149, v14, v13);
  // pto: %rope_col_inline2076__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %rope_col_inline2076__tile
  uint64_t v154 = (uint64_t) v30;
  TASSIGN(v153, v154);
  pipe_barrier(PIPE_V);
  TCOLEXPANDMUL(v153, v147, v151);
  // pto: %17
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %17
  uint64_t v156 = (uint64_t) v29;
  TASSIGN(v155, v156);
  pipe_barrier(PIPE_V);
  TMULS(v155, v153, v16);
  // pto: %18
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v157 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %18
  uint64_t v158 = (uint64_t) v29;
  TASSIGN(v157, v158);
  pipe_barrier(PIPE_V);
  TCVT(v157, v155, v10, v13);
  // pto: %rope_dup_f_inline2074__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %rope_dup_f_inline2074__tile
  uint64_t v160 = (uint64_t) v29;
  TASSIGN(v159, v160);
  pipe_barrier(PIPE_V);
  TCVT(v159, v157, v14, v13);
  // pto: %19
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v161 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %19
  uint64_t v162 = (uint64_t) v29;
  TASSIGN(v161, v162);
  pipe_barrier(PIPE_V);
  TMULS(v161, v159, v15);
  // pto: %rope_lane_inline2073__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %rope_lane_inline2073__tile
  uint64_t v164 = (uint64_t) v29;
  TASSIGN(v163, v164);
  pipe_barrier(PIPE_V);
  TSUB(v163, v153, v161);
  // pto: %20
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %20
  uint64_t v166 = (uint64_t) v30;
  TASSIGN(v165, v166);
  pipe_barrier(PIPE_V);
  TADDS(v165, v153, v18);
  // pto: %21
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %21
  uint64_t v168 = (uint64_t) v29;
  TASSIGN(v167, v168);
  TMULS(v167, v163, v15);
  // pto: %22
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %22
  uint64_t v170 = (uint64_t) v30;
  TASSIGN(v169, v170);
  pipe_barrier(PIPE_V);
  TSUB(v169, v165, v167);
  // pto: %rope_swap_idx_inline2079__tile
  Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %rope_swap_idx_inline2079__tile
  uint64_t v172 = (uint64_t) v30;
  TASSIGN(v171, v172);
  pipe_barrier(PIPE_V);
  TCVT(v171, v169, v14, v13);
  // pto: %gather_acc_init
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %gather_acc_init
  uint64_t v174 = (uint64_t) v29;
  TASSIGN(v173, v174);
  for (int64_t i175 = v29; i175 < v22; i175 += v24) {
    // pto: %gather_inp_row
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v176 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gather_inp_row
    uint64_t v177 = (uint64_t) v31;
    TASSIGN(v176, v177);
    // pto: %slice_view
    int64_t v178 = (int64_t) ((uint64_t) i175 * (uint64_t) v35);
    // pto: %slice_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v179;
    // pto: %slice_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v180 = v179;
    // pto: %slice_view
    uint64_t v181 = (uint64_t) ((int64_t) ((uint64_t) v178 + (uint64_t) v31));
    TASSIGN(v180, v181);
    // pto: %gather_idx_row
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v182 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gather_idx_row
    uint64_t v183 = (uint64_t) v30;
    TASSIGN(v182, v183);
    // pto: %40
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184;
    // pto: %40
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v185 = v184;
    // pto: %40
    uint64_t v186 = (uint64_t) ((int64_t) ((uint64_t) v178 + (uint64_t) v30));
    TASSIGN(v185, v186);
    // pto: %gather_row_tmp
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gather_row_tmp
    uint64_t v188 = (uint64_t) v27;
    TASSIGN(v187, v188);
    // pto: %gather_row
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v189 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v24, v25);
    // pto: %gather_row
    uint64_t v190 = (uint64_t) v32;
    TASSIGN(v189, v190);
    pipe_barrier(PIPE_V);
    TGATHER(v189, v180, v185, v187);
    // pto: %assemble_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v191;
    // pto: %assemble_view
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v192 = v191;
    // pto: %assemble_view
    uint64_t v193 = (uint64_t) v178;
    TASSIGN(v192, v193);
    pipe_barrier(PIPE_V);
    TMOV(v192, v189);
  }
  // pto: %swapped_inline2071__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %swapped_inline2071__tile
  uint64_t v195 = (uint64_t) v29;
  TASSIGN(v194, v195);
  // pto: %23
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v196 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %23
  uint64_t v197 = (uint64_t) v31;
  TASSIGN(v196, v197);
  TMUL(v196, v145, v49);
  // pto: %24
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v198 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %24
  uint64_t v199 = (uint64_t) v30;
  TASSIGN(v198, v199);
  pipe_barrier(PIPE_V);
  TMUL(v198, v194, v58);
  // pto: %rope_rot_inline2070__tile
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v200 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %rope_rot_inline2070__tile
  uint64_t v201 = (uint64_t) v31;
  TASSIGN(v200, v201);
  pipe_barrier(PIPE_V);
  TADD(v200, v196, v198);
  // pto: %25
  Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v202 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v25);
  // pto: %25
  uint64_t v203 = (uint64_t) v31;
  TASSIGN(v202, v203);
  pipe_barrier(PIPE_V);
  TCVT(v202, v200, v12, v13);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  // pto: %normed_kv_inline2164__rv_v2_pview
  pto::Shape<1, 1, 1, 16, 64> v204 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %normed_kv_inline2164__rv_v2_pview
  pto::Stride<2048, 2048, 2048, 128, 1> v205 = pto::Stride<2048, 2048, 2048, 128, 1>();
  // pto: %normed_kv_inline2164__rv_v2_pview
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v206 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v4 + (v25 + v51 * v23), v204, v205);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TSTORE(v206, v202);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}