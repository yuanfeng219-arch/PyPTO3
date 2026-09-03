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

AICORE void kv_rms_norm_rope(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ int32_t* v6, int64_t v7, int64_t v8, int64_t v9, int64_t v10, int64_t v11, int32_t v12, int32_t v13) {
  RoundMode v14 = RoundMode::CAST_TRUNC;
  unsigned v15 = 448;
  unsigned v16 = 384;
  RoundMode v17 = RoundMode::CAST_RINT;
  SaturationMode v18 = SaturationMode::OFF;
  RoundMode v19 = RoundMode::CAST_ROUND;
  const float v20 = 64.0f;
  const float v21 = 2.0f;
  const float v22 = 0.5f;
  const int32_t v23 = 0;
  const float v24 = 1.0f;
  const int64_t v25 = 448;
  const int64_t v26 = 384;
  const float v27 = 9.99999997E-7f;
  const float v28 = 0.001953125f;
  const int64_t v29 = 128;
  const float v30 = 0.0f;
  const int64_t v31 = 16;
  const int64_t v32 = 64;
  const int64_t v33 = 1;
  const int64_t v34 = 512;
  const int64_t v35 = 12288;
  const int64_t v36 = 16384;
  const int64_t v37 = 8192;
  const int64_t v38 = 32768;
  const int64_t v39 = 20480;
  const int64_t v40 = 0;
  const int64_t v41 = 28672;
  const int64_t v42 = 256;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %kv_fp32_inline1920__rv_v7_view
  int64_t v43 = v9 * v34;
  // pto: %kv_fp32_inline1920__rv_v7_view
  int64_t v44 = v33 * v43;
  // pto: %kv_fp32_inline1920__rv_v7_view
  pto::Shape<1, 1, 1, -1, -1> v45 = pto::Shape<1, 1, 1, -1, -1>(v33, v33, v33, v9, v34);
  // pto: %kv_fp32_inline1920__rv_v7_view
  pto::Stride<-1, -1, -1, -1, -1> v46 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v44, v44, v43, v34, v33);
  // pto: %kv_fp32_inline1920__rv_v7_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v45, v46);
  // pto: %kv_view_inline1909__ssa_v0_view
  int64_t v48 = v10 * v34;
  // pto: %kv_view_inline1909__ssa_v0_view
  int64_t v49 = v33 * v48;
  // pto: %kv_view_inline1909__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v50 = pto::Shape<1, 1, 1, -1, -1>(v33, v33, v33, v10, v34);
  // pto: %kv_view_inline1909__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v51 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v49, v49, v48, v34, v33);
  // pto: %kv_view_inline1909__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v52 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v50, v51);
  // pto: %kv_cos_il_inline1258__ssa_v0_view
  int64_t v53 = v11 * v32;
  // pto: %kv_cos_il_inline1258__ssa_v0_view
  int64_t v54 = v33 * v53;
  // pto: %kv_cos_il_inline1258__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v55 = pto::Shape<1, 1, 1, -1, -1>(v33, v33, v33, v11, v32);
  // pto: %kv_cos_il_inline1258__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v56 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v54, v54, v53, v32, v33);
  // pto: %kv_cos_il_inline1258__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v57 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v55, v56);
  // pto: %kv_sin_signed_inline1301__ssa_v0_view
  int64_t v58 = v11 * v32;
  // pto: %kv_sin_signed_inline1301__ssa_v0_view
  int64_t v59 = v33 * v58;
  // pto: %kv_sin_signed_inline1301__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v60 = pto::Shape<1, 1, 1, -1, -1>(v33, v33, v33, v11, v32);
  // pto: %kv_sin_signed_inline1301__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v61 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v59, v59, v58, v32, v33);
  // pto: %kv_sin_signed_inline1301__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v62 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v5, v60, v61);
  // pto: %tg_idx_inline1916__ssa_v0, %51
  int64_t v63 = (int64_t) ((uint64_t) ((int64_t) v12) * (uint64_t) v31);
  // pto: %52
  int64_t v64 = (int64_t) ((uint64_t) v7 - (uint64_t) v63);
  // pto: %53
  int64_t v65 = v64 < v31 ? v64 : v31;
  // pto: %54
  int64_t v66 = (int64_t) ((uint64_t) v8 + (uint64_t) v63);
  // pto: %55
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  if (v65 == v31) {
    // pto: %kv_sq_sum_inline1961__tile
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %kv_sq_sum_inline1961__tile
    uint64_t v68 = (uint64_t) v41;
    TASSIGN(v67, v68);
    TEXPANDS(v67, v30);
    for (int64_t i69 = v40; i69 < v34; i69 += v29) {
      // pto: %kv_chunk_inline1941__tile
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %kv_chunk_inline1941__tile
      uint64_t v71 = (uint64_t) v40;
      TASSIGN(v70, v71);
      // pto: %56
      int64_t v72 = v63 < v40 ? v40 : v63;
      // pto: %kv_fp32_inline1920__rv_v7_pview
      pto::Shape<1, 1, 1, 16, 64> v73 = pto::Shape<1, 1, 1, 16, 64>();
      // pto: %kv_fp32_inline1920__rv_v7_pview
      pto::Stride<8192, 8192, 8192, 512, 1> v74 = pto::Stride<8192, 8192, 8192, 512, 1>();
      // pto: %kv_fp32_inline1920__rv_v7_pview, %57
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v75 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v1 + ((v40 + v72 * v34) + (i69 < v40 ? v40 : i69)), v73, v74);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v70, v75);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %0
      uint64_t v77 = (uint64_t) v39;
      TASSIGN(v76, v77);
      // pto: %59
      int64_t v78 = (int64_t) ((uint64_t) i69 + (uint64_t) v32);
      // pto: %61
      pto::Shape<1, 1, 1, 16, 64> v79 = pto::Shape<1, 1, 1, 16, 64>();
      // pto: %61
      pto::Stride<8192, 8192, 8192, 512, 1> v80 = pto::Stride<8192, 8192, 8192, 512, 1>();
      // pto: %61, %60
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v81 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v1 + ((v40 + v72 * v34) + (v78 < v40 ? v40 : v78)), v79, v80);
      TLOAD(v76, v81);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %kv_sq_inline1968__tile
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %kv_sq_inline1968__tile
      uint64_t v83 = (uint64_t) v38;
      TASSIGN(v82, v83);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TMUL(v82, v70, v70);
      // pto: %tmp_tile
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v84 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
      // pto: %tmp_tile
      uint64_t v85 = (uint64_t) v40;
      TASSIGN(v84, v85);
      // pto: %t__tile
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %t__tile
      uint64_t v87 = (uint64_t) v37;
      TASSIGN(v86, v87);
      pipe_barrier(PIPE_V);
      TROWSUM(v86, v82, v84);
      // pto: %kv_row_sum_inline1946__tile
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %kv_row_sum_inline1946__tile
      uint64_t v89 = (uint64_t) v37;
      TASSIGN(v88, v89);
      // pto: %1
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %1
      uint64_t v91 = (uint64_t) v40;
      TASSIGN(v90, v91);
      pipe_barrier(PIPE_V);
      TADD(v90, v67, v88);
      // pto: %2
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %2
      uint64_t v93 = (uint64_t) v36;
      TASSIGN(v92, v93);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TMUL(v92, v76, v76);
      // pto: %3
      Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v29);
      // pto: %3
      uint64_t v95 = (uint64_t) v39;
      TASSIGN(v94, v95);
      // pto: %4
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
      // pto: %4
      uint64_t v97 = (uint64_t) v35;
      TASSIGN(v96, v97);
      pipe_barrier(PIPE_V);
      TROWSUM(v96, v92, v94);
      // pto: %5
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %5
      uint64_t v99 = (uint64_t) v35;
      TASSIGN(v98, v99);
      // pto: %6
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %6
      uint64_t v101 = (uint64_t) v41;
      TASSIGN(v100, v101);
      pipe_barrier(PIPE_V);
      TADD(v100, v90, v98);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    // pto: %7
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %7
    uint64_t v103 = (uint64_t) v40;
    TASSIGN(v102, v103);
    pipe_barrier(PIPE_V);
    TMULS(v102, v67, v28);
    // pto: %8
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %8
    uint64_t v105 = (uint64_t) v40;
    TASSIGN(v104, v105);
    pipe_barrier(PIPE_V);
    TADDS(v104, v102, v27);
    // pto: %rsqrt_tmp
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v106 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %rsqrt_tmp
    uint64_t v107 = (uint64_t) v39;
    TASSIGN(v106, v107);
    // pto: %kv_inv_rms_inline1942__tile
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v108 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %kv_inv_rms_inline1942__tile
    uint64_t v109 = (uint64_t) v35;
    TASSIGN(v108, v109);
    pipe_barrier(PIPE_V);
    TRSQRT(v108, v104, v106);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    // pto: %kv_inv_rms_t_inline1949__tile
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
    // pto: %kv_inv_rms_t_inline1949__tile
    uint64_t v111 = (uint64_t) v35;
    TASSIGN(v110, v111);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int64_t i112 = v40; i112 < v26; i112 += v29) {
      // pto: %9
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %9
      uint64_t v114 = (uint64_t) v40;
      TASSIGN(v113, v114);
      // pto: %62
      int64_t v115 = v63 < v40 ? v40 : v63;
      // pto: %63
      int64_t v116 = i112 < v40 ? v40 : i112;
      // pto: %64
      pto::Shape<1, 1, 1, 16, 64> v117 = pto::Shape<1, 1, 1, 16, 64>();
      // pto: %64
      pto::Stride<8192, 8192, 8192, 512, 1> v118 = pto::Stride<8192, 8192, 8192, 512, 1>();
      // pto: %64
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v119 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v1 + ((v40 + v115 * v34) + v116), v117, v118);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v113, v119);
      // pto: %10
      Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %10
      uint64_t v121 = (uint64_t) v41;
      TASSIGN(v120, v121);
      // pto: %gamma_ckv__ssa_v0_pview
      pto::Shape<1, 1, 1, 1, 64> v122 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %gamma_ckv__ssa_v0_pview
      pto::Stride<64, 64, 64, 64, 1> v123 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %gamma_ckv__ssa_v0_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v124 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v40 + v116), v122, v123);
      TLOAD(v120, v124);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %11
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %11
      uint64_t v126 = (uint64_t) v39;
      TASSIGN(v125, v126);
      // pto: %67
      int64_t v127 = (int64_t) ((uint64_t) i112 + (uint64_t) v32);
      // pto: %68
      int64_t v128 = v127 < v40 ? v40 : v127;
      // pto: %69
      pto::Shape<1, 1, 1, 16, 64> v129 = pto::Shape<1, 1, 1, 16, 64>();
      // pto: %69
      pto::Stride<8192, 8192, 8192, 512, 1> v130 = pto::Stride<8192, 8192, 8192, 512, 1>();
      // pto: %69
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v131 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v1 + ((v40 + v115 * v34) + v128), v129, v130);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v125, v131);
      // pto: %12
      Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v132 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %12
      uint64_t v133 = (uint64_t) v37;
      TASSIGN(v132, v133);
      // pto: %72
      pto::Shape<1, 1, 1, 1, 64> v134 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %72
      pto::Stride<64, 64, 64, 64, 1> v135 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %72
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v136 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v40 + v128), v134, v135);
      TLOAD(v132, v136);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %gamma_kv_cast_inline1948__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gamma_kv_cast_inline1948__tile
      uint64_t v138 = (uint64_t) v38;
      TASSIGN(v137, v138);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TCVT(v137, v120, v19, v18);
      // pto: %gamma_kv_chunk_inline1951__tile
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gamma_kv_chunk_inline1951__tile
      uint64_t v140 = (uint64_t) v38;
      TASSIGN(v139, v140);
      // pto: %13
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %13
      uint64_t v142 = (uint64_t) v40;
      TASSIGN(v141, v142);
      TROWEXPANDMUL(v141, v113, v110);
      // pto: %kv_normed_inline1925__tile
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %kv_normed_inline1925__tile
      uint64_t v144 = (uint64_t) v40;
      TASSIGN(v143, v144);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v143, v141, v139);
      // pto: %kv_normed_bf16_inline1908__tile
      Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %kv_normed_bf16_inline1908__tile
      uint64_t v146 = (uint64_t) v40;
      TASSIGN(v145, v146);
      pipe_barrier(PIPE_V);
      TCVT(v145, v143, v17, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // pto: %73
      int64_t v147 = v66 < v40 ? v40 : v66;
      // pto: %kv_view_inline1909__iter_v1_pview
      pto::Shape<1, 1, 1, 16, 64> v148 = pto::Shape<1, 1, 1, 16, 64>();
      // pto: %kv_view_inline1909__iter_v1_pview
      pto::Stride<8192, 8192, 8192, 512, 1> v149 = pto::Stride<8192, 8192, 8192, 512, 1>();
      // pto: %kv_view_inline1909__iter_v1_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v150 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v2 + ((v40 + v147 * v34) + v116), v148, v149);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v150, v145);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      // pto: %14
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %14
      uint64_t v152 = (uint64_t) v36;
      TASSIGN(v151, v152);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TCVT(v151, v132, v19, v18);
      // pto: %15
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %15
      uint64_t v154 = (uint64_t) v36;
      TASSIGN(v153, v154);
      // pto: %16
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %16
      uint64_t v156 = (uint64_t) v39;
      TASSIGN(v155, v156);
      TROWEXPANDMUL(v155, v125, v110);
      // pto: %17
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v157 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %17
      uint64_t v158 = (uint64_t) v39;
      TASSIGN(v157, v158);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v157, v155, v153);
      // pto: %18
      Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
      // pto: %18
      uint64_t v160 = (uint64_t) v39;
      TASSIGN(v159, v160);
      pipe_barrier(PIPE_V);
      TCVT(v159, v157, v17, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      // pto: %kv_view_inline1909__tile_pview
      pto::Shape<1, 1, 1, 16, 64> v161 = pto::Shape<1, 1, 1, 16, 64>();
      // pto: %kv_view_inline1909__tile_pview
      pto::Stride<8192, 8192, 8192, 512, 1> v162 = pto::Stride<8192, 8192, 8192, 512, 1>();
      // pto: %kv_view_inline1909__tile_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v163 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v2 + ((v40 + v147 * v34) + v128), v161, v162);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v163, v159);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    // pto: %19
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %19
    uint64_t v165 = (uint64_t) v40;
    TASSIGN(v164, v165);
    // pto: %79
    int64_t v166 = v63 < v40 ? v40 : v63;
    // pto: %80
    pto::Shape<1, 1, 1, 16, 64> v167 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %80
    pto::Stride<8192, 8192, 8192, 512, 1> v168 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %80
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v169 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v1 + (v26 + v166 * v34), v167, v168);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v164, v169);
    // pto: %20
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %20
    uint64_t v171 = (uint64_t) v38;
    TASSIGN(v170, v171);
    // pto: %81
    pto::Shape<1, 1, 1, 1, 64> v172 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %81
    pto::Stride<64, 64, 64, 64, 1> v173 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %81
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v174 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + v16, v172, v173);
    TLOAD(v170, v174);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    // pto: %21
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v175 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %21
    uint64_t v176 = (uint64_t) v39;
    TASSIGN(v175, v176);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TCVT(v175, v170, v19, v18);
    // pto: %22
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %22
    uint64_t v178 = (uint64_t) v39;
    TASSIGN(v177, v178);
    // pto: %23
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v179 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %23
    uint64_t v180 = (uint64_t) v40;
    TASSIGN(v179, v180);
    TROWEXPANDMUL(v179, v164, v110);
    // pto: %24
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v181 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %24
    uint64_t v182 = (uint64_t) v40;
    TASSIGN(v181, v182);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v181, v179, v177);
    // pto: %25
    Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %25
    uint64_t v184 = (uint64_t) v40;
    TASSIGN(v183, v184);
    pipe_barrier(PIPE_V);
    TCVT(v183, v181, v17, v18);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %83
    int64_t v185 = v66 < v40 ? v40 : v66;
    // pto: %kv_view_inline1909__rv_v2_main_pview
    pto::Shape<1, 1, 1, 16, 64> v186 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %kv_view_inline1909__rv_v2_main_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v187 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %kv_view_inline1909__rv_v2_main_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v188 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v2 + (v26 + v185 * v34), v186, v187);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v188, v183);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    // pto: %26
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v189 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %26
    uint64_t v190 = (uint64_t) v40;
    TASSIGN(v189, v190);
    // pto: %84
    pto::Shape<1, 1, 1, 1, 64> v191 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %84
    pto::Stride<64, 64, 64, 64, 1> v192 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %84
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v193 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + v15, v191, v192);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    TLOAD(v189, v193);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    // pto: %gamma_rope_cast_inline1955__tile
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v194 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %gamma_rope_cast_inline1955__tile
    uint64_t v195 = (uint64_t) v39;
    TASSIGN(v194, v195);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    TCVT(v194, v189, v19, v18);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    // pto: %gamma_rope_inline1957__tile
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v196 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %gamma_rope_inline1957__tile
    uint64_t v197 = (uint64_t) v39;
    TASSIGN(v196, v197);
    // pto: %kv_rope_chunk_inline1959__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v198 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_rope_chunk_inline1959__tile
    uint64_t v199 = (uint64_t) v40;
    TASSIGN(v198, v199);
    // pto: %86
    pto::Shape<1, 1, 1, 16, 64> v200 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %86
    pto::Stride<8192, 8192, 8192, 512, 1> v201 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %86
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v202 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v1 + (v25 + v166 * v34), v200, v201);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    TLOAD(v198, v202);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    // pto: %27
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %27
    uint64_t v204 = (uint64_t) v40;
    TASSIGN(v203, v204);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
    TROWEXPANDMUL(v203, v198, v110);
    // pto: %kv_rope_norm_chunk_inline1945__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v205 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_rope_norm_chunk_inline1945__tile
    uint64_t v206 = (uint64_t) v40;
    TASSIGN(v205, v206);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v205, v203, v196);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    // pto: %kv_cos_il_full_inline1963__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_cos_il_full_inline1963__tile
    uint64_t v208 = (uint64_t) v39;
    TASSIGN(v207, v208);
    // pto: %kv_cos_il_inline1258__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 64> v209 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %kv_cos_il_inline1258__ssa_v0_pview
    pto::Stride<1024, 1024, 1024, 64, 1> v210 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %kv_cos_il_inline1258__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v211 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v4 + (v40 + v185 * v32), v209, v210);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    TLOAD(v207, v211);
    // pto: %kv_sin_signed_full_inline1956__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_sin_signed_full_inline1956__tile
    uint64_t v213 = (uint64_t) v38;
    TASSIGN(v212, v213);
    // pto: %kv_sin_signed_inline1301__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 64> v214 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %kv_sin_signed_inline1301__ssa_v0_pview
    pto::Stride<1024, 1024, 1024, 64, 1> v215 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %kv_sin_signed_inline1301__ssa_v0_pview
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v216 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v5 + (v40 + v185 * v32), v214, v215);
    TLOAD(v212, v216);
    // pto: %kv_swap_idx_full_inline1953__tile
    Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v217 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_swap_idx_full_inline1953__tile
    uint64_t v218 = (uint64_t) v36;
    TASSIGN(v217, v218);
    // pto: %kv_swap_idx_inline1305__ssa_v0_pview
    pto::Shape<1, 1, 1, 16, 64> v219 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %kv_swap_idx_inline1305__ssa_v0_pview
    pto::Stride<1024, 1024, 1024, 64, 1> v220 = pto::Stride<1024, 1024, 1024, 64, 1>();
    // pto: %kv_swap_idx_inline1305__ssa_v0_pview
    GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v221 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v6 + (v40 + v185 * v32), v219, v220);
    TLOAD(v217, v221);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    // pto: %gather_acc_init
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v222 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %gather_acc_init
    uint64_t v223 = (uint64_t) v41;
    TASSIGN(v222, v223);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    for (int64_t i224 = v40; i224 < v31; i224 += v33) {
      // pto: %gather_inp_row
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v225 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gather_inp_row
      uint64_t v226 = (uint64_t) v40;
      TASSIGN(v225, v226);
      // pto: %slice_view
      int64_t v227 = (int64_t) ((uint64_t) i224 * (uint64_t) v42);
      // pto: %slice_view
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228;
      // pto: %slice_view
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v229 = v228;
      // pto: %slice_view
      uint64_t v230 = (uint64_t) v227;
      TASSIGN(v229, v230);
      // pto: %gather_idx_row
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v231 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gather_idx_row
      uint64_t v232 = (uint64_t) v36;
      TASSIGN(v231, v232);
      // pto: %90
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v233;
      // pto: %90
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234 = v233;
      // pto: %90
      uint64_t v235 = (uint64_t) ((int64_t) ((uint64_t) v227 + (uint64_t) v36));
      TASSIGN(v234, v235);
      // pto: %gather_row_tmp
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gather_row_tmp
      uint64_t v237 = (uint64_t) v37;
      TASSIGN(v236, v237);
      // pto: %gather_row
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gather_row
      uint64_t v239 = (uint64_t) v35;
      TASSIGN(v238, v239);
      pipe_barrier(PIPE_V);
      TGATHER(v238, v229, v234, v236);
      // pto: %assemble_view
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240;
      // pto: %assemble_view
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v241 = v240;
      // pto: %assemble_view
      uint64_t v242 = (uint64_t) ((int64_t) ((uint64_t) v227 + (uint64_t) v41));
      TASSIGN(v241, v242);
      pipe_barrier(PIPE_V);
      TMOV(v241, v238);
    }
    // pto: %kv_swapped_full_inline1912__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v243 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_swapped_full_inline1912__tile
    uint64_t v244 = (uint64_t) v41;
    TASSIGN(v243, v244);
    // pto: %28
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v245 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %28
    uint64_t v246 = (uint64_t) v40;
    TASSIGN(v245, v246);
    TMUL(v245, v205, v207);
    // pto: %29
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v247 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %29
    uint64_t v248 = (uint64_t) v39;
    TASSIGN(v247, v248);
    pipe_barrier(PIPE_V);
    TMUL(v247, v243, v212);
    // pto: %kv_rope_rot_full_inline1921__tile
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v249 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_rope_rot_full_inline1921__tile
    uint64_t v250 = (uint64_t) v40;
    TASSIGN(v249, v250);
    pipe_barrier(PIPE_V);
    TADD(v249, v245, v247);
    // pto: %kv_rope_i16_full_inline1965__tile
    Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v251 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_rope_i16_full_inline1965__tile
    uint64_t v252 = (uint64_t) v40;
    TASSIGN(v251, v252);
    pipe_barrier(PIPE_V);
    TCVT(v251, v249, v17, v18);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    // pto: %kv_view_inline1909__rv_v2_pview
    pto::Shape<1, 1, 1, 16, 64> v253 = pto::Shape<1, 1, 1, 16, 64>();
    // pto: %kv_view_inline1909__rv_v2_pview
    pto::Stride<8192, 8192, 8192, 512, 1> v254 = pto::Stride<8192, 8192, 8192, 512, 1>();
    // pto: %kv_view_inline1909__rv_v2_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v255 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v2 + (v25 + v185 * v34), v253, v254);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v255, v251);
  } else {
    // pto: %kv_reduce_tmp_inline1967__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_reduce_tmp_inline1967__ssa_v0
    uint64_t v257 = (uint64_t) v40;
    TASSIGN(v256, v257);
    // pto: %kv_sq_sum_tail_inline1970__ssa_v0
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v258 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %kv_sq_sum_tail_inline1970__ssa_v0
    uint64_t v259 = (uint64_t) v36;
    TASSIGN(v258, v259);
    TEXPANDS(v258, v30);
    for (int64_t i260 = v40; i260 < v34; i260 += v29) {
      // pto: %kv_chunk_tail_inline1939__ssa_v0
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %kv_chunk_tail_inline1939__ssa_v0
      uint64_t v262 = (uint64_t) v39;
      TASSIGN(v261, v262);
      // pto: %93
      int64_t v263 = v63 < v40 ? v40 : v63;
      // pto: %95
      __gm__ float* v264 = PTOAS__GLOBAL_TENSOR_DATA(v47);
      // pto: %95
      int64_t v265 = v65 * v34;
      // pto: %95
      int64_t v266 = v33 * v265;
      // pto: %95
      pto::Shape<1, 1, 1, -1, 64> v267 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
      // pto: %95
      pto::Stride<-1, -1, -1, -1, -1> v268 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v266, v266, v265, v34, v33);
      // pto: %95, %94
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v269 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v264 + ((v40 + v263 * v34) + (i260 < v40 ? v40 : i260) * v33), v267, v268);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v261, v269);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %30
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v270 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %30
      uint64_t v271 = (uint64_t) v38;
      TASSIGN(v270, v271);
      // pto: %97
      int64_t v272 = (int64_t) ((uint64_t) i260 + (uint64_t) v32);
      // pto: %99
      __gm__ float* v273 = PTOAS__GLOBAL_TENSOR_DATA(v47);
      // pto: %99
      int64_t v274 = v65 * v34;
      // pto: %99
      int64_t v275 = v33 * v274;
      // pto: %99
      pto::Shape<1, 1, 1, -1, 64> v276 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
      // pto: %99
      pto::Stride<-1, -1, -1, -1, -1> v277 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v275, v275, v274, v34, v33);
      // pto: %99, %98
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v278 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v273 + ((v40 + v263 * v34) + (v272 < v40 ? v40 : v272) * v33), v276, v277);
      TLOAD(v270, v278);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %kv_sq_tail_inline1971__ssa_v0
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %kv_sq_tail_inline1971__ssa_v0
      uint64_t v280 = (uint64_t) v39;
      TASSIGN(v279, v280);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TMUL(v279, v261, v261);
      // pto: %t__tmp_v165
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v33);
      // pto: %t__tmp_v165
      uint64_t v282 = (uint64_t) v41;
      TASSIGN(v281, v282);
      pipe_barrier(PIPE_V);
      TROWSUM(v281, v279, v256);
      // pto: %kv_row_sum_tail_inline1973__ssa_v0
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v65);
      // pto: %kv_row_sum_tail_inline1973__ssa_v0
      uint64_t v284 = (uint64_t) v41;
      TASSIGN(v283, v284);
      // pto: %kv_sq_sum_tail_inline1970__ssa_v3
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %kv_sq_sum_tail_inline1970__ssa_v3
      uint64_t v286 = (uint64_t) v39;
      TASSIGN(v285, v286);
      pipe_barrier(PIPE_V);
      TADD(v285, v258, v283);
      // pto: %31
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %31
      uint64_t v288 = (uint64_t) v38;
      TASSIGN(v287, v288);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TMUL(v287, v270, v270);
      // pto: %32
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v33);
      // pto: %32
      uint64_t v290 = (uint64_t) v37;
      TASSIGN(v289, v290);
      pipe_barrier(PIPE_V);
      TROWSUM(v289, v287, v256);
      // pto: %33
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v291 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v65);
      // pto: %33
      uint64_t v292 = (uint64_t) v37;
      TASSIGN(v291, v292);
      // pto: %34
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
      // pto: %34
      uint64_t v294 = (uint64_t) v36;
      TASSIGN(v293, v294);
      pipe_barrier(PIPE_V);
      TADD(v293, v285, v291);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID1);
    // pto: %t__tmp_v166
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %t__tmp_v166
    uint64_t v296 = (uint64_t) v40;
    TASSIGN(v295, v296);
    pipe_barrier(PIPE_V);
    TMULS(v295, v258, v28);
    // pto: %t__tmp_v167
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %t__tmp_v167
    uint64_t v298 = (uint64_t) v40;
    TASSIGN(v297, v298);
    pipe_barrier(PIPE_V);
    TADDS(v297, v295, v27);
    // pto: %t__tmp_v168
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v299 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %t__tmp_v168
    uint64_t v300 = (uint64_t) v40;
    TASSIGN(v299, v300);
    pipe_barrier(PIPE_V);
    TSQRT(v299, v297);
    // pto: %kv_inv_rms_tail_inline1958__ssa_v0
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v301 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %kv_inv_rms_tail_inline1958__ssa_v0
    uint64_t v302 = (uint64_t) v35;
    TASSIGN(v301, v302);
    pipe_barrier(PIPE_V);
    TRECIP(v301, v299);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    // pto: %kv_inv_rms_t_tail_inline1947__ssa_v0
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v303 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v33);
    // pto: %kv_inv_rms_t_tail_inline1947__ssa_v0
    uint64_t v304 = (uint64_t) v35;
    TASSIGN(v303, v304);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t i305 = v40; i305 < v26; i305 += v29) {
      // pto: %kv_chunk_tail_inline1939__ssa_v1
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %kv_chunk_tail_inline1939__ssa_v1
      uint64_t v307 = (uint64_t) v40;
      TASSIGN(v306, v307);
      // pto: %100
      int64_t v308 = v63 < v40 ? v40 : v63;
      // pto: %101
      int64_t v309 = i305 < v40 ? v40 : i305;
      // pto: %102
      __gm__ float* v310 = PTOAS__GLOBAL_TENSOR_DATA(v47);
      // pto: %102
      int64_t v311 = v65 * v34;
      // pto: %102
      int64_t v312 = v33 * v311;
      // pto: %102
      pto::Shape<1, 1, 1, -1, 64> v313 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
      // pto: %102
      pto::Stride<-1, -1, -1, -1, -1> v314 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v312, v312, v311, v34, v33);
      // pto: %102
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v315 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v310 + ((v40 + v308 * v34) + v309 * v33), v313, v314);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      TLOAD(v306, v315);
      // pto: %gamma_kv_input_tail_inline1960__ssa_v0
      Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gamma_kv_input_tail_inline1960__ssa_v0
      uint64_t v317 = (uint64_t) v41;
      TASSIGN(v316, v317);
      // pto: %104
      pto::Shape<1, 1, 1, 1, 64> v318 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %104
      pto::Stride<64, 64, 64, 64, 1> v319 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %104
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v320 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v40 + v309), v318, v319);
      TLOAD(v316, v320);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %35
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %35
      uint64_t v322 = (uint64_t) v39;
      TASSIGN(v321, v322);
      // pto: %106
      int64_t v323 = (int64_t) ((uint64_t) i305 + (uint64_t) v32);
      // pto: %107
      int64_t v324 = v323 < v40 ? v40 : v323;
      // pto: %108
      __gm__ float* v325 = PTOAS__GLOBAL_TENSOR_DATA(v47);
      // pto: %108
      int64_t v326 = v65 * v34;
      // pto: %108
      int64_t v327 = v33 * v326;
      // pto: %108
      pto::Shape<1, 1, 1, -1, 64> v328 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
      // pto: %108
      pto::Stride<-1, -1, -1, -1, -1> v329 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v327, v327, v326, v34, v33);
      // pto: %108
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v330 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v325 + ((v40 + v308 * v34) + v324 * v33), v328, v329);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      TLOAD(v321, v330);
      // pto: %36
      Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %36
      uint64_t v332 = (uint64_t) v37;
      TASSIGN(v331, v332);
      // pto: %111
      pto::Shape<1, 1, 1, 1, 64> v333 = pto::Shape<1, 1, 1, 1, 64>();
      // pto: %111
      pto::Stride<64, 64, 64, 64, 1> v334 = pto::Stride<64, 64, 64, 64, 1>();
      // pto: %111
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v335 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v40 + v324), v333, v334);
      TLOAD(v331, v335);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %gamma_kv_cast_tail_inline1905__ssa_v0
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gamma_kv_cast_tail_inline1905__ssa_v0
      uint64_t v337 = (uint64_t) v38;
      TASSIGN(v336, v337);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(v336, v316, v19, v18);
      // pto: %gamma_kv_chunk_tail_inline1919__ssa_v0
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %gamma_kv_chunk_tail_inline1919__ssa_v0
      uint64_t v339 = (uint64_t) v38;
      TASSIGN(v338, v339);
      // pto: %t__tmp_v169
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %t__tmp_v169
      uint64_t v341 = (uint64_t) v40;
      TASSIGN(v340, v341);
      TROWEXPANDMUL(v340, v306, v303);
      // pto: %kv_normed_tail_inline1904__ssa_v0
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %kv_normed_tail_inline1904__ssa_v0
      uint64_t v343 = (uint64_t) v40;
      TASSIGN(v342, v343);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v342, v340, v338);
      // pto: %kv_normed_bf16_tail_inline1903__ssa_v0
      Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %kv_normed_bf16_tail_inline1903__ssa_v0
      uint64_t v345 = (uint64_t) v40;
      TASSIGN(v344, v345);
      pipe_barrier(PIPE_V);
      TCVT(v344, v342, v17, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      v344.SetValidShape(v65, v32);
      // pto: %112
      int64_t v346 = v66 < v40 ? v40 : v66;
      // pto: %kv_view_inline1909__ssa_v0_pview
      __gm__ bfloat16_t* v347 = PTOAS__GLOBAL_TENSOR_DATA(v52);
      // pto: %kv_view_inline1909__ssa_v0_pview
      int64_t v348 = v65 * v34;
      // pto: %kv_view_inline1909__ssa_v0_pview
      int64_t v349 = v33 * v348;
      // pto: %kv_view_inline1909__ssa_v0_pview
      pto::Shape<1, 1, 1, -1, 64> v350 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
      // pto: %kv_view_inline1909__ssa_v0_pview
      pto::Stride<-1, -1, -1, -1, -1> v351 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v349, v349, v348, v34, v33);
      // pto: %kv_view_inline1909__ssa_v0_pview
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v352 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v347 + ((v40 + v346 * v34) + v309 * v33), v350, v351);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v352, v344);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %37
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %37
      uint64_t v354 = (uint64_t) v36;
      TASSIGN(v353, v354);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TCVT(v353, v331, v19, v18);
      // pto: %38
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
      // pto: %38
      uint64_t v356 = (uint64_t) v36;
      TASSIGN(v355, v356);
      // pto: %39
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %39
      uint64_t v358 = (uint64_t) v39;
      TASSIGN(v357, v358);
      TROWEXPANDMUL(v357, v321, v303);
      // pto: %40
      Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %40
      uint64_t v360 = (uint64_t) v39;
      TASSIGN(v359, v360);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v359, v357, v355);
      // pto: %41
      Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v361 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
      // pto: %41
      uint64_t v362 = (uint64_t) v39;
      TASSIGN(v361, v362);
      pipe_barrier(PIPE_V);
      TCVT(v361, v359, v17, v18);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      v361.SetValidShape(v65, v32);
      // pto: %118
      __gm__ bfloat16_t* v363 = PTOAS__GLOBAL_TENSOR_DATA(v52);
      // pto: %118
      int64_t v364 = v65 * v34;
      // pto: %118
      int64_t v365 = v33 * v364;
      // pto: %118
      pto::Shape<1, 1, 1, -1, 64> v366 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
      // pto: %118
      pto::Stride<-1, -1, -1, -1, -1> v367 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v365, v365, v364, v34, v33);
      // pto: %118
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v368 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v363 + ((v40 + v346 * v34) + v324 * v33), v366, v367);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v368, v361);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
    }
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    // pto: %43
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %43
    uint64_t v370 = (uint64_t) v40;
    TASSIGN(v369, v370);
    // pto: %119
    int64_t v371 = v63 < v40 ? v40 : v63;
    // pto: %120
    __gm__ float* v372 = PTOAS__GLOBAL_TENSOR_DATA(v47);
    // pto: %120
    int64_t v373 = v65 * v34;
    // pto: %120
    int64_t v374 = v33 * v373;
    // pto: %120
    pto::Shape<1, 1, 1, -1, 64> v375 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
    // pto: %120
    pto::Stride<-1, -1, -1, -1, -1> v376 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v374, v374, v373, v34, v33);
    // pto: %120
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v377 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v372 + ((v40 + v371 * v34) + v26 * v33), v375, v376);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
    TLOAD(v369, v377);
    // pto: %44
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %44
    uint64_t v379 = (uint64_t) v38;
    TASSIGN(v378, v379);
    // pto: %121
    pto::Shape<1, 1, 1, 1, 64> v380 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %121
    pto::Stride<64, 64, 64, 64, 1> v381 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %121
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v382 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + v16, v380, v381);
    TLOAD(v378, v382);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %45
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %45
    uint64_t v384 = (uint64_t) v39;
    TASSIGN(v383, v384);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TCVT(v383, v378, v19, v18);
    // pto: %46
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %46
    uint64_t v386 = (uint64_t) v39;
    TASSIGN(v385, v386);
    // pto: %47
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %47
    uint64_t v388 = (uint64_t) v40;
    TASSIGN(v387, v388);
    TROWEXPANDMUL(v387, v369, v303);
    // pto: %48
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %48
    uint64_t v390 = (uint64_t) v40;
    TASSIGN(v389, v390);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v389, v387, v385);
    // pto: %49
    Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v391 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %49
    uint64_t v392 = (uint64_t) v40;
    TASSIGN(v391, v392);
    pipe_barrier(PIPE_V);
    TCVT(v391, v389, v17, v18);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    v391.SetValidShape(v65, v32);
    // pto: %123
    int64_t v393 = v66 < v40 ? v40 : v66;
    // pto: %124
    __gm__ bfloat16_t* v394 = PTOAS__GLOBAL_TENSOR_DATA(v52);
    // pto: %124
    int64_t v395 = v65 * v34;
    // pto: %124
    int64_t v396 = v33 * v395;
    // pto: %124
    pto::Shape<1, 1, 1, -1, 64> v397 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
    // pto: %124
    pto::Stride<-1, -1, -1, -1, -1> v398 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v396, v396, v395, v34, v33);
    // pto: %124
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v399 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v394 + ((v40 + v393 * v34) + v26 * v33), v397, v398);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    TSTORE(v399, v391);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    // pto: %gamma_rope_input_tail_inline1901__ssa_v0
    Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v400 = Tile<TileType::Vec, bfloat16_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %gamma_rope_input_tail_inline1901__ssa_v0
    uint64_t v401 = (uint64_t) v40;
    TASSIGN(v400, v401);
    // pto: %125
    pto::Shape<1, 1, 1, 1, 64> v402 = pto::Shape<1, 1, 1, 1, 64>();
    // pto: %125
    pto::Stride<64, 64, 64, 64, 1> v403 = pto::Stride<64, 64, 64, 64, 1>();
    // pto: %125
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v404 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + v15, v402, v403);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    TLOAD(v400, v404);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %gamma_rope_cast_tail_inline1900__ssa_v0
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %gamma_rope_cast_tail_inline1900__ssa_v0
    uint64_t v406 = (uint64_t) v39;
    TASSIGN(v405, v406);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v405, v400, v19, v18);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    // pto: %gamma_rope_tail_inline1899__ssa_v0
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %gamma_rope_tail_inline1899__ssa_v0
    uint64_t v408 = (uint64_t) v39;
    TASSIGN(v407, v408);
    // pto: %kv_rope_chunk_tail_inline1907__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %kv_rope_chunk_tail_inline1907__ssa_v0
    uint64_t v410 = (uint64_t) v40;
    TASSIGN(v409, v410);
    // pto: %127
    __gm__ float* v411 = PTOAS__GLOBAL_TENSOR_DATA(v47);
    // pto: %127
    int64_t v412 = v65 * v34;
    // pto: %127
    int64_t v413 = v33 * v412;
    // pto: %127
    pto::Shape<1, 1, 1, -1, 64> v414 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
    // pto: %127
    pto::Stride<-1, -1, -1, -1, -1> v415 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v413, v413, v412, v34, v33);
    // pto: %127
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v416 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v411 + ((v40 + v371 * v34) + v25 * v33), v414, v415);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    TLOAD(v409, v416);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %t__tmp_v170
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v417 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %t__tmp_v170
    uint64_t v418 = (uint64_t) v40;
    TASSIGN(v417, v418);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPANDMUL(v417, v409, v303);
    // pto: %kv_rope_norm_tail_inline1898__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %kv_rope_norm_tail_inline1898__ssa_v0
    uint64_t v420 = (uint64_t) v40;
    TASSIGN(v419, v420);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v419, v417, v407);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    // pto: %kv_cos_il_tail_inline1929__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %kv_cos_il_tail_inline1929__ssa_v0
    uint64_t v422 = (uint64_t) v39;
    TASSIGN(v421, v422);
    // pto: %129
    __gm__ float* v423 = PTOAS__GLOBAL_TENSOR_DATA(v57);
    // pto: %129
    int64_t v424 = v65 * v32;
    // pto: %129
    int64_t v425 = v33 * v424;
    // pto: %129
    pto::Shape<1, 1, 1, -1, 64> v426 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
    // pto: %129
    pto::Stride<-1, -1, -1, -1, -1> v427 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v425, v425, v424, v32, v33);
    // pto: %129
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v428 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v423 + ((v40 + v393 * v32) + v40 * v33), v426, v427);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    TLOAD(v421, v428);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %kv_sin_signed_tail_inline1917__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %kv_sin_signed_tail_inline1917__ssa_v0
    uint64_t v430 = (uint64_t) v38;
    TASSIGN(v429, v430);
    // pto: %131
    __gm__ float* v431 = PTOAS__GLOBAL_TENSOR_DATA(v62);
    // pto: %131
    int64_t v432 = v65 * v32;
    // pto: %131
    int64_t v433 = v33 * v432;
    // pto: %131
    pto::Shape<1, 1, 1, -1, 64> v434 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
    // pto: %131
    pto::Stride<-1, -1, -1, -1, -1> v435 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v433, v433, v432, v32, v33);
    // pto: %131
    GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v436 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v431 + ((v40 + v393 * v32) + v40 * v33), v434, v435);
    TLOAD(v429, v436);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    // pto: %t__tmp_v171
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v437 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %t__tmp_v171
    uint64_t v438 = (uint64_t) v36;
    TASSIGN(v437, v438);
    TEXPANDS(v437, v24);
    // pto: %t__tmp_v172
    Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v439 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %t__tmp_v172
    uint64_t v440 = (uint64_t) v41;
    TASSIGN(v439, v440);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
    TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v439, v23);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    // pto: %t__tmp_v173
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v441 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v32);
    // pto: %t__tmp_v173
    uint64_t v442 = (uint64_t) v41;
    TASSIGN(v441, v442);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TCVT(v441, v439, v19, v18);
    // pto: %kv_col_inline1897__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_col_inline1897__ssa_v0
    uint64_t v444 = (uint64_t) v36;
    TASSIGN(v443, v444);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v443, v437, v441);
    // pto: %t__tmp_v174
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %t__tmp_v174
    uint64_t v446 = (uint64_t) v41;
    TASSIGN(v445, v446);
    pipe_barrier(PIPE_V);
    TMULS(v445, v443, v22);
    // pto: %t__tmp_v175
    Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %t__tmp_v175
    uint64_t v448 = (uint64_t) v41;
    TASSIGN(v447, v448);
    pipe_barrier(PIPE_V);
    TCVT(v447, v445, v14, v18);
    // pto: %kv_dup_f_inline1952__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v449 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_dup_f_inline1952__ssa_v0
    uint64_t v450 = (uint64_t) v41;
    TASSIGN(v449, v450);
    pipe_barrier(PIPE_V);
    TCVT(v449, v447, v19, v18);
    // pto: %t__tmp_v176
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v451 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %t__tmp_v176
    uint64_t v452 = (uint64_t) v41;
    TASSIGN(v451, v452);
    pipe_barrier(PIPE_V);
    TMULS(v451, v449, v21);
    // pto: %kv_lane_inline1896__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v453 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_lane_inline1896__ssa_v0
    uint64_t v454 = (uint64_t) v41;
    TASSIGN(v453, v454);
    pipe_barrier(PIPE_V);
    TSUB(v453, v443, v451);
    // pto: %t__tmp_v177
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %t__tmp_v177
    uint64_t v456 = (uint64_t) v36;
    TASSIGN(v455, v456);
    pipe_barrier(PIPE_V);
    TADDS(v455, v443, v24);
    // pto: %t__tmp_v178
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %t__tmp_v178
    uint64_t v458 = (uint64_t) v41;
    TASSIGN(v457, v458);
    TMULS(v457, v453, v21);
    // pto: %kv_swap_f_inline1895__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v459 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_swap_f_inline1895__ssa_v0
    uint64_t v460 = (uint64_t) v36;
    TASSIGN(v459, v460);
    pipe_barrier(PIPE_V);
    TSUB(v459, v455, v457);
    set_flag(PIPE_V, PIPE_S, EVENT_ID2);
    // pto: %t__tmp_v179
    Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v461 = Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %t__tmp_v179
    uint64_t v462 = (uint64_t) v41;
    TASSIGN(v461, v462);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
    TCI<Tile<TileType::Vec, int32_t, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v461, v23);
    set_flag(PIPE_S, PIPE_V, EVENT_ID1);
    // pto: %t__tmp_v180
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v463 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %t__tmp_v180
    uint64_t v464 = (uint64_t) v41;
    TASSIGN(v463, v464);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    TCVT(v463, v461, v19, v18);
    // pto: %kv_row_seed_inline1966__ssa_v0
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v31);
    // pto: %kv_row_seed_inline1966__ssa_v0
    uint64_t v466 = (uint64_t) v37;
    TASSIGN(v465, v466);
    pipe_barrier(PIPE_V);
    TMULS(v465, v463, v20);
    // pto: %t__tmp_v181
    Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v467 = Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v31);
    // pto: %t__tmp_v181
    uint64_t v468 = (uint64_t) v41;
    TASSIGN(v467, v468);
    pipe_barrier(PIPE_V);
    TEXPANDS(v467, v24);
    // pto: %kv_row_grid_inline1893__ssa_v0
    Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v31);
    // pto: %kv_row_grid_inline1893__ssa_v0
    uint64_t v470 = (uint64_t) v41;
    TASSIGN(v469, v470);
    pipe_barrier(PIPE_V);
    TCOLEXPANDMUL(v469, v467, v465);
    // pto: %transpose_tmp
    Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v471 = Tile<TileType::Vec, float, 64, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v31);
    // pto: %transpose_tmp
    uint64_t v472 = (uint64_t) v37;
    TASSIGN(v471, v472);
    // pto: %kv_row_offset_inline1892__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v473 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_row_offset_inline1892__ssa_v0
    uint64_t v474 = (uint64_t) v35;
    TASSIGN(v473, v474);
    pipe_barrier(PIPE_V);
    TTRANS(v473, v469, v471);
    // pto: %t__tmp_v182
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v475 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %t__tmp_v182
    uint64_t v476 = (uint64_t) v36;
    TASSIGN(v475, v476);
    pipe_barrier(PIPE_V);
    TADD(v475, v459, v473);
    // pto: %kv_swap_idx_tail_inline1891__ssa_v0
    Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_swap_idx_tail_inline1891__ssa_v0
    uint64_t v478 = (uint64_t) v36;
    TASSIGN(v477, v478);
    pipe_barrier(PIPE_V);
    TCVT(v477, v475, v19, v18);
    // pto: %kv_gather_tmp_inline1890__ssa_v0
    Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v479 = Tile<TileType::Vec, int32_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_gather_tmp_inline1890__ssa_v0
    uint64_t v480 = (uint64_t) v41;
    TASSIGN(v479, v480);
    // pto: %kv_swapped_tail_inline1894__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v481 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %kv_swapped_tail_inline1894__ssa_v0
    uint64_t v482 = (uint64_t) v37;
    TASSIGN(v481, v482);
    pipe_barrier(PIPE_V);
    TGATHER(v481, v419, v477, v479);
    // pto: %t__tmp_v183
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v483 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %t__tmp_v183
    uint64_t v484 = (uint64_t) v40;
    TASSIGN(v483, v484);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(v483, v419, v421);
    // pto: %t__tmp_v184
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v485 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v31, v32);
    // pto: %t__tmp_v184
    uint64_t v486 = (uint64_t) v39;
    TASSIGN(v485, v486);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v485, v481, v429);
    // pto: %kv_rope_rot_tail_inline1927__ssa_v0
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v487 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %kv_rope_rot_tail_inline1927__ssa_v0
    uint64_t v488 = (uint64_t) v40;
    TASSIGN(v487, v488);
    pipe_barrier(PIPE_V);
    TADD(v487, v483, v485);
    // pto: %kv_rope_i16_tail_inline1889__ssa_v0
    Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v489 = Tile<TileType::Vec, bfloat16_t, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v65, v32);
    // pto: %kv_rope_i16_tail_inline1889__ssa_v0
    uint64_t v490 = (uint64_t) v40;
    TASSIGN(v489, v490);
    pipe_barrier(PIPE_V);
    TCVT(v489, v487, v17, v18);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    v489.SetValidShape(v65, v32);
    // pto: %133
    __gm__ bfloat16_t* v491 = PTOAS__GLOBAL_TENSOR_DATA(v52);
    // pto: %133
    int64_t v492 = v65 * v34;
    // pto: %133
    int64_t v493 = v33 * v492;
    // pto: %133
    pto::Shape<1, 1, 1, -1, 64> v494 = pto::Shape<1, 1, 1, -1, 64>(v33, v33, v33, v65, v32);
    // pto: %133
    pto::Stride<-1, -1, -1, -1, -1> v495 = pto::Stride<-1, -1, -1, -1, -1>(v33 * v493, v493, v492, v34, v33);
    // pto: %133
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v496 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 64>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v491 + ((v40 + v393 * v34) + v25 * v33), v494, v495);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    TSTORE(v496, v489);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}