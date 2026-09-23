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

AICORE void hc_pre_rms(__gm__ float* v1, __gm__ float* v2, int64_t v3, int64_t v4, int64_t v5, int32_t v6, int32_t v7) {
  const float v8 = 9.99999997E-7f;
  const float v9 = 6.10351563E-5f;
  const int64_t v10 = 1536;
  const int64_t v11 = 1024;
  const int64_t v12 = 512;
  const int64_t v13 = 4;
  const float v14 = 0.0f;
  const int64_t v15 = 8;
  const int64_t v16 = 1;
  const int64_t v17 = 16384;
  const int64_t v18 = 114880;
  const int64_t v19 = 98496;
  const int64_t v20 = 82112;
  const int64_t v21 = 82080;
  const int64_t v22 = 65664;
  const int64_t v23 = 49280;
  const int64_t v24 = 32896;
  const int64_t v25 = 32864;
  const int64_t v26 = 32800;
  const int64_t v27 = 16416;
  const int64_t v28 = 32;
  const int64_t v29 = 0;
  const int64_t v30 = 131296;
  const int64_t v31 = 114912;
  const int64_t v32 = 65696;
  const int64_t v33 = 32832;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %x_flat_inline1497__ssa_v0_view
  int64_t v34 = v4 * v17;
  // pto: %x_flat_inline1497__ssa_v0_view
  int64_t v35 = v16 * v34;
  // pto: %x_flat_inline1497__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v36 = pto::Shape<1, 1, 1, -1, -1>(v16, v16, v16, v4, v17);
  // pto: %x_flat_inline1497__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v37 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v35, v35, v34, v17, v16);
  // pto: %x_flat_inline1497__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v38 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v36, v37);
  // pto: %inv_rms_inline1463__ssa_v0_view
  int64_t v39 = v5 * v16;
  // pto: %inv_rms_inline1463__ssa_v0_view
  int64_t v40 = v16 * v39;
  // pto: %inv_rms_inline1463__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v41 = pto::Shape<1, 1, 1, -1, -1>(v16, v16, v16, v5, v16);
  // pto: %inv_rms_inline1463__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v42 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v40, v40, v39, v16, v5);
  // pto: %inv_rms_inline1463__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v41, v42);
  // pto: %sq_sum_inline1490__phi_v5
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %sq_sum_inline1490__phi_v5
  uint64_t v45 = (uint64_t) v30;
  TASSIGN(v44, v45);
  // pto: %61
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %61
  uint64_t v47 = (uint64_t) v26;
  TASSIGN(v46, v47);
  // pto: %69
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v48 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %69
  uint64_t v49 = (uint64_t) v22;
  TASSIGN(v48, v49);
  // pto: %77
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v50 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %77
  uint64_t v51 = (uint64_t) v33;
  TASSIGN(v50, v51);
  // pto: %t_inline1518__ssa_v0, %44
  int64_t v52 = (int64_t) ((uint64_t) ((int64_t) v6) * (uint64_t) v15);
  // pto: %45
  int64_t v53 = (int64_t) ((uint64_t) v4 - (uint64_t) v52);
  // pto: %46
  int64_t v54 = v53 < v15 ? v53 : v15;
  // pto: %sq_sum_inline1490__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %sq_sum_inline1490__tile
  uint64_t v56 = (uint64_t) v33;
  TASSIGN(v55, v56);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  TEXPANDS(v55, v14);
  for (int64_t i57 = v29; i57 < v28; i57 += v13) {
    // pto: %47
    int64_t v58 = (int64_t) ((uint64_t) i57 * (uint64_t) v12);
    // pto: %49
    int64_t v59 = (int64_t) ((uint64_t) v58 + (uint64_t) v12);
    // pto: %51
    int64_t v60 = (int64_t) ((uint64_t) v58 + (uint64_t) v11);
    // pto: %53
    int64_t v61 = (int64_t) ((uint64_t) v58 + (uint64_t) v10);
    // pto: %54
    bool v62 = v54 == v15;
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    if (v62) {
      // pto: %x_chunk_full_inline1502__tile
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %x_chunk_full_inline1502__tile
      uint64_t v64 = (uint64_t) v32;
      TASSIGN(v63, v64);
      // pto: %x_flat_inline1497__ssa_v0_pview
      pto::Shape<1, 1, 1, 8, 512> v65 = pto::Shape<1, 1, 1, 8, 512>();
      // pto: %x_flat_inline1497__ssa_v0_pview
      pto::Stride<131072, 131072, 131072, 16384, 1> v66 = pto::Stride<131072, 131072, 131072, 16384, 1>();
      // pto: %55, %x_flat_inline1497__ssa_v0_pview, %56
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v67 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v1 + ((v29 + (v52 < v29 ? v29 : v52) * v17) + (v58 < v29 ? v29 : v58)), v65, v66);
      TLOAD(v63, v67);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %x_sq_full_inline1510__tile
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %x_sq_full_inline1510__tile
      uint64_t v69 = (uint64_t) v32;
      TASSIGN(v68, v69);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TMUL(v68, v63, v63);
      // pto: %tmp_tile
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %tmp_tile
      uint64_t v71 = (uint64_t) v31;
      TASSIGN(v70, v71);
      // pto: %t__tile
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
      // pto: %t__tile
      uint64_t v73 = (uint64_t) v30;
      TASSIGN(v72, v73);
      pipe_barrier(PIPE_V);
      TROWSUM(v72, v68, v70);
      // pto: %x_sq_row_full_inline1511__tile
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %x_sq_row_full_inline1511__tile
      uint64_t v75 = (uint64_t) v30;
      TASSIGN(v74, v75);
      // pto: %0
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %0
      uint64_t v77 = (uint64_t) v30;
      TASSIGN(v76, v77);
      pipe_barrier(PIPE_V);
      TADD(v76, v55, v74);
    } else {
      // pto: %x_chunk_tail_inline1513__tile
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v12);
      // pto: %x_chunk_tail_inline1513__tile
      uint64_t v79 = (uint64_t) v32;
      TASSIGN(v78, v79);
      // pto: %59
      __gm__ float* v80 = PTOAS__GLOBAL_TENSOR_DATA(v38);
      // pto: %59
      int64_t v81 = v54 * v17;
      // pto: %59
      int64_t v82 = v16 * v81;
      // pto: %59
      pto::Shape<1, 1, 1, -1, 512> v83 = pto::Shape<1, 1, 1, -1, 512>(v16, v16, v16, v54, v12);
      // pto: %59
      pto::Stride<-1, -1, -1, -1, -1> v84 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v82, v82, v81, v17, v16);
      // pto: %57, %59, %58
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v85 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v80 + ((v29 + (v52 < v29 ? v29 : v52) * v17) + (v58 < v29 ? v29 : v58) * v16), v83, v84);
      TLOAD(v78, v85);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %x_sq_tail_inline1503__tile
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v12);
      // pto: %x_sq_tail_inline1503__tile
      uint64_t v87 = (uint64_t) v32;
      TASSIGN(v86, v87);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TMUL(v86, v78, v78);
      // pto: %1
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %1
      uint64_t v89 = (uint64_t) v31;
      TASSIGN(v88, v89);
      // pto: %2
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v16);
      // pto: %2
      uint64_t v91 = (uint64_t) v29;
      TASSIGN(v90, v91);
      pipe_barrier(PIPE_V);
      TROWSUM(v90, v86, v88);
      // pto: %x_sq_row_tail_inline1537__tile
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v92 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v54);
      // pto: %x_sq_row_tail_inline1537__tile
      uint64_t v93 = (uint64_t) v29;
      TASSIGN(v92, v93);
      // pto: %3
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %3
      uint64_t v95 = (uint64_t) v32;
      TASSIGN(v94, v95);
      pipe_barrier(PIPE_V);
      TADD(v94, v55, v92);
      // pto: %sq_sum_inline1490__tile_mv
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %sq_sum_inline1490__tile_mv
      uint64_t v97 = (uint64_t) v30;
      TASSIGN(v96, v97);
      pipe_barrier(PIPE_V);
      TMOV(v96, v94);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    if (v62) {
      // pto: %4
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v98 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %4
      uint64_t v99 = (uint64_t) v28;
      TASSIGN(v98, v99);
      // pto: %64
      pto::Shape<1, 1, 1, 8, 512> v100 = pto::Shape<1, 1, 1, 8, 512>();
      // pto: %64
      pto::Stride<131072, 131072, 131072, 16384, 1> v101 = pto::Stride<131072, 131072, 131072, 16384, 1>();
      // pto: %62, %64, %63
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v102 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v1 + ((v29 + (v52 < v29 ? v29 : v52) * v17) + (v59 < v29 ? v29 : v59)), v100, v101);
      TLOAD(v98, v102);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %5
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %5
      uint64_t v104 = (uint64_t) v28;
      TASSIGN(v103, v104);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TMUL(v103, v98, v98);
      // pto: %6
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %6
      uint64_t v106 = (uint64_t) v27;
      TASSIGN(v105, v106);
      // pto: %7
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
      // pto: %7
      uint64_t v108 = (uint64_t) v26;
      TASSIGN(v107, v108);
      pipe_barrier(PIPE_V);
      TROWSUM(v107, v103, v105);
      // pto: %8
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %8
      uint64_t v110 = (uint64_t) v26;
      TASSIGN(v109, v110);
      // pto: %9
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %9
      uint64_t v112 = (uint64_t) v26;
      TASSIGN(v111, v112);
      pipe_barrier(PIPE_V);
      TADD(v111, v44, v109);
    } else {
      // pto: %10
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v12);
      // pto: %10
      uint64_t v114 = (uint64_t) v28;
      TASSIGN(v113, v114);
      // pto: %67
      __gm__ float* v115 = PTOAS__GLOBAL_TENSOR_DATA(v38);
      // pto: %67
      int64_t v116 = v54 * v17;
      // pto: %67
      int64_t v117 = v16 * v116;
      // pto: %67
      pto::Shape<1, 1, 1, -1, 512> v118 = pto::Shape<1, 1, 1, -1, 512>(v16, v16, v16, v54, v12);
      // pto: %67
      pto::Stride<-1, -1, -1, -1, -1> v119 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v117, v117, v116, v17, v16);
      // pto: %65, %67, %66
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v120 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v115 + ((v29 + (v52 < v29 ? v29 : v52) * v17) + (v59 < v29 ? v29 : v59) * v16), v118, v119);
      TLOAD(v113, v120);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %11
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v12);
      // pto: %11
      uint64_t v122 = (uint64_t) v28;
      TASSIGN(v121, v122);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TMUL(v121, v113, v113);
      // pto: %12
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %12
      uint64_t v124 = (uint64_t) v27;
      TASSIGN(v123, v124);
      // pto: %13
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v16);
      // pto: %13
      uint64_t v126 = (uint64_t) v25;
      TASSIGN(v125, v126);
      pipe_barrier(PIPE_V);
      TROWSUM(v125, v121, v123);
      // pto: %14
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v54);
      // pto: %14
      uint64_t v128 = (uint64_t) v25;
      TASSIGN(v127, v128);
      // pto: %15
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %15
      uint64_t v130 = (uint64_t) v28;
      TASSIGN(v129, v130);
      pipe_barrier(PIPE_V);
      TADD(v129, v44, v127);
      // pto: %16
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %16
      uint64_t v132 = (uint64_t) v26;
      TASSIGN(v131, v132);
      pipe_barrier(PIPE_V);
      TMOV(v131, v129);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    if (v62) {
      // pto: %17
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %17
      uint64_t v134 = (uint64_t) v24;
      TASSIGN(v133, v134);
      // pto: %72
      pto::Shape<1, 1, 1, 8, 512> v135 = pto::Shape<1, 1, 1, 8, 512>();
      // pto: %72
      pto::Stride<131072, 131072, 131072, 16384, 1> v136 = pto::Stride<131072, 131072, 131072, 16384, 1>();
      // pto: %70, %72, %71
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v137 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v1 + ((v29 + (v52 < v29 ? v29 : v52) * v17) + (v60 < v29 ? v29 : v60)), v135, v136);
      TLOAD(v133, v137);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %18
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v138 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %18
      uint64_t v139 = (uint64_t) v24;
      TASSIGN(v138, v139);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TMUL(v138, v133, v133);
      // pto: %19
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v140 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %19
      uint64_t v141 = (uint64_t) v23;
      TASSIGN(v140, v141);
      // pto: %20
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v142 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
      // pto: %20
      uint64_t v143 = (uint64_t) v22;
      TASSIGN(v142, v143);
      pipe_barrier(PIPE_V);
      TROWSUM(v142, v138, v140);
      // pto: %21
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v144 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %21
      uint64_t v145 = (uint64_t) v22;
      TASSIGN(v144, v145);
      // pto: %22
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %22
      uint64_t v147 = (uint64_t) v22;
      TASSIGN(v146, v147);
      pipe_barrier(PIPE_V);
      TADD(v146, v46, v144);
    } else {
      // pto: %23
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v148 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v12);
      // pto: %23
      uint64_t v149 = (uint64_t) v24;
      TASSIGN(v148, v149);
      // pto: %75
      __gm__ float* v150 = PTOAS__GLOBAL_TENSOR_DATA(v38);
      // pto: %75
      int64_t v151 = v54 * v17;
      // pto: %75
      int64_t v152 = v16 * v151;
      // pto: %75
      pto::Shape<1, 1, 1, -1, 512> v153 = pto::Shape<1, 1, 1, -1, 512>(v16, v16, v16, v54, v12);
      // pto: %75
      pto::Stride<-1, -1, -1, -1, -1> v154 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v152, v152, v151, v17, v16);
      // pto: %73, %75, %74
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v155 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v150 + ((v29 + (v52 < v29 ? v29 : v52) * v17) + (v60 < v29 ? v29 : v60) * v16), v153, v154);
      TLOAD(v148, v155);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %24
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v12);
      // pto: %24
      uint64_t v157 = (uint64_t) v24;
      TASSIGN(v156, v157);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TMUL(v156, v148, v148);
      // pto: %25
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v158 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %25
      uint64_t v159 = (uint64_t) v23;
      TASSIGN(v158, v159);
      // pto: %26
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v160 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v16);
      // pto: %26
      uint64_t v161 = (uint64_t) v21;
      TASSIGN(v160, v161);
      pipe_barrier(PIPE_V);
      TROWSUM(v160, v156, v158);
      // pto: %27
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v162 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v54);
      // pto: %27
      uint64_t v163 = (uint64_t) v21;
      TASSIGN(v162, v163);
      // pto: %28
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v164 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %28
      uint64_t v165 = (uint64_t) v24;
      TASSIGN(v164, v165);
      pipe_barrier(PIPE_V);
      TADD(v164, v46, v162);
      // pto: %29
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v166 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %29
      uint64_t v167 = (uint64_t) v22;
      TASSIGN(v166, v167);
      pipe_barrier(PIPE_V);
      TMOV(v166, v164);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
    if (v62) {
      // pto: %30
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v168 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %30
      uint64_t v169 = (uint64_t) v20;
      TASSIGN(v168, v169);
      // pto: %80
      pto::Shape<1, 1, 1, 8, 512> v170 = pto::Shape<1, 1, 1, 8, 512>();
      // pto: %80
      pto::Stride<131072, 131072, 131072, 16384, 1> v171 = pto::Stride<131072, 131072, 131072, 16384, 1>();
      // pto: %78, %80, %79
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND> v172 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 512>, pto::Stride<131072, 131072, 131072, 16384, 1>, pto::Layout::ND>(v1 + ((v29 + (v52 < v29 ? v29 : v52) * v17) + (v61 < v29 ? v29 : v61)), v170, v171);
      TLOAD(v168, v172);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %31
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %31
      uint64_t v174 = (uint64_t) v20;
      TASSIGN(v173, v174);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TMUL(v173, v168, v168);
      // pto: %32
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v175 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %32
      uint64_t v176 = (uint64_t) v19;
      TASSIGN(v175, v176);
      // pto: %33
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
      // pto: %33
      uint64_t v178 = (uint64_t) v18;
      TASSIGN(v177, v178);
      pipe_barrier(PIPE_V);
      TROWSUM(v177, v173, v175);
      // pto: %34
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v179 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %34
      uint64_t v180 = (uint64_t) v18;
      TASSIGN(v179, v180);
      // pto: %35
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v181 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %35
      uint64_t v182 = (uint64_t) v33;
      TASSIGN(v181, v182);
      pipe_barrier(PIPE_V);
      TADD(v181, v48, v179);
    } else {
      // pto: %36
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v12);
      // pto: %36
      uint64_t v184 = (uint64_t) v20;
      TASSIGN(v183, v184);
      // pto: %83
      __gm__ float* v185 = PTOAS__GLOBAL_TENSOR_DATA(v38);
      // pto: %83
      int64_t v186 = v54 * v17;
      // pto: %83
      int64_t v187 = v16 * v186;
      // pto: %83
      pto::Shape<1, 1, 1, -1, 512> v188 = pto::Shape<1, 1, 1, -1, 512>(v16, v16, v16, v54, v12);
      // pto: %83
      pto::Stride<-1, -1, -1, -1, -1> v189 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v187, v187, v186, v17, v16);
      // pto: %81, %83, %82
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v190 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v185 + ((v29 + (v52 < v29 ? v29 : v52) * v17) + (v61 < v29 ? v29 : v61) * v16), v188, v189);
      TLOAD(v183, v190);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %37
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v191 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v12);
      // pto: %37
      uint64_t v192 = (uint64_t) v20;
      TASSIGN(v191, v192);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TMUL(v191, v183, v183);
      // pto: %38
      Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %38
      uint64_t v194 = (uint64_t) v19;
      TASSIGN(v193, v194);
      // pto: %39
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v54, v16);
      // pto: %39
      uint64_t v196 = (uint64_t) v18;
      TASSIGN(v195, v196);
      pipe_barrier(PIPE_V);
      TROWSUM(v195, v191, v193);
      // pto: %40
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v197 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v54);
      // pto: %40
      uint64_t v198 = (uint64_t) v18;
      TASSIGN(v197, v198);
      // pto: %41
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
      // pto: %41
      uint64_t v200 = (uint64_t) v33;
      TASSIGN(v199, v200);
      pipe_barrier(PIPE_V);
      TADD(v199, v48, v197);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  }
  // pto: %42
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %42
  uint64_t v202 = (uint64_t) v32;
  TASSIGN(v201, v202);
  pipe_barrier(PIPE_V);
  TMULS(v201, v55, v9);
  // pto: %sq_mean_inline1514__tile
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %sq_mean_inline1514__tile
  uint64_t v204 = (uint64_t) v32;
  TASSIGN(v203, v204);
  pipe_barrier(PIPE_V);
  TADDS(v203, v201, v8);
  // pto: %rsqrt_tmp
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v205 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %rsqrt_tmp
  uint64_t v206 = (uint64_t) v31;
  TASSIGN(v205, v206);
  // pto: %43
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v16, v15);
  // pto: %43
  uint64_t v208 = (uint64_t) v28;
  TASSIGN(v207, v208);
  pipe_barrier(PIPE_V);
  TRSQRT(v207, v203, v205);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %inv_inline1481__tile
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v209 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v15, v16);
  // pto: %inv_inline1481__tile
  uint64_t v210 = (uint64_t) v28;
  TASSIGN(v209, v210);
  // pto: %inv_rms_inline1463__ssa_v0_pview
  __gm__ float* v211 = PTOAS__GLOBAL_TENSOR_DATA(v43);
  // pto: %inv_rms_inline1463__ssa_v0_pview
  int64_t v212 = v15 * v16;
  // pto: %inv_rms_inline1463__ssa_v0_pview
  int64_t v213 = v16 * v212;
  // pto: %inv_rms_inline1463__ssa_v0_pview
  pto::Shape<1, 1, 1, 8, 1> v214 = pto::Shape<1, 1, 1, 8, 1>(v16, v16, v16, v15, v16);
  // pto: %inv_rms_inline1463__ssa_v0_pview
  pto::Stride<-1, -1, -1, -1, -1> v215 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v213, v213, v212, v16, v5);
  // pto: %84, %inv_rms_inline1463__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v216 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v211 + ((v29 + (v52 < v29 ? v29 : v52) * v16) + v29 * v5), v214, v215);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v216, v209);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}