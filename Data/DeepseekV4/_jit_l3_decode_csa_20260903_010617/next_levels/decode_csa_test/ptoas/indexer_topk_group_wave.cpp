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

AICORE void indexer_topk_group_wave(__gm__ int32_t* v1, __gm__ int32_t* v2, __gm__ float* v3, __gm__ float* v4, int64_t v5, int64_t v6, int32_t v7, int32_t v8) {
  const int64_t v9 = 2048;
  const int64_t v10 = 4096;
  const int32_t v11 = 4096;
  const int32_t v12 = 1024;
  const int32_t v13 = 256;
  const int32_t v14 = 64;
  const int64_t v15 = 16384;
  const float v16 = -3.40282347E+38f;
  const int32_t v17 = 0;
  const int64_t v18 = 16;
  const int64_t v19 = 48;
  const int64_t v20 = 2;
  const int64_t v21 = 8192;
  const int64_t v22 = 8191;
  const int64_t v23 = 4;
  const int64_t v24 = 8;
  const int64_t v25 = 1024;
  const int64_t v26 = 262144;
  const int64_t v27 = 1;
  const int64_t v28 = 131072;
  const int64_t v29 = 135168;
  const int64_t v30 = 65536;
  const int64_t v31 = 0;
  pto::MrgSortExecutedNumList v32 = pto::MrgSortExecutedNumList{0, 0, 0, 0};
  const int64_t v33 = 4097;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %score_arena__ssa_v0_view
  int64_t v34 = v5 * v26;
  // pto: %score_arena__ssa_v0_view
  int64_t v35 = v27 * v34;
  // pto: %score_arena__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v36 = pto::Shape<1, 1, 1, -1, -1>(v27, v27, v27, v5, v26);
  // pto: %score_arena__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v37 = pto::Stride<-1, -1, -1, -1, -1>(v27 * v35, v35, v34, v26, v27);
  // pto: %score_arena__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v38 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v36, v37);
  // pto: %worker__ssa_v0
  int64_t v39 = (int64_t) v7;
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  // pto: %global_group_base__rv_v2
  int64_t v40;
  v40 = v31;
  for (int64_t i41 = v31; i41 < v5; i41 += v27) {
    // pto: %global_group_base__rv_v2
    int64_t v42 = v40;
    // pto: %position__tile
    int32_t v43 = (v1)[i41];
    // pto: %0, %t__tile
    int32_t v44 = (v2)[i41 / v24];
    // pto: %1, %2
    int64_t v45 = (int64_t) v44 / v23;
    // pto: %3, %4, %5
    int64_t v46 = (int64_t) ((uint64_t) ((int64_t) v43) + (uint64_t) v27) / v23;
    // pto: %6
    int64_t v47 = v45 < v46 ? v45 : v46;
    // pto: %7
    int64_t v48 = v47 < v26 ? v47 : v26;
    // pto: %8
    int64_t v49 = v48 < v31 ? v31 : v48;
    // pto: %9, %10
    int64_t v50 = (int64_t) ((uint64_t) v49 + (uint64_t) v22) / v21;
    // pto: %11, %12
    int64_t v51 = (int64_t) ((uint64_t) v50 + (uint64_t) v27) / v20;
    // pto: %14, %13, %15
    for (int64_t j52 = (int64_t) ((uint64_t) v39 + (uint64_t) (v42 % v19)) % v19; j52 < v51; j52 += v19) {
      // pto: %16
      int64_t v53 = (int64_t) ((uint64_t) j52 * (uint64_t) v20);
      // pto: %17
      int64_t v54 = (int64_t) ((uint64_t) v50 - (uint64_t) v53);
      // pto: %19, %20
      int64_t v55 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) i41 * (uint64_t) v18)) + (uint64_t) j52);
      // pto: %18, %21
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      if ((v54 < v20 ? v54 : v20) == v27) {
        int64_t v56 = (int64_t) ((uint64_t) j52 * (uint64_t) v15);
        // pto: %23
        int64_t v57 = (int64_t) ((uint64_t) v49 - (uint64_t) v56);
        // pto: %24
        int64_t v58 = v57 < v21 ? v57 : v21;
        // pto: %25
        int32_t v59 = (int32_t) v56;
        // pto: %t__tmp_v1
        Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v60 = Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v1
        uint64_t v61 = (uint64_t) v31;
        TASSIGN(v60, v61);
        TCI<Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v60, v17);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        // pto: %leaf_indices_inline125__ssa_v0
        Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62 = Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %leaf_indices_inline125__ssa_v0
        uint64_t v63 = (uint64_t) v30;
        TASSIGN(v62, v63);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        TADDS(v62, v60, v59);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %leaf_scores_raw_inline124__ssa_v0
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v64 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v58);
        // pto: %leaf_scores_raw_inline124__ssa_v0
        uint64_t v65 = (uint64_t) v31;
        TASSIGN(v64, v65);
        // pto: %score_arena__ssa_v0_pview
        __gm__ float* v66 = PTOAS__GLOBAL_TENSOR_DATA(v38);
        // pto: %score_arena__ssa_v0_pview
        int64_t v67 = v27 * v26;
        // pto: %score_arena__ssa_v0_pview
        int64_t v68 = v27 * v67;
        // pto: %score_arena__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, -1> v69 = pto::Shape<1, 1, 1, 1, -1>(v27, v27, v27, v27, v58);
        // pto: %score_arena__ssa_v0_pview
        pto::Stride<-1, -1, -1, -1, -1> v70 = pto::Stride<-1, -1, -1, -1, -1>(v27 * v68, v68, v67, v26, v27);
        // pto: %26, %score_arena__ssa_v0_pview, %27
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v71 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v66 + ((v31 + (i41 < v31 ? v31 : i41) * v26) + (v56 < v31 ? v31 : v56) * v27), v69, v70);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v64, v71);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %leaf_scores_inline121__ssa_v0
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v72 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v21);
        // pto: %leaf_scores_inline121__ssa_v0
        uint64_t v73 = (uint64_t) v31;
        TASSIGN(v72, v73);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TFILLPAD(v72, v64);
        // pto: %t__tmp_v2
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v2
        uint64_t v75 = (uint64_t) v29;
        TASSIGN(v74, v75);
        TEXPANDS(v74, v16);
        // pto: %leaf_scores_v1_inline118__ssa_v0
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v76 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v21);
        // pto: %leaf_scores_v1_inline118__ssa_v0
        uint64_t v77 = (uint64_t) v29;
        TASSIGN(v76, v77);
        pipe_barrier(PIPE_V);
        TMAX(v76, v72, v74);
        // pto: %t__tmp_v3
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v3
        uint64_t v79 = (uint64_t) v30;
        TASSIGN(v78, v79);
        // pto: %pairs_inline120__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v80 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_inline120__ssa_v0
        uint64_t v81 = (uint64_t) v31;
        TASSIGN(v80, v81);
        // pto: %sort32_src_view
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v82;
        // pto: %sort32_src_view
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v83 = v82;
        TRESHAPE(v83, v76);
        // pto: %sort32_idx_view
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v84;
        // pto: %sort32_idx_view
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v85 = v84;
        TRESHAPE(v85, v78);
        // pto: %sort32_dst_view
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v86;
        // pto: %sort32_dst_view
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v87 = v86;
        TRESHAPE(v87, v80);
        pipe_barrier(PIPE_V);
        TSORT32(v87, v83, v85);
        // pto: %pairs_v1_inline119__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v88 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v1_inline119__ssa_v0
        uint64_t v89 = (uint64_t) v30;
        TASSIGN(v88, v89);
        pipe_barrier(PIPE_V);
        TMRGSORT(v88, v80, v14);
        // pto: %pairs_v2_inline123__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v90 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v2_inline123__ssa_v0
        uint64_t v91 = (uint64_t) v31;
        TASSIGN(v90, v91);
        pipe_barrier(PIPE_V);
        TMRGSORT(v90, v88, v13);
        // pto: %pairs_v3_inline117__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v92 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v3_inline117__ssa_v0
        uint64_t v93 = (uint64_t) v30;
        TASSIGN(v92, v93);
        pipe_barrier(PIPE_V);
        TMRGSORT(v92, v90, v12);
        // pto: %pairs_v4_inline122__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v94 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v4_inline122__ssa_v0
        uint64_t v95 = (uint64_t) v31;
        TASSIGN(v94, v95);
        pipe_barrier(PIPE_V);
        TMRGSORT(v94, v92, v11);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %t__tmp_v4
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v96 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v25);
        // pto: %t__tmp_v4
        uint64_t v97 = (uint64_t) v31;
        TASSIGN(v96, v97);
        // pto: %slice_view
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v98;
        // pto: %slice_view
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v99 = v98;
        // pto: %slice_view
        uint64_t v100 = (uint64_t) v31;
        TASSIGN(v99, v100);
        // pto: %pair_arena__ssa_v0_pview
        pto::Shape<1, 1, 1, 1, 1024> v101 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %pair_arena__ssa_v0_pview
        pto::Stride<1024, 1024, 1024, 1024, 1> v102 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %28, %pair_arena__ssa_v0_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v103 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v4 + (v31 + (v55 < v31 ? v31 : v55) * v25), v101, v102);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v103, v99);
      } else {
        // pto: %29
        int64_t v104 = (int64_t) ((uint64_t) v39 * (uint64_t) v20);
        // pto: %30
        int64_t v105 = (int64_t) ((uint64_t) v104 + (uint64_t) v10);
        int64_t v106 = (int64_t) ((uint64_t) j52 * (uint64_t) v15);
        // pto: %32
        int64_t v107 = (int64_t) ((uint64_t) v49 - (uint64_t) v106);
        // pto: %33
        int64_t v108 = v107 < v21 ? v107 : v21;
        // pto: %34
        int32_t v109 = (int32_t) v106;
        // pto: %t__tmp_v5
        Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v5
        uint64_t v111 = (uint64_t) v31;
        TASSIGN(v110, v111);
        TCI<Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v110, v17);
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        // pto: %leaf_indices_inline135__ssa_v0
        Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v112 = Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %leaf_indices_inline135__ssa_v0
        uint64_t v113 = (uint64_t) v30;
        TASSIGN(v112, v113);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        TADDS(v112, v110, v109);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %leaf_scores_raw_inline134__ssa_v0
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v108);
        // pto: %leaf_scores_raw_inline134__ssa_v0
        uint64_t v115 = (uint64_t) v31;
        TASSIGN(v114, v115);
        // pto: %35
        int64_t v116 = i41 < v31 ? v31 : i41;
        // pto: %37
        __gm__ float* v117 = PTOAS__GLOBAL_TENSOR_DATA(v38);
        // pto: %37
        int64_t v118 = v27 * v26;
        // pto: %37
        int64_t v119 = v27 * v118;
        // pto: %37
        pto::Shape<1, 1, 1, 1, -1> v120 = pto::Shape<1, 1, 1, 1, -1>(v27, v27, v27, v27, v108);
        // pto: %37
        pto::Stride<-1, -1, -1, -1, -1> v121 = pto::Stride<-1, -1, -1, -1, -1>(v27 * v119, v119, v118, v26, v27);
        // pto: %37, %36
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v122 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v117 + ((v31 + v116 * v26) + (v106 < v31 ? v31 : v106) * v27), v120, v121);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v114, v122);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %leaf_scores_inline131__ssa_v0
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v123 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v21);
        // pto: %leaf_scores_inline131__ssa_v0
        uint64_t v124 = (uint64_t) v31;
        TASSIGN(v123, v124);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        TFILLPAD(v123, v114);
        // pto: %t__tmp_v6
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v6
        uint64_t v126 = (uint64_t) v29;
        TASSIGN(v125, v126);
        TEXPANDS(v125, v16);
        // pto: %leaf_scores_v1_inline128__ssa_v0
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v127 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v21);
        // pto: %leaf_scores_v1_inline128__ssa_v0
        uint64_t v128 = (uint64_t) v29;
        TASSIGN(v127, v128);
        pipe_barrier(PIPE_V);
        TMAX(v127, v123, v125);
        // pto: %t__tmp_v7
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v7
        uint64_t v130 = (uint64_t) v30;
        TASSIGN(v129, v130);
        // pto: %pairs_inline130__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v131 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_inline130__ssa_v0
        uint64_t v132 = (uint64_t) v31;
        TASSIGN(v131, v132);
        // pto: %38
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v133;
        // pto: %38
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v134 = v133;
        TRESHAPE(v134, v127);
        // pto: %39
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v135;
        // pto: %39
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v136 = v135;
        TRESHAPE(v136, v129);
        // pto: %40
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v137;
        // pto: %40
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v138 = v137;
        TRESHAPE(v138, v131);
        pipe_barrier(PIPE_V);
        TSORT32(v138, v134, v136);
        // pto: %pairs_v1_inline129__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v139 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v1_inline129__ssa_v0
        uint64_t v140 = (uint64_t) v30;
        TASSIGN(v139, v140);
        pipe_barrier(PIPE_V);
        TMRGSORT(v139, v131, v14);
        // pto: %pairs_v2_inline133__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v141 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v2_inline133__ssa_v0
        uint64_t v142 = (uint64_t) v31;
        TASSIGN(v141, v142);
        pipe_barrier(PIPE_V);
        TMRGSORT(v141, v139, v13);
        // pto: %pairs_v3_inline127__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v143 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v3_inline127__ssa_v0
        uint64_t v144 = (uint64_t) v30;
        TASSIGN(v143, v144);
        pipe_barrier(PIPE_V);
        TMRGSORT(v143, v141, v12);
        // pto: %pairs_v4_inline132__ssa_v0
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v145 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v4_inline132__ssa_v0
        uint64_t v146 = (uint64_t) v31;
        TASSIGN(v145, v146);
        pipe_barrier(PIPE_V);
        TMRGSORT(v145, v143, v11);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // pto: %t__tmp_v8
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v147 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v25);
        // pto: %t__tmp_v8
        uint64_t v148 = (uint64_t) v31;
        TASSIGN(v147, v148);
        // pto: %41
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v149;
        // pto: %41
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v150 = v149;
        // pto: %41
        uint64_t v151 = (uint64_t) v31;
        TASSIGN(v150, v151);
        // pto: %43
        pto::Shape<1, 1, 1, 1, 1024> v152 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %43
        pto::Stride<1024, 1024, 1024, 1024, 1> v153 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %42, %43
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v154 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v4 + (v31 + (v105 < v31 ? v31 : v105) * v25), v152, v153);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(v154, v150);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
        // pto: %44, %45
        int64_t v155 = (int64_t) ((uint64_t) ((int64_t) ((uint64_t) v53 + (uint64_t) v27)) * (uint64_t) v21);
        // pto: %46
        int64_t v156 = (int64_t) ((uint64_t) v49 - (uint64_t) v155);
        // pto: %47
        int64_t v157 = v156 < v21 ? v156 : v21;
        // pto: %48
        int32_t v158 = (int32_t) v155;
        // pto: %t__tmp_v9
        Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v9
        uint64_t v160 = (uint64_t) v31;
        TASSIGN(v159, v160);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID1);
        TCI<Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v159, v17);
        set_flag(PIPE_S, PIPE_V, EVENT_ID2);
        // pto: %leaf_indices_inline135__ssa_v1
        Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v161 = Tile<TileType::Vec, int32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %leaf_indices_inline135__ssa_v1
        uint64_t v162 = (uint64_t) v30;
        TASSIGN(v161, v162);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
        TADDS(v161, v159, v158);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        // pto: %leaf_scores_raw_inline134__ssa_v1
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v157);
        // pto: %leaf_scores_raw_inline134__ssa_v1
        uint64_t v164 = (uint64_t) v31;
        TASSIGN(v163, v164);
        // pto: %51
        __gm__ float* v165 = PTOAS__GLOBAL_TENSOR_DATA(v38);
        // pto: %51
        int64_t v166 = v27 * v26;
        // pto: %51
        int64_t v167 = v27 * v166;
        // pto: %51
        pto::Shape<1, 1, 1, 1, -1> v168 = pto::Shape<1, 1, 1, 1, -1>(v27, v27, v27, v27, v157);
        // pto: %51
        pto::Stride<-1, -1, -1, -1, -1> v169 = pto::Stride<-1, -1, -1, -1, -1>(v27 * v167, v167, v166, v26, v27);
        // pto: %51, %50
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v170 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v165 + ((v31 + v116 * v26) + (v155 < v31 ? v31 : v155) * v27), v168, v169);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        TLOAD(v163, v170);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %leaf_scores_inline131__ssa_v1
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v171 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v21);
        // pto: %leaf_scores_inline131__ssa_v1
        uint64_t v172 = (uint64_t) v31;
        TASSIGN(v171, v172);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TFILLPAD(v171, v163);
        // pto: %t__tmp_v10
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v10
        uint64_t v174 = (uint64_t) v29;
        TASSIGN(v173, v174);
        TEXPANDS(v173, v16);
        // pto: %leaf_scores_v1_inline128__ssa_v1
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v175 = Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v21);
        // pto: %leaf_scores_v1_inline128__ssa_v1
        uint64_t v176 = (uint64_t) v29;
        TASSIGN(v175, v176);
        pipe_barrier(PIPE_V);
        TMAX(v175, v171, v173);
        // pto: %t__tmp_v11
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v21);
        // pto: %t__tmp_v11
        uint64_t v178 = (uint64_t) v30;
        TASSIGN(v177, v178);
        // pto: %pairs_inline130__ssa_v1
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v179 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_inline130__ssa_v1
        uint64_t v180 = (uint64_t) v31;
        TASSIGN(v179, v180);
        // pto: %52
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v181;
        // pto: %52
        Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v182 = v181;
        TRESHAPE(v182, v175);
        // pto: %53
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183;
        // pto: %53
        Tile<TileType::Vec, uint32_t, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v184 = v183;
        TRESHAPE(v184, v177);
        // pto: %54
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v185;
        // pto: %54
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v186 = v185;
        TRESHAPE(v186, v179);
        pipe_barrier(PIPE_V);
        TSORT32(v186, v182, v184);
        // pto: %pairs_v1_inline129__ssa_v1
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v187 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v1_inline129__ssa_v1
        uint64_t v188 = (uint64_t) v30;
        TASSIGN(v187, v188);
        pipe_barrier(PIPE_V);
        TMRGSORT(v187, v179, v14);
        // pto: %pairs_v2_inline133__ssa_v1
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v189 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v2_inline133__ssa_v1
        uint64_t v190 = (uint64_t) v31;
        TASSIGN(v189, v190);
        pipe_barrier(PIPE_V);
        TMRGSORT(v189, v187, v13);
        // pto: %pairs_v3_inline127__ssa_v1
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v191 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v3_inline127__ssa_v1
        uint64_t v192 = (uint64_t) v30;
        TASSIGN(v191, v192);
        pipe_barrier(PIPE_V);
        TMRGSORT(v191, v189, v12);
        // pto: %pairs_v4_inline132__ssa_v1
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v193 = Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v15);
        // pto: %pairs_v4_inline132__ssa_v1
        uint64_t v194 = (uint64_t) v31;
        TASSIGN(v193, v194);
        pipe_barrier(PIPE_V);
        TMRGSORT(v193, v191, v11);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        // pto: %t__tmp_v12
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v195 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v27, v25);
        // pto: %t__tmp_v12
        uint64_t v196 = (uint64_t) v31;
        TASSIGN(v195, v196);
        // pto: %55
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v197;
        // pto: %55
        Tile<TileType::Vec, float, 1, 16384, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v198 = v197;
        // pto: %55
        uint64_t v199 = (uint64_t) v31;
        TASSIGN(v198, v199);
        int64_t v200 = (int64_t) ((uint64_t) v104 + (uint64_t) v33);
        // pto: %58
        pto::Shape<1, 1, 1, 1, 1024> v201 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %58
        pto::Stride<1024, 1024, 1024, 1024, 1> v202 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %57, %58
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v203 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v4 + (v31 + (v200 < v31 ? v31 : v200) * v25), v201, v202);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        TSTORE(v203, v198);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        // pto: %left_inline141__ssa_v0
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v204 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v25);
        // pto: %left_inline141__ssa_v0
        uint64_t v205 = (uint64_t) v29;
        TASSIGN(v204, v205);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(v204, v154);
        // pto: %right_inline140__ssa_v0
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v206 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v25);
        // pto: %right_inline140__ssa_v0
        uint64_t v207 = (uint64_t) v28;
        TASSIGN(v206, v207);
        TLOAD(v206, v203);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %merge_tmp_inline138__ssa_v0
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v208 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v9);
        // pto: %merge_tmp_inline138__ssa_v0
        uint64_t v209 = (uint64_t) v31;
        TASSIGN(v208, v209);
        // pto: %merged_all_inline139__ssa_v0
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v9);
        // pto: %merged_all_inline139__ssa_v0
        uint64_t v211 = (uint64_t) v30;
        TASSIGN(v210, v211);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        TMRGSORT<Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, false>(v210, v32, v208, v204, v206);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        // pto: %merged_inline137__ssa_v0
        Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v212 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v27, v25);
        // pto: %merged_inline137__ssa_v0
        uint64_t v213 = (uint64_t) v30;
        TASSIGN(v212, v213);
        // pto: %64
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v214;
        // pto: %64
        Tile<TileType::Vec, float, 1, 2048, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v215 = v214;
        // pto: %64
        uint64_t v216 = (uint64_t) v30;
        TASSIGN(v215, v216);
        // pto: %66
        pto::Shape<1, 1, 1, 1, 1024> v217 = pto::Shape<1, 1, 1, 1, 1024>();
        // pto: %66
        pto::Stride<1024, 1024, 1024, 1024, 1> v218 = pto::Stride<1024, 1024, 1024, 1024, 1>();
        // pto: %65, %66
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v219 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v4 + (v31 + (v55 < v31 ? v31 : v55) * v25), v217, v218);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        TSTORE(v219, v215);
      }
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
    // pto: %67
    v40 = (int64_t) ((uint64_t) v42 + (uint64_t) v51);
  }
  // pto: %global_group_base__rv_v2
  int64_t v220 = v40;
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}