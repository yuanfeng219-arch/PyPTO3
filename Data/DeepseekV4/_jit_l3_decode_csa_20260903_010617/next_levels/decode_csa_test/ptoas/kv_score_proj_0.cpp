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

AICORE void kv_score_proj_0(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  const int64_t v10 = 256;
  const int64_t v11 = 2;
  const int64_t v12 = 32;
  const int64_t v13 = 16;
  const int64_t v14 = 8;
  const int64_t v15 = 512;
  const int64_t v16 = 1;
  const int64_t v17 = 4096;
  const int64_t v18 = 32768;
  const int64_t v19 = 49152;
  const int64_t v20 = 16384;
  const int64_t v21 = 131072;
  const int64_t v22 = 98304;
  const int64_t v23 = 81920;
  const int64_t v24 = 2048;
  const int64_t v25 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %x_flat_inline2162__ssa_v0_view
  int64_t v26 = v7 * v17;
  // pto: %x_flat_inline2162__ssa_v0_view
  int64_t v27 = v16 * v26;
  // pto: %x_flat_inline2162__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v16, v16, v16, v7, v17);
  // pto: %x_flat_inline2162__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v27, v27, v26, v17, v16);
  // pto: %x_flat_inline2162__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v30 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v28, v29);
  // pto: %kv_acc_inline2107__phi_v5
  Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v31 = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v13, v12);
  // pto: %kv_acc_inline2107__phi_v5
  uint64_t v32 = (uint64_t) v25;
  TASSIGN(v31, v32);
  // pto: %score_acc_inline2103__phi_v5
  Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v33 = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v13, v12);
  // pto: %score_acc_inline2103__phi_v5
  uint64_t v34 = (uint64_t) v24;
  TASSIGN(v33, v34);
  // pto: %idx_inline2108__ssa_v0
  int64_t v35 = (int64_t) v8;
  // pto: %19, %20
  int64_t v36 = (int64_t) ((uint64_t) (v35 / v14) * (uint64_t) v13);
  // pto: %21, %22
  int64_t v37 = (int64_t) ((uint64_t) (v35 % v14) * (uint64_t) v12);
  // pto: %kv_acc_inline2107__tile
  Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v13, v12);
  // pto: %kv_acc_inline2107__tile
  uint64_t v39 = (uint64_t) v25;
  TASSIGN(v38, v39);
  // pto: %score_acc_inline2103__tile
  Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v13, v12);
  // pto: %score_acc_inline2103__tile
  uint64_t v41 = (uint64_t) v24;
  TASSIGN(v40, v41);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  for (int64_t i42 = v25; i42 < v14; i42 += v11) {
    // pto: %23
    int64_t v43 = (int64_t) ((uint64_t) i42 * (uint64_t) v15);
    // pto: %24
    int64_t v44 = (int64_t) ((uint64_t) v6 - (uint64_t) v36);
    // pto: %25
    int64_t v45 = v44 < v13 ? v44 : v13;
    // pto: %27
    int64_t v46 = (int64_t) ((uint64_t) v43 + (uint64_t) v15);
    // pto: %x_tile_inline2096__tile
    Tile<TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v47 = Tile<TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v45, v15);
    // pto: %x_tile_inline2096__tile
    uint64_t v48 = (uint64_t) v23;
    TASSIGN(v47, v48);
    // pto: %30
    int64_t v49 = v36 < v25 ? v25 : v36;
    // pto: %31
    int64_t v50 = v43 < v25 ? v25 : v43;
    // pto: %x_flat_inline2162__ssa_v0_pview
    __gm__ bfloat16_t* v51 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %x_flat_inline2162__ssa_v0_pview
    int64_t v52 = v45 * v17;
    // pto: %x_flat_inline2162__ssa_v0_pview
    int64_t v53 = v16 * v52;
    // pto: %x_flat_inline2162__ssa_v0_pview
    pto::Shape<1, 1, 1, -1, 512> v54 = pto::Shape<1, 1, 1, -1, 512>(v16, v16, v16, v45, v15);
    // pto: %x_flat_inline2162__ssa_v0_pview
    pto::Stride<-1, -1, -1, -1, -1> v55 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v53, v53, v52, v17, v16);
    // pto: %x_flat_inline2162__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v56 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v51 + ((v25 + v49 * v17) + v50 * v16), v54, v55);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v47, v56);
    // pto: %wkv_tile_inline2114__tile
    Tile<TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v15);
    // pto: %wkv_tile_inline2114__tile
    uint64_t v58 = (uint64_t) v22;
    TASSIGN(v57, v58);
    // pto: %32
    int64_t v59 = v37 < v25 ? v25 : v37;
    // pto: %inner_wkv__ssa_v0_pview
    pto::Shape<1, 1, 1, 32, 512> v60 = pto::Shape<1, 1, 1, 32, 512>();
    // pto: %inner_wkv__ssa_v0_pview
    pto::Stride<131072, 131072, 131072, 4096, 1> v61 = pto::Stride<131072, 131072, 131072, 4096, 1>();
    // pto: %inner_wkv__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND> v62 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND>(v2 + ((v25 + v59 * v17) + v50), v60, v61);
    TLOAD(v57, v62);
    // pto: %wgate_tile_inline2100__tile
    Tile<TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v63 = Tile<TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v15);
    // pto: %wgate_tile_inline2100__tile
    uint64_t v64 = (uint64_t) v21;
    TASSIGN(v63, v64);
    // pto: %inner_wgate__ssa_v0_pview
    pto::Shape<1, 1, 1, 32, 512> v65 = pto::Shape<1, 1, 1, 32, 512>();
    // pto: %inner_wgate__ssa_v0_pview
    pto::Stride<131072, 131072, 131072, 4096, 1> v66 = pto::Stride<131072, 131072, 131072, 4096, 1>();
    // pto: %inner_wgate__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND> v67 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND>(v3 + ((v25 + v59 * v17) + v50), v65, v66);
    TLOAD(v63, v67);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %0
    Tile<TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v68 = Tile<TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v45, v15);
    // pto: %0
    uint64_t v69 = (uint64_t) v25;
    TASSIGN(v68, v69);
    // pto: %37
    int64_t v70 = v46 < v25 ? v25 : v46;
    // pto: %38
    __gm__ bfloat16_t* v71 = PTOAS__GLOBAL_TENSOR_DATA(v30);
    // pto: %38
    int64_t v72 = v45 * v17;
    // pto: %38
    int64_t v73 = v16 * v72;
    // pto: %38
    pto::Shape<1, 1, 1, -1, 512> v74 = pto::Shape<1, 1, 1, -1, 512>(v16, v16, v16, v45, v15);
    // pto: %38
    pto::Stride<-1, -1, -1, -1, -1> v75 = pto::Stride<-1, -1, -1, -1, -1>(v16 * v73, v73, v72, v17, v16);
    // pto: %38
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v76 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v71 + ((v25 + v49 * v17) + v70 * v16), v74, v75);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TLOAD(v68, v76);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %1
    Tile<TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v15);
    // pto: %1
    uint64_t v78 = (uint64_t) v20;
    TASSIGN(v77, v78);
    // pto: %41
    pto::Shape<1, 1, 1, 32, 512> v79 = pto::Shape<1, 1, 1, 32, 512>();
    // pto: %41
    pto::Stride<131072, 131072, 131072, 4096, 1> v80 = pto::Stride<131072, 131072, 131072, 4096, 1>();
    // pto: %41
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND> v81 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND>(v2 + ((v25 + v59 * v17) + v70), v79, v80);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v77, v81);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    // pto: %2
    Tile<TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Mat, bfloat16_t, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v15);
    // pto: %2
    uint64_t v83 = (uint64_t) v19;
    TASSIGN(v82, v83);
    // pto: %44
    pto::Shape<1, 1, 1, 32, 512> v84 = pto::Shape<1, 1, 1, 32, 512>();
    // pto: %44
    pto::Stride<131072, 131072, 131072, 4096, 1> v85 = pto::Stride<131072, 131072, 131072, 4096, 1>();
    // pto: %44
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND> v86 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 512>, pto::Stride<131072, 131072, 131072, 4096, 1>, pto::Layout::ND>(v3 + ((v25 + v59 * v17) + v70), v84, v85);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    TLOAD(v82, v86);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    // pto: %45
    v31.SetValidShape(v45, v12);
    v33.SetValidShape(v45, v12);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    if (v43 == v25) {
      // pto: %wkv_tile_inline2114__tile_t
      Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %wkv_tile_inline2114__tile_t
      uint64_t v88 = (uint64_t) v22;
      TASSIGN(v87, v88);
      // pto: %x_tile_inline2096__tile_Left
      Tile<TileType::Left, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Left, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v45, v15);
      // pto: %x_tile_inline2096__tile_Left
      uint64_t v90 = (uint64_t) v25;
      TASSIGN(v89, v90);
      TMOV(v89, v47);
      // pto: %wkv_tile_inline2114__tile_t_Right
      Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %wkv_tile_inline2114__tile_t_Right
      uint64_t v92 = (uint64_t) v25;
      TASSIGN(v91, v92);
      TMOV(v91, v87);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %3
      Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v93 = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v45, v12);
      // pto: %3
      uint64_t v94 = (uint64_t) v25;
      TASSIGN(v93, v94);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      TMATMUL(v93, v89, v91);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      // pto: %wgate_tile_inline2100__tile_t
      Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %wgate_tile_inline2100__tile_t
      uint64_t v96 = (uint64_t) v21;
      TASSIGN(v95, v96);
      // pto: %wgate_tile_inline2100__tile_t_Right
      Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v97 = Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %wgate_tile_inline2100__tile_t_Right
      uint64_t v98 = (uint64_t) v25;
      TASSIGN(v97, v98);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      TMOV(v97, v95);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      // pto: %4
      Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v99 = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v45, v12);
      // pto: %4
      uint64_t v100 = (uint64_t) v24;
      TASSIGN(v99, v100);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL(v99, v89, v97);
    } else {
      // pto: %5
      Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %5
      uint64_t v102 = (uint64_t) v22;
      TASSIGN(v101, v102);
      // pto: %6
      Tile<TileType::Left, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Left, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v45, v15);
      // pto: %6
      uint64_t v104 = (uint64_t) v25;
      TASSIGN(v103, v104);
      TMOV(v103, v47);
      // pto: %7
      Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %7
      uint64_t v106 = (uint64_t) v25;
      TASSIGN(v105, v106);
      TMOV(v105, v101);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      TMATMUL_ACC(v38, v38, v103, v105);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      // pto: %9
      Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %9
      uint64_t v108 = (uint64_t) v21;
      TASSIGN(v107, v108);
      // pto: %10
      Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
      // pto: %10
      uint64_t v110 = (uint64_t) v25;
      TASSIGN(v109, v110);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      TMOV(v109, v107);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      TMATMUL_ACC(v40, v40, v103, v109);
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    // pto: %12
    Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
    // pto: %12
    uint64_t v112 = (uint64_t) v20;
    TASSIGN(v111, v112);
    // pto: %13
    Tile<TileType::Left, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Left, bfloat16_t, 16, 512, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v45, v15);
    // pto: %13
    uint64_t v114 = (uint64_t) v20;
    TASSIGN(v113, v114);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v113, v68);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    // pto: %14
    Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
    // pto: %14
    uint64_t v116 = (uint64_t) v18;
    TASSIGN(v115, v116);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    TMOV(v115, v111);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
    // pto: %15
    Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v13, v12);
    // pto: %15
    uint64_t v118 = (uint64_t) v25;
    TASSIGN(v117, v118);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
    TMATMUL_ACC(v117, v117, v113, v115);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    // pto: %16
    Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Mat, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
    // pto: %16
    uint64_t v120 = (uint64_t) v19;
    TASSIGN(v119, v120);
    // pto: %17
    Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Right, bfloat16_t, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v15, v12);
    // pto: %17
    uint64_t v122 = (uint64_t) v18;
    TASSIGN(v121, v122);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    TMOV(v121, v119);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
    // pto: %18
    Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v13, v12);
    // pto: %18
    uint64_t v124 = (uint64_t) v24;
    TASSIGN(v123, v124);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
    TMATMUL_ACC(v123, v123, v113, v121);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %46
  int64_t v125 = v36 < v25 ? v25 : v36;
  // pto: %47
  int64_t v126 = v37 < v25 ? v25 : v37;
  // pto: %kv_proj_pad_inline2129__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 32> v127 = pto::Shape<1, 1, 1, 16, 32>();
  // pto: %kv_proj_pad_inline2129__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 256, 1> v128 = pto::Stride<4096, 4096, 4096, 256, 1>();
  // pto: %kv_proj_pad_inline2129__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v129 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v4 + ((v25 + v125 * v10) + v126), v127, v128);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v129, v38);
  // pto: %score_proj_pad_inline2143__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 32> v130 = pto::Shape<1, 1, 1, 16, 32>();
  // pto: %score_proj_pad_inline2143__ssa_v0_pview
  pto::Stride<4096, 4096, 4096, 256, 1> v131 = pto::Stride<4096, 4096, 4096, 256, 1>();
  // pto: %score_proj_pad_inline2143__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v132 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v5 + ((v25 + v125 * v10) + v126), v130, v131);
  TSTORE(v132, v40);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}