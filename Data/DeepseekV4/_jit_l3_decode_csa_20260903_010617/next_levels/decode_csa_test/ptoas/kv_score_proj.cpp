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

AICORE void kv_score_proj(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, int64_t v6, int64_t v7, int32_t v8, int32_t v9) {
  const int64_t v10 = 1024;
  const int64_t v11 = 256;
  const int64_t v12 = 2;
  const int64_t v13 = 8;
  const int64_t v14 = 64;
  const int64_t v15 = 16;
  const int64_t v16 = 512;
  const int64_t v17 = 1;
  const int64_t v18 = 24576;
  const int64_t v19 = 32768;
  const int64_t v20 = 8192;
  const int64_t v21 = 81920;
  const int64_t v22 = 16384;
  const int64_t v23 = 229376;
  const int64_t v24 = 163840;
  const int64_t v25 = 147456;
  const int64_t v26 = 4096;
  const int64_t v27 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  // pto: %x_flat_inline2021__ssa_v0_view
  int64_t v28 = v7 * v26;
  // pto: %x_flat_inline2021__ssa_v0_view
  int64_t v29 = v17 * v28;
  // pto: %x_flat_inline2021__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v17, v17, v17, v7, v26);
  // pto: %x_flat_inline2021__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v29, v29, v28, v26, v17);
  // pto: %x_flat_inline2021__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v30, v31);
  // pto: %kv_acc_inline2010__phi_v5
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v33 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v15, v14);
  // pto: %kv_acc_inline2010__phi_v5
  uint64_t v34 = (uint64_t) v27;
  TASSIGN(v33, v34);
  // pto: %score_acc_inline2014__phi_v5
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v35 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v15, v14);
  // pto: %score_acc_inline2014__phi_v5
  uint64_t v36 = (uint64_t) v26;
  TASSIGN(v35, v36);
  // pto: %idx_inline2012__ssa_v0
  int64_t v37 = (int64_t) v8;
  // pto: %37, %38
  int64_t v38 = (int64_t) ((uint64_t) (v37 / v15) * (uint64_t) v15);
  // pto: %39, %40
  int64_t v39 = (int64_t) ((uint64_t) (v37 % v15) * (uint64_t) v14);
  // pto: %kv_acc_inline2010__tile
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %kv_acc_inline2010__tile
  uint64_t v41 = (uint64_t) v27;
  TASSIGN(v40, v41);
  // pto: %score_acc_inline2014__tile
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v15, v14);
  // pto: %score_acc_inline2014__tile
  uint64_t v43 = (uint64_t) v26;
  TASSIGN(v42, v43);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int64_t i44 = v27; i44 < v13; i44 += v12) {
    // pto: %41
    int64_t v45 = (int64_t) ((uint64_t) i44 * (uint64_t) v16);
    // pto: %42
    int64_t v46 = (int64_t) ((uint64_t) v6 - (uint64_t) v38);
    // pto: %43
    int64_t v47 = v46 < v15 ? v46 : v15;
    // pto: %45
    int64_t v48 = (int64_t) ((uint64_t) v45 + (uint64_t) v16);
    // pto: %x_tile_inline2033__tile
    Tile<TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v49 = Tile<TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v16);
    // pto: %x_tile_inline2033__tile
    uint64_t v50 = (uint64_t) v25;
    TASSIGN(v49, v50);
    // pto: %48
    int64_t v51 = v38 < v27 ? v27 : v38;
    // pto: %49
    int64_t v52 = v45 < v27 ? v27 : v45;
    // pto: %x_flat_inline2021__ssa_v0_pview
    __gm__ bfloat16_t* v53 = PTOAS__GLOBAL_TENSOR_DATA(v32);
    // pto: %x_flat_inline2021__ssa_v0_pview
    int64_t v54 = v47 * v26;
    // pto: %x_flat_inline2021__ssa_v0_pview
    int64_t v55 = v17 * v54;
    // pto: %x_flat_inline2021__ssa_v0_pview
    pto::Shape<1, 1, 1, -1, 512> v56 = pto::Shape<1, 1, 1, -1, 512>(v17, v17, v17, v47, v16);
    // pto: %x_flat_inline2021__ssa_v0_pview
    pto::Stride<-1, -1, -1, -1, -1> v57 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v55, v55, v54, v26, v17);
    // pto: %x_flat_inline2021__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v58 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v53 + ((v27 + v51 * v26) + v52 * v17), v56, v57);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v49, v58);
    // pto: %wkv_tile_inline2007__tile
    Tile<TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v59 = Tile<TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v16);
    // pto: %wkv_tile_inline2007__tile
    uint64_t v60 = (uint64_t) v24;
    TASSIGN(v59, v60);
    // pto: %50
    int64_t v61 = v39 < v27 ? v27 : v39;
    // pto: %cmp_wkv__ssa_v0_pview
    pto::Shape<1, 1, 1, 64, 512> v62 = pto::Shape<1, 1, 1, 64, 512>();
    // pto: %cmp_wkv__ssa_v0_pview
    pto::Stride<262144, 262144, 262144, 4096, 1> v63 = pto::Stride<262144, 262144, 262144, 4096, 1>();
    // pto: %cmp_wkv__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND> v64 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND>(v2 + ((v27 + v61 * v26) + v52), v62, v63);
    TLOAD(v59, v64);
    // pto: %wgate_tile_inline1998__tile
    Tile<TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v16);
    // pto: %wgate_tile_inline1998__tile
    uint64_t v66 = (uint64_t) v23;
    TASSIGN(v65, v66);
    // pto: %cmp_wgate__ssa_v0_pview
    pto::Shape<1, 1, 1, 64, 512> v67 = pto::Shape<1, 1, 1, 64, 512>();
    // pto: %cmp_wgate__ssa_v0_pview
    pto::Stride<262144, 262144, 262144, 4096, 1> v68 = pto::Stride<262144, 262144, 262144, 4096, 1>();
    // pto: %cmp_wgate__ssa_v0_pview
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND> v69 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND>(v3 + ((v27 + v61 * v26) + v52), v67, v68);
    TLOAD(v65, v69);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %0
    Tile<TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v70 = Tile<TileType::Mat, bfloat16_t, 16, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v16);
    // pto: %0
    uint64_t v71 = (uint64_t) v27;
    TASSIGN(v70, v71);
    // pto: %55
    int64_t v72 = v48 < v27 ? v27 : v48;
    // pto: %56
    __gm__ bfloat16_t* v73 = PTOAS__GLOBAL_TENSOR_DATA(v32);
    // pto: %56
    int64_t v74 = v47 * v26;
    // pto: %56
    int64_t v75 = v17 * v74;
    // pto: %56
    pto::Shape<1, 1, 1, -1, 512> v76 = pto::Shape<1, 1, 1, -1, 512>(v17, v17, v17, v47, v16);
    // pto: %56
    pto::Stride<-1, -1, -1, -1, -1> v77 = pto::Stride<-1, -1, -1, -1, -1>(v17 * v75, v75, v74, v26, v17);
    // pto: %56
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v78 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v73 + ((v27 + v51 * v26) + v72 * v17), v76, v77);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    TLOAD(v70, v78);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    // pto: %1
    Tile<TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v16);
    // pto: %1
    uint64_t v80 = (uint64_t) v22;
    TASSIGN(v79, v80);
    // pto: %59
    pto::Shape<1, 1, 1, 64, 512> v81 = pto::Shape<1, 1, 1, 64, 512>();
    // pto: %59
    pto::Stride<262144, 262144, 262144, 4096, 1> v82 = pto::Stride<262144, 262144, 262144, 4096, 1>();
    // pto: %59
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND> v83 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND>(v2 + ((v27 + v61 * v26) + v72), v81, v82);
    TLOAD(v79, v83);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    // pto: %2
    Tile<TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v84 = Tile<TileType::Mat, bfloat16_t, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v14, v16);
    // pto: %2
    uint64_t v85 = (uint64_t) v21;
    TASSIGN(v84, v85);
    // pto: %62
    pto::Shape<1, 1, 1, 64, 512> v86 = pto::Shape<1, 1, 1, 64, 512>();
    // pto: %62
    pto::Stride<262144, 262144, 262144, 4096, 1> v87 = pto::Stride<262144, 262144, 262144, 4096, 1>();
    // pto: %62
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND> v88 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 512>, pto::Stride<262144, 262144, 262144, 4096, 1>, pto::Layout::ND>(v3 + ((v27 + v61 * v26) + v72), v86, v87);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v84, v88);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    // pto: %63
    v33.SetValidShape(v47, v14);
    v35.SetValidShape(v47, v14);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    if (v45 == v27) {
      // pto: %wkv_tile_inline2007__tile_t
      Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v16, v14);
      // pto: %wkv_tile_inline2007__tile_t
      uint64_t v90 = (uint64_t) v24;
      TASSIGN(v89, v90);
      // pto: %kv_acc_inline2010__tile_l0_init_storage
      Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v91 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v15, v14);
      // pto: %kv_acc_inline2010__tile_l0_init_storage
      uint64_t v92 = (uint64_t) v27;
      TASSIGN(v91, v92);
      v91.SetValidShape(v47, v14);
      // pto: %kv_acc_inline2010__tile_l0_a
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v93 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
      // pto: %kv_acc_inline2010__tile_l0_a
      uint64_t v94 = (uint64_t) v27;
      TASSIGN(v93, v94);
      TEXTRACT(v93, v49, v27, v27);
      // pto: %kv_acc_inline2010__tile_l0_b
      Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v95 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
      // pto: %kv_acc_inline2010__tile_l0_b
      uint64_t v96 = (uint64_t) v27;
      TASSIGN(v95, v96);
      TEXTRACT(v95, v89, v27, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %3
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v97 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
      // pto: %3
      uint64_t v98 = (uint64_t) v20;
      TASSIGN(v97, v98);
      TEXTRACT(v97, v49, v27, v11);
      // pto: %4
      Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
      // pto: %4
      uint64_t v100 = (uint64_t) v19;
      TASSIGN(v99, v100);
      TEXTRACT(v99, v89, v11, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      TMATMUL(v91, v93, v95);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v91, v91, v97, v99);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      // pto: %wgate_tile_inline1998__tile_t
      Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v16, v14);
      // pto: %wgate_tile_inline1998__tile_t
      uint64_t v102 = (uint64_t) v23;
      TASSIGN(v101, v102);
      // pto: %score_acc_inline2014__tile_l0_init_storage
      Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal> v103 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Normal>(v15, v14);
      // pto: %score_acc_inline2014__tile_l0_init_storage
      uint64_t v104 = (uint64_t) v26;
      TASSIGN(v103, v104);
      v103.SetValidShape(v47, v14);
      // pto: %score_acc_inline2014__tile_l0_a
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v105 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
      // pto: %score_acc_inline2014__tile_l0_a
      uint64_t v106 = (uint64_t) v27;
      TASSIGN(v105, v106);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      TEXTRACT(v105, v49, v27, v27);
      // pto: %score_acc_inline2014__tile_l0_b
      Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v107 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
      // pto: %score_acc_inline2014__tile_l0_b
      uint64_t v108 = (uint64_t) v27;
      TASSIGN(v107, v108);
      TEXTRACT(v107, v101, v27, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      // pto: %6
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v109 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
      // pto: %6
      uint64_t v110 = (uint64_t) v20;
      TASSIGN(v109, v110);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      TEXTRACT(v109, v49, v27, v11);
      // pto: %7
      Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v111 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
      // pto: %7
      uint64_t v112 = (uint64_t) v19;
      TASSIGN(v111, v112);
      TEXTRACT(v111, v101, v11, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      TMATMUL(v103, v105, v107);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      TMATMUL_ACC(v103, v103, v109, v111);
    } else {
      // pto: %9
      Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v113 = Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v16, v14);
      // pto: %9
      uint64_t v114 = (uint64_t) v24;
      TASSIGN(v113, v114);
      // pto: %10
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v115 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
      // pto: %10
      uint64_t v116 = (uint64_t) v27;
      TASSIGN(v115, v116);
      TEXTRACT(v115, v49, v27, v27);
      // pto: %11
      Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
      // pto: %11
      uint64_t v118 = (uint64_t) v27;
      TASSIGN(v117, v118);
      TEXTRACT(v117, v113, v27, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      // pto: %12
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v119 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
      // pto: %12
      uint64_t v120 = (uint64_t) v20;
      TASSIGN(v119, v120);
      TEXTRACT(v119, v49, v27, v11);
      // pto: %13
      Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
      // pto: %13
      uint64_t v122 = (uint64_t) v19;
      TASSIGN(v121, v122);
      TEXTRACT(v121, v113, v11, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      TMATMUL_ACC(v40, v40, v115, v117);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      TMATMUL_ACC(v40, v40, v119, v121);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      // pto: %16
      Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v16, v14);
      // pto: %16
      uint64_t v124 = (uint64_t) v23;
      TASSIGN(v123, v124);
      // pto: %17
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v125 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
      // pto: %17
      uint64_t v126 = (uint64_t) v27;
      TASSIGN(v125, v126);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      TEXTRACT(v125, v49, v27, v27);
      // pto: %18
      Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
      // pto: %18
      uint64_t v128 = (uint64_t) v27;
      TASSIGN(v127, v128);
      TEXTRACT(v127, v123, v27, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
      // pto: %19
      Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v129 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
      // pto: %19
      uint64_t v130 = (uint64_t) v20;
      TASSIGN(v129, v130);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      TEXTRACT(v129, v49, v27, v11);
      // pto: %20
      Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
      // pto: %20
      uint64_t v132 = (uint64_t) v19;
      TASSIGN(v131, v132);
      TEXTRACT(v131, v123, v11, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
      TMATMUL_ACC(v42, v42, v125, v127);
      pipe_barrier(PIPE_M);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
      TMATMUL_ACC(v42, v42, v129, v131);
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    // pto: %23
    Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %23
    uint64_t v134 = (uint64_t) v22;
    TASSIGN(v133, v134);
    // pto: %24
    Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v135 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
    // pto: %24
    uint64_t v136 = (uint64_t) v22;
    TASSIGN(v135, v136);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TEXTRACT(v135, v70, v27, v27);
    // pto: %25
    Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
    // pto: %25
    uint64_t v138 = (uint64_t) v27;
    TASSIGN(v137, v138);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    TEXTRACT(v137, v133, v27, v27);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // pto: %26
    Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v139 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
    // pto: %26
    uint64_t v140 = (uint64_t) v18;
    TASSIGN(v139, v140);
    TEXTRACT(v139, v70, v27, v11);
    // pto: %27
    Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v141 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
    // pto: %27
    uint64_t v142 = (uint64_t) v19;
    TASSIGN(v141, v142);
    TEXTRACT(v141, v133, v11, v27);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    // pto: %28
    Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v15, v14);
    // pto: %28
    uint64_t v144 = (uint64_t) v27;
    TASSIGN(v143, v144);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL_ACC(v143, v143, v135, v137);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    TMATMUL_ACC(v143, v143, v139, v141);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    // pto: %30
    Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Mat, bfloat16_t, 512, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v16, v14);
    // pto: %30
    uint64_t v146 = (uint64_t) v21;
    TASSIGN(v145, v146);
    // pto: %31
    Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v147 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
    // pto: %31
    uint64_t v148 = (uint64_t) v22;
    TASSIGN(v147, v148);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    TEXTRACT(v147, v70, v27, v27);
    // pto: %32
    Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
    // pto: %32
    uint64_t v150 = (uint64_t) v27;
    TASSIGN(v149, v150);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    TEXTRACT(v149, v145, v27, v27);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    // pto: %33
    Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal> v151 = Tile<TileType::Left, bfloat16_t, 16, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Normal>(v47, v11);
    // pto: %33
    uint64_t v152 = (uint64_t) v18;
    TASSIGN(v151, v152);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    TEXTRACT(v151, v70, v27, v11);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    // pto: %34
    Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Right, bfloat16_t, 256, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v11, v14);
    // pto: %34
    uint64_t v154 = (uint64_t) v19;
    TASSIGN(v153, v154);
    TEXTRACT(v153, v145, v11, v27);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    // pto: %35
    Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v15, v14);
    // pto: %35
    uint64_t v156 = (uint64_t) v26;
    TASSIGN(v155, v156);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL_ACC(v155, v155, v147, v149);
    pipe_barrier(PIPE_M);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    TMATMUL_ACC(v155, v155, v151, v153);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  // pto: %64
  int64_t v157 = v38 < v27 ? v27 : v38;
  // pto: %65
  int64_t v158 = v39 < v27 ? v27 : v39;
  // pto: %cmp4_kv_proj_pad_inline2031__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 64> v159 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %cmp4_kv_proj_pad_inline2031__ssa_v0_pview
  pto::Stride<16384, 16384, 16384, 1024, 1> v160 = pto::Stride<16384, 16384, 16384, 1024, 1>();
  // pto: %cmp4_kv_proj_pad_inline2031__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND> v161 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(v4 + ((v27 + v157 * v10) + v158), v159, v160);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v161, v40);
  // pto: %cmp4_score_proj_pad_inline2019__ssa_v0_pview
  pto::Shape<1, 1, 1, 16, 64> v162 = pto::Shape<1, 1, 1, 16, 64>();
  // pto: %cmp4_score_proj_pad_inline2019__ssa_v0_pview
  pto::Stride<16384, 16384, 16384, 1024, 1> v163 = pto::Stride<16384, 16384, 16384, 1024, 1>();
  // pto: %cmp4_score_proj_pad_inline2019__ssa_v0_pview
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND> v164 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<16384, 16384, 16384, 1024, 1>, pto::Layout::ND>(v5 + ((v27 + v157 * v10) + v158), v162, v163);
  TSTORE(v164, v42);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}