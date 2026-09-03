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

AICORE void indexer_score_leaf_wave_aic(__gm__ int32_t* v1, __gm__ float* v2, __gm__ int32_t* v3, __gm__ int8_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ int32_t* v7, __gm__ int8_t* v8, __gm__ float* v9, __gm__ float* v10, int64_t v11, int64_t v12, int64_t v13, int64_t v14, int64_t v15, int32_t v16, int32_t v17) {
  const int64_t v18 = 2;
  const int64_t v19 = 32;
  const int64_t v20 = 31;
  const int64_t v21 = 24;
  const int64_t v22 = 8191;
  const int64_t v23 = 4;
  const int64_t v24 = 8;
  const int64_t v25 = 64;
  const int64_t v26 = 128;
  const int64_t v27 = 262144;
  const int64_t v28 = 1;
  const int64_t v29 = 4096;
  const int64_t v30 = 12288;
  const int64_t v31 = 8192;
  const int64_t v32 = 0;
  const int32_t v33 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  auto v34 = TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>(v10, v33, v33);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  // pto: %global_leaf_base_inline51_inline2273__rv_v2
  int64_t v35;
  v35 = v32;
  for (int64_t i36 = v32; i36 < v11; i36 += v28) {
    // pto: %global_leaf_base_inline51_inline2273__rv_v2
    int64_t v37 = v35;
    // pto: %10
    int64_t v38 = i36 / v24;
    // pto: %position_inline54_inline2275__tile
    int32_t v39 = (v1)[i36];
    // pto: %t__tile
    int32_t v40 = (v3)[v38];
    // pto: %11, %12
    int64_t v41 = (int64_t) v40 / v23;
    // pto: %13, %14, %15
    int64_t v42 = (int64_t) ((uint64_t) ((int64_t) v39) + (uint64_t) v28) / v23;
    // pto: %16
    int64_t v43 = v41 < v42 ? v41 : v42;
    // pto: %17
    int64_t v44 = v43 < v27 ? v43 : v27;
    // pto: %18
    int64_t v45 = v44 < v32 ? v32 : v44;
    // pto: %19, %20
    int64_t v46 = (int64_t) ((uint64_t) v45 + (uint64_t) v22) / v31;
    // pto: %worker_inline75_inline2270__ssa_v0, %22, %21, %23
    for (int64_t j47 = (int64_t) ((uint64_t) ((int64_t) v16) + (uint64_t) (v37 % v21)) % v21; j47 < v46; j47 += v21) {
      // pto: %24
      int64_t v48 = (int64_t) ((uint64_t) j47 * (uint64_t) v31);
      // pto: %25
      int64_t v49 = (int64_t) ((uint64_t) v45 - (uint64_t) v48);
      // pto: %27
      int64_t v50 = (int64_t) ((uint64_t) i36 * (uint64_t) v25);
      // pto: %query_vector_inline70_inline2291__tile
      Tile<TileType::Mat, int8_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v51 = Tile<TileType::Mat, int8_t, 64, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v25, v26);
      // pto: %query_vector_inline70_inline2291__tile
      uint64_t v52 = (uint64_t) v32;
      TASSIGN(v51, v52);
      // pto: %qr_hadamard_i8_inline2177__rv_v2_pview
      pto::Shape<1, 1, 1, 64, 128> v53 = pto::Shape<1, 1, 1, 64, 128>();
      // pto: %qr_hadamard_i8_inline2177__rv_v2_pview
      pto::Stride<8192, 8192, 8192, 128, 1> v54 = pto::Stride<8192, 8192, 8192, 128, 1>();
      // pto: %28, %qr_hadamard_i8_inline2177__rv_v2_pview
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v55 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v4 + (v32 + (v50 < v32 ? v32 : v50) * v26), v53, v54);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v51, v55);
      // pto: %26, %29, %30
      int64_t v56 = (int64_t) ((uint64_t) (v49 < v31 ? v49 : v31) + (uint64_t) v20) / v19;
      // pto: %31
      int64_t v57 = v56 / v18;
      // pto: %32
      int64_t v58 = (int64_t) ((uint64_t) v57 * (uint64_t) v18);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      for (int64_t k59 = v32; k59 < v58; k59 += v18) {
        // pto: %33
        int64_t v60 = (int64_t) ((uint64_t) k59 * (uint64_t) v19);
        // pto: %37
        int64_t v61 = (int64_t) ((uint64_t) v38 * (uint64_t) v31);
        // pto: %38, %34, %35, %36
        int32_t v62 = (v7)[(int64_t) ((uint64_t) v61 + (uint64_t) ((int64_t) ((uint64_t) v48 + (uint64_t) v60) / v19))];
        // pto: %39, %40
        int64_t v63 = (int64_t) ((uint64_t) ((int64_t) v62) * (uint64_t) v19);
        // pto: %47, %43, %42, %44, %45
        int32_t v64 = (v7)[(int64_t) ((uint64_t) v61 + (uint64_t) ((int64_t) ((uint64_t) v48 + (uint64_t) ((int64_t) ((uint64_t) v60 + (uint64_t) v19))) / v19))];
        // pto: %48, %49
        int64_t v65 = (int64_t) ((uint64_t) ((int64_t) v64) * (uint64_t) v19);
        // pto: %kv_i8_inline38_inline2300__tile
        Tile<TileType::Mat, int8_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v66 = Tile<TileType::Mat, int8_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v26);
        // pto: %kv_i8_inline38_inline2300__tile
        uint64_t v67 = (uint64_t) v31;
        TASSIGN(v66, v67);
        // pto: %kv_cache_i8_flat_inline46_inline2265__ssa_v0_pview
        pto::Shape<1, 1, 1, 32, 128> v68 = pto::Shape<1, 1, 1, 32, 128>();
        // pto: %kv_cache_i8_flat_inline46_inline2265__ssa_v0_pview
        pto::Stride<4096, 4096, 4096, 128, 1> v69 = pto::Stride<4096, 4096, 4096, 128, 1>();
        // pto: %50, %kv_cache_i8_flat_inline46_inline2265__ssa_v0_pview
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v70 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v8 + (v32 + (v63 < v32 ? v32 : v63) * v26), v68, v69);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(v66, v70);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        // pto: %0
        Tile<TileType::Mat, int8_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Mat, int8_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v26);
        // pto: %0
        uint64_t v72 = (uint64_t) v30;
        TASSIGN(v71, v72);
        // pto: %52
        pto::Shape<1, 1, 1, 32, 128> v73 = pto::Shape<1, 1, 1, 32, 128>();
        // pto: %52
        pto::Stride<4096, 4096, 4096, 128, 1> v74 = pto::Stride<4096, 4096, 4096, 128, 1>();
        // pto: %51, %52
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v75 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v8 + (v32 + (v65 < v32 ? v32 : v65) * v26), v73, v74);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v71, v75);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        // pto: %query_vector_inline70_inline2291__tile_t
        Tile<TileType::Mat, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Mat, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v25);
        // pto: %query_vector_inline70_inline2291__tile_t
        uint64_t v77 = (uint64_t) v32;
        TASSIGN(v76, v77);
        // pto: %kv_i8_inline38_inline2300__tile_Left
        Tile<TileType::Left, int8_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Left, int8_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v26);
        // pto: %kv_i8_inline38_inline2300__tile_Left
        uint64_t v79 = (uint64_t) v32;
        TASSIGN(v78, v79);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TMOV(v78, v66);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        // pto: %query_vector_inline70_inline2291__tile_t_Right
        Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v25);
        // pto: %query_vector_inline70_inline2291__tile_t_Right
        uint64_t v81 = (uint64_t) v32;
        TASSIGN(v80, v81);
        TMOV(v80, v76);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        // pto: %score_i32_inline37_inline2302__tile
        Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v82 = Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v19, v25);
        // pto: %score_i32_inline37_inline2302__tile
        uint64_t v83 = (uint64_t) v32;
        TASSIGN(v82, v83);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        TMATMUL(v82, v78, v80);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TPUSH<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v34, v82);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
        // pto: %1
        Tile<TileType::Mat, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v84 = Tile<TileType::Mat, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v25);
        // pto: %1
        uint64_t v85 = (uint64_t) v32;
        TASSIGN(v84, v85);
        // pto: %2
        Tile<TileType::Left, int8_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Left, int8_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v26);
        // pto: %2
        uint64_t v87 = (uint64_t) v29;
        TASSIGN(v86, v87);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        TMOV(v86, v71);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        // pto: %3
        Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v88 = Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v25);
        // pto: %3
        uint64_t v89 = (uint64_t) v31;
        TASSIGN(v88, v89);
        TMOV(v88, v84);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        // pto: %4
        Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v19, v25);
        // pto: %4
        uint64_t v91 = (uint64_t) v32;
        TASSIGN(v90, v91);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
        TMATMUL(v90, v86, v88);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
        TPUSH<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v34, v90);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      }
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      // pto: %53, %54
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
      if ((int64_t) ((uint64_t) v56 - (uint64_t) v58) == v28) {
        // pto: %59, %60, %56, %57, %58
        int32_t v92 = (v7)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) v38 * (uint64_t) v31)) + (uint64_t) ((int64_t) ((uint64_t) v48 + (uint64_t) ((int64_t) ((uint64_t) v57 * (uint64_t) v25))) / v19))];
        // pto: %61, %62
        int64_t v93 = (int64_t) ((uint64_t) ((int64_t) v92) * (uint64_t) v19);
        // pto: %5
        Tile<TileType::Mat, int8_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Mat, int8_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v26);
        // pto: %5
        uint64_t v95 = (uint64_t) v31;
        TASSIGN(v94, v95);
        // pto: %64
        pto::Shape<1, 1, 1, 32, 128> v96 = pto::Shape<1, 1, 1, 32, 128>();
        // pto: %64
        pto::Stride<4096, 4096, 4096, 128, 1> v97 = pto::Stride<4096, 4096, 4096, 128, 1>();
        // pto: %63, %64
        GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v98 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v8 + (v32 + (v93 < v32 ? v32 : v93) * v26), v96, v97);
        TLOAD(v94, v98);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        // pto: %6
        Tile<TileType::Mat, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Mat, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v25);
        // pto: %6
        uint64_t v100 = (uint64_t) v32;
        TASSIGN(v99, v100);
        // pto: %7
        Tile<TileType::Left, int8_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v101 = Tile<TileType::Left, int8_t, 32, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v19, v26);
        // pto: %7
        uint64_t v102 = (uint64_t) v32;
        TASSIGN(v101, v102);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        TMOV(v101, v94);
        // pto: %8
        Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v103 = Tile<TileType::Right, int8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v26, v25);
        // pto: %8
        uint64_t v104 = (uint64_t) v32;
        TASSIGN(v103, v104);
        TMOV(v103, v99);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        // pto: %9
        Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v105 = Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v19, v25);
        // pto: %9
        uint64_t v106 = (uint64_t) v32;
        TASSIGN(v105, v106);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        TMATMUL(v105, v101, v103);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        TPUSH<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Acc, int32_t, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v34, v105);
      }
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    }
    // pto: %65
    v35 = (int64_t) ((uint64_t) v37 + (uint64_t) v46);
  }
  // pto: %global_leaf_base_inline51_inline2273__rv_v2
  int64_t v107 = v35;
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void indexer_score_leaf_wave_aiv(__gm__ int32_t* v1, __gm__ float* v2, __gm__ int32_t* v3, __gm__ int8_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ int32_t* v7, __gm__ int8_t* v8, __gm__ float* v9, __gm__ float* v10, int64_t v11, int64_t v12, int64_t v13, int64_t v14, int64_t v15, int32_t v16, int32_t v17, int32_t v18) {
  SaturationMode v19 = SaturationMode::OFF;
  RoundMode v20 = RoundMode::CAST_NONE;
  const int32_t v21 = 64;
  const int32_t v22 = 32;
  const float v23 = 0.0f;
  const int64_t v24 = 2;
  const int64_t v25 = 32;
  const int64_t v26 = 31;
  const int64_t v27 = 24;
  const int64_t v28 = 8192;
  const int64_t v29 = 8191;
  const int64_t v30 = 4;
  const int64_t v31 = 8;
  const int64_t v32 = 0;
  const int64_t v33 = 64;
  const int64_t v34 = 128;
  const int64_t v35 = 262144;
  const int64_t v36 = 1;
  const int64_t v37 = 66432;
  const int64_t v38 = 41856;
  const int64_t v39 = 50048;
  const int64_t v40 = 41728;
  const int64_t v41 = 17152;
  const int64_t v42 = 25344;
  const int64_t v43 = 17024;
  const int64_t v44 = 16896;
  const int64_t v45 = 16640;
  const int64_t v46 = 16384;
  const int32_t v47 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %205
  int64_t v48 = (int64_t) ((uint64_t) v15 * (uint64_t) v25);
  // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_view
  int64_t v49 = v48 * v36;
  // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_view
  int64_t v50 = v36 * v49;
  // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v51 = pto::Shape<1, 1, 1, -1, -1>(v36, v36, v36, v48, v36);
  // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v52 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v50, v50, v49, v36, v48);
  // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v9, v51, v52);
  auto v54 = TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>(v10, v47, v47);
  // pto: %subblock_idx, %75
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  if ((int64_t) v18 == v32) {
    // pto: %global_leaf_base_inline51_inline2273__rv_v2
    int64_t v55;
    v55 = v32;
    for (int64_t i56 = v32; i56 < v11; i56 += v36) {
      // pto: %global_leaf_base_inline51_inline2273__rv_v2
      int64_t v57 = v55;
      // pto: %76
      int64_t v58 = i56 / v31;
      // pto: %position_inline54_inline2275__tile
      int32_t v59 = (v1)[i56];
      // pto: %77
      int32_t v60 = (v3)[v58];
      // pto: %78, %79
      int64_t v61 = (int64_t) v60 / v30;
      // pto: %80, %81, %82
      int64_t v62 = (int64_t) ((uint64_t) ((int64_t) v59) + (uint64_t) v36) / v30;
      // pto: %83
      int64_t v63 = v61 < v62 ? v61 : v62;
      // pto: %84
      int64_t v64 = v63 < v35 ? v63 : v35;
      // pto: %85
      int64_t v65 = v64 < v32 ? v32 : v64;
      // pto: %86, %87
      int64_t v66 = (int64_t) ((uint64_t) v65 + (uint64_t) v29) / v28;
      // pto: %worker_inline75_inline2270__ssa_v0, %89, %88, %90
      for (int64_t j67 = (int64_t) ((uint64_t) ((int64_t) v16) + (uint64_t) (v57 % v27)) % v27; j67 < v66; j67 += v27) {
        // pto: %91
        int64_t v68 = (int64_t) ((uint64_t) j67 * (uint64_t) v28);
        // pto: %92
        int64_t v69 = (int64_t) ((uint64_t) v65 - (uint64_t) v68);
        // pto: %93
        int64_t v70 = v69 < v28 ? v69 : v28;
        // pto: %94
        int64_t v71 = (int64_t) ((uint64_t) i56 * (uint64_t) v33);
        // pto: %t__tile
        Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v33, v36);
        // pto: %t__tile
        uint64_t v73 = (uint64_t) v46;
        TASSIGN(v72, v73);
        // pto: %qr_hadamard_scale_dq_inline2234__ssa_v1_pview
        pto::Shape<1, 1, 1, 64, 1> v74 = pto::Shape<1, 1, 1, 64, 1>();
        // pto: %qr_hadamard_scale_dq_inline2234__ssa_v1_pview
        pto::Stride<64, 64, 64, 1, 16384> v75 = pto::Stride<64, 64, 64, 1, 16384>();
        // pto: %95, %qr_hadamard_scale_dq_inline2234__ssa_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 16384>, pto::Layout::DN> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 1>, pto::Stride<64, 64, 64, 1, 16384>, pto::Layout::DN>(v5 + (v32 + (v71 < v32 ? v32 : v71)), v74, v75);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v72, v76);
        // pto: %query_scale_inline73_inline2197__tile
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v77 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %query_scale_inline73_inline2197__tile
        uint64_t v78 = (uint64_t) v46;
        TASSIGN(v77, v78);
        // pto: %query_weight_inline76_inline2243__tile
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v79 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v33);
        // pto: %query_weight_inline76_inline2243__tile
        uint64_t v80 = (uint64_t) v45;
        TASSIGN(v79, v80);
        // pto: %96
        int64_t v81 = i56 < v32 ? v32 : i56;
        // pto: %weights_inline2244__ssa_v1_pview
        pto::Shape<1, 1, 1, 1, 64> v82 = pto::Shape<1, 1, 1, 1, 64>();
        // pto: %weights_inline2244__ssa_v1_pview
        pto::Stride<64, 64, 64, 64, 1> v83 = pto::Stride<64, 64, 64, 64, 1>();
        // pto: %weights_inline2244__ssa_v1_pview
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v84 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v6 + (v32 + v81 * v33), v82, v83);
        TLOAD(v79, v84);
        // pto: %97, %98
        int64_t v85 = (int64_t) ((uint64_t) v70 + (uint64_t) v26) / v25;
        // pto: %99
        int64_t v86 = v85 / v24;
        // pto: %100
        int64_t v87 = (int64_t) ((uint64_t) v86 * (uint64_t) v24);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        for (int64_t k88 = v32; k88 < v87; k88 += v24) {
          // pto: %101
          int64_t v89 = (int64_t) ((uint64_t) k88 * (uint64_t) v25);
          // pto: %102
          int64_t v90 = (int64_t) ((uint64_t) v68 + (uint64_t) v89);
          // pto: %105
          int64_t v91 = (int64_t) ((uint64_t) v58 * (uint64_t) v28);
          // pto: %106, %103, %104
          int32_t v92 = (v7)[(int64_t) ((uint64_t) v91 + (uint64_t) (v90 / v25))];
          // pto: %107, %108
          int64_t v93 = (int64_t) ((uint64_t) ((int64_t) v92) * (uint64_t) v25);
          // pto: %109
          int64_t v94 = (int64_t) ((uint64_t) v70 - (uint64_t) v89);
          // pto: %110
          int64_t v95 = v94 < v25 ? v94 : v25;
          // pto: %112
          int64_t v96 = (int64_t) ((uint64_t) v89 + (uint64_t) v25);
          // pto: %113
          int64_t v97 = (int64_t) ((uint64_t) v68 + (uint64_t) v96);
          // pto: %117, %114, %115
          int32_t v98 = (v7)[(int64_t) ((uint64_t) v91 + (uint64_t) (v97 / v25))];
          // pto: %118, %119
          int64_t v99 = (int64_t) ((uint64_t) ((int64_t) v98) * (uint64_t) v25);
          // pto: %120
          int64_t v100 = (int64_t) ((uint64_t) v70 - (uint64_t) v96);
          // pto: %121
          int64_t v101 = v100 < v25 ? v100 : v25;
          // pto: %kv_scale_inline32_inline2184__tile
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v102 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %kv_scale_inline32_inline2184__tile
          uint64_t v103 = (uint64_t) v44;
          TASSIGN(v102, v103);
          // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_pview
          __gm__ float* v104 = PTOAS__GLOBAL_TENSOR_DATA(v53);
          // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_pview
          int64_t v105 = v25 * v36;
          // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_pview
          int64_t v106 = v36 * v105;
          // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_pview
          pto::Shape<1, 1, 1, 32, 1> v107 = pto::Shape<1, 1, 1, 32, 1>(v36, v36, v36, v25, v36);
          // pto: %kv_scale_flat_inline50_inline2214__ssa_v0_pview
          pto::Stride<-1, -1, -1, -1, -1> v108 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v106, v106, v105, v36, v48);
          // pto: %122, %kv_scale_flat_inline50_inline2214__ssa_v0_pview
          GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v109 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v104 + ((v32 + (v93 < v32 ? v32 : v93) * v36) + v32 * v48), v107, v108);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          TLOAD(v102, v109);
          // pto: %0
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v110 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %0
          uint64_t v111 = (uint64_t) v43;
          TASSIGN(v110, v111);
          // pto: %124
          __gm__ float* v112 = PTOAS__GLOBAL_TENSOR_DATA(v53);
          // pto: %124
          int64_t v113 = v25 * v36;
          // pto: %124
          int64_t v114 = v36 * v113;
          // pto: %124
          pto::Shape<1, 1, 1, 32, 1> v115 = pto::Shape<1, 1, 1, 32, 1>(v36, v36, v36, v25, v36);
          // pto: %124
          pto::Stride<-1, -1, -1, -1, -1> v116 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v114, v114, v113, v36, v48);
          // pto: %123, %124
          GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v117 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v112 + ((v32 + (v99 < v32 ? v32 : v99) * v36) + v32 * v48), v115, v116);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          TLOAD(v110, v117);
          // pto: %score_i32_inline37_inline2302__tile_Vec
          Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v21);
          TPOP<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v54, v118);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          // pto: %score_fp32_inline35_inline2254__tile
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v119 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %score_fp32_inline35_inline2254__tile
          uint64_t v120 = (uint64_t) v42;
          TASSIGN(v119, v120);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          TCVT(v119, v118, v20, v19);
          TFREE<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v54);
          // pto: %score_fp32_v1_inline34_inline2251__tile
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %score_fp32_v1_inline34_inline2251__tile
          uint64_t v122 = (uint64_t) v42;
          TASSIGN(v121, v122);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v121, v119, v77);
          // pto: %score_fp32_v2_inline62_inline2294__tile
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v123 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %score_fp32_v2_inline62_inline2294__tile
          uint64_t v124 = (uint64_t) v42;
          TASSIGN(v123, v124);
          pipe_barrier(PIPE_V);
          TMAXS(v123, v121, v23);
          // pto: %score_fp32_v3_inline33_inline2187__tile
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v125 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %score_fp32_v3_inline33_inline2187__tile
          uint64_t v126 = (uint64_t) v41;
          TASSIGN(v125, v126);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v125, v123, v79);
          // pto: %tmp_tile
          Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v127 = Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v34);
          // pto: %tmp_tile
          uint64_t v128 = (uint64_t) v42;
          TASSIGN(v127, v128);
          // pto: %1
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v129 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %1
          uint64_t v130 = (uint64_t) v40;
          TASSIGN(v129, v130);
          pipe_barrier(PIPE_V);
          TROWSUM(v129, v125, v127);
          // pto: %score_inline58_inline2180__rm_a0_tmp_v0
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %score_inline58_inline2180__rm_a0_tmp_v0
          uint64_t v132 = (uint64_t) v40;
          TASSIGN(v131, v132);
          // pto: %score_inline58_inline2180__rm_a1_tmp_v1
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v133 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %score_inline58_inline2180__rm_a1_tmp_v1
          uint64_t v134 = (uint64_t) v44;
          TASSIGN(v133, v134);
          // pto: %score_inline58_inline2180__row_major_tmp_v2
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v135 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %score_inline58_inline2180__row_major_tmp_v2
          uint64_t v136 = (uint64_t) v42;
          TASSIGN(v135, v136);
          pipe_barrier(PIPE_V);
          TMUL(v135, v131, v133);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          // pto: %score_inline58_inline2180__tile
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v137 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %score_inline58_inline2180__tile
          uint64_t v138 = (uint64_t) v42;
          TASSIGN(v137, v138);
          // pto: %score_row_inline31_inline2296__tile
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v139 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %score_row_inline31_inline2296__tile
          uint64_t v140 = (uint64_t) v42;
          TASSIGN(v139, v140);
          v139.SetValidShape(v36, v95);
          // pto: %score_valid_inline30_inline2277__tile
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v141 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v36, v25);
          // pto: %score_valid_inline30_inline2277__tile
          uint64_t v142 = (uint64_t) v42;
          TASSIGN(v141, v142);
          pipe_barrier(PIPE_V);
          TFILLPAD(v141, v139);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          // pto: %score_arena_inline44_inline2267__iter_v5_pview
          pto::Shape<1, 1, 1, 1, 32> v143 = pto::Shape<1, 1, 1, 1, 32>();
          // pto: %score_arena_inline44_inline2267__iter_v5_pview
          pto::Stride<262144, 262144, 262144, 262144, 1> v144 = pto::Stride<262144, 262144, 262144, 262144, 1>();
          // pto: %score_arena_inline44_inline2267__iter_v5_pview, %126
          GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<262144, 262144, 262144, 262144, 1>, pto::Layout::ND> v145 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<262144, 262144, 262144, 262144, 1>, pto::Layout::ND>(v2 + ((v32 + v81 * v35) + (v90 < v32 ? v32 : v90)), v143, v144);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v145, v141);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          // pto: %127
          Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v21);
          TPOP<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v54, v146);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          // pto: %3
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v147 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %3
          uint64_t v148 = (uint64_t) v39;
          TASSIGN(v147, v148);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
          TCVT(v147, v146, v20, v19);
          TFREE<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v54);
          // pto: %4
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v149 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %4
          uint64_t v150 = (uint64_t) v39;
          TASSIGN(v149, v150);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v149, v147, v77);
          // pto: %5
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v151 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %5
          uint64_t v152 = (uint64_t) v39;
          TASSIGN(v151, v152);
          pipe_barrier(PIPE_V);
          TMAXS(v151, v149, v23);
          // pto: %6
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v153 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %6
          uint64_t v154 = (uint64_t) v38;
          TASSIGN(v153, v154);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v153, v151, v79);
          // pto: %7
          Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v34);
          // pto: %7
          uint64_t v156 = (uint64_t) v39;
          TASSIGN(v155, v156);
          // pto: %8
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v157 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %8
          uint64_t v158 = (uint64_t) v37;
          TASSIGN(v157, v158);
          pipe_barrier(PIPE_V);
          TROWSUM(v157, v153, v155);
          // pto: %9
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v159 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %9
          uint64_t v160 = (uint64_t) v37;
          TASSIGN(v159, v160);
          // pto: %10
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v161 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %10
          uint64_t v162 = (uint64_t) v43;
          TASSIGN(v161, v162);
          // pto: %11
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v163 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %11
          uint64_t v164 = (uint64_t) v39;
          TASSIGN(v163, v164);
          pipe_barrier(PIPE_V);
          TMUL(v163, v159, v161);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          // pto: %12
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v165 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %12
          uint64_t v166 = (uint64_t) v39;
          TASSIGN(v165, v166);
          // pto: %13
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v167 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %13
          uint64_t v168 = (uint64_t) v39;
          TASSIGN(v167, v168);
          v167.SetValidShape(v36, v101);
          // pto: %15
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v169 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v36, v25);
          // pto: %15
          uint64_t v170 = (uint64_t) v39;
          TASSIGN(v169, v170);
          pipe_barrier(PIPE_V);
          TFILLPAD(v169, v167);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          // pto: %score_arena_inline44_inline2267__tile_pview
          pto::Shape<1, 1, 1, 1, 32> v171 = pto::Shape<1, 1, 1, 1, 32>();
          // pto: %score_arena_inline44_inline2267__tile_pview
          pto::Stride<262144, 262144, 262144, 262144, 1> v172 = pto::Stride<262144, 262144, 262144, 262144, 1>();
          // pto: %score_arena_inline44_inline2267__tile_pview, %130
          GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<262144, 262144, 262144, 262144, 1>, pto::Layout::ND> v173 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<262144, 262144, 262144, 262144, 1>, pto::Layout::ND>(v2 + ((v32 + v81 * v35) + (v97 < v32 ? v32 : v97)), v171, v172);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v173, v169);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        // pto: %131, %132
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        if ((int64_t) ((uint64_t) v85 - (uint64_t) v87) == v36) {
          int64_t v174 = (int64_t) ((uint64_t) v86 * (uint64_t) v33);
          // pto: %134
          int64_t v175 = (int64_t) ((uint64_t) v68 + (uint64_t) v174);
          // pto: %137, %138, %135, %136
          int32_t v176 = (v7)[(int64_t) ((uint64_t) ((int64_t) ((uint64_t) v58 * (uint64_t) v28)) + (uint64_t) (v175 / v25))];
          // pto: %139, %140
          int64_t v177 = (int64_t) ((uint64_t) ((int64_t) v176) * (uint64_t) v25);
          // pto: %141
          Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v178 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v21);
          TPOP<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v54, v178);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          // pto: %16
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v179 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %16
          uint64_t v180 = (uint64_t) v42;
          TASSIGN(v179, v180);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          TCVT(v179, v178, v20, v19);
          TFREE<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v54);
          // pto: %17
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v181 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %17
          uint64_t v182 = (uint64_t) v42;
          TASSIGN(v181, v182);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v181, v179, v77);
          // pto: %18
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %18
          uint64_t v184 = (uint64_t) v42;
          TASSIGN(v183, v184);
          pipe_barrier(PIPE_V);
          TMAXS(v183, v181, v23);
          // pto: %19
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v185 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v33);
          // pto: %19
          uint64_t v186 = (uint64_t) v39;
          TASSIGN(v185, v186);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v185, v183, v79);
          // pto: %20
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %20
          uint64_t v188 = (uint64_t) v41;
          TASSIGN(v187, v188);
          // pto: %143
          __gm__ float* v189 = PTOAS__GLOBAL_TENSOR_DATA(v53);
          // pto: %143
          int64_t v190 = v25 * v36;
          // pto: %143
          int64_t v191 = v36 * v190;
          // pto: %143
          pto::Shape<1, 1, 1, 32, 1> v192 = pto::Shape<1, 1, 1, 32, 1>(v36, v36, v36, v25, v36);
          // pto: %143
          pto::Stride<-1, -1, -1, -1, -1> v193 = pto::Stride<-1, -1, -1, -1, -1>(v36 * v191, v191, v190, v36, v48);
          // pto: %142, %143
          GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v194 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v189 + ((v32 + (v177 < v32 ? v32 : v177) * v36) + v32 * v48), v192, v193);
          TLOAD(v187, v194);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
          // pto: %21
          Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v195 = Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v34);
          // pto: %21
          uint64_t v196 = (uint64_t) v42;
          TASSIGN(v195, v196);
          // pto: %22
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v197 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %22
          uint64_t v198 = (uint64_t) v38;
          TASSIGN(v197, v198);
          pipe_barrier(PIPE_V);
          TROWSUM(v197, v185, v195);
          // pto: %23
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %23
          uint64_t v200 = (uint64_t) v38;
          TASSIGN(v199, v200);
          // pto: %24
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %24
          uint64_t v202 = (uint64_t) v41;
          TASSIGN(v201, v202);
          // pto: %25
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v203 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %25
          uint64_t v204 = (uint64_t) v42;
          TASSIGN(v203, v204);
          pipe_barrier(PIPE_V);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
          TMUL(v203, v199, v201);
          // pto: %26
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v205 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v25, v36);
          // pto: %26
          uint64_t v206 = (uint64_t) v42;
          TASSIGN(v205, v206);
          // pto: %27
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36, v25);
          // pto: %27
          uint64_t v208 = (uint64_t) v42;
          TASSIGN(v207, v208);
          // pto: %144
          int64_t v209 = (int64_t) ((uint64_t) v70 - (uint64_t) v174);
          // pto: %145
          int64_t v210 = v209 < v25 ? v209 : v25;
          v207.SetValidShape(v36, v210);
          // pto: %29
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v211 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v36, v25);
          // pto: %29
          uint64_t v212 = (uint64_t) v42;
          TASSIGN(v211, v212);
          pipe_barrier(PIPE_V);
          TFILLPAD(v211, v207);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
          // pto: %score_arena_inline44_inline2267__rv_v6_main_pview
          pto::Shape<1, 1, 1, 1, 32> v213 = pto::Shape<1, 1, 1, 1, 32>();
          // pto: %score_arena_inline44_inline2267__rv_v6_main_pview
          pto::Stride<262144, 262144, 262144, 262144, 1> v214 = pto::Stride<262144, 262144, 262144, 262144, 1>();
          // pto: %score_arena_inline44_inline2267__rv_v6_main_pview, %148
          GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<262144, 262144, 262144, 262144, 1>, pto::Layout::ND> v215 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<262144, 262144, 262144, 262144, 1>, pto::Layout::ND>(v2 + ((v32 + v81 * v35) + (v175 < v32 ? v32 : v175)), v213, v214);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
          TSTORE(v215, v211);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      }
      // pto: %149
      v55 = (int64_t) ((uint64_t) v57 + (uint64_t) v66);
    }
    // pto: %global_leaf_base_inline51_inline2273__rv_v2
    int64_t v216 = v55;
  } else {
    // pto: %154
    int64_t v217;
    v217 = v32;
    for (int64_t i218 = v32; i218 < v11; i218 += v36) {
      // pto: %154
      int64_t v219 = v217;
      // pto: %156
      int32_t v220 = (v1)[i218];
      // pto: %155, %157
      int32_t v221 = (v3)[i218 / v31];
      // pto: %158, %159
      int64_t v222 = (int64_t) v221 / v30;
      // pto: %160, %161, %162
      int64_t v223 = (int64_t) ((uint64_t) ((int64_t) v220) + (uint64_t) v36) / v30;
      // pto: %163
      int64_t v224 = v222 < v223 ? v222 : v223;
      // pto: %164
      int64_t v225 = v224 < v35 ? v224 : v35;
      // pto: %165
      int64_t v226 = v225 < v32 ? v32 : v225;
      // pto: %166, %167
      int64_t v227 = (int64_t) ((uint64_t) v226 + (uint64_t) v29) / v28;
      // pto: %150, %169, %168, %170
      for (int64_t j228 = (int64_t) ((uint64_t) ((int64_t) v16) + (uint64_t) (v219 % v27)) % v27; j228 < v227; j228 += v27) {
        // pto: %173, %172
        int64_t v229 = (int64_t) ((uint64_t) v226 - (uint64_t) ((int64_t) ((uint64_t) j228 * (uint64_t) v28)));
        // pto: %30
        Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
        // pto: %30
        uint64_t v231 = (uint64_t) v46;
        TASSIGN(v230, v231);
        // pto: %31
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
        // pto: %31
        uint64_t v233 = (uint64_t) v46;
        TASSIGN(v232, v233);
        // pto: %32
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
        // pto: %32
        uint64_t v235 = (uint64_t) v45;
        TASSIGN(v234, v235);
        // pto: %174, %175, %176
        int64_t v236 = (int64_t) ((uint64_t) (v229 < v28 ? v229 : v28) + (uint64_t) v26) / v25;
        // pto: %177, %178
        int64_t v237 = (int64_t) ((uint64_t) (v236 / v24) * (uint64_t) v24);
        for (int64_t k238 = v32; k238 < v237; k238 += v24) {
          // pto: %193
          Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v239 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v21);
          v239.SetValidShape(v32, v32);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
          TPOP<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v54, v239);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
          // pto: %33
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %33
          uint64_t v241 = (uint64_t) v42;
          TASSIGN(v240, v241);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
          pipe_barrier(PIPE_V);
          TCVT(v240, v239, v20, v19);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
          TFREE<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v54);
          // pto: %34
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %34
          uint64_t v243 = (uint64_t) v42;
          TASSIGN(v242, v243);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v242, v240, v232);
          // pto: %35
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %35
          uint64_t v245 = (uint64_t) v42;
          TASSIGN(v244, v245);
          pipe_barrier(PIPE_V);
          TMAXS(v244, v242, v23);
          // pto: %36
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %36
          uint64_t v247 = (uint64_t) v41;
          TASSIGN(v246, v247);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v246, v244, v234);
          // pto: %37
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v248 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %37
          uint64_t v249 = (uint64_t) v44;
          TASSIGN(v248, v249);
          // pto: %38
          Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v250 = Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %38
          uint64_t v251 = (uint64_t) v42;
          TASSIGN(v250, v251);
          // pto: %39
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v252 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %39
          uint64_t v253 = (uint64_t) v43;
          TASSIGN(v252, v253);
          pipe_barrier(PIPE_V);
          TROWSUM(v252, v246, v250);
          // pto: %40
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %40
          uint64_t v255 = (uint64_t) v43;
          TASSIGN(v254, v255);
          // pto: %41
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %41
          uint64_t v257 = (uint64_t) v44;
          TASSIGN(v256, v257);
          // pto: %42
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v258 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %42
          uint64_t v259 = (uint64_t) v42;
          TASSIGN(v258, v259);
          pipe_barrier(PIPE_V);
          TMUL(v258, v254, v256);
          // pto: %43
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v260 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %43
          uint64_t v261 = (uint64_t) v42;
          TASSIGN(v260, v261);
          // pto: %44
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v262 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %44
          uint64_t v263 = (uint64_t) v42;
          TASSIGN(v262, v263);
          v262.SetValidShape(v32, v32);
          // pto: %46
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v264 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v32, v32);
          // pto: %46
          uint64_t v265 = (uint64_t) v42;
          TASSIGN(v264, v265);
          pipe_barrier(PIPE_V);
          TFILLPAD(v264, v262);
          // pto: %194
          Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v21);
          v266.SetValidShape(v32, v32);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
          TPOP<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v54, v266);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
          // pto: %47
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v267 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %47
          uint64_t v268 = (uint64_t) v39;
          TASSIGN(v267, v268);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
          TCVT(v267, v266, v20, v19);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
          TFREE<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v54);
          // pto: %48
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v269 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %48
          uint64_t v270 = (uint64_t) v39;
          TASSIGN(v269, v270);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v269, v267, v232);
          // pto: %49
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v271 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %49
          uint64_t v272 = (uint64_t) v39;
          TASSIGN(v271, v272);
          pipe_barrier(PIPE_V);
          TMAXS(v271, v269, v23);
          // pto: %50
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v273 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %50
          uint64_t v274 = (uint64_t) v38;
          TASSIGN(v273, v274);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v273, v271, v234);
          // pto: %51
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v275 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %51
          uint64_t v276 = (uint64_t) v40;
          TASSIGN(v275, v276);
          // pto: %52
          Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277 = Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %52
          uint64_t v278 = (uint64_t) v39;
          TASSIGN(v277, v278);
          // pto: %53
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %53
          uint64_t v280 = (uint64_t) v37;
          TASSIGN(v279, v280);
          pipe_barrier(PIPE_V);
          TROWSUM(v279, v273, v277);
          // pto: %54
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %54
          uint64_t v282 = (uint64_t) v37;
          TASSIGN(v281, v282);
          // pto: %55
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %55
          uint64_t v284 = (uint64_t) v40;
          TASSIGN(v283, v284);
          // pto: %56
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %56
          uint64_t v286 = (uint64_t) v39;
          TASSIGN(v285, v286);
          pipe_barrier(PIPE_V);
          TMUL(v285, v281, v283);
          // pto: %57
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %57
          uint64_t v288 = (uint64_t) v39;
          TASSIGN(v287, v288);
          // pto: %58
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %58
          uint64_t v290 = (uint64_t) v39;
          TASSIGN(v289, v290);
          v289.SetValidShape(v32, v32);
          // pto: %60
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v291 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v32, v32);
          // pto: %60
          uint64_t v292 = (uint64_t) v39;
          TASSIGN(v291, v292);
          pipe_barrier(PIPE_V);
          TFILLPAD(v291, v289);
        }
        // pto: %195, %196
        if ((int64_t) ((uint64_t) v236 - (uint64_t) v237) == v36) {
          // pto: %203
          Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293 = Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v22, v21);
          v293.SetValidShape(v32, v32);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
          TPOP<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, Tile<TileType::Vec, int32_t, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v54, v293);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
          // pto: %61
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v294 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %61
          uint64_t v295 = (uint64_t) v42;
          TASSIGN(v294, v295);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
          pipe_barrier(PIPE_V);
          TCVT(v294, v293, v20, v19);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
          TFREE<TPipe<0, Direction::DIR_C2V, 8192, 2, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v54);
          // pto: %62
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %62
          uint64_t v297 = (uint64_t) v42;
          TASSIGN(v296, v297);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v296, v294, v232);
          // pto: %63
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %63
          uint64_t v299 = (uint64_t) v42;
          TASSIGN(v298, v299);
          pipe_barrier(PIPE_V);
          TMAXS(v298, v296, v23);
          // pto: %64
          Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v300 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %64
          uint64_t v301 = (uint64_t) v39;
          TASSIGN(v300, v301);
          pipe_barrier(PIPE_V);
          TCOLEXPANDMUL(v300, v298, v234);
          // pto: %65
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %65
          uint64_t v303 = (uint64_t) v41;
          TASSIGN(v302, v303);
          // pto: %66
          Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %66
          uint64_t v305 = (uint64_t) v42;
          TASSIGN(v304, v305);
          // pto: %67
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %67
          uint64_t v307 = (uint64_t) v38;
          TASSIGN(v306, v307);
          pipe_barrier(PIPE_V);
          TROWSUM(v306, v300, v304);
          // pto: %68
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %68
          uint64_t v309 = (uint64_t) v38;
          TASSIGN(v308, v309);
          // pto: %69
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %69
          uint64_t v311 = (uint64_t) v41;
          TASSIGN(v310, v311);
          // pto: %70
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %70
          uint64_t v313 = (uint64_t) v42;
          TASSIGN(v312, v313);
          pipe_barrier(PIPE_V);
          TMUL(v312, v308, v310);
          // pto: %71
          Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %71
          uint64_t v315 = (uint64_t) v42;
          TASSIGN(v314, v315);
          // pto: %72
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32, v32);
          // pto: %72
          uint64_t v317 = (uint64_t) v42;
          TASSIGN(v316, v317);
          v316.SetValidShape(v32, v32);
          // pto: %74
          Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null> v318 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Min, CompactMode::Null>(v32, v32);
          // pto: %74
          uint64_t v319 = (uint64_t) v42;
          TASSIGN(v318, v319);
          pipe_barrier(PIPE_V);
          TFILLPAD(v318, v316);
        }
      }
      // pto: %204
      v217 = (int64_t) ((uint64_t) v219 + (uint64_t) v227);
    }
    // pto: %154
    int64_t v320 = v217;
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}