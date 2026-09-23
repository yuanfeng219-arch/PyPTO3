#include "pto/pto-inst.hpp"
using namespace pto;

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

AICORE void dcr_xgamma(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ bfloat16_t* v5, int32_t v6, int32_t v7) {
  SaturationMode v8 = SaturationMode::OFF;
  RoundMode v9 = RoundMode::CAST_ROUND;
  const int64_t v10 = 1024;
  const int64_t v11 = 1;
  const int64_t v12 = 5120;
  const int64_t v13 = 16;
  const int64_t v14 = 65536;
  const int64_t v15 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v16 = (int64_t) ((uint64_t) ((int64_t) v6) * (uint64_t) v10);
  Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v17 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v10);
  uint64_t v18 = (uint64_t) v15;
  TASSIGN(v17, v18);
  pto::Shape<1, 1, 1, 16, 1024> v19 = pto::Shape<1, 1, 1, 16, 1024>();
  pto::Stride<81920, 81920, 81920, 5120, 1> v20 = pto::Stride<81920, 81920, 81920, 5120, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v21 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v15 + v15 * v12 + v16 * v11), v19, v20);
  TLOAD(v17, v21);
  Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v22 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v10);
  uint64_t v23 = (uint64_t) v14;
  TASSIGN(v22, v23);
  pto::Shape<1, 1, 1, 16, 1024> v24 = pto::Shape<1, 1, 1, 16, 1024>();
  pto::Stride<81920, 81920, 81920, 5120, 1> v25 = pto::Stride<81920, 81920, 81920, 5120, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v26 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v2 + (v15 + v15 * v12 + v16 * v11), v24, v25);
  TLOAD(v22, v26);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v10);
  uint64_t v28 = (uint64_t) v15;
  TASSIGN(v27, v28);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v27, v17, v22);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  pto::Shape<1, 1, 1, 16, 1024> v29 = pto::Shape<1, 1, 1, 16, 1024>();
  pto::Stride<81920, 81920, 81920, 5120, 1> v30 = pto::Stride<81920, 81920, 81920, 5120, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v31 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v3 + (v15 + v15 * v12 + v16 * v11), v29, v30);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v31, v27);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v10);
  uint64_t v33 = (uint64_t) v14;
  TASSIGN(v32, v33);
  pto::Shape<1, 1, 1, 1, 1024> v34 = pto::Shape<1, 1, 1, 1, 1024>();
  pto::Stride<5120, 5120, 5120, 5120, 1> v35 = pto::Stride<5120, 5120, 5120, 5120, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v4 + (v15 + v15 * v12 + v16 * v11), v34, v35);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  TLOAD(v32, v36);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v10);
  uint64_t v38 = (uint64_t) v15;
  TASSIGN(v37, v38);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  TCOLEXPANDMUL(v37, v27, v32);
  Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v39 = Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v13, v10);
  uint64_t v40 = (uint64_t) v15;
  TASSIGN(v39, v40);
  pipe_barrier(PIPE_V);
  TCVT(v39, v37, v9, v8);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  pto::Shape<1, 1, 1, 16, 1024> v41 = pto::Shape<1, 1, 1, 16, 1024>();
  pto::Stride<81920, 81920, 81920, 5120, 1> v42 = pto::Stride<81920, 81920, 81920, 5120, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v43 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v5 + (v15 + v15 * v12 + v16 * v11), v41, v42);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  TSTORE(v43, v39);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}