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

AICORE void down_seed(__gm__ float* v1) {
  const int64_t v2 = 4096;
  const float v3 = 0.0f;
  const int64_t v4 = 1024;
  const int64_t v5 = 2;
  const int64_t v6 = 4;
  const int64_t v7 = 1;
  const int64_t v8 = 5120;
  const int64_t v9 = 16;
  const int64_t v10 = 65536;
  const int64_t v11 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v12 = (size_t) v11; v12 < ((size_t) v6); v12 += (size_t) v5) {
    int64_t v13 = (int64_t) ((uint64_t) ((int64_t) v12) * (uint64_t) v4);
    Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v14 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v4);
    uint64_t v15 = (uint64_t) v11;
    TASSIGN(v14, v15);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v14, v3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v16 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v4);
    uint64_t v17 = (uint64_t) v10;
    TASSIGN(v16, v17);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TEXPANDS(v16, v3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pto::Shape<1, 1, 1, 16, 1024> v18 = pto::Shape<1, 1, 1, 16, 1024>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v19 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v20 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v11 + v11 * v8 + v13 * v7), v18, v19);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v20, v14);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 1024> v21 = pto::Shape<1, 1, 1, 16, 1024>();
    pto::Stride<81920, 81920, 81920, 5120, 1> v22 = pto::Stride<81920, 81920, 81920, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v11 + v11 * v8 + (int64_t) ((uint64_t) v13 + (uint64_t) v4) * v7), v21, v22);
    pipe_barrier(PIPE_MTE3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v23, v16);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  }
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v4);
  uint64_t v25 = (uint64_t) v11;
  TASSIGN(v24, v25);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  TEXPANDS(v24, v3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  pto::Shape<1, 1, 1, 16, 1024> v26 = pto::Shape<1, 1, 1, 16, 1024>();
  pto::Stride<81920, 81920, 81920, 5120, 1> v27 = pto::Stride<81920, 81920, 81920, 5120, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND> v28 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<81920, 81920, 81920, 5120, 1>, pto::Layout::ND>(v1 + (v11 + v11 * v8 + v2 * v7), v26, v27);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v28, v24);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}