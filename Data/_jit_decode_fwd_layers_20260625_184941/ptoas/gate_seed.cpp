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

AICORE void gate_seed(__gm__ float* v1) {
  const int64_t v2 = 16384;
  const float v3 = 0.0f;
  const int64_t v4 = 1024;
  const int64_t v5 = 2;
  const int64_t v6 = 1;
  const int64_t v7 = 17408;
  const int64_t v8 = 16;
  const int64_t v9 = 65536;
  const int64_t v10 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v11 = (size_t) v10; v11 < ((size_t) v8); v11 += (size_t) v5) {
    int64_t v12 = (int64_t) ((uint64_t) ((int64_t) v11) * (uint64_t) v4);
    Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v13 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v4);
    uint64_t v14 = (uint64_t) v10;
    TASSIGN(v13, v14);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v13, v3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v15 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v4);
    uint64_t v16 = (uint64_t) v9;
    TASSIGN(v15, v16);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    TEXPANDS(v15, v3);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pto::Shape<1, 1, 1, 16, 1024> v17 = pto::Shape<1, 1, 1, 16, 1024>();
    pto::Stride<278528, 278528, 278528, 17408, 1> v18 = pto::Stride<278528, 278528, 278528, 17408, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND>(v1 + (v10 + v10 * v7 + v12 * v6), v17, v18);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v19, v13);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 1024> v20 = pto::Shape<1, 1, 1, 16, 1024>();
    pto::Stride<278528, 278528, 278528, 17408, 1> v21 = pto::Stride<278528, 278528, 278528, 17408, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND> v22 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND>(v1 + (v10 + v10 * v7 + (int64_t) ((uint64_t) v12 + (uint64_t) v4) * v6), v20, v21);
    pipe_barrier(PIPE_MTE3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v22, v15);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  }
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v23 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v8, v4);
  uint64_t v24 = (uint64_t) v10;
  TASSIGN(v23, v24);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  TEXPANDS(v23, v3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  pto::Shape<1, 1, 1, 16, 1024> v25 = pto::Shape<1, 1, 1, 16, 1024>();
  pto::Stride<278528, 278528, 278528, 17408, 1> v26 = pto::Stride<278528, 278528, 278528, 17408, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND> v27 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1024>, pto::Stride<278528, 278528, 278528, 17408, 1>, pto::Layout::ND>(v1 + (v10 + v10 * v7 + v2 * v6), v25, v26);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  pipe_barrier(PIPE_MTE3);
  TSTORE(v27, v23);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}