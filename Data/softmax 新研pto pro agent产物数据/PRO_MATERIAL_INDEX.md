# PyPTO-Pro 全部资料索引

> **自动生成时间**: 2026-08-31
> **说明**: 本索引 §A/§C 由扫描命令动态生成，§B 为官方指定算子固定清单。§C 扫描 `tutorials/`。每次执行须重新扫描 §A/§C，§B 以官方最新指定清单为准。

---

## §A API 文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`）

> 搜索范围：`docs/pypto_pro/api/` 递归获取所有 `.md` 文档，共 **312** 个。

### A.1 API 总索引

| 文档 | 路径 |
|------|------|
| PyPTO-Pro API 总索引 | （缓存中未发现 `docs/pypto_api_list.md`，以 §A.2+ 分组目录为准） |

### A.2 SIMD-API（SIMD 向量/矩阵 API，含 VF 指令）（258 文档）

| # | 文档 | 路径 | 子类别 |
|---|------|------|--------|
| 1 | AccToVecMode | `docs/pypto_pro/api/SIMD-API/basic_data_structures/AccToVecMode.md` | basic_data_structures |
| 2 | DataType | `docs/pypto_pro/api/SIMD-API/basic_data_structures/DataType.md` | basic_data_structures |
| 3 | MemorySpace | `docs/pypto_pro/api/SIMD-API/basic_data_structures/MemorySpace.md` | basic_data_structures |
| 4 | PipeType | `docs/pypto_pro/api/SIMD-API/basic_data_structures/PipeType.md` | basic_data_structures |
| 5 | Ptr | `docs/pypto_pro/api/SIMD-API/basic_data_structures/Ptr.md` | basic_data_structures |
| 6 | ReluPreMode | `docs/pypto_pro/api/SIMD-API/basic_data_structures/ReluPreMode.md` | basic_data_structures |
| 7 | Scalar | `docs/pypto_pro/api/SIMD-API/basic_data_structures/Scalar.md` | basic_data_structures |
| 8 | Tensor | `docs/pypto_pro/api/SIMD-API/basic_data_structures/Tensor.md` | basic_data_structures |
| 9 | TensorLayout | `docs/pypto_pro/api/SIMD-API/basic_data_structures/TensorLayout.md` | basic_data_structures |
| 10 | TilePad | `docs/pypto_pro/api/SIMD-API/basic_data_structures/TilePad.md` | basic_data_structures |
| 11 | TileType | `docs/pypto_pro/api/SIMD-API/basic_data_structures/TileType.md` | basic_data_structures |
| 12 | index | `docs/pypto_pro/api/SIMD-API/basic_data_structures/index.md` | basic_data_structures |
| 13 | index | `docs/pypto_pro/api/SIMD-API/index.md` | （根） |
| 14 | index | `docs/pypto_pro/api/SIMD-API/operation/atomic_operations/index.md` | operation/atomic_operations |
| 15 | store_atomic | `docs/pypto_pro/api/SIMD-API/operation/atomic_operations/store_atomic.md` | operation/atomic_operations |
| 16 | dcci | `docs/pypto_pro/api/SIMD-API/operation/cache_control/dcci.md` | operation/cache_control |
| 17 | index | `docs/pypto_pro/api/SIMD-API/operation/cache_control/index.md` | operation/cache_control |
| 18 | index | `docs/pypto_pro/api/SIMD-API/operation/controlflow/index.md` | operation/controlflow |
| 19 | range | `docs/pypto_pro/api/SIMD-API/operation/controlflow/range.md` | operation/controlflow |
| 20 | section_vector_section_cube | `docs/pypto_pro/api/SIMD-API/operation/controlflow/section_vector_section_cube.md` | operation/controlflow |
| 21 | index | `docs/pypto_pro/api/SIMD-API/operation/index.md` | operation |
| 22 | index | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/index.md` | operation/matrix_computation |
| 23 | matmul | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/matmul.md` | operation/matrix_computation |
| 24 | matmul_acc | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/matmul_acc.md` | operation/matrix_computation |
| 25 | matmul_mx | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/matmul_mx.md` | operation/matrix_computation |
| 26 | matmul_mx_acc | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/matmul_mx_acc.md` | operation/matrix_computation |
| 27 | phase | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/phase.md` | operation/matrix_computation |
| 28 | set_mm_layout_transform | `docs/pypto_pro/api/SIMD-API/operation/matrix_computation/set_mm_layout_transform.md` | operation/matrix_computation |
| 29 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/index.md` | operation/memory_data_movement |
| 30 | insert | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/insert.md` | operation/memory_data_movement |
| 31 | load | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/load.md` | operation/memory_data_movement |
| 32 | load_tile | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/load_tile.md` | operation/memory_data_movement |
| 33 | move | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/move.md` | operation/memory_data_movement |
| 34 | ssbuf_load | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/ssbuf_load.md` | operation/memory_data_movement |
| 35 | ssbuf_store | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/ssbuf_store.md` | operation/memory_data_movement |
| 36 | store | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/store.md` | operation/memory_data_movement |
| 37 | store_tile | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/store_tile.md` | operation/memory_data_movement |
| 38 | subview | `docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/subview.md` | operation/memory_data_movement |
| 39 | comparison | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/comparison/comparison.md` | operation/memory_vector_computation/comparison |
| 40 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/comparison/index.md` | operation/memory_vector_computation/comparison |
| 41 | axpy | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/composite_computation/axpy.md` | operation/memory_vector_computation/composite_computation |
| 42 | fused_mul_add | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/composite_computation/fused_mul_add.md` | operation/memory_vector_computation/composite_computation |
| 43 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/composite_computation/index.md` | operation/memory_vector_computation/composite_computation |
| 44 | abs | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/abs.md` | operation/memory_vector_computation/elementwise |
| 45 | add | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/add.md` | operation/memory_vector_computation/elementwise |
| 46 | and_ | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/and_.md` | operation/memory_vector_computation/elementwise |
| 47 | div | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/div.md` | operation/memory_vector_computation/elementwise |
| 48 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/index.md` | operation/memory_vector_computation/elementwise |
| 49 | log | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/log.md` | operation/memory_vector_computation/elementwise |
| 50 | maximum | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/maximum.md` | operation/memory_vector_computation/elementwise |
| 51 | minimum | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/minimum.md` | operation/memory_vector_computation/elementwise |
| 52 | mul | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/mul.md` | operation/memory_vector_computation/elementwise |
| 53 | neg | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/neg.md` | operation/memory_vector_computation/elementwise |
| 54 | relu | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/relu.md` | operation/memory_vector_computation/elementwise |
| 55 | shl | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/shl.md` | operation/memory_vector_computation/elementwise |
| 56 | shr | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/shr.md` | operation/memory_vector_computation/elementwise |
| 57 | sqrt | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/sqrt.md` | operation/memory_vector_computation/elementwise |
| 58 | sub | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/sub.md` | operation/memory_vector_computation/elementwise |
| 59 | xor | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/elementwise/xor.md` | operation/memory_vector_computation/elementwise |
| 60 | fillpad | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fillpad.md` | operation/memory_vector_computation |
| 61 | add_relu | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/add_relu.md` | operation/memory_vector_computation/fused_vector_computation |
| 62 | add_relu_cast | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/add_relu_cast.md` | operation/memory_vector_computation/fused_vector_computation |
| 63 | addc | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/addc.md` | operation/memory_vector_computation/fused_vector_computation |
| 64 | fused_mul_add_relu | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/fused_mul_add_relu.md` | operation/memory_vector_computation/fused_vector_computation |
| 65 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/index.md` | operation/memory_vector_computation/fused_vector_computation |
| 66 | mul_add_dst | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/mul_add_dst.md` | operation/memory_vector_computation/fused_vector_computation |
| 67 | mul_cast | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/mul_cast.md` | operation/memory_vector_computation/fused_vector_computation |
| 68 | partadd | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partadd.md` | operation/memory_vector_computation/fused_vector_computation |
| 69 | partmax | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmax.md` | operation/memory_vector_computation/fused_vector_computation |
| 70 | partmin | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmin.md` | operation/memory_vector_computation/fused_vector_computation |
| 71 | partmul | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/partmul.md` | operation/memory_vector_computation/fused_vector_computation |
| 72 | sub_relu | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/sub_relu.md` | operation/memory_vector_computation/fused_vector_computation |
| 73 | sub_relu_cast | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/fused_vector_computation/sub_relu_cast.md` | operation/memory_vector_computation/fused_vector_computation |
| 74 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/index.md` | operation/memory_vector_computation |
| 75 | argmax | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/argmax.md` | operation/memory_vector_computation/math_functions |
| 76 | argmin | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/argmin.md` | operation/memory_vector_computation/math_functions |
| 77 | exp | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/exp.md` | operation/memory_vector_computation/math_functions |
| 78 | expand_div | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_div.md` | operation/memory_vector_computation/math_functions |
| 79 | expand_max | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_max.md` | operation/memory_vector_computation/math_functions |
| 80 | expand_min | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_min.md` | operation/memory_vector_computation/math_functions |
| 81 | expand_mul | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_mul.md` | operation/memory_vector_computation/math_functions |
| 82 | expand_sub | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expand_sub.md` | operation/memory_vector_computation/math_functions |
| 83 | expands | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/expands.md` | operation/memory_vector_computation/math_functions |
| 84 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/index.md` | operation/memory_vector_computation/math_functions |
| 85 | max | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/max.md` | operation/memory_vector_computation/math_functions |
| 86 | min | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/min.md` | operation/memory_vector_computation/math_functions |
| 87 | recip | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/recip.md` | operation/memory_vector_computation/math_functions |
| 88 | rsqrt | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/rsqrt.md` | operation/memory_vector_computation/math_functions |
| 89 | sum | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/math_functions/sum.md` | operation/memory_vector_computation/math_functions |
| 90 | gather | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/gather.md` | operation/memory_vector_computation/scatter_gather |
| 91 | gatherb | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/gatherb.md` | operation/memory_vector_computation/scatter_gather |
| 92 | gathermask | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/gathermask.md` | operation/memory_vector_computation/scatter_gather |
| 93 | histogram | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/histogram.md` | operation/memory_vector_computation/scatter_gather |
| 94 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/index.md` | operation/memory_vector_computation/scatter_gather |
| 95 | scatter | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/scatter_gather/scatter.md` | operation/memory_vector_computation/scatter_gather |
| 96 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/selection/index.md` | operation/memory_vector_computation/selection |
| 97 | select | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/selection/select.md` | operation/memory_vector_computation/selection |
| 98 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/index.md` | operation/memory_vector_computation/sorting |
| 99 | mrgsort | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/mrgsort.md` | operation/memory_vector_computation/sorting |
| 100 | mrgsort2 | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/mrgsort2.md` | operation/memory_vector_computation/sorting |
| 101 | sort32 | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/sorting/sort32.md` | operation/memory_vector_computation/sorting |
| 102 | getval | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/getval.md` | operation/memory_vector_computation/transpose_and_element_access |
| 103 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/index.md` | operation/memory_vector_computation/transpose_and_element_access |
| 104 | set_stride | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/set_stride.md` | operation/memory_vector_computation/transpose_and_element_access |
| 105 | set_validshape | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/set_validshape.md` | operation/memory_vector_computation/transpose_and_element_access |
| 106 | setval | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/setval.md` | operation/memory_vector_computation/transpose_and_element_access |
| 107 | transpose | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/transpose.md` | operation/memory_vector_computation/transpose_and_element_access |
| 108 | cast | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/type_conversion/cast.md` | operation/memory_vector_computation/type_conversion |
| 109 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/type_conversion/index.md` | operation/memory_vector_computation/type_conversion |
| 110 | index | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/index.md` | operation/memory_vector_computation/vector_mask |
| 111 | reset_mask | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/reset_mask.md` | operation/memory_vector_computation/vector_mask |
| 112 | set_mask_count | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/set_mask_count.md` | operation/memory_vector_computation/vector_mask |
| 113 | set_mask_norm | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/set_mask_norm.md` | operation/memory_vector_computation/vector_mask |
| 114 | set_vec_mask | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/vector_mask/set_vec_mask.md` | operation/memory_vector_computation/vector_mask |
| 115 | dequant | `docs/pypto_pro/api/SIMD-API/operation/quantization/dequant.md` | operation/quantization |
| 116 | index | `docs/pypto_pro/api/SIMD-API/operation/quantization/index.md` | operation/quantization |
| 117 | quant | `docs/pypto_pro/api/SIMD-API/operation/quantization/quant.md` | operation/quantization |
| 118 | addptr | `docs/pypto_pro/api/SIMD-API/operation/resource_management/addptr.md` | operation/resource_management |
| 119 | index | `docs/pypto_pro/api/SIMD-API/operation/resource_management/index.md` | operation/resource_management |
| 120 | make_ptr | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_ptr.md` | operation/resource_management |
| 121 | make_tensor | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_tensor.md` | operation/resource_management |
| 122 | make_tile | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_tile.md` | operation/resource_management |
| 123 | make_tile_group | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_tile_group.md` | operation/resource_management |
| 124 | barrier | `docs/pypto_pro/api/SIMD-API/operation/synchronization/barrier.md` | operation/synchronization |
| 125 | index | `docs/pypto_pro/api/SIMD-API/operation/synchronization/index.md` | operation/synchronization |
| 126 | mutex_lock_mutex_unlock | `docs/pypto_pro/api/SIMD-API/operation/synchronization/mutex_lock_mutex_unlock.md` | operation/synchronization |
| 127 | set_cross_core_wait_cross_core | `docs/pypto_pro/api/SIMD-API/operation/synchronization/set_cross_core_wait_cross_core.md` | operation/synchronization |
| 128 | sync_all | `docs/pypto_pro/api/SIMD-API/operation/synchronization/sync_all.md` | operation/synchronization |
| 129 | sync_src_sync_dst | `docs/pypto_pro/api/SIMD-API/operation/synchronization/sync_src_sync_dst.md` | operation/synchronization |
| 130 | get_block_idx | `docs/pypto_pro/api/SIMD-API/operation/system_variables/get_block_idx.md` | operation/system_variables |
| 131 | get_block_num | `docs/pypto_pro/api/SIMD-API/operation/system_variables/get_block_num.md` | operation/system_variables |
| 132 | get_subblock_idx | `docs/pypto_pro/api/SIMD-API/operation/system_variables/get_subblock_idx.md` | operation/system_variables |
| 133 | get_subblock_num | `docs/pypto_pro/api/SIMD-API/operation/system_variables/get_subblock_num.md` | operation/system_variables |
| 134 | index | `docs/pypto_pro/api/SIMD-API/operation/system_variables/index.md` | operation/system_variables |
| 135 | fill_index | `docs/pypto_pro/api/SIMD-API/operation/transpose_and_element_access/fill_index.md` | operation/transpose_and_element_access |
| 136 | index | `docs/pypto_pro/api/SIMD-API/operation/transpose_and_element_access/index.md` | operation/transpose_and_element_access |
| 137 | Array | `docs/pypto_pro/api/SIMD-API/operation/utilities/Array.md` | operation/utilities |
| 138 | index | `docs/pypto_pro/api/SIMD-API/operation/utilities/index.md` | operation/utilities |
| 139 | make_tuple | `docs/pypto_pro/api/SIMD-API/operation/utilities/make_tuple.md` | operation/utilities |
| 140 | struct | `docs/pypto_pro/api/SIMD-API/operation/utilities/struct.md` | operation/utilities |
| 141 | struct_array | `docs/pypto_pro/api/SIMD-API/operation/utilities/struct_array.md` | operation/utilities |
| 142 | de_interleave | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/de_interleave.md` | operation/vf_computation/advanced_computation |
| 143 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/index.md` | operation/vf_computation/advanced_computation |
| 144 | interleave | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/interleave.md` | operation/vf_computation/advanced_computation |
| 145 | pack | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/pack.md` | operation/vf_computation/advanced_computation |
| 146 | unpack | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/unpack.md` | operation/vf_computation/advanced_computation |
| 147 | unsqueeze | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/advanced_computation/unsqueeze.md` | operation/vf_computation/advanced_computation |
| 148 | abs | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/abs.md` | operation/vf_computation/basic_arithmetic |
| 149 | add | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/add.md` | operation/vf_computation/basic_arithmetic |
| 150 | addc | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/addc.md` | operation/vf_computation/basic_arithmetic |
| 151 | adds | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/adds.md` | operation/vf_computation/basic_arithmetic |
| 152 | div | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/div.md` | operation/vf_computation/basic_arithmetic |
| 153 | exp | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/exp.md` | operation/vf_computation/basic_arithmetic |
| 154 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/index.md` | operation/vf_computation/basic_arithmetic |
| 155 | leaky_relu | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/leaky_relu.md` | operation/vf_computation/basic_arithmetic |
| 156 | ln | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/ln.md` | operation/vf_computation/basic_arithmetic |
| 157 | log | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/log.md` | operation/vf_computation/basic_arithmetic |
| 158 | log10 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/log10.md` | operation/vf_computation/basic_arithmetic |
| 159 | log2 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/log2.md` | operation/vf_computation/basic_arithmetic |
| 160 | max | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/max.md` | operation/vf_computation/basic_arithmetic |
| 161 | maxs | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/maxs.md` | operation/vf_computation/basic_arithmetic |
| 162 | min | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/min.md` | operation/vf_computation/basic_arithmetic |
| 163 | mins | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/mins.md` | operation/vf_computation/basic_arithmetic |
| 164 | mul | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/mul.md` | operation/vf_computation/basic_arithmetic |
| 165 | mull | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/mull.md` | operation/vf_computation/basic_arithmetic |
| 166 | muls | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/muls.md` | operation/vf_computation/basic_arithmetic |
| 167 | neg | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/neg.md` | operation/vf_computation/basic_arithmetic |
| 168 | prelu | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/prelu.md` | operation/vf_computation/basic_arithmetic |
| 169 | relu | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/relu.md` | operation/vf_computation/basic_arithmetic |
| 170 | sqrt | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/sqrt.md` | operation/vf_computation/basic_arithmetic |
| 171 | sub | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/sub.md` | operation/vf_computation/basic_arithmetic |
| 172 | subc | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/subc.md` | operation/vf_computation/basic_arithmetic |
| 173 | eq | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/eq.md` | operation/vf_computation/comparison_and_selection |
| 174 | ge | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/ge.md` | operation/vf_computation/comparison_and_selection |
| 175 | gt | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/gt.md` | operation/vf_computation/comparison_and_selection |
| 176 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/index.md` | operation/vf_computation/comparison_and_selection |
| 177 | le | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/le.md` | operation/vf_computation/comparison_and_selection |
| 178 | lt | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/lt.md` | operation/vf_computation/comparison_and_selection |
| 179 | ne | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/ne.md` | operation/vf_computation/comparison_and_selection |
| 180 | select | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/select.md` | operation/vf_computation/comparison_and_selection |
| 181 | squeeze | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/comparison_and_selection/squeeze.md` | operation/vf_computation/comparison_and_selection |
| 182 | abs_sub | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/abs_sub.md` | operation/vf_computation/composite_computation |
| 183 | axpy | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/axpy.md` | operation/vf_computation/composite_computation |
| 184 | exp_sub | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/exp_sub.md` | operation/vf_computation/composite_computation |
| 185 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/index.md` | operation/vf_computation/composite_computation |
| 186 | mul_add_dst | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/mul_add_dst.md` | operation/vf_computation/composite_computation |
| 187 | mul_dst_add | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/mul_dst_add.md` | operation/vf_computation/composite_computation |
| 188 | muls_cast | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/muls_cast.md` | operation/vf_computation/composite_computation |
| 189 | clear_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/clear_spr.md` | operation/vf_computation/data_movement |
| 190 | create_addr_reg | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/create_addr_reg.md` | operation/vf_computation/data_movement |
| 191 | full | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/full.md` | operation/vf_computation/data_movement |
| 192 | gather | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/gather.md` | operation/vf_computation/data_movement |
| 193 | get_mask_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/get_mask_spr.md` | operation/vf_computation/data_movement |
| 194 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/index.md` | operation/vf_computation/data_movement |
| 195 | load | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load.md` | operation/vf_computation/data_movement |
| 196 | load_align | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_align.md` | operation/vf_computation/data_movement |
| 197 | load_unalign | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_unalign.md` | operation/vf_computation/data_movement |
| 198 | load_unalign_init | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_unalign_init.md` | operation/vf_computation/data_movement |
| 199 | load_unalign_pre | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_unalign_pre.md` | operation/vf_computation/data_movement |
| 200 | mem_bar | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/mem_bar.md` | operation/vf_computation/data_movement |
| 201 | move | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/move.md` | operation/vf_computation/data_movement |
| 202 | scatter | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/scatter.md` | operation/vf_computation/data_movement |
| 203 | store | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store.md` | operation/vf_computation/data_movement |
| 204 | store_align | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store_align.md` | operation/vf_computation/data_movement |
| 205 | store_unalign | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store_unalign.md` | operation/vf_computation/data_movement |
| 206 | store_unalign_post | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store_unalign_post.md` | operation/vf_computation/data_movement |
| 207 | unalign_reg_for_store | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/unalign_reg_for_store.md` | operation/vf_computation/data_movement |
| 208 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/index.md` | operation/vf_computation |
| 209 | and_ | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/and_.md` | operation/vf_computation/logical_computation |
| 210 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/index.md` | operation/vf_computation/logical_computation |
| 211 | not_ | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/not_.md` | operation/vf_computation/logical_computation |
| 212 | or_ | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/or_.md` | operation/vf_computation/logical_computation |
| 213 | shift_left | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/shift_left.md` | operation/vf_computation/logical_computation |
| 214 | shift_right | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/shift_right.md` | operation/vf_computation/logical_computation |
| 215 | xor | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/logical_computation/xor.md` | operation/vf_computation/logical_computation |
| 216 | create_mask | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/create_mask.md` | operation/vf_computation/mask_operations |
| 217 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/index.md` | operation/vf_computation/mask_operations |
| 218 | mask_gen_with_reg_tensor | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/mask_gen_with_reg_tensor.md` | operation/vf_computation/mask_operations |
| 219 | update_mask | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/update_mask.md` | operation/vf_computation/mask_operations |
| 220 | mask_reg | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_reg.md` | operation/vf_computation |
| 221 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/index.md` | operation/vf_computation/reduction |
| 222 | pair_reduce_sum | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/pair_reduce_sum.md` | operation/vf_computation/reduction |
| 223 | reduce_max | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_max.md` | operation/vf_computation/reduction |
| 224 | reduce_min | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_min.md` | operation/vf_computation/reduction |
| 225 | reduce_sum | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_sum.md` | operation/vf_computation/reduction |
| 226 | reg_tensor | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reg_tensor.md` | operation/vf_computation |
| 227 | arange | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/sorting_and_indexing/arange.md` | operation/vf_computation/sorting_and_indexing |
| 228 | histograms | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/sorting_and_indexing/histograms.md` | operation/vf_computation/sorting_and_indexing |
| 229 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/sorting_and_indexing/index.md` | operation/vf_computation/sorting_and_indexing |
| 230 | get_ctrl_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/get_ctrl_spr.md` | operation/vf_computation/special_reg_access |
| 231 | get_saturation_flag | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/get_saturation_flag.md` | operation/vf_computation/special_reg_access |
| 232 | get_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/get_spr.md` | operation/vf_computation/special_reg_access |
| 233 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/index.md` | operation/vf_computation/special_reg_access |
| 234 | reset_ctrl_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/reset_ctrl_spr.md` | operation/vf_computation/special_reg_access |
| 235 | set_ctrl_spr | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/set_ctrl_spr.md` | operation/vf_computation/special_reg_access |
| 236 | set_saturation_flag | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/special_reg_access/set_saturation_flag.md` | operation/vf_computation/special_reg_access |
| 237 | astype | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/astype.md` | operation/vf_computation/type_conversion |
| 238 | bit_cast | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/bit_cast.md` | operation/vf_computation/type_conversion |
| 239 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/index.md` | operation/vf_computation/type_conversion |
| 240 | truncate | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/truncate.md` | operation/vf_computation/type_conversion |
| 241 | BinType | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/BinType.md` | operation/vf_computation/types |
| 242 | CastLayout | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/CastLayout.md` | operation/vf_computation/types |
| 243 | DataCopyMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/DataCopyMode.md` | operation/vf_computation/types |
| 244 | DuplicatePos | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/DuplicatePos.md` | operation/vf_computation/types |
| 245 | HistType | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/HistType.md` | operation/vf_computation/types |
| 246 | IndexOrder | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/IndexOrder.md` | operation/vf_computation/types |
| 247 | LoadDist | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/LoadDist.md` | operation/vf_computation/types |
| 248 | MaskPattern | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/MaskPattern.md` | operation/vf_computation/types |
| 249 | MaskWidth | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/MaskWidth.md` | operation/vf_computation/types |
| 250 | MemBarMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/MemBarMode.md` | operation/vf_computation/types |
| 251 | MergeMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/MergeMode.md` | operation/vf_computation/types |
| 252 | PackPart | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/PackPart.md` | operation/vf_computation/types |
| 253 | SaturateMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/SaturateMode.md` | operation/vf_computation/types |
| 254 | SaturationFlagMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/SaturationFlagMode.md` | operation/vf_computation/types |
| 255 | SqueezeMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/SqueezeMode.md` | operation/vf_computation/types |
| 256 | StoreDist | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/StoreDist.md` | operation/vf_computation/types |
| 257 | VFRoundMode | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/VFRoundMode.md` | operation/vf_computation/types |
| 258 | index | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/types/index.md` | operation/vf_computation/types |

### A.3 SIMT-API（SIMT 标量/线程 API）（43 文档）

| # | 文档 | 路径 | 子类别 |
|---|------|------|--------|
| 1 | atomic_add | `docs/pypto_pro/api/SIMT-API/atomic/atomic_add.md` | atomic |
| 2 | atomic_and | `docs/pypto_pro/api/SIMT-API/atomic/atomic_and.md` | atomic |
| 3 | atomic_cas | `docs/pypto_pro/api/SIMT-API/atomic/atomic_cas.md` | atomic |
| 4 | atomic_dec | `docs/pypto_pro/api/SIMT-API/atomic/atomic_dec.md` | atomic |
| 5 | atomic_exch | `docs/pypto_pro/api/SIMT-API/atomic/atomic_exch.md` | atomic |
| 6 | atomic_inc | `docs/pypto_pro/api/SIMT-API/atomic/atomic_inc.md` | atomic |
| 7 | atomic_max | `docs/pypto_pro/api/SIMT-API/atomic/atomic_max.md` | atomic |
| 8 | atomic_min | `docs/pypto_pro/api/SIMT-API/atomic/atomic_min.md` | atomic |
| 9 | atomic_or | `docs/pypto_pro/api/SIMT-API/atomic/atomic_or.md` | atomic |
| 10 | atomic_sub | `docs/pypto_pro/api/SIMT-API/atomic/atomic_sub.md` | atomic |
| 11 | atomic_xor | `docs/pypto_pro/api/SIMT-API/atomic/atomic_xor.md` | atomic |
| 12 | index | `docs/pypto_pro/api/SIMT-API/atomic/index.md` | atomic |
| 13 | block_dim | `docs/pypto_pro/api/SIMT-API/execution/block_dim.md` | execution |
| 14 | block_idx | `docs/pypto_pro/api/SIMT-API/execution/block_idx.md` | execution |
| 15 | grid_dim | `docs/pypto_pro/api/SIMT-API/execution/grid_dim.md` | execution |
| 16 | index | `docs/pypto_pro/api/SIMT-API/execution/index.md` | execution |
| 17 | launch | `docs/pypto_pro/api/SIMT-API/execution/launch.md` | execution |
| 18 | linear_thread_idx | `docs/pypto_pro/api/SIMT-API/execution/linear_thread_idx.md` | execution |
| 19 | thread_idx | `docs/pypto_pro/api/SIMT-API/execution/thread_idx.md` | execution |
| 20 | index | `docs/pypto_pro/api/SIMT-API/index.md` | （根） |
| 21 | abs | `docs/pypto_pro/api/SIMT-API/scalar_compute/abs.md` | scalar_compute |
| 22 | cast | `docs/pypto_pro/api/SIMT-API/scalar_compute/cast.md` | scalar_compute |
| 23 | ceil | `docs/pypto_pro/api/SIMT-API/scalar_compute/ceil.md` | scalar_compute |
| 24 | cos | `docs/pypto_pro/api/SIMT-API/scalar_compute/cos.md` | scalar_compute |
| 25 | exp | `docs/pypto_pro/api/SIMT-API/scalar_compute/exp.md` | scalar_compute |
| 26 | exp2 | `docs/pypto_pro/api/SIMT-API/scalar_compute/exp2.md` | scalar_compute |
| 27 | floor | `docs/pypto_pro/api/SIMT-API/scalar_compute/floor.md` | scalar_compute |
| 28 | fma | `docs/pypto_pro/api/SIMT-API/scalar_compute/fma.md` | scalar_compute |
| 29 | index | `docs/pypto_pro/api/SIMT-API/scalar_compute/index.md` | scalar_compute |
| 30 | isinf | `docs/pypto_pro/api/SIMT-API/scalar_compute/isinf.md` | scalar_compute |
| 31 | isnan | `docs/pypto_pro/api/SIMT-API/scalar_compute/isnan.md` | scalar_compute |
| 32 | log | `docs/pypto_pro/api/SIMT-API/scalar_compute/log.md` | scalar_compute |
| 33 | log1p | `docs/pypto_pro/api/SIMT-API/scalar_compute/log1p.md` | scalar_compute |
| 34 | log2 | `docs/pypto_pro/api/SIMT-API/scalar_compute/log2.md` | scalar_compute |
| 35 | max | `docs/pypto_pro/api/SIMT-API/scalar_compute/max.md` | scalar_compute |
| 36 | min | `docs/pypto_pro/api/SIMT-API/scalar_compute/min.md` | scalar_compute |
| 37 | rint | `docs/pypto_pro/api/SIMT-API/scalar_compute/rint.md` | scalar_compute |
| 38 | round | `docs/pypto_pro/api/SIMT-API/scalar_compute/round.md` | scalar_compute |
| 39 | rsqrt | `docs/pypto_pro/api/SIMT-API/scalar_compute/rsqrt.md` | scalar_compute |
| 40 | sin | `docs/pypto_pro/api/SIMT-API/scalar_compute/sin.md` | scalar_compute |
| 41 | sqrt | `docs/pypto_pro/api/SIMT-API/scalar_compute/sqrt.md` | scalar_compute |
| 42 | tanh | `docs/pypto_pro/api/SIMT-API/scalar_compute/tanh.md` | scalar_compute |
| 43 | trunc | `docs/pypto_pro/api/SIMT-API/scalar_compute/trunc.md` | scalar_compute |

### A.4 Utils-API（调试与语法糖工具）（10 文档）

| # | 文档 | 路径 | 子类别 |
|---|------|------|--------|
| 1 | dump_data | `docs/pypto_pro/api/Utils-API/debugging/dump_data.md` | debugging |
| 2 | index | `docs/pypto_pro/api/Utils-API/debugging/index.md` | debugging |
| 3 | printf | `docs/pypto_pro/api/Utils-API/debugging/printf.md` | debugging |
| 4 | pto_assert | `docs/pypto_pro/api/Utils-API/debugging/pto_assert.md` | debugging |
| 5 | trap | `docs/pypto_pro/api/Utils-API/debugging/trap.md` | debugging |
| 6 | index | `docs/pypto_pro/api/Utils-API/index.md` | （根） |
| 7 | const | `docs/pypto_pro/api/Utils-API/python_syntax_sugar/const.md` | python_syntax_sugar |
| 8 | index | `docs/pypto_pro/api/Utils-API/python_syntax_sugar/index.md` | python_syntax_sugar |
| 9 | max | `docs/pypto_pro/api/Utils-API/python_syntax_sugar/max.md` | python_syntax_sugar |
| 10 | min | `docs/pypto_pro/api/Utils-API/python_syntax_sugar/min.md` | python_syntax_sugar |

### A.5 API 根目录（1 文档）

| # | 文档 | 路径 |
|---|------|------|
| 1 | index | `docs/pypto_pro/api/index.md` |

---

## §B 官方指定算子样例

> **重要**：以下为官方明确允许 agent 开发算子时参考的算子代码，是**唯一的算子写法参考来源**。`$PYPTO_DEVKIT_DIR/pro_ops/` 下其余文件**不得**作为样例参考或索引对象。

**清单一致性核对**：清单期望 13 个；缓存实际 13 个。其中 12 个完全一致；`pro_ops/matmul/test_matmul_8K_example.py` 存在大小写不一致（清单 `8K` vs 缓存 `8k`，文件实际存在，非 softmax 相关，记为非阻断注意项，见 EXPLORE_REPORT §8.2）。softmax 官方样例 `pro_ops/vf_api/test_softmax_tile_group_vf.py` 完全一致。

| # | 算子名称 | 缓存相对路径 | 类型 | 描述 |
|---|---------|-------------|------|------|
| 1 | add | `pro_ops/element_wise/test_add.py` | elementwise |  |
| 2 | matmul_8k_example | `pro_ops/matmul/test_matmul_8K_example.py` | matmul 入门 |  |
| 3 | matmul_perf_asw_4k | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset.py` | matmul 性能 |  |
| 4 | matmul_perf_asw_8k_k128 | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset.py` | matmul 性能 |  |
| 5 | matmul_perf_asw_4k_dynamic | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | matmul 性能（动态轴） |  |
| 6 | matmul_perf_asw_8k_k128_dynamic | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset_dynamic.py` | matmul 性能（动态轴） |  |
| 7 | fa_perf_tkv_preload | `pro_ops/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py` | FlashAttention 生产级 |  |
| 8 | fa_tilingkey_attn_mask | `pro_ops/fa/test_fa_tilingkey_attn_mask.py` | FlashAttention 教学 |  |
| 9 | fa_with_mask | `pro_ops/fa/test_fa_with_mask.py` | FlashAttention 性能（mask + NBuffer + auto_mutex） |  |
| 10 | flex_attention | `pro_ops/fa/test_flex_attention.py` | FlexAttention 性能 |  |
| 11 | quant_lightning_indexer_vf | `pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py` | VF TopK |  |
| 12 | layernorm_tile_group_vf | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | VF LayerNorm |  |
| 13 | softmax_tile_group_vf | `pro_ops/vf_api/test_softmax_tile_group_vf.py` | VF Softmax |  |

---

## §C 教程与设计指南（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials`）

> 扫描 `tutorials/` 递归获取所有 `.md` 文档，共 **40** 个。

| # | 来源 | 文档 | 路径 |
|---|------|------|------|
| 1 | tutorials | aclnn_operator_project_development | `docs/pypto_pro/tutorials/advanced_programming/aclnn_operator_project_development.md` |
| 2 | tutorials | advanced_ai_core_programming_model | `docs/pypto_pro/tutorials/advanced_programming/advanced_ai_core_programming_model.md` |
| 3 | tutorials | ai_framework_operator_adaptation | `docs/pypto_pro/tutorials/advanced_programming/ai_framework_operator_adaptation.md` |
| 4 | tutorials | aot_compilation_optimization | `docs/pypto_pro/tutorials/advanced_programming/aot_compilation_optimization.md` |
| 5 | tutorials | auto_parallel_pipeline | `docs/pypto_pro/tutorials/advanced_programming/auto_parallel_pipeline.md` |
| 6 | tutorials | hardware_implementation | `docs/pypto_pro/tutorials/advanced_programming/hardware_implementation.md` |
| 7 | tutorials | index | `docs/pypto_pro/tutorials/advanced_programming/index.md` |
| 8 | tutorials | operator_graph_integration_development | `docs/pypto_pro/tutorials/advanced_programming/operator_graph_integration_development.md` |
| 9 | tutorials | superkernel | `docs/pypto_pro/tutorials/advanced_programming/superkernel.md` |
| 10 | tutorials | functional_debugging | `docs/pypto_pro/tutorials/debugging_and_optimization/functional_debugging.md` |
| 11 | tutorials | index | `docs/pypto_pro/tutorials/debugging_and_optimization/index.md` |
| 12 | tutorials | performance_optimization | `docs/pypto_pro/tutorials/debugging_and_optimization/performance_optimization.md` |
| 13 | tutorials | index | `docs/pypto_pro/tutorials/index.md` |
| 14 | tutorials | introduction | `docs/pypto_pro/tutorials/introduction.md` |
| 15 | tutorials | JIT_compilation | `docs/pypto_pro/tutorials/operator_development/compilation_and_execution/JIT_compilation.md` |
| 16 | tutorials | index | `docs/pypto_pro/tutorials/operator_development/compilation_and_execution/index.md` |
| 17 | tutorials | offline_binary_compilation | `docs/pypto_pro/tutorials/operator_development/compilation_and_execution/offline_binary_compilation.md` |
| 18 | tutorials | index | `docs/pypto_pro/tutorials/operator_development/index.md` |
| 19 | tutorials | kernel_function | `docs/pypto_pro/tutorials/operator_development/kernel_function.md` |
| 20 | tutorials | Cube_matrix_computation | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Cube_matrix_computation.md` |
| 21 | tutorials | Python_programming_overview | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Python_programming_overview.md` |
| 22 | tutorials | Reg_vector_computation | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Reg_vector_computation.md` |
| 23 | tutorials | Tile_vector_computation | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Tile_vector_computation.md` |
| 24 | tutorials | TilingData | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/TilingData.md` |
| 25 | tutorials | index | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/index.md` |
| 26 | tutorials | multi_core_partitioning_and_Tiling | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md` |
| 27 | tutorials | tail_block_handling | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tail_block_handling.md` |
| 28 | tutorials | tiling_key | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tiling_key.md` |
| 29 | tutorials | abstract_hardware_architecture | `docs/pypto_pro/tutorials/programming_paradigm/abstract_hardware_architecture.md` |
| 30 | tutorials | index | `docs/pypto_pro/tutorials/programming_paradigm/index.md` |
| 31 | tutorials | programming_paradigm_overview | `docs/pypto_pro/tutorials/programming_paradigm/programming_paradigm_overview.md` |
| 32 | tutorials | simt_programming | `docs/pypto_pro/tutorials/programming_paradigm/simt_programming.md` |
| 33 | tutorials | Add_operator | `docs/pypto_pro/tutorials/quick_start/SIMD/Add_operator.md` |
| 34 | tutorials | CV_fused_operator | `docs/pypto_pro/tutorials/quick_start/SIMD/CV_fused_operator.md` |
| 35 | tutorials | HelloWorld | `docs/pypto_pro/tutorials/quick_start/SIMD/HelloWorld.md` |
| 36 | tutorials | Matmul_operator | `docs/pypto_pro/tutorials/quick_start/SIMD/Matmul_operator.md` |
| 37 | tutorials | index | `docs/pypto_pro/tutorials/quick_start/SIMD/index.md` |
| 38 | tutorials | Add_operator | `docs/pypto_pro/tutorials/quick_start/SIMT/Add_operator.md` |
| 39 | tutorials | index | `docs/pypto_pro/tutorials/quick_start/SIMT/index.md` |
| 40 | tutorials | index | `docs/pypto_pro/tutorials/quick_start/index.md` |
