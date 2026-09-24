#include "kernel.cpp"
__global__ AICORE void softmax_tile_group_kernel(__gm__ half* x, __gm__ half* y, int32_t __pypto_dyn_x_0, int32_t __pypto_dyn_x_1, int32_t __pypto_dyn_y_0, int32_t __pypto_dyn_y_1)
{
    softmax_tile_group_kernel_impl(x, y, __pypto_dyn_x_0, __pypto_dyn_x_1, __pypto_dyn_y_0, __pypto_dyn_y_1);
}
extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y, int32_t __pypto_dyn_x_0, int32_t __pypto_dyn_x_1, int32_t __pypto_dyn_y_0, int32_t __pypto_dyn_y_1)
{{
    softmax_tile_group_kernel<<<blockDim, nullptr, stream>>>((half *)x, (half *)y, __pypto_dyn_x_0, __pypto_dyn_x_1, __pypto_dyn_y_0, __pypto_dyn_y_1);
}}
