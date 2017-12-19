
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define MAT_TILE_DIM  32
#define MAX_THREADS   1024
#define ceil(num,denom) (((num)-1) / (denom) + 1)

#define CONST_M       50
#define CONST_K       5
#define CONST_C       1
#define H_IN         28
#define W_IN         28
#define H_OUT        24
#define W_OUT        24

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float weights[CONST_M][CONST_K][CONST_K];

__global__ void forward_kernel(
    const float* x, float* y
)
{
    #define x4d(i3,i2,i1,i0) x[(i3)*(CONST_C*H_IN*W_IN) + (i2)*(H_IN*W_IN) + (i1)*(W_IN) + i0]
    #define y4d(i3,i2,i1,i0) y[(i3)*(CONST_M*H_OUT*W_OUT) + (i2)*(H_OUT*W_OUT) + (i1)*(W_OUT) + i0]
    #define xs2d(i1,i0) xs[(i1)*(W_IN) + i0]

    extern __shared__ float xs[];

    const unsigned int batch = blockIdx.x;

    const unsigned int x_col = threadIdx.x;
    const unsigned int x_row = threadIdx.y;
    const unsigned int y_row = x_row;
    const unsigned int y_col = x_col;

    unsigned int fmap, k_y, k_x;

    /* Load x into shared memory xs */
    xs2d(x_row, x_col) = x4d(batch, 0, x_row, x_col);

    __syncthreads();

    if (y_row < H_OUT && y_col < W_OUT)
    {
        for(fmap = 0; fmap < CONST_M; ++fmap)
        {
            float val = 0.0;

            for (k_y = 0; k_y < CONST_K; k_y++)
            {
                for (k_x = 0; k_x < CONST_K; k_x++)
                {
                    val += weights[fmap][k_y][k_x] * xs2d(y_row + k_y, y_col + k_x);
                }
            }

            y4d(batch, fmap, y_row, y_col) = val;
        }
    }

    #undef xs2d
    #undef y4d
    #undef x4d
}


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called,
   so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(
    mshadow::Tensor<gpu, 4, float> &y,
    const mshadow::Tensor<gpu, 4, float> &x,
    const mshadow::Tensor<gpu, 4, float> &w
)
{
    using namespace mshadow;

    // Use mxnet's CHECK_EQ to do assertions.
    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions
    const int batches = x.shape_[0];
    // const int fmaps = w.shape_[0];
    // const int channels = w.shape_[1];
    // const int h_in = x.shape_[2];
    // const int w_in = x.shape_[3];
    // const int k_dim = w.shape_[3];

    cudaMemcpyToSymbol(weights, w.dptr_, CONST_M*CONST_K*CONST_K * sizeof(float));

    /******************************** KERNEL **********************************/

    dim3 gridDim(batches, 1, 1);
    dim3 blockDim(W_IN, H_IN, 1);

    size_t xshared_size = sizeof(float) * H_IN * W_IN;

    forward_kernel<<<gridDim, blockDim, xshared_size, s>>>(
        x.dptr_, y.dptr_
    );

    /**************************************************************************/

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}


/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(
    mshadow::Tensor<gpu, 4, DType> &y,
    const mshadow::Tensor<gpu, 4, DType> &x,
    const mshadow::Tensor<gpu, 4, DType> &w
)
{
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif