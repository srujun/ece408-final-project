
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define MAT_TILE_DIM  32
#define MAX_THREADS   1024
#define ceil(num,denom) (((num)-1) / (denom) + 1)
#define CONST_M       50
#define CONST_K       5
#define CONST_C       1

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// __constant__ float weights[CONST_M][CONST_K][CONST_K];
__constant__ float weights[50][5][5];

__global__ void unroll_kernel(
    const float* x, float* y/*,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int H_xu, const int W_xu,
    const int B, const int C, const int M, const int K*/
)
{
    #define x4d(i3,i2,i1,i0) x[(i3)*(CONST_C*28*28) + (i2)*(28*28) + (i1)*(28) + i0]
    #define y4d(i3,i2,i1,i0) y[(i3)*(CONST_M*24*24) + (i2)*(24*24) + (i1)*(24) + i0]

    __shared__ float xs[28][28];

    const unsigned int batch = blockIdx.x;
    // const int fmap = blockIdx.y;
    // const int channel = 0;

    const unsigned int x_col = threadIdx.x;
    const unsigned int x_row = threadIdx.y;
    const unsigned int y_row = x_row;
    const unsigned int y_col = x_col;

    unsigned int fmap, k_y, k_x;

    /* Load x into shared memory xs */
    xs[x_row][x_col] = x4d(batch, 0, x_row, x_col);

    __syncthreads();

    if (y_row < 24 && y_col < 24)
    {
        for(fmap = 0; fmap < 25; ++fmap)
        {
            float val1 = 0.0;
            float val2 = 0.0;

            for (k_y = 0; k_y < 5; k_y++)
            {
                for (k_x = 0; k_x < 5; k_x++)
                {
                    // const int xs_row = y_row + k_y;
                    // const int xs_col = y_col + k_x;
                    // float xs_val = xs[xs_row][xs_col];
                    val1 += weights[fmap   ][k_y][k_x] * xs[y_row + k_y][y_col + k_x];
                    val2 += weights[fmap+25][k_y][k_x] * xs[y_row + k_y][y_col + k_x];
                }
            }

            y4d(batch, fmap   , y_row, y_col) = val1;
            y4d(batch, fmap+25, y_row, y_col) = val2;
        }
    }

    #undef y4d
    #undef x4d
}


// void printDim3(char* dimType, dim3 toPrint)
// {
//     fprintf(stdout, "%s(%d, %d, %d)\n", dimType, toPrint.x, toPrint.y, toPrint.z);
// }


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

    // const int h_out = h_in - k_dim + 1;
    // const int w_out = w_in - k_dim + 1;

    // const int wts_rows = fmaps;
    // const int wts_cols = channels * k_dim * k_dim;
    // const int H_xu = channels * k_dim * k_dim;
    // const int W_xu = h_out * w_out;

    // cudaMemcpyToSymbol(weights, w.dptr_, wts_rows * wts_cols * sizeof(float));
    cudaMemcpyToSymbol(weights, w.dptr_, 50*5*5 * sizeof(float));

    /****************************** UNROLLING *********************************/

    // dim3 grid_unroll(batches, 1, 1);
    // dim3 block_unroll(w_in, h_in, 1);
    dim3 grid_unroll(batches, 1, 1);
    dim3 block_unroll(28, 28, 1);

    // fprintf(stdout, "\nUnroll Kernel:\n");
    // printDim3((char *)"grid", grid_unroll);
    // printDim3((char *)"block", block_unroll);

    // size_t xshared_size = sizeof(float) * h_in * w_in;

    unroll_kernel<<<grid_unroll, block_unroll, 0, s>>>(
        x.dptr_, y.dptr_/*,
        h_in, w_in,
        h_out, w_out,
        H_xu, W_xu,
        batches, channels, fmaps, k_dim*/
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