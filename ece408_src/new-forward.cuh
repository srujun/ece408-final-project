
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define MAT_TILE_DIM  16
#define MAX_THREADS   1024
#define ceil(num,denom) (((num)-1) / (denom) + 1)

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


// Compute C = A * B
__global__ void matrixMultiplyShared(
    float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns
)
{
    __shared__ float subTileA[MAT_TILE_DIM][MAT_TILE_DIM];
    __shared__ float subTileB[MAT_TILE_DIM][MAT_TILE_DIM];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    // int batch = blockIdx.z;

    float val = 0;

    int numIters = ceil(numAColumns, MAT_TILE_DIM);
    for (int i = 0; i < numIters; i++)
    {
        int idxArow = row;
        int idxAcol = MAT_TILE_DIM * i + threadIdx.x;
        int idxBrow = MAT_TILE_DIM * i + threadIdx.y;
        int idxBcol = col;

        if (idxArow < numARows && idxAcol < numAColumns)
            subTileA[threadIdx.y][threadIdx.x] = A[numAColumns * idxArow + idxAcol];
        else
            subTileA[threadIdx.y][threadIdx.x] = 0;

        if (idxBrow < numBRows && idxBcol < numBColumns)
            subTileB[threadIdx.y][threadIdx.x] = B[numBColumns * idxBrow + idxBcol];
        else
            subTileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        if (row < numCRows && col < numCColumns)
        {
            for (int j = 0; j < MAT_TILE_DIM; j++)
            val += subTileA[threadIdx.y][j] * subTileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < numCRows && col < numCColumns)
    {
        int idxC = numCColumns * row + col;
        C[idxC] = val;
    }
}


__global__ void forward_kernel(
    float *y, const float *x, const float *k,
    const int B, const int M, const int C, const int H, const int W, const int K
)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    #define y4d(i3,i2,i1,i0) y[(i3)*(M*H_out*W_out) + (i2)*(H_out*W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3)*(C*H*W) + (i2)*(H*W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3)*(C*K*K) + (i2)*(K*K) + (i1)*(K) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int batch = blockIdx.x;
    int fmap = blockIdx.y;
    int xval = threadIdx.x;
    int yval = threadIdx.y;

    float acc = 0;

    for (int channel = 0; channel < C; ++channel)
    {
        for (int k_y = 0; k_y < K; ++k_y)
        {
            for (int k_x = 0; k_x < K; ++k_x)
            {
                acc += x4d(batch, channel, yval + k_y, xval + k_x) *
                       k4d(fmap, channel, k_y, k_x);
            }
        }
    }

    y4d(batch, fmap, yval, xval) = acc;

    #undef y4d
    #undef x4d
    #undef k4d
}


__global__ void unroll_kernel(
    const float* x, float* x_unroll,
    const int H_in, const int W_in, const int H_out, const int W_out,
    const int H_xu, const int W_xu,
    const int B, const int C, const int K
)
{
    // x_unroll will has dim: [B, C*K*K, H_out*W_out]
    // CHECK_EQ(H_xu, C*K*K);
    // CHECK_EQ(W_xu, H_out*W_out);

    #define x4d(i3,i2,i1,i0) x[(i3)*(C*H_in*W_in) + (i2)*(H_in*W_in) + (i1)*(W_in) + i0]
    #define xu3d(i2,i1,i0) x_unroll[(i2)*(H_xu*W_xu) + (i1)*(H_xu) + i0]
    // used to be (i1)*(W_xu) + i0

    int batch = blockIdx.x;
    int tId = blockDim.y * blockIdx.y + threadIdx.x;

    if (tId < C * W_xu)
    {
        int channel = tId / W_xu;
        int unroll_col = tId % W_xu;

        // indices in x
        int x_row = unroll_col / W_out;
        int x_col = unroll_col % W_out;

        // indices in x_unroll
        int h_unroll = x_row * W_out + x_col;
        int w_base = channel * K * K;

        for (int k_y = 0; k_y < K; ++k_y)
        {
            for (int k_x = 0; k_x < K; ++k_x)
            {
                int w_unroll = w_base + k_y * K + k_x;
                // TODO: swap index names for xu3d
                xu3d(batch, h_unroll, w_unroll) = x4d(batch, channel, x_row + k_y, x_col + k_x);
            }
        }
    }
}


void printDim3(char* dimType, dim3 toPrint)
{
    fprintf(stdout, "%s(%d, %d, %d)\n", dimType, toPrint.x, toPrint.y, toPrint.z);
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
    // cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions
    const int batches = x.shape_[0];
    const int fmaps = w.shape_[0];
    const int channels = w.shape_[1];
    const int h_in = x.shape_[2];
    const int w_in = x.shape_[3];
    const int k_dim = w.shape_[3];

    const int h_out = h_in - k_dim + 1;
    const int w_out = w_in - k_dim + 1;

    // will create 3 matrices: wts, x_unroll, newy
    const int wts_rows = fmaps;
    const int wts_cols = channels * k_dim * k_dim;
    const int x_unroll_rows = channels * k_dim * k_dim;
    const int x_unroll_cols = h_out * w_out;
    // const int newy_rows = fmaps;
    // const int newy_cols = h_out * w_out;

    Tensor<gpu, 2, float> wts = Tensor<gpu, 2, float>(w.dptr_, Shape2(wts_rows, wts_cols));
    Tensor<gpu, 3, float> x_unroll(Shape3(batches, x_unroll_rows, x_unroll_cols));
    // Tensor<gpu, 2, float> newy = Tensor<gpu, 2, float>(y.dptr_, Shape2(newy_rows, newy_cols));

    AllocSpace(&x_unroll);

    /****************************** UNROLLING *********************************/

    const int threads_per_img = channels * h_out * w_out;
    const int blocks_per_img = ceil(threads_per_img, MAX_THREADS);
    dim3 grid_unroll(batches, blocks_per_img, 1);
    dim3 block_unroll(MAX_THREADS, 1, 1);

    fprintf(stdout, "\nUnroll Kernel:\n");
    printDim3((char *)"grid", grid_unroll);
    printDim3((char *)"block", block_unroll);

    unroll_kernel<<<grid_unroll, block_unroll>>>(
        x.dptr_, x_unroll.dptr_,
        h_in, w_in, h_out, w_out,
        x_unroll_rows, x_unroll_cols,
        batches, channels, k_dim
    );

    /**************************************************************************/

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    /*************************** MATRIX MULTIPLY ******************************/

    const int numCRows = wts_rows;
    const int numCColumns = x_unroll_cols;
    dim3 grid_mm(ceil(numCColumns, MAT_TILE_DIM), ceil(numCRows, MAT_TILE_DIM), 1);//batches);
    dim3 block_mm(MAT_TILE_DIM, MAT_TILE_DIM, 1);

    fprintf(stdout, "\nMatrix Multiply Kernel:\n");
    printDim3((char *)"grid", grid_mm);
    printDim3((char *)"block", block_mm);

    for(int batch = 0; batch < batches; batch++)
    {
        matrixMultiplyShared<<<grid_mm, block_mm>>>(
            wts.dptr_, &(x_unroll.dptr_[batch * (x_unroll_rows * x_unroll_cols)]), &(y.dptr_[batch * (channels * h_out * w_out)]),
            wts_rows, wts_cols,
            x_unroll_rows, x_unroll_cols,
            wts_rows, x_unroll_cols
        );
    }
    // A -> wts
    // B -> x_unroll
    // C -> y
    // matrixMultiplyShared<<<grid_mm, block_mm>>>(
    //     wts.dptr_, x_unroll.dptr_, y.dptr_,
    //     wts_rows, wts_cols,
    //     x_unroll_rows, x_unroll_cols,
    //     wts_rows, x_unroll_cols
    // );

    /**************************************************************************/

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    FreeSpace(&x_unroll);
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