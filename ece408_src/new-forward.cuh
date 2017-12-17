
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

__constant__ float weights[CONST_M][CONST_C * CONST_K * CONST_K];

// Compute C = A * B
__global__ void matrixMultiplyShared(
    float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns
)
{
    // __shared__ float subTileA[MAT_TILE_DIM][MAT_TILE_DIM];
    __shared__ float subTileB[MAT_TILE_DIM][MAT_TILE_DIM];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int batch = blockIdx.z;

    float val = 0;

    int numIters = ceil(numAColumns, MAT_TILE_DIM);
    for (int i = 0; i < numIters; i++)
    {
        int idxArow = row;
        // int idxAcol = MAT_TILE_DIM * i + threadIdx.x;
        int idxBrow = MAT_TILE_DIM * i + threadIdx.y;
        int idxBcol = col;

        int idxCONSTcol = MAT_TILE_DIM * i;

        // if (idxArow < numARows && idxAcol < numAColumns)
        //     subTileA[threadIdx.y][threadIdx.x] = A[numAColumns * idxArow + idxAcol];
        // else
        //     subTileA[threadIdx.y][threadIdx.x] = 0;

        if (idxBrow < numBRows && idxBcol < numBColumns)
            subTileB[threadIdx.y][threadIdx.x] = B[(numBRows * numBColumns) * batch + numBColumns * idxBrow + idxBcol];
        else
            subTileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        if (row < numCRows && col < numCColumns)
        {
            for (int j = 0; j < MAT_TILE_DIM; j++)
                // val += subTileA[threadIdx.y][j] * subTileB[j][threadIdx.x];
                val += weights[idxArow][idxCONSTcol + j] * subTileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < numCRows && col < numCColumns)
    {
        int idxC = (numCRows * numCColumns) * batch + numCColumns * row + col;
        C[idxC] = val;
    }
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
    #define xu3d(i2,i1,i0) x_unroll[(i2)*(H_xu*W_xu) + (i1)*(W_xu) + i0]

    int batch = blockIdx.y;
    int tId = blockDim.x * blockIdx.x + threadIdx.x;

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
                xu3d(batch, w_unroll, h_unroll) = x4d(batch, channel, x_row + k_y, x_col + k_x);
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
    const int H_xu = channels * k_dim * k_dim;
    const int W_xu = h_out * w_out;
    // const int newy_rows = fmaps;
    // const int newy_cols = h_out * w_out;

    // Tensor<gpu, 2, float> wts = Tensor<gpu, 2, float>(w.dptr_, Shape2(wts_rows, wts_cols));
    Tensor<gpu, 3, float> x_unroll(Shape3(W_xu, H_xu, batches));
    // Tensor<gpu, 2, float> newy = Tensor<gpu, 2, float>(y.dptr_, Shape2(newy_rows, newy_cols));

    AllocSpace(&x_unroll);

    /****************************** UNROLLING *********************************/

    const int threads_per_img = channels * h_out * w_out;
    const int blocks_per_img = ceil(threads_per_img, MAX_THREADS);
    dim3 grid_unroll(blocks_per_img, batches, 1);
    dim3 block_unroll(MAX_THREADS, 1, 1);

    fprintf(stdout, "\nUnroll Kernel:\n");
    printDim3((char *)"grid", grid_unroll);
    printDim3((char *)"block", block_unroll);

    unroll_kernel<<<grid_unroll, block_unroll>>>(
        x.dptr_, x_unroll.dptr_,
        h_in, w_in, h_out, w_out,
        H_xu, W_xu,
        batches, channels, k_dim
    );

    /**************************************************************************/

    cudaMemcpyToSymbol(weights, w.dptr_, wts_rows * wts_cols * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    /*************************** MATRIX MULTIPLY ******************************/

    const int numCRows = wts_rows;
    const int numCColumns = W_xu;
    dim3 grid_mm(ceil(numCColumns, MAT_TILE_DIM), ceil(numCRows, MAT_TILE_DIM), batches);
    dim3 block_mm(MAT_TILE_DIM, MAT_TILE_DIM, 1);

    fprintf(stdout, "\nMatrix Multiply Kernel:\n");
    printDim3((char *)"grid", grid_mm);
    printDim3((char *)"block", block_mm);

    // A -> wts
    // B -> x_unroll
    // C -> y
    matrixMultiplyShared<<<grid_mm, block_mm>>>(
        w.dptr_, x_unroll.dptr_, y.dptr_,
        wts_rows, wts_cols,
        H_xu, W_xu,
        numCRows, numCColumns
    );

    /**************************************************************************/

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.

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