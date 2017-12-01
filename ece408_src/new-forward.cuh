
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


__global__ void forward_kernel(float *y, const float *x, const float *k,
                               const int B, const int M, const int C,
                               const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    // #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    // #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    // #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    // const int batches = B;
    // const int fmaps = M;
    const int channels = C;
    const int img_height = H - K + 1;
    const int img_width = W - K + 1;
    const int kernel_size = K;

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * img_height * img_width) + (i2)*(img_height * img_width) + (i1)*(img_width) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
    int batch = blockIdx.x;
    int fmap = blockIdx.y;
    int xval = threadIdx.x;
    int yval = threadIdx.y;

    float acc = 0;

    for (int channel = 0; channel < channels; ++channel)
    {
        for (int k_y = 0; k_y < kernel_size; ++k_y)
        {
            for (int k_x = 0; k_x < kernel_size; ++k_x)
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


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                         const mshadow::Tensor<gpu, 4, float> &x,
                         const mshadow::Tensor<gpu, 4, float> &w)
{
    // Use mxnet's CHECK_EQ to do assertions.

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions
    const int batches = x.shape_[0];
    const int fmaps = y.shape_[1];
    const int channels = x.shape_[1];
    const int img_height = x.shape_[2];
    const int img_width = x.shape_[3];
    const int kernel_size = w.shape_[3];

    fprintf(stdout, "batches = %d\n", batches);
    fprintf(stdout, "fmaps = %d\n", fmaps);
    fprintf(stdout, "channels = %d\n", channels);
    fprintf(stdout, "img_height = %d\n", img_height);
    fprintf(stdout, "img_width = %d\n", img_width);
    fprintf(stdout, "kernel_size = %d\n", kernel_size);

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);

    // const int BLOCK_SIZE = 16;
    const int grid_width = (img_width - kernel_size + 1);// / BLOCK_SIZE;
    const int grid_height = (img_height - kernel_size + 1);// / BLOCK_SIZE;

    dim3 dimGrid(batches, fmaps, 1);
    dim3 dimBlock(grid_width, grid_height, 1);

    fprintf(stdout, "\n");
    fprintf(stdout, "grid(%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    fprintf(stdout, "block(%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);

    // Call the kernel
    // forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    forward_kernel<<<dimGrid, dimBlock, 0, s>>>(
        y.dptr_, x.dptr_, w.dptr_,
        batches, fmaps, channels,
        img_height, img_width, kernel_size
    );

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif