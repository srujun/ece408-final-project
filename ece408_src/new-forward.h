
#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// This function is called by new-inl.h
// Any code you write should be executed by this function
template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */


    const int batch_size = x.shape_[0];
    const int out_fmap_size = y.shape_[1];
    const int in_fmap_size = x.shape_[1];
    const int img_height = x.shape_[2];
    const int img_width = x.shape_[3];
    const int kernel_size = k.shape_[3];

    for (int batch = 0; batch < batch_size; ++batch) {
        // loop through the iage batches
        for (int out_fmap = 0; out_fmap < out_fmap_size; ++out_fmap) {
            // loop through the output feature maps

            for (int yval = 0; yval < img_height; ++yval) {
                for (int xval = 0; xval < img_width; ++xval) {
                    // loop through each element of the input image

                    y[batch][out_fmap][yval][xval] = 0;

                    for (int in_fmap = 0; in_fmap < in_fmap_size; ++in_fmap) {
                        // loop through the output feature maps

                        for (int k_y = 0; k_y < kernel_size; ++k_y) {
                            for (int k_x = 0; k_x < kernel_size; ++k_x) {
                                // loop through the kernel elements
                                y[batch][out_fmap][yval][xval] +=
                                    x[batch][in_fmap][yval + k_y][xval + k_x] *
                                    k[out_fmap][in_fmap][k_y][k_x];
                            }
                        }
                    }
                }
            }
        }
    }

}
}
}

#endif