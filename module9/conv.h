#pragma once

#include <cublas_v2.h>

constexpr int NUM_FILTERS = 2;
constexpr int KERNEL_SIZE = 3;

static const float SOBEL_FILTERS[NUM_FILTERS][9] = {
    {-1, 0, 1, -2, 0, 2, -1, 0, 1}, // Sobel X
    {-1, -2, -1, 0, 0, 0, 1, 2, 1}  // Sobel Y
};

// Geometry of a convolution.
struct ConvShape
{
    int width = 0;  // input width
    int height = 0; // input height
    int num_filters =
        0; // number of filters being applied as part of the convolution
    int kernel_size = 0; // the filter side length
    int out_w = 0;       // output image width
    int out_h = 0;       // output image height

    ConvShape() = default;
    ConvShape(int w, int h);

    // The number of elements touched by each application of the kernel, which
    // corresponds to the row length of the im2col matrix call this quantity P
    int patchElements() const;

    // The number of elements in the output image, which corresponds to the
    // column length of the im2col matrix call this quantity N
    int numOutputElements() const;
};

// an im2col implementation that helps flatten the image into a matrix for GEMM
void im2col(const float *d_img, float *d_cols, ConvShape s);

// calls cublasSgemm on the im2col output to perform the convolution
void gemmConv(cublasHandle_t handle, const float *d_w, const float *d_cols,
              float *d_out, ConvShape s);
