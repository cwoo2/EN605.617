#pragma once

#include "conv.h"
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <vector>

// Generate an image with a gradient, a filled disc, and some noise.
std::vector<uint8_t> generateImage(int width, int height);

// Writes a P5 PGM (basically 8-bit grayscale image) to the given path.
void writePgm(const char *path, const uint8_t *img, int width, int height);

// Scale a float PGM into 8-bit values and write to disk.
void saveFloatMap(const float *map, int width, int height, const char *path);

// Print performance report
void report(const ConvShape &s, float ms_cols, float ms_gemm, float ms_npp);

struct Context
{
    ConvShape shape;
    Npp8u *d_raw = nullptr;  // holds the raw image
    Npp8u *d_blur = nullptr; // holds the blurred image
    float *d_img =
        nullptr; // holds the blurred image converted to float for GEMM
    float *d_cols = nullptr; // holds the im2col output, P x N (P = patch size)
    float *d_filters = nullptr; // filter bank, P x F (F = number of filters)
    float *d_out = nullptr; // feature maps, N x F (N = number of output pixels)
    float *d_mag = nullptr; // edge magnitudes, (N elements)
    float *d_interleaved =
        nullptr; // Interleaved Sobel X and Y convolution results (2N elements)
    Npp8u *d_scratch = nullptr; // NPP max reduction scratch buffer
    Npp32f *d_max = nullptr;    // NPP max reduction result
    cublasHandle_t handle = nullptr;

    // Allocate device memory and create the cuBLAS handle.
    void allocate(int width, int height);

    // Free all device memory and destroy the cuBLAS handle.
    void release();

    // Blur the raw image and convert to float.
    void nppBlurAndConvert(const std::vector<uint8_t> &img);

    // Compute the magnitude of the edges by combining the Sobel X and Y
    // results.
    void computeMagnitude();

    // Normalize the intensities and save the edges to disk.
    bool saveEdges();

    // Save the feature maps to disk as a PGM for each filter.
    void saveFilterMaps();
    float initContext(int size);
};