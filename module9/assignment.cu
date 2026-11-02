// EN605.617 Module 9: convolution as GEMM with NPP preprocessing and
// postprocessing. See module9-npp-cublas-conv-gemm.md for the design.

#include "assignment.h"
#include "check.h"
#include "conv.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

constexpr int DEFAULT_SIZE = 1024;

std::vector<uint8_t> generateImage(int width, int height)
{
    std::vector<uint8_t> img(static_cast<size_t>(width) * height);

    std::random_device rd;
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> noise(-8.0f, 8.0f);

    const float cx = 0.5f * width;
    const float cy = 0.5f * height;
    const float radius = 0.3f * std::min(width, height);
    const float radius2 = radius * radius;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float value = 40.0f + 160.0f * x / std::max(1, width - 1);

            const float dx = x - cx;
            const float dy = y - cy;

            if (dx * dx + dy * dy < radius2)
                value = 230.0f;

            value += noise(gen);
            value = std::min(255.0f, std::max(0.0f, value));

            img[static_cast<size_t>(y) * width + x] =
                static_cast<uint8_t>(value + 0.5f);
        }
    }

    return img;
}

void writePgm(const char *path, const uint8_t *img, int width, int height)
{
    std::FILE *f = std::fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "cannot open %s for writing\n", path);
        std::exit(1);
    }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    std::fwrite(img, 1, static_cast<size_t>(width) * height, f);
    std::fclose(f);
}

void saveFloatMap(const float *map, int width, int height, const char *path)
{
    const int count = width * height;
    const auto lo_hi = std::minmax_element(map, map + count);
    const float lo = *lo_hi.first;
    const float range = *lo_hi.second - lo;
    std::vector<uint8_t> scaled(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        const float scale = range > 0.0f ? (map[i] - lo) / range : 0.0f;
        scaled[i] = static_cast<uint8_t>(scale * 255.0f + 0.5f);
    }
    writePgm(path, scaled.data(), width, height);
    printf("wrote %s\n", path);
}

void report(const ConvShape &s, float ms_cols, float ms_gemm, float ms_npp)
{
    printf("%dx%d -> %dx%d, %d filters, K=%d\n", s.width, s.height, s.out_w,
           s.out_h, s.num_filters, s.patchElements());
    printf("im2col: %.2f ms | gemm: %.2f ms | npp: %.2f ms\n", ms_cols, ms_gemm,
           ms_npp);
}

// Times one launch (or any block of GPU work) with CUDA events.
template <typename F> float timeKernelMs(F &&launch)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

void Context::allocate(int width, int height)
{
    shape = ConvShape(width, height);
    const NppiSize roi = {shape.out_w, shape.out_h};
    const size_t pixels = static_cast<size_t>(width) * height;
    const size_t rows = static_cast<size_t>(shape.numOutputElements());
    const size_t patch = static_cast<size_t>(shape.patchElements());
    const size_t weights = static_cast<size_t>(shape.num_filters) * patch;

    CUDA_CHECK(cudaMalloc(&d_raw, pixels * sizeof(Npp8u)));
    CUDA_CHECK(cudaMalloc(&d_blur, pixels * sizeof(Npp8u)));
    CUDA_CHECK(cudaMalloc(&d_img, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cols, rows * patch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filters, weights * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, shape.num_filters * rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mag, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_interleaved, 2 * rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(Npp32f)));

    size_t scratch = 0;
    NPP_CHECK(nppiMaxGetBufferHostSize_32f_C1R(roi, &scratch));
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch));

    CUDA_CHECK(cudaMemcpy(d_filters, SOBEL_FILTERS, weights * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasCreate(&handle));
}

void Context::release()
{
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_raw));
    CUDA_CHECK(cudaFree(d_blur));
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_filters));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_mag));
    CUDA_CHECK(cudaFree(d_interleaved));
    CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_max));
}

void Context::nppBlurAndConvert(const std::vector<uint8_t> &img)
{
    // distances between rows, in bytes
    const int step8 = shape.width * static_cast<int>(sizeof(Npp8u));
    const int stepf = shape.width * static_cast<int>(sizeof(Npp32f));

    const NppiSize roi = {shape.width,
                          shape.height}; // ROI is just the whole image
    const NppiPoint origin = {0, 0};     // origin is just 0,0

    CUDA_CHECK(
        cudaMemcpy(d_raw, img.data(), img.size(), cudaMemcpyHostToDevice));
    NPP_CHECK(nppiFilterGaussBorder_8u_C1R(d_raw, step8, roi, origin, d_blur,
                                           step8, roi, NPP_MASK_SIZE_3_X_3,
                                           NPP_BORDER_REPLICATE));
    NPP_CHECK(nppiConvert_8u32f_C1R(d_blur, step8, d_img, stepf, roi));
}

void Context::computeMagnitude()
{
    const int count = shape.numOutputElements();
    const int row_bytes = shape.out_w * static_cast<int>(sizeof(Npp32f));
    Npp8u *base = reinterpret_cast<Npp8u *>(d_interleaved);

    // nppiMagnitude_32fc32f_C1R expects the values to be interleaved, but the
    // GEMM output (d_out) has the X results in the first half and the Y results
    // in the second. To interleave, use two strided cudaMemcpy2d calls
    CUDA_CHECK(cudaMemcpy2D(base, 2 * sizeof(Npp32f), d_out, sizeof(Npp32f),
                            sizeof(Npp32f), count, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy2D(base + sizeof(Npp32f), 2 * sizeof(Npp32f),
                            d_out + count, sizeof(Npp32f), sizeof(Npp32f),
                            count, cudaMemcpyDeviceToDevice));

    const NppiSize roi = {shape.out_w, shape.out_h};
    NPP_CHECK(nppiMagnitude_32fc32f_C1R(
        reinterpret_cast<const Npp32fc *>(d_interleaved), 2 * row_bytes, d_mag,
        row_bytes, roi));
}

bool Context::saveEdges()
{
    const NppiSize roi = {shape.out_w, shape.out_h};
    const int stepf = shape.out_w * static_cast<int>(sizeof(Npp32f));
    const int step8 = shape.out_w * static_cast<int>(sizeof(Npp8u));

    // Find max intensity
    NPP_CHECK(nppiMax_32f_C1R(d_mag, stepf, roi, d_scratch, d_max));
    Npp32f peak = 0.0f;
    CUDA_CHECK(
        cudaMemcpy(&peak, d_max, sizeof(Npp32f), cudaMemcpyDeviceToHost));
    if (!(peak > 0.0f))
    {
        fprintf(stderr, "edge map is empty\n");
        return false;
    }

    // Normalize to 0-255 and convert to 8-bit
    NPP_CHECK(nppiMulC_32f_C1IR(255.0f / peak, d_mag, stepf, roi));
    NPP_CHECK(
        nppiConvert_32f8u_C1R(d_mag, stepf, d_blur, step8, roi, NPP_RND_NEAR));

    // Save to disk
    std::vector<uint8_t> host(static_cast<size_t>(shape.out_w) * shape.out_h);
    CUDA_CHECK(
        cudaMemcpy(host.data(), d_blur, host.size(), cudaMemcpyDeviceToHost));
    const std::string path = "edges_" + std::to_string(shape.out_w) + ".pgm";
    writePgm(path.c_str(), host.data(), shape.out_w, shape.out_h);
    printf("wrote %s (peak %.3f)\n", path.c_str(), peak);
    return true;
}

void Context::saveFilterMaps()
{
    const int rows = shape.numOutputElements();
    std::vector<float> host(static_cast<size_t>(rows));
    for (int k = 0; k < shape.num_filters; ++k)
    {
        CUDA_CHECK(cudaMemcpy(host.data(), d_out + k * rows,
                              host.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        char name[32];
        std::snprintf(name, sizeof(name), "filter_sobel%s_%d.pgm",
                      k == 0 ? "X" : "Y", shape.width);
        saveFloatMap(host.data(), shape.out_w, shape.out_h, name);
    }
}

float Context::initContext(int size)
{
    const std::vector<uint8_t> img = generateImage(size, size);
    const std::string path = "input_" + std::to_string(size) + ".pgm";
    writePgm(path.c_str(), img.data(), size, size);
    return timeKernelMs([&] { nppBlurAndConvert(img); });
}

int main(int argc, char **argv)
{
    const int size = argc > 1 ? std::atoi(argv[1]) : DEFAULT_SIZE;
    if (size < 8)
    {
        fprintf(stderr, "image size must be at least 8\n");
        return 1;
    }

    Context ctx;
    ctx.allocate(size, size);
    const float ms_front = ctx.initContext(size);

    const float ms_cols =
        timeKernelMs([&] { im2col(ctx.d_img, ctx.d_cols, ctx.shape); });
    const float ms_gemm = timeKernelMs([&] {
        gemmConv(ctx.handle, ctx.d_filters, ctx.d_cols, ctx.d_out, ctx.shape);
    });

    const float ms_back = timeKernelMs([&] { ctx.computeMagnitude(); });

    const bool edges_ok = ctx.saveEdges();
    ctx.saveFilterMaps();
    report(ctx.shape, ms_cols, ms_gemm, ms_front + ms_back);
    ctx.release();
    return edges_ok ? 0 : 1;
}
