#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int DEFAULT_WIDTH = 640;
constexpr int DEFAULT_HEIGHT = 480;
constexpr int DEFAULT_FRAMES = 300;
constexpr int DEFAULT_STREAMS = 4;
constexpr int MAX_STREAMS = 8;
constexpr unsigned FRAME_SEED = 1337u;

constexpr int KERNEL_WIDTH = 5;
constexpr int KERNEL_HEIGHT = 5;
constexpr int KERNEL_RADIUS = KERNEL_WIDTH / 2;
constexpr int KERNEL_MARGIN = 2 * KERNEL_RADIUS;
constexpr int KERNEL_SIZE = KERNEL_HEIGHT * KERNEL_WIDTH;

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int TILE_W = BLOCK_X + KERNEL_MARGIN;
constexpr int TILE_H = BLOCK_Y + KERNEL_MARGIN;

// 5x5 Gaussian kernel
#define GAUSSIAN_LITERALS                                                                                              \
    {0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f, 0.0133f, 0.0596f, 0.0983f, 0.0596f,                                  \
     0.0133f, 0.0219f, 0.0983f, 0.1621f, 0.0983f, 0.0219f, 0.0133f, 0.0596f, 0.0983f,                                  \
     0.0596f, 0.0133f, 0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f}

__constant__ float GAUSSIAN_KERNEL[KERNEL_SIZE];
const float GAUSSIAN_KERNEL_HOST[KERNEL_SIZE] = GAUSSIAN_LITERALS;

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// Generate a random video frame
__host__ inline void generate_frame(uchar4 *frame, int width, int height)
{
    for (int i = 0; i < width * height; ++i)
    {
        frame[i] = make_uchar4(rand() % 256, rand() % 256, rand() % 256, 255);
    }
}

__host__ inline float clamp(float v)
{
    if (v < 0.0f)
        return 0.0f;
    else if (v > 255.0f)
        return 255.0f;
    else
        return v;
}

__host__ void invert_and_blur_cpu(const uchar4 *in, uchar4 *out, int width, int height)
{
    // Invert the rgb channels of the frame
    std::vector<uchar4> inverted(static_cast<size_t>(width) * height);
    for (int i = 0; i < width * height; ++i)
    {
        inverted[i] = make_uchar4(255 - in[i].x, 255 - in[i].y, 255 - in[i].z, in[i].w);
    }

    // Apply the 5x5 Gaussian kernel (blur)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int ky = 0; ky < KERNEL_HEIGHT; ++ky)
            {
                for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
                {
                    const int sx = x + kx - KERNEL_RADIUS;
                    const int sy = y + ky - KERNEL_RADIUS;
                    const int cx = sx < 0 ? 0 : (sx >= width ? width - 1 : sx);
                    const int cy = sy < 0 ? 0 : (sy >= height ? height - 1 : sy);

                    const uchar4 pixel = inverted[cy * width + cx];
                    const float weight = GAUSSIAN_KERNEL_HOST[ky * KERNEL_WIDTH + kx];

                    acc[0] += pixel.x * weight;
                    acc[1] += pixel.y * weight;
                    acc[2] += pixel.z * weight;
                    acc[3] += pixel.w * weight;
                }
            }

            out[y * width + x] = make_uchar4(
                static_cast<unsigned char>(clamp(acc[0]) + 0.5f), static_cast<unsigned char>(clamp(acc[1]) + 0.5f),
                static_cast<unsigned char>(clamp(acc[2]) + 0.5f), static_cast<unsigned char>(clamp(acc[3]) + 0.5f));
        }
    }
}

// CUDA kernel timing helper
template <typename F> static float time_kernel_ms(F &&launch)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// Kernels and device functions
__device__ inline float clamp_gpu(float v)
{
    if (v < 0.0f)
        return 0.0f;
    else if (v > 255.0f)
        return 255.0f;
    else
        return v;
}

// Invert each pixel's R, G, and B channels
__global__ void invert_kernel(const uchar4 *in, uchar4 *out, int num_pixels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pixels)
    {
        out[i] = make_uchar4(255 - in[i].x, 255 - in[i].y, 255 - in[i].z, in[i].w);
    }
}

// Apply 5x5 Gaussian blur using a shared memory tile
__global__ void gaussian_blur_kernel(const uchar4 *in, uchar4 *out, int width, int height)
{
    __shared__ uchar4 tile[TILE_H][TILE_W];

    const int x0 = blockIdx.x * BLOCK_X;
    const int y0 = blockIdx.y * BLOCK_Y;

    // Prepare the shared memory tile
    for (int i = threadIdx.y * BLOCK_X + threadIdx.x; i < TILE_H * TILE_W; i += BLOCK_X * BLOCK_Y)
    {
        const int row = i / TILE_W;
        const int col = i % TILE_W;

        int gx = x0 + col - KERNEL_RADIUS;
        int gy = y0 + row - KERNEL_RADIUS;
        gx = gx < 0 ? 0 : (gx >= width ? width - 1 : gx);
        gy = gy < 0 ? 0 : (gy >= height ? height - 1 : gy);

        tile[row][col] = in[gy * width + gx];
    }

    __syncthreads();

    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;

    if (x < width && y < height)
    {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int ky = 0; ky < KERNEL_HEIGHT; ++ky)
        {
            for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
            {
                const uchar4 sample = tile[threadIdx.y + ky][threadIdx.x + kx];
                const float weight = GAUSSIAN_KERNEL[ky * KERNEL_WIDTH + kx];

                acc[0] += sample.x * weight;
                acc[1] += sample.y * weight;
                acc[2] += sample.z * weight;
                acc[3] += sample.w * weight;
            }
        }

        out[y * width + x] = make_uchar4(
            static_cast<unsigned char>(clamp_gpu(acc[0]) + 0.5f), static_cast<unsigned char>(clamp_gpu(acc[1]) + 0.5f),
            static_cast<unsigned char>(clamp_gpu(acc[2]) + 0.5f), static_cast<unsigned char>(clamp_gpu(acc[3]) + 0.5f));
    }
}

// Run the CPU version
__host__ float run_cpu(const uchar4 *host_in, uchar4 *host_ref, int width, int height, int num_frames)
{
    const int num_pixels = width * height;

    const auto start = std::chrono::high_resolution_clock::now();
    for (int f = 0; f < num_frames; ++f)
    {
        const uchar4 *src = host_in + static_cast<size_t>(f) * num_pixels;
        uchar4 *dst = host_ref + static_cast<size_t>(f) * num_pixels;

        invert_and_blur_cpu(src, dst, width, height);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> elapsed = end - start;
    return static_cast<float>(elapsed.count());
}

// Run the CUDA kernels with blocking memory transfers of each frame to/from the GPU
__host__ float run_kernels_blocking(const uchar4 *host_in, uchar4 *host_out, int width, int height, int num_frames)
{
    const size_t frame_bytes = static_cast<size_t>(width) * height * sizeof(uchar4);
    const int num_pixels = width * height;

    // allocate device memory
    uchar4 *d_in = nullptr;
    uchar4 *d_temp = nullptr;
    uchar4 *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, frame_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, frame_bytes));

    const int invert_threads = 256;
    const int invert_blocks = (num_pixels + invert_threads - 1) / invert_threads;
    const dim3 blur_block(BLOCK_X, BLOCK_Y);
    const dim3 blur_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

    // Warm up the GPU before timing
    CUDA_CHECK(cudaMemcpy(d_in, host_in, frame_bytes, cudaMemcpyHostToDevice));
    invert_kernel<<<invert_blocks, invert_threads>>>(d_in, d_temp, num_pixels);
    gaussian_blur_kernel<<<blur_grid, blur_block>>>(d_temp, d_out, width, height);
    CUDA_CHECK(cudaMemcpy(host_out, d_out, frame_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run the kernels on each frame sequentially
    const float ms = time_kernel_ms([&] {
        for (int f = 0; f < num_frames; ++f)
        {
            const uchar4 *src = host_in + static_cast<size_t>(f) * num_pixels;
            uchar4 *dst = host_out + static_cast<size_t>(f) * num_pixels;

            CUDA_CHECK(cudaMemcpy(d_in, src, frame_bytes, cudaMemcpyHostToDevice));
            invert_kernel<<<invert_blocks, invert_threads>>>(d_in, d_temp, num_pixels);
            gaussian_blur_kernel<<<blur_grid, blur_block>>>(d_temp, d_out, width, height);
            CUDA_CHECK(cudaMemcpy(dst, d_out, frame_bytes, cudaMemcpyDeviceToHost));
        }
    });

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_out));

    return ms;
}

// Run the CUDA kernels with streaming memory transfers of each frame to/from the GPU
// Uses events
__host__ float run_kernels_streaming(const uchar4 *host_in, uchar4 *host_out, int width, int height,
                                     const int num_frames, const int num_streams)
{
    const size_t frame_bytes = static_cast<size_t>(width) * height * sizeof(uchar4);
    const int num_pixels = width * height;

    uchar4 *d_in[MAX_STREAMS];
    uchar4 *d_temp[MAX_STREAMS];
    uchar4 *d_out[MAX_STREAMS];
    cudaStream_t streams[MAX_STREAMS];
    cudaEvent_t copy_done[MAX_STREAMS];

    for (int i = 0; i < num_streams; ++i)
    {
        CUDA_CHECK(cudaMalloc(&d_in[i], frame_bytes));
        CUDA_CHECK(cudaMalloc(&d_temp[i], frame_bytes));
        CUDA_CHECK(cudaMalloc(&d_out[i], frame_bytes));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming));
    }

    const int invert_threads = 256;
    const int invert_blocks = (num_pixels + invert_threads - 1) / invert_threads;
    const dim3 blur_block(BLOCK_X, BLOCK_Y);
    const dim3 blur_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

    // Warm-up on one stream before timing
    CUDA_CHECK(cudaMemcpyAsync(d_in[0], host_in, frame_bytes, cudaMemcpyHostToDevice, streams[0]));
    invert_kernel<<<invert_blocks, invert_threads, 0, streams[0]>>>(d_in[0], d_temp[0], num_pixels);
    gaussian_blur_kernel<<<blur_grid, blur_block, 0, streams[0]>>>(d_temp[0], d_out[0], width, height);
    CUDA_CHECK(cudaMemcpyAsync(host_out, d_out[0], frame_bytes, cudaMemcpyDeviceToHost, streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    const float ms = time_kernel_ms([&] {
        for (int i = 0; i < num_frames; ++i)
        {
            const int stream_idx = i % num_streams;
            const uchar4 *src = host_in + static_cast<size_t>(i) * num_pixels;
            const auto stream = streams[stream_idx];

            CUDA_CHECK(cudaMemcpyAsync(d_in[stream_idx], src, frame_bytes, cudaMemcpyHostToDevice, stream));
            invert_kernel<<<invert_blocks, invert_threads, 0, stream>>>(d_in[stream_idx], d_temp[stream_idx],
                                                                        num_pixels);
            gaussian_blur_kernel<<<blur_grid, blur_block, 0, stream>>>(d_temp[stream_idx], d_out[stream_idx], width,
                                                                       height);
            CUDA_CHECK(cudaMemcpyAsync(host_out + static_cast<size_t>(i) * num_pixels, d_out[stream_idx], frame_bytes,
                                       cudaMemcpyDeviceToHost, stream));

            // Signal that this frame is done being processed and copied back to the host
            CUDA_CHECK(cudaEventRecord(copy_done[stream_idx], stream));
        }

        // Wait on each stream's completion event before returning
        for (int i = 0; i < num_streams; ++i)
        {
            CUDA_CHECK(cudaEventSynchronize(copy_done[i]));
        }
    });

    for (int i = 0; i < num_streams; ++i)
    {
        CUDA_CHECK(cudaEventDestroy(copy_done[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFree(d_in[i]));
        CUDA_CHECK(cudaFree(d_temp[i]));
        CUDA_CHECK(cudaFree(d_out[i]));
    }

    return ms;
}

int main(int argc, char **argv)
{
    const int num_frames = argc >= 2 ? atoi(argv[1]) : DEFAULT_FRAMES;
    const int width = argc >= 3 ? atoi(argv[2]) : DEFAULT_WIDTH;
    const int height = argc >= 4 ? atoi(argv[3]) : DEFAULT_HEIGHT;
    int num_streams = argc >= 5 ? atoi(argv[4]) : DEFAULT_STREAMS;
    num_streams = num_streams < 1 ? 1 : (num_streams > MAX_STREAMS ? MAX_STREAMS : num_streams);

    const size_t frame_bytes = static_cast<size_t>(width) * height * sizeof(uchar4);
    const int frame_pixels = width * height;

    CUDA_CHECK(cudaMemcpyToSymbol(GAUSSIAN_KERNEL, GAUSSIAN_KERNEL_HOST, sizeof(GAUSSIAN_KERNEL_HOST)));

    uchar4 *host_in = nullptr;
    uchar4 *host_out = nullptr;
    uchar4 *host_ref = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_in, static_cast<size_t>(num_frames) * frame_bytes));
    CUDA_CHECK(cudaMallocHost(&host_out, static_cast<size_t>(num_frames) * frame_bytes));
    CUDA_CHECK(cudaMallocHost(&host_ref, static_cast<size_t>(num_frames) * frame_bytes));

    // Seed rng and generate a number of frames to represent a video
    srand(FRAME_SEED);
    for (int f = 0; f < num_frames; ++f)
    {
        generate_frame(host_in + static_cast<size_t>(f) * frame_pixels, width, height);
    }

    // Run CPU version
    const float cpu_ms = run_cpu(host_in, host_ref, width, height, num_frames);

    // Run blocking version
    const float blocking_ms = run_kernels_blocking(host_in, host_out, width, height, num_frames);

    // Run streaming version
    const float streaming_ms = run_kernels_streaming(host_in, host_out, width, height, num_frames, num_streams);

    // Print summary of results
    printf("Video: %dx%d, frames=%d, streams=%d\n", width, height, num_frames, num_streams);
    printf("CPU version: %10.3f ms\n", cpu_ms);
    printf("Blocking GPU version: %10.3f ms\n", blocking_ms);
    printf("Streaming GPU version: %10.3f ms\n", streaming_ms);

    CUDA_CHECK(cudaFreeHost(host_in));
    CUDA_CHECK(cudaFreeHost(host_out));
    CUDA_CHECK(cudaFreeHost(host_ref));

    return 0;
}
