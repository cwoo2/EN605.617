#include <chrono>
#include <random>

#define KERNEL_HEIGHT 5
#define KERNEL_WIDTH 5
#define KERNEL_RADIUS (KERNEL_WIDTH / 2)
#define KERNEL_MARGIN 2 * KERNEL_RADIUS
#define KERNEL_SIZE (KERNEL_HEIGHT * KERNEL_WIDTH)
#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define IMAGE_SIZE_BYTES IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float)

#define BLOCK_X 128
#define BLOCK_Y 8
#define TILE_H (BLOCK_Y + KERNEL_MARGIN)
#define TILE_W (BLOCK_X + KERNEL_MARGIN)
#define TILE_ELEMENTS (TILE_H * TILE_W)

#define ROWS_PER_THREAD 8 // rows per thread for register memory kernel

// 5x5 Gaussian kernel
#define GAUSSIAN_LITERALS                                                                                              \
    {0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f, 0.0133f, 0.0596f, 0.0983f, 0.0596f,                                  \
     0.0133f, 0.0219f, 0.0983f, 0.1621f, 0.0983f, 0.0219f, 0.0133f, 0.0596f, 0.0983f,                                  \
     0.0596f, 0.0133f, 0.0030f, 0.0133f, 0.0219f, 0.0133f, 0.0030f}

const float GAUSSIAN_KERNEL[KERNEL_SIZE] = GAUSSIAN_LITERALS;

// constant memory
__constant__ float constant_kernel[KERNEL_SIZE];

// Generate a random grayscale image of dimensions (width x height)
__host__ void generate_image(float *const image, const int width, const int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int rand_val = rand() % 256; // Generate a random integer between 0 and 255
            image[y * width + x] = static_cast<float>(rand_val);
        }
    }
}

__host__ void convolution_cpu(float *const input, float *output, int width, int height)
{
    // Since the kernel is 5x5, iteration starts and ends 2 pixels away from the edges
    for (int y = KERNEL_HEIGHT / 2; y < height - KERNEL_HEIGHT / 2; ++y)
    {
        for (int x = KERNEL_WIDTH / 2; x < width - KERNEL_WIDTH / 2; ++x)
        {
            // Apply the kernel to the current pixel and its neighbors
            float sum = 0.0f;
            for (int ky = 0; ky < KERNEL_HEIGHT; ++ky)
            {
                for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
                {
                    int ix = x + kx - KERNEL_WIDTH / 2;
                    int iy = y + ky - KERNEL_HEIGHT / 2;

                    sum += input[iy * width + ix] * GAUSSIAN_KERNEL[ky * KERNEL_WIDTH + kx];
                }
            }
            output[y * width + x] = sum;
        }
    }
}

// Performs convolution using the Gaussian kernel on the GPU with the kernel stored in global memory
__global__ void convolution_global(const float *const input, float *output, const float *const mask, const int width,
                                   const int height)
{
    // Calculate the global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= KERNEL_RADIUS && x < width - KERNEL_RADIUS && y >= KERNEL_RADIUS && y < height - KERNEL_RADIUS)
    {
        float sum = 0.0f;

        // Apply the convolution kernel
        for (int ky = 0; ky < KERNEL_HEIGHT; ++ky)
        {
            for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
            {
                int ix = x + kx - KERNEL_RADIUS;
                int iy = y + ky - KERNEL_RADIUS;

                sum += input[iy * width + ix] * mask[ky * KERNEL_WIDTH + kx];
            }
        }

        output[y * width + x] = sum;
    }
}

// Performs convolution using the Gaussian kernel on the GPU with the kernel stored in constant memory
__global__ void convolution_constant(const float *const input, float *output, const int width, const int height)
{
    // Calculate the global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the valid output range
    if (x >= KERNEL_RADIUS && x < width - KERNEL_RADIUS && y >= KERNEL_RADIUS && y < height - KERNEL_RADIUS)
    {
        float sum = 0.0f;

        // Apply the convolution kernel
        for (int ky = 0; ky < KERNEL_HEIGHT; ++ky)
        {
            for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
            {
                int ix = x + kx - KERNEL_RADIUS;
                int iy = y + ky - KERNEL_RADIUS;

                sum += input[iy * width + ix] * constant_kernel[ky * KERNEL_WIDTH + kx];
            }
        }

        // Write the result to the output image
        output[y * width + x] = sum;
    }
}

__global__ void convolution_shared(const float *const input, float *output, const float *const mask, const int width,
                                   const int height)
{
    // Create a shared memory tile that is the size of a block plus the 2px margins
    // on each side due to the 5x5 kernel. This allows each thread to access its pixel
    // and its neighbors without going out of bounds
    __shared__ float tile[TILE_H][TILE_W];

    const int x0 = blockIdx.x * BLOCK_X;
    const int y0 = blockIdx.y * BLOCK_Y;

    // Fill out the tile once, then synchronize
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < TILE_ELEMENTS; i += BLOCK_X * BLOCK_Y)
    {
        const int r = i / TILE_W;
        const int c = i - r * TILE_W;
        const int gy = y0 - KERNEL_RADIUS + r;
        const int gx = x0 - KERNEL_RADIUS + c;

        // margin pixels that fall outside the image read as zero
        tile[r][c] = (gy >= 0 && gy < height && gx >= 0 && gx < width) ? input[gy * width + gx] : 0.0f;
    }

    __syncthreads();

    const int x = x0 + threadIdx.x;
    const int y = y0 + threadIdx.y;

    if (x >= KERNEL_RADIUS && x < width - KERNEL_RADIUS && y >= KERNEL_RADIUS && y < height - KERNEL_RADIUS)
    {
        float sum = 0.0f;

        for (int ky = 0; ky < KERNEL_HEIGHT; ++ky)
        {
            for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
            {
                sum += tile[threadIdx.y + ky][threadIdx.x + kx] * mask[ky * KERNEL_WIDTH + kx];
            }
        }

        // Write the result to the output image
        output[y * width + x] = sum;
    }
}

__global__ void convolution_hardcoded(const float *const input, float *output, const int width, const int height)
{
    // Hardcode the kernel directly
    float mask[KERNEL_SIZE] = GAUSSIAN_LITERALS;

    // Calculate the global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure the thread is within the valid output range
    if (x >= KERNEL_RADIUS && x < width - KERNEL_RADIUS && y >= KERNEL_RADIUS && y < height - KERNEL_RADIUS)
    {
        float sum = 0.0f;

        // Apply the convolution kernel
        for (int ky = 0; ky < KERNEL_HEIGHT; ++ky)
        {
            for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
            {
                int ix = x + kx - KERNEL_RADIUS;
                int iy = y + ky - KERNEL_RADIUS;

                sum += input[iy * width + ix] * mask[ky * KERNEL_WIDTH + kx];
            }
        }

        // Write the result to the output image
        output[y * width + x] = sum;
    }
}

// Each thread is responsible for ROWS_PER_THREAD rows and 1 column of pixels
__global__ void convolution_register(const float *input, float *output, const int width, const int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = blockIdx.y * ROWS_PER_THREAD;
    float acc[ROWS_PER_THREAD] = {};

    // Iterate over the rows, including the extra rows needed for the kernel radius
    for (int r = 0; r < ROWS_PER_THREAD + KERNEL_MARGIN; ++r)
    {
        const int iy = y0 + r - KERNEL_RADIUS;
        float slice[KERNEL_WIDTH];

        // Load a horizontal slice of 5 pixels from (iy, x-2...x+2) into registers
        for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
        {
            const int ix = x + kx - KERNEL_RADIUS;

            // margin pixels that fall outside the image read as zero
            slice[kx] = (iy >= 0 && iy < height && ix >= 0 && ix < width) ? input[iy * width + ix] : 0.0f;
        }

        // Reuse the current slice by computing the dot product of the slice and a
        // row of the kernel for up to 5 rows. For example, the topmost slice
        // only contributes to the convolution of pixel 0 (the topmost pixel), but the next slice
        // contributes to that of pixels 0 and 1, etc.
        for (int j = 0; j < ROWS_PER_THREAD; ++j)
        {
            // If the current row ky is beyond the bounds of the kernel, skip it
            const int ky = r - j;
            if (ky >= 0 && ky < KERNEL_HEIGHT)
            {
                // Accumulate the dot product for pixel j
                for (int kx = 0; kx < KERNEL_WIDTH; ++kx)
                    acc[j] += slice[kx] * constant_kernel[ky * KERNEL_WIDTH + kx];
            }
        }
    }

    // Write out results
    for (int j = 0; j < ROWS_PER_THREAD; ++j)
    {
        const int y = y0 + j;
        if (x >= KERNEL_RADIUS && x < width - KERNEL_RADIUS && y >= KERNEL_RADIUS && y < height - KERNEL_RADIUS)
            output[y * width + x] = acc[j];
    }
}

// kernel timing helper using CUDA events
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

int main(int argc, char **argv)
{
    // seed rng
    srand(1337);

    float *image = nullptr;
    float *output_cpu = nullptr;

    cudaMallocHost(&image, IMAGE_SIZE_BYTES);
    cudaMallocHost(&output_cpu, IMAGE_SIZE_BYTES);

    // Generate a random image
    generate_image(image, IMAGE_WIDTH, IMAGE_HEIGHT);

    // Perform convolution on the CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    convolution_cpu(image, output_cpu, IMAGE_WIDTH, IMAGE_HEIGHT);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("CPU time: %.3f ms\n", cpu_duration.count());

    float *d_image, *d_output, *d_global_kernel;

    // Allocate GPU memory
    cudaMalloc(&d_image, IMAGE_SIZE_BYTES);
    cudaMalloc(&d_output, IMAGE_SIZE_BYTES);
    cudaMemset(d_output, 0, IMAGE_SIZE_BYTES);
    cudaMalloc(&d_global_kernel, KERNEL_SIZE * sizeof(float));

    // Copy stuff to GPU
    cudaMemcpy(d_image, image, IMAGE_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_global_kernel, GAUSSIAN_KERNEL, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constant_kernel, GAUSSIAN_KERNEL, KERNEL_SIZE * sizeof(float));

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(IMAGE_WIDTH / BLOCK_X, IMAGE_HEIGHT / BLOCK_Y);

    printf("Block size: (%d, %d), Grid size: (%d, %d)\n", block.x, block.y, grid.x, grid.y);

    const dim3 block_reg(128, 1);
    const dim3 grid_reg(IMAGE_WIDTH / 128, IMAGE_HEIGHT / ROWS_PER_THREAD);

    printf("Register memory kernel block size: (%d, %d), grid size: (%d, %d)\n", block_reg.x, block_reg.y, grid_reg.x,
           grid_reg.y);

    // Warmup kernel (CUDA lazy loads)
    convolution_global<<<grid, block>>>(d_image, d_output, d_global_kernel, IMAGE_WIDTH, IMAGE_HEIGHT);

    float gpuMs = time_kernel_ms(
        [&] { convolution_global<<<grid, block>>>(d_image, d_output, d_global_kernel, IMAGE_WIDTH, IMAGE_HEIGHT); });
    printf("Global memory kernel time: %.3f ms\n", gpuMs);

    gpuMs =
        time_kernel_ms([&] { convolution_constant<<<grid, block>>>(d_image, d_output, IMAGE_WIDTH, IMAGE_HEIGHT); });
    printf("Constant memory kernel time: %.3f ms\n", gpuMs);

    cudaMemcpy(output_cpu, d_output, IMAGE_SIZE_BYTES, cudaMemcpyDeviceToHost);

    gpuMs = time_kernel_ms(
        [&] { convolution_shared<<<grid, block>>>(d_image, d_output, d_global_kernel, IMAGE_WIDTH, IMAGE_HEIGHT); });
    printf("Shared memory kernel time: %.3f ms\n", gpuMs);

    gpuMs = time_kernel_ms(
        [&] { convolution_register<<<grid_reg, block_reg>>>(d_image, d_output, IMAGE_WIDTH, IMAGE_HEIGHT); });
    printf("Register memory kernel time: %.3f ms\n", gpuMs);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_global_kernel);

    // Clean up
    cudaFreeHost(image);
    cudaFreeHost(output_cpu);

    return 0;
}