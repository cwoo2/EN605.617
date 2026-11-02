#include "check.h"
#include "conv.h"

constexpr int THREADS_PER_BLOCK = 256;

// im2col: one row per output pixel, one column per patch element
__global__ void im2colKernel(const float *img, float *cols, ConvShape s)
{
    // The size of each patch, (# pixels in the kernel)
    // This is the length of each row
    const int patch_size = s.kernel_size * s.kernel_size;
    const int total = s.out_w * s.out_h * patch_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    // Calculate the row and column of the im2col matrix for this thread
    const int row = idx / patch_size;
    const int col = idx % patch_size;

    // Calculate the kernel coordinates for this column, row major in the kernel
    const int kx = col % s.kernel_size;
    const int ky = col / s.kernel_size;

    const int py = row / s.out_w;
    const int px = row % s.out_w;

    // input pixel (py + ky, px + kx) -> patch element (ky, kx)
    cols[idx] = img[(py + ky) * s.width + (px + kx)];
}

ConvShape::ConvShape(int w, int h)
    : width(w), height(h), num_filters(NUM_FILTERS), kernel_size(KERNEL_SIZE),
      out_w(width - kernel_size + 1), out_h(height - kernel_size + 1)
{
}

int ConvShape::patchElements() const
{
    return kernel_size * kernel_size;
}

int ConvShape::numOutputElements() const
{
    return out_w * out_h;
}

void im2col(const float *d_img, float *d_cols, ConvShape s)
{
    const int total = s.numOutputElements() * s.patchElements();
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    im2colKernel<<<blocks, THREADS_PER_BLOCK>>>(d_img, d_cols, s);
    CUDA_CHECK(cudaGetLastError());
}

// A = d_cols (P x N), where P is the patch size and N is the number of output
// pixels, transposed to N x P, ldb = P
// B = d_filters (P x F), where F is the number of filters, ldb = P
// C = d_out  (N x F), ldc = N
void gemmConv(cublasHandle_t handle, const float *d_filters,
              const float *d_cols, float *d_out, ConvShape s)
{
    const int n_out = s.numOutputElements();
    const int P = s.patchElements();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_out,
                             s.num_filters, P, &alpha, d_cols, P, d_filters, P,
                             &beta, d_out, n_out));
}
