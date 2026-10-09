// EN605.617 Module 6 - CUDA Streams and Events
//
// Demonstrates CUDA streams and events by processing a simulated video stream
// (synthetic frames) through a per-frame pipeline of two kernels:
//   1. color inversion
//   2. 5x5 Gaussian blur
//
// Three execution modes are timed against each other:
//   A - pinned host memory, blocking cudaMemcpy, default stream (naive)
//   B - pinned host memory, cudaMemcpyAsync, one stream (depth-1)
//   C - pinned host memory, cudaMemcpyAsync, N streams (pipelined)
//
// Based on the work of Andrew Krepps.

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Frame / video parameters
// ---------------------------------------------------------------------------

constexpr int FRAME_WIDTH = 1920;
constexpr int FRAME_HEIGHT = 1080;
constexpr int NUM_FRAMES = 300;
constexpr int NUM_STREAMS = 4;

// ---------------------------------------------------------------------------
// 5x5 Gaussian kernel (same literals used in module 5)
// ---------------------------------------------------------------------------

constexpr int KERNEL_WIDTH = 5;
constexpr int KERNEL_HEIGHT = 5;
constexpr int KERNEL_RADIUS = KERNEL_WIDTH / 2;
constexpr int KERNEL_MARGIN = 2 * KERNEL_RADIUS;
constexpr int KERNEL_SIZE = KERNEL_HEIGHT * KERNEL_WIDTH;

int main(int argc, char **argv)
{
    printf("EN605.617 Module 6 - CUDA Streams and Events\n");
    printf("Frame: %dx%d, frames: %d, streams: %d\n", FRAME_WIDTH, FRAME_HEIGHT, NUM_FRAMES, NUM_STREAMS);
    return 0;
}
