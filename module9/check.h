#pragma once

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <npp.h>

// Helpers for checking CUDA, NPP, and cuBLAS calls for errors.
#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess)                                               \
        {                                                                      \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err_));                                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define NPP_CHECK(call)                                                        \
    do                                                                         \
    {                                                                          \
        NppStatus status_ = (call);                                            \
        if (status_ != NPP_SUCCESS)                                            \
        {                                                                      \
            fprintf(stderr, "%s:%d: NPP error: %d\n", __FILE__, __LINE__,      \
                    static_cast<int>(status_));                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do                                                                         \
    {                                                                          \
        cublasStatus_t status_ = (call);                                       \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                  \
        {                                                                      \
            fprintf(stderr, "%s:%d: cuBLAS error: %s\n", __FILE__, __LINE__,   \
                    cublasGetStatusString(status_));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
