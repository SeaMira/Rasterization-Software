#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " (" << cudaGetErrorString(err) << ")" << std::endl;  \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)
