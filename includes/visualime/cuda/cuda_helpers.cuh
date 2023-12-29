
#ifndef VISUALIME_CUDA_HELPERS_H
#define VISUALIME_CUDA_HELPERS_H

#ifdef VISUALIME_USE_CUDA
#include <cstdlib>
#include <cstdio>
#include "cuda_runtime.h"

namespace visualime::cuda_helper {
    inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
        if (code != cudaSuccess) {
            std::printf("GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }
}

#define CHECK_ERROR(ans) { visualime::cuda_helper::gpu_assert((ans), __FILE__, __LINE__); }

#endif //VISUALIME_USE_CUDA

#endif //VISUALIME_CUDA_HELPERS_H