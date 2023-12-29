
#ifndef VISUALIME_CUDA_RENDERER_H
#define VISUALIME_CUDA_RENDERER_H

#include "visualime/renderer.h"
#include "vector"
#include "visualime/math_utils.h"
#include "visualime/primitives.h"
#include "visualime/canvas.h"
#include "iostream"
#include "memory"
#include "visualime/cuda/use_cuda.h"

namespace visualime::renderer {
#ifdef VISUALIME_USE_CUDA
    class cuda_renderer : public renderer_base {
    public:
        cuda_renderer(unsigned int width, unsigned int height, math_utils::interval render_border = {0, 1});
        ~cuda_renderer() override;
        void render(const std::vector<std::shared_ptr<primitive::primitive_base>>& primitives,
                    unsigned char* d_data, float* d_depth) override;
    private:
        unsigned int _width;
        unsigned int _height;
        math_utils::interval _render_border;
    };
#endif //VISUALIME_USE_CUDA
}


#endif //VISUALIME_CUDA_RENDERER_H