//
// Created by alexzms on 2023/12/21.
//

#ifndef VISUALIME_CUDA_ENGINE_H
#define VISUALIME_CUDA_ENGINE_H

#include "visualime/cuda/use_cuda.h"

#include "visualime/renderer.h"
#include "vector"
#include "visualime/math_utils.h"
#include "visualime/engine.h"
#include "iostream"


namespace visualime::engine {
#ifdef VISUALIME_USE_CUDA
    class cuda_engine: public engine_base {
    public:
        cuda_engine(unsigned int width, unsigned int height, math_utils::interval render_border = {0, 1});
        ~cuda_engine() override;
        void copy_to(uchar3* d_ptr);
        void render(const std::vector<std::shared_ptr<primitive::primitive_base>>& primitives,
                    std::shared_ptr<canvas::canvas_base> background_canvas) override;
        [[nodiscard]] unsigned int get_width() const { return _width; }
        [[nodiscard]] unsigned int get_height() const { return _height; }
        [[nodiscard]] unsigned char* get_data() const { return _d_data; }
    private:
        unsigned int _width;
        unsigned int _height;
        unsigned char* _d_data;
        unsigned char* _d_data_buffer;
        renderer::simple_renderer _renderer;
        math_utils::interval _render_border;
        float* _d_depth;
    };
#endif
}

#endif //VISUALIME_CUDA_ENGINE_H
