#include "visualime/cuda/use_cuda.h"

#ifdef VISUALIME_USE_CUDA
#include "visualime/cuda/cuda_engine.cuh"
#include "visualime/scenes.h"
#include "visualime/engine.h"
#include "visualime/cuda/cuda_helpers.cuh"

namespace visualime::engine {
    cuda_engine::cuda_engine(unsigned int width, unsigned int height, const math_utils::interval render_border):
            _width(width), _height(height), _render_border(render_border), _d_data(nullptr), _d_data_buffer(nullptr),
            _d_depth(nullptr), _renderer(width, height, render_border) {
        std::cout << "Engine initialized (" << width << "x" << height <<")" << std::endl;
        cudaMalloc((void**)&_d_data, _width * _height * 3 * sizeof(unsigned char));
        cudaMalloc((void**)&_d_data_buffer, _width * _height * 3 * sizeof(unsigned char));
        cudaMalloc((void**)&_d_depth, _width * _height * sizeof(float));
        cudaMemset(_d_data, 0, _width * _height * 3 * sizeof(unsigned char));
        cudaMemset(_d_data_buffer, 0, _width * _height * 3 * sizeof(unsigned char));
        cudaMemset(_d_depth, 0, _width * _height * sizeof(float));
    }
    cuda_engine::~cuda_engine() {
        cudaFree(_d_data);
        cudaFree(_d_data_buffer);
        cudaFree(_d_depth);
    }

    void cuda_engine::render(const std::vector<std::shared_ptr<primitive::primitive_base>> &primitives,
                             std::shared_ptr<canvas::canvas_base> background_canvas)
    {
        // copy from _background_canvas
        if (background_canvas == nullptr) {                                         // from canvas or blank
            cudaMemset(_d_data, 0, _width * _height * 3 * sizeof(unsigned char));
        } else {
            cudaMemcpy(_d_data, background_canvas->get_data(), _width * _height * 3 * sizeof(unsigned char),
                       cudaMemcpyHostToDevice);
        }
        memset(_d_depth, 0, _width * _height * sizeof(float));
        _renderer.render(primitives, _d_data_buffer, _d_depth);
        auto tmp = _d_data;                                           // swap buffer
        _d_data = _d_data_buffer;
        _d_data_buffer = tmp;
    }

    void cuda_engine::copy_to(uchar3 *d_ptr) {
        cudaMemcpy(d_ptr, _d_data, _width * _height * 3 * sizeof(unsigned char),
                   cudaMemcpyDeviceToDevice);
    }
}
#endif //VISUALIME_USE_CUDA