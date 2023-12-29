
#include "visualime/cuda/use_cuda.h"
#include "visualime/cuda/cuda_renderer.h"
#include "visualime/cuda/cuda_helpers.cuh"
#include "visualime/cuda/use_cuda.h"
#include "visualime/cuda/cuda_helpers.cuh"
#include "visualime/math_utils.h"
#include "visualime/primitives.h"
#include "visualime/canvas.h"
#include "visualime/renderer.h"
#include "iostream"


namespace visualime::renderer {
#ifdef VISUALIME_USE_CUDA
    cuda_renderer::cuda_renderer(unsigned int width, unsigned int height, const math_utils::interval render_border):
            _width(width), _height(height), _render_border(render_border) {
        std::cout << "Renderer initialized (" << width << "x" << height <<")" << std::endl;
    }
    cuda_renderer::~cuda_renderer() = default;

    __device__ math_utils::aabb clamp_bounding_box(const math_utils::aabb& bounding_box,
                                                   const math_utils::interval& render_border) {
        math_utils::aabb result = bounding_box;
        result.x_inter = result.x_inter.clamped(render_border);
        result.y_inter = result.y_inter.clamped(render_border);
        return result;
    }

    __global__ void primitive_based_render_kernel(unsigned char* d_data, float* d_depth, unsigned int width,
                                                  unsigned int height,
                                                  math_utils::interval render_border, unsigned int primitive_count,
                                                  primitive::primitive_base** d_primitives) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;
        unsigned int idx = y * width + x;
        float depth = 1.0f;
        for (unsigned int i = 0; i < primitive_count; ++i) {
            // atomic
            auto primitive = d_primitives[i];
            if (primitive == nullptr) continue;
            auto bounding_box = primitive->get_aabb();
            bounding_box = clamp_bounding_box(bounding_box, render_border);
            if (bounding_box.is_empty()) continue;
            glm::vec2 point = {x / static_cast<float>(width), y / static_cast<float>(height)};
            if (bounding_box.contains(point)) {
                depth = primitive->get_depth(point);
                atomicExch(&d_depth[idx], depth);
                d_data[idx * 3] = primitive->show_color(point).r;
                d_data[idx * 3 + 1] = primitive->show_color(point).g;
                d_data[idx * 3 + 2] = primitive->show_color(point).b;
            }
        }
    }

    void cuda_renderer::render(const std::vector<std::shared_ptr<primitive::primitive_base>> &primitives,
                               unsigned char *d_data, float *d_depth) {
        unsigned int primitive_count = primitives.size();
        primitive::primitive_base** d_primitives;
        cudaMalloc((void**)&d_primitives, primitive_count * sizeof(primitive::primitive_base*));
        for (unsigned int i = 0; i < primitive_count; ++i) {
            cudaMalloc((void**)&d_primitives[i], sizeof(primitive::primitive_base));
            cudaMemcpy(d_primitives[i], primitives[i].get(), sizeof(primitive::primitive_base), cudaMemcpyHostToDevice);
        }
        dim3 block_size(16, 16);
        dim3 grid_size((_width + block_size.x - 1) / block_size.x, (_height + block_size.y - 1) / block_size.y);
        primitive_based_render_kernel<<<grid_size, block_size>>>(d_data, d_depth, _width,
                                                                 _height, _render_border,
                                                                 primitive_count,
                                                                 d_primitives);
        cudaDeviceSynchronize();
        for (unsigned int i = 0; i < primitive_count; ++i) {
            cudaFree(d_primitives[i]);
        }
        cudaFree(d_primitives);
    }
#endif //VISUALIME_USE_CUDA
}