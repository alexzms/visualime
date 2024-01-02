//
// Created by alexzms on 2023/12/17.
//

#ifndef UNIT_TESTS_RENDERER_H
#define UNIT_TESTS_RENDERER_H

#include "memory"
#include "vector"
#include "primitives.h"
#include "math_utils.h"
#include "omp.h"
#include "iostream"

namespace visualime::renderer {

    class orthogonal_renderer {
    public:
        orthogonal_renderer(unsigned int width, unsigned int height, const math_utils::interval& render_border):
                        _width(width), _height(height), _render_border(render_border) {}

        void render_single(const std::shared_ptr<primitive::primitive_base>& primitive,
                           unsigned char* target_rgb, float* target_depth) {
            if (target_rgb == nullptr || target_depth == nullptr) {
                std::cout << "[visualime][renderer] target_rgb or target_depth is nullptr" << std::endl;
                return;
            }
            if (primitive == nullptr)                                             // deleted primitive
                return;
            auto bounding_box = primitive->get_aabb().clamped(_render_border);
            if (bounding_box.is_empty())
                return;                                // if empty, skip
            for (int k = (int)ceil(bounding_box.y_inter.min * _height);        // ceil to avoid missing pixels
                 k < bounding_box.y_inter.max * _height;
                 ++k) {
                for (int j = (int)ceil(bounding_box.x_inter.min * _width);
                     j < bounding_box.x_inter.max * _width;
                     ++j) {

                    glm::vec2 point = {j / static_cast<double>(_width), k / static_cast<double>(_height)};
                    glm::vec<3, unsigned char> color = primitive->show_color(point);
                    auto depth = (float)primitive->get_depth(point);
                    unsigned offset = k * _width + j;
                    if (depth > target_depth[offset]) {
                        target_depth[offset] = depth;
                        target_rgb[offset * 3] = color.r;
                        target_rgb[offset * 3 + 1] = color.g;
                        target_rgb[offset * 3 + 2] = color.b;
                    }
                }
            }
        }


        void render(const std::vector<std::shared_ptr<primitive::primitive_base>>& primitives,
                    unsigned char* target_rgb, float* target_depth) {
            if (target_rgb == nullptr || target_depth == nullptr) {
                std::cout << "[renderer] target_rgb or target_depth is nullptr" << std::endl;
                return;
            }
            #pragma omp parallel for
            for (int i = 0; i < primitives.size(); ++i) {
                const auto& primitive = primitives[i];
                if (primitive == nullptr)                                             // deleted primitvie
                    continue;
                auto bounding_box = primitive->get_aabb().clamped(_render_border);
                if (bounding_box.is_empty())
                    continue;                                // if empty, skip this
                for (int k = (int)ceil(bounding_box.y_inter.min * _height);        // ceil to avoid missing pixels
                     k < bounding_box.y_inter.max * _height;
                     ++k) {
                    for (int j = (int)ceil(bounding_box.x_inter.min * _width);
                         j < bounding_box.x_inter.max * _width;
                         ++j) {

                        glm::vec2 point = {j / static_cast<double>(_width), k / static_cast<double>(_height)};
                        glm::vec<3, unsigned char> color = primitive->show_color(point);
                        auto depth = (float)primitive->get_depth(point);
                        unsigned offset = k * _width + j;
                        if (depth > target_depth[offset]) {
                            target_depth[offset] = depth;
                            target_rgb[offset * 3] = color.r;
                            target_rgb[offset * 3 + 1] = color.g;
                            target_rgb[offset * 3 + 2] = color.b;
                        }
                    }
                }
            }
        }

    private:
        unsigned int _width;
        unsigned int _height;
        math_utils::interval _render_border;
    };

    class perspective_renderer {
        // TODO
    };

}

#endif //UNIT_TESTS_RENDERER_H
