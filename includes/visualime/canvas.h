//
// Created by alexzms on 2023/12/17.
//

#ifndef VISUALIME_CANVAS_H
#define VISUALIME_CANVAS_H

#include "primitives.h"
#include "renderer.h"
#include "glm/glm.hpp"

namespace visualime::canvas {
    class canvas_base : public primitive::primitive_base {
    public:
        // no need to write a virtual ~constructor
        virtual bool draw_pixel(const glm::vec<2, unsigned>& unnormalized_pos, const glm::vec<3, unsigned char>& color)
                                                                                                    { return false; }
        virtual bool draw_primitive(const std::shared_ptr<primitive::primitive_base>& primitive) { return false; }
        virtual bool draw_primitive_and_store(const std::shared_ptr<primitive::primitive_base>& primitive)
                                                                                                 { return false; }
        virtual bool draw_all() { return false; }                           // draw what's stored conveniently
        virtual bool flush() = 0;                                           // flush is to clear to black
        [[nodiscard]] virtual const unsigned char* get_data() const = 0;
    };

    class fullscreen_orthogonal_canvas : public canvas_base {
    public:                                                                 // fullscreen canvas has optimization
        fullscreen_orthogonal_canvas(unsigned int width, unsigned int height, double depth = 0):
                                _renderer(width, height, {0, 1}),
                                _width(width), _height(height), _depth(depth) {
            _data = new unsigned char[_width * _height * 3];
            _data_depth = new float[_width * _height];
            memset(_data, 0, _width * _height * 3 * sizeof(unsigned char));
            memset(_data_depth, 0, _width * _height * sizeof (float));
        }
        ~fullscreen_orthogonal_canvas() override {
            delete[] _data;
        }
        [[nodiscard]] const unsigned char* get_data() const override {
             return _data;
        }

        bool draw_primitive(const std::shared_ptr<primitive::primitive_base>& primitive) override {
            _renderer.render_single(primitive, _data, _data_depth);
            return true;
        }
        bool draw_primitive_and_store(const std::shared_ptr<primitive::primitive_base>& primitive) override {
            _renderer.render_single(primitive, _data, _data_depth);
            _primitives.emplace_back(primitive);
            return true;
        }
        bool draw_all() override {
            _renderer.render(_primitives, _data, _data_depth);
            return true;
        }
        bool flush() override {
            memset(_data, 0, _width * _height * 3 * sizeof(unsigned char));
            memset(_data_depth, 0, _width * _height * sizeof(float));
            return true;
        }

        bool draw_pixel(const glm::vec<2, unsigned>& unnormalized_pos,
                        const glm::vec<3, unsigned char>& color) override {
                                                                            // !!!in range of _width, _height!!!
            auto offset = unnormalized_pos.y * _width + unnormalized_pos.x;
            _data[offset * 3] = color.r;
            _data[offset * 3 + 1] = color.g;
            _data[offset * 3 + 2] = color.b;                                // pixel is always on top! wierd but fast
            return true;
        }

        [[nodiscard]] glm::vec<3, unsigned char> show_color(const glm::vec2 &point) const override {
                                                                            // it's highly not recommended to call
                                                                            // show_color() for canvas
            auto w = static_cast<unsigned>(point.x * (float)_width);
            auto h = static_cast<unsigned>(point.y * (float)_height);
            auto offset = h * _width + w;
            return {
                _data[offset * 3],                       // static_cast keeps it's original value
                _data[offset * 3 + 1],
                _data[offset * 3 + 2],
            };
        }

        [[nodiscard]] double get_depth(const glm::vec2 &point) const override {
            return _depth;
        }
        [[nodiscard]] math_utils::aabb get_aabb() const override {
            return {glm::vec2{0, 0}, glm::vec2{1, 1}};  // from 0,0 to 1,1 bounding box(full screen)
        }
        [[nodiscard]] glm::vec2 get_position() const override {
            return {0.5, 0.5};                                       // center of the screen
        }
        [[nodiscard]] double get_rotation() const override {
            return 0;                                                      // does not support rotation
        }
        /*
         * set_position and set_rotation are not supported, default is return false
         */

    private:
        unsigned int _width;
        unsigned int _height;
        unsigned char* _data;
        float* _data_depth;
        double _depth;
        std::vector<std::shared_ptr<primitive::primitive_base>> _primitives;    // stored primitives
        renderer::orthogonal_renderer _renderer;                                // also need a renderer to draw primitives
    };

}

#endif //VISUALIME_CANVAS_H
