//
// Created by alexzms on 2023/12/10.
//

#ifndef VISUALIME_ENGINE_H
#define VISUALIME_ENGINE_H

#include "iostream"
#include "primitives.h"
#include "opengl_wrapper.h"
#include "vector"
#include "math_utils.h"
#include "renderer.h"
#include "canvas.h"


namespace visualime::engine {
    class engine_base {
    public:
        virtual ~engine_base() = default;
        virtual void render(const std::vector<std::shared_ptr<primitive::primitive_base>>& primitives,
                            std::shared_ptr<canvas::canvas_base> background_canvas) = 0;
    };

    class simple_engine: public engine_base {
    public:
        simple_engine(unsigned int width, unsigned int height, const math_utils::interval render_border = {0, 1}):
                    _width(width), _height(height), _render_border(render_border),
                    _renderer(width, height, render_border) {
            std::cout << "Engine initialized (" << width << "x" << height <<")" << std::endl;
            _data = new unsigned char[_width * _height * 3];
            _data_buffer = new unsigned char[_width * _height * 3];
            _depth = new float[_width * _height];
            memset(_data, 0, _width * _height * 3 * sizeof(unsigned char));
            memset(_data_buffer, 0, _width * _height * 3 * sizeof(unsigned char));
            memset(_depth, 0, _width * _height * sizeof(float));
        }
        ~simple_engine() override {
            delete[] _data;
            delete[] _data_buffer;
            delete[] _depth;
        }

        void render(const std::vector<std::shared_ptr<primitive::primitive_base>>& primitives,
                    std::shared_ptr<canvas::canvas_base> background_canvas) override {
            // copy from _background_canvas
            if (background_canvas == nullptr) {                                         // from canvas or blank
                memset(_data, 0, _width * _height * 3 * sizeof(unsigned char));
            } else {
                memcpy(_data, background_canvas->get_data(), _width * _height * 3 * sizeof(unsigned char));
            }
            memset(_depth, 0, _width * _height * sizeof(float));
            _renderer.render(primitives, _data_buffer, _depth);
            auto tmp = _data;                                           // swap buffer
            _data = _data_buffer;
            _data_buffer = tmp;
        }

        [[nodiscard]] unsigned int get_width() const { return _width; }
        [[nodiscard]] unsigned int get_height() const { return _height; }
        [[nodiscard]] unsigned char* get_data() const { return _data; }
    private:
        unsigned int _width;
        unsigned int _height;
        unsigned char* _data;
        unsigned char* _data_buffer;
        renderer::simple_renderer _renderer;
        math_utils::interval _render_border;
        float* _depth;
    };
}

#endif //VISUALIME_ENGINE_H
