//
// Created by alexzms on 2023/12/10.
//

#ifndef VISUALIME_ENGINE_H
#define VISUALIME_ENGINE_H

#include "iostream"
#include "primitives.h"
#include "opengl_wrapper.h"
#include "vector"
#include "omp.h"

namespace visualime::engine {
    class engine_base {
    public:
        virtual ~engine_base() = default;
        virtual void render(std::vector<std::shared_ptr<primitive::primitive_base>> primitives) = 0;
    };

    class simple_engine: public engine_base {
    public:
        simple_engine(unsigned int width, unsigned int height): _width(width), _height(height) {
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

        void render(std::vector<std::shared_ptr<primitive::primitive_base>> primitives) override {
            memset(_data_buffer, 0, _width * _height * 3 * sizeof(unsigned char));
            memset(_depth, 0, _width * _height * sizeof(float));
            // openmp
            #pragma omp parallel for
            for (int i = 0; i < _height; ++i) {
                for (int j = 0; j != _width; ++j) {
                    glm::vec3 color = {0, 0, 0};
                    unsigned offset = i * _width + j;
                    bool update = false;
                    for (auto& primitive: primitives) {
                        glm::vec3 temp_color = primitive->show_color(j / static_cast<double>(_width),
                                                                     i / static_cast<double>(_height));
                        float depth = primitive->get_depth(
                                j / static_cast<double>(_width),
                                i / static_cast<double>(_height));
                        if (depth > _depth[offset]) {
                            _depth[offset] = depth;
                            color = temp_color;
                            update = true;
                        }
                    }
                    if (update) {
                        _data_buffer[offset * 3] = static_cast<unsigned char>(color.r);
                        _data_buffer[offset * 3 + 1] = static_cast<unsigned char>(color.g);
                        _data_buffer[offset * 3 + 2] = static_cast<unsigned char>(color.b);
                    }
                }
            }
            auto tmp = _data;                               // swap buffer
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
        float* _depth;
    };
}

#endif //VISUALIME_ENGINE_H
