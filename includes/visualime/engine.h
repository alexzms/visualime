//
// Created by alexzms on 2023/12/10.
//

#ifndef VISUALIME_ENGINE_H
#define VISUALIME_ENGINE_H

#include "iostream"
#include "primitives.h"
#include "opengl_wrapper.h"
#include "vector"

namespace visualime::engine {
    class engine_base {
    public:
        virtual ~engine_base() = default;
        virtual void render(std::vector<std::shared_ptr<primitive::primitive_base>> primitives) = 0;
    };
//    class simple_engine: public engine_base {
//    public:
//        simple_engine(unsigned int width, unsigned int height): _width(width), _height(height) {
//            _data = new unsigned char[_width * _height * 3];
//            _data_buffer = new unsigned char[_width * _height * 3];
//        }
//        ~simple_engine() override {
//            delete[] _data;
//            delete[] _data_buffer;
//        }
//
//        void add_primitive(const std::shared_ptr<primitive::primitive_base>& primitive) {
//            _primitives.emplace_back(primitive);
//        }
//
//        void add_primitive(std::shared_ptr<primitive::primitive_base>&& primitive) {
//            _primitives.emplace_back(std::move(primitive));
//        }
//
//        void remove_primitive(const std::shared_ptr<primitive::primitive_base>& primitive) {
//            _primitives.erase(std::remove(_primitives.begin(), _primitives.end(), primitive), _primitives.end());
//        }
//
//        void render() override {
//            for (int i = 0; i < _height; ++i) {
//                for (int j = 0; j < _width; ++j) {
//                    glm::vec3 color = {0, 0, 0};
//                    for (auto& primitive: _primitives) {
//                        color += primitive->should_show(_width, _height);
//                    }
//                    _data_buffer[i * _width + j] = static_cast<unsigned char>(color.r);
//                    _data_buffer[i * _width + j + 1] = static_cast<unsigned char>(color.g);
//                    _data_buffer[i * _width + j + 2] = static_cast<unsigned char>(color.b);
//                }
//            }
//            auto tmp = _data;                               // swap buffer
//            _data = _data_buffer;
//            _data_buffer = tmp;
//        }
//
//        [[nodiscard]] unsigned int get_width() const { return _width; }
//        [[nodiscard]] unsigned int get_height() const { return _height; }
//        [[nodiscard]] unsigned char* get_data() const { return _data; }
//
//    private:
//        unsigned int _width;
//        unsigned int _height;
//        unsigned char* _data;
//        unsigned char* _data_buffer;
//        std::vector<std::shared_ptr<primitive::primitive_base>> _primitives;
//    };

    class simple_engine: public engine_base {
    public:
        simple_engine(unsigned int width, unsigned int height): _width(width), _height(height) {
            std::cout << "Engine initialized (" << width << "x" << height <<")" << std::endl;
            _data = new unsigned char[_width * _height * 3];
            _data_buffer = new unsigned char[_width * _height * 3];
        }
        ~simple_engine() override {
            delete[] _data;
            delete[] _data_buffer;
        }

        void render(std::vector<std::shared_ptr<primitive::primitive_base>> primitives) override {
            for (int i = 0; i != _height; ++i) {
                for (int j = 0; j != _width; ++j) {
                    glm::vec3 color = {0, 0, 0};
                    for (auto& primitive: primitives) {
                        color += primitive->should_show(j / static_cast<double>(_width),
                                                        i / static_cast<double>(_height));
                    }
                    unsigned offset = i * _width + j;
                    _data_buffer[offset * 3] = static_cast<unsigned char>(color.r);
                    _data_buffer[offset * 3 + 1] = static_cast<unsigned char>(color.g);
                    _data_buffer[offset * 3 + 2] = static_cast<unsigned char>(color.b);
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
    };
}

#endif //VISUALIME_ENGINE_H
