#ifndef VISUALIME_LIBRARY_H
#define VISUALIME_LIBRARY_H

#include <glad/glad.h>
#include <mutex>
#include "material.h"
#include "primitives.h"
#include "engine.h"
#include "opengl_wrapper.h"
#include "thread"
#include <ctime>
#include "glm/glm.hpp"
#include "glm/gtx/string_cast.hpp"


namespace visualime {
    class scene2d {
    public:
        std::function<void(const glm::vec2&)> on_mouse_left_click = [](const glm::vec2& pos){
            std::cout << "[scene2d:default] left mouse clicked at " << glm::to_string(pos) << std::endl;
        };
        std::function<void(const glm::vec2&)> on_mouse_right_click = [](const glm::vec2& pos){
            std::cout << "[scene2d:default] right mouse clicked at " << glm::to_string(pos) << std::endl;
        };


        scene2d(unsigned int width, unsigned int height, double super_sampling_ratio = 1.0, bool fullscreen = false):
                _width(width), _height(height),
                _engine(static_cast<unsigned>(width * super_sampling_ratio) ,
                        static_cast<unsigned>(height * super_sampling_ratio)),
                _opengl_wrapper(static_cast<GLsizei>(width),
                                static_cast<GLsizei>(height),
                                fullscreen, super_sampling_ratio) {}
        ~scene2d() {
            if (_run_thread.joinable()) {
                _run_thread.join();
            }
        }

        bool should_quit() {
            return _opengl_wrapper.should_close();
        }

        void run(bool show_fps = false) {
            std::cout << "Calling run in scene2d" << std::endl;
            _opengl_wrapper.init();
            double _prev_time = glfwGetTime();
            unsigned counter = 0;
            glm::vec2 mouse_click_pos;
            while (!_opengl_wrapper.should_close()) {
                if (show_fps) {
                    double _curr_time = glfwGetTime();
                    if (_curr_time - _prev_time > 1) {
                        std::cout << "FPS: " << counter << std::endl;
                        counter = 0;
                        _prev_time = _curr_time;
                    }
                }
                _mutex.lock();
                if (_scene_changed) {
                    _engine.render(_primitives);
                    _scene_changed = false;
                }
                _mutex.unlock();

                if (_opengl_wrapper.get_left_mouse_click(mouse_click_pos))
                    on_mouse_left_click(mouse_click_pos);

                if (_opengl_wrapper.get_right_mouse_click(mouse_click_pos))
                    on_mouse_right_click(mouse_click_pos);

                _opengl_wrapper.render_call(_engine.get_data());
                _opengl_wrapper.swap_buffers();
                _opengl_wrapper.poll_events();
                counter += 1;
            }
            _opengl_wrapper.destroy();
            _running = false;
        }

        void refresh() {
            _mutex.lock();
            _scene_changed = true;
            _mutex.unlock();
        }

        void launch(bool show_fps = false) {
            _running = true;
            _run_thread = std::thread(&scene2d::run, this, show_fps);
        }

        [[nodiscard]] bool is_running() const {
            return _running;
        }

        size_t add_primitive(const std::shared_ptr<primitive::primitive_base>& primitive) {
            _mutex.lock();
            _primitives.emplace_back(primitive);
            _mutex.unlock();
            return _primitives.size() - 1;
        }

        void delete_primitive(size_t index) {
            if (index >= _primitives.size()) {
                std::cout << "[visualime][delete_primitive] index out of range" << std::endl;
                return;
            }
            _mutex.lock();
            _primitives[index] = nullptr;
            _mutex.unlock();
        }

        void change_primitive_position_normalized(size_t index, const glm::vec2& position) {

            if (index >= _primitives.size()) {
                std::cout << "[visualime][change_primitive_position] index out of range" << std::endl;
                return;
            }
            _mutex.lock();
            bool success = _primitives[index]->set_position(position);
            if (!success) {
                std::cout << "[visualime][change_primitive_position] failed to change position" << std::endl;
            }
            _mutex.unlock();
        }

        size_t add_circle_normalized(glm::vec3 color = {200, 60, 50},
                        glm::vec2 position = {0.3, 0.5},
                        double depth = 0.1,
                        double radius = 0.1) {
            return add_primitive(std::make_shared<primitive::solid_circle>(position, depth, radius, color));
        }

        bool change_line_start_end(size_t index, glm::vec2 start, glm::vec2 end) {
            if (index >= _primitives.size()) {
                std::cout << "[visualime][change_line_start_end] index out of range" << std::endl;
                return false;
            }
            auto rectangle_ptr = std::dynamic_pointer_cast<primitive::solid_rectangle>(_primitives[index]);
            if (rectangle_ptr == nullptr) {
                std::cout << "[visualime][change_line_start_end] not a line" << std::endl;
                return false;
            }
            double rotation = std::atan2(end.y - start.y, end.x - start.x);
            rectangle_ptr->set_rotation(rotation);
            rectangle_ptr->set_position((start + end) * 0.5f);
            rectangle_ptr->set_width(glm::length(end - start));
            return true;
        }

        size_t add_line_normalized(glm::vec3 color = {200, 60, 50},
                      glm::vec2 start = {0.3, 0.5},
                      glm::vec2 end = {0.5, 0.5},
                      double depth = 0.1,
                      double height = 0.1) {
            double rotation = std::atan2(end.y - start.y, end.x - start.x);
            auto rectangle_ptr = std::make_shared<primitive::solid_rectangle>((start + end) * 0.5f, depth, glm::length(end - start), height, color, rotation);
            return add_primitive(rectangle_ptr);
        }

        [[nodiscard]] std::thread& get_run_thread() { return _run_thread; }
    private:
        unsigned int _width;
        unsigned int _height;
        std::vector<std::shared_ptr<primitive::primitive_base>> _primitives;
        engine::simple_engine _engine;
        opengl_wrapper::simple_opengl_wrapper _opengl_wrapper;
        bool _scene_changed = false;
        bool _running = false;
        std::thread _run_thread;
        std::mutex _mutex;
        double super_sampling_ratio = 1;
    };
}
#endif //VISUALIME_LIBRARY_H
