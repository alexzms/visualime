#ifndef VISUALIME_LIBRARY_H
#define VISUALIME_LIBRARY_H

#include <mutex>
#include "material.h"
#include "primitives.h"
#include "engine.h"
#include "opengl_wrapper.h"
#include "thread"
#include <ctime>
#include <functional>
#include "glm/glm.hpp"
#include "canvas.h"
#include "chrono"
#include "cuda/cuda_opengl_wrapper.cuh"

namespace visualime::scene {
    class scene2d {
    public:
        std::function<void(const glm::vec2&)> on_mouse_left_click = [](const glm::vec2& pos){
            std::cout << "[scene2d:default] left mouse clicked at " << pos.x << ", " << pos.y << std::endl;
        };
        std::function<void(const glm::vec2&)> on_mouse_right_click = [](const glm::vec2& pos){
            std::cout << "[scene2d:default] right mouse clicked at " << pos.x << ", " << pos.y << std::endl;
        };


        scene2d(GLsizei width, GLsizei height, double super_sampling_ratio = 1.0, bool fullscreen = false):
                _width(width), _height(height),
                _background_canvas(std::make_shared<canvas::fullscreen_orthogonal_canvas>
                                    (static_cast<unsigned>(width * super_sampling_ratio),
                                     static_cast<unsigned>(height * super_sampling_ratio))),
                _opengl_wrapper(static_cast<GLsizei>(width),
                                static_cast<GLsizei>(height),
                                fullscreen, super_sampling_ratio),
                _engine(static_cast<unsigned>(width * super_sampling_ratio) ,
                        static_cast<unsigned>(height * super_sampling_ratio),
                        _opengl_wrapper.get_internal_data()) {}
        ~scene2d() {
            if (_run_thread.joinable()) {
                _run_thread.join();
            }
        }

        void run(bool show_fps = false, bool show_performance = false, unsigned int lock_framerate = 60) {
            std::cout << "[scene2d] calling run(start thread)" << std::endl;
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
                auto frame_start = std::chrono::high_resolution_clock::now();
                _mutex.lock();
                if (_scene_changed) {
                    _scene_changed = false;
                    auto render_start = std::chrono::high_resolution_clock::now();
                    _engine.render(_primitives, _background_canvas);
                    auto render_end = std::chrono::high_resolution_clock::now();
                    if (show_performance) {
                        std::cout << "[scene][performance] _engine.render(): "
                                  << std::chrono::duration_cast<std::chrono::microseconds>(render_end - render_start).count()
                                  << " us" << std::endl;
                    }
                    auto render_call_start = std::chrono::high_resolution_clock::now();
                    _opengl_wrapper.render_call_internal_data();
                    _opengl_wrapper.swap_buffers();
                    auto render_call_end = std::chrono::high_resolution_clock::now();
                    if (show_performance) {
                        std::cout << "[scene][performance] _opengl_wrapper.render_call(): "
                        << std::chrono::duration_cast<std::chrono::microseconds>
                                (render_call_end - render_call_start).count() << " us" << std::endl;
                    }
                }
                _mutex.unlock();

                if (_opengl_wrapper.get_left_mouse_click(mouse_click_pos))
                    on_mouse_left_click(mouse_click_pos);

                if (_opengl_wrapper.get_right_mouse_click(mouse_click_pos))
                    on_mouse_right_click(mouse_click_pos);
                _opengl_wrapper.poll_events();
                auto frame_end = std::chrono::high_resolution_clock::now();
                if (lock_framerate > 0) {                                      // lock_frame <= 0: unlimited
                    auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>
                            (frame_end - frame_start).count();
                    if (frame_duration < 1000000 / lock_framerate) {
                        std::this_thread::sleep_for(std::chrono::microseconds
                                (math_utils::sleep_time_corrected(lock_framerate, frame_duration)));
                    }
                }
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

        void launch(bool show_fps = false, bool show_performance = false, unsigned int lock_framerate = 60) {
            _running = true;
            _run_thread = std::thread(&scene2d::run, this, show_fps, show_performance, lock_framerate);
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

        size_t add_circle_normalized(glm::vec<3, unsigned char> color = {200, 60, 50},
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

        size_t add_line_normalized(glm::vec<3, unsigned char> color = {200, 60, 50},
                      glm::vec2 start = {0.3, 0.5},
                      glm::vec2 end = {0.5, 0.5},
                      double depth = 0.1,
                      double height = 0.1) {
            double rotation = std::atan2(end.y - start.y, end.x - start.x);
            auto rectangle_ptr = std::make_shared<primitive::solid_rectangle>((start + end) * 0.5f, depth, glm::length(end - start), height, color, rotation);
            return add_primitive(rectangle_ptr);
        }

        bool draw_pixel(const glm::vec<2, unsigned>& unnormalized_pos, const glm::vec<3, unsigned char>& color) {
            return _background_canvas->draw_pixel(unnormalized_pos, color);
        }

        bool draw_primitive(const std::shared_ptr<primitive::primitive_base>& primitive) {
            return _background_canvas->draw_primitive(primitive);
        }

        bool draw_primitive_and_store(const std::shared_ptr<primitive::primitive_base>& primitive) {
            return _background_canvas->draw_primitive_and_store(primitive);
        }

        [[nodiscard]] std::thread& get_run_thread() { return _run_thread; }
    private:
        GLsizei _width;
        GLsizei _height;
        std::vector<std::shared_ptr<primitive::primitive_base>> _primitives;
        visualime::opengl_wrapper::simple_opengl_wrapper _opengl_wrapper;     // declare before _engine, for .get_data()
        engine::orthogonal_engine _engine;
        bool _scene_changed = false;
        bool _running = false;
        std::thread _run_thread;
        std::mutex _mutex;
        double super_sampling_ratio = 1;
        std::shared_ptr<canvas::fullscreen_orthogonal_canvas> _background_canvas;
    };

#ifdef VISUALIME_USE_CUDA
    class canvas_scene_cuda_2d {
    public:
        std::function<void(const glm::vec2&)> on_mouse_left_click = [](const glm::vec2& pos){
            std::cout << "[canvas_scene_cuda_2d:default] left mouse clicked at " << pos.x << ", " << pos.y << std::endl;
        };
        std::function<void(const glm::vec2&)> on_mouse_right_click = [](const glm::vec2& pos){
            std::cout << "[canvas_scene_cuda_2d:default] right mouse clicked at " << pos.x << ", " << pos.y << std::endl;
        };

        canvas_scene_cuda_2d(GLsizei width, GLsizei height, double super_sampling_ratio = 1.0, bool fullscreen = false):
                _width(width), _height(height),
                _opengl_wrapper(_width, _height, fullscreen, super_sampling_ratio) {}

        [[nodiscard]] uchar3* get_d_ptr() const { return _opengl_wrapper.get_d_ptr(); }
        [[nodiscard]] size_t get_d_ptr_size() const { return _opengl_wrapper.get_d_ptr_size(); }

        void run(bool show_fps = false, unsigned int lock_framerate = 60) {
            std::cout << "[canvas_scene_cuda_2d] calling run(start thread)" << std::endl;
            _opengl_wrapper.init();
            _running = true;
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
                auto frame_start = std::chrono::high_resolution_clock::now();
                _mutex.lock();
                if (_scene_changed) {                                       // memory is managed by _opengl_wrapper
                    _opengl_wrapper.render_call();                          // we should change the _opengl_wrapper.get_d_ptr()
                    _opengl_wrapper.swap_buffers();                         // on the outside, then call .refresh()
                    _scene_changed = false;
                }
                _mutex.unlock();
                if (_opengl_wrapper.get_left_mouse_click(mouse_click_pos))
                    on_mouse_left_click(mouse_click_pos);

                if (_opengl_wrapper.get_right_mouse_click(mouse_click_pos))
                    on_mouse_right_click(mouse_click_pos);
                _opengl_wrapper.poll_events();
                auto frame_end = std::chrono::high_resolution_clock::now();
                if (lock_framerate > 0) {                                     // lock_frame <= 0: unlimited
                    auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>
                            (frame_end - frame_start).count();
                    if (frame_duration < 1000000 / lock_framerate) {
                        std::this_thread::sleep_for(std::chrono::microseconds
                                                            (math_utils::sleep_time_corrected(lock_framerate, frame_duration)));
                    }
                }
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

        void launch(bool show_fps = false, unsigned int lock_framerate = 60) {
            _run_thread = std::thread(&canvas_scene_cuda_2d::run, this, show_fps, lock_framerate);
        }
        [[nodiscard]] std::thread& get_run_thread() { return _run_thread; }
        [[nodiscard]] bool is_running() const { return _running; }
        void wait_for_running(unsigned int milliseconds = 100) const {
            while (!_running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
            }
        }


    private:
        GLsizei _width;
        GLsizei _height;
        visualime::opengl_wrapper::cuda_opengl_wrapper _opengl_wrapper;
        std::thread _run_thread;
        std::mutex _mutex;
        bool _running = false;
        bool _scene_changed = false;
    };
#endif //VISUALIME_USE_CUDA
}
#endif //VISUALIME_LIBRARY_H
