#ifndef VISUALIME_LIBRARY_H
#define VISUALIME_LIBRARY_H

#include <glad/glad.h>
#include <mutex>
#include "material.h"
#include "primitives.h"
#include "engine.h"
#include "opengl_wrapper.h"
#include "thread"
#include "time.h"


namespace visualime {
    void test_success_import() {
        std::cout << "Hello, World from Visualime!" << std::endl;
    }
    class test_scene2d {
    public:
        test_scene2d(unsigned int width,unsigned int height,double super_sampling_ratio = 1.0,bool fullscreen = false):
                _engine(static_cast<unsigned>(width * super_sampling_ratio) ,
                        static_cast<unsigned>(height * super_sampling_ratio)),
                _opengl_wrapper(static_cast<GLsizei>(width),
                                static_cast<GLsizei>(height),
                                fullscreen, super_sampling_ratio) {}
        ~test_scene2d() {
            if (_run_thread.joinable()) {
                _run_thread.join();
            }
        }

        bool should_quit() {
            return _opengl_wrapper.should_close();
        }

        void run() {
            std::cout << "Calling run in test_scene2d" << std::endl;
            _opengl_wrapper.init();
            double _prev_time = glfwGetTime();
            unsigned counter = 0;
            while (!_opengl_wrapper.should_close()) {
//                double _curr_time = glfwGetTime();
//                if (_curr_time - _prev_time > 1) {
//                    std::cout << "FPS: " << counter << std::endl;
//                    counter = 0;
//                    _prev_time = _curr_time;
//                }
                _mutex.lock();
                if (_scene_changed) {
                    _engine.render(_primitives);
                    _scene_changed = false;
                }
                _mutex.unlock();
                _opengl_wrapper.render_call(_engine.get_data());
                _opengl_wrapper.swap_buffers();
                _opengl_wrapper.poll_events();
                counter += 1;
            }
            _opengl_wrapper.destroy();
        }

        void launch() {
            _run_thread = std::thread(&test_scene2d::run, this);
        }

        size_t add_primitive(const std::shared_ptr<primitive::primitive_base>& primitive) {
            _mutex.lock();
            _scene_changed = true;
            _primitives.emplace_back(primitive);
            _mutex.unlock();
            return _primitives.size() - 1;
        }

        void remove_primitive(size_t index) {
            _mutex.lock();
            _scene_changed = true;
            _primitives.erase(_primitives.begin() + index);
            _mutex.unlock();
        }

        void add_circle(glm::vec3 color = {200, 60, 50},
                        glm::vec3 position = {0.5, 0.3, 0.1},
                        double radius = 0.1) {

            add_primitive(std::make_shared<primitive::solid_circle>(color, position, radius));
        }

        [[nodiscard]] std::thread& get_run_thread() { return _run_thread; }
    private:
        std::vector<std::shared_ptr<primitive::primitive_base>> _primitives;
        engine::simple_engine _engine;
        opengl_wrapper::simple_opengl_wrapper _opengl_wrapper;
        bool _scene_changed = false;
        std::thread _run_thread;
        std::mutex _mutex;
        double super_sampling_ratio = 1;
    };
}
#endif //VISUALIME_LIBRARY_H
