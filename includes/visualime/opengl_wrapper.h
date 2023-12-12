//
// Created by alexzms on 2023/12/10.
//

#ifndef VISUALIME_OPENGL_WRAPPER_H
#define VISUALIME_OPENGL_WRAPPER_H

#include "iostream"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "GLFW/glfw3.h"
//#include "glad/glad.h"
#include "Shader.h"
#include "functional"

namespace visualime::opengl_wrapper {
    class simple_opengl_wrapper {
    public:
        simple_opengl_wrapper(GLsizei width, GLsizei height, bool fullscreen, double super_sampling_ratio = 1.0):
        _width(width), _height(height), _fullscreen(fullscreen), _super_sampling_ratio(super_sampling_ratio) {
            _vertices = new float[5 * 4]{
                // positions          // texture coords
                1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
                1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // bottom right
                -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // bottom left
                -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // top left
            };
            _indices = new unsigned int[6]{
                0, 1, 3, // first triangle
                1, 2, 3  // second triangle
            };
            _vertices_size = 5 * 4 * sizeof(float);
            _indices_size = 6 * sizeof(unsigned int);
            _data = new unsigned char[(size_t)(_width * _height * 3 * super_sampling_ratio * super_sampling_ratio)];
            vert_shader = new char[1024]{
                "#version 460\n"
                "layout (location = 0) in vec3 aPos;\n"
                "layout (location = 1) in vec2 aTexCoord;\n"
                "\n"
                "out vec2 TexCoord;\n"
                "\n"
                "void main()\n"
                "{\n"
                "    gl_Position = vec4(aPos, 1.0);\n"
                "    TexCoord = aTexCoord;\n"
                "}\n"
            };
            frag_shader = new char[1024]{
                "#version 460\n"
                "\n"
                "in vec2 TexCoord;\n"
                "\n"
                "uniform sampler2D tex;\n"
                "\n"
                "// simply output the color of the texture as fragment color\n"
                "out vec4 FragColor;\n"
                "\n"
                "void main()\n"
                "{\n"
                "    FragColor = texture(tex, TexCoord);\n"
                "}\n"
            };
        }
        ~simple_opengl_wrapper() {
            std::cout << "[wrapper]Destroying simple_opengl_wrapper" << std::endl;
            delete[] _vertices;
            delete[] _indices;
            delete[] _data;
            delete[] vert_shader;
            delete[] frag_shader;
        }


        [[nodiscard]] GLFWwindow *get_window() const { return _window; }

        void init() {                                                 // init opengl(thread-sensitive context)
            std::cout << "[wrapper]Launching OpenGL(" << _width << "x" << _height << ") with super sampling "
                        << _super_sampling_ratio << "x" << std::endl;

            glfwInit();
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            _window = glfwCreateWindow(_width, _height, "Visualime",
                                  _fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);
            if (_window == nullptr) {
                std::cout << "[wrapper]Failed to create GLFW _window" << std::endl;
                glfwTerminate();
                exit(-1);
            }
            glfwMakeContextCurrent(_window);
            glfwSetFramebufferSizeCallback(_window,
                                           [](GLFWwindow* window, int width, int height) {
                        glViewport(0, 0, width, height);
                    });
            glfwSetWindowUserPointer(_window, this);
            glfwSetCursorPosCallback(_window, static_mouse_callback);
            glfwSetScrollCallback(_window, static_scroll_callback);
            if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
                std::cout << "[wrapper]Failed to initialize GLAD" << std::endl;
                exit(-1);
            }
            glViewport(0, 0, _width, _height);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            _shader = Shader(vert_shader, frag_shader);
            _shader.use();
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, _vertices_size, _vertices, GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indices_size, _indices, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)nullptr);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);

            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(
                    GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)(_width * _super_sampling_ratio),
                    (GLsizei)(_height * _super_sampling_ratio), 0, GL_RGB, GL_UNSIGNED_BYTE, _data
            );
            glBindTexture(GL_TEXTURE_2D, 0);
            initialized = true;
            std::cout << "[wrapper]OpenGL Launch Success" << std::endl;
        }

        void destroy() {
            if (initialized) {
                glDeleteVertexArrays(1, &VAO);
                glDeleteBuffers(1, &VBO);
                glDeleteBuffers(1, &EBO);
                glDeleteTextures(1, &texture);
                glfwDestroyWindow(_window);
                glfwTerminate();
                initialized = false;
            }
        }

        void render_call(unsigned char* data) {
            if (!initialized) { std::cout << "[call render_call()]OpenGL not initialized" << std::endl; return;}
            process_input(_window);
            glClearColor((GLfloat)(sin(glfwGetTime()) + 1)/2, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            _shader.use();
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                    GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)(_width * _super_sampling_ratio),
                    (GLsizei)(_height * _super_sampling_ratio), 0, GL_RGB, GL_UNSIGNED_BYTE, data
            );
            _shader.setInt("tex", 0);
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

            glBindTexture(GL_TEXTURE_2D, 0);
            glBindVertexArray(0);
        }

        void swap_buffers() {
            if (!initialized) { std::cout << "[wrapper][call swap_buffer()]OpenGL not initialized" << std::endl; return;}
            glfwSwapBuffers(_window);
        }

        void poll_events() const {
            if (!initialized) { std::cout << "[wrapper][call pull_events()]OpenGL not initialized" << std::endl; return;}
            glfwPollEvents();
        }

        bool should_close() {
            if (!initialized) { return false; }
            return glfwWindowShouldClose(_window);
        }

    private:
        GLsizei _width;
        GLsizei _height;
        double _super_sampling_ratio;
        unsigned char* _data;
        bool _fullscreen;
        Shader _shader;
        float* _vertices;                                           // will init asa rectangle in front of the camera
        unsigned int* _indices;
        GLsizei _vertices_size, _indices_size;
        char* vert_shader;
        char* frag_shader;
        GLuint VAO{}, VBO{}, EBO{}, texture{};
        GLFWwindow* _window{};
        double last_width{}, last_height{}, scroll_offset{};
        bool initialized = false;

        std::function<void(GLFWwindow *window, double x, double y)> mouse_callback =
                [this](GLFWwindow *window, double x, double y) {
                    // get mouse position and write to lastX and lastY
                    last_width = x;
                    last_height = y;
                };
        std::function<void(GLFWwindow *window, double x, double y)> scroll_callback =
                [this](GLFWwindow *window, double x, double y) {
                    scroll_offset = y;
                };

        static void process_input(GLFWwindow *window) {
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
                glfwSetWindowShouldClose(window, true);
        }

        static void static_mouse_callback(GLFWwindow *window, double x, double y) {
            auto *wrapper = static_cast<simple_opengl_wrapper*>(glfwGetWindowUserPointer(window));
            wrapper->mouse_callback(window, x, y);
        }

        static void static_scroll_callback(GLFWwindow *window, double x, double y) {
            auto *wrapper = static_cast<simple_opengl_wrapper*>(glfwGetWindowUserPointer(window));
            wrapper->scroll_callback(window, x, y);
        }
    };
}

#endif //VISUALIME_OPENGL_WRAPPER_H
