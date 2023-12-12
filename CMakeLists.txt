cmake_minimum_required(VERSION 3.26)
project(visualime)
project(unit_tests)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP)
set(TARGETS visualime unit_tests)

include_directories(E:/ComputerGraphics/libraries/OpenGL/includes)
include_directories(E:/ComputerGraphics/libraries/Utilities/includes)
include_directories(includes)
link_directories(E:/ComputerGraphics/libraries/OpenGL/lib)
link_directories(E:/ComputerGraphics/libraries/Utilities/lib)

add_library(visualime src/library.cpp src/glad.c
        includes/visualime/primitives.h
        includes/visualime/material.h
        includes/visualime/engine.h
        includes/visualime/opengl_wrapper.h
        includes/visualime/Shader.h)

add_executable(unit_tests src/unit_test/test.cpp src/glad.c)
target_link_libraries(unit_tests PUBLIC glfw3.dll opengl32.lib)

if(OpenMP_CXX_FOUND)
    foreach (target ${TARGETS})
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endforeach ()
endif()
