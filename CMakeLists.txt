cmake_minimum_required(VERSION 3.26)
project(visualime LANGUAGES CXX CUDA)
project(unit_tests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(TARGETS visualime unit_tests)

set_source_files_properties(
    src/glad.c
    includes/visualime/primitives.h
    includes/visualime/material.h
    includes/visualime/engine.h
    includes/visualime/opengl_wrapper.h
    includes/visualime/Shader.h
    includes/visualime/math_utils.h
    includes/visualime/visualime.h
    includes/visualime/renderer.h
    includes/visualime/scenes.h
    includes/visualime/canvas.h
        PROPERTIES
    LANGUAGE CXX
)

set_source_files_properties(
    includes/visualime/cuda/cuda_opengl_wrapper.cuh
    includes/visualime/cuda/use_cuda.h
    includes/visualime/cuda/cuda_helpers.cuh
    src/cuda/cuda_opengl_wrapper.cu
    PROPERTIES
    LANGUAGE CUDA
)

set(SOURCES
    src/glad.c
    src/cuda/cuda_opengl_wrapper.cu
    src/Shader.cpp
)

set(HEADERS
        includes/visualime/visualime.h
#        includes/visualime/primitives.h
#        includes/visualime/material.h
#        includes/visualime/engine.h
#        includes/visualime/opengl_wrapper.h
#        includes/visualime/math_utils.h

#        includes/visualime/renderer.h
#        includes/visualime/scenes.h
#        includes/visualime/canvas.h
#        includes/visualime/cuda/cuda_opengl_wrapper.cuh
#        includes/visualime/cuda/use_cuda.h
#        includes/visualime/cuda/cuda_helpers.cuh
)

add_library(visualime ${SOURCES} ${HEADERS} )
#target_compile_definitions(visualime PRIVATE VISUALIME_USE_CUDA)

add_executable(unit_tests
        src/glad.c
        src/unit_test/test.cpp
        src/cuda/cuda_opengl_wrapper.cu
        src/cuda/cuda_opengl_wrapper.cu
        src/Shader.cpp
)

include_directories(E:/ComputerGraphics/libraries/OpenGL/includes)
include_directories(E:/ComputerGraphics/libraries/Utilities/includes)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/include")
include_directories(includes)
include_directories(includes/visualime)
link_directories(E:/ComputerGraphics/libraries/OpenGL/lib)
link_directories(E:/ComputerGraphics/libraries/Utilities/lib)
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64")


target_link_libraries(unit_tests PUBLIC glfw3.dll opengl32.lib)
target_link_libraries(unit_tests PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)
target_link_libraries(visualime PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        cudart
)

set_target_properties(visualime PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(unit_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    foreach (target ${TARGETS})
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endforeach ()
endif()

#install(TARGETS visualime
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib
#        RUNTIME DESTINATION bin
#)
