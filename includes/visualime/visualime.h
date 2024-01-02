//
// Created by alexzms on 2023/12/16.
//

#ifndef VISUALIME_VISUALIME_H
#define VISUALIME_VISUALIME_H

#include <glad/glad.h>
#include "material.h"
#include "primitives.h"
#include "engine.h"
#include "opengl_wrapper.h"
#include "math_utils.h"
#include "scenes.h"
#include "renderer.h"
#include "Shader.h"
#include "canvas.h"
#include "cuda/cuda_helpers.cuh"
#include "cuda/cuda_opengl_wrapper.cuh"

namespace visualime {
    void test_success_import() {
        std::cout << "[success import]Hello, World from Visualime!" << std::endl;
    }
}

#endif //VISUALIME_VISUALIME_H
