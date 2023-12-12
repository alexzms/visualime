# Visualime - A Simple 2D Visualization Library for C++

## Done
- [x] Circle primitive
- [x] OpenGL rendering
- [x] Depth buffer
- [x] OpenMP parallelization

## TODO
- [ ] *Box primitive
- [ ] Triangle primitive
- [ ] Triangle mesh primitive
- [ ] *CUDA parallelization

(*) means it's highly wanted.

## Dependencies
- [CMake](https://cmake.org)
- [OpenGL](https://www.opengl.org)
- [GLFW3](https://www.glfw.org)
- [OpenMP](https://www.openmp.org) (optional)
- [CUDA](https://developer.nvidia.com/cuda-downloads) (optional)

## Installation
### Windows(Only tested on Windows 11)
1. Install [CMake](https://cmake.org/download/) and [Visual Studio](https://visualstudio.microsoft.com/ko/downloads/).
2. Install [Git](https://git-scm.com/downloads).
3. Install [CUDA](https://developer.nvidia.com/cuda-downloads) if you want to use CUDA.
4. Install [OpenMP](https://www.openmp.org/resources/openmp-compilers-tools/) if you want to use OpenMP.
5. Install [GLFW3](https://www.glfw.org/download.html).
6. Clone this repository.
7. Open CMake GUI and set source code path and build path.
8. Click `Configure` button and select your Visual Studio version.
9. Click `Generate` button.
10. Open `Visualime.sln` and build `ALL_BUILD` project.
11. Find `Visualime.dll` and `Visualime.lib` in `build/Debug` or `build/Release` directory.
12. Copy `Visualime.dll` and `Visualime.lib` to your project directory.
13. Copy `include` directory to your project directory.
14. Add `Visualime.lib` to your project.
15. Done!

## Usage
```cpp
#include "visualime/library.h"

#include "random"

static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_real_distribution<double> dist(0.0, 1.0);

glm::vec3 random_color() {
    return {static_cast<unsigned char>(dist(mt) * 255),
            static_cast<unsigned char>(dist(mt) * 255),
            static_cast<unsigned char>(dist(mt) * 255)};
}

int main() {
    std::cout << "Hello, World from Visualime!" << std::endl;

    visualime::test_scene2d test_scene(800, 800, 1.0, false);
    test_scene.add_circle();
    test_scene.launch();

    double x, y, z, r;
    while (std::cin >> x >> y >> z >> r) {
        test_scene.add_circle(random_color(), {x, y, z}, r);
    }

    if (test_scene.get_run_thread().joinable()) {
        test_scene.get_run_thread().join();
    }
    std::cout << "Test scene finished" << std::endl;
    return 0;
}
```
This can be found in `src/unit_test/test.cpp`.