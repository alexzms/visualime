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

    visualime::test_scene2d test_scene(800, 800, 2.0, false);
    test_scene.add_circle();
    test_scene.launch();

    double x, y, r;
    while (std::cin >> x >> y >> r) {
        test_scene.add_circle(random_color(), {x, y, 0}, r);
    }


    if (test_scene.get_run_thread().joinable()) {
        test_scene.get_run_thread().join();
    }
    std::cout << "Test scene finished" << std::endl;
    return 0;
}