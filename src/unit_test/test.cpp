#include "visualime/scenes.h"

#include "random"
#include "glm/glm.hpp"

static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_real_distribution<double> dist(0.0, 1.0);

glm::vec3 random_color() {
    return {static_cast<unsigned char>(dist(mt) * 255),
            static_cast<unsigned char>(dist(mt) * 255),
            static_cast<unsigned char>(dist(mt) * 255)};
}

auto main() -> int {
    std::cout << "Hello, World from Visualime!" << '\n';

    visualime::scene2d test_scene(800, 800, 1.0, false);
    test_scene.add_circle_normalized();
    auto rectangle_ptr = std::make_shared<visualime::primitive::solid_rectangle>(glm::vec2{0.5, 0.5}, 0.1, 0.1, 0.3, random_color());
    auto circle_ptr = std::make_shared<visualime::primitive::solid_circle>(glm::vec2{0.5, 0.5}, 0.1, 0.3, random_color());
    rectangle_ptr->set_rotation(3.5 * visualime::math_utils::PI / 8);
    test_scene.add_primitive(rectangle_ptr);
    size_t line_index = test_scene.add_line_normalized(random_color(), {0.1, 0.1}, {0.9, 0.9}, 0.1, 0.01);
    test_scene.launch(true);
//    test_scene.delete_primitive(0);
//    test_scene.delete_primitive(1);
//    test_scene.delete_primitive(2);
    double rotation = 0, end = 0.9;
    for (int i = 0; i != 600; ++i) {
        // rotate rectangle
        rotation += 0.1;
        end -= 0.01;
        if (!rectangle_ptr->set_rotation(rotation)) {
            std::cout << "Failed to rotate rectangle" << std::endl;
        }
        if (!test_scene.change_line_start_end(line_index, {0.1, 0.1}, {end, end})) {
            std::cout << "Failed to change line" << std::endl;
        }
        test_scene.refresh();
        // sleep
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!test_scene.is_running()) break;
    }

//    double x;
//    double y;
//    double z;
//    double r;
//    while (std::cin >> x >> y >> z >> r) {
//        test_scene.add_circle(random_color(), {x, y, z}, r);
//    }

    if (test_scene.get_run_thread().joinable()) {
        test_scene.get_run_thread().join();
    }
    std::cout << "Test scene finished" << std::endl;
    return 0;
}