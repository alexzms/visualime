//
// Created by alexzms on 2023/12/10.
//

#ifndef VISUALIME_PRIMITIVES_H
#define VISUALIME_PRIMITIVES_H

#include "memory"
#include "material.h"
#include "glm/glm.hpp"

namespace visualime::primitive {
    class primitive_base {
    public:
        virtual ~primitive_base() = default;
        [[nodiscard]] virtual glm::vec3 show_color(double width, double height) const = 0;
        [[nodiscard]] virtual float get_depth(double width, double height) const = 0;
    };

    class solid_circle: public primitive_base {
    public:

        explicit solid_circle(glm::vec3 color = {200, 60, 50},
                     glm::vec3 position = {0, 0, 0},
                     double radius = 0.5):
                     _position(std::move(position)),
                     _material(std::make_shared<material::solid_color>(color)),
                     _radius(radius) {}
        [[nodiscard]] glm::vec3 show_color(double width, double height) const override {
            if (should_show(width, height)) {
                return _material->get_color(width, height);
            }
            return {0, 0, 0};
        }
        [[nodiscard]] float get_depth(double width, double height) const override {
            if (should_show(width, height)) {
                return _position.z;
            } else {
                return 0;
            }
        }

    private:
        glm::vec3 _position;                                    // position is vec3 for overlapping
        std::shared_ptr<material::material_base> _material;
        double _radius;

        [[nodiscard]] bool should_show(double width, double height) const {
            glm::vec2 center = {_position.x, _position.y};
            auto distance = glm::distance(center, {width, height});
            return distance < _radius;
        }
    };
}


#endif //VISUALIME_PRIMITIVES_H
