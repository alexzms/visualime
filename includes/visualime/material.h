//
// Created by alexzms on 2023/12/10.
//

#ifndef VISUALIME_MATERIAL_H
#define VISUALIME_MATERIAL_H

#include "glm/glm.hpp"

namespace visualime::material {
    class material_base {
    public:
        virtual ~material_base() = default;
        [[nodiscard]] virtual glm::vec3 get_color(double u, double v) const = 0;
    };

    class solid_color : public material_base {
    public:
        explicit solid_color(glm::vec3 color): _color(color) {}
        [[nodiscard]] glm::vec3 get_color(double u, double v) const override { return _color; }
    private:
        glm::vec3 _color;
    };
}

#endif //VISUALIME_MATERIAL_H
