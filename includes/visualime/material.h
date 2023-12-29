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
        [[nodiscard]] virtual glm::vec<3, unsigned char> get_color(const glm::vec2& uv) const = 0;
    };

    class solid_color : public material_base {
    public:
        explicit solid_color(const glm::vec<3, unsigned char>& color): _color(color) {}
        [[nodiscard]] glm::vec<3, unsigned char> get_color(const glm::vec2& uv) const override { return _color; }
    private:
        glm::vec<3, unsigned char> _color;
    };
}

#endif //VISUALIME_MATERIAL_H
