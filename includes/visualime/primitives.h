//
// Created by alexzms on 2023/12/10.
//

#ifndef VISUALIME_PRIMITIVES_H
#define VISUALIME_PRIMITIVES_H

#include "memory"
#include "material.h"
#include "glm/glm.hpp"
#include "math_utils.h"
#include "algorithm"

namespace visualime::primitive {                              // !!!all the coordinates are in [0, 1]!!!
    class primitive_base {
    public:
        virtual ~primitive_base() = default;

        [[nodiscard]] virtual glm::vec3 show_color(const glm::vec2& point) const = 0;
        [[nodiscard]] virtual double get_depth(const glm::vec2& point) const = 0;
        [[nodiscard]] virtual math_utils::aabb get_aabb() const = 0;
        [[nodiscard]] virtual bool is_deleted() const { return true; }
        virtual bool set_position(const glm::vec2& position) { return false; }
        virtual bool set_rotation(double angle) { return false; }         // !!!radian!!!
    };

    class deleted_primitive: public primitive_base {
    public:
        [[nodiscard]] glm::vec3 show_color(const glm::vec2& point) const override { return {0, 0, 0}; }
        [[nodiscard]] double get_depth(const glm::vec2& point) const override { return 0; }
        [[nodiscard]] math_utils::aabb get_aabb() const override { return math_utils::aabb{}; }
        [[nodiscard]] bool is_deleted() const override { return true; }
    };

    class solid_circle: public primitive_base {
    public:
        solid_circle(const glm::vec2& position, double depth, double radius, const glm::vec3& color): _depth(depth),
             _position(position), _material(std::make_shared<material::solid_color>(color)), _radius(radius),
             bounding_box(math_utils::interval{position.x - radius, position.x + radius},
                          math_utils::interval{position.y - radius, position.y + radius})
             {}
        [[nodiscard]] glm::vec3 show_color(const glm::vec2& point) const override {
            if (!bounding_box.contains(point))
                return {0, 0, 0};
            if (!should_show(point))
                return {0, 0, 0};

            return _material->get_color(point);                   // use coordinate as uv(that's not right)
        }
        [[nodiscard]] double get_depth(const glm::vec2& point) const override {
            if (!bounding_box.contains(point))
                return 0;
            if (!should_show(point))
                return 0;

            return _depth;
        }
        bool set_rotation(double angle) override {
            this->_angle = angle;                               // bounding box no need to update since it's a circle
            return true;
        }
        bool set_position(const glm::vec2& position) override {
                                                                // change position; is this a good design?
            bounding_box.x_inter.min = position.x - _radius;
            bounding_box.x_inter.max = position.x + _radius;
            bounding_box.y_inter.min = position.y - _radius;
            bounding_box.y_inter.max = position.y + _radius;
                                                                // WARN: currently z is not used
            _position = position;
            return true;
        }
        [[nodiscard]] math_utils::aabb get_aabb() const override {
            return bounding_box;
        }

    private:
        glm::vec2 _position;                                    // position is vec3 for overlapping
        std::shared_ptr<material::material_base> _material;
        double _radius, _depth;
        double _angle{0};
        math_utils::aabb bounding_box;

        [[nodiscard]] glm::vec2 rotate_point(const glm::vec2& point) const {
            // rotate_point a point around the center of the rectangle, by _angle
            glm::vec2 relative = point - _position;
            glm::vec2 rotated_point{0, 0};
            rotated_point.x = relative.x * cos(_angle) - relative.y * sin(_angle);
            rotated_point.y = relative.x * sin(_angle) + relative.y * cos(_angle);
            return rotated_point + _position;
        }

        [[nodiscard]] glm::vec2 rotate_point(const glm::vec2& point, double angle) const {
            // rotate_point a point around the center of the rectangle, by _angle
            glm::vec2 relative = point - _position;
            glm::vec2 rotated_point{0, 0};
            rotated_point.x = relative.x * cos(angle) - relative.y * sin(angle);
            rotated_point.y = relative.x * sin(angle) + relative.y * cos(angle);
            return rotated_point + _position;
        }

        [[nodiscard]] bool should_show(const glm::vec2& point) const {
                                                                                    // point has not been rotated
            glm::vec2 point_rotate = rotate_point(point, -_angle);            // rotate_point back!
            glm::vec2 center = {_position.x, _position.y};
            auto distance = glm::distance(center, point_rotate);
            return distance < _radius;
        }
    };

    class solid_rectangle: public primitive_base {
    public:
        explicit solid_rectangle(const glm::vec2& position, double depth,
                                 double width, double height, const glm::vec3& color, double rotation = 0):
            _position(position), _width(width), _height(height), _depth(depth), _angle(rotation),
            _material(std::make_shared<material::solid_color>(color)),
            _bbox(math_utils::interval{position.x - width / 2, position.x + width / 2},
                  math_utils::interval{position.y - height / 2, position.y + height / 2}) {
            _no_rotate_bbox = _bbox;                                              // record this for later check
            recompute_bbox();                                                     // compute for rotation
        }

        [[nodiscard]] glm::vec3 show_color(const glm::vec2& point) const override {
            if (!_bbox.contains(point))
                return {0, 0, 0};
            if (!should_show(point))
                return {0, 0, 0};

            return _material->get_color(point);                             // use coordinate as uv(that's not right)
        }
        [[nodiscard]] double get_depth(const glm::vec2& point) const override {
            if (!_bbox.contains(point))
                return 0;
            if (!should_show(point))
                return 0;

            return _depth;
        }
        bool set_width(double width) {
            _width = width;
            recompute_bbox();
            recompute_no_rotate_bbox();
            return true;
        }
        bool set_height(double height) {
            _height = height;
            recompute_bbox();
            recompute_no_rotate_bbox();
            return true;
        }
        bool set_width_height(double width, double height) {
            _width = width;
            _height = height;
            recompute_bbox();
            recompute_no_rotate_bbox();
            return true;
        }
        bool set_position(const glm::vec2& position) override {
            _position = position;
            recompute_bbox();
            recompute_no_rotate_bbox();
            return true;
        }
        bool set_rotation(double angle) override {
            this->_angle = angle;
            recompute_bbox();
            recompute_no_rotate_bbox();
            return true;
        }
        [[nodiscard]] math_utils::aabb get_aabb() const override {
            return _bbox;
        }

    private:
        glm::vec2 _position;
        double _depth;
        std::shared_ptr<material::material_base> _material;
        math_utils::aabb _bbox, _no_rotate_bbox;
        double _width, _height, _angle{0};

        void recompute_bbox() {
            glm::vec2 left_top = {_position.x - _width / 2, _position.y + _height / 2};
            glm::vec2 right_top = {_position.x + _width / 2, _position.y + _height / 2};
            glm::vec2 left_bottom = {_position.x - _width / 2, _position.y - _height / 2};
            glm::vec2 right_bottom = {_position.x + _width / 2, _position.y - _height / 2};
            left_top = rotate_point(left_top);
            right_top = rotate_point(right_top);
            left_bottom = rotate_point(left_bottom);
            right_bottom = rotate_point(right_bottom);
            _bbox.x_inter.min = std::min({left_top.x, right_top.x, left_bottom.x, right_bottom.x});
            _bbox.x_inter.max = std::max({left_top.x, right_top.x, left_bottom.x, right_bottom.x});
            _bbox.y_inter.min = std::min({left_top.y, right_top.y, left_bottom.y, right_bottom.y});
            _bbox.y_inter.max = std::max({left_top.y, right_top.y, left_bottom.y, right_bottom.y});
        }

        void recompute_no_rotate_bbox() {
            _no_rotate_bbox.x_inter.min = _position.x - _width / 2;
            _no_rotate_bbox.x_inter.max = _position.x + _width / 2;
            _no_rotate_bbox.y_inter.min = _position.y - _height / 2;
            _no_rotate_bbox.y_inter.max = _position.y + _height / 2;
        }

        [[nodiscard]] glm::vec2 rotate_point(const glm::vec2& point) const {
            // rotate_point a point around the center of the rectangle, by _angle
            glm::vec2 relative = point - _position;
            glm::vec2 rotated_point{0, 0};
            rotated_point.x = relative.x * cos(_angle) - relative.y * sin(_angle);
            rotated_point.y = relative.x * sin(_angle) + relative.y * cos(_angle);
            return rotated_point + _position;
        }

        [[nodiscard]] glm::vec2 rotate_point(const glm::vec2& point, double angle) const {
            // rotate_point a point around the center of the rectangle, by _angle
            glm::vec2 relative = point - _position;
            glm::vec2 rotated_point{0, 0};
            rotated_point.x = relative.x * cos(angle) - relative.y * sin(angle);
            rotated_point.y = relative.x * sin(angle) + relative.y * cos(angle);
            return rotated_point + _position;
        }

        [[nodiscard]] bool should_show(const glm::vec2& point) const {
                                                                                    // point has not been rotated
            glm::vec2 point_rotate = rotate_point(point, -_angle);            // rotate_point back!
            return _no_rotate_bbox.contains(point_rotate);
        }
    };
}


#endif //VISUALIME_PRIMITIVES_H
