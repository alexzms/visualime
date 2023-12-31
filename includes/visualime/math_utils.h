//
// Created by alexzms on 2023/12/15.
//

#ifndef UNIT_TESTS_MATH_UTILS_H
#define UNIT_TESTS_MATH_UTILS_H

#include <glm/glm.hpp>
#include "limits"

namespace visualime::math_utils {
    double infinity = std::numeric_limits<double>::infinity();
    double PI = 3.1415926535897932385;

    [[nodiscard]] inline double degrees_to_radians(double degrees) {
        return degrees * PI / 180.0;
    }

    [[nodiscard]] inline double radians_to_degrees(double radians) {
        return radians * 180.0 / PI;
    }

    [[nodiscard]] inline glm::vec2 rotate_point(const glm::vec2& point, double angle) {
        double x = point.x * cos(angle) - point.y * sin(angle);
        double y = point.x * sin(angle) + point.y * cos(angle);
        return {x, y};
    }

    [[nodiscard]] inline glm::vec2 rotate_point(const glm::vec2& point, const glm::vec2& center, double angle) {
        glm::vec2 point_relative = point - center;
        glm::vec2 point_relative_rotated = rotate_point(point_relative, angle);
        return point_relative_rotated + center;
    }

    [[nodiscard]] inline long long sleep_time_corrected(unsigned int framerate, long long frame_duration) {
        if (framerate <= 60) return 1000000 / framerate - frame_duration - 2200;
        return 1000000 / framerate - frame_duration - 20;
    }

    class interval {
    public:
        double min, max;                                                            // no need for over-wrapping
        interval(double min, double max): min(min), max(max) {}
        interval(const interval& i1, const interval& i2):                           // combine constructor
                                    min(fmin(i1.min, i2.min)), max(fmax(i1.max, i2.max)) {}
        interval(): min(+infinity), max(-infinity) {}                               // default is empty

        [[nodiscard]] bool contains(double value) const {
            return min < value && value < max;
        }

        void clamp(const interval& inter) {
                                                                                    // clamp to smallest(change)
            this->min = fmax(inter.min, this->min);
            this->max = fmin(inter.max, this->max);
        }
        [[nodiscard]] interval clamped(const interval& inter) const {
                                                                                    // clamp to smallest(return)
            return {fmax(inter.min, this->min), fmin(inter.max, this->max)};
        }
        [[nodiscard]] bool is_empty() const {
            return min > max;
        }
    };

    class aabb {
    public:
        interval x_inter, y_inter;
        aabb() = default;                                                   // default is empty aabb
        aabb(const interval& x, const interval& y): x_inter(x), y_inter(y) {}
        aabb(const glm::vec2& p1, const glm::vec2& p2) {
            x_inter = interval{fmin(p1.x, p2.x), fmax(p1.x, p2.x)};
            y_inter = interval{fmin(p1.y, p2.y), fmax(p1.y, p2.y)};
        }
        aabb(const aabb& a1, const aabb& a2) {
                                                                            // merge to largest
            x_inter = interval{a1.x_inter, a2.x_inter};
            y_inter = interval{a1.y_inter, a2.y_inter};
        }
        [[nodiscard]] bool contains(const glm::vec2& point) const {
            return x_inter.contains(point.x) && y_inter.contains(point.y);
        }
        [[nodiscard]] bool is_empty() const {
            return x_inter.is_empty() || y_inter.is_empty();
        }
        void clamp(const aabb& aabb) {
                                                                            // clamp to smallest(change)
            x_inter.clamp(aabb.x_inter);
            y_inter.clamp(aabb.y_inter);
        }
        [[nodiscard]] aabb clamped(const aabb& aabb) const {
                                                                            // clamp to smallest(return)
            return {
                    x_inter.clamped(aabb.x_inter),
                    y_inter.clamped(aabb.y_inter),
            };
        }
        [[nodiscard]] aabb clamped(const interval& x_limit, const interval& y_limit) const {
            return {
                    x_inter.clamped(x_limit),
                    y_inter.clamped(y_limit),
            };
        }
        [[nodiscard]] aabb clamped(const interval& universal_interval_limitation) const {
                                                                            // no matter what axis, clamp to min,max
            return this->clamped(
                    universal_interval_limitation,
                    universal_interval_limitation
                    );
        }
    };
}


#endif //UNIT_TESTS_MATH_UTILS_H
