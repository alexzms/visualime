//
// Created by alexzms on 2023/12/11.
//

#ifndef VISUALIME_SHADER_H
#define VISUALIME_SHADER_H


//
// Created by alexzms on 2023/10/1.
//

#ifndef LEARN_OPENGL_CLION_SHADER_H
#define LEARN_OPENGL_CLION_SHADER_H

#include <glm/gtc/type_ptr.hpp>
#include "glad/glad.h"
#include "sstream"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Shader{
public:
    // shader ID
    unsigned int ID{};
    bool functional = false;

    Shader() = default;
    Shader(const char* vertex_shader_code, const char* fragment_shader_code);

    // activate program
    void use() const;
    // uniform value set
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
    void setVec3(const std::string &name, float x, float y, float z) const;
    void setVec3(const std::string &name, glm::vec3 vec) const;
    void setMat4(const std::string &name, glm::mat4 mat) const;
};

#endif //LEARN_OPENGL_CLION_SHADER_H

#endif //VISUALIME_SHADER_H
