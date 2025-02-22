#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class ComputeShader
{
    public:
        unsigned int ID;
        // constructor generates the shader on the fly
        // ------------------------------------------------------------------------
        ComputeShader(const char* computePath);
        
        // Activate this shader
        void use();

        // set uniform functions
        void setBool(const std::string &name, bool value) const;
        void setInt(const std::string &name, int value) const;
        void setFloat(const std::string &name, float value) const;

        void setVec2(const std::string &name, const glm::vec2 &value) const;
        void setVec2(const std::string &name, float x, float y) const;
        void setVec2I(const std::string &name, const glm::vec2 &value) const;
        void setVec2I(const std::string &name, int x, int y) const;

        void setVec3(const std::string &name, const glm::vec3 &value) const;
        void setVec3(const std::string &name, float x, float y, float z) const;
        void setVec3I(const std::string &name, const glm::vec3 &value) const;
        void setVec3I(const std::string &name, int x, int y, int z) const;

        void setVec4(const std::string &name, const glm::vec4 &value) const;
        void setVec4(const std::string &name, float x, float y, float z, float w) const;
        void setVec4I(const std::string &name, const glm::vec4 &value) const;
        void setVec4I(const std::string &name, int x, int y, int z, int w);

        void setMat2(const std::string &name, const glm::mat2 &mat) const;
        void setMat3(const std::string &name, const glm::mat3 &mat) const;
        void setMat4(const std::string &name, const glm::mat4 &mat) const;

    private:
        // utility function for checking shader compilation/linking errors.
        void checkCompileErrors(GLuint shader, std::string type);
};

#endif