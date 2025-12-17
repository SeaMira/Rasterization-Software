#ifndef _SHADER_PROGRAM_H
#define _SHADER_PROGRAM_H

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

/**
 * @class ShaderProgram
 * 
 * @brief Compiles shader programs.
 * 
 * It's in charge of loading and compiling general shaders.
 */
class ShaderProgram
{
    public:
        unsigned int ID; ///< Shader program ID
        std::string m_shaderProgramName; ///< Shader program name
        
        /**
         * @brief ShaderProgram class constructor.
         * 
         * Recieves the name of files with shaders and the constructor loads it, compiles, verifies
         * everything is correct and generates a program.
         * 
         * @param vertexPath file name with a vertex shader.
         * @param fragmentPath file name with a fragment shader.
         * @param geometryPath file name with a geometry shader.
         * @param shaderProgramName name of the shader program.
         */
        ShaderProgram(const char* vertexPath = "", const char* fragmentPath = "", const char* geometryPath = "", const char* shaderProgramName = "", 
            const char* vertexShaderName = "", const char* fragmentShaderName = "", const char* geometryShaderName = "");
        
        /**
         * @brief ShaderProgram class destructor.
         * 
         * Deletes the program.
         */
        ~ShaderProgram() { glDeleteProgram(ID); };

        /**
         * @brief Compiles and attaches a shader to the program frome
         * a source file.
         * 
         * @param shaderPath file name with a shader.
         * @param shaderType type of the shader to be compiled.
         * @param shaderName name of the shader.
         */
        void attachShader(const char* shaderPath, GLenum shaderType, const char* shaderName);

        /**
         * @brief Links the shaders program.
         */
        void linkProgram();

        /**
         * @brief sets this program to use.
         */
        void use();

        /**
         * @brief Loads a uniform bool on the shader.
         * 
         * @param name Uniform bool name in shader.
         * @param value Uniform bool value.
         */
        void setBool(const std::string &name, bool value) const;

        /**
         * @brief Loads a uniform int on the shader.
         * 
         * @param name Uniform int name in shader.
         * @param value Uniform int value.
         */
        void setInt(const std::string &name, int value) const;
        
        /**
         * @brief Loads a uniform unsigned int on the shader.
         * 
         * @param name Uniform unsigned int name in shader.
         * @param value Uniform unsigned int value.
         */
        void setUint(const std::string &name, unsigned int value) const;

        /**
         * @brief Loads a uniform float on the shader.
         * 
         * @param name Uniform float name in shader.
         * @param value Uniform float value.
         */
        void setFloat(const std::string &name, float value) const;

        /**
         * @brief Loads a uniform vec2 on the shader.
         * 
         * @param name Uniform vec2 name in shader.
         * @param value Uniform vec2 value.
         */
        void setVec2(const std::string &name, const glm::vec2 &value) const;

        /**
         * @brief Loads a uniform vec2 on the shader.
         * 
         * @param name Uniform vec2 name in shader.
         * @param x first float value of the vec2.
         * @param y second float value of the vec2.
         */
        void setVec2(const std::string &name, float x, float y) const;

        /**
         * @brief Loads a uniform integers vec2 on the shader.
         * 
         * @param name Uniform integers vec2 name in shader.
         * @param value Uniform integers vec2 value.
         */
        void setVec2I(const std::string &name, const glm::vec2 &value) const;

        /**
         * @brief Loads a uniform integers vec2 on the shader.
         * 
         * @param name Uniform integers vec2 name in shader.
         * @param x first integer value of the vec2.
         * @param y second integer value of the vec2.
         */
        void setVec2I(const std::string &name, int x, int y) const;

        /**
         * @brief Loads a uniform vec3 on the shader.
         * 
         * @param name Uniform integers vec3 name in shader.
         * @param value Uniform integers vec3 value.
         */
        void setVec3(const std::string &name, const glm::vec3 &value) const;

        /**
         * @brief Loads a uniform vec3 on the shader.
         * 
         * @param name Uniform vec3 name in shader.
         * @param x first float value of the vec3.
         * @param y second float value of the vec3.
         * @param z third float value of the vec3.
         */
        void setVec3(const std::string &name, float x, float y, float z) const;

        /**
         * @brief Loads a uniform integers vec3 on the shader.
         * 
         * @param name Uniform integers vec3 name in shader.
         * @param value Uniform integers vec3 value.
         */
        void setVec3I(const std::string &name, const glm::vec3 &value) const;

        /**
         * @brief Loads a uniform integers vec3 on the shader.
         * 
         * @param name Uniform integers vec3 name in shader.
         * @param x first integer value of the vec3.
         * @param y second integer value of the vec3.
         * @param z third integer value of the vec3.
         */
        void setVec3I(const std::string &name, int x, int y, int z) const;

        /**
         * @brief Loads a uniform vec4 on the shader.
         * 
         * @param name Uniform vec4 name in shader.
         * @param value Uniform vec4 value.
         */
        void setVec4(const std::string &name, const glm::vec4 &value) const;

        /**
         * @brief Loads a uniform vec4 on the shader.
         * 
         * @param name Uniform vec4 name in shader.
         * @param x first float value of the vec4.
         * @param y second float value of the vec4.
         * @param z third float value of the vec4.
         * @param w fourth float value of the vec4.
         */
        void setVec4(const std::string &name, float x, float y, float z, float w) const;

        /**
         * @brief Loads a uniform integers vec4 on the shader.
         * 
         * @param name Uniform integers vec4 name in shader.
         * @param value Uniform integers vec4 value.
         */
        void setVec4I(const std::string &name, const glm::vec4 &value) const;

        /**
         * @brief Loads a uniform integers vec4 on the shader.
         * 
         * @param name Uniform integers vec4 name in shader.
         * @param x first integer value of the vec4.
         * @param y second integer value of the vec4.
         * @param z third integer value of the vec4.
         * @param w fourth integer value of the vec4.
         */
        void setVec4I(const std::string &name, int x, int y, int z, int w);

        /**
         * @brief Loads a uniform 2x2 matrix on the shader.
         * 
         * @param name Uniform 2x2 matrix name in shader.
         * @param mat Uniform 2x2 matrix value.
         */
        void setMat2(const std::string &name, const glm::mat2 &mat) const;

        /**
         * @brief Loads a uniform 3x3 matrix on the shader.
         * 
         * @param name Uniform 3x3 matrix name in shader.
         * @param mat Uniform 3x3 matrix value.
         */
        void setMat3(const std::string &name, const glm::mat3 &mat) const;

        /**
         * @brief Loads a uniform 4x4 matrix on the shader.
         * 
         * @param name Uniform 4x4 matrix name in shader.
         * @param mat Uniform 4x4 matrix value.
         */
        void setMat4(const std::string &name, const glm::mat4 &mat) const;

    private:
        /**
         * @brief Checking compile errors function.
         * 
         * Utility function for checking shader compilation/linking errors.
         * 
         * @param shader shader/program id to check.
         * @param type if is a shader or a program to check.
         */
        void checkCompileErrors(GLuint shader, std::string type);
};

#endif // _SHADER_PROGRAM_H