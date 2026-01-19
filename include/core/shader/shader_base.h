#ifndef _SHADER_BASE_H_
#define _SHADER_BASE_H_

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "core/interfaces/i_shader.h"

/**
 * @class ShaderBase
 * 
 * @brief Base class for all shader programs.
 * 
 * Provides common functionality for shader programs, including
 * uniform setting methods. Follows the Template Method pattern
 * for shader compilation and the Liskov Substitution Principle (LSP)
 * by implementing the IShader interface.
 * 
 * This class consolidates duplicate code from ComputeShader and
 * ShaderProgram classes, following the DRY (Don't Repeat Yourself)
 * principle.
 */
class ShaderBase : public IShader
{
public:
    /**
     * @brief Constructs a shader base with a name.
     * 
     * @param name Name of the shader for debugging.
     */
    explicit ShaderBase(const std::string& name = "Unnamed Shader");

    /**
     * @brief Virtual destructor - deletes the shader program.
     */
    virtual ~ShaderBase();

    // Prevent copying
    ShaderBase(const ShaderBase&) = delete;
    ShaderBase& operator=(const ShaderBase&) = delete;

    // Allow moving
    ShaderBase(ShaderBase&& other) noexcept;
    ShaderBase& operator=(ShaderBase&& other) noexcept;

    // IShader interface implementation

    /**
     * @brief Activates this shader program.
     */
    void use() override;

    /**
     * @brief Gets the shader program ID.
     * @return OpenGL program ID.
     */
    unsigned int getId() const override { return m_programId; }

    /**
     * @brief Gets the shader name.
     * @return Shader name string.
     */
    const std::string& getName() const override { return m_name; }

    // Uniform setters

    void setBool(const std::string& name, bool value) const override;
    void setInt(const std::string& name, int value) const override;
    void setUint(const std::string& name, unsigned int value) const override;
    void setFloat(const std::string& name, float value) const override;
    
    void setVec2(const std::string& name, const glm::vec2& value) const override;
    void setVec3(const std::string& name, const glm::vec3& value) const override;
    void setVec4(const std::string& name, const glm::vec4& value) const override;
    void setVec2I(const std::string& name, const glm::ivec2& value) const override;
    void setMat4(const std::string& name, const glm::mat4& value) const override;

    // Additional uniform setters (not in interface but commonly needed)

    /**
     * @brief Sets a vec2 uniform from components.
     * 
     * @param name Uniform name.
     * @param x X component.
     * @param y Y component.
     */
    void setVec2(const std::string& name, float x, float y) const;

    /**
     * @brief Sets a vec3 uniform from components.
     * 
     * @param name Uniform name.
     * @param x X component.
     * @param y Y component.
     * @param z Z component.
     */
    void setVec3(const std::string& name, float x, float y, float z) const;

    /**
     * @brief Sets a vec4 uniform from components.
     * 
     * @param name Uniform name.
     * @param x X component.
     * @param y Y component.
     * @param z Z component.
     * @param w W component.
     */
    void setVec4(const std::string& name, float x, float y, float z, float w) const;

    /**
     * @brief Sets an integer vec2 uniform from components.
     * 
     * @param name Uniform name.
     * @param x X component.
     * @param y Y component.
     */
    void setVec2I(const std::string& name, int x, int y) const;

    /**
     * @brief Sets an integer vec3 uniform.
     * 
     * @param name Uniform name.
     * @param value Vec3 value.
     */
    void setVec3I(const std::string& name, const glm::ivec3& value) const;

    /**
     * @brief Sets an integer vec3 uniform from components.
     * 
     * @param name Uniform name.
     * @param x X component.
     * @param y Y component.
     * @param z Z component.
     */
    void setVec3I(const std::string& name, int x, int y, int z) const;

    /**
     * @brief Sets an integer vec4 uniform.
     * 
     * @param name Uniform name.
     * @param value Vec4 value.
     */
    void setVec4I(const std::string& name, const glm::ivec4& value) const;

    /**
     * @brief Sets a mat2 uniform.
     * 
     * @param name Uniform name.
     * @param value Matrix value.
     */
    void setMat2(const std::string& name, const glm::mat2& value) const;

    /**
     * @brief Sets a mat3 uniform.
     * 
     * @param name Uniform name.
     * @param value Matrix value.
     */
    void setMat3(const std::string& name, const glm::mat3& value) const;

protected:
    /**
     * @brief Reads shader source code from a file.
     * 
     * @param filePath Path to the shader file.
     * @return Shader source code as string.
     * @throws std::runtime_error if file cannot be read.
     */
    std::string readShaderFile(const std::string& filePath) const;

    /**
     * @brief Compiles a shader from source.
     * 
     * @param source Shader source code.
     * @param shaderType GL shader type (GL_VERTEX_SHADER, etc.).
     * @param shaderName Name for error messages.
     * @return Compiled shader ID.
     * @throws std::runtime_error on compilation failure.
     */
    GLuint compileShader(const std::string& source, GLenum shaderType, 
                         const std::string& shaderName) const;

    /**
     * @brief Links the shader program.
     * 
     * @throws std::runtime_error on linking failure.
     */
    void linkProgram();

    /**
     * @brief Creates a new shader program.
     */
    void createProgram();

    /**
     * @brief Attaches a shader to the program.
     * 
     * @param shaderId Shader to attach.
     */
    void attachShader(GLuint shaderId);

    /**
     * @brief Detaches and deletes a shader.
     * 
     * @param shaderId Shader to cleanup.
     */
    void cleanupShader(GLuint shaderId);

    /**
     * @brief Gets a uniform location, with caching.
     * 
     * @param name Uniform name.
     * @return Uniform location, or -1 if not found.
     */
    GLint getUniformLocation(const std::string& name) const;

protected:
    GLuint m_programId = 0;     ///< OpenGL program ID
    std::string m_name;         ///< Shader name for debugging

private:
    /**
     * @brief Checks for shader/program errors.
     * 
     * @param object Shader or program ID.
     * @param type "SHADER" or "PROGRAM".
     * @param name Object name for error messages.
     */
    void checkErrors(GLuint object, const std::string& type, 
                     const std::string& name) const;

    mutable std::unordered_map<std::string, GLint> m_uniformCache; ///< Uniform location cache
};

#endif // _SHADER_BASE_H_
