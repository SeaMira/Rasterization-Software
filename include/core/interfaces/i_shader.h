#ifndef _I_SHADER_H_
#define _I_SHADER_H_

#include <string>
#include <glm/glm.hpp>

/**
 * @interface IShader
 * 
 * @brief Interface for shader programs.
 * 
 * Defines a common interface for both vertex/fragment shader programs
 * and compute shader programs. Follows the Liskov Substitution Principle (LSP)
 * by ensuring all shader types can be used interchangeably where uniforms
 * need to be set.
 * 
 * @note This interface provides uniform setting methods that are common
 *       to all shader types in OpenGL.
 */
class IShader
{
public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~IShader() = default;

    /**
     * @brief Activates this shader program for use.
     */
    virtual void use() = 0;

    /**
     * @brief Gets the shader program ID.
     * 
     * @return OpenGL shader program ID.
     */
    virtual unsigned int getId() const = 0;

    /**
     * @brief Gets the shader name for debugging purposes.
     * 
     * @return Shader program name.
     */
    virtual const std::string& getName() const = 0;

    // Uniform setters - common to all shader types

    /**
     * @brief Sets a boolean uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Boolean value to set.
     */
    virtual void setBool(const std::string& name, bool value) const = 0;

    /**
     * @brief Sets an integer uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Integer value to set.
     */
    virtual void setInt(const std::string& name, int value) const = 0;

    /**
     * @brief Sets an unsigned integer uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Unsigned integer value to set.
     */
    virtual void setUint(const std::string& name, unsigned int value) const = 0;

    /**
     * @brief Sets a float uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Float value to set.
     */
    virtual void setFloat(const std::string& name, float value) const = 0;

    /**
     * @brief Sets a vec2 uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Vec2 value to set.
     */
    virtual void setVec2(const std::string& name, const glm::vec2& value) const = 0;

    /**
     * @brief Sets a vec3 uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Vec3 value to set.
     */
    virtual void setVec3(const std::string& name, const glm::vec3& value) const = 0;

    /**
     * @brief Sets a vec4 uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Vec4 value to set.
     */
    virtual void setVec4(const std::string& name, const glm::vec4& value) const = 0;

    /**
     * @brief Sets an integer vec2 uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Integer vec2 value to set.
     */
    virtual void setVec2I(const std::string& name, const glm::ivec2& value) const = 0;

    /**
     * @brief Sets a mat4 uniform value.
     * 
     * @param name Uniform variable name in the shader.
     * @param value Mat4 value to set.
     */
    virtual void setMat4(const std::string& name, const glm::mat4& value) const = 0;
};

#endif // _I_SHADER_H_
