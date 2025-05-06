#ifndef _TEXTURE_H_
#define _TEXTURE_H_

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>

/**
 * @class Texture 
 * 
 * @brief Texture class wrapper.
 * 
 * Texture class wrapper for using textures on shaders. Used by Canvas objects since a texture works as viewport.
 */
class Texture {
public:


    /**
     * @brief Texture class constructor.
     * 
     * Constructor that allows to generate a Texture with a target, a certain image format (channels and size),
     * width and height of the texture, type and pointer to image data. 
     * 
     * @param target buffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param format format of the pixel data.
     * @param type data type of the pixel.
     * @param unit index of image unit to which bind the texture.
     * @param data pointer to the image data in memory.
     * 
     * More info on
     * <a href="https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml">glTexImage2D</a>
     */
    Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, 
        GLenum format, GLenum type, GLuint unit, void* data = nullptr);
        
    /**
     * @brief Texture class constructor.
     * 
     * Similar to the other constructor but uses glTexStorage2D instead of glTexImage2D. Doesn´t
     * need type or format.
     * 
     * @param target buffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param unit index of image unit to which bind the texture.
     * 
     * More info on
     * <a href="https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexStorage2D.xhtml">glTexStorage2D</a>
     */
    Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, GLuint unit);
    
    /**
     * @brief Texture class constructor.
     * 
     * Similar to the other constructors but uses glTexImage2D. Can set the values
     * for parameters.
     * 
     * @param target buffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param format format of the pixel data.
     * @param type data type of the pixel.
     * @param unit index of image unit to which bind the texture.
     * @param data pointer to the image data in memory.
     * @param min_filter_param parameter value for texture min filter.
     * @param max_filter_param parameter value for texture max filter.
     * @param wrap_s_param parameter value for texture wrap for coordinate s.
     * @param wrap_t_param parameter value for texture wrap for coordinate t.
     * 
     * More info on
     * <a href="https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexStorage2D.xhtml">glTexStorage2D</a>
     */
    Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, 
        GLenum format, GLenum type, GLuint unit, void* data,
        GLint min_filter_param, GLint mag_filter_param, GLint wrap_s_param, GLint wrap_t_param);
    
    /**
     * @brief Default constructor.
     * 
     * Default constructor, in case of late setup.
     */
    Texture() = default;

    /**
     * @brief Class destructor.
     * 
     * Class destructor, deletes textures. 
     */
    ~Texture();

    /**
     * @brief Texture class late setup.
     * 
     * Setup that allows to generate a Texture with a target, a certain image format (channels and size),
     * width and height of the texture, type and pointer to image data. 
     * 
     * @param target buffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param format format of the pixel data.
     * @param type data type of the pixel.
     * @param unit index of image unit to which bind the texture.
     * @param data pointer to the image data in memory.
     * 
     * More info on
     * <a href="https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml">glTexImage2D</a>
     */
    void setup(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, 
        GLenum format, GLenum type, GLuint unit, void* data = nullptr);

    /**
     * @brief Texture class late setup.
     * 
     * Similar to the other setup but uses glTexStorage2D instead of glTexImage2D. Doesn´t
     * need type or format.
     * 
     * @param target buffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param unit index of image unit to which bind the texture.
     * 
     * More info on
     * <a href="https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexStorage2D.xhtml">glTexStorage2D</a>
     */
    void setup(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, GLuint unit);

    /**
     * @brief Texture class late setup.
     * 
     * Similar to the other setup but uses glTexImage2D. Can set the values
     * for parameters.
     * 
     * @param target buffer target to set.
     * @param internalFormat image format, specifies the number of color components.
     * @param width width of the texture.
     * @param height height of the texture.
     * @param format format of the pixel data.
     * @param type data type of the pixel.
     * @param unit index of image unit to which bind the texture.
     * @param data pointer to the image data in memory.
     * @param min_filter_param parameter value for texture min filter.
     * @param max_filter_param parameter value for texture max filter.
     * @param wrap_s_param parameter value for texture wrap for coordinate s.
     * @param wrap_t_param parameter value for texture wrap for coordinate t.
     * 
     * More info on
     * <a href="https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexStorage2D.xhtml">glTexStorage2D</a>
     */
    void setup(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, 
        GLenum format, GLenum type, GLuint unit, void* data,
        GLint min_filter_param, GLint mag_filter_param, GLint wrap_s_param, GLint wrap_t_param);


    /**
     * @brief Binds textures.
     * 
     * Binds and actives textures with the binding point as texture unit.
     * 
     */
    void bind() const;

    /**
     * @brief Binds image textures.
     * 
     * @param access access types to image from shaders: GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE.
     * @param format pixel format used for image formatted stores on shaders.
     */
    void bindImage(GLenum access, GLenum format) const;
    
    /**
     * @brief Unbinds image textures.
     * 
     * @param access access types to image from shaders: GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE.
     * @param format pixel format used for image formatted stores on shaders.
     */
    void unbindImage(GLenum access, GLenum format) const;

    /**
     * @brief Unbind textures. 
     */
    void unbind() const;

    /**
     * @brief Gets texture ID.
     * 
     * @return Id of the texture.
     */
    GLuint getId() const;

private:
    GLuint m_id; ///< Id of the texture.
    GLuint m_unit; ///< Id of the texture.
    GLenum m_target; ///< target texture to set.
    friend class Canvas; ///< Friend class Canvas so it can access private atributes.
};

#endif