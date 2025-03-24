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
     * @param data pointer to the image data in memory.
     * 
     * More info on @link https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml glTexImage2D @endlink
     */
    Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height, 
        GLenum format, GLenum type, void* data = nullptr);

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
     * 
     * More info on @link https://registry.khronos.org/OpenGL-Refpages/es3.0/html/glTexStorage2D.xhtml glTexStorage2D @endlink
     */
    Texture(GLenum target, GLenum internalFormat, GLsizei width, GLsizei height);
    
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
     * @brief Binds textures.
     * 
     * Binds and actives textures.
     * 
     * @param unit texture unit to make active.
     */
    void bind(GLuint unit = 0) const;

    /**
     * @brief Binds image textures.
     * 
     * @param unit index of the image unit to which to bind the texture.
     * @param access access types to image from shaders: GL_READ_ONLY, GL_WRITE_ONLY, or GL_READ_WRITE.
     * @param format pixel format used for image formatted stores on shaders.
     */
    void bindImage(GLuint unit, GLenum access, GLenum format) const;

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
    GLenum m_target; ///< target texture to set.
    friend class Canvas; ///< Friend class Canvas so it can access private atributes.
};

#endif