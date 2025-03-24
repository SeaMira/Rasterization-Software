#ifndef _FRAME_BUFFER_H
#define _FRAME_BUFFER_H

#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include "vis/gl/texture.h"

/**
 * @class Framebuffer
 * 
 * @brief Framebuffer class for handling framebuffers.
 * 
 * The Framebuffer class is used to handle framebuffers in OpenGL, attaching textures to them and 
 * checking if they are complete.
 */
class Framebuffer {
public:

    /**
     * @brief Constructor for the Framebuffer class.
     * 
     * Constructor for the Framebuffer class, initializes the framebuffer.
     */
    Framebuffer();

    /**
     * @brief Destructor for the Framebuffer class.
     * 
     * Destructor for the Framebuffer class, deletes the framebuffer.
     */
    ~Framebuffer();

    /**
     * @brief attach a texture to the framebuffer.
     * 
     * Attaches a texture to the framebuffer at a certain attachment point.
     * 
     * @param attachment the attachment point to attach the texture.
     * @param texture the texture to attach.
     * @param framebuffer the framebuffer target to attach the texture to.
     */
    void attachTexture(GLenum attachment, const Texture& texture, 
        GLenum framebuffer = GL_FRAMEBUFFER);

    /**
     * @brief attach a texture image to the framebuffer.
     * 
     * Attaches a texture image to the framebuffer at a certain attachment point.
     * 
     * @param attachment the attachment point to attach the texture.
     * @param texture the texture to attach.
     * @param framebuffer the framebuffer target to attach the texture to.
     */
    void attachTextureImage(GLenum attachment, const Texture& texture, 
        GLenum framebuffer);

    /**
     * @brief bind the framebuffer.
     * 
     * Binds the framebuffer with a certain target.
     * 
     * @param target the target to bind the framebuffer.
     */
    void bind(GLenum target = GL_FRAMEBUFFER) const;

    /**
     * @brief unbind the framebuffer.
     * 
     * Unbinds the framebuffer.
     */
    void unbind() const;

    /**
     * @brief check if the framebuffer is complete.
     * 
     * Checks if the framebuffer is complete.
     * 
     * @return true if the framebuffer is complete, false otherwise.
     */
    bool isComplete() const;

    /**
     * @brief get the id of the framebuffer.
     * 
     * Gets the id of the framebuffer.
     * 
     * @return the id of the framebuffer.
     */
    GLuint getId() const { return m_id; }

private:
    GLuint m_id; ///< the id of the framebuffer.
};

#endif