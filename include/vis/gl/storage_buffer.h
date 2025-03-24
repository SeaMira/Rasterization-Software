#ifndef _STORAGE_BUFFER_H
#define _STORAGE_BUFFER_H

#include <glad/glad.h>
#include "vis/gl/texture.h"


/**
 * @class StorageBuffer
 * 
 * @brief StorageBuffer class for handling storage buffers.
 * 
 * The StorageBuffer class is used to handle storage buffers in OpenGL (SSBOs), generating buffer data and checking if they are complete.
 */
class StorageBuffer
{
public:
    /**
     * @brief Constructor for the StorageBuffer class.
     * 
     * Constructor for the StorageBuffer class, generates buffers and saves atributes.
     * 
     * @param target the target of the storage buffer.
     */
    StorageBuffer(GLenum target);

    /**
     * @brief Destructor for the StorageBuffer class.
     * 
     * Destructor for the StorageBuffer class, deletes the storage buffer.
     */
    ~StorageBuffer();

    /**
     * @brief generate buffer data for the storage buffer.
     * 
     * Generates buffer data for the storage buffer at a certain index.
     * 
     * @param size the size of the buffer data.
     * @param index the index of the buffer data.
     * @param data the data pointer to be stored in the buffer.
     * @param usage the usage of the buffer data.
     */
    void generateBufferData(int size, GLuint index,
        const void * data = nullptr, GLenum usage = GL_DYNAMIC_DRAW) const;
    // void generateBufferStorage(int size, GLuint index,
    //     const void * data = nullptr, GLbitfield flags = GL_DYNAMIC_STORAGE_BIT) const;
    
    /**
     * @brief Binds the buffer.
     * 
     * Binds the buffer on its target.
     */
    void bind() const;

    /**
     * @brief Unbinds the buffer.
     */
    void unbind() const;

    /**
     * @brief Checks if buffer is complete.
     * Checks if buffer is valid with glIsBuffer.
     */
    bool isComplete() const;

    /**
     * @brief Gets the buffer ID.
     * 
     * @return ID of the buffer
     */
    GLuint getId() const { return m_id; }

private:
    GLuint m_id; ///< id of the buffer
    GLenum m_target; ///< buffer's target
};

#endif