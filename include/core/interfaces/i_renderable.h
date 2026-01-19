#ifndef _I_RENDERABLE_H_
#define _I_RENDERABLE_H_

#include <glad/glad.h>

/**
 * @interface IRenderable
 * 
 * @brief Interface for renderable entities in the scene.
 * 
 * Provides a common interface for all entities that can be rendered.
 * Follows the Interface Segregation Principle (ISP) by defining only
 * the minimum required methods for rendering.
 * 
 * @note Classes implementing this interface should provide GPU-compatible
 *       data structures for efficient rendering.
 */
class IRenderable
{
public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~IRenderable() = default;

    /**
     * @brief Gets the size of the renderable data in bytes.
     * 
     * Used for GPU buffer allocation.
     * 
     * @return Size of the data structure in bytes.
     */
    virtual GLsizeiptr getDataSize() const = 0;

    /**
     * @brief Gets a pointer to the raw data for GPU upload.
     * 
     * @return Const void pointer to the data.
     */
    virtual const void* getData() const = 0;

    /**
     * @brief Gets the number of elements in this renderable.
     * 
     * @return Number of elements (vertices, primitives, etc.).
     */
    virtual size_t getCount() const = 0;
};

#endif // _I_RENDERABLE_H_
