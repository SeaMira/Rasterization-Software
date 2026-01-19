#ifndef _I_CULLABLE_H_
#define _I_CULLABLE_H_

#include <glm/glm.hpp>

/**
 * @interface ICullable
 * 
 * @brief Interface for entities that can be culled against a view frustum.
 * 
 * Provides a common interface for frustum culling operations.
 * Follows the Interface Segregation Principle (ISP) by separating
 * culling concerns from rendering concerns.
 */
class ICullable
{
public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~ICullable() = default;

    /**
     * @brief Gets the bounding sphere center for frustum culling.
     * 
     * @return 3D position of the bounding sphere center.
     */
    virtual glm::vec3 getBoundingCenter() const = 0;

    /**
     * @brief Gets the bounding sphere radius for frustum culling.
     * 
     * @return Radius of the bounding sphere.
     */
    virtual float getBoundingRadius() const = 0;

    /**
     * @brief Checks if the entity is visible.
     * 
     * @return True if the entity should be rendered.
     */
    virtual bool isVisible() const = 0;

    /**
     * @brief Sets the visibility state of the entity.
     * 
     * @param visible New visibility state.
     */
    virtual void setVisible(bool visible) = 0;
};

#endif // _I_CULLABLE_H_
