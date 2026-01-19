#ifndef _I_CULLING_STRATEGY_H_
#define _I_CULLING_STRATEGY_H_

#include <vector>
#include <memory>

// Forward declarations
class Scene;
class Camera;
struct CullingResult;

/**
 * @struct CullingResult
 * 
 * @brief Results from a culling operation.
 * 
 * Contains statistics and visibility information after
 * performing a culling pass.
 */
struct CullingResult
{
    int visibleSphereCount = 0;     ///< Number of spheres passing frustum culling
    int drawnSphereCount = 0;       ///< Number of spheres actually drawn (after occlusion)
    int visibleCylinderCount = 0;   ///< Number of cylinders passing frustum culling
    int drawnCylinderCount = 0;     ///< Number of cylinders actually drawn (after occlusion)
};

/**
 * @interface ICullingStrategy
 * 
 * @brief Interface for culling strategies.
 * 
 * Defines a common interface for different culling algorithms
 * (frustum culling, occlusion culling, etc.). Follows the Strategy Pattern
 * allowing different culling methods to be swapped at runtime.
 * 
 * Also follows the Single Responsibility Principle (SRP) by separating
 * culling logic from rendering logic.
 */
class ICullingStrategy
{
public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~ICullingStrategy() = default;

    /**
     * @brief Initializes the culling strategy with required resources.
     * 
     * @param screenWidth Width of the render target.
     * @param screenHeight Height of the render target.
     */
    virtual void initialize(int screenWidth, int screenHeight) = 0;

    /**
     * @brief Performs culling on the scene.
     * 
     * @param scene The scene containing entities to cull.
     * @param camera The camera for view-dependent culling.
     * @return CullingResult containing visibility statistics.
     */
    virtual CullingResult performCulling(Scene& scene, const Camera& camera) = 0;

    /**
     * @brief Cleans up any resources used by this culling pass.
     */
    virtual void cleanup() = 0;

    /**
     * @brief Gets the name of this culling strategy for debugging.
     * 
     * @return Strategy name (e.g., "GPU Frustum Culling", "Hi-Z Occlusion").
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Checks if this strategy supports occlusion culling.
     * 
     * @return True if occlusion culling is supported.
     */
    virtual bool supportsOcclusionCulling() const = 0;
};

#endif // _I_CULLING_STRATEGY_H_
