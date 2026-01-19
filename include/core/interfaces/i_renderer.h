#ifndef _I_RENDERER_H_
#define _I_RENDERER_H_

#include <memory>
#include <string>

// Forward declarations
class Scene;
class Camera;
class RenderConfig;
struct CullingResult;

/**
 * @interface IRenderer
 * 
 * @brief Interface for scene renderers.
 * 
 * Defines a common interface for different rendering implementations.
 * Follows the Dependency Inversion Principle (DIP) by allowing
 * high-level modules to depend on this abstraction rather than
 * concrete renderer implementations.
 * 
 * Also follows the Open/Closed Principle (OCP) by allowing
 * new renderer types to be added without modifying existing code.
 */
class IRenderer
{
public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~IRenderer() = default;

    /**
     * @brief Initializes the renderer with required resources.
     * 
     * @param config Rendering configuration parameters.
     */
    virtual void initialize(const RenderConfig& config) = 0;

    /**
     * @brief Creates necessary GPU buffers for the scene.
     * 
     * @param scene Scene containing data to buffer.
     */
    virtual void createBuffers(Scene& scene) = 0;

    /**
     * @brief Renders the scene from the camera's perspective.
     * 
     * @param scene Scene to render.
     * @param camera Camera for view/projection.
     * @return CullingResult with visibility statistics.
     */
    virtual CullingResult render(Scene& scene, Camera& camera) = 0;

    /**
     * @brief Cleans up renderer resources.
     */
    virtual void cleanup() = 0;

    /**
     * @brief Gets the renderer name for debugging.
     * 
     * @return Renderer implementation name.
     */
    virtual std::string getName() const = 0;

    /**
     * @brief Checks if the renderer is properly initialized.
     * 
     * @return True if ready to render.
     */
    virtual bool isInitialized() const = 0;
};

#endif // _I_RENDERER_H_
