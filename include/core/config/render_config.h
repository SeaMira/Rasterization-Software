#ifndef _RENDER_CONFIG_H_
#define _RENDER_CONFIG_H_

#include <string>
#include <glad/glad.h>

/**
 * @class RenderConfig
 * 
 * @brief Configuration class for rendering parameters.
 * 
 * Encapsulates all rendering configuration parameters in a single class,
 * following the Single Responsibility Principle (SRP). This eliminates
 * the need for global variables and provides a clean interface for
 * configuration management.
 * 
 * Uses the Builder pattern for flexible configuration construction.
 */
class RenderConfig
{
public:
    /**
     * @class Builder
     * 
     * @brief Builder for constructing RenderConfig instances.
     */
    class Builder; // Forward declaration; defined after RenderConfig is complete

    /**
     * @brief Gets the screen width.
     * @return Screen width in pixels.
     */
    int getScreenWidth() const { return m_screenWidth; }

    /**
     * @brief Gets the screen height.
     * @return Screen height in pixels.
     */
    int getScreenHeight() const { return m_screenHeight; }

    /**
     * @brief Gets the window title.
     * @return Window title string.
     */
    const std::string& getTitle() const { return m_title; }

    /**
     * @brief Checks if window should be shown.
     * @return True if window should be visible.
     */
    bool isShown() const { return m_shown; }

    /**
     * @brief Checks if occlusion culling is enabled.
     * @return True if occlusion culling is enabled.
     */
    bool hasOcclusionCulling() const { return m_occlusionCulling; }

    /**
     * @brief Gets the X work group size for per-pixel operations.
     * @return Work group X dimension.
     */
    GLuint getWorkGroupSizeXPerPixel() const { return m_workGroupSizeXPerPixel; }

    /**
     * @brief Gets the Y work group size for per-pixel operations.
     * @return Work group Y dimension.
     */
    GLuint getWorkGroupSizeYPerPixel() const { return m_workGroupSizeYPerPixel; }

    /**
     * @brief Gets the X work group size for per-sphere operations.
     * @return Work group X dimension.
     */
    GLuint getWorkGroupSizeXPerSphere() const { return m_workGroupSizeXPerSphere; }

    /**
     * @brief Gets the X work group size for per-cylinder operations.
     * @return Work group X dimension.
     */
    GLuint getWorkGroupSizeXPerCylinder() const { return m_workGroupSizeXPerCylinder; }

    /**
     * @brief Gets the downsample level for Hi-Z.
     * @return Downsample level.
     */
    int getDownsampleLevel() const { return m_downsampleLevel; }

    /**
     * @brief Calculates the downsample work group X size.
     * @return Calculated work group X size for downsampling.
     */
    int getDownsampleWorkGroupSizeX() const { return m_workGroupSizeXPerPixel / m_downsampleLevel; }

    /**
     * @brief Calculates the downsample work group Y size.
     * @return Calculated work group Y size for downsampling.
     */
    int getDownsampleWorkGroupSizeY() const { return m_workGroupSizeYPerPixel / m_downsampleLevel; }

private:
    int m_screenWidth = 1024;                   ///< Screen width in pixels
    int m_screenHeight = 1024;                  ///< Screen height in pixels
    std::string m_title = "Rasterization App";  ///< Window title
    bool m_shown = true;                        ///< Window visibility
    bool m_occlusionCulling = false;            ///< Occlusion culling flag
    
    GLuint m_workGroupSizeXPerPixel = 16;       ///< Per-pixel work group X
    GLuint m_workGroupSizeYPerPixel = 16;       ///< Per-pixel work group Y
    GLuint m_workGroupSizeXPerSphere = 256;     ///< Per-sphere work group X
    GLuint m_workGroupSizeXPerCylinder = 256;   ///< Per-cylinder work group X
    
    int m_downsampleLevel = 4;                  ///< Hi-Z downsample level
};

/**
 * @class Builder
 * 
 * @brief Builder for constructing RenderConfig instances.
 * 
 * Allows fluent construction of RenderConfig objects with
 * optional parameters having sensible defaults.
 */
class RenderConfig::Builder
{
public:
    /**
     * @brief Sets the screen width.
     * 
     * @param width Screen width in pixels.
     * @return Reference to this builder for chaining.
     */
    Builder& screenWidth(int width) { m_config.m_screenWidth = width; return *this; }

    /**
     * @brief Sets the screen height.
     * 
     * @param height Screen height in pixels.
     * @return Reference to this builder for chaining.
     */
    Builder& screenHeight(int height) { m_config.m_screenHeight = height; return *this; }

    /**
     * @brief Sets the window title.
     * 
     * @param title Window title string.
     * @return Reference to this builder for chaining.
     */
    Builder& title(const std::string& title) { m_config.m_title = title; return *this; }

    /**
     * @brief Sets whether the window should be shown.
     * 
     * @param shown True to show the window.
     * @return Reference to this builder for chaining.
     */
    Builder& shown(bool shown) { m_config.m_shown = shown; return *this; }

    /**
     * @brief Sets whether occlusion culling is enabled.
     * 
     * @param enabled True to enable occlusion culling.
     * @return Reference to this builder for chaining.
     */
    Builder& occlusionCulling(bool enabled) { m_config.m_occlusionCulling = enabled; return *this; }

    /**
     * @brief Sets the work group size for per-pixel compute shaders.
     * 
     * @param x X dimension of work group.
     * @param y Y dimension of work group.
     * @return Reference to this builder for chaining.
     */
    Builder& workGroupSizePerPixel(GLuint x, GLuint y) 
    { 
        m_config.m_workGroupSizeXPerPixel = x; 
        m_config.m_workGroupSizeYPerPixel = y; 
        return *this; 
    }

    /**
     * @brief Sets the work group size for per-sphere compute shaders.
     * 
     * @param x X dimension of work group.
     * @return Reference to this builder for chaining.
     */
    Builder& workGroupSizePerSphere(GLuint x) { m_config.m_workGroupSizeXPerSphere = x; return *this; }

    /**
     * @brief Sets the work group size for per-cylinder compute shaders.
     * 
     * @param x X dimension of work group.
     * @return Reference to this builder for chaining.
     */
    Builder& workGroupSizePerCylinder(GLuint x) { m_config.m_workGroupSizeXPerCylinder = x; return *this; }

    /**
     * @brief Sets the downsample level for Hi-Z pyramid.
     * 
     * @param level Downsample level (power of 2).
     * @return Reference to this builder for chaining.
     */
    Builder& downsampleLevel(int level) { m_config.m_downsampleLevel = level; return *this; }

    /**
     * @brief Builds the RenderConfig object.
     * 
     * @return Configured RenderConfig instance.
     */
    RenderConfig build() const { return m_config; }

private:
    RenderConfig m_config;
};

#endif // _RENDER_CONFIG_H_
