#ifndef _SCENE_H_
#define _SCENE_H_

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glad/glad.h>

#include "core/config/scene_config.h"
#include "geometry/sphere/sphere.h"
#include "geometry/cylinder/cylinder.h"

/**
 * @class Scene
 * 
 * @brief Container class for scene entities and related data.
 * 
 * Manages the collection of renderable entities (spheres, cylinders)
 * and provides methods for scene manipulation. Follows the Single
 * Responsibility Principle (SRP) by focusing solely on scene data
 * management.
 * 
 * The Scene class acts as an aggregate root for scene entities,
 * providing a clean interface for accessing and modifying scene
 * contents.
 */
class Scene
{
public:
    /**
     * @brief Constructs an empty scene.
     */
    Scene() = default;

    /**
     * @brief Constructs a scene with configuration.
     * 
     * @param config Scene configuration parameters.
     */
    explicit Scene(const SceneConfig& config);

    /**
     * @brief Copy constructor is deleted to prevent expensive copies.
     */
    Scene(const Scene&) = delete;

    /**
     * @brief Copy assignment is deleted to prevent expensive copies.
     */
    Scene& operator=(const Scene&) = delete;

    /**
     * @brief Move constructor.
     */
    Scene(Scene&&) noexcept = default;

    /**
     * @brief Move assignment operator.
     */
    Scene& operator=(Scene&&) noexcept = default;

    /**
     * @brief Default destructor.
     */
    ~Scene() = default;

    // ============ Sphere Management ============

    /**
     * @brief Adds a sphere to the scene.
     * 
     * @param sphere Sphere to add.
     */
    void addSphere(const glm::vec4& sphere);

    /**
     * @brief Adds multiple spheres to the scene.
     * 
     * @param spheres Vector of spheres to add.
     */
    void addSpheres(const std::vector<glm::vec4>& spheres);
    
    /**
     * @brief Adds multiple spheres to the scene from geometry::Sphere.
     * 
     * @param spheres Vector of geometry::Spheres to add.
     */
    void addSpheresFromClass(const std::vector<geometry::Sphere>& spheres);

    /**
     * @brief Gets all spheres in the scene.
     * 
     * @return Const reference to the sphere vector.
     */
    const std::vector<glm::vec4>& getSpheres() const { return m_spheres; }

    /**
     * @brief Gets mutable access to spheres.
     * 
     * @return Reference to the sphere vector.
     */
    std::vector<glm::vec4>& getSpheresMutable() { return m_spheres; }

    /**
     * @brief Gets the number of spheres in the scene.
     * 
     * @return Sphere count.
     */
    size_t getSphereCount() const { return m_spheres.size(); }
    
    /**
     * @brief Gets the data reference with the number of spheres in the scene.
     * 
     * @return Sphere count.
     */
    int& getSphereCountData() { sphereCount = m_spheres.size(); return sphereCount; }

    /**
     * @brief Reserves memory for spheres.
     * 
     * @param count Number of spheres to reserve space for.
     */
    void reserveSpheres(size_t count) { m_spheres.reserve(count); }

    /**
     * @brief Clears all spheres from the scene.
     */
    void clearSpheres() { m_spheres.clear(); }

    // ============ Cylinder Management ============

    /**
     * @brief Adds a cylinder to the scene.
     * 
     * @param cylinder Cylinder to add.
     */
    void addCylinder(const CylinderIndex& cylinder);

    /**
     * @brief Adds multiple cylinders to the scene.
     * 
     * @param cylinders Vector of cylinders to add.
     */
    void addCylinders(const std::vector<CylinderIndex>& cylinders);

    /**
     * @brief Gets all cylinders in the scene.
     * 
     * @return Const reference to the cylinder vector.
     */
    const std::vector<CylinderIndex>& getCylinders() const { return m_cylinders; }

    /**
     * @brief Gets mutable access to cylinders.
     * 
     * @return Reference to the cylinder vector.
     */
    std::vector<CylinderIndex>& getCylindersMutable() { return m_cylinders; }

    /**
     * @brief Gets the number of cylinders in the scene.
     * 
     * @return Cylinder count.
     */
    size_t getCylinderCount() const { return m_cylinders.size(); }
    
    /**
     * @brief Gets the reference of the number of cylinders in the scene.
     * 
     * @return Cylinder count.
     */
    int& getCylinderCountData() { cylinderCount = m_cylinders.size(); return cylinderCount; }
    
    /**
     * @brief Reserves memory for cylinders.
     * 
     * @param count Number of cylinders to reserve space for.
     */
    void reserveCylinders(size_t count) { m_cylinders.reserve(count); }

    /**
     * @brief Clears all cylinders from the scene.
     */
    void clearCylinders() { m_cylinders.clear(); }

    // ============ Scene Operations ============

    /**
     * @brief Clears all entities from the scene.
     */
    void clear();

    /**
     * @brief Gets the total entity count.
     * 
     * @return Total number of entities (spheres + cylinders).
     */
    size_t getTotalEntityCount() const { return m_spheres.size() + m_cylinders.size(); }

    /**
     * @brief Checks if the scene is empty.
     * 
     * @return True if no entities exist.
     */
    bool isEmpty() const { return m_spheres.empty() && m_cylinders.empty(); }

    /**
     * @brief Gets the scene configuration.
     * 
     * @return Const reference to scene configuration.
     */
    const SceneConfig& getConfig() const { return m_config; }

    /**
     * @brief Computes the bounding box of the scene.
     * 
     * @param outMin Output minimum corner of bounding box.
     * @param outMax Output maximum corner of bounding box.
     */
    void computeBoundingBox(glm::vec3& outMin, glm::vec3& outMax) const;

    /**
     * @brief Computes the center of the scene.
     * 
     * @return Center point of the scene.
     */
    glm::vec3 computeCenter() const;

    /**
     * @brief Gets the size of sphere data in bytes for GPU upload.
     * 
     * @return Size in bytes.
     */
    GLsizeiptr getSphereDataSize() const 
    { 
        return static_cast<GLsizeiptr>(m_spheres.size() * sizeof(glm::vec4)); 
    }

    /**
     * @brief Gets the size of cylinder data in bytes for GPU upload.
     * 
     * @return Size in bytes.
     */
    GLsizeiptr getCylinderDataSize() const 
    { 
        return static_cast<GLsizeiptr>(m_cylinders.size() * sizeof(CylinderIndex)); 
    }

    /**
     * @brief Gets a pointer to sphere data for GPU upload.
     * 
     * @return Const void pointer to sphere data.
     */
    const void* getSphereData() const { return m_spheres.data(); }

    /**
     * @brief Gets a pointer to cylinder data for GPU upload.
     * 
     * @return Const void pointer to cylinder data.
     */
    const void* getCylinderData() const { return m_cylinders.data(); }

private:
    std::vector<glm::vec4> m_spheres;      ///< Collection of spheres
    std::vector<CylinderIndex> m_cylinders;  ///< Collection of cylinders (index pairs + radius)
    SceneConfig m_config;               ///< Scene configuration

    int sphereCount = 0;
    int cylinderCount = 0;
};

#endif // _SCENE_H_
