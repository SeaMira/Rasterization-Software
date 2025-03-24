#ifndef _BASIC_LOADER_H_
#define _BASIC_LOADER_H_

#include <chemfiles.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <glm/glm.hpp>

/**
 * @class ChemFilesLoader
 * @brief A class to load and process chemical file data using the ChemFiles library.
 * 
 * This class provides functionality to load chemical files, store sphere information (positions and radiuses),
 * and prepare the ChemFiles library for use. The loader interacts with files on the filesystem and
 * processes them into a usable format.
 */
class ChemFilesLoader
{
public:
    /**
     * @brief Default constructor for the ChemFilesLoader class.
     * 
     * Initializes a default instance of the loader.
     */
    ChemFilesLoader() = default;

    /**
     * @brief Constructs a ChemFilesLoader instance with a specified file path.
     * 
     * Initializes a loader instance and loads the chemical file at the specified path.
     * 
     * @param path The path to the chemical file to be loaded.
     */
    ChemFilesLoader(const std::filesystem::path & path);

    /**
     * @brief Deleted copy constructor.
     * 
     * The copy constructor is deleted to prevent copying of ChemFilesLoader instances.
     */
    ChemFilesLoader(const ChemFilesLoader&) = delete;

    /**
     * @brief Deleted copy assignment operator.
     * 
     * The copy assignment operator is deleted to prevent assignment of ChemFilesLoader instances.
     */
    ChemFilesLoader& operator=(const ChemFilesLoader&) = delete;

    /**
     * @brief Move constructor for the ChemFilesLoader class.
     * @note This constructor transfers ownership of resources.
     */
    ChemFilesLoader(ChemFilesLoader&&) noexcept = default;

    /**
     * @brief Move assignment operator for the ChemFilesLoader class.
     * @note This operator transfers ownership of resources.
     */
    ChemFilesLoader& operator=(ChemFilesLoader&&) noexcept = default;

    /**
     * @brief Destructor for the ChemFilesLoader class.
     * 
     * Cleans up any resources held by the instance.
     */
    ~ChemFilesLoader() = default;

    /**
     * @brief Prepares the ChemFiles library for use.
     * 
     * This function is responsible for any initialization required for ChemFiles to function properly.
     * Rises a callback for any warning.
     */
    static void prepareChemfiles(); 
    
    /**
     * @brief Loads the chemical data from the specified file path.
     * @param path The path to the chemical file to load.
     * 
     * This function processes the specified file and stores the necessary data (specifically the atoms positions and radiuses).
     */
    void load( const std::filesystem::path & path );

    /**
     * @brief Returns a reference to the vector of sphere position data.
     * @return A reference to a vector of glm::vec4 containing the sphere positions and their radius.
     * 
     * Atoms are represented as spheres, so sphere positions are stored as glm::vec4, where the x, y, z coordinates represent the 
     * position, and the w component the radius.
     */
    std::vector<glm::vec4> & getSphereInfo();

private:
    std::vector<glm::vec4> m_positions; /**< A vector of glm::vec4 containing the sphere positions and their radius. Each sphere represents an atom. */


};

#endif // _BASIC_LOADER_H_