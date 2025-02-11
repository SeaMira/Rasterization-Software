#ifndef _BASIC_LOADER_H_
#define _BASIC_LOADER_H_

#include <chemfiles.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <glm/glm.hpp>


class ChemFilesLoader
{
public:
    ChemFilesLoader() = default;
    ChemFilesLoader(const std::string& filename);
    ChemFilesLoader(const ChemFilesLoader&) = delete;
    ChemFilesLoader& operator=(const ChemFilesLoader&) = delete;
    ChemFilesLoader(ChemFilesLoader&&) noexcept = default;
    ChemFilesLoader& operator=(ChemFilesLoader&&) noexcept = default;
    ~ChemFilesLoader() = default;

    static void prepareChemfiles(); 
    
    void load( const std::filesystem::path & path );

    std::vector<std::pair<glm::vec3, float>> & getSphereInfo();

private:
    std::vector<std::pair<glm::vec3, float>> m_positions;


};

#endif // _BASIC_LOADER_H_