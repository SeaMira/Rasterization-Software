#include <iostream>
#include <vector>
#include <filesystem>
#include "molecule_loader/basic_loader.h"

int main(int argc, char* argv[])
{
    std::filesystem::path path = "molecules/1AGA.mmtf";
    ChemFilesLoader loader(path);
    std::vector<std::pair<glm::vec3, float>> positions = loader.getSphereInfo();

    std::cout << "Number of atoms: " << positions.size() << std::endl;

    for (std::pair<glm::vec3, float>& atom : positions)
    {
        std::cout << "Position: (" << atom.first.x << ", " << atom.first.y << ", " << atom.first.z << ") Radius: " << atom.second << std::endl;
    }
    return 0;
}