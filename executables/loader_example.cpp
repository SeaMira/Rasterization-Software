#include <iostream>
#include <vector>
#include <filesystem>
#include <glm/glm.hpp>
#include "molecule_loader/basic_loader.h"

int main(int argc, char* argv[])
{
    std::filesystem::path path = "molecules/1AGA.mmtf";
    ChemFilesLoader loader(path);
    std::vector<glm::vec4>& positions = loader.getSphereInfo();

    std::cout << "Number of atoms: " << positions.size() << std::endl;

    for (glm::vec4& atom : positions)
    {
        std::cout << "Position: (" << atom.x << ", " << atom.y << ", " << atom.z << ") Radius: " << atom.w << std::endl;
    }
    return 0;
}