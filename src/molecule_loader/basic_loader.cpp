#include <filesystem>
#include <stdexcept>
#include <set>
#include <tuple>
#include "molecule_loader/basic_loader.h"
#include "utils/math_defines.h"

ChemFilesLoader::ChemFilesLoader(const std::filesystem::path & path)
{
    load(path);
}

std::set<std::tuple<float,float,float,float>> unique_atoms;
std::set<std::pair<int,int>> unique_bonds;

void ChemFilesLoader::load(const std::filesystem::path & path)
{
    if (!std::filesystem::exists(path)) {
        std::cerr << "Molecule file " << path.string() <<  " not found." << std::endl;
        throw std::runtime_error("Molecule file '" + path.string() + "' not found.");
    }

    prepareChemfiles();

    chemfiles::Trajectory trayectory { path.string() };

    try 
    {
        chemfiles::Trajectory trajectory { path.string() };

        #ifndef NDEBUG
            std::cerr << "Debug: " << trajectory.nsteps() << " frames found" << std::endl;
        #endif

        if (trajectory.nsteps() == 0) 
            throw std::runtime_error("Trajectory is empty");
        
        int atoms_amount = 0;
        int bounds_amount = 0;
        for (size_t i = 0; i < trajectory.nsteps(); ++i) 
        {
            chemfiles::Frame frame = trajectory.read_step(i);

            // Obtén las posiciones de los átomos
            auto positions = frame.positions();
            auto topology = frame.topology();
            auto bonds = topology.bonds();

            // Itera sobre todos los átomos en el frame
            for (size_t j = 0; j < frame.size(); ++j) 
            {
                atoms_amount++;
                auto position = positions[j];
                auto atom = frame[j];

                // Obtén el radio del átomo (si está disponible)
                float radius = SymbolRadius[atom.atomic_number().value_or( 0 )];

                auto atom_key = std::make_tuple(position[0], position[1], position[2], radius * 0.2f);

                if (unique_atoms.insert(atom_key).second) 
                {
                    // Solo se inserta si no estaba antes
                    m_positions.push_back({position[0], position[1], position[2], radius * 0.2f});
                    atoms_amount++;
                }
            }

            for (const auto& bond : bonds) 
            {
                bounds_amount++;
                auto atom1 = bond[0];
                auto atom2 = bond[1];
    
                if (atom1 > atom2) std::swap(atom1, atom2); // normalizar orden

                auto bond_key = std::make_pair(atom1, atom2);

                if (unique_bonds.insert(bond_key).second) 
                {
                    m_bonds.push_back(bond_key);
                    bounds_amount++;
                }
            }
        }

        std::cout << "Atoms amount: " << atoms_amount << std::endl;
        std::cout << "Bonds amount: " << bounds_amount << std::endl;

    } 
    catch (const chemfiles::Error& e) 
    {
        std::cerr << "Error loading molecule file: " << e.what() << std::endl;
        throw;
    }

} 

std::pair<glm::vec4, glm::vec4> ChemFilesLoader::getBond(int index) const 
{ 
    // std::cout << "Getting bond at index: " << index << std::endl;
    // std::cout << "Bond atoms indexes: " << m_bonds[index].first << ", " << m_bonds[index].second << std::endl;
    return {m_positions[m_bonds[index].first], m_positions[m_bonds[index].second]}; 
}

void ChemFilesLoader::prepareChemfiles()
{
    std::cout << "Chemfiles version: " << CHEMFILES_VERSION << std::endl;
    #ifndef NDEBUG
        chemfiles::warning_callback_t callback = [](const std::string & p_log) { std::cerr << "Warning: " << p_log << std::endl; };
    #else
        chemfiles::warning_callback_t callback = [](const std::string & p_log) { /* std::cerr << "Warning: " << p_log << std::endl; */ };
    #endif
        chemfiles::set_warning_callback(callback);
}

std::vector<glm::vec4> & ChemFilesLoader::getSphereInfo()
{
    return m_positions;
}

std::vector<std::pair<int, int>> & ChemFilesLoader::getBondsInfo()
{
    return m_bonds;
}