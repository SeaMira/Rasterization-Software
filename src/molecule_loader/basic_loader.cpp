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

void ChemFilesLoader::load(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        std::cerr << "Molecule file " << path << " not found." << std::endl;
        throw std::runtime_error("Molecule file '" + path.string() + "' not found.");
    }

    prepareChemfiles();
    m_positions.clear();
    m_bonds.clear();

    try {
        chemfiles::Trajectory trajectory{ path.string() };

        #ifndef NDEBUG
            std::cerr << "Debug: " << trajectory.nsteps() << " frames found" << std::endl;
        #endif

        int n_steps = trajectory.nsteps();
        if (n_steps == 0)
            throw std::runtime_error("Trajectory is empty");


        std::unordered_map<glm::vec4, int, Vec4Hash, Vec4Equal> atom_index_map;
        std::unordered_set<std::pair<int,int>, BondHash, BondEqual> bond_set; 
        
        int frames_atoms_sum = 0;
        int frames_bonds_sum = 0;

        for (int i = 0; i < n_steps; ++i) 
        {
            chemfiles::Frame frame = trajectory.read_step(i);
            frames_atoms_sum += frame.size();
            frames_bonds_sum += frame.topology().bonds().size();
        }
        
        // reserva inicial
        atom_index_map.reserve(frames_atoms_sum); 
        bond_set.reserve(frames_bonds_sum);

        for (int i = 0; i < n_steps; ++i) 
        {
            chemfiles::Frame frame = trajectory.read_step(i);
            auto positions = frame.positions();
            auto topology  = frame.topology();
            auto bonds     = topology.bonds();

            std::vector<int> global_indices(frame.size());

            // --- Procesar átomos ---
            for (int j = 0; j < frame.size(); ++j) 
            {
                auto position = positions[j];
                auto atom     = frame[j];
                float radius  = SymbolRadius[atom.atomic_number().value_or(0)];

                glm::vec4 atom_vec(position[0], position[1], position[2], radius * 0.2f);

                auto it = atom_index_map.find(atom_vec);
                if (it != atom_index_map.end()) 
                    global_indices[j] = it->second;
                else 
                {
                    int new_index = static_cast<int>(m_positions.size());
                    m_positions.emplace_back(position[0], position[1], position[2], radius * 0.2f);
                    atom_index_map.emplace(m_positions.back(), new_index);
                    global_indices[j] = new_index;
                }
            }

            // --- Procesar enlaces ---
            for (const auto& bond : bonds) 
            {
                int a = global_indices[bond[0]];
                int b = global_indices[bond[1]];
                auto [min_a, max_b] = std::minmax(a, b);
                if (bond_set.insert({min_a, max_b}).second) 
                {
                    m_bonds.emplace_back(min_a, max_b);
                }
            }
        }

        std::cout << "Atoms amount: " << m_positions.size() << std::endl;
        std::cout << "Bonds amount: " << m_bonds.size() << std::endl;

    } catch (const chemfiles::Error& e) {
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