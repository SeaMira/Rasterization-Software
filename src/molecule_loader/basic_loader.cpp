#include <filesystem>
#include <stdexcept>

#include "molecule_loader/basic_loader.h"
#include "utils/math_defines.h"

ChemFilesLoader::ChemFilesLoader(const std::filesystem::path & path)
{
    load(path);
}

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
        

        for (size_t i = 0; i < trajectory.nsteps(); ++i) 
        {
            chemfiles::Frame frame = trajectory.read();

            // Obtén las posiciones de los átomos
            auto positions = frame.positions();

            // Itera sobre todos los átomos en el frame
            for (size_t j = 0; j < frame.size(); ++j) 
            {
                auto position = positions[j];
                auto atom = frame[j];

                // Obtén el radio del átomo (si está disponible)
                float radius = SymbolRadius[atom.atomic_number().value_or( 0 )];

                m_positions.push_back({position[0], position[1], position[2], radius});
            }
        }

    } 
    catch (const chemfiles::Error& e) 
    {
        std::cerr << "Error loading molecule file: " << e.what() << std::endl;
        throw;
    }

} 

void ChemFilesLoader::prepareChemfiles()
{
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