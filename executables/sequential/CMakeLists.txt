# ==============================================================================
# executables/sequential/CMakeLists.txt - Versiones secuenciales (CPU)
# ==============================================================================
# 
# Ejecutables:
#   - seq_brute_force      : Rasterización bruta de esferas
#   - seq_cylinder_proj    : Proyección de cilindros en CPU
#
# ==============================================================================

message(STATUS "Configurando versiones secuenciales (CPU)...")

# ------------------------------------------------------------------------------
# Versión bruta secuencial (esferas)
# ------------------------------------------------------------------------------
add_executable(seq_brute_force
    ${CMAKE_SOURCE_DIR}/executables/sequential/brute_sequencial.cpp
)

target_link_libraries(seq_brute_force PRIVATE 
    lib_core
    lib_sequential
)

target_include_directories(seq_brute_force PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(seq_brute_force PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/sequential"
    OUTPUT_NAME "brute_sequencial"
)

# ------------------------------------------------------------------------------
# Versión secuencial con proyección de cilindros
# ------------------------------------------------------------------------------
add_executable(seq_cylinder_proj
    ${CMAKE_SOURCE_DIR}/executables/sequential/cylinder_projection_cpu.cpp
)

target_link_libraries(seq_cylinder_proj PRIVATE 
    lib_core
    lib_sequential
)

target_include_directories(seq_cylinder_proj PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

set_target_properties(seq_cylinder_proj PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/sequential"
    OUTPUT_NAME "cylinder_projection_cpu"
)

message(STATUS "  - seq_brute_force")
message(STATUS "  - seq_cylinder_proj")
