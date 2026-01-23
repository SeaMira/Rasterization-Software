# ==============================================================================
# CMakeLists.txt - Proyecto de Rasterización por Software
# ==============================================================================
#
# Este proyecto implementa varias versiones de un rasterizador:
#
#   1. SEQUENTIAL    - Rasterización en CPU (SDL Renderer)
#   2. FST_PARALLEL  - Primera versión con OpenGL compute shaders
#   3. SND_PARALLEL  - Segunda versión con OpenGL compute shaders
#   4. STANDARD      - Pipeline estándar de OpenGL
#   5. CUDA          - Interop CUDA-OpenGL para kernels de rasterización
#
# Uso:
#   cmake -B build -DBUILD_SEQUENTIAL_VERSION=ON -DBUILD_CUDA_VERSION=OFF
#   cmake --build build
#
# ==============================================================================

cmake_minimum_required(VERSION 3.20)

# Detectar si CUDA está disponible para habilitarlo condicionalmente
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(rasterization_software LANGUAGES CXX C CUDA)
    set(CUDA_AVAILABLE ON)
else()
    project(rasterization_software LANGUAGES CXX C)
    set(CUDA_AVAILABLE OFF)
    message(STATUS "CUDA no encontrado - versiones CUDA deshabilitadas")
endif()

# ==============================================================================
# OPCIONES DE BUILD
# ==============================================================================
# Activa/desactiva las versiones que deseas compilar

option(BUILD_SEQUENTIAL_VERSION   "Compilar versión secuencial (CPU)"              ON)
option(BUILD_FST_PARALLEL_VERSION "Compilar primera versión paralela (OpenGL)"     ON)
option(BUILD_SND_PARALLEL_VERSION "Compilar segunda versión paralela (OpenGL)"     ON)
option(BUILD_STANDARD_VERSION     "Compilar versión estándar del pipeline OpenGL"  ON)
option(BUILD_CUDA_VERSION         "Compilar versiones con CUDA interop"            ${CUDA_AVAILABLE})

# Validar que CUDA esté disponible si se solicita BUILD_CUDA_VERSION
if(BUILD_CUDA_VERSION AND NOT CUDA_AVAILABLE)
    message(WARNING "CUDA no está disponible - BUILD_CUDA_VERSION se desactivará")
    set(BUILD_CUDA_VERSION OFF)
endif()

# ==============================================================================
# INFORMACIÓN DE CONFIGURACIÓN
# ==============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS "  Proyecto de Rasterización por Software")
message(STATUS "============================================================")
message(STATUS "")
message(STATUS "Versiones a compilar:")
message(STATUS "  - Sequential (CPU):       ${BUILD_SEQUENTIAL_VERSION}")
message(STATUS "  - First Parallel (OpenGL): ${BUILD_FST_PARALLEL_VERSION}")
message(STATUS "  - Second Parallel (OpenGL): ${BUILD_SND_PARALLEL_VERSION}")
message(STATUS "  - Standard Pipeline:       ${BUILD_STANDARD_VERSION}")
message(STATUS "  - CUDA Interop:            ${BUILD_CUDA_VERSION}")
message(STATUS "")

# ==============================================================================
# INCLUIR MÓDULOS CMAKE
# ==============================================================================

# Configuración de compiladores (C++20, CUDA 17)
include(${CMAKE_SOURCE_DIR}/cmake/CompilerSettings.cmake)

# ==============================================================================
# BIBLIOTECAS DEL PROYECTO
# ==============================================================================

# Dependencias externas (OpenGL, SDL3, GLM, ImGui, chemfiles)
include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)

# Configuración de CUDA (arquitecturas, flags)
if(BUILD_CUDA_VERSION)
include(${CMAKE_SOURCE_DIR}/cmake/CUDASettings.cmake)
endif()

add_subdirectory(src)

# Funciones helper para crear targets
include(${CMAKE_SOURCE_DIR}/cmake/BuildTargets.cmake)

# Configuración de directorios de salida
include(${CMAKE_SOURCE_DIR}/cmake/OutputDirectories.cmake)


# ==============================================================================
# EJECUTABLES
# ==============================================================================

add_subdirectory(executables)

# ==============================================================================
# POST-BUILD: Assets y directorios
# ==============================================================================

# Copiar assets al directorio de build
setup_assets_copy()

# Crear estructura de directorios para media
setup_media_directories()

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS "  Configuración completada")
message(STATUS "============================================================")
message(STATUS "")
message(STATUS "Directorio de build:    ${CMAKE_BINARY_DIR}")
message(STATUS "Ejecutables en:         ${CMAKE_BINARY_DIR}/bin/")
message(STATUS "")
if(BUILD_CUDA_VERSION)
    message(STATUS "Arquitecturas CUDA:     ${CUDA_ARCH}")
endif()
message(STATUS "")
