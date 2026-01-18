# ==============================================================================
# src/CMakeLists.txt - Bibliotecas del proyecto de rasterización
# ==============================================================================
# 
# Este archivo define las bibliotecas principales del proyecto:
#   - lib_core      : Componentes comunes (UI, UX, visualización, geometría)
#   - lib_sequential: Funciones específicas para versión secuencial CPU
#   - lib_parallel  : Funciones específicas para versiones paralelas OpenGL
#   - lib_cuda      : Wrappers para interop CUDA-OpenGL
#
# ==============================================================================

# ==============================================================================
# SECCIÓN 1: Biblioteca Core (componentes compartidos)
# ==============================================================================

set(CORE_HEADERS
    # Algorithms
    ${CMAKE_SOURCE_DIR}/include/algorithms/frustum_cull.h
    ${CMAKE_SOURCE_DIR}/include/algorithms/occlusion_cull.h
    
    # Geometry
    ${CMAKE_SOURCE_DIR}/include/geometry/cylinder/cylinder.h
    ${CMAKE_SOURCE_DIR}/include/geometry/sphere/sphere.h
    
    # Molecule Loader
    ${CMAKE_SOURCE_DIR}/include/molecule_loader/basic_loader.h
    
    # UI
    ${CMAKE_SOURCE_DIR}/include/ui/components/scene_info.h
    ${CMAKE_SOURCE_DIR}/include/ui/components/camera_info.h
    ${CMAKE_SOURCE_DIR}/include/ui/components/input_info.h
    ${CMAKE_SOURCE_DIR}/include/ui/components/benchmark_info.h
    ${CMAKE_SOURCE_DIR}/include/ui/app_ui.h
    ${CMAKE_SOURCE_DIR}/include/ui/component.h
    ${CMAKE_SOURCE_DIR}/include/ui/element.h
    
    # Utils
    ${CMAKE_SOURCE_DIR}/include/utils/math_defines.h
    ${CMAKE_SOURCE_DIR}/include/utils/benchmark_resources.h
    # scene_descriptor.h es específico de CUDA
    
    # UX
    ${CMAKE_SOURCE_DIR}/include/ux/camera_controller.h
    ${CMAKE_SOURCE_DIR}/include/ux/camera.h
    ${CMAKE_SOURCE_DIR}/include/ux/input.h
    ${CMAKE_SOURCE_DIR}/include/ux/cinematic/camera_controlled_movement.h
    ${CMAKE_SOURCE_DIR}/include/ux/cinematic/benchmark.h
    ${CMAKE_SOURCE_DIR}/include/ux/profiler/profiler.h
    
    # Visualization (OpenGL)
    ${CMAKE_SOURCE_DIR}/include/vis/compute_shader_program.h
    ${CMAKE_SOURCE_DIR}/include/vis/shader_program.h
    ${CMAKE_SOURCE_DIR}/include/vis/window.h
    ${CMAKE_SOURCE_DIR}/include/vis/canvas.h
    ${CMAKE_SOURCE_DIR}/include/vis/gl/frame_buffer.h
    ${CMAKE_SOURCE_DIR}/include/vis/gl/texture.h
    ${CMAKE_SOURCE_DIR}/include/vis/gl/storage_buffer.h
    ${CMAKE_SOURCE_DIR}/include/vis/gl/depth_downsample.h
    ${CMAKE_SOURCE_DIR}/include/vis/gl/depth_data.h
    
    # Application
    ${CMAKE_SOURCE_DIR}/include/appSDLGL.h
    ${CMAKE_SOURCE_DIR}/include/appSDLRenderer.h
)

set(CORE_SOURCES
    # Algorithms
    ${CMAKE_SOURCE_DIR}/src/algorithms/frustum_cull.cpp
    ${CMAKE_SOURCE_DIR}/src/algorithms/occlusion_cull.cpp
    
    # Geometry
    ${CMAKE_SOURCE_DIR}/src/geometry/cylinder/cylinder.cpp
    ${CMAKE_SOURCE_DIR}/src/geometry/sphere/sphere.cpp
    
    # Molecule Loader
    ${CMAKE_SOURCE_DIR}/src/molecule_loader/basic_loader.cpp
    
    # UI
    ${CMAKE_SOURCE_DIR}/src/ui/components/scene_info.cpp
    ${CMAKE_SOURCE_DIR}/src/ui/components/camera_info.cpp
    ${CMAKE_SOURCE_DIR}/src/ui/components/input_info.cpp
    ${CMAKE_SOURCE_DIR}/src/ui/components/benchmark_info.cpp
    ${CMAKE_SOURCE_DIR}/src/ui/app_ui.cpp
    ${CMAKE_SOURCE_DIR}/src/ui/component.cpp
    
    # Utils
    ${CMAKE_SOURCE_DIR}/src/utils/benchmark_resources.cpp
    # scene_descriptor.cpp es específico de CUDA
    
    # UX
    ${CMAKE_SOURCE_DIR}/src/ux/camera_controller.cpp
    ${CMAKE_SOURCE_DIR}/src/ux/camera.cpp
    ${CMAKE_SOURCE_DIR}/src/ux/input.cpp
    ${CMAKE_SOURCE_DIR}/src/ux/cinematic/camera_controlled_movement.cpp
    ${CMAKE_SOURCE_DIR}/src/ux/cinematic/benchmark.cpp
    ${CMAKE_SOURCE_DIR}/src/ux/profiler/profiler.cpp
    
    # Visualization (OpenGL)
    ${CMAKE_SOURCE_DIR}/src/vis/compute_shader_program.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/shader_program.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/window.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/canvas.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/gl/frame_buffer.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/gl/texture.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/gl/storage_buffer.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/gl/depth_downsample.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/gl/depth_data.cpp
    
    # Application
    ${CMAKE_SOURCE_DIR}/src/appSDLGL.cpp
    ${CMAKE_SOURCE_DIR}/src/appSDLRenderer.cpp
)

add_library(lib_core STATIC ${CORE_HEADERS} ${CORE_SOURCES})

target_include_directories(lib_core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
)

target_link_libraries(lib_core PUBLIC 
    ${COMMON_GRAPHICS_LIBS}
    chemfiles
)

set_target_properties(lib_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ==============================================================================
# SECCIÓN 2: Biblioteca Sequential (versión CPU)
# ==============================================================================

set(SEQUENTIAL_HEADERS
    ${CMAKE_SOURCE_DIR}/include/utils/sequential/common.h
    ${CMAKE_SOURCE_DIR}/include/utils/sequential/aux_functions_sphere.h
    ${CMAKE_SOURCE_DIR}/include/utils/sequential/aux_functions_cylinder.h
)

set(SEQUENTIAL_SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils/sequential/common.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/sequential/aux_functions_sphere.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/sequential/aux_functions_cylinder.cpp
)

add_library(lib_sequential STATIC ${SEQUENTIAL_HEADERS} ${SEQUENTIAL_SOURCES})

target_include_directories(lib_sequential PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
)

target_link_libraries(lib_sequential PUBLIC glm)

set_target_properties(lib_sequential PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# OpenMP para paralelización en CPU
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP encontrado: ${OpenMP_CXX_VERSION}")
    target_link_libraries(lib_sequential PRIVATE OpenMP::OpenMP_CXX)
endif()

# ==============================================================================
# SECCIÓN 3: Biblioteca Parallel (versiones OpenGL compute shaders)
# ==============================================================================

set(PARALLEL_HEADERS
    ${CMAKE_SOURCE_DIR}/include/utils/parallel/aux_functions.h
    ${CMAKE_SOURCE_DIR}/include/utils/parallel/cuda_checks.h
)

set(PARALLEL_SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils/parallel/aux_functions.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/parallel/cuda_checks.cpp
)

add_library(lib_parallel STATIC ${PARALLEL_HEADERS} ${PARALLEL_SOURCES})

target_include_directories(lib_parallel PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${EXTERN_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
)

target_link_libraries(lib_parallel PUBLIC 
    glm 
    glad
)

set_target_properties(lib_parallel PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ==============================================================================
# SECCIÓN 4: Biblioteca CUDA (interop OpenGL-CUDA)
# Sólo se compila si CUDA está disponible y BUILD_CUDA_VERSION está activo
# ==============================================================================

if(BUILD_CUDA_VERSION)
    set(CUDA_INTEROP_HEADERS
        ${CMAKE_SOURCE_DIR}/include/vis/gl/cu/canvas_cu.h
        ${CMAKE_SOURCE_DIR}/include/vis/gl/cu/texture_cu.h
        ${CMAKE_SOURCE_DIR}/include/vis/gl/cu/depth_data_cu.h
        ${CMAKE_SOURCE_DIR}/include/vis/gl/cu/storage_buffer_cu.h
        ${CMAKE_SOURCE_DIR}/include/vis/gl/cu/depth_downsample_cu.h
    )

    set(CUDA_INTEROP_SOURCES
        ${CMAKE_SOURCE_DIR}/src/vis/gl/cu/canvas_cu.cpp
        ${CMAKE_SOURCE_DIR}/src/vis/gl/cu/texture_cu.cpp
        ${CMAKE_SOURCE_DIR}/src/vis/gl/cu/depth_data_cu.cpp
        ${CMAKE_SOURCE_DIR}/src/vis/gl/cu/storage_buffer_cu.cpp
        ${CMAKE_SOURCE_DIR}/src/vis/gl/cu/depth_downsample_cu.cpp
    )

    add_library(lib_cuda_interop STATIC ${CUDA_INTEROP_HEADERS} ${CUDA_INTEROP_SOURCES})

    target_include_directories(lib_cuda_interop PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
        ${EXTERN_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
    )

    get_cuda_libraries(CUDA_LIBS)
    target_link_libraries(lib_cuda_interop PUBLIC 
        lib_core
        lib_parallel
        ${CUDA_LIBS}
    )

    configure_cuda_target(lib_cuda_interop)
    
    set_target_properties(lib_cuda_interop PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()
