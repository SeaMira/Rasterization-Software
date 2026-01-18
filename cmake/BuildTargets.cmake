# ==============================================================================
# BuildTargets.cmake - Funciones helper para crear targets ejecutables
# ==============================================================================

# ------------------------------------------------------------------------------
# add_rasterizer_executable
# Crea un ejecutable estándar para el proyecto de rasterización
# 
# Parámetros:
#   TARGET_NAME     - Nombre del target
#   SOURCE_FILE     - Archivo fuente principal (.cpp)
#   DEPENDENCIES    - Library target de dependencias a linkear
# ------------------------------------------------------------------------------
function(add_rasterizer_executable TARGET_NAME SOURCE_FILE DEPENDENCIES)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    
    target_link_libraries(${TARGET_NAME} PRIVATE 
        ${DEPENDENCIES}
        ${COMMON_GRAPHICS_LIBS}
    )
    
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${EXTERN_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
    )
    
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${DEPENDENCIES}"
    )
endfunction()

# ------------------------------------------------------------------------------
# add_cuda_rasterizer_executable
# Crea un ejecutable CUDA con kernels para el proyecto
# 
# Parámetros:
#   TARGET_NAME     - Nombre del target
#   SOURCE_FILE     - Archivo fuente principal (.cpp)
#   KERNEL_DIR      - Directorio con archivos .cu
#   DEPENDENCIES    - Library target de dependencias a linkear
# ------------------------------------------------------------------------------
function(add_cuda_rasterizer_executable TARGET_NAME SOURCE_FILE KERNEL_DIR DEPENDENCIES)
    # Recolectar todos los archivos .cu del directorio de kernels
    file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${KERNEL_DIR}/*.cu")
    
    add_executable(${TARGET_NAME} ${SOURCE_FILE} ${KERNEL_SOURCES})
    
    # Marcar los archivos .cu como CUDA
    set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE CUDA)
    
    target_link_libraries(${TARGET_NAME} PRIVATE 
        ${DEPENDENCIES}
        ${COMMON_GRAPHICS_LIBS}
    )
    
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${EXTERN_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
    )
    
    # Configurar propiedades CUDA
    configure_cuda_target(${TARGET_NAME})
    
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/cuda"
    )
endfunction()

# ------------------------------------------------------------------------------
# add_sequential_executable  
# Crea un ejecutable para versión secuencial (CPU only)
# ------------------------------------------------------------------------------
function(add_sequential_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    
    target_link_libraries(${TARGET_NAME} PRIVATE 
        lib_sequential
        lib_core
        ${COMMON_GRAPHICS_LIBS}
    )
    
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${EXTERN_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
    )
    
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/sequential"
    )
endfunction()

# ------------------------------------------------------------------------------
# add_opengl_executable
# Crea un ejecutable para versiones paralelas OpenGL (compute shaders)
# ------------------------------------------------------------------------------
function(add_opengl_executable TARGET_NAME SOURCE_FILE VERSION_NAME)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    
    target_link_libraries(${TARGET_NAME} PRIVATE 
        lib_parallel
        lib_core
        ${COMMON_GRAPHICS_LIBS}
    )
    
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${EXTERN_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
    )
    
    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${VERSION_NAME}"
    )
endfunction()
