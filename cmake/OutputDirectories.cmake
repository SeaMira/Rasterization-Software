# ==============================================================================
# OutputDirectories.cmake - Configuración de directorios de salida
# ==============================================================================

# Configurar directorios de salida para binarios
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ------------------------------------------------------------------------------
# Crear estructura de directorios para media (CSV e imágenes)
# ------------------------------------------------------------------------------
function(setup_media_directories)
    # Directorios CSV
    set(CSV_DIRS
        media/csv/sequential
        media/csv/sequential_w_cyl
        media/csv/fst_parallel/cpu_cull
        media/csv/fst_parallel/gpu_cull
        media/csv/fst_parallel_w_cyl/cpu_cull
        media/csv/fst_parallel_w_cyl/gpu_cull
        media/csv/snd_parallel/cpu_cull
        media/csv/snd_parallel/gpu_cull
        media/csv/snd_parallel_w_cyl/cpu_cull
        media/csv/snd_parallel_w_cyl/gpu_cull
        media/csv/standard_version/cpu_cull
        media/csv/standard_version/gpu_cull
        media/csv/cuda/gpu_cull
    )
    
    # Directorios de imágenes
    set(IMG_DIRS
        media/img/sequential
        media/img/sequential_w_cyl
        media/img/fst_parallel/cpu_cull
        media/img/fst_parallel/gpu_cull
        media/img/fst_parallel_w_cyl/cpu_cull
        media/img/fst_parallel_w_cyl/gpu_cull
        media/img/snd_parallel/cpu_cull
        media/img/snd_parallel/gpu_cull
        media/img/snd_parallel_w_cyl/cpu_cull
        media/img/snd_parallel_w_cyl/gpu_cull
        media/img/standard_version/cpu_cull
        media/img/standard_version/gpu_cull
        media/img/cuda/gpu_cull
    )
    
    # Crear todos los directorios
    foreach(dir ${CSV_DIRS} ${IMG_DIRS})
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${dir})
    endforeach()
endfunction()

# ------------------------------------------------------------------------------
# Copiar assets al directorio de build
# ------------------------------------------------------------------------------
function(setup_assets_copy)
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/assets"
        COMMAND ${CMAKE_COMMAND} -E copy_directory 
            ${CMAKE_SOURCE_DIR}/assets 
            ${CMAKE_BINARY_DIR}/assets
        DEPENDS ${CMAKE_SOURCE_DIR}/assets
        COMMENT "Copiando assets al directorio de build..."
    )
    
    add_custom_target(copy_assets ALL 
        DEPENDS "${CMAKE_BINARY_DIR}/assets"
    )
endfunction()
