# ==============================================================================
# CUDASettings.cmake - Configuración de CUDA
# ==============================================================================

# Buscar CUDA
find_package(CUDA REQUIRED)

# Arquitecturas GPU soportadas
# Ampere (RTX 30xx): 86
# Turing (RTX 20xx, GTX 16xx): 75
# Ada Lovelace (RTX 40xx): 89
set(CUDA_ARCH "75;86;89" CACHE STRING "CUDA architectures to compile for")

# Flags adicionales de CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Función para obtener las bibliotecas CUDA necesarias
function(get_cuda_libraries OUTPUT_VAR)
    if(WIN32)
        set(${OUTPUT_VAR}
            "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib"
            "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib"
            PARENT_SCOPE
        )
    else()
        set(${OUTPUT_VAR}
            cuda
            cudart_static
            PARENT_SCOPE
        )
    endif()
endfunction()

# Función para configurar propiedades CUDA en un target
function(configure_cuda_target TARGET_NAME)
    target_compile_options(${TARGET_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    )
    set_target_properties(${TARGET_NAME} PROPERTIES 
        CUDA_ARCHITECTURES "${CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    get_cuda_libraries(CUDA_LIBS)
    target_link_libraries(${TARGET_NAME} PUBLIC ${CUDA_LIBS})
endfunction()
