cmake_minimum_required(VERSION 3.20)

project(compute_shaders_project LANGUAGES CXX C CUDA)
set(CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)

# ===============================
# Arch GPU para CUDA
# ===============================
find_package(CUDA REQUIRED)
set(CUDA_ARCH "75;86;89") # ajusta según tus tarjetas
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_subdirectory(src)
add_subdirectory(extern)

# Define SDL3 as static lib
set(SDL_SHARED OFF CACHE BOOL "" FORCE)   # Deactivates dynamic version
set(SDL_STATIC ON CACHE BOOL "" FORCE)    # Activates static version

target_link_libraries(raster_dependencies PUBLIC  glad glm chemfiles)
target_link_libraries(sequential_dependencies PUBLIC glm)
target_link_libraries(parallel_dependencies PUBLIC glm glad)

function(add_subproject project_name executable)
	add_executable(${project_name} ${executable})
	target_link_libraries(${project_name} PUBLIC raster_dependencies SDL3-static glad imgui)
	target_include_directories(${project_name} PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include extern/imgui)
endfunction()

function(add_cuda_subproject project_name executable kernel_dir)
    file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS "${kernel_dir}/*.cu")

	add_executable(${project_name} ${executable} ${KERNEL_SOURCES})

    set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE CUDA)

	target_link_libraries(${project_name} PUBLIC raster_dependencies SDL3-static glad imgui)
	target_include_directories(${project_name} PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include extern/imgui)
endfunction()

# Sequential attempt define
add_subproject(brute_sequencial "${CMAKE_SOURCE_DIR}/executables/seq_version/brute_sequencial.cpp")

# Sequential cylinder projection attempt define
add_subproject(cylinder_projection_cpu "${CMAKE_SOURCE_DIR}/executables/seq_version/cylinder_projection_cpu.cpp")

# First parallel attempt define
add_subproject(fst_cpu "${CMAKE_SOURCE_DIR}/executables/fst_parallel_attempt/cpu_cull/sphs.cpp")
add_subproject(fst_cpu_w_cylinders "${CMAKE_SOURCE_DIR}/executables/fst_parallel_attempt/cpu_cull/w_cyls.cpp")
add_subproject(fst_gpu "${CMAKE_SOURCE_DIR}/executables/fst_parallel_attempt/gpu_cull/sphs.cpp")
add_subproject(fst_gpu_w_cylinders "${CMAKE_SOURCE_DIR}/executables/fst_parallel_attempt/gpu_cull/w_cyls.cpp")

# Second parallel attempt define
add_subproject(snd_cpu "${CMAKE_SOURCE_DIR}/executables/snd_parallel_attempt/cpu_cull/sphs.cpp")
add_subproject(snd_cpu_w_cylinders "${CMAKE_SOURCE_DIR}/executables/snd_parallel_attempt/cpu_cull/w_cyls.cpp")
add_subproject(snd_gpu "${CMAKE_SOURCE_DIR}/executables/snd_parallel_attempt/gpu_cull/sphs.cpp")
add_subproject(snd_gpu_w_cylinders "${CMAKE_SOURCE_DIR}/executables/snd_parallel_attempt/gpu_cull/w_cyls.cpp")

# Standard pipeline version
add_subproject(standard_pipeline "${CMAKE_SOURCE_DIR}/executables/standard_version/gpu_cull/standard_version.cpp")
add_subproject(standard_pipeline_cpu_cull "${CMAKE_SOURCE_DIR}/executables/standard_version/cpu_cull/standard_version.cpp")


# First CUDA version
add_cuda_subproject(CUDA_fst 
    "${CMAKE_SOURCE_DIR}/executables/cuda_fst/gpu_cull/w_cyls.cpp"
    "${CMAKE_SOURCE_DIR}/kernels/fst/gpu_cull"    
)

target_link_libraries(brute_sequencial PUBLIC sequential_dependencies)
target_link_libraries(cylinder_projection_cpu PUBLIC sequential_dependencies)

function(add_cuda_props project_name )
	target_compile_options(${project_name} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
    )
	set_target_properties(${project_name} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}")
	target_link_libraries(${project_name} PUBLIC 
        parallel_dependencies
        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib"
        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib"
    )
endfunction()


target_link_libraries(fst_cpu PUBLIC parallel_dependencies)
target_link_libraries(fst_cpu_w_cylinders PUBLIC parallel_dependencies)
target_link_libraries(fst_gpu PUBLIC parallel_dependencies)
target_link_libraries(fst_gpu_w_cylinders PUBLIC parallel_dependencies)

target_link_libraries(snd_cpu PUBLIC parallel_dependencies)
target_link_libraries(snd_cpu_w_cylinders PUBLIC parallel_dependencies)
target_link_libraries(snd_gpu PUBLIC parallel_dependencies)
target_link_libraries(snd_gpu_w_cylinders PUBLIC parallel_dependencies)

target_link_libraries(standard_pipeline PUBLIC parallel_dependencies)
target_link_libraries(standard_pipeline_cpu_cull PUBLIC parallel_dependencies)


add_cuda_props(CUDA_fst)



add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets
    DEPENDS ${CMAKE_SOURCE_DIR}/assets
)

add_custom_target(copy_assets ALL DEPENDS "${CMAKE_BINARY_DIR}/assets")

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/seq)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/seq_w_cyl)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/fst_parallel/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/fst_parallel/gpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/fst_parallel_w_cyl/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/fst_parallel_w_cyl/gpu_cull)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/scnd_parallel/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/scnd_parallel/gpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/scnd_parallel_w_cyl/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/scnd_parallel_w_cyl/gpu_cull)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/standard_version/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/csv/standard_version/gpu_cull)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/seq)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/seq_w_cyl)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/fst_parallel/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/fst_parallel/gpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/fst_parallel_w_cyl/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/fst_parallel_w_cyl/gpu_cull)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/scnd_parallel/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/scnd_parallel/gpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/scnd_parallel_w_cyl/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/scnd_parallel_w_cyl/gpu_cull)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/standard_version/cpu_cull)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/standard_version/gpu_cull)