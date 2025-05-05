cmake_minimum_required(VERSION 3.20)

project(compute_shaders_project LANGUAGES CXX C)
set(CXX_STANDARD 20)

find_package(OpenGL REQUIRED)

add_subdirectory(src)
add_subdirectory(extern)

# Define SDL3 as static lib
set(SDL_SHARED OFF CACHE BOOL "" FORCE)   # Deactivates dynamic version
set(SDL_STATIC ON CACHE BOOL "" FORCE)    # Activates static version

target_link_libraries(raster_dependencies PUBLIC  glad glm chemfiles)
target_link_libraries(sequential_dependencies PUBLIC glm)
target_link_libraries(parallel_dependencies PUBLIC glm)

function(add_subproject project_name executable)
	add_executable(${project_name} ${executable})
	target_link_libraries(${project_name} PUBLIC raster_dependencies SDL3-static glad imgui)
	target_include_directories(${project_name} PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include extern/imgui)
endfunction()

# Sequential attempt define
add_subproject(brute_sequencial "${CMAKE_SOURCE_DIR}/executables/seq_version/brute_sequencial.cpp")

# Sequential cylinder projection attempt define
add_subproject(cylinder_projection_cpu "${CMAKE_SOURCE_DIR}/executables/seq_version/cylinder_projection_cpu.cpp")

# First parallel attempt define
add_subproject(fst_parallel "${CMAKE_SOURCE_DIR}/executables/fst_parallel_attempt/fst_parallel.cpp")
add_subproject(fst_parallel_w_cylinders "${CMAKE_SOURCE_DIR}/executables/fst_parallel_attempt/fst_parallel_w_cylinders.cpp")

# Second parallel attempt define
add_subproject(snd_parallel "${CMAKE_SOURCE_DIR}/executables/snd_parallel_attempt/bbox_intersect.cpp")

# Third parallel attempt define (same as second with parallel FC but without atomics)
add_subproject(thrd_parallel "${CMAKE_SOURCE_DIR}/executables/thrd_parallel_attempt/thrd_parallel_attempt.cpp")

target_link_libraries(brute_sequencial PUBLIC sequential_dependencies)
target_link_libraries(cylinder_projection_cpu PUBLIC sequential_dependencies)
target_link_libraries(fst_parallel PUBLIC parallel_dependencies)
target_link_libraries(fst_parallel_w_cylinders PUBLIC parallel_dependencies)
target_link_libraries(snd_parallel PUBLIC parallel_dependencies)
target_link_libraries(thrd_parallel PUBLIC parallel_dependencies)


add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets
    DEPENDS ${CMAKE_SOURCE_DIR}/assets
)

add_custom_target(copy_assets ALL DEPENDS "${CMAKE_BINARY_DIR}/assets")

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/off/seq)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/off/seq_w_cyl)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/off/fst_parallel)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/off/fst_parallel_w_cyl)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/off/scnd_parallel)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/seq)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/seq_w_cyl)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/fst_parallel)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/fst_parallel_w_cyl)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media/img/scnd_parallel)