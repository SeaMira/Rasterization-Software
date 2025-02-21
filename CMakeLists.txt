cmake_minimum_required(VERSION 3.20)

function(add_dependency_folder folder_target input_folder output_folder)	
	file(GLOB_RECURSE folder_content ${input_folder}/*.*)
	add_custom_target(  ${folder_target} COMMAND ${CMAKE_COMMAND} -E copy_directory ${input_folder}/ ${output_folder}/)
	set_property(TARGET ${folder_target} APPEND PROPERTY SOURCES ${folder_content}) 
endfunction()


project(compute_shaders_project LANGUAGES CXX C)
set(CXX_STANDARD 20)

find_package(OpenGL REQUIRED)

add_subdirectory(src)
add_subdirectory(extern)

# Define SDL3 as static lib
set(SDL_SHARED OFF CACHE BOOL "" FORCE)   # Deactivates dynamic version
set(SDL_STATIC ON CACHE BOOL "" FORCE)    # Activates static version

add_dependency_folder(data_folder_bin "${CMAKE_CURRENT_SOURCE_DIR}/assets" "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
add_dependency_folder(data_folder "${CMAKE_CURRENT_SOURCE_DIR}/assets" "${CMAKE_BINARY_DIR}")

target_link_libraries(raster_dependencies PUBLIC  glad glm chemfiles)
target_link_libraries(sequential_dependencies PUBLIC glm)

function(add_subproject project_name executable)
	add_executable(${project_name} ${executable})
	target_link_libraries(${project_name} PUBLIC raster_dependencies SDL3-static glad)
	target_include_directories(${project_name} PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include)
endfunction()

add_subproject(controllers_example "${CMAKE_SOURCE_DIR}/executables/controllers_example.cpp")
add_subproject(shaders_example "${CMAKE_SOURCE_DIR}/executables/shaders_example.cpp")
add_subproject(loader_example "${CMAKE_SOURCE_DIR}/executables/loader_example.cpp")
add_subproject(brute_sequencial "${CMAKE_SOURCE_DIR}/executables/brute_sequencial.cpp")
add_subproject(fst_parallel "${CMAKE_SOURCE_DIR}/executables/fst_parallel.cpp")
add_subproject(gltexstor_ex "${CMAKE_SOURCE_DIR}/executables/gltexstor_ex.cpp")

target_link_libraries(brute_sequencial PUBLIC sequential_dependencies)


add_dependencies(shaders_example data_folder_bin)
add_dependencies(shaders_example data_folder)
add_dependencies(loader_example data_folder_bin)
add_dependencies(loader_example data_folder)
