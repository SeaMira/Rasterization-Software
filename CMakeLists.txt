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

add_executable(controllers_example "controllers_example.cpp")
add_executable(shaders_example "shaders_example.cpp")
add_executable(loader_example "loader_example.cpp")

add_dependency_folder(data_folder_bin "${CMAKE_CURRENT_SOURCE_DIR}/assets" "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
add_dependency_folder(data_folder "${CMAKE_CURRENT_SOURCE_DIR}/assets" "${CMAKE_BINARY_DIR}")

target_link_libraries(raster_dependencies PUBLIC  glad glm)

target_link_libraries(controllers_example PUBLIC raster_dependencies SDL3-static glad)
target_link_libraries(shaders_example PUBLIC raster_dependencies SDL3-static glad)
target_link_libraries(loader_example PUBLIC raster_dependencies)

target_include_directories(controllers_example PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include)
target_include_directories(shaders_example PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include)
target_include_directories(loader_example PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include)

add_dependencies(shaders_example data_folder_bin)
add_dependencies(shaders_example data_folder)
add_dependencies(loader_example data_folder_bin)
add_dependencies(loader_example data_folder)

# set(SHADER_FILES
#     ${CMAKE_SOURCE_DIR}/assets/shaders/raster.compute
# )

# # Copia los archivos .cs al directorio de salida del ejecutable
# add_custom_command(TARGET shaders_example POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     ${SHADER_FILES}
#     $<TARGET_FILE_DIR:shaders_example>
# )
