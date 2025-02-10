cmake_minimum_required(VERSION 3.20)

project(compute_shaders_project LANGUAGES CXX C)
set(CXX_STANDARD 20)

find_package(OpenGL REQUIRED)
# find_package(SDL2 REQUIRED)

add_subdirectory(src)
add_subdirectory(extern)

# Definir SDL3 como una biblioteca est�tica
set(SDL_SHARED OFF CACHE BOOL "" FORCE)   # Desactiva la versi�n din�mica
set(SDL_STATIC ON CACHE BOOL "" FORCE)    # Activa la versi�n est�tica

add_executable(controllers_example "controllers_example.cpp")
add_executable(shaders_example "shaders_example.cpp")

target_link_libraries(raster_dependencies PUBLIC  glad glm)

target_link_libraries(controllers_example PUBLIC raster_dependencies SDL3-static glad)
target_link_libraries(shaders_example PUBLIC raster_dependencies SDL3-static glad)

target_include_directories(controllers_example PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include)
target_include_directories(shaders_example PUBLIC ${OPENGL_INCLUDE_DIRS} extern/SDL/include)


set(SHADER_FILES
    ${CMAKE_SOURCE_DIR}/assets/shaders/raster.compute
)

# Copia los archivos .cs al directorio de salida del ejecutable
add_custom_command(TARGET shaders_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${SHADER_FILES}
    $<TARGET_FILE_DIR:shaders_example>
)