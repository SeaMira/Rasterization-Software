# ==============================================================================
# Dependencies.cmake - Configuración de dependencias externas
# ==============================================================================

# OpenGL
find_package(OpenGL REQUIRED)

# SDL3 configuración
set(SDL_SHARED OFF CACHE BOOL "Build SDL as shared library" FORCE)
set(SDL_STATIC ON CACHE BOOL "Build SDL as static library" FORCE)
set(SDL2_DISABLE_UNINSTALL ON CACHE BOOL "Disable uninstall target" FORCE)
set(SDL_TEST OFF CACHE BOOL "Build SDL tests" FORCE)
set(SDL_TEST_ENABLED_BY_DEFAULT OFF CACHE BOOL "SDL tests enabled by default" FORCE)
set(SDL_TESTS OFF CACHE BOOL "Build SDL tests" FORCE)

# Agregar subdirectorios de extern
add_subdirectory(${CMAKE_SOURCE_DIR}/extern)

# Definir las bibliotecas externas principales
set(EXTERN_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/extern/glm
    ${CMAKE_SOURCE_DIR}/extern/glad/include
    ${CMAKE_SOURCE_DIR}/extern/chemfiles/include
    ${CMAKE_SOURCE_DIR}/extern/SDL/include
    ${CMAKE_SOURCE_DIR}/extern/imgui
    ${CMAKE_SOURCE_DIR}/extern/imgui/backends
    CACHE INTERNAL "External include directories"
)

# Bibliotecas comunes para versiones gráficas
set(COMMON_GRAPHICS_LIBS
    glad
    glm
    SDL3-static
    imgui
    ${OPENGL_LIBRARIES}
    CACHE INTERNAL "Common graphics libraries"
)
