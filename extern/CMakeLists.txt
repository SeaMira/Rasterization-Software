add_subdirectory(glm)
add_subdirectory(chemfiles)

if(NOT TARGET SDL2::SDL2)
    # Defines SDL3 as a static library
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)   # Deactivates dynamic version
    set(SDL_STATIC ON CACHE BOOL "" FORCE)    # Activates static version
    
    set(SDL2_DISABLE_UNINSTALL      ON  CACHE BOOL "" FORCE) # Name clash between SDL2 and chemfiles
    set(SDL_TEST                    OFF CACHE BOOL "" FORCE)
    set(SDL_TEST_ENABLED_BY_DEFAULT OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS                   OFF CACHE BOOL "" FORCE)
    add_subdirectory(SDL)
endif()

if(NOT TARGET glad)
    add_subdirectory(glad)
endif()

set(raster_extern_includes
    ${CMAKE_CURRENT_SOURCE_DIR}/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/glad
    ${CMAKE_CURRENT_SOURCE_DIR}/chemfiles/include
    ${CMAKE_CURRENT_SOURCE_DIR}/SDL/include
)

set(raster_extern_libraries
    glm::glm
    chemfiles
)