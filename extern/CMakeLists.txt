add_subdirectory(glm)
add_subdirectory(chemfiles)
# add_subdirectory(imgui)

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


set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

set(imgui_source
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp

    ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer3.cpp
)

set(imgui_includes
    ${IMGUI_DIR}/imgui.h
    ${IMGUI_DIR}/imconfig.h
    ${IMGUI_DIR}/imgui_internal.h
    ${IMGUI_DIR}/imstb_rectpack.h
    ${IMGUI_DIR}/imstb_textedit.h
    ${IMGUI_DIR}/imstb_truetype.h
    ${IMGUI_DIR}/misc/cpp/imgui_stdlib.h

    ${IMGUI_DIR}/backends/imgui_impl_sdl3.h
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.h
    ${IMGUI_DIR}/backends/imgui_impl_sdlrenderer3.h
)

add_library(imgui STATIC
    ${imgui_source}
    ${imgui_includes}
)
target_link_libraries(imgui PUBLIC 
${SDL3_LIBRARIES} 
SDL3-static
)

# Incluir directorios de ImGui
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${OPENGL_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/SDL/include
)

set(raster_extern_includes
    ${CMAKE_CURRENT_SOURCE_DIR}/glm
    ${CMAKE_CURRENT_SOURCE_DIR}/glad
    ${CMAKE_CURRENT_SOURCE_DIR}/chemfiles/include
    ${CMAKE_CURRENT_SOURCE_DIR}/SDL/include
    ${IMGUI_DIR}
)

set(raster_extern_libraries
    glm::glm
    chemfiles
    imgui
)