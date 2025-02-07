set(raster_includes
    ${CMAKE_SOURCE_DIR}/include/algorithms/frustum_cull.h
    ${CMAKE_SOURCE_DIR}/include/algorithms/occlusion_cull.h

    ${CMAKE_SOURCE_DIR}/include/geometry/cylinder/cylinder.h
    ${CMAKE_SOURCE_DIR}/include/geometry/sphere/sphere.h

    ${CMAKE_SOURCE_DIR}/include/utils/math_defines.h
    ${CMAKE_SOURCE_DIR}/include/utils/scene_descriptor.h

    ${CMAKE_SOURCE_DIR}/include/ux/camera_controller.h
    ${CMAKE_SOURCE_DIR}/include/ux/camera.h
    ${CMAKE_SOURCE_DIR}/include/ux/input.h

    ${CMAKE_SOURCE_DIR}/include/vis/shader_program.h
    ${CMAKE_SOURCE_DIR}/include/vis/window.h
)

set(raster_source
    ${CMAKE_SOURCE_DIR}/src/algorithms/frustum_cull.cpp
    ${CMAKE_SOURCE_DIR}/src/algorithms/occlusion_cull.cpp

    ${CMAKE_SOURCE_DIR}/src/geometry/cylinder/cylinder.cpp
    ${CMAKE_SOURCE_DIR}/src/geometry/sphere/sphere.cpp

    ${CMAKE_SOURCE_DIR}/src/utils/scene_descriptor.cpp

    ${CMAKE_SOURCE_DIR}/src/ux/camera_controller.cpp
    ${CMAKE_SOURCE_DIR}/src/ux/camera.cpp
    ${CMAKE_SOURCE_DIR}/src/ux/input.cpp

    ${CMAKE_SOURCE_DIR}/src/vis/shader_program.cpp
    ${CMAKE_SOURCE_DIR}/src/vis/window.cpp
)

add_library(raster_dependencies STATIC
    ${raster_includes} ${raster_source} ${raster_extern_includes}
)

set_property(TARGET raster_dependencies PROPERTY CXX_STANDARD 20)

target_include_directories(raster_dependencies PUBLIC 
    ${OPENGL_INCLUDE_DIRS} 
    "${CMAKE_SOURCE_DIR}/extern/SDL/include" 
    "${CMAKE_SOURCE_DIR}/include" 
    ${raster_extern_includes}
)

target_link_libraries(raster_dependencies PUBLIC 
    ${SDL3_LIBRARIES} 
    glad 
    glm 
    chemfiles
)
