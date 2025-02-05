cmake_minimum_required(VERSION 3.20)

project(Glad)

find_package(OpenGL REQUIRED)
add_library(glad src/glad.c)
target_link_libraries(glad PUBLIC OpenGL::GL)
target_include_directories(glad PUBLIC include/)