cmake_minimum_required(VERSION 3.1)
project(rand-util CXX)
set(CMAKE_CXX_STANDARD 11)

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/rand-util.hpp)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/rand-util.cpp)

add_library(rand-util STATIC ${headers} ${sources})
target_include_directories(rand-util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h*")
install(TARGETS rand-util ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
