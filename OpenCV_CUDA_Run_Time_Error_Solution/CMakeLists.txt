cmake_minimum_required(VERSION 3.0)

# Project name
project(HelloWorld)

# Find Opencv
find_package(OpenCV REQUIRED)

# Add executable
add_executable(main src/main.cpp)

# OpenCV Include path
target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})

# Link Libraries
target_link_libraries(main ${OpenCV_LIBS})