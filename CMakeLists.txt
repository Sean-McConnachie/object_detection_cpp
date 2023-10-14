#add_compile_options(-O3)

cmake_minimum_required(VERSION 3.26)
project(object_detection_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(object_detection_cpp main.cpp
        feature.cpp
        feature.h
        constants.h
        image.cpp
        image.h
        utils.cpp
        utils.h)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
