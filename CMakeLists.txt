add_compile_options(-O3)

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
        utils.h
        learner.cpp
        learner.h
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
