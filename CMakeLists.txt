cmake_minimum_required(VERSION 3.18)
project(ClmapPicking)
set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0072 NEW)

# Open3D
list(APPEND CMAKE_PREFIX_PATH "C:/open3d/open3d-devel-windows-amd64-0.19.0")

# Zivid
set(Zivid_DIR "C:/Program Files/Zivid/CMake/Zivid")

find_package(Open3D REQUIRED)
find_package(Zivid 2.13.0 COMPONENTS Core REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# include the src folder
add_subdirectory(src)

add_executable(my_app main.cpp)

# Fix: prevent Windows headers from defining min/max macros
target_compile_definitions(my_app PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

target_link_libraries(my_app PRIVATE
    my_modules
    Open3D::Open3D
    Zivid::Core
    ${OpenCV_LIBS}
)





