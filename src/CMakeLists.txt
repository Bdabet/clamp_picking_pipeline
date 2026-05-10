# bundle all modules into one library
add_library(my_modules
    segment_zdf_pcd_file.cpp
    scene_representation.cpp
)


target_include_directories(my_modules PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    "C:/Program Files/Zivid/include"        # add this
)

target_compile_definitions(my_modules PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

target_link_libraries(my_modules PUBLIC
    Open3D::Open3D
    Zivid::Core
    ${OpenCV_LIBS}
)
