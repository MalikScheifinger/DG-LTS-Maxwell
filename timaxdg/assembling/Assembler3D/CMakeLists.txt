add_library(assembler_3D STATIC
    Assembler3D.cpp
    Assembler3DLocal.cpp)

target_include_directories(assembler_3D PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include)

DEAL_II_SETUP_TARGET(assembler_3D)
target_link_libraries(assembler_3D data)