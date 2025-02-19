add_library(assembler_1D STATIC
    Assembler1D.cpp
    Assembler1DLocal.cpp)

target_include_directories(assembler_1D PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include)

DEAL_II_SETUP_TARGET(assembler_1D)
target_link_libraries(assembler_1D data tools)