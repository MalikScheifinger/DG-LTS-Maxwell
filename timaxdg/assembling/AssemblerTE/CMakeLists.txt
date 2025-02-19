add_library(assembler_TE STATIC
    AssemblerTE.cpp
    AssemblerTELocal.cpp)

target_include_directories(assembler_TE PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include)

DEAL_II_SETUP_TARGET(assembler_TE)
target_link_libraries(assembler_TE data tools)