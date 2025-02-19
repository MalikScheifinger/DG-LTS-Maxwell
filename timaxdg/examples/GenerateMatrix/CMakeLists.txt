
add_executable(GenerateMatrix GenerateMatrix.cpp)
target_link_libraries(GenerateMatrix
        data
        tools
        assembler_3D
        assembler_TE)
DEAL_II_SETUP_TARGET(GenerateMatrix)