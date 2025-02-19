
project(tests)

# 3D tests
add_executable(3DMatrixAssemblingTest assembling/3DMatrixAssemblingTest.cpp)
target_link_libraries(3DMatrixAssemblingTest
        gtest
        gtest_main
        data
        assembler_3D
        tools)
DEAL_II_SETUP_TARGET(3DMatrixAssemblingTest)

# 2D tests
add_executable(TEMatrixAssemblingTest assembling/TEMatrixAssemblingTest.cpp)
target_link_libraries(TEMatrixAssemblingTest
        gtest
        gtest_main
        data
        assembler_TE
        tools)
DEAL_II_SETUP_TARGET(TEMatrixAssemblingTest)

# copy test resources to build tree
file(COPY assembling/matrices DESTINATION .)