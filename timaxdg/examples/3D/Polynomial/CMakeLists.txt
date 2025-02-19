add_executable(Polynomial_3D Polynomial.cpp)
target_link_libraries(Polynomial_3D
        data
        data_types
        tools
        time_integration
        assembler_3D)
DEAL_II_SETUP_TARGET(Polynomial_3D)