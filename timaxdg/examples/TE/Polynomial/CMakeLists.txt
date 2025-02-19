add_executable(Polynomial_TE Polynomial.cpp)
target_link_libraries(Polynomial_TE
        data
        data_types
        tools
        time_integration
        assembler_TE)
DEAL_II_SETUP_TARGET(Polynomial_TE)