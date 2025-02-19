add_executable(convg_cavity_1D convergence_cavity_1D.cpp)
target_link_libraries(convg_cavity_1D
        data
        data_1D
        data_types
        tools
        time_integration
        assembler_1D)
DEAL_II_SETUP_TARGET(convg_cavity_1D)