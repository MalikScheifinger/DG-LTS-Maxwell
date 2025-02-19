add_executable(convg_time_cavity_TE convergence_time_cavity.cpp)
target_link_libraries(convg_time_cavity_TE
        data
        data_types
        tools
        time_integration
        tmp_data_types
        assembler_TE
        solution_TE)
DEAL_II_SETUP_TARGET(convg_time_cavity_TE)