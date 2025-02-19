add_executable(LTSvsLI comparision_LTS_LI.cpp)
target_link_libraries(LTSvsLI
        data
        data_types
        tools
        time_integration
        tmp_data_types
        assembler_TE
        solution_TE)
DEAL_II_SETUP_TARGET(LTSvsLI)