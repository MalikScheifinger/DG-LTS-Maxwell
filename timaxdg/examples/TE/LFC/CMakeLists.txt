add_executable(LFC_TE LFC.cpp)
target_link_libraries(LFC_TE
        data
        data_types
        tools
        time_integration
        assembler_TE)
DEAL_II_SETUP_TARGET(LFC_TE)