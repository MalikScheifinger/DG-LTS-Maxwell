
add_executable(Minimal_Locally_Implicit MinimalLocallyImplicit.cpp)
target_link_libraries(Minimal_Locally_Implicit
        data
        data_types
        tools
        time_integration
        assembler_TE)
DEAL_II_SETUP_TARGET(Minimal_Locally_Implicit)