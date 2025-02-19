add_subdirectory(Polynomial)
add_subdirectory(LocallyImplicit)
add_subdirectory(LFC)

add_executable(Minimal_TE Minimal.cpp)
target_link_libraries(Minimal_TE
        data
        data_types
        tools
        time_integration
        assembler_TE)
DEAL_II_SETUP_TARGET(Minimal_TE)