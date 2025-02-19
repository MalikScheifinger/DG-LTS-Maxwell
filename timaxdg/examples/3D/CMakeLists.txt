add_subdirectory(Polynomial)

add_executable(Minimal_3D Minimal.cpp)
target_link_libraries(Minimal_3D
        data
        data_types
        tools
        time_integration
        assembler_3D)
DEAL_II_SETUP_TARGET(Minimal_3D)