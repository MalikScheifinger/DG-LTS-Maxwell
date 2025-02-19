
add_executable(MassAssemblyBenchmark MassAssemblyBenchmark.cpp)
target_link_libraries(MassAssemblyBenchmark
        benchmark::benchmark
        data
        assembler_3D
        tools)
DEAL_II_SETUP_TARGET(MassAssemblyBenchmark)

add_executable(CurlAssemblyBenchmark CurlAssemblyBenchmark.cpp)
target_link_libraries(CurlAssemblyBenchmark
        benchmark::benchmark
        data
        assembler_3D
        tools)
DEAL_II_SETUP_TARGET(CurlAssemblyBenchmark)

