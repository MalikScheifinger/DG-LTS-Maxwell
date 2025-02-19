
add_library(solution_TE STATIC
        SolutionTE.cpp)

target_include_directories(solution_TE PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include)

DEAL_II_SETUP_TARGET(solution_TE)