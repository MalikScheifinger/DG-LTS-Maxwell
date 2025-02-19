
add_library(data_types
        VectorMassMatrix.cpp)
target_include_directories(data_types PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(data_types)

DEAL_II_SETUP_TARGET(data_types)