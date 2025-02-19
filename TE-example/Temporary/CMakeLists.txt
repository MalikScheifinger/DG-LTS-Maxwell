
add_library(tmp_data_types
        VectorMassOperator.cpp)
target_include_directories(tmp_data_types PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tmp_data_types)

DEAL_II_SETUP_TARGET(tmp_data_types)