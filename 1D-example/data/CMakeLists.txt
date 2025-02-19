add_library(data_1D STATIC 
    Solutions1D.cpp)

target_include_directories(data_1D PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include)

DEAL_II_SETUP_TARGET(data_1D) 