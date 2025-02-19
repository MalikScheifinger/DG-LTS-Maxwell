
add_library(tools STATIC NumberSpaces.cpp)

target_include_directories(tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
DEAL_II_SETUP_TARGET(tools)