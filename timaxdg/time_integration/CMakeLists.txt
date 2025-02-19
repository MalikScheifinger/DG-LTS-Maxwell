
add_library(time_integration
        Leapfrog.h
        Leapfrog.hh
        CrankNicolson.h
        CrankNicolson.hh
        LocallyImplicit.h
        LocallyImplicit.hh
        TimeIntegrator.h
        LeapfrogChebychev.h
        LeapfrogChebychev.hh)
target_include_directories(time_integration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(time_integration data_types tmp_data_types)

DEAL_II_SETUP_TARGET(time_integration)