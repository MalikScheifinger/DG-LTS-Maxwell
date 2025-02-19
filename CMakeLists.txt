cmake_minimum_required(VERSION 3.0)
project(Maxwell-dG-LTS)

### needed for gcc to compile with c++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

### SETUP DEALII ###
FIND_PACKAGE(deal.II 9.5.0 QUIET
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        )
IF (NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
ENDIF ()
DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET(DEAL_II_WITH_TBB OFF)

# add dg-maxwell project
add_subdirectory(timaxdg/assembling)
add_subdirectory(timaxdg/data)
add_subdirectory(timaxdg/data_types)
add_subdirectory(timaxdg/tools)
add_subdirectory(timaxdg/time_integration)

### Add examples ###
add_subdirectory(1D-example)
add_subdirectory(TE-example)
