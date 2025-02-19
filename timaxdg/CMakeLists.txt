cmake_minimum_required(VERSION 3.0)
project(DGMaxwell)

### needed for gcc to compile with c++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


### SETUP DEALII ###
FIND_PACKAGE(deal.II 9.3.0 QUIET
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

### Add library components ###
add_subdirectory(assembling)
add_subdirectory(data)
add_subdirectory(data_types)
add_subdirectory(tools)
add_subdirectory(time_integration)

### Add examples ###
add_subdirectory(examples)

### Setup doxygen ###
option(BUILD_DOC "Build documentation" ON)

find_package(Doxygen)
if (DOXYGEN_FOUND)

    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
