CMAKE_MINIMUM_REQUIRED( VERSION 3.15)

Project( mcflow LANGUAGES CXX)

#SET(CMAKE_VERBOSE_MAKEFILE ON)

#default build type is Release
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

SET( PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR} )
SET( CMAKE_INSTALL_PREFIX ${PROJECT_ROOT} )
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)


include(FetchContent)
FetchContent_Declare(openmesh
GIT_REPOSITORY https://www.graphics.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
GIT_TAG        OpenMesh-8.1
)
FetchContent_MakeAvailable(openmesh)

SUBDIRS( src )


