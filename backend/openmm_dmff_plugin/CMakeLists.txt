#---------------------------------------------------
# OpenMM Deepmd Plugin
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
SET(CMAKE_VERBOSE_MAKEFILE ON)

project(OpenMMDMFF)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")
INCLUDE_DIRECTORIES("${OPENMM_DIR}/include")
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")


# Include the cppflow
SET(CPPFLOW_DIR "/usr/local/cppflow/" CACHE PATH "Where cppflow is installed")
INCLUDE_DIRECTORIES("${CPPFLOW_DIR}/include")
LINK_DIRECTORIES("${CPPFLOW_DIR}/lib")

# Include the tensorflow
SET(TENSORFLOW_DIR "/usr/local/tensorflow/" CACHE PATH "Where libtensorflow is installed")
INCLUDE_DIRECTORIES("${TENSORFLOW_DIR}/include")
LINK_DIRECTORIES("${TENSORFLOW_DIR}/lib")

# Specify the C++ version we are building for.
SET (CMAKE_CXX_STANDARD 17)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wl,--allow-multiple-definition  -Wl,--no-as-needed")

# Set if high precision (double) is used. By default, it is off.
OPTION(USE_HIGH_PRECISION "Use high precision (double) for calculations" OFF)

IF(USE_HIGH_PRECISION)
    message(STATUS "USE_HIGH_PRECISION is enabled")
    ADD_DEFINITIONS(-DUSE_HIGH_PRECISION)
endif ()

if (USE_HIGH_PRECISION)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHIGH_PRECISION")
else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif ()

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX "${OPENMM_DIR}" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# Put all the tests and libraries in a single output directory.
IF(NOT EXECUTABLE_OUTPUT_PATH)
  SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}
      CACHE INTERNAL "Single output directory for building all executables.")
ENDIF()
IF(NOT LIBRARY_OUTPUT_PATH)
  SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}
      CACHE INTERNAL "Single output directory for building all libraries.")
ENDIF()
SET(${PROJECT_NAME}_EXECUTABLE_DIR ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})
SET(${PROJECT_NAME}_LIBRARY_DIR    ${LIBRARY_OUTPUT_PATH}/${CMAKE_CFG_INTDIR})

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(DMFF_PLUGIN_SOURCE_SUBDIRS openmmapi serialization)

# Set the library name
SET(DMFF_LIBRARY_NAME OpenMMDMFF)
SET(SHARED_DMFF_TARGET ${DMFF_LIBRARY_NAME})

# These are all the places to search for header files that are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${DMFF_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# Create the library.

ADD_LIBRARY(${SHARED_DMFF_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_DMFF_TARGET}
    PROPERTIES COMPILE_FLAGS "-DDMFF_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_DMFF_TARGET} OpenMM)
TARGET_LINK_LIBRARIES(${SHARED_DMFF_TARGET} tensorflow_cc)

INSTALL_TARGETS(/lib RUNTIME_DIRECTORY /lib ${SHARED_DMFF_TARGET})

# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
INSTALL (FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

# Enable testing
# Set no testing here.
ENABLE_TESTING()
ADD_SUBDIRECTORY(serialization/tests)

# Use reference platform default.
ADD_SUBDIRECTORY(platforms/reference)

FIND_PACKAGE(CUDA QUIET)
IF(CUDA_FOUND)
    message(STATUS "CUDA found, building CUDA implementation")
    SET(PLUGIN_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA: ON")
ELSE(CUDA_FOUND)
    message(STATUS "CUDA not found, not building CUDA implementation")
    SET(PLUGIN_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA: OFF")
ENDIF(CUDA_FOUND)
IF(PLUGIN_BUILD_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(PLUGIN_BUILD_CUDA_LIB)


# Build the Python API
FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
IF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    message(STATUS "Python and SWIG found, building Python wrappers")
    SET(PLUGIN_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    message(FATAL_ERROR "Python or SWIG not found, not building Python wrappers")
    SET(PLUGIN_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
IF(PLUGIN_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(PLUGIN_BUILD_PYTHON_WRAPPERS)

