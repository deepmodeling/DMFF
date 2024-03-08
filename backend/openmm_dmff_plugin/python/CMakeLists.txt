set(WRAP_FILE OpenMMDMFFPluginWrapper.cpp)
set(MODULE_NAME OpenMMDMFFPlugin)

# Execute SWIG to generate source code for the Python module.
add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "-I{DMFF_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDMFFPlugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDMFFPlugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Set up the git version name
# Get the latest abbreviated commit hash of the working branch
execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
execute_process(
    COMMAND git describe --tags --abbrev=0
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

# Compile the Python module.
add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(DMFFPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(DMFFPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDMFFPlugin DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/OpenMMDMFFPlugin/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/OpenMMDMFFPlugin/__init__.py)
add_custom_command(TARGET PythonInstall
    COMMAND "${PYTHON_EXECUTABLE}" setup.py build
    COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/OpenMMDMFFPlugin.py ${CMAKE_CURRENT_BINARY_DIR}/OpenMMDMFFPlugin
    COMMAND "${PYTHON_EXECUTABLE}" setup.py install
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
