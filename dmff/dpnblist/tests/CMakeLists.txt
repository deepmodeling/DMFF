set(TEST_EXE dpnblist_test)

add_executable(${TEST_EXE}
    main.cpp
    test_box.cpp
    read_lmp.cpp
    test_cellSchAlgCPU.cpp
    test_cellSchAlgGPU.cpp
    test_octreeSchAlgCPU.cpp
    test_octreeSchAlgGPU.cpp
    #test_hashSchAlgCPU.cpp
    #test_hashSchAlgGPU.cpp
)

find_package(doctest QUIET)
if (NOT doctest_FOUND)
    set(DOCTEST_DIR ${CMAKE_SOURCE_DIR}/external/doctest-2.4.11)
    # add_subdirectory(${CMAKE_SOURCE_DIR}/external/doctest-2.4.11)
    include(${CMAKE_SOURCE_DIR}/external/doctest-2.4.11/scripts/cmake/doctest.cmake)
endif()

target_include_directories(${TEST_EXE} PRIVATE
    ${DOCTEST_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/SchAlg
)

# Link libraries
if (TARGET doctest::doctest)
    target_link_libraries(${TEST_EXE} PRIVATE doctest::doctest dpnblist)
else()
    target_link_libraries(${TEST_EXE} PRIVATE dpnblist)
endif()

# Discover tests if Doctest was not found and manually included
if (NOT doctest_FOUND)
    doctest_discover_tests(${TEST_EXE})
endif()
