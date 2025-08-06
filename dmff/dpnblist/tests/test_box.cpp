// #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "../src/box.h" // Assuming Box.h is your header file
#include <iostream>

TEST_CASE("Test Box construction and getters") {
    std::vector<float> init_lengths = {0.0f, 0.0f, 0.0f};
    dpnblist::Box box(init_lengths);

    // Check default values
    CHECK(box.get_lengths_cpu() == std::vector<float>{0.0f, 0.0f, 0.0f});
    CHECK(box.get_angles_cpu() == std::vector<float>{90.0f, 90.0f, 90.0f});

    // Test setting lengths and angles
    std::vector<float> lengths = {10.0f, 20.0f, 30.0f};
    std::vector<float> angles = {60.0f, 70.0f, 80.0f};
    box.set_lengths_and_angles(lengths, angles);
    CHECK(box.get_lengths_cpu() == std::vector<float>{10.0f, 20.0f, 30.0f});
    CHECK(box.get_angles_cpu() == std::vector<float>{60.0f, 70.0f, 80.0f});

    // Test setting periodicity
    box.set_periodic(true, false, true);
    CHECK(box.get_periodic_cpu() == std::vector<bool>{true, false, true});
}

TEST_CASE("Test wrapping methods") {
    std::vector<float> lengths = {10.0f, 10.0f, 10.0f};
    dpnblist::Box box(lengths);

    // Test wrapping with vector<float>
    std::vector<float> position1 = {15.0f, 15.0f, 15.0f};
    box.wrap(position1);
    CHECK(position1 == std::vector<float>{5.0f, 5.0f, 5.0f});

    // Test wrapping with array<float, 3>
    std::array<float, 3> position2 = {25.0f, 25.0f, 25.0f};
    box.wrap(position2);
    CHECK(position2 == std::array<float, 3>{5.0f, 5.0f, 5.0f});

    // Test wrapping with vec3_float
    dpnblist::vec3_float position3 = {35.0f, 35.0f, 35.0f};
    box.wrap(position3);
    CHECK(position3.x == doctest::Approx(5.0f));
    CHECK(position3.y == doctest::Approx(5.0f));
    CHECK(position3.z == doctest::Approx(5.0f));
}

TEST_CASE("Test distance calculation methods") {
    std::vector<float> lengths = {10.0f, 10.0f, 10.0f};
    dpnblist::Box box(lengths);

    // Test calc_sqrt_distance with vector<float>
    std::vector<float> point1 = {0.0f, 0.0f, 0.0f};
    std::vector<float> point2 = {3.0f, 4.0f, 0.0f};
    CHECK(box.calc_sqrt_distance(point1, point2) == doctest::Approx(25.0f));

    // Test calc_sqrt_distance with vec3_float
    dpnblist::vec3_float r1 = {0.0f, 0.0f, 0.0f};
    dpnblist::vec3_float r2 = {3.0f, 4.0f, 0.0f};
    CHECK(box.calc_sqrt_distance(r1, r2) == doctest::Approx(25.0f));

    // Test calc_distance2 with array<float, 3>
    std::array<float, 3> r3 = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> r4 = {3.0f, 4.0f, 0.0f};
    CHECK(box.calc_distance2(r3, r4) == doctest::Approx(25.0f));
}
