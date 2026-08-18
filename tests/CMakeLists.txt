find_package(GTest REQUIRED)

add_executable(parallax_tests
    smoke_test.cpp)

target_link_libraries(parallax_tests
    PRIVATE
        GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(parallax_tests)