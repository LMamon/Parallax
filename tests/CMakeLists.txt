find_package(GTest REQUIRED)

add_executable(parallax_tests
    smoke_test.cpp)

add_executable(parallax_product_store_test
    product_store_test.cpp
)

target_link_libraries(parallax_tests
    PRIVATE
        parallax_core
        GTest::gtest_main
)

target_link_libraries(parallax_product_store_test
    PRIVATE
        parallax_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(parallax_tests)
gtest_discover_tests(parallax_product_store_test)