#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <chrono>
#include "factorama/stopwatch.hpp"

using namespace factorama;

TEST_CASE("Stopwatch basic timing", "[stopwatch]")
{
    Stopwatch sw;

    sw.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    sw.stop();

    double elapsed = sw.elapsed();

    // Should be approximately 0.1 seconds, with some tolerance
    REQUIRE(elapsed >= 0.09);
    REQUIRE(elapsed <= 0.15);
}

TEST_CASE("Stopwatch accumulates time across multiple start/stop cycles", "[stopwatch]")
{
    Stopwatch sw;

    // First cycle
    sw.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sw.stop();

    // Second cycle
    sw.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sw.stop();

    double elapsed = sw.elapsed();

    // Should be approximately 0.1 seconds total
    REQUIRE(elapsed >= 0.09);
    REQUIRE(elapsed <= 0.15);
}

TEST_CASE("Stopwatch reset clears accumulated time", "[stopwatch]")
{
    Stopwatch sw;

    sw.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sw.stop();

    sw.reset();

    REQUIRE(sw.elapsed() == 0.0);

    // After reset, can start timing again
    sw.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sw.stop();

    double elapsed = sw.elapsed();
    REQUIRE(elapsed >= 0.04);
    REQUIRE(elapsed <= 0.08);
}

TEST_CASE("Stopwatch ignores redundant start/stop calls", "[stopwatch]")
{
    Stopwatch sw;

    // Multiple starts should only count from first start
    sw.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sw.start();  // Should do nothing
    sw.stop();

    double elapsed1 = sw.elapsed();

    // Multiple stops should not affect the time
    sw.stop();  // Should do nothing

    double elapsed2 = sw.elapsed();
    REQUIRE(elapsed1 == elapsed2);
}
