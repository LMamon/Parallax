#include <parallax/core/runtime.hpp>

#include <csignal>
#include <cstdlib>

namespace {
    volatile std::sig_atomic_t stop_requested = 0;
    void signalHandler(int) { stop_requested = 1; }
} 

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    parallax::core::Runtime runtime;
    const std::filesystem::path camera_config_path = "config/camera/stereo.yaml";
    const std::filesystem::path calibration_directory = "config/camera/calibration/results/rectification";

    if (!runtime.initialize(camera_config_path, calibration_directory)) {
        return EXIT_FAILURE;
    }

    runtime.run(stop_requested);

    return EXIT_SUCCESS;
}