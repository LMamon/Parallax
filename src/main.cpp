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
    const std::filesystem::path sensor_extrinsics_path = "config/sensors/extrinsics.yaml";
    const std::filesystem::path nanoowl_engine_path = "models/owl_image_encoder_patch32_fp16.engine";
    const std::filesystem::path calibration_directory = "config/camera/calibration/results/rectification";

    if (!runtime.initialize(camera_config_path,
                            sensor_extrinsics_path, 
                            calibration_directory, 
                            nanoowl_engine_path)) {

        return EXIT_FAILURE;
    }

    runtime.run(stop_requested);

    return EXIT_SUCCESS;
}