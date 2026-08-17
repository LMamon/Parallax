#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>
#include <parallax/camera/stereo_camera.hpp>
#include <parallax/core/pipeline.hpp>
#include <parallax/core/sensor_frame.hpp>

#include <csignal>
#include <cstdlib>
#include <iostream>

namespace {

    volatile std::sig_atomic_t running = 1;
    void signalHandler(int) {
        running = 0;
    }
} 

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    parallax::camera::CameraConfig config;

    if (!config.loadFromFile("config/camera/stereo.yaml")) {
        std::cerr << "Failed to load camera configuration\n";
        return EXIT_FAILURE;
    }

    parallax::camera::StereoCamera camera(config);

    if (!camera.initialize()) {
        std::cerr << "Failed to initialize camera\n";
        return EXIT_FAILURE;
    }

    parallax::core::Pipeline pipeline;
    
    if (!pipeline.initialize(config, "config/camera/calibration/results/rectification")) {
        std::cerr << "Failed to initialize processing pipeline\n";
        camera.shutdown();
        return EXIT_FAILURE;
    }
    
    parallax::camera::RawFrame raw_frame{};
    parallax::core::SensorFrame sensor_frame{};
    int failed_frames = 0;

    while (running) {
        if (!camera.capture(raw_frame)) {
            if (++failed_frames >= 10) {
                std::cerr << "Camera failed to produce a valid frame\n";
                break;
            }
            continue;
        }
        failed_frames = 0;

        if (!pipeline.process(raw_frame, sensor_frame)) {
            std::cerr << "Pipeline processing failed\n";
            camera.release(raw_frame);
            break;
        }
        
        if (!camera.release(raw_frame)) {
            std::cerr << "Failed to release camera frame\n";
            break;
        }

        // Future Runtime::dispatch() will consume frame here.

        
    }

    // pipeline.shutdown();
    // camera.shutdown();

    return EXIT_SUCCESS;
}