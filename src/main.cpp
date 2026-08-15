#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>
#include <parallax/camera/stereo_camera.hpp>
#include <parallax/camera/arducam_controls.hpp>
#include <parallax/stereo/calibration.hpp>
#include <parallax/stereo/rectification.hpp>
#include <parallax/stereo/matcher.hpp>

#include <parallax/isp/isp.hpp>

#include <parallax/vpi/stream.hpp>

#include <cstring>
#include <vector>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <filesystem>

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

    parallax::isp::ISP isp;

    if (!isp.initialize(config)) {
        std::cerr << "Failed to initialize ISP\n";
        return EXIT_FAILURE;
    }

    parallax::stereo::StereoCalibration calibration;

    if (!calibration.load("config/camera/calibration/results/rectification")) {
        std::cerr << "Failed to load stereo calibration\n";
        return EXIT_FAILURE;
    }

    parallax::vpi::Stream vpi_stream;

    if (!vpi_stream.initialize(VPI_BACKEND_CUDA)) {
        std::cerr << "Failed to initialize VPI stream\n";
        return EXIT_FAILURE;
    }

    parallax::stereo::StereoRectifier rectifier;

    if (!rectifier.initialize(calibration, isp.rgb(), isp.gray(), vpi_stream.handle())) {
        std::cerr << "Failed to initialize stereo rectifier\n";
        return EXIT_FAILURE;   
    }

    parallax::stereo::StereoMatcher matcher;

    if (!matcher.initialize(rectifier.gray(), vpi_stream.handle())) {
        std::cerr << "Failed to intialize stereo matcher\n";
        return EXIT_FAILURE;
    }

    parallax::camera::RawFrame raw_frame{};
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

        if (!isp.process(raw_frame)) {
            std::cerr << "ISP processing failed\n";
            camera.release(raw_frame);
            break;
        }

        if (!isp.synchronize()) {
            std::cerr << "ISP synchronization failed\n";
            camera.release(raw_frame);
            break;
        }
        
        if (!camera.release(raw_frame)) {
            std::cerr << "Failed to release camera frame\n";
            break;
        }

        if (!rectifier.process()) {
            std::cerr << "Stereo rectification failed\n";
            break;
        }

        if (!matcher.process()) {
            std::cerr << "Stereo matching failed\n";
            break;
        }
        // Remove this when the live pipeline is ready.
        break;
    }


    if (!vpi_stream.synchronize()) {
        std::cerr << "Failed to synchronize VPI stream during shutdown\n";
    }

    matcher.shutdown();
    rectifier.shutdown();
    vpi_stream.shutdown();
    isp.shutdown();
    camera.shutdown();

    return EXIT_SUCCESS;
}