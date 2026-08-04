#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/stereo_camera.hpp>

#include <csignal>
#include <cstdlib>
#include <iostream>

using namespace parallax::camera;

namespace {
    volatile std::sig_atomic_t running = 1;

    void signalHandler(int) {
        running = 0;
    }
}

    int main() {
        std::signal(SIGINT, signalHandler);

        CameraConfig config;

        if (!config.loadFromFile("config/camera.yaml")) return EXIT_FAILURE;

        StereoCamera camera(config);

        if (!camera.initialize()) return EXIT_FAILURE;

        RawFrame frame{};

        while (running) {

            if (!camera.capture(frame)) continue;

            //
            // ISP
            //
            // stereo
            //
            // tracking
            //
            // visualization
            //

            if (!camera.release(frame)) {
                std::cerr << "Failed to release frame\n";
                break;
            }
        }

        camera.shutdown();

        return EXIT_SUCCESS;
}