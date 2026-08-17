#include <parallax/core/pipeline.hpp>

#include <iostream>

namespace parallax::core {
    Pipeline::~Pipeline() { shutdown(); }

    bool Pipeline::initialize(const parallax::camera::CameraConfig& config, const std::filesystem::path& calibration) {
        if (initialized_) return true;
        if (!isp_.initialize(config)) {
            std::cerr << "Pipeline: failed to initialize ISP\n";
            shutdown();
            return false;
        }

        if (!calibration_.load(calibration)) {
            std::cerr << "Pipeline: failed to load stereo calibration\n";
            shutdown();
            return false;
        }

        if(!vpi_stream_.initialize(VPI_BACKEND_CUDA)) {
            std::cerr << "Pipeline: failed to initialize VPI stream\n";
            shutdown();
            return false;
        }

        if (!rectifier_.initialize(calibration_, isp_.rgb(), isp_.gray(), vpi_stream_.handle())) {
            std::cerr << "Pipeline: failed to initialize stereo rectifier\n";
            shutdown();
            return false;
        }

        if (!matcher_.initialize(rectifier_.gray(), vpi_stream_.handle())) {
            std::cerr << "Pipeline: failed to initialize stereo matcher\n";
            shutdown();
            return false;
        }

        sequence_ = 0;
        initialized_ = true;

        return true;
    }

    bool Pipeline::process(const parallax::camera::RawFrame& input, SensorFrame& output) {
        if (!initialized_) return false;
        if (!isp_.process(input)) {
            std::cerr << "Pipeline: ISP processing failed\n";
            return false;
        }
        // sensitive area 
        /*
        * ISP uses its own CUDA stream while the downstream rectifier/matcher 
        * share a VPI stream. Synchronize so that VPI cannot consume ISP
        * output before the CUDA work completes.
        *
        * !!!This is a synchronization boundary between the two execution
        * streams!!!
        */

        if (!isp_.synchronize()) {
            std::cerr << "Pipeline: ISP synchronization failed\n";
            return false;
        }

        if (!rectifier_.process()) {
            std::cerr << "Pipeline: stereo rectification failed\n";
            return false;
        }

        if (!matcher_.process()) {
            std::cerr << "Pipeline: stereo matching failed\n";
            return false;
        }
        // !!!Do not sync VPI stream here
        // sensitive area
        output.rgb = &rectifier_.rgb();
        output.stereo = &matcher_.output();

        output.sequence = sequence_++;
        output.timestamp = std::chrono::steady_clock::now();
        
        return true;
    }

    void Pipeline::shutdown() {
        if (initialized_) {
            if (!vpi_stream_.synchronize()) {
                std::cerr << "Pipeline: failed to synchronize VPI stream during shutdown\n";
            }
        }
        matcher_.shutdown();
        rectifier_.shutdown();
        vpi_stream_.shutdown();
        isp_.shutdown();

        sequence_ = 0;
        initialized_ = false;
    }

}