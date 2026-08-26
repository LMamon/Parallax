#include <parallax/core/pipeline.hpp>
#include <parallax/cuda/depth.cuh>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

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

        if(!vpi_stream_.initialize(VPI_BACKEND_CUDA | VPI_BACKEND_VIC | VPI_BACKEND_OFA)) {
            std::cerr << "Pipeline: failed to initialize VPI stream\n";
            shutdown();
            return false;
        }

        if (!rectifier_.initialize(calibration_, isp_.rgb(), isp_.gray(), vpi_stream_.handle())) {
            std::cerr << "Pipeline: failed to initialize stereo rectifier\n";
            shutdown();
            return false;
        }

        if (!charuco_pose_.initialize(rectifier_.gray().width, 
                                      rectifier_.gray().height, 
                                      calibration_)) {
                                        
            std::cerr << "Pipeline: failed to initialize ChArUco pose\n";
            shutdown();
            return false;
        }

        if (!matcher_.initialize(rectifier_.gray(), vpi_stream_.handle())) {
            std::cerr << "Pipeline: failed to initialize stereo matcher\n";
            shutdown();
            return false;
        }

        depth_.width = matcher_.output().width;
        depth_.height = matcher_.output().height;

        if (!depth_.depth.allocate(depth_.width, depth_.height, 1, sizeof(float))) {
            std::cerr << "Pipeline: failed to allocate depth buffer\n";
            shutdown();
            return false;
        }

        fps_window_start_ = std::chrono::steady_clock::now();
        fps_frame_count_ = 0;
        pipeline_fps_ = 0.0;

        sequence_ = 0;
        initialized_ = true;

        return true;
    }

    bool Pipeline::synchronize() {
        if (!initialized_) return false;
        
        if (!vpi_stream_.synchronize()) {
            std::cerr << "Pipeline: VPI synchronization failed\n";
            return false;
        }
        return true;
    }

    bool Pipeline::process(const parallax::camera::RawFrame& input, SensorFrame& output) {
        if (!initialized_) return false;
        using Clock = std::chrono::steady_clock;
        auto ms = [](auto start, auto end) {
            return std::chrono::duration<double, std::milli>(end - start).count();
        };

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
        const auto t_after_isp_sync = Clock::now();

        if (!rectifier_.process(vpi_stream_.handle())) {
            std::cerr << "Pipeline: stereo rectification failed\n";
            return false;
        }

        if (!matcher_.process(vpi_stream_.handle())) {
            std::cerr << "Pipeline: stereo matching failed\n";
            return false;
        }

        if (!parallax::cuda::disparityToDepth(matcher_.output().disparity,
                                            depth_.depth,
                                            static_cast<float>(calibration_.metadata().virtual_fx),
                                            static_cast<float>(calibration_.metadata().baseline_mm / 1000.0),
                                            parallax::isp::StereoMatchFrame::DisparityScale,
                                            vpi_stream_.cudaHandle())) {

            std::cerr << "Pipeline: depth conversion failed\n";
            return false;
        }

        if (!vpi_stream_.synchronize()) {
            std::cerr << "Pipeline: sync after depth failed\n";
            return false;
        }
        const auto t_depth_done = Clock::now();

        ++timing_frames;
        // if (!vpi_stream_.synchronize()) {
        //     std::cerr << "Pipeline: VPI synchronization failed before pose\n";
        //     return false;
        // }

        if (!charuco_pose_.process(rectifier_.gray(), output.pose)) {
            std::cerr << "Pipeline: ChArUco processing failed\n";
            return false;
        }

        output.pose.depth_valid = false;
        output.pose.depth_m = 0.0f;

        if (output.pose.pose_valid && output.pose.plane_valid) {
            const int x = static_cast<int>(std::lround(output.pose.projected_center.x));
            const int y = static_cast<int>(std::lround(output.pose.projected_center.y));

            if (x >= 0 &&
                x < static_cast<int>(depth_.width) &&
                y >= 0 &&
                y < static_cast<int>(depth_.height)) {

                float depth_m = 0.0f;

                const auto* row = reinterpret_cast<const std::uint8_t*>(depth_.depth.data()) +
                                  static_cast<std::size_t>(y) * depth_.depth.pitch();

                const float* pixel = reinterpret_cast<const float*>(row) + x;

                if (cudaMemcpy(&depth_m,
                               pixel,
                               sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess) {

                    if (std::isfinite(depth_m) && depth_m > 0.0f) {
                        output.pose.depth_m = depth_m;
                        output.pose.depth_valid = true;
                    }
                }
            }
        }

        // !!!Do not sync VPI stream here
        // sensitive area
        output.rgb = &rectifier_.rgb();
        output.stereo = &matcher_.output();
        output.depth = &depth_;

        output.metadata.observation.source = parallax::core::SourceId::StereoCamera;
        output.metadata.observation.sequence = sequence_++;
        output.metadata.timestamp = std::chrono::steady_clock::now();
        output.metadata.valid = true;

        ++fps_frame_count_;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - fps_window_start_).count();

        if (elapsed >= 1.0) {
            pipeline_fps_ = static_cast<double>(fps_frame_count_) / elapsed;

            std::cout << std::fixed << std::setprecision(1)
                      << "Pipeline: " << pipeline_fps_ << " FPS | board="
                      << (output.pose.board_detected ? "yes" : "no")
                      << " | pose="
                      << (output.pose.pose_valid ? "yes" : "no")
                      << " | depth=";

            if (output.pose.depth_valid) {
                std::cout << std::setprecision(2)
                          << output.pose.depth_m << " m";
            } else {
                std::cout << "--";
            }

            std::cout << " | plane="
                      << (output.pose.plane_valid ? "yes" : "no")
                      << '\n';
            
            fps_frame_count_ = 0;
            fps_window_start_ = now;

            timing_frames = 0;
        }
        return true;
    }

    void Pipeline::shutdown() {
        if (initialized_) {
            if (!vpi_stream_.synchronize()) {
                std::cerr << "Pipeline: failed to synchronize VPI stream during shutdown\n";
            }
        }
        
        depth_.depth.release();
        depth_.width = 0;
        depth_.height = 0;
        charuco_pose_.shutdown();
        matcher_.shutdown();
        rectifier_.shutdown();
        vpi_stream_.shutdown();
        isp_.shutdown();

        sequence_ = 0;
        initialized_ = false;
    }
}