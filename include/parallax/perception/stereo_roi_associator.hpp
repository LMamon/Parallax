#pragma once

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/cuda/depth_roi.cuh>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/image_space_mapper.hpp>
#include <parallax/perception/object3d.hpp>
#include <parallax/stereo/calibration.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace parallax::perception {

    struct RectifiedCameraModel {
        float fx_px = 0.0F;
        float fy_px = 0.0F;
        float cx_px = 0.0F;
        float cy_px = 0.0F;

        std::string coordinate_frame;

        [[nodiscard]] bool valid() const noexcept;
    };

    class StereoRoiAssociator {
        public:
            static constexpr std::size_t MaxObjects = 64;
            static constexpr std::uint32_t RoiRadius = 3;
            static constexpr std::uint32_t MinValidSamples = 5;

            explicit StereoRoiAssociator(const stereo::StereoCalibration& calibration,
                                         std::string coordinate_frame = "camera_left_optical");

            // Production initialization uses the exact calibration maps and P1
            // consumed by the rectified stereo/depth path.
            bool initialize();

            // Explicit initialization keeps geometry unit-testable without
            // requiring calibration files or hardware-specific fixtures.
            bool initialize(std::uint32_t width,
                            std::uint32_t height,
                            const std::vector<float>& rectified_to_rgb_x,
                            const std::vector<float>& rectified_to_rgb_y,
                            RectifiedCameraModel camera_model);

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

            bool associate(const DetectionSet& detections,
                           const core::ProductMetadata& semantic_metadata,
                           const core::Product<isp::DepthFrame>& depth,
                           core::ExecutionContext& context,
                           Object3DSet& output);

        private:
            [[nodiscard]] bool backProject(
                const cv::Point2f& rectified_point,
                float depth_m,
                std::array<float, 3>& xyz) const noexcept;

            const stereo::StereoCalibration& calibration_;
            std::string coordinate_frame_;

            RectifiedCameraModel camera_model_{};
            ImageSpaceMapper mapper_;

            // Fixed scratch capacity prevents per-frame CUDA allocation.
            cuda::CudaBuffer requests_device_;
            cuda::CudaBuffer results_device_;

            std::array<cuda::DepthRoiRequest, MaxObjects> requests_host_{};
            std::array<cuda::DepthRoiResult, MaxObjects> results_host_{};

            std::uint32_t image_width_ = 0;
            std::uint32_t image_height_ = 0;

            bool initialized_ = false;
    };
}