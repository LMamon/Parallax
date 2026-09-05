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

namespace parallax::perception {

    class StereoRoiAssociator {
        public:
            static constexpr std::size_t MaxObjects = 64;
            static constexpr std::uint32_t RoiRadius = 3;
            static constexpr std::uint32_t MinValidSamples = 5;

            explicit StereoRoiAssociator(const stereo::StereoCalibration& calibration);

            bool initialize();

            bool associate(const DetectionSet& detections,
                           const core::ProductMetadata& semantic_metadata,
                           const core::Product<isp::DepthFrame>& depth,
                           core::ExecutionContext& context,
                           Object3DSet& output);

        private:
            [[nodiscard]] bool backProject(const cv::Point2f& rectified_point,
                                           float depth_m,
                                           std::array<float, 3>& xyz) const noexcept;

            const stereo::StereoCalibration& calibration_;

            ImageSpaceMapper mapper_;

            cuda::CudaBuffer requests_device_;
            cuda::CudaBuffer results_device_;

            std::array<cuda::DepthRoiRequest, MaxObjects> requests_host_{};
            std::array<cuda::DepthRoiResult, MaxObjects> results_host_{};

            bool initialized_ = false;
    };
}