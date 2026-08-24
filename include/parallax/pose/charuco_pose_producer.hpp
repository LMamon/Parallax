#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>

#include <parallax/pose/charuco_pose.hpp>
#include <parallax/stereo/calibration.hpp>

#include <vector>

namespace parallax::pose {
    /**
     * CharucoPose remains responsible for board detection and calibrated 6-DoF
     * pose estimation. This producer declares the image dependency and exposes
     * pose as a named graph product without coupling pose estimation to stereo
     * disparity or metric depth.
     *
     * StereoCalibration is immutable startup geometry. It remains an explicit
     * constructor dependency rather than a per-frame ProductStore entry.
     */
    class CharucoPoseProducer final : public parallax::core::Producer {
        public:
            CharucoPoseProducer(CharucoPose& pose,
                                const parallax::stereo::StereoCalibration& calibration,
                                parallax::core::ProductStore& store);

            [[nodiscard]] std::string_view name() const noexcept override;
            
            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;
        
            parallax::core::SubmitResult submit() override;

        private:
            CharucoPose& pose_;
            // camera geometry is fixed for pose estimator
            const parallax::stereo::StereoCalibration& calibration_;
            parallax::core::ProductStore& store_;

            // ChArUco pose consumes the calibrated rectified image path only.
            // Disparity and Depth are intentionally absent: board pose remains
            // usable without activating stereo matching
            const std::vector<parallax::core::ProductId> inputs_{
                parallax::core::ProductId::RectifiedGray
            };

            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::Pose
            };
    };
}
