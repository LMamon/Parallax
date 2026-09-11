#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include <cuvslam/cuvslam2.h>

#include <parallax/core/sensor_extrinsics.hpp>
#include <parallax/stereo/calibration.hpp>

namespace parallax::localization {

    /**
     * Non-owning view of one synchronized rectified stereo observation.
     *
     * The producer/runtime remains responsible for buffer lifetime and
     * accelerator readiness. CuVslamLocalizer only borrows these pointers for
     * the duration of the synchronous cuVSLAM Track() call.
     */
    struct CuVslamFrame {
        const void* left = nullptr;
        const void* right = nullptr;

        std::int32_t width = 0;
        std::int32_t height = 0;

        // Pitch is bytes per row. cuVSLAM ignores it for host memory but
        // requires it when the supplied image lives in device memory.
        std::int32_t left_pitch = 0;
        std::int32_t right_pitch = 0;

        std::int64_t timestamp_ns = 0;

        // RectifiedGray is expected to remain CUDA-resident in the normal
        // Parallax path, but the adapter can also accept host-backed test data.
        bool gpu_memory = true;
    };

    /**
     * Thin result wrapper around the cuVSLAM estimate.
     *
     * This is deliberately not a ProductStore contract. Phase 17 commit 3 will
     * define Parallax-native localization products independently of NVIDIA API
     * types.
     */
    struct CuVslamPoseEstimate {
        std::int64_t timestamp_ns = 0;
        std::optional<cuvslam::PoseWithCovariance> world_from_rig;
        std::optional<cuvslam::Pose> slam_world_from_rig;

        std::vector<cuvslam::Observation> observations;
        std::vector<cuvslam::Landmark> landmarks;
        [[nodiscard]] bool valid() const noexcept { return world_from_rig.has_value(); }
    };

    /**
     * Owns the persistent cuVSLAM odometry session.
     *
     * The rig frame is Parallax's existing stereo_body frame. Camera
     * extrinsics are derived from the existing SensorExtrinsics plus the
     * rectification transforms rather than redefining the rig around either
     * optical camera.
     */
    class CuVslamLocalizer {
        public:
            CuVslamLocalizer() = default;
            ~CuVslamLocalizer();

            CuVslamLocalizer(const CuVslamLocalizer&) = delete;
            CuVslamLocalizer& operator=(const CuVslamLocalizer&) = delete;

            CuVslamLocalizer(CuVslamLocalizer&&) = delete;
            CuVslamLocalizer& operator=(CuVslamLocalizer&&) = delete;

            bool initialize(const stereo::StereoCalibration& calibration, const core::SensorExtrinsics& extrinsics);

            /**
             * Submit one synchronized rectified stereo observation.
             *
             * cuVSLAM Track() is synchronous. The caller may therefore release
             * or recycle the borrowed image buffers once this method returns.
             *
             * Throws for malformed input or non-monotonic timestamps. Actual
             * cuVSLAM tracking loss is represented by a valid call returning
             * world_from_rig == nullopt.
             */
            CuVslamPoseEstimate track(const CuVslamFrame& frame);

            bool reset();
            void shutdown() noexcept;

            [[nodiscard]] bool initialized() const noexcept { return odometry_ != nullptr; }

            /**
             * Exposed for focused rig-contract tests.
             *
             * The returned Camera::rig_from_camera transforms are expressed
             * from each rectified virtual optical frame into stereo_body.
             */
            static cuvslam::Rig makeRig(const stereo::StereoCalibration& calibration, const core::SensorExtrinsics& extrinsics);

        private:
            std::unique_ptr<cuvslam::Odometry> odometry_;
            std::unique_ptr<cuvslam::Slam> slam_;

            // Retaining these lightweight configuration objects allows reset()
            // to restart the stateful estimator without rereading calibration.
            std::optional<cuvslam::Rig> rig_;
            std::optional<cuvslam::Odometry::Config> config_;
            std::optional<cuvslam::Slam::Config> slam_config_;

            std::int32_t image_width_ = 0;
            std::int32_t image_height_ = 0;

            // Sequencing primarily belongs to CuVslamProducer, but rejecting
            // reordered frames here prevents direct adapter misuse from
            // corrupting the stateful cuVSLAM session.
            std::int64_t last_timestamp_ns_ = -1;
    };
}