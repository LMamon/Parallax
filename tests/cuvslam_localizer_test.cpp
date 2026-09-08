#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>

#include "parallax/core/sensor_extrinsics.hpp"
#include "parallax/localization/cuvslam_localizer.hpp"
#include "parallax/stereo/calibration.hpp"

namespace {
    constexpr double kTolerance = 1e-6;
    constexpr double kCameraParameterTolerance = 1e-4;

    parallax::stereo::StereoCalibration loadCalibration() {
        parallax::stereo::StereoCalibration calibration;

        const std::filesystem::path calibration_path = std::filesystem::path(PARALLAX_SOURCE_DIR) / "config/camera/calibration/results/rectification";

        EXPECT_TRUE(calibration.load(calibration_path));

        return calibration;
    }

    parallax::core::SensorExtrinsics loadExtrinsics() {
        parallax::core::SensorExtrinsics extrinsics;
        const std::filesystem::path extrinsics_path = std::filesystem::path(PARALLAX_SOURCE_DIR) / "config/sensors/extrinsics.yaml";

        EXPECT_TRUE(extrinsics.loadFromFile(extrinsics_path.string()));
        return extrinsics;
    }

    std::array<double, 3> rotateVector(const std::array<float, 4>& quaternion, const std::array<double, 3>& vector) {
        const double x = quaternion[0];
        const double y = quaternion[1];
        const double z = quaternion[2];
        const double w = quaternion[3];

        // Convert cuVSLAM's xyzw quaternion to a rotation matrix so the test
        // verifies the stereo geometry independently of the adapter helpers.
        const std::array<double, 9> rotation{1.0 - 2.0 * (y * y + z * z),
                                             2.0 * (x * y - z * w),
                                             2.0 * (x * z + y * w),

                                             2.0 * (x * y + z * w),
                                             1.0 - 2.0 * (x * x + z * z),
                                             2.0 * (y * z - x * w),

                                             2.0 * (x * z - y * w),
                                             2.0 * (y * z + x * w),
                                             1.0 - 2.0 * (x * x + y * y)};

        return {rotation[0] * vector[0] + rotation[1] * vector[1] + rotation[2] * vector[2],
                rotation[3] * vector[0] + rotation[4] * vector[1] + rotation[5] * vector[2],
                rotation[6] * vector[0] + rotation[7] * vector[1] + rotation[8] * vector[2]};
    }

    TEST(CuVslamLocalizerTest, RigUsesRectifiedStereoCalibration) {
        const auto calibration = loadCalibration();
        const auto extrinsics = loadExtrinsics();

        ASSERT_TRUE(calibration.loaded());

        const auto rig = parallax::localization::CuVslamLocalizer::makeRig(calibration, extrinsics);
        ASSERT_EQ(rig.cameras.size(), 2U);

        const auto& left = rig.cameras[0];
        const auto& right = rig.cameras[1];

        EXPECT_EQ(left.size[0], calibration.metadata().image_width);
        EXPECT_EQ(left.size[1], calibration.metadata().image_height);
        EXPECT_EQ(right.size[0], calibration.metadata().image_width);
        EXPECT_EQ(right.size[1], calibration.metadata().image_height);

        const auto& p1 = calibration.P1();
        const auto& p2 = calibration.P2();

        // P1 and P2 own the intrinsics of the rectified images that cuVSLAM
        // consumes. Raw-camera intrinsics would describe the wrong image space.
        EXPECT_FLOAT_EQ(left.focal[0], static_cast<float>(p1[0]));
        EXPECT_FLOAT_EQ(left.focal[1], static_cast<float>(p1[5]));
        EXPECT_FLOAT_EQ(left.principal[0], static_cast<float>(p1[2]));
        EXPECT_FLOAT_EQ(left.principal[1], static_cast<float>(p1[6]));

        EXPECT_FLOAT_EQ(right.focal[0], static_cast<float>(p2[0]));
        EXPECT_FLOAT_EQ(right.focal[1], static_cast<float>(p2[5]));
        EXPECT_FLOAT_EQ(right.principal[0], static_cast<float>(p2[2]));
        EXPECT_FLOAT_EQ(right.principal[1], static_cast<float>(p2[6]));
    }

    TEST(CuVslamLocalizerTest, RectifiedCameraOriginsPreservePhysicalBaseline) {
        const auto calibration = loadCalibration();
        const auto extrinsics = loadExtrinsics();

        ASSERT_TRUE(calibration.loaded());

        const auto rig = parallax::localization::CuVslamLocalizer::makeRig(calibration, extrinsics);

        ASSERT_EQ(rig.cameras.size(), 2U);

        const auto& left = rig.cameras[0].rig_from_camera;
        const auto& right = rig.cameras[1].rig_from_camera;

        const std::array<double, 3> delta{right.translation[0] - left.translation[0],
                                          right.translation[1] - left.translation[1],
                                          right.translation[2] - left.translation[2]};

        const double baseline = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);

        EXPECT_NEAR(baseline, calibration.metadata().baseline_mm / 1000.0, kTolerance);

        // Rectification rotates the virtual image planes but does not change
        // the physical baseline. Both rectified cameras share an orientation.
        for (std::size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(left.rotation[i], right.rotation[i], kTolerance);
        }
    }

    TEST(CuVslamLocalizerTest, RightCameraLiesOnRectifiedPositiveXAxis) {
        const auto calibration = loadCalibration();
        const auto extrinsics = loadExtrinsics();

        ASSERT_TRUE(calibration.loaded());

        const auto rig = parallax::localization::CuVslamLocalizer::makeRig(calibration, extrinsics);
        ASSERT_EQ(rig.cameras.size(), 2U);

        const auto& left = rig.cameras[0].rig_from_camera;
        const auto& right = rig.cameras[1].rig_from_camera;

        const double baseline_m = calibration.metadata().baseline_mm / 1000.0;

        // In rectified-left coordinates the right camera center is +baseline
        // along x. Rotate that displacement into stereo_body and compare it
        // with the camera-center displacement produced by the adapter.
        const auto expected_delta = rotateVector(left.rotation, {baseline_m, 0.0, 0.0});

        EXPECT_NEAR(right.translation[0] - left.translation[0], expected_delta[0], kTolerance);
        EXPECT_NEAR(right.translation[1] - left.translation[1], expected_delta[1], kTolerance);
        EXPECT_NEAR(right.translation[2] - left.translation[2], expected_delta[2], kTolerance);
    }

    TEST(CuVslamLocalizerTest, LeftCameraKeepsConfiguredPhysicalOrigin) {
        const auto calibration = loadCalibration();
        const auto extrinsics = loadExtrinsics();

        ASSERT_TRUE(calibration.loaded());

        const auto rig = parallax::localization::CuVslamLocalizer::makeRig(calibration, extrinsics);

        ASSERT_EQ(rig.cameras.size(), 2U);

        const auto& left = rig.cameras[0].rig_from_camera;

        // Rectification rotates around the optical center, so the left camera
        // origin remains the one already established by SensorExtrinsics.
        EXPECT_NEAR(left.translation[0], extrinsics.left_camera.translation_m[0], kTolerance);
        EXPECT_NEAR(left.translation[1], extrinsics.left_camera.translation_m[1], kTolerance);
        EXPECT_NEAR(left.translation[2], extrinsics.left_camera.translation_m[2], kTolerance);
    }
}