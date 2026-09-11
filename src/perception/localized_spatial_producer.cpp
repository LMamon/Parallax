#include <parallax/perception/localized_spatial_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/localization/localization.hpp>
#include <parallax/perception/localized_spatial_observation.hpp>
#include <parallax/perception/object3d.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace parallax::perception {
    namespace {

        [[nodiscard]] std::array<double, 9> quaternionToMatrix(const std::array<double, 4>& q) {
            double x = q[0];
            double y = q[1];
            double z = q[2];
            double w = q[3];

            const double norm = std::sqrt(x * x + y * y + z * z + w * w);
            if (!std::isfinite(norm) || norm <= 1.0e-9) {
                throw std::invalid_argument("localized spatial producer received invalid camera rotation");
            }

            x /= norm;
            y /= norm;
            z /= norm;
            w /= norm;

            return {1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),

                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),

                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y)};
        }

        [[nodiscard]] std::array<double, 9> transpose(const std::array<double, 9>& m) {
            return {m[0], m[3], m[6],
                    m[1], m[4], m[7],
                    m[2], m[5], m[8]};
        }

        [[nodiscard]] std::array<double, 9> multiply(const std::array<double, 9>& a, const std::array<double, 9>& b) {

            std::array<double, 9> out{};

            for (std::size_t row = 0; row < 3; ++row) {
                for (std::size_t column = 0; column < 3; ++column) {
                    out[row * 3 + column] = a[row * 3 + 0] * b[0 * 3 + column] +
                                            a[row * 3 + 1] * b[1 * 3 + column] +
                                            a[row * 3 + 2] * b[2 * 3 + column];
                }
            }

            return out;
        }

        [[nodiscard]] std::array<float, 3> transformPoint(const std::array<double, 9>& rotation,
                                                          const std::array<double, 3>& translation,
                                                          const std::array<float, 3>& point) {

            return {static_cast<float>(rotation[0] * point[0] +
                                       rotation[1] * point[1] +
                                       rotation[2] * point[2] +
                                       translation[0]),

                    static_cast<float>(rotation[3] * point[0] +
                                       rotation[4] * point[1] +
                                       rotation[5] * point[2] +
                                       translation[1]),

                    static_cast<float>(rotation[6] * point[0] +
                                       rotation[7] * point[1] +
                                       rotation[8] * point[2] +
                                       translation[2])
            };
        }

        [[nodiscard]] std::array<double, 9> poseRotation(const localization::LocalizationPose& pose) {
            return quaternionToMatrix({pose.rotation_xyzw[0],
                                       pose.rotation_xyzw[1],
                                       pose.rotation_xyzw[2],
                                       pose.rotation_xyzw[3]});
        }
    }

    LocalizedSpatialProducer::LocalizedSpatialProducer(const stereo::StereoCalibration& calibration,
                                                       const core::SensorExtrinsics& extrinsics,
                                                       core::ProductStore& products) : products_(products) {

        if (!calibration.loaded()) {
            throw std::invalid_argument("localized spatial producer requires calibration");
        }

        // Object3D points use P1 rectified-left geometry, so undo R1 before
        // applying the physical left-camera -> body extrinsic.
        const auto body_from_raw = quaternionToMatrix(extrinsics.left_camera.rotation_xyzw);
        body_from_rectified_rotation_ = multiply(body_from_raw, transpose(calibration.R1()));

        body_from_rectified_translation_ = extrinsics.left_camera.translation_m;
    }

    std::string_view LocalizedSpatialProducer::name() const noexcept {
        return "perception.localized_spatial";
    }

    const std::vector<core::ProductId>& LocalizedSpatialProducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<core::ProductId>& LocalizedSpatialProducer::outputs() const noexcept {
        return outputs_;
    }

    const std::vector<core::CompatibleInputRequirement>& LocalizedSpatialProducer::compatible_inputs() const noexcept {
        return compatible_inputs_;
    }

    core::ExecutionPolicy LocalizedSpatialProducer::execution_policy() const noexcept {
        core::ExecutionPolicy policy{};
        policy.drop_policy = core::DropPolicy::Supersede;
        policy.affinity = core::ResourceAffinity::Cpu;
        policy.stateful = false;
        return policy;
    }

    core::SubmitResult LocalizedSpatialProducer::submit(core::ExecutionContext&) {
        const auto objects = products_.latest<Object3DSet>(core::ProductId::Object3D);
        if (!objects || !objects->valid() || !objects->payload) return core::SubmitResult::NoWork;

        const auto pose_history = products_.history<localization::LocalizationPose>(core::ProductId::LocalizationPose);
        std::shared_ptr<const core::Product<localization::LocalizationPose>> selected_pose{};

        auto selected_delta = std::chrono::steady_clock::duration::max();
        for (const auto& candidate : pose_history) {
            if (!candidate || !candidate->valid() || !candidate->payload) {
                continue;
            }

            // LocalizationPose comes from the same stereo observation domain.
            if (candidate->metadata.observation.source != objects->metadata.observation.source) {
                continue;
            }

            const auto delta = candidate->metadata.timestamp >= objects->metadata.timestamp
                                ? candidate->metadata.timestamp - objects->metadata.timestamp
                                : objects->metadata.timestamp - candidate->metadata.timestamp;

            if (delta < selected_delta) {
                selected_pose = candidate;
                selected_delta = delta;
            }
        }

        if (!selected_pose || selected_delta > MaxPoseDelta) return core::SubmitResult::NoWork;

        const auto& localization_pose = *selected_pose->payload;
        const auto world_from_body_rotation = poseRotation(localization_pose);

        const std::array<double, 3> world_from_body_translation{localization_pose.translation_m[0],
                                                                localization_pose.translation_m[1],
                                                                localization_pose.translation_m[2]};

        const auto world_from_rectified_rotation = multiply(world_from_body_rotation, body_from_rectified_rotation_);

        const auto body_camera_origin_in_world = transformPoint(world_from_body_rotation,
                                                                world_from_body_translation,
                                                                {static_cast<float>(body_from_rectified_translation_[0]),
                                                                static_cast<float>(body_from_rectified_translation_[1]),
                                                                static_cast<float>(body_from_rectified_translation_[2])});

        const std::array<double, 3> world_from_rectified_translation{body_camera_origin_in_world[0],
                                                                     body_camera_origin_in_world[1],
                                                                     body_camera_origin_in_world[2]};

        auto localized = std::make_shared<LocalizedSpatialObservation>();
        localized->objects = *objects->payload;
        localized->localization_observation = selected_pose->metadata.observation;
        localized->localization_epoch = localization_pose.epoch;
        localized->localization_time_delta = selected_delta;

        for (auto& object : localized->objects.objects) {
            object.position_m = transformPoint(world_from_rectified_rotation, world_from_rectified_translation, object.position_m);

            for (auto& corner : object.image_supported_corners_m) {
                corner = transformPoint(world_from_rectified_rotation, world_from_rectified_translation, corner);
            }

            for (auto& point : object.surface_points_m) {
                point = transformPoint(world_from_rectified_rotation, world_from_rectified_translation, point);
            }

            object.coordinate_frame = "localization_world";
        }

        // if (!localized->valid()) return core::SubmitResult::Failed;
        if (!localized->valid()) {
            std::cerr << "[LocalizedSpatial] invalid output"
                    << " object_obs=" << objects->metadata.observation.sequence
                    << " pose_obs=" << selected_pose->metadata.observation.sequence
                    << " epoch=" << localization_pose.epoch
                    << " objects=" << localized->objects.objects.size() << '\n';

            return core::SubmitResult::Failed;
        }

        auto metadata = objects->metadata;
        metadata.production_timestamp = std::chrono::steady_clock::now();

        products_.publish(core::make_product<LocalizedSpatialObservation>(core::ProductId::LocalizedSpatialObservation,
                                                                          metadata,
                                                                          std::move(localized)));

        return core::SubmitResult::Submitted;
    }
}