#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path.cwd()

def write(rel, text):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(f"updated {rel}")

def replace(rel, old, new):
    p = ROOT / rel
    s = p.read_text()
    if old not in s:
        raise RuntimeError(f"expected text not found in {rel}: {old[:80]!r}")
    p.write_text(s.replace(old, new))
    print(f"updated {rel}")

# Guard against applying to a substantially different tree.
required = [
    "include/parallax/mapping/spatial_tsdf_producer.hpp",
    "src/mapping/spatial_tsdf_producer.cpp",
    "include/parallax/core/runtime.hpp",
    "src/core/runtime.cpp",
    "src/mapping/CMakeLists.txt",
]
for rel in required:
    if not (ROOT / rel).exists():
        sys.exit(f"Run this from the Parallax repository root; missing {rel}")

write("include/parallax/mapping/spatial_map.hpp", r'''#pragma once

#include <parallax/mapping/mapping_config.hpp>

#include <cuda_runtime.h>
#include <nvblox/core/cuda_stream.h>
#include <nvblox/mapper/mapper.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace parallax::mapping {

// Runtime-owned persistent nvblox state. Producers may derive representations
// from this state, but the state itself is never copied into ProductStore.
class SpatialMap final {
public:
    SpatialMap(const MappingConfig& config, cudaStream_t cuda_stream)
        : config_(config),
          cuda_stream_(cuda_stream),
          nvblox_stream_(std::make_shared<nvblox::CudaStreamNonOwning>(&cuda_stream_)) {
        if (cuda_stream_ == nullptr) {
            throw std::invalid_argument("spatial map requires CUDA stream");
        }
    }

    void reset(std::uint64_t epoch) {
        mapper_ = std::make_unique<nvblox::Mapper>(
            config_.voxel_size_m,
            nvblox::BlockMemoryPoolParams{},
            nvblox::ProjectiveLayerType::kTsdf,
            nvblox_stream_);
        mapper_->tsdf_integrator().max_integration_distance_m(
            config_.max_integration_distance_m);
        mapper_->tsdf_integrator().truncation_distance_vox(
            config_.tsdf_truncation_distance_vox);
        mapper_->tsdf_integrator().max_weight(config_.tsdf_max_weight);
        mapper_->color_integrator().max_integration_distance_m(
            config_.max_integration_distance_m);
        epoch_ = epoch;
    }

    [[nodiscard]] bool initialized() const noexcept { return static_cast<bool>(mapper_); }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

    nvblox::Mapper& mapper() {
        if (!mapper_) throw std::logic_error("spatial map is not initialized");
        return *mapper_;
    }

    const nvblox::Mapper& mapper() const {
        if (!mapper_) throw std::logic_error("spatial map is not initialized");
        return *mapper_;
    }

    [[nodiscard]] nvblox::CudaStream* stream() noexcept { return nvblox_stream_.get(); }
    [[nodiscard]] cudaStream_t cudaStream() const noexcept { return cuda_stream_; }

private:
    const MappingConfig& config_;
    cudaStream_t cuda_stream_ = nullptr;
    std::shared_ptr<nvblox::CudaStreamNonOwning> nvblox_stream_;
    std::unique_ptr<nvblox::Mapper> mapper_;
    std::uint64_t epoch_ = 0;
};

}  // namespace parallax::mapping
''')

write("include/parallax/mapping/spatial_map_state.hpp", r'''#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace parallax::mapping {

// Lightweight publication describing a new persistent-map revision. The
// nvblox map remains owned by SpatialMap and GPU resident.
struct SpatialMapState {
    std::uint64_t localization_epoch = 0;
    std::uint64_t revision = 0;
    std::uint64_t integrated_frames = 0;
    std::uint64_t epoch_resets = 0;
    std::size_t allocated_blocks = 0;

    std::array<float, 9> world_from_camera_rotation{};
    std::array<float, 3> world_from_camera_translation_m{};
};

}  // namespace parallax::mapping
''')

write("include/parallax/mapping/spatial_tsdf_producer.hpp", r'''#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/mapping/mapping_config.hpp>
#include <parallax/mapping/spatial_map.hpp>
#include <parallax/mapping/spatial_map_state.hpp>
#include <parallax/core/sensor_extrinsics.hpp>
#include <parallax/localization/localization.hpp>
#include <parallax/stereo/calibration.hpp>

#include <nvblox/sensors/camera.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::mapping {

// Persistent-map integrator. Despite the historical class name, this producer
// no longer materializes a TSDF visualization product. It only advances the
// GPU-resident SpatialMap and publishes a lightweight map revision.
class SpatialTsdfProducer final : public parallax::core::Producer {
public:
    SpatialTsdfProducer(const parallax::stereo::StereoCalibration& calibration,
                        const parallax::core::SensorExtrinsics& extrinsics,
                        const MappingConfig& config,
                        SpatialMap& map,
                        parallax::core::ProductStore& products);

    std::string_view name() const noexcept override;
    const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
    const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
    const std::vector<parallax::core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;
    parallax::core::ExecutionPolicy execution_policy() const noexcept override;
    parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

private:
    static constexpr std::size_t DepthHistoryCapacity = 4;

    void resetForEpoch(std::uint64_t epoch);
    nvblox::Transform worldFromRectifiedCamera(
        const parallax::localization::LocalizationPose& pose) const;

    const parallax::stereo::StereoCalibration& calibration_;
    const MappingConfig& config_;
    SpatialMap& map_;
    parallax::core::ProductStore& products_;
    nvblox::Camera camera_;

    std::array<double, 9> body_from_rectified_rotation_{};
    std::array<double, 3> body_from_rectified_translation_{};

    std::optional<std::uint64_t> epoch_;
    std::optional<parallax::core::SourceObservation> last_integrated_;
    std::uint64_t integrated_frames_ = 0;
    std::uint64_t epoch_resets_ = 0;
    std::uint64_t map_revision_ = 0;

    const std::vector<parallax::core::ProductId> inputs_{
        parallax::core::ProductId::LocalizationPose,
        parallax::core::ProductId::Depth};
    const std::vector<parallax::core::ProductId> outputs_{
        parallax::core::ProductId::SpatialMapState};
    const std::vector<parallax::core::CompatibleInputRequirement> compatible_inputs_{
        {parallax::core::ProductId::Depth, DepthHistoryCapacity}};
};

}  // namespace parallax::mapping
''')

write("src/mapping/spatial_tsdf_producer.cpp", r'''#include <parallax/mapping/spatial_tsdf_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/isp/frame_types.hpp>

#include <Eigen/Geometry>

#include <cmath>
#include <memory>
#include <stdexcept>

namespace parallax::mapping {
namespace {

std::array<double, 9> quaternionToMatrix(const std::array<double, 4>& q) {
    Eigen::Quaterniond quaternion(q[3], q[0], q[1], q[2]);
    const double norm = quaternion.norm();
    if (!std::isfinite(norm) || norm <= 1.0e-9) {
        throw std::invalid_argument("spatial TSDF received invalid rotation");
    }
    quaternion.normalize();
    const Eigen::Matrix3d m = quaternion.toRotationMatrix();
    return {m(0,0), m(0,1), m(0,2),
            m(1,0), m(1,1), m(1,2),
            m(2,0), m(2,1), m(2,2)};
}

std::array<double, 9> transpose(const std::array<double, 9>& m) {
    return {m[0], m[3], m[6],
            m[1], m[4], m[7],
            m[2], m[5], m[8]};
}

std::array<double, 9> multiply(const std::array<double, 9>& a,
                               const std::array<double, 9>& b) {
    std::array<double, 9> out{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            out[row * 3 + col] = a[row * 3 + 0] * b[0 * 3 + col] +
                                 a[row * 3 + 1] * b[1 * 3 + col] +
                                 a[row * 3 + 2] * b[2 * 3 + col];
        }
    }
    return out;
}

}  // namespace

SpatialTsdfProducer::SpatialTsdfProducer(
    const parallax::stereo::StereoCalibration& calibration,
    const parallax::core::SensorExtrinsics& extrinsics,
    const MappingConfig& config,
    SpatialMap& map,
    parallax::core::ProductStore& products)
    : calibration_(calibration), config_(config), map_(map), products_(products) {
    if (!calibration.loaded()) {
        throw std::invalid_argument("spatial TSDF requires calibration");
    }

    const auto& p1 = calibration.P1();
    camera_ = nvblox::Camera(static_cast<float>(p1[0]), static_cast<float>(p1[5]),
                             static_cast<float>(p1[2]), static_cast<float>(p1[6]),
                             static_cast<int>(calibration.metadata().image_width),
                             static_cast<int>(calibration.metadata().image_height));

    const auto body_from_raw = quaternionToMatrix(extrinsics.left_camera.rotation_xyzw);
    body_from_rectified_rotation_ = multiply(body_from_raw, transpose(calibration.R1()));
    body_from_rectified_translation_ = extrinsics.left_camera.translation_m;
}

std::string_view SpatialTsdfProducer::name() const noexcept {
    return "mapping.spatial_map";
}

const std::vector<parallax::core::ProductId>& SpatialTsdfProducer::inputs() const noexcept {
    return inputs_;
}

const std::vector<parallax::core::ProductId>& SpatialTsdfProducer::outputs() const noexcept {
    return outputs_;
}

const std::vector<parallax::core::CompatibleInputRequirement>&
SpatialTsdfProducer::compatible_inputs() const noexcept {
    return compatible_inputs_;
}

parallax::core::ExecutionPolicy SpatialTsdfProducer::execution_policy() const noexcept {
    parallax::core::ExecutionPolicy policy{};
    policy.target_hz = config_.integration_rate_hz;
    policy.drop_policy = parallax::core::DropPolicy::Supersede;
    policy.priority = 5;
    policy.affinity = parallax::core::ResourceAffinity::Gpu;
    policy.stateful = true;
    return policy;
}

void SpatialTsdfProducer::resetForEpoch(std::uint64_t epoch) {
    map_.reset(epoch);
    epoch_ = epoch;
    last_integrated_.reset();
}

nvblox::Transform SpatialTsdfProducer::worldFromRectifiedCamera(
    const parallax::localization::LocalizationPose& pose) const {
    Eigen::Quaternionf world_from_body(
        pose.rotation_xyzw[3], pose.rotation_xyzw[0],
        pose.rotation_xyzw[1], pose.rotation_xyzw[2]);
    world_from_body.normalize();

    Eigen::Matrix3f body_from_rectified;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            body_from_rectified(row, col) =
                static_cast<float>(body_from_rectified_rotation_[row * 3 + col]);
        }
    }

    const Eigen::Quaternionf body_from_rectified_q(body_from_rectified);
    const Eigen::Vector3f body_from_rectified_t(
        static_cast<float>(body_from_rectified_translation_[0]),
        static_cast<float>(body_from_rectified_translation_[1]),
        static_cast<float>(body_from_rectified_translation_[2]));
    const Eigen::Vector3f world_from_body_t(
        pose.translation_m[0], pose.translation_m[1], pose.translation_m[2]);

    nvblox::Transform transform = nvblox::Transform::Identity();
    transform.linear() = (world_from_body * body_from_rectified_q).toRotationMatrix();
    transform.translation() = world_from_body * body_from_rectified_t + world_from_body_t;
    return transform;
}

parallax::core::SubmitResult SpatialTsdfProducer::submit(
    parallax::core::ExecutionContext& context) {
    const auto pose = products_.latest<parallax::localization::LocalizationPose>(
        parallax::core::ProductId::LocalizationPose);
    if (!pose || !pose->valid() || !pose->payload) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (last_integrated_ && pose->metadata.observation == *last_integrated_) {
        return parallax::core::SubmitResult::NoWork;
    }

    const auto depth = products_.find_observation<parallax::isp::DepthFrame>(
        parallax::core::ProductId::Depth, pose->metadata.observation);
    if (!depth || !depth->valid() || !depth->payload) {
        return parallax::core::SubmitResult::NoWork;
    }

    if (!epoch_ || *epoch_ != pose->payload->epoch) {
        if (epoch_) ++epoch_resets_;
        resetForEpoch(pose->payload->epoch);
    }

    if (!context.waitFor(depth->completion, map_.cudaStream())) {
        return parallax::core::SubmitResult::Failed;
    }

    const auto& frame = *depth->payload;
    nvblox::DepthImageConstView depth_view(
        static_cast<int>(frame.height), static_cast<int>(frame.width),
        static_cast<int>(frame.depth.pitch()), 1, frame.depth.dataAs<float>());
    nvblox::MaskedDepthImageConstView masked_depth(
        depth_view, nvblox::kMaskActiveEverywhere);

    const nvblox::Transform world_from_camera =
        worldFromRectifiedCamera(*pose->payload);
    map_.mapper().integrateDepth(masked_depth, world_from_camera, camera_);

    ++integrated_frames_;
    ++map_revision_;
    last_integrated_ = pose->metadata.observation;

    auto state = std::make_shared<SpatialMapState>();
    state->localization_epoch = pose->payload->epoch;
    state->revision = map_revision_;
    state->integrated_frames = integrated_frames_;
    state->epoch_resets = epoch_resets_;
    state->allocated_blocks = map_.mapper().tsdf_layer().numAllocatedBlocks();

    const auto& r = world_from_camera.linear();
    state->world_from_camera_rotation = {
        r(0,0), r(0,1), r(0,2),
        r(1,0), r(1,1), r(1,2),
        r(2,0), r(2,1), r(2,2)};
    const auto& t = world_from_camera.translation();
    state->world_from_camera_translation_m = {t.x(), t.y(), t.z()};

    auto metadata = pose->metadata;
    metadata.production_timestamp = parallax::core::ExecutionContext::now();
    metadata.valid = true;
    std::shared_ptr<const SpatialMapState> published = std::move(state);
    products_.publish(parallax::core::make_product(
        parallax::core::ProductId::SpatialMapState,
        metadata,
        std::move(published)));

    return parallax::core::SubmitResult::Submitted;
}

}  // namespace parallax::mapping
''')

write("include/parallax/mapping/tsdf_snapshot_producer.hpp", r'''#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/mapping/mapping_config.hpp>
#include <parallax/mapping/spatial_map.hpp>
#include <parallax/mapping/spatial_map_state.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::mapping {

class TsdfSnapshotProducer final : public parallax::core::Producer {
public:
    TsdfSnapshotProducer(const MappingConfig& config,
                         SpatialMap& map,
                         parallax::core::ProductStore& products);

    std::string_view name() const noexcept override;
    const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
    const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
    parallax::core::ExecutionPolicy execution_policy() const noexcept override;
    parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

private:
    const MappingConfig& config_;
    SpatialMap& map_;
    parallax::core::ProductStore& products_;
    std::optional<std::uint64_t> last_revision_;

    const std::vector<parallax::core::ProductId> inputs_{
        parallax::core::ProductId::SpatialMapState};
    const std::vector<parallax::core::ProductId> outputs_{
        parallax::core::ProductId::SpatialTsdf};
};

}  // namespace parallax::mapping
''')

write("src/mapping/tsdf_snapshot_producer.cpp", r'''#include <parallax/mapping/tsdf_snapshot_producer.hpp>
#include <parallax/mapping/spatial_tsdf_state.hpp>
#include <parallax/mapping/tsdf_snapshot.hpp>
#include <parallax/core/execution_context.hpp>

#include <memory>

namespace parallax::mapping {

TsdfSnapshotProducer::TsdfSnapshotProducer(
    const MappingConfig& config,
    SpatialMap& map,
    parallax::core::ProductStore& products)
    : config_(config), map_(map), products_(products) {}

std::string_view TsdfSnapshotProducer::name() const noexcept {
    return "mapping.tsdf_snapshot";
}

const std::vector<parallax::core::ProductId>& TsdfSnapshotProducer::inputs() const noexcept {
    return inputs_;
}

const std::vector<parallax::core::ProductId>& TsdfSnapshotProducer::outputs() const noexcept {
    return outputs_;
}

parallax::core::ExecutionPolicy TsdfSnapshotProducer::execution_policy() const noexcept {
    parallax::core::ExecutionPolicy policy{};
    policy.target_hz = config_.tsdf_visualization_rate_hz;
    policy.drop_policy = parallax::core::DropPolicy::Supersede;
    policy.priority = 2;
    policy.affinity = parallax::core::ResourceAffinity::Gpu;
    policy.stateful = false;
    return policy;
}

parallax::core::SubmitResult TsdfSnapshotProducer::submit(
    parallax::core::ExecutionContext&) {
    const auto map_state = products_.latest<SpatialMapState>(
        parallax::core::ProductId::SpatialMapState);
    if (!map_state || !map_state->valid() || !map_state->payload || !map_.initialized()) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (last_revision_ && *last_revision_ == map_state->payload->revision) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (map_.epoch() != map_state->payload->localization_epoch) {
        return parallax::core::SubmitResult::NoWork;
    }

    auto state = std::make_shared<SpatialTsdfState>();
    state->localization_epoch = map_state->payload->localization_epoch;
    state->integrated_frames = map_state->payload->integrated_frames;
    state->epoch_resets = map_state->payload->epoch_resets;
    state->allocated_blocks = map_state->payload->allocated_blocks;

    const auto& c = map_state->payload->world_from_camera_translation_m;
    const nvblox::Vector3f center(c[0], c[1], c[2]);
    const TsdfSnapshotConfig snapshot_config{
        config_.voxel_size_m,
        config_.debug_columns,
        config_.debug_rows,
        config_.debug_slices,
        config_.debug_surface_band_m};

    if (!buildTsdfSnapshot(map_.mapper().tsdf_layer(),
                           center,
                           map_.stream(),
                           snapshot_config,
                           state.get())) {
        return parallax::core::SubmitResult::Failed;
    }

    auto metadata = map_state->metadata;
    metadata.production_timestamp = parallax::core::ExecutionContext::now();
    metadata.valid = true;
    std::shared_ptr<const SpatialTsdfState> published = std::move(state);
    products_.publish(parallax::core::make_product(
        parallax::core::ProductId::SpatialTsdf,
        metadata,
        std::move(published)));
    last_revision_ = map_state->payload->revision;
    return parallax::core::SubmitResult::Submitted;
}

}  // namespace parallax::mapping
''')

write("include/parallax/mapping/spatial_mesh_producer.hpp", r'''#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/mapping/mapping_config.hpp>
#include <parallax/mapping/spatial_map.hpp>
#include <parallax/mapping/spatial_map_state.hpp>
#include <parallax/stereo/calibration.hpp>

#include <nvblox/sensors/camera.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::mapping {

class SpatialMeshProducer final : public parallax::core::Producer {
public:
    SpatialMeshProducer(const MappingConfig& config,
                        const parallax::stereo::StereoCalibration& calibration,
                        SpatialMap& map,
                        parallax::core::ProductStore& products);

    std::string_view name() const noexcept override;
    const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
    const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
    const std::vector<parallax::core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;
    parallax::core::ExecutionPolicy execution_policy() const noexcept override;
    parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

private:
    static constexpr std::size_t RectifiedRgbHistoryCapacity = 4;

    const MappingConfig& config_;
    SpatialMap& map_;
    parallax::core::ProductStore& products_;
    nvblox::Camera camera_;
    std::optional<std::uint64_t> last_revision_;
    std::uint64_t mesh_revision_ = 0;

    const std::vector<parallax::core::ProductId> inputs_{
        parallax::core::ProductId::SpatialMapState,
        parallax::core::ProductId::RectifiedRgb};
    const std::vector<parallax::core::ProductId> outputs_{
        parallax::core::ProductId::SpatialMesh};
    const std::vector<parallax::core::CompatibleInputRequirement> compatible_inputs_{
        {parallax::core::ProductId::RectifiedRgb, RectifiedRgbHistoryCapacity}};
};

}  // namespace parallax::mapping
''')

write("src/mapping/spatial_mesh_producer.cpp", r'''#include <parallax/mapping/spatial_mesh_producer.hpp>
#include <parallax/mapping/spatial_mesh_snapshot.hpp>
#include <parallax/mapping/spatial_mesh_state.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/isp/frame_types.hpp>

#include <memory>

namespace parallax::mapping {

SpatialMeshProducer::SpatialMeshProducer(
    const MappingConfig& config,
    const parallax::stereo::StereoCalibration& calibration,
    SpatialMap& map,
    parallax::core::ProductStore& products)
    : config_(config), map_(map), products_(products) {
    const auto& p1 = calibration.P1();
    camera_ = nvblox::Camera(static_cast<float>(p1[0]), static_cast<float>(p1[5]),
                             static_cast<float>(p1[2]), static_cast<float>(p1[6]),
                             static_cast<int>(calibration.metadata().image_width),
                             static_cast<int>(calibration.metadata().image_height));
}

std::string_view SpatialMeshProducer::name() const noexcept {
    return "mapping.spatial_mesh";
}

const std::vector<parallax::core::ProductId>& SpatialMeshProducer::inputs() const noexcept {
    return inputs_;
}

const std::vector<parallax::core::ProductId>& SpatialMeshProducer::outputs() const noexcept {
    return outputs_;
}

const std::vector<parallax::core::CompatibleInputRequirement>&
SpatialMeshProducer::compatible_inputs() const noexcept {
    return compatible_inputs_;
}

parallax::core::ExecutionPolicy SpatialMeshProducer::execution_policy() const noexcept {
    parallax::core::ExecutionPolicy policy{};
    policy.target_hz = config_.mesh_update_rate_hz;
    policy.drop_policy = parallax::core::DropPolicy::Supersede;
    policy.priority = 2;
    policy.affinity = parallax::core::ResourceAffinity::Gpu;
    policy.stateful = false;
    return policy;
}

parallax::core::SubmitResult SpatialMeshProducer::submit(
    parallax::core::ExecutionContext& context) {
    const auto map_state = products_.latest<SpatialMapState>(
        parallax::core::ProductId::SpatialMapState);
    if (!map_state || !map_state->valid() || !map_state->payload || !map_.initialized()) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (last_revision_ && *last_revision_ == map_state->payload->revision) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (map_.epoch() != map_state->payload->localization_epoch) {
        return parallax::core::SubmitResult::NoWork;
    }

    // Phase A preserves the existing color-mesh behavior but moves all appearance
    // and mesh materialization work out of the persistent integrator.
    if (!config_.color_enabled) {
        return parallax::core::SubmitResult::NoWork;
    }

    const auto rgb = products_.find_observation<parallax::isp::RectifiedStereoFrame>(
        parallax::core::ProductId::RectifiedRgb, map_state->metadata.observation);
    if (!rgb || !rgb->valid() || !rgb->payload || !rgb->payload->left.isAllocated()) {
        return parallax::core::SubmitResult::NoWork;
    }
    if (!context.waitFor(rgb->completion, map_.cudaStream())) {
        return parallax::core::SubmitResult::Failed;
    }

    const auto& cf = *rgb->payload;
    nvblox::ColorImageConstView color_view(
        static_cast<int>(cf.height), static_cast<int>(cf.width),
        static_cast<int>(cf.left.pitch()), 3,
        reinterpret_cast<const nvblox::Color*>(cf.left.dataAs<std::uint8_t>()));
    const nvblox::MaskedColorImageConstView masked_color(
        color_view, nvblox::kMaskActiveEverywhere);

    nvblox::Transform world_from_camera = nvblox::Transform::Identity();
    const auto& r = map_state->payload->world_from_camera_rotation;
    world_from_camera.linear() <<
        r[0], r[1], r[2],
        r[3], r[4], r[5],
        r[6], r[7], r[8];
    const auto& t = map_state->payload->world_from_camera_translation_m;
    world_from_camera.translation() = nvblox::Vector3f(t[0], t[1], t[2]);

    map_.mapper().integrateColor(masked_color, world_from_camera, camera_);
    map_.mapper().updateFlatColorMesh();

    auto state = std::make_shared<SpatialMeshState>();
    state->localization_epoch = map_state->payload->localization_epoch;
    state->integrated_frames = map_state->payload->integrated_frames;
    state->mesh_revision = ++mesh_revision_;

    if (!buildSpatialMeshSnapshot(map_.mapper().flat_color_mesh(), state.get())) {
        return parallax::core::SubmitResult::NoWork;
    }

    auto metadata = map_state->metadata;
    metadata.production_timestamp = parallax::core::ExecutionContext::now();
    metadata.valid = true;
    std::shared_ptr<const SpatialMeshState> published = std::move(state);
    products_.publish(parallax::core::make_product(
        parallax::core::ProductId::SpatialMesh,
        metadata,
        std::move(published)));
    last_revision_ = map_state->payload->revision;
    return parallax::core::SubmitResult::Submitted;
}

}  // namespace parallax::mapping
''')

# Product vocabulary: lightweight persistent-map revision becomes the baseline root.
replace("include/parallax/core/product_id.hpp",
        "        LocalOccupancy,\n        SpatialTsdf,\n        SpatialMesh",
        "        LocalOccupancy,\n        SpatialMapState,\n        SpatialTsdf,\n        SpatialMesh")

# Config: visualization gets an independent 1 Hz policy.
replace("include/parallax/mapping/mapping_config.hpp",
        "        float tsdf_max_weight = 5.0F;\n        float mesh_update_rate_hz = 2.0F;",
        "        float tsdf_max_weight = 5.0F;\n        float tsdf_visualization_rate_hz = 1.0F;\n        float mesh_update_rate_hz = 2.0F;")
replace("src/mapping/mapping_config.cpp",
        "            if (n[\"max_weight\"]) tsdf_max_weight=n[\"max_weight\"].as<float>();\n        }",
        "            if (n[\"max_weight\"]) tsdf_max_weight=n[\"max_weight\"].as<float>();\n            if (n[\"visualization_rate_hz\"]) tsdf_visualization_rate_hz=n[\"visualization_rate_hz\"].as<float>();\n        }")
replace("src/mapping/mapping_config.cpp",
        "        !positive(integration_rate_hz) || !positive(tsdf_truncation_distance_vox) || !positive(tsdf_max_weight) ||\n        !positive(mesh_update_rate_hz)",
        "        !positive(integration_rate_hz) || !positive(tsdf_truncation_distance_vox) || !positive(tsdf_max_weight) ||\n        !positive(tsdf_visualization_rate_hz) || !positive(mesh_update_rate_hz)")
replace("config/mapping.yaml",
        "    max_weight: 5.0\n  mesh:",
        "    max_weight: 5.0\n    visualization_rate_hz: 1.0\n  mesh:")

# Mapping library sources.
replace("src/mapping/CMakeLists.txt",
        "    spatial_tsdf_producer.cpp\n    spatial_mesh_snapshot.cpp\n    tsdf_snapshot.cpp",
        "    spatial_tsdf_producer.cpp\n    tsdf_snapshot_producer.cpp\n    spatial_mesh_producer.cpp\n    spatial_mesh_snapshot.cpp\n    tsdf_snapshot.cpp")

# Runtime ownership/includes.
replace("include/parallax/core/runtime.hpp",
        "#include <parallax/mapping/spatial_tsdf_producer.hpp>\n#include <parallax/mapping/mapping_config.hpp>",
        "#include <parallax/mapping/spatial_map.hpp>\n#include <parallax/mapping/spatial_tsdf_producer.hpp>\n#include <parallax/mapping/tsdf_snapshot_producer.hpp>\n#include <parallax/mapping/spatial_mesh_producer.hpp>\n#include <parallax/mapping/mapping_config.hpp>")
replace("include/parallax/core/runtime.hpp",
        "            std::unique_ptr<parallax::localization::CuVslamProducer> cuvslam_producer_;\n            std::unique_ptr<parallax::mapping::SpatialTsdfProducer> spatial_tsdf_producer_;",
        "            std::unique_ptr<parallax::localization::CuVslamProducer> cuvslam_producer_;\n            std::unique_ptr<parallax::mapping::SpatialMap> spatial_map_;\n            std::unique_ptr<parallax::mapping::SpatialTsdfProducer> spatial_tsdf_producer_;\n            std::unique_ptr<parallax::mapping::TsdfSnapshotProducer> tsdf_snapshot_producer_;\n            std::unique_ptr<parallax::mapping::SpatialMeshProducer> spatial_mesh_producer_;")

old_ctor = '''        spatial_tsdf_producer_ = std::make_unique<parallax::mapping::SpatialTsdfProducer>(pipeline_.calibration(),
                                                                                          sensor_extrinsics_,
                                                                                          mapping_config_,
                                                                                          context_.products(),
                                                                                          resolver_,
                                                                                          context_.stereoLane().cudaHandle());'''
new_ctor = '''        spatial_map_ = std::make_unique<parallax::mapping::SpatialMap>(
            mapping_config_, context_.stereoLane().cudaHandle());

        spatial_tsdf_producer_ = std::make_unique<parallax::mapping::SpatialTsdfProducer>(
            pipeline_.calibration(), sensor_extrinsics_, mapping_config_,
            *spatial_map_, context_.products());

        tsdf_snapshot_producer_ = std::make_unique<parallax::mapping::TsdfSnapshotProducer>(
            mapping_config_, *spatial_map_, context_.products());

        spatial_mesh_producer_ = std::make_unique<parallax::mapping::SpatialMeshProducer>(
            mapping_config_, pipeline_.calibration(), *spatial_map_, context_.products());'''
replace("src/core/runtime.cpp", old_ctor, new_ctor)
replace("src/core/runtime.cpp",
        "        graph_.register_producer(*cuvslam_producer_);\n        graph_.register_producer(*spatial_tsdf_producer_);",
        "        graph_.register_producer(*cuvslam_producer_);\n        graph_.register_producer(*spatial_tsdf_producer_);\n        graph_.register_producer(*tsdf_snapshot_producer_);\n        graph_.register_producer(*spatial_mesh_producer_);")
replace("src/core/runtime.cpp",
        "        resolver_.acquire(ProductId::SpatialTsdf, DemandSource::RuntimeBaseline);",
        "        // Persistent mapping is baseline state. TSDF and mesh are derived,\n        // demand-driven materializations and are not baseline products.\n        resolver_.acquire(ProductId::SpatialMapState, DemandSource::RuntimeBaseline);")

print("\nApplied Phase A mapping separation.")
print("Persistent nvblox state is now runtime-owned by SpatialMap.")
print("SpatialTsdfProducer only integrates depth and publishes SpatialMapState revisions.")
print("TSDF snapshot and mesh materialization are independent producers.")
print("\nNext:")
print("  git diff --check")
print("  git diff --stat")
print("  ./scripts/test.sh")
print("  ./scripts/run.sh")
