#include <parallax/mapping/local_occupancy_producer.hpp>
#include <parallax/mapping/occupancy_snapshot.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/isp/frame_types.hpp>

#include <Eigen/Geometry>

#include <cmath>
#include <stdexcept>

namespace parallax::mapping {
    namespace {

        std::array<double, 9> quaternionToMatrix(const std::array<double, 4>& q) {
            Eigen::Quaterniond quaternion(q[3], q[0], q[1], q[2]);
            const double norm = quaternion.norm();

            if (!std::isfinite(norm) || norm <= 1.0e-9) {
                throw std::invalid_argument("local occupancy received invalid rotation");
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

        std::array<double, 9> multiply(const std::array<double, 9>& a, const std::array<double, 9>& b) {
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

    }

    LocalOccupancyProducer::LocalOccupancyProducer(const parallax::stereo::StereoCalibration& calibration,
                                                   const parallax::core::SensorExtrinsics& extrinsics,
                                                   parallax::core::ProductStore& products,
                                                   const parallax::core::DependencyResolver& resolver,
                                                   cudaStream_t cuda_stream) : 
                                                calibration_(calibration),
                                                products_(products),
                                                resolver_(resolver),
                                                cuda_stream_(cuda_stream),
                                                nvblox_stream_(std::make_shared<nvblox::CudaStreamNonOwning>(&cuda_stream_)) {

        if (!calibration.loaded() || cuda_stream_ == nullptr) {
            throw std::invalid_argument("local occupancy requires calibration and CUDA stream");
        }

        const auto& p1 = calibration.P1();
        camera_ = nvblox::Camera(static_cast<float>(p1[0]), static_cast<float>(p1[5]),
                                 static_cast<float>(p1[2]), static_cast<float>(p1[6]),
                                 static_cast<int>(calibration.metadata().image_width),
                                 static_cast<int>(calibration.metadata().image_height));

        const auto body_from_raw = quaternionToMatrix(extrinsics.left_camera.rotation_xyzw);

        body_from_rectified_rotation_ = multiply(body_from_raw, transpose(calibration.R1()));
        body_from_rectified_translation_ = extrinsics.left_camera.translation_m;

        resetForEpoch(0);
        epoch_.reset();
    }

    std::string_view LocalOccupancyProducer::name() const noexcept {
        return "mapping.local_occupancy";
    }

    const std::vector<parallax::core::ProductId>&
    LocalOccupancyProducer::inputs() const noexcept { return inputs_; }

    const std::vector<parallax::core::ProductId>&
    LocalOccupancyProducer::outputs() const noexcept { return outputs_; }

    const std::vector<parallax::core::CompatibleInputRequirement>&
    LocalOccupancyProducer::compatible_inputs() const noexcept {
        return compatible_inputs_;
    }

    parallax::core::ExecutionPolicy LocalOccupancyProducer::execution_policy() const noexcept {
        parallax::core::ExecutionPolicy policy{};
        policy.target_hz = 10.0;
        policy.drop_policy = parallax::core::DropPolicy::Supersede;
        policy.priority = 5;
        policy.affinity = parallax::core::ResourceAffinity::Gpu;
        policy.stateful = true;

        return policy;
    }

    void LocalOccupancyProducer::resetForEpoch(std::uint64_t epoch) {
        mapper_ = std::make_unique<nvblox::Mapper>(VoxelSizeM,
                                                   nvblox::BlockMemoryPoolParams{},
                                                   nvblox::ProjectiveLayerType::kOccupancy,
                                                   nvblox_stream_);

        mapper_->occupancy_decay_integrator().decay_to_free(false);
        epoch_ = epoch;
        last_integrated_.reset();
    }

    nvblox::Transform LocalOccupancyProducer::worldFromRectifiedCamera(const parallax::localization::LocalizationPose& pose) const {

        Eigen::Quaternionf world_from_body(pose.rotation_xyzw[3], pose.rotation_xyzw[0], pose.rotation_xyzw[1], pose.rotation_xyzw[2]);
        world_from_body.normalize();

        Eigen::Matrix3f body_from_rectified;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                body_from_rectified(row, col) = static_cast<float>(body_from_rectified_rotation_[row * 3 + col]);
            }
        }

        const Eigen::Quaternionf body_from_rectified_q(body_from_rectified);
        const Eigen::Vector3f body_from_rectified_t(static_cast<float>(body_from_rectified_translation_[0]),
                                                    static_cast<float>(body_from_rectified_translation_[1]),
                                                    static_cast<float>(body_from_rectified_translation_[2]));

        const Eigen::Vector3f world_from_body_t(pose.translation_m[0], pose.translation_m[1], pose.translation_m[2]);

        nvblox::Transform transform = nvblox::Transform::Identity();
        transform.linear() = (world_from_body * body_from_rectified_q).toRotationMatrix();
        transform.translation() = world_from_body * body_from_rectified_t + world_from_body_t;

        return transform;
    }

    parallax::core::SubmitResult LocalOccupancyProducer::submit(
        parallax::core::ExecutionContext& context) {

        const auto pose = products_.latest<parallax::localization::LocalizationPose>(parallax::core::ProductId::LocalizationPose);
        if (!pose || !pose->valid() || !pose->payload) {
            return parallax::core::SubmitResult::NoWork;
        }

        if (last_integrated_ && pose->metadata.observation == *last_integrated_) {
            return parallax::core::SubmitResult::NoWork;
        }

        const auto depth = products_.find_observation<parallax::isp::DepthFrame>(parallax::core::ProductId::Depth,
                                                                                 pose->metadata.observation);

        if (!depth || !depth->valid() || !depth->payload) {
            return parallax::core::SubmitResult::NoWork;
        }

        if (!epoch_ || *epoch_ != pose->payload->epoch) {
            if (epoch_) ++epoch_resets_;
            resetForEpoch(pose->payload->epoch);
        }

        if (!context.waitFor(depth->completion, cuda_stream_)) {
            return parallax::core::SubmitResult::Failed;
        }

        const auto& frame = *depth->payload;
        nvblox::DepthImageConstView depth_view(static_cast<int>(frame.height),
                                              static_cast<int>(frame.width),
                                              static_cast<int>(frame.depth.pitch()),
                                              1,
                                              frame.depth.dataAs<float>());

        nvblox::MaskedDepthImageConstView masked_depth(depth_view, nvblox::kMaskActiveEverywhere);

        const nvblox::Transform world_from_camera = worldFromRectifiedCamera(*pose->payload);

        mapper_->integrateDepth(masked_depth, world_from_camera, camera_);
        mapper_->decayOccupancyExcludeLastView<nvblox::Camera>();
        mapper_->clearOutsideRadius(world_from_camera.translation(), KeepRadiusM);

        ++integrated_frames_;
        last_integrated_ = pose->metadata.observation;

        /*
        * The nvblox occupancy layer is runtime-baseline state. Keep integrating it
        * even with no viewer attached, but do not copy its bounded visualization
        * window back to the CPU unless Foxglove is actually subscribed.
        *
        * DependencyResolver already owns synchronized demand accounting, so this
        * check does not introduce a second subscription registry.
        */
        if (resolver_.demand(parallax::core::ProductId::LocalOccupancy, parallax::core::DemandSource::FoxgloveSubscriber) == 0) {
            return parallax::core::SubmitResult::Submitted;
        }

        auto state = std::make_shared<LocalOccupancyState>();
        state->localization_epoch = pose->payload->epoch;
        state->integrated_frames = integrated_frames_;
        state->epoch_resets = epoch_resets_;

        state->allocated_blocks = mapper_->occupancy_layer().numAllocatedBlocks();
        state->allocated_bytes = mapper_->occupancy_layer().numAllocatedBytes();

        const OccupancySnapshotConfig snapshot_config{VoxelSizeM, 54, 54, 27};
        if (!buildOccupancySnapshot(mapper_->occupancy_layer(),
                                    world_from_camera.translation(),
                                    cuda_stream_,
                                    snapshot_config,
                                    state.get())) {
            return parallax::core::SubmitResult::Failed;
        }

        auto metadata = pose->metadata;
        metadata.production_timestamp = parallax::core::ExecutionContext::now();
        metadata.valid = true;

        std::shared_ptr<const LocalOccupancyState> published_state = std::move(state);
        products_.publish(parallax::core::make_product(parallax::core::ProductId::LocalOccupancy, metadata, std::move(published_state)));

        return parallax::core::SubmitResult::Submitted;
    }

}
