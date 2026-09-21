#include <parallax/mapping/local_occupancy_producer.hpp>

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

        constexpr float OccupancyEpsilon = 1.0e-6F;
        std::uint8_t classifyOccupancy(float log_odds) {
            if (log_odds > OccupancyEpsilon) return static_cast<std::uint8_t>(OccupancyCell::Occupied);
            if (log_odds < -OccupancyEpsilon) return static_cast<std::uint8_t>(OccupancyCell::Free);
            return static_cast<std::uint8_t>(OccupancyCell::Unknown);
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

    bool LocalOccupancyProducer::buildSnapshot(
        const nvblox::Vector3f& center,
        LocalOccupancyState* state) {

        if (state == nullptr || mapper_ == nullptr) return false;

        const auto& layer = mapper_->occupancy_layer();
        using Block = nvblox::VoxelBlock<nvblox::OccupancyVoxel>;

        const float size_x = static_cast<float>(SnapshotColumns) * VoxelSizeM;
        const float size_y = static_cast<float>(SnapshotRows) * VoxelSizeM;
        const float size_z = static_cast<float>(SnapshotSlices) * VoxelSizeM;
        
        const auto snap = [](float value) {
            return std::floor(value / VoxelSizeM) * VoxelSizeM;
        };

        const nvblox::Vector3f origin(snap(center.x() - 0.5F * size_x),
                                      snap(center.y() - 0.5F * size_y),
                                      snap(center.z() - 0.5F * size_z));

        const nvblox::Vector3f upper(origin.x() + size_x, origin.y() + size_y, origin.z() + size_z);

        const float block_size = layer.block_size();
        const auto indices = layer.getAllBlockIndices();
        std::vector<nvblox::Index3D> selected_indices;
        std::vector<const Block*> selected_blocks;

        for (const auto& index : indices) {
            const nvblox::Vector3f block_min(static_cast<float>(index.x()) * block_size,
                                             static_cast<float>(index.y()) * block_size,
                                             static_cast<float>(index.z()) * block_size);

            const nvblox::Vector3f block_max = block_min + nvblox::Vector3f::Constant(block_size);

            const bool intersects = block_max.x() > origin.x() && block_min.x() < upper.x() &&
                                    block_max.y() > origin.y() && block_min.y() < upper.y() &&
                                    block_max.z() > origin.z() && block_min.z() < upper.z();

            if (!intersects) continue;

            const auto block = layer.getBlockAtIndex(index);
            if (block) {
                selected_indices.push_back(index);
                selected_blocks.push_back(block.get());
            }
        }

        std::vector<Block> host_blocks(selected_blocks.size());

        for (std::size_t i = 0; i < selected_blocks.size(); ++i) {
            if (cudaMemcpyAsync(&host_blocks[i], selected_blocks[i], sizeof(Block),
                                cudaMemcpyDeviceToHost, cuda_stream_) != cudaSuccess) {
                return false;
            }
        }

        // One synchronization covers the bounded block snapshot. Publisher never
        // touches the live nvblox layer.
        if (cudaStreamSynchronize(cuda_stream_) != cudaSuccess) return false;

        state->origin_m = {origin.x(), origin.y(), origin.z()};
        state->column_count = SnapshotColumns;
        state->row_count = SnapshotRows;
        state->slice_count = SnapshotSlices;
        
        state->cells.assign(static_cast<std::size_t>(SnapshotColumns) * SnapshotRows * SnapshotSlices,
                            static_cast<std::uint8_t>(OccupancyCell::Unknown));

        for (std::size_t block_i = 0; block_i < host_blocks.size(); ++block_i) {
            const auto& index = selected_indices[block_i];
            const auto& block = host_blocks[block_i];

            for (int vx = 0; vx < Block::kVoxelsPerSide; ++vx) {
                for (int vy = 0; vy < Block::kVoxelsPerSide; ++vy) {
                    for (int vz = 0; vz < Block::kVoxelsPerSide; ++vz) {
                        const float x = static_cast<float>(index.x()) * block_size +
                                        (static_cast<float>(vx) + 0.5F) * VoxelSizeM;

                        const float y = static_cast<float>(index.y()) * block_size +
                                        (static_cast<float>(vy) + 0.5F) * VoxelSizeM;

                        const float z = static_cast<float>(index.z()) * block_size +
                                        (static_cast<float>(vz) + 0.5F) * VoxelSizeM;

                        const int gx = static_cast<int>(std::floor((x - origin.x()) / VoxelSizeM));
                        const int gy = static_cast<int>(std::floor((y - origin.y()) / VoxelSizeM));
                        const int gz = static_cast<int>(std::floor((z - origin.z()) / VoxelSizeM));

                        if (gx < 0 || gy < 0 || gz < 0 ||
                            gx >= static_cast<int>(SnapshotColumns) ||
                            gy >= static_cast<int>(SnapshotRows) ||
                            gz >= static_cast<int>(SnapshotSlices)) continue;

                        const std::size_t linear = (static_cast<std::size_t>(gz) * SnapshotRows +
                                                    static_cast<std::size_t>(gy)) * SnapshotColumns +
                                                    static_cast<std::size_t>(gx);

                        state->cells[linear] = classifyOccupancy(block.voxels[vx][vy][vz].log_odds);
                    }
                }
            }
        }

        return state->gridValid();
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

        if (!buildSnapshot(world_from_camera.translation(), state.get())) {
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
