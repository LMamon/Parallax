#include <parallax/mapping/occupancy_snapshot.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace parallax::mapping {
    namespace {
        constexpr float OccupancyEpsilon = 1.0e-6F;
    }

    std::uint8_t classifyOccupancyLogOdds(float log_odds) noexcept {
        if (log_odds > OccupancyEpsilon) {
            return static_cast<std::uint8_t>(OccupancyCell::Occupied);
        }
        if (log_odds < -OccupancyEpsilon) {
            return static_cast<std::uint8_t>(OccupancyCell::Free);
        }
        return static_cast<std::uint8_t>(OccupancyCell::Unknown);
    }

    bool buildOccupancySnapshot(const nvblox::OccupancyLayer& layer,
                                const nvblox::Vector3f& center,
                                cudaStream_t stream,
                                const OccupancySnapshotConfig& config,
                                LocalOccupancyState* state) {

        if (state == nullptr || stream == nullptr ||
            !(config.voxel_size_m > 0.0F) ||
            config.column_count == 0 || config.row_count == 0 || config.slice_count == 0) {
            return false;
        }

        using Block = nvblox::VoxelBlock<nvblox::OccupancyVoxel>;

        const float size_x = static_cast<float>(config.column_count) * config.voxel_size_m;
        const float size_y = static_cast<float>(config.row_count) * config.voxel_size_m;
        const float size_z = static_cast<float>(config.slice_count) * config.voxel_size_m;

        const auto snap = [&config](float value) {
            return std::floor(value / config.voxel_size_m) * config.voxel_size_m;
        };

        const nvblox::Vector3f origin(snap(center.x() - 0.5F * size_x),
                                      snap(center.y() - 0.5F * size_y),
                                      snap(center.z() - 0.5F * size_z));
        const nvblox::Vector3f upper(origin.x() + size_x,
                                     origin.y() + size_y,
                                     origin.z() + size_z);

        const float block_size = layer.block_size();
        const auto indices = layer.getAllBlockIndices();
        std::vector<nvblox::Index3D> selected_indices;
        std::vector<const Block*> selected_blocks;

        for (const auto& index : indices) {
            const nvblox::Vector3f block_min(static_cast<float>(index.x()) * block_size,
                                             static_cast<float>(index.y()) * block_size,
                                             static_cast<float>(index.z()) * block_size);
            const nvblox::Vector3f block_max =
                block_min + nvblox::Vector3f::Constant(block_size);

            const bool intersects = block_max.x() > origin.x() && block_min.x() < upper.x() &&
                                    block_max.y() > origin.y() && block_min.y() < upper.y() &&
                                    block_max.z() > origin.z() && block_min.z() < upper.z();
            if (!intersects) continue;

            const auto block = layer.getBlockAtIndex(index);
            if (!block) continue;

            selected_indices.push_back(index);
            selected_blocks.push_back(block.get());
        }

        std::vector<Block> host_blocks(selected_blocks.size());
        for (std::size_t i = 0; i < selected_blocks.size(); ++i) {
            if (cudaMemcpyAsync(&host_blocks[i], selected_blocks[i], sizeof(Block),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
                return false;
            }
        }

        // One synchronization covers the bounded block snapshot. The completed
        // product owns CPU data; downstream consumers never touch the live layer.
        if (cudaStreamSynchronize(stream) != cudaSuccess) return false;

        state->voxel_size_m = config.voxel_size_m;
        state->origin_m = {origin.x(), origin.y(), origin.z()};
        state->column_count = config.column_count;
        state->row_count = config.row_count;
        state->slice_count = config.slice_count;
        state->cells.assign(static_cast<std::size_t>(config.column_count) *
                                config.row_count * config.slice_count,
                            static_cast<std::uint8_t>(OccupancyCell::Unknown));

        for (std::size_t block_i = 0; block_i < host_blocks.size(); ++block_i) {
            const auto& index = selected_indices[block_i];
            const auto& block = host_blocks[block_i];

            for (int vx = 0; vx < Block::kVoxelsPerSide; ++vx) {
                for (int vy = 0; vy < Block::kVoxelsPerSide; ++vy) {
                    for (int vz = 0; vz < Block::kVoxelsPerSide; ++vz) {
                        const float x = static_cast<float>(index.x()) * block_size +
                                        (static_cast<float>(vx) + 0.5F) * config.voxel_size_m;
                        const float y = static_cast<float>(index.y()) * block_size +
                                        (static_cast<float>(vy) + 0.5F) * config.voxel_size_m;
                        const float z = static_cast<float>(index.z()) * block_size +
                                        (static_cast<float>(vz) + 0.5F) * config.voxel_size_m;

                        const int gx = static_cast<int>(std::floor((x - origin.x()) / config.voxel_size_m));
                        const int gy = static_cast<int>(std::floor((y - origin.y()) / config.voxel_size_m));
                        const int gz = static_cast<int>(std::floor((z - origin.z()) / config.voxel_size_m));

                        if (gx < 0 || gy < 0 || gz < 0 ||
                            gx >= static_cast<int>(config.column_count) ||
                            gy >= static_cast<int>(config.row_count) ||
                            gz >= static_cast<int>(config.slice_count)) continue;

                        const std::size_t linear =
                            (static_cast<std::size_t>(gz) * config.row_count +
                             static_cast<std::size_t>(gy)) * config.column_count +
                            static_cast<std::size_t>(gx);

                        state->cells[linear] =
                            classifyOccupancyLogOdds(block.voxels[vx][vy][vz].log_odds);
                    }
                }
            }
        }

        return state->gridValid();
    }

}
