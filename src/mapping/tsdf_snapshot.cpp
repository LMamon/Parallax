#include <parallax/mapping/tsdf_snapshot.hpp>
#include <parallax/mapping/mapping_metrics.hpp>

#include <nvblox/core/cuda_stream.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace parallax::mapping {

    std::uint8_t classifyTsdfVoxel(float distance_m,
                                   float weight,
                                   float surface_band_m) noexcept {
        if (!std::isfinite(distance_m) || !std::isfinite(weight) ||
            weight <= 0.0F || surface_band_m <= 0.0F) {
            return static_cast<std::uint8_t>(TsdfCell::Unknown);
        }

        return std::abs(distance_m) <= surface_band_m
            ? static_cast<std::uint8_t>(TsdfCell::Surface)
            : static_cast<std::uint8_t>(TsdfCell::Unknown);
    }

    bool buildTsdfSnapshot(const nvblox::TsdfLayer& layer,
                           const nvblox::Vector3f& center,
                           nvblox::CudaStream* stream,
                           const TsdfSnapshotConfig& config,
                           SpatialTsdfState* state) {
        if (!state || !stream || config.voxel_size_m <= 0.0F ||
            config.column_count == 0 ||
            config.row_count == 0 ||
            config.slice_count == 0) {
            return false;
        }

        const float sx =
            static_cast<float>(config.column_count) * config.voxel_size_m;
        const float sy =
            static_cast<float>(config.row_count) * config.voxel_size_m;
        const float sz =
            static_cast<float>(config.slice_count) * config.voxel_size_m;

        const auto snap = [&](float value) {
            return std::floor(value / config.voxel_size_m) *
                   config.voxel_size_m;
        };

        const nvblox::Vector3f origin(
            snap(center.x() - 0.5F * sx),
            snap(center.y() - 0.5F * sy),
            snap(center.z() - 0.5F * sz));

        const std::size_t cell_count =
            static_cast<std::size_t>(config.column_count) *
            static_cast<std::size_t>(config.row_count) *
            static_cast<std::size_t>(config.slice_count);

        /*
         * Query voxel centers through nvblox's public layer API.
         *
         * For device-backed layers getVoxels() performs the required
         * device-to-host copies and synchronizes the supplied CUDA stream
         * before returning.
         */
        std::vector<nvblox::Vector3f> positions;
        positions.reserve(cell_count);

        for (std::uint32_t gz = 0; gz < config.slice_count; ++gz) {
            for (std::uint32_t gy = 0; gy < config.row_count; ++gy) {
                for (std::uint32_t gx = 0; gx < config.column_count; ++gx) {
                    positions.emplace_back(
                        origin.x() +
                            (static_cast<float>(gx) + 0.5F) *
                                config.voxel_size_m,
                        origin.y() +
                            (static_cast<float>(gy) + 0.5F) *
                                config.voxel_size_m,
                        origin.z() +
                            (static_cast<float>(gz) + 0.5F) *
                                config.voxel_size_m);
                }
            }
        }

        std::vector<nvblox::TsdfVoxel> voxels;
        std::vector<bool> success_flags;

        layer.getVoxels(
            positions,
            &voxels,
            &success_flags,
            stream);

        if (voxels.size() != cell_count ||
            success_flags.size() != cell_count) {
            return false;
        }

        std::uint64_t copied_voxels=0; for(const bool ok:success_flags) if(ok) ++copied_voxels;
        if(copied_voxels){ auto& m=mapping_metrics(); m.tsdf_d2h_transfers.fetch_add(copied_voxels); m.tsdf_d2h_bytes.fetch_add(copied_voxels*sizeof(nvblox::TsdfVoxel)); }
        state->voxel_size_m = config.voxel_size_m;
        state->origin_m = {origin.x(), origin.y(), origin.z()};
        state->column_count = config.column_count;
        state->row_count = config.row_count;
        state->slice_count = config.slice_count;

        state->cells.assign(
            cell_count,
            static_cast<std::uint8_t>(TsdfCell::Unknown));

        for (std::size_t i = 0; i < cell_count; ++i) {
            if (!success_flags[i]) {
                continue;
            }

            state->cells[i] = classifyTsdfVoxel(
                voxels[i].distance,
                voxels[i].weight,
                config.surface_band_m);
        }

        return state->gridValid();
    }

}