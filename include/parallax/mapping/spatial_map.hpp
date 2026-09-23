#pragma once

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
