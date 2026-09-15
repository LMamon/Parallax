#pragma once
#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>
#include <parallax/core/fixed_payload_pool.hpp>
#include <parallax/isp/auto_control.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/isp/isp_config.hpp>
#include <parallax/isp/isp_kernels.cuh>
#include <cuda_runtime.h>
#include <nppdefs.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace parallax::isp {
class ISP {
public:
    ISP();
    ~ISP();
    ISP(const ISP&) = delete;
    ISP& operator=(const ISP&) = delete;
    ISP(ISP&&) = delete;
    ISP& operator=(ISP&&) = delete;
    static constexpr std::size_t OutputSlotCount = 9;
    struct OutputSlot { StereoRgbFrame rgb{}; StereoGrayFrame gray{}; };

    bool initialize(const parallax::camera::CameraConfig& camera_config, const IspConfig& isp_config);
    bool process(const parallax::camera::RawFrame& input, OutputSlot& output);
    [[nodiscard]] std::shared_ptr<OutputSlot> acquireOutput() { return output_pool_.acquire(); }
    bool synchronize();
    void shutdown();
    bool downloadRaw(std::uint16_t* host_data, std::size_t host_pitch) const;
    bool tryGetStatistics(IspStatistics& statistics);
    void setWhiteBalance(const std::array<float, 3>& gains);
    const StereoRgbFrame& rgb() const noexcept { return output_pool_.prototype()->rgb; }
    const StereoGrayFrame& gray() const noexcept { return output_pool_.prototype()->gray; }
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

private:
    bool upload(const parallax::camera::RawFrame& input);
    bool demosaic(const parallax::cuda::CudaBuffer& bayer, parallax::cuda::CudaBuffer& rgb16);
    bool maybeEnqueueStatistics();
    CanonicalRgbParameters colorParametersSnapshot() const;

    cudaStream_t stream_{};
    NppStreamContext npp_context_{};
    GpuBayerFrame gpu_input_{};
    parallax::cuda::CudaBuffer left_bayer16_;
    parallax::cuda::CudaBuffer right_bayer16_;
    parallax::cuda::CudaBuffer left_linear_rgb16_;
    parallax::cuda::CudaBuffer right_linear_rgb16_;
    parallax::core::FixedPayloadPool<OutputSlot, OutputSlotCount> output_pool_;
    IspConfig config_{};
    float linear_white_level_ = 1.0F;
    std::uint64_t frame_counter_ = 0;
    std::uint32_t statistics_interval_frames_ = 1;
    mutable std::mutex color_mutex_;
    std::array<float, 3> white_balance_{1.0F, 1.0F, 1.0F};
    mutable std::mutex statistics_mutex_;
    DeviceIspStatistics* device_statistics_ = nullptr;
    DeviceIspStatistics* host_statistics_ = nullptr;
    cudaEvent_t statistics_event_ = nullptr;
    bool statistics_pending_ = false;
    std::uint64_t pending_statistics_sequence_ = 0;
    bool initialized_ = false;
};
} // namespace parallax::isp
