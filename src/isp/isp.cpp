#include <parallax/isp/isp.hpp>
#include <nppcore.h>
#include <nppi_color_conversion.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>

namespace parallax::isp {
    namespace { bool pitchFitsNpp(std::size_t pitch) { return pitch <= static_cast<std::size_t>(INT_MAX); } }

    ISP::ISP() = default;
    ISP::~ISP() { shutdown(); }

    bool ISP::initialize(const parallax::camera::CameraConfig& camera_config, const IspConfig& isp_config) {
        if (initialized_) return true;
        
        if (!isp_config.enable || camera_config.bayer_pattern != parallax::camera::BayerPattern::GRBG ||
            camera_config.width <= 0 || camera_config.height <= 0 || (camera_config.width % 2) != 0) return false;

        config_ = isp_config;
        white_balance_ = config_.white_balance;
        linear_white_level_ = static_cast<float>(1023U - config_.black_level);

        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) { 
            shutdown(); 
            return false; 
        }
        
        if (nppGetStreamContext(&npp_context_) != NPP_SUCCESS) { 
            shutdown(); 
            return false; 
        }
        npp_context_.hStream = stream_;

        if (!gpu_input_.buffer.allocate(camera_config.width, camera_config.height, 1, sizeof(std::uint16_t))) { 
            shutdown(); 
            return false; 
        }
        gpu_input_.width = camera_config.width;
        gpu_input_.height = camera_config.height;
        gpu_input_.pattern = camera_config.bayer_pattern;

        const auto eye_width = static_cast<std::uint32_t>(camera_config.width / 2);
        const auto height = static_cast<std::uint32_t>(camera_config.height);
        
        if (!left_bayer16_.allocate(eye_width, height, 1, sizeof(std::uint16_t)) ||
            !right_bayer16_.allocate(eye_width, height, 1, sizeof(std::uint16_t)) ||
            !left_linear_rgb16_.allocate(eye_width, height, 3, sizeof(std::uint16_t)) ||
            !right_linear_rgb16_.allocate(eye_width, height, 3, sizeof(std::uint16_t))) { 

                shutdown(); 
                return false; 
            }

        if (!pitchFitsNpp(left_bayer16_.pitch()) || !pitchFitsNpp(right_bayer16_.pitch()) ||
            !pitchFitsNpp(left_linear_rgb16_.pitch()) || !pitchFitsNpp(right_linear_rgb16_.pitch())) { 
                
                shutdown(); 
                return false; 
            }

        if (!output_pool_.initialize([&](OutputSlot& slot, std::size_t index) {

            slot.rgb.width = eye_width; 
            slot.rgb.height = height; 
            slot.rgb.storage_slot = static_cast<std::uint32_t>(index);
            slot.gray.width = eye_width; 
            slot.gray.height = height; 
            slot.gray.storage_slot = static_cast<std::uint32_t>(index);
            
            return slot.rgb.left.allocate(eye_width, height, 3, sizeof(std::uint8_t)) &&
                slot.rgb.right.allocate(eye_width, height, 3, sizeof(std::uint8_t)) &&
                slot.gray.left.allocate(eye_width, height, 1, sizeof(std::uint8_t)) &&
                slot.gray.right.allocate(eye_width, height, 1, sizeof(std::uint8_t));
        })) { 
            shutdown(); 
            return false; 
        }

        if (cudaMalloc(reinterpret_cast<void**>(&device_statistics_), sizeof(DeviceIspStatistics)) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void**>(&host_statistics_), sizeof(DeviceIspStatistics)) != cudaSuccess ||
            cudaEventCreateWithFlags(&statistics_event_, cudaEventDisableTiming) != cudaSuccess) { shutdown(); return false; }

        const float source_fps = static_cast<float>(std::max(camera_config.frame_rate, 1));
        statistics_interval_frames_ = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(
                                                              std::lround(source_fps / config_.statistics.update_hz)));

        initialized_ = true;
        return true;
    }

    bool ISP::downloadRaw(std::uint16_t* host_data, std::size_t host_pitch) const {
        return initialized_ && host_data && gpu_input_.buffer.downloadAsync(host_data, host_pitch, stream_);
    }

    bool ISP::upload(const parallax::camera::RawFrame& input) {
        if (input.width != gpu_input_.width || input.height != gpu_input_.height) return false;

        return gpu_input_.buffer.uploadAsync(input.data, input.width * sizeof(std::uint16_t), stream_);
    }

    bool ISP::demosaic(const parallax::cuda::CudaBuffer& bayer, parallax::cuda::CudaBuffer& rgb16) {
        const NppiSize size{static_cast<int>(bayer.width()), static_cast<int>(bayer.height())};
        const NppiRect roi{0, 0, static_cast<int>(bayer.width()), static_cast<int>(bayer.height())};
        
        const NppStatus status = nppiCFAToRGB_16u_C1C3R_Ctx(bayer.dataAs<Npp16u>(), 
                                                            static_cast<int>(bayer.pitch()), 
                                                            size, 
                                                            roi,
                                                            rgb16.dataAs<Npp16u>(), 
                                                            static_cast<int>(rgb16.pitch()),
                                                            NPPI_BAYER_GRBG, 
                                                            NPPI_INTER_UNDEFINED, 
                                                            npp_context_);

        if (status != NPP_SUCCESS) std::cerr << "ISP: NPP demosaic failed: " << status << '\n';

        return status == NPP_SUCCESS;
    }

    CanonicalRgbParameters ISP::colorParametersSnapshot() const {
        std::lock_guard<std::mutex> lock(color_mutex_);
        CanonicalRgbParameters p{};
        
        for (int i = 0; i < 3; ++i) p.white_balance[i] = white_balance_[i];
        for (int i = 0; i < 9; ++i) p.color_matrix[i] = config_.color_matrix[i];
        
        p.inverse_gamma = 1.0F / config_.gamma;
        p.linear_white_level = linear_white_level_;
        
        return p;
    }

    bool ISP::maybeEnqueueStatistics() {
        ++frame_counter_;
        
        if ((frame_counter_ % statistics_interval_frames_) != 0U) return true;

        std::lock_guard<std::mutex> lock(statistics_mutex_);
        if (statistics_pending_) return true;
        
        if (cudaMemsetAsync(device_statistics_, 0, sizeof(DeviceIspStatistics), stream_) != cudaSuccess) return false;
        
        if (!collectIspStatistics(left_linear_rgb16_, device_statistics_, linear_white_level_, config_.statistics.sample_stride, stream_)) return false;
        
        if (cudaMemcpyAsync(host_statistics_, device_statistics_, sizeof(DeviceIspStatistics), cudaMemcpyDeviceToHost, stream_) != cudaSuccess) return false;
        
        if (cudaEventRecord(statistics_event_, stream_) != cudaSuccess) return false;
        pending_statistics_sequence_ = frame_counter_;
        statistics_pending_ = true;
        
        return true;
    }

    bool ISP::process(const parallax::camera::RawFrame& input, OutputSlot& output) {
        if (!initialized_ || !upload(input)) return false;
        
        if (!prepareStereoBayer(gpu_input_, left_bayer16_, right_bayer16_, config_.black_level, stream_)) return false;
        if (!demosaic(left_bayer16_, left_linear_rgb16_) || !demosaic(right_bayer16_, right_linear_rgb16_)) return false;
        
        const auto parameters = colorParametersSnapshot();
        if (!formCanonicalStereo(left_linear_rgb16_, right_linear_rgb16_, output.rgb, output.gray, parameters, stream_)) return false;
        
        return maybeEnqueueStatistics();
    }

    bool ISP::tryGetStatistics(IspStatistics& statistics) {
        if (!initialized_) return false;
        
        std::lock_guard<std::mutex> lock(statistics_mutex_);
        if (!statistics_pending_) return false;
        
        const auto status = cudaEventQuery(statistics_event_);
        if (status == cudaErrorNotReady) return false;
        
        if (status != cudaSuccess) { 
            statistics_pending_ = false;
            return false; 
        }
        
        for (std::size_t i = 0; i < 256; ++i) {
            statistics.luminance_histogram[i] = host_statistics_->luminance_histogram[i];
            statistics.red_sum = host_statistics_->red_sum;
            statistics.green_sum = host_statistics_->green_sum;
            statistics.blue_sum = host_statistics_->blue_sum;
            statistics.color_samples = host_statistics_->color_samples;
            
            statistics.total_samples = host_statistics_->total_samples;
            statistics.saturated_samples = host_statistics_->saturated_samples;
            statistics.sequence = pending_statistics_sequence_;
            statistics_pending_ = false;
        }
        
        return statistics.valid();
    }

    void ISP::setWhiteBalance(const std::array<float, 3>& gains) {
        std::lock_guard<std::mutex> lock(color_mutex_);
        white_balance_ = gains;
    }

    bool ISP::synchronize() { return initialized_ && stream_ && cudaStreamSynchronize(stream_) == cudaSuccess; }

    void ISP::shutdown() {
        if (stream_) cudaStreamSynchronize(stream_);
        
        if (statistics_event_) { cudaEventDestroy(statistics_event_); statistics_event_ = nullptr; }
        if (host_statistics_) { cudaFreeHost(host_statistics_); host_statistics_ = nullptr; }
        
        if (device_statistics_) { cudaFree(device_statistics_); device_statistics_ = nullptr; }
        
        left_bayer16_.release(); right_bayer16_.release(); left_linear_rgb16_.release(); right_linear_rgb16_.release();
        gpu_input_.buffer.release(); output_pool_.reset();
        
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
     
        statistics_pending_ = false; frame_counter_ = 0; initialized_ = false;
    }
}
