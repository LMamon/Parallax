#pragma once

#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>
#include <parallax/core/fixed_payload_pool.hpp>
#include <parallax/isp/frame_types.hpp>

#include <cuda_runtime.h>
#include <memory>

namespace parallax::isp {

    class ISP {
        public:
            ISP();
            ~ISP();

            ISP(const ISP&) = delete;
            ISP& operator=(const ISP&) = delete;

            ISP(ISP&&) = delete;
            ISP& operator=(ISP&&) = delete;

            static constexpr std::size_t OutputSlotCount = 3;

            struct OutputSlot {
                StereoRgbFrame rgb{};
                StereoGrayFrame gray{};
            };

            bool initialize(const parallax::camera::CameraConfig& config); //maybe change interface to params instead of config
            bool process(const parallax::camera::RawFrame& input, OutputSlot& output);

            [[nodiscard]] std::shared_ptr<OutputSlot> acquireOutput() {
                return output_pool_.acquire();
            }

            bool synchronize();

            void shutdown();

            bool downloadRaw(std::uint16_t* host_data, std::size_t host_pitch) const;

            const StereoRgbFrame& rgb() const noexcept {
                return output_pool_.prototype()->rgb;
            }

            const StereoGrayFrame& gray() const noexcept {
                return output_pool_.prototype()->gray;
            }

            [[nodiscard]] cudaStream_t stream() const noexcept {
                return stream_;
            }

        private:
            bool upload(const parallax::camera::RawFrame& input);

            // Current ISP work runs on an ISP-owned CUDA stream.
            // keep stream ownership for now; do synchronization later
            cudaStream_t stream_{};

            // CUDA-owned, pitch-linear storage allocated by CudaBuffer via
            // cudaMallocPitch. Raw Bayer data is uploaded from V4L2 and
            // consumed directly by the custom CUDA demosaic/split kernel.
            GpuBayerFrame gpu_input_;

            /**
             * Three preallocated generations bound accelerator retention.
             *
             * The pool owns one baseline reference to each allocation. Published
             * products lease a slot through shared ownership. ISP never overwrites a
             * slot while any product/consumer still holds that generation.
             */
            parallax::core::FixedPayloadPool<OutputSlot, OutputSlotCount> output_pool_;

            bool initialized_ = false;
        };
}