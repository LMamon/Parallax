#pragma once

#include <vpi/Stream.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace parallax::vpi {

    class Stream {
        public:
            Stream() = default;
            ~Stream();

            Stream(const Stream&) = delete;
            Stream& operator=(const Stream&) = delete;

            Stream(Stream&& other) noexcept;
            Stream& operator=(Stream&& other) noexcept;

            bool initialize(std::uint64_t backends);
            bool synchronize();

            void shutdown() noexcept;

            [[nodiscard]] bool initialized() const noexcept {
                return stream_ != nullptr;
            }

            [[nodiscard]] VPIStream handle() const noexcept { return stream_; }
            [[nodiscard]] cudaStream_t cudaHandle() const noexcept { return cuda_stream_; }

        private:
            VPIStream stream_ = nullptr;
            cudaStream_t cuda_stream_ = nullptr;
        };

}