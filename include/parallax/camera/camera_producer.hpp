#pragma once

#include <parallax/camera/frame_types.hpp>
#include <parallax/camera/stereo_camera.hpp>

#include <parallax/core/producer.hpp>
#include <parallax/core/product.hpp>
#include <parallax/core/product_store.hpp>

#include <cstdint>
#include <vector>

namespace parallax::camera {
    /**
     * StereoCamera still owns camera configuration, streaming, capture, and V4L2
     * buffer release. CameraProducer only gives that existing work a graph facing
     * product boundary
     * 
     * Runtime does not execute this producer yet, will migrate producers one at a time
     * while pipeline remains the working compatibility path. Runtime execution moves 
     * onto the producer graph after image/geometry path has been represented and verified
     */
     class CameraProducer final : public parallax::core::Producer {
        public:
            CameraProducer(StereoCamera& camera, parallax::core::ProductStore& store);
            [[nodiscard]] std::string_view name() const noexcept override;
            [[nodiscard]]
            const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]]
            const std::vector<parallax::core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;
            parallax::core::SubmitResult submit() override;

        private:
            StereoCamera& camera_;
            parallax::core::ProductStore& store_;

            std::uint64_t next_sequence_ = 0;

            //Camera capture is a graph source so there are no upstream products
            const std::vector<parallax::core::ProductId> inputs_{};
            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::RawStereo
            };
     };
}