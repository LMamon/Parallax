#include <parallax/isp/isp_producer.hpp>
#include <parallax/camera/frame_types.hpp>

#include <memory>


namespace parallax::isp {
    IspProducer::IspProducer(ISP& isp, parallax::core::ProductStore& store) : 
                             isp_(isp),
                             store_(store) {}

    std::string_view IspProducer::name() const noexcept {
        return "isp";
    }

    const std::vector<parallax::core::ProductId>& IspProducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<parallax::core::ProductId>& IspProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy IspProducer::execution_policy() const noexcept {
        // keeping existing implementation rather than changing streams or memory
        // ownership while graph boundaries are established.
        return {parallax::core::ResourceAffinity::Gpu, false};
    }

    parallax::core::SubmitResult IspProducer::submit() {
        const auto raw = store_.latest<parallax::camera::RawFrame>(parallax::core::ProductId::RawStereo);

        if (!raw || !raw->valid()) {
            return parallax::core::SubmitResult::Failed;
        }

        if (!isp_.process(*raw->payload)) {
            return parallax::core::SubmitResult::Failed;
        }

        /**
         * ISP owns a CUDA stream separate from the downstream VPI stream.
         * keep existing synchronization boundary before publishing ISP
         * outputs as ready for downstream graph consumers
         */
        if(!isp_.synchronize()) {
            return parallax::core::SubmitResult::Failed;
        }

        // these payloads remain in device storage for ISP. ProductStore receives
        // non-owning shared handles rather than copies/host downloads

        auto rgb = std::shared_ptr<const parallax::isp::StereoRgbFrame>(&isp_.rgb(),
                                                                        [](const parallax::isp::StereoRgbFrame*)
                                                                        {});

        auto gray = std::shared_ptr<const parallax::isp::StereoGrayFrame>(&isp_.gray(),
                                                                        [](const parallax::isp::StereoGrayFrame*)
                                                                        {});

        store_.publish(parallax::core::make_product(parallax::core::ProductId::RgbLeft,
                                                    raw->metadata,
                                                    std::move(rgb)));

        store_.publish(parallax::core::make_product(parallax::core::ProductId::GrayStereo,
                                                    raw->metadata,
                                                    std::move(gray)));

        return parallax::core::SubmitResult::Submitted;
    }
}