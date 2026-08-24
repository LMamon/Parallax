#include <parallax/isp/isp_producer.hpp>

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
        /**
         *  The final graph path will obtain RawStereo from ProductStore, call the
         *  existing ISP::process(), and publish handles to ISP::rgb()/gray()
         *  The device buffers remain ISP-owned. Product publication must therefore
         *  preserve ISP lifetime rather than copy these frames through host memory.
         *  return parallax::core::SubmitResult::NoWork;
         */

        return parallax::core::SubmitResult::NoWork;
    }
}