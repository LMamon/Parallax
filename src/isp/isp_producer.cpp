#include <parallax/isp/isp_producer.hpp>
#include <parallax/camera/frame_types.hpp>

#include <parallax/core/execution_context.hpp>
#include <cuda_runtime.h>
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
        parallax::core::ExecutionPolicy policy{};
        policy.affinity = parallax::core::ResourceAffinity::Gpu;
        policy.stateful = false;
        return policy;
    }

    parallax::core::SubmitResult IspProducer::submit(parallax::core::ExecutionContext& context) {
        const auto raw = store_.latest<parallax::camera::RawFrame>(parallax::core::ProductId::RawStereo);

        if (!raw || !raw->valid()) {
            return parallax::core::SubmitResult::Failed;
        }

        auto output = isp_.acquireOutput();

        if (!output) {
            /**
             * Every preallocated ISP generation is still retained by a consumer.
             * Do not overwrite one and do not allocate another large frame.
             *
             * Later policy work will expose this as a supersede/backpressure
             * counter. For now NoWork preserves the bounded-memory contract.
             */
            return parallax::core::SubmitResult::NoWork;
        }

        if (!isp_.process(*raw->payload, *output)) {
            return parallax::core::SubmitResult::Failed;
        }

        /**

        * SYNCHRONIZATION INVENTORY — ORDERING ONLY
        *
        * ISP submits upload/demosaic work to its CUDA stream while rectification
        * executes through the downstream VPI stream. This host synchronization
        * currently prevents VPI from consuming incomplete ISP-owned buffers.
        *
        * ISP output remains CUDA-resident. Record completion on the ISP-owned CUDA
        * stream so downstream accelerator work can depend on these pixels without
        * blocking the host thread.
        */
        auto completion = context.recordCudaCompletion(isp_.stream());
        if (!completion.valid()) {
            return parallax::core::SubmitResult::Failed;
        }

        // these payloads remain in device storage for ISP. ProductStore receives
        // non-owning shared handles rather than copies/host downloads

        /**
         * Both published products lease the same preallocated ISP generation.
         *
         * The aliasing shared_ptr points at the requested frame member while its
         * control block owns OutputSlot. The slot therefore cannot return to the
         * ISP pool until both RGB/gray publications and all downstream consumers
         * release it.
         */
        std::shared_ptr<const parallax::isp::StereoRgbFrame> rgb(output, &output->rgb);
        std::shared_ptr<const parallax::isp::StereoGrayFrame> gray(output, &output->gray);
        std::shared_ptr<const void> raw_lifetime = raw->payload;

        store_.publish(parallax::core::make_product(parallax::core::ProductId::RgbLeft,
                                                    raw->metadata,
                                                    std::move(rgb),
                                                    completion,
                                                    raw_lifetime));

        store_.publish(parallax::core::make_product(parallax::core::ProductId::GrayStereo,
                                                    raw->metadata,
                                                    std::move(gray),
                                                    completion,
                                                    std::move(raw_lifetime)));

        return parallax::core::SubmitResult::Submitted;
    }
}