#include <parallax/stereo/depth_producer.hpp>
#include <parallax/cuda/depth.cuh>
#include <parallax/core/execution_context.hpp>

#include <memory>

namespace parallax::stereo {
    DepthProducer::DepthProducer(const StereoCalibration& calibration,
                                 parallax::core::ProductStore& store) :
                                 calibration_(calibration),
                                 store_(store) {}

    std::string_view DepthProducer::name() const noexcept {
        return "stereo.depth";
    }

    const std::vector<parallax::core::ProductId>& DepthProducer::inputs() const noexcept {
        return inputs_;
    }
    
    const std::vector<parallax::core::ProductId>& DepthProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy DepthProducer::execution_policy() const noexcept {
        parallax::core::ExecutionPolicy policy{};
        policy.drop_policy = parallax::core::DropPolicy::Supersede;
        policy.affinity = parallax::core::ResourceAffinity::Gpu;
        policy.stateful = false;
        return policy;
    }

    parallax::core::SubmitResult DepthProducer::submit(parallax::core::ExecutionContext& context) {
        (void)context;
        const auto disparity = store_.latest<parallax::isp::StereoMatchFrame>(parallax::core::ProductId::Disparity);

        if (!disparity || !disparity->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }
        
        if (!output_pool_initialized_) {
            const auto width = disparity->payload->width;
            const auto height = disparity->payload->height;

            if (width == 0 || height == 0) {
                return parallax::core::SubmitResult::Failed;
            }

            if (!output_pool_.initialize([&](parallax::isp::DepthFrame& depth, std::size_t index) {
                        depth.width = width;
                        depth.height = height;
                        depth.storage_slot = static_cast<std::uint32_t>(index);

                        return depth.depth.allocate(width, height, 1, sizeof(float));
                    })) {

                return parallax::core::SubmitResult::Failed;
            }

            output_pool_initialized_ = true;
        }

        auto& lane = context.stereoLane();
        if (!context.waitFor(disparity->completion, lane)) {
            return parallax::core::SubmitResult::Failed;
        }

        auto depth = output_pool_.acquire();
        if (!depth) {
            /**
             * Every preallocated depth generation is still retained.
             * Never overwrite retained depth and never allocate another large
             * device buffer in steady state.
             */
            return parallax::core::SubmitResult::NoWork;
        }

        if (!parallax::cuda::disparityToDepth(disparity->payload->disparity,
                                              depth->depth,
                                              static_cast<float>(calibration_.metadata().virtual_fx),
                                              static_cast<float>(calibration_.metadata().baseline_mm / 1000.0),
                                              parallax::isp::StereoMatchFrame::DisparityScale,
                                              lane.cudaHandle())) {

            return parallax::core::SubmitResult::Failed;
        }
        /**
         * Depth remains device-resident when published. Record completion on the
         * CUDA stream that produced it so CPU consumers can wait only when they
         * actually need to observe depth values.
         */
        auto completion = context.recordCudaCompletion(lane.cudaHandle());
        if (!completion.valid()) {
            return parallax::core::SubmitResult::Failed;
        }

        std::shared_ptr<const void> input_lifetime = disparity->payload;
        store_.publish(parallax::core::make_product(parallax::core::ProductId::Depth,
                                                    disparity->metadata,
                                                    std::shared_ptr<const parallax::isp::DepthFrame>(
                                                        std::move(depth)),
                                                        std::move(completion),
                                                        std::move(input_lifetime)));

        return parallax::core::SubmitResult::Submitted;
    }
}