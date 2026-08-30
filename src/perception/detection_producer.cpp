#include <parallax/perception/detection_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>

#include <chrono>
#include <memory>

namespace parallax::perception {

    DetectionProducer::DetectionProducer(NanoOwlBridge& detector, 
                                         parallax::core::ProductStore& products) noexcept 
                                         : detector_(detector),
                                         products_(products) {}

    std::string_view DetectionProducer::name() const noexcept {
        return "DetectionProducer";
    }

    const std::vector<parallax::core::ProductId>& DetectionProducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<parallax::core::ProductId>& DetectionProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy DetectionProducer::execution_policy() const noexcept {
        parallax::core::ExecutionPolicy policy{};

        /*
         * NanoOWL is deliberately slower than the camera path.
         *
         * ProductStore remains latest-value storage, so skipped camera
         * generations do not accumulate into a detector queue.
         */
        policy.target_hz = 5.0;
        policy.max_input_age_ms = 250.0;
        policy.drop_policy = parallax::core::DropPolicy::Supersede;
        policy.priority = 0;
        policy.affinity = parallax::core::ResourceAffinity::Gpu;
        policy.stateful = false;

        return policy;
    }

    bool DetectionProducer::setQuery(const std::string& query, std::uint64_t revision) {

        if (query.empty() || revision == 0) {
            return false;
        }

        if (query == query_ && revision == query_revision_) {
            return true;
        }

        if (!detector_.setQuery(query, revision)) {
            return false;
        }

        query_ = query;
        query_revision_ = revision;

        return true;
    }

    parallax::core::SubmitResult DetectionProducer::submit(parallax::core::ExecutionContext& context) {

        if (query_.empty() || query_revision_ == 0) {
            return parallax::core::SubmitResult::NoWork;
        }

        auto input = products_.latest<parallax::isp::StereoRgbFrame>(parallax::core::ProductId::RgbLeft);

        if (!input || !input->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }

        auto stream = context.neuralCudaLane();

        /*
         * Depend on the exact RGB generation being inferred.
         *
         * This inserts accelerator ordering onto the neural CUDA lane rather
         * than synchronizing the host.
         */
        if (!context.waitFor(input->completion, stream)) {
            return parallax::core::SubmitResult::Failed;
        }

        auto detections = std::make_shared<DetectionSet>();

        if (!detector_.predict(*input->payload, stream, *detections)) {
            return parallax::core::SubmitResult::Failed;
        }

        /*
         * Reject a result whose query identity no longer matches the producer's
         * active query. This becomes important once query replacement is wired
         * from RequestController.
         */
        if (detections->query_revision != query_revision_ || detections->query != query_) {

            return parallax::core::SubmitResult::NoWork;
        }

        auto metadata = input->metadata;
        metadata.production_timestamp = std::chrono::steady_clock::now();

        /*
         * DetectionSet contains compact host-visible metadata. Its source
         * observation remains the RGB generation actually inferred, regardless
         * of how many newer camera observations now exist in ProductStore.
         */
        products_.publish(parallax::core::make_product(parallax::core::ProductId::Detection,
                                                       metadata,
                                                       std::move(detections),
                                                       parallax::core::CompletionHandle::cpu_ready()));

        return parallax::core::SubmitResult::Submitted;
    }

}