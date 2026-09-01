#include <parallax/perception/detection_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>

#include <chrono>
#include <memory>
#include <utility>
#include <iostream>

namespace parallax::perception {

    DetectionProducer::DetectionProducer(NanoOwlBridge& detector, 
                                         parallax::core::ProductStore& products) noexcept 
                                         : detector_(detector),
                                         products_(products) {}

    std::string_view DetectionProducer::name() const noexcept {
        return "detection";
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
        policy.target_hz = 10.0;
        policy.max_input_age_ms = 250.0;
        policy.drop_policy = parallax::core::DropPolicy::Supersede;
        policy.priority = 0;
        policy.affinity = parallax::core::ResourceAffinity::Gpu;
        policy.stateful = false;

        return policy;
    }

    bool DetectionProducer::setQuery(const std::string& query, std::uint64_t revision) {
        if (query.empty() || revision == 0) {
            std::cerr << "[Detection] rejected invalid query\n";
            return false;
        }

        if (query == query_ && revision == query_revision_) return true;

        std::cerr << "[Detection] applying query=\"" << query
                  << "\" revision=" << revision << '\n';

        if (!detector_.setQuery(query, revision)) {
            std::cerr << "[Detection] NanoOWL setQuery failed\n";
            return false;
        }

        query_ = query;
        query_revision_ = revision;

        std::cerr << "[Detection] query ready\n";
        return true;
    }

    parallax::core::SubmitResult DetectionProducer::submit(parallax::core::ExecutionContext& context) {
        std::cerr << "[Detection] submit called query=\"" << query_
                  << "\" revision=" << query_revision_ << '\n';

        if (query_.empty() || query_revision_ == 0) {
            std::cerr << "[Detection] NoWork: query not configured\n";
            return parallax::core::SubmitResult::NoWork;
        }

        auto input = products_.latest<parallax::isp::StereoRgbFrame>(parallax::core::ProductId::RgbLeft);
        if (!input || !input->valid()) {
            std::cerr << "[Detection] NoWork: no valid RgbLeft product\n";
            return parallax::core::SubmitResult::NoWork;
        }

        std::cerr << "[Detection] input observation=" << input->metadata.observation.sequence << '\n';

        const cudaStream_t stream = context.neuralCudaLane();
        if (stream == nullptr) {
            std::cerr << "[Detection] Failed: neural CUDA lane unavailable\n";
            return parallax::core::SubmitResult::Failed;
        }
        /*
         * Depend on the exact RGB generation being inferred.
         *
         * This inserts accelerator ordering onto the neural CUDA lane rather
         * than synchronizing the host.
         */
        if (!context.waitFor(input->completion, stream)) {
            std::cerr << "[Detection] Failed: RGB completion wait rejected\n";
            return parallax::core::SubmitResult::Failed;
        }

        auto detections = std::make_shared<DetectionSet>();

        std::cerr << "[Detection] running NanoOWL\n";
        if (!detector_.predict(*input->payload, stream, *detections)) {
            std::cerr << "[Detection] Failed: NanoOWL predict returned false\n";
            return parallax::core::SubmitResult::Failed;
        }

        std::cerr << "[Detection] NanoOWL returned"
                  << " query=\"" << detections->query
                  << "\" revision=" << detections->query_revision
                  << " count=" << detections->size() << '\n';
        /*
         * Reject a result whose query identity no longer matches the producer's
         * active query. This becomes important once query replacement is wired
         * from RequestController.
         */
        if (detections->query_revision != query_revision_ || detections->query != query_) {
            std::cerr << "[Detection] NoWork: result query was superseded\n";
            return parallax::core::SubmitResult::NoWork;
        }

        auto metadata = input->metadata;
        metadata.production_timestamp = std::chrono::steady_clock::now();

        /*
         * DetectionSet contains compact host-visible metadata. Its source
         * observation remains the RGB generation actually inferred, regardless
         * of how many newer camera observations now exist in ProductStore.
         */
        products_.publish(parallax::core::make_product<DetectionSet>(parallax::core::ProductId::Detection,
                                                       metadata,
                                                       std::move(detections),
                                                       parallax::core::CompletionHandle::cpu_ready()));

        std::cerr << "[Detection] published Detection product observation="
                  << metadata.observation.sequence << '\n';

        return parallax::core::SubmitResult::Submitted;
    }
}