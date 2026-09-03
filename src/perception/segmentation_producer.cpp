#include <parallax/perception/segmentation_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/segmentation.hpp>
#include <parallax/perception/segmentation_compatibility.hpp>

#include <chrono>
#include <memory>
#include <utility>

namespace parallax::perception {
    SegmentationProducer::SegmentationProducer(EfficientVitSam& segmenter,
                                               parallax::core::ProductStore& products,
                                               std::filesystem::path encoder_engine,
                                               std::filesystem::path decoder_engine) noexcept : segmenter_(segmenter),
                                               products_(products),
                                               encoder_engine_(std::move(encoder_engine)),
                                               decoder_engine_(std::move(decoder_engine)) {}


    std::string_view SegmentationProducer::name() const noexcept {
        return "segmentation";
    }

    const std::vector<parallax::core::ProductId>& SegmentationProducer::inputs() const noexcept {
        return inputs_;
    }


    const std::vector<parallax::core::ProductId>& SegmentationProducer::outputs() const noexcept {
        return outputs_;
    }


    const std::vector<parallax::core::CompatibleInputRequirement>& SegmentationProducer::compatible_inputs() const noexcept {
        return compatible_inputs_;
    }


    parallax::core::ExecutionPolicy SegmentationProducer::execution_policy() const noexcept {
        parallax::core::ExecutionPolicy policy{};

        // Segmentation follows explicit demand, not camera cadence.
        // Newer prompts supersede obsolete pending work.
        policy.target_hz = 0.0;
        policy.max_input_age_ms = 0.0;
        policy.drop_policy = parallax::core::DropPolicy::Supersede;
        policy.priority = 0;
        policy.affinity = parallax::core::ResourceAffinity::Gpu;
        policy.stateful = false;

        return policy;
    }

    parallax::core::SubmitResult SegmentationProducer::submit(parallax::core::ExecutionContext& context) {
        const auto detection = products_.latest<DetectionSet>(parallax::core::ProductId::Detection);

        if (!detection || !detection->valid() || !detection->payload || detection->payload->empty()) {
            return parallax::core::SubmitResult::NoWork;
        }

        // One segmentation result is produced for the newest detection prompt.
        // Re-running the same observation/query adds no useful information.
        if (detection->metadata.observation == last_observation_ &&
            detection->payload->query_revision == last_query_revision_) {

            return parallax::core::SubmitResult::NoWork;
        }

        const auto rgb = find_segmentation_rgb(products_, *detection);

        if (!rgb) return parallax::core::SubmitResult::NoWork;

        const cudaStream_t stream = context.neuralCudaLane();
        if (stream == nullptr) return parallax::core::SubmitResult::Failed;

        // The exact retained RGB generation may still have accelerator work
        // completing. Order this neural submission without blocking the host
        if (!context.waitFor(rgb->completion, stream)) {
            return parallax::core::SubmitResult::Failed;
        }

        if (!segmenter_.initialized()) {
            if (!segmenter_.initialize(encoder_engine_, decoder_engine_)) {
                return parallax::core::SubmitResult::Failed;
            }
        }

        // Phase 13 currently segments the highest-confidence detection box.
        // Multi-mask selection remains a downstream policy rather than hidden
        // inside the EfficientViT-SAM adapter.
        std::size_t selected = 0;

        for (std::size_t i = 1; i < detection->payload->scores.size(); ++i) {
            if (detection->payload->scores[i] > detection->payload->scores[selected]) {
                selected = i;
            }
        }

        EfficientVitSamResult result{};

        if (!segmenter_.segment(*rgb->payload,
                                detection->payload->boxes[selected],
                                stream,
                                result)) {

            return parallax::core::SubmitResult::Failed;
        }

        const auto completion = context.recordCudaCompletion(stream);

        if (!completion.valid()) return parallax::core::SubmitResult::Failed;

        auto mask = std::make_shared<SegmentationMask>();

        mask->source_observation = detection->metadata.observation;
        mask->image_space = detection->payload->image_space;
        mask->query_revision = detection->payload->query_revision;
        mask->width = result.width;
        mask->height = result.height;
        mask->pitch_bytes = result.pitch_bytes;
        mask->layout = MaskLayout::RowMajor;
        mask->representation = MaskRepresentation::CudaDevice;
        mask->confidence = result.confidence;
        mask->mask_valid = result.valid();
        mask->storage = std::move(result.storage);

        if (!mask->valid()) return parallax::core::SubmitResult::Failed;

        auto metadata = detection->metadata;

        metadata.production_timestamp = std::chrono::steady_clock::now();
        products_.publish(parallax::core::make_product<SegmentationMask>(parallax::core::ProductId::Segmentation,
                                                                         metadata,
                                                                         std::move(mask),
                                                                         completion));

        last_observation_ = detection->metadata.observation;
        last_query_revision_ = detection->payload->query_revision;

        return parallax::core::SubmitResult::Submitted;
    }
}