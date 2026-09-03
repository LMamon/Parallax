#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/perception/efficientvit_sam.hpp>

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace parallax::perception {

    class SegmentationProducer final : public parallax::core::Producer {
        public:
            SegmentationProducer(EfficientVitSam& segmenter,
                                 parallax::core::ProductStore& products,
                                 std::filesystem::path encoder_engine,
                                 std::filesystem::path decoder_engine) noexcept;

            [[nodiscard]] std::string_view name() const noexcept override;

            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;
            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;

            parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

        private:
            EfficientVitSam& segmenter_;
            parallax::core::ProductStore& products_;

            std::filesystem::path encoder_engine_;
            std::filesystem::path decoder_engine_;

            std::uint64_t last_query_revision_ = 0;
            parallax::core::SourceObservation last_observation_{};

            const std::vector<parallax::core::ProductId> inputs_{
                parallax::core::ProductId::Detection,
                parallax::core::ProductId::RgbLeft
            };

            const std::vector<parallax::core::ProductId> outputs_{parallax::core::ProductId::Segmentation};
            const std::vector<parallax::core::CompatibleInputRequirement> compatible_inputs_{
                                                                    {parallax::core::ProductId::RgbLeft, 2}
            };
    };
}