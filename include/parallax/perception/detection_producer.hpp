#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/perception/nanoowl_bridge.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace parallax::perception {

    class DetectionProducer final : public parallax::core::Producer {
        public:
            DetectionProducer(NanoOwlBridge& detector, parallax::core::ProductStore& products) noexcept;

            [[nodiscard]] std::string_view name() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;

            /**
             * Replace the active query without changing graph demand.
             *
             * RequestController owns application intent and demand.
             * DetectionProducer owns only the query state required to execute
             * the currently requested detector work.
             */
            bool setQuery(const std::string& query, std::uint64_t revision);

            parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

        private:
            NanoOwlBridge& detector_;
            parallax::core::ProductStore& products_;

            std::string query_;
            std::uint64_t query_revision_ = 0;

            const std::vector<parallax::core::ProductId> inputs_{parallax::core::ProductId::RgbLeft};
            const std::vector<parallax::core::ProductId> outputs_{parallax::core::ProductId::Detection};
    };
}