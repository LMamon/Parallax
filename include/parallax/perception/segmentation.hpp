#pragma once

#include <parallax/core/product.hpp>
#include <parallax/perception/image_space.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace parallax::perception {

    enum class MaskLayout : std::uint8_t {
        Unknown = 0, RowMajor
    };

    enum class MaskRepresentation : std::uint8_t {
        Unknown = 0, Host, CudaDevice
    };

    struct SegmentationMask {
        parallax::core::SourceObservation source_observation{};
        ImageSpace image_space = ImageSpace::Unknown;

        std::uint64_t query_revision = 0;

        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::size_t pitch_bytes = 0;

        MaskLayout layout = MaskLayout::Unknown;
        MaskRepresentation representation = MaskRepresentation::Unknown;

        float confidence = 0.0F;
        bool mask_valid = false;

        // The product owns mask storage without requiring a host-visible mask.
        std::shared_ptr<const void> storage{};

        [[nodiscard]] bool valid() const noexcept {
            return source_observation.valid() &&
                   image_space != ImageSpace::Unknown &&
                   query_revision != 0 &&
                   width != 0 &&
                   height != 0 &&
                   pitch_bytes != 0 &&
                   layout != MaskLayout::Unknown &&
                   representation != MaskRepresentation::Unknown &&
                   mask_valid &&
                   static_cast<bool>(storage);
        }
    };
}