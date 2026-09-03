#pragma once

#include <parallax/core/product_store.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/image_space.hpp>

#include <memory>

namespace parallax::perception {

    [[nodiscard]] inline std::shared_ptr<const parallax::core::Product<parallax::isp::StereoRgbFrame>>
    
    find_segmentation_rgb(const parallax::core::ProductStore& products,
                          const parallax::core::Product<DetectionSet>& detection) {

        if (!detection.valid() || !detection.payload || detection.payload->image_space != ImageSpace::RgbLeft) {
            return {};
        }

        const auto rgb = products.find_observation<parallax::isp::StereoRgbFrame>(
                                                   parallax::core::ProductId::RgbLeft,
                                                   detection.metadata.observation);

        if (!rgb || !rgb->valid() || !rgb->payload) {
            return {};
        }

        if (!parallax::core::same_source_observation(*rgb, detection.metadata.observation)) {
            return {};
        }
        return rgb;
    }
}
