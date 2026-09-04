#pragma once

#include <parallax/core/product.hpp>

#include <cstdint>

namespace parallax::tracking {

    enum class TrackerFrameDecision : std::uint8_t {
        Accept = 0,
        RejectDuplicate,
        RejectOlder,
        RejectSource,
        ResetGap
    };

    struct TrackerFrameOrder {
        static constexpr std::uint64_t MaxSequenceGap = 3;

        TrackerFrameDecision decision = TrackerFrameDecision::Accept;
        std::uint64_t skipped = 0;

        [[nodiscard]] bool accepted() const noexcept {
            return decision == TrackerFrameDecision::Accept;
        }

        [[nodiscard]] bool requires_reset() const noexcept {
            return decision == TrackerFrameDecision::ResetGap ||
                   decision == TrackerFrameDecision::RejectSource;
        }
    };

    [[nodiscard]] inline TrackerFrameOrder evaluate_tracker_frame(const core::SourceObservation& previous,
                                                                  const core::SourceObservation& candidate) noexcept {

        if (!candidate.valid()) {
            return {TrackerFrameDecision::RejectSource, 0};
        }

        // The first valid observation establishes the tracker clock.
        if (!previous.valid()) {
            return {TrackerFrameDecision::Accept, 0};
        }

        if (candidate.source != previous.source) {
            return {TrackerFrameDecision::RejectSource, 0};
        }

        if (candidate.sequence == previous.sequence) {
            return {TrackerFrameDecision::RejectDuplicate, 0};
        }

        if (candidate.sequence < previous.sequence) {
            return {TrackerFrameDecision::RejectOlder, 0};
        }

        const std::uint64_t gap = candidate.sequence - previous.sequence;

        // Small skips stay useful in real time. Large gaps invalidate DCF state
        // instead of replaying old camera generations to catch back up.
        if (gap > TrackerFrameOrder::MaxSequenceGap) {
            return {TrackerFrameDecision::ResetGap, gap - 1};
        }

        return {TrackerFrameDecision::Accept, gap - 1};
    }
}