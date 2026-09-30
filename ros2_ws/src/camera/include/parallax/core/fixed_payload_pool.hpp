#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace parallax::core {

    /**
     * Small fixed-capacity pool for preallocated product payloads.
     *
     * This is intentionally not a general allocator:
     * - capacity is compile-time fixed;
     * - slots are created during initialization;
     * - acquire() never allocates;
     * - the pool never grows.
     *
     * Each slot has one baseline shared_ptr owned by the pool. A slot is
     * reusable only when that is the sole remaining reference.
     *
     * Product/shared_ptr lifetime therefore acts as the lease that prevents
     * accelerator memory from being overwritten while a consumer still holds it.
     */
    template <typename T, std::size_t Capacity>
    class FixedPayloadPool {
        static_assert(Capacity > 0);

    public:
        FixedPayloadPool() = default;

        FixedPayloadPool(const FixedPayloadPool&) = delete;
        FixedPayloadPool& operator=(const FixedPayloadPool&) = delete;

        template <typename InitializeSlot> bool initialize(InitializeSlot&& initialize_slot) {
            std::lock_guard<std::mutex> lock(mutex_);

            if (initialized_) return true;

            for (std::size_t i = 0; i < Capacity; ++i) {
                auto slot = std::make_shared<T>();

                if (!initialize_slot(*slot, i)) {
                    slots_ = {};
                    initialized_ = false;
                    return false;
                }

                slots_[i] = std::move(slot);
            }

            initialized_ = true;
            return true;
        }

        /**
         * Acquire one currently unleased slot.
         *
         * use_count()==1 means only the pool owns the slot. Returning a copy
         * increments the count before another acquire can select it.
         */
        [[nodiscard]] std::shared_ptr<T> acquire() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) return {};

            for (auto& slot : slots_) {
                if (slot && slot.use_count() == 1) {
                    auto acquired = slot;

                    std::size_t in_use = 0;
                    for (const auto& candidate : slots_) {
                        if (candidate && candidate.use_count() > 1) ++in_use;
                    }

                    high_water_mark_ = std::max(high_water_mark_, in_use);
                    return acquired;
                }
            }
            return {};
        }

        [[nodiscard]] const T* prototype() const noexcept { return slots_[0].get(); }
        [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Capacity; }

        void reset() noexcept {
            std::lock_guard<std::mutex> lock(mutex_);
            slots_ = {};
            initialized_ = false;
            high_water_mark_ = 0;
        }

        [[nodiscard]] std::size_t in_use() const noexcept {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!initialized_) return 0;

            std::size_t count = 0;
            for (const auto& slot : slots_) {
                if (slot && slot.use_count() > 1) ++count;
            }
            return count;
        }

        [[nodiscard]] bool initialized() const noexcept { return initialized_; }

        [[nodiscard]] std::size_t high_water_mark() const noexcept {
            std::lock_guard<std::mutex> lock(mutex_);
            return high_water_mark_;
        }

    private:
        std::array<std::shared_ptr<T>, Capacity> slots_{};
        std::size_t high_water_mark_ = 0;
        mutable std::mutex mutex_;
        bool initialized_ = false;
    };
}