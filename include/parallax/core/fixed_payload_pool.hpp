#pragma once

#include <array>
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
                if (slot && slot.use_count() == 1) return slot;
            }

            return {};
        }

        [[nodiscard]] const T* prototype() const noexcept { return slots_[0].get(); }
        [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Capacity; }

        void reset() noexcept {
            std::lock_guard<std::mutex> lock(mutex_);
            slots_ = {};
            initialized_ = false;
        }

        [[nodiscard]] bool initialized() const noexcept { return initialized_; }

    private:
        std::array<std::shared_ptr<T>, Capacity> slots_{};
        mutable std::mutex mutex_;
        bool initialized_ = false;
    };
}