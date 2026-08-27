#include <parallax/core/product_store.hpp>

namespace parallax::core {
    void ProductStore::set_history_capacity(ProductId id, std::size_t capacity) {
        std::unique_lock lock(mutex_);

        if (capacity == 0) {
            histories_.erase(id);
            return;
        }

        auto& history = histories_[id];
        history.capacity = capacity;

        while (history.entries.size() > capacity) {
            history.entries.pop_front();
        }
    }

    std::size_t ProductStore::history_capacity(ProductId id) const noexcept {
        std::shared_lock lock(mutex_);

        const auto it = histories_.find(id);
        if (it == histories_.end()) return 0;

        return it->second.capacity;
    }

    std::optional<ProductMetadata> ProductStore::metadata(ProductId id) const noexcept {
        std::shared_lock lock(mutex_);

        const auto it = entries_.find(id);
        if (it == entries_.end()) return std::nullopt;

        return it->second.metadata;
    }
    
    bool ProductStore::contains(ProductId id) const noexcept {
        std::shared_lock lock(mutex_);
        return entries_.find(id) != entries_.end();
    }

    void ProductStore::clear() noexcept {
        std::unique_lock lock(mutex_);

        entries_.clear();

        for (auto& [id, history] : histories_) {
            (void)id;
            history.entries.clear();
        }
    }
}