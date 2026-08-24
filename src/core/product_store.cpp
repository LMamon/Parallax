#include <parallax/core/product_store.hpp>

namespace parallax::core {
    bool ProductStore::contains(ProductId id) const noexcept {
        return entries_.find(id) != entries_.end();
    }

    void ProductStore::set_history_capacity(ProductId id, std::size_t capacity) {
        if (capacity == 0) {
            histories_.erase(id);
            return;
        }

        auto& history = histories_[id];
        history.capacity = capacity;

        // shrinking capacity drops the oldest retained products first.
        while (history.entries.size() > history.capacity) history.entries.pop_front();
    }

    std::size_t ProductStore::history_capacity(ProductId id) const noexcept {
        const auto it = histories_.find(id);

        if (it == histories_.end()) return 0;

        return it->second.capacity;
    }

    void ProductStore::clear() noexcept {
        // clearing only releases the stores references. any consumer hold a Product<T>
        // or its payload can keep that alloc alive
        entries_.clear();
        histories_.clear();
    }

}