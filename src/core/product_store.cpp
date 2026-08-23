#include <parallax/core/product_store.hpp>

namespace parallax::core {
    bool ProductStore::contains(ProductId id) const noexcept {
        return entries_.find(id) != entries_.end();
    }

    void ProductStore::clear() noexcept {
        // clearing only releases the stores references. any consumer hold a Product<T>
        // or its payload can keep that alloc alive
        entries_.clear();
    }

}