#include <parallax/core/dependency_resolver.hpp>

#include <algorithm>
#include <exception>
#include <unordered_set>

namespace parallax::core {

    DependencyResolver::ProducerList DependencyResolver::resolve(ProductId product) const {
        return resolve(std::vector<ProductId>{product});
    }

    DependencyResolver::ProducerList DependencyResolver::resolve(const std::vector<ProductId>& products) const {
        ProducerList ordered;
        std::unordered_set<const Producer*> visited;

        const auto visit = [&](const auto& self, Producer* producer) -> void {
            if (producer == nullptr) return;
            if (!visited.insert(producer).second) return;

            // Dependencies go first so the returned plan is already executable.
            for (Producer* dependency : graph_.dependencies_of(*producer)) {
                self(self, dependency);
            }

            ordered.push_back(producer);
        };

        for (ProductId product : products) {
            visit(visit, graph_.producer_for(product));
        }

        return ordered;
    }

    void DependencyResolver::acquire(ProductId product, DemandSource source) {
        std::lock_guard<std::mutex> lock(demand_mutex_);

        auto& counts = demand_[product];
        const bool was_active = total(counts) != 0;

        ++count_for(counts, source);

        if (!was_active) {
            ++active_revision_;
        }
    }

    void DependencyResolver::release(ProductId product, DemandSource source) {
        std::lock_guard<std::mutex> lock(demand_mutex_);

        const auto it = demand_.find(product);
        if (it == demand_.end()) return;

        auto& count = count_for(it->second, source);

        if (count == 0) return;

        --count;

        if (total(it->second) == 0) {
            demand_.erase(it);
            ++active_revision_;
        }
    }

    std::size_t DependencyResolver::demand(ProductId product, DemandSource source) const noexcept {
        std::lock_guard<std::mutex> lock(demand_mutex_);

        const auto it = demand_.find(product);
        if (it == demand_.end()) return 0;

        return count_for(it->second, source);
    }

    std::size_t DependencyResolver::total_demand(ProductId product) const noexcept {
        std::lock_guard<std::mutex> lock(demand_mutex_);

        const auto it = demand_.find(product);
        if (it == demand_.end()) return 0;

        return total(it->second);
    }

    DemandSnapshot DependencyResolver::active_demand() const {
        std::lock_guard<std::mutex> lock(demand_mutex_);

        DemandSnapshot snapshot;
        snapshot.revision = active_revision_;
        snapshot.products.reserve(demand_.size());

        for (const auto& [product, counts] : demand_) {
            if (total(counts) != 0) {
                snapshot.products.push_back(product);
            }
        }

        // Stable ordering keeps independently demanded branches deterministic.
        std::sort(snapshot.products.begin(),snapshot.products.end(), [](ProductId left, ProductId right) {
                return static_cast<std::uint8_t>(left) < static_cast<std::uint8_t>(right);
            }
        );

        return snapshot;
    }

    std::size_t& DependencyResolver::count_for(DemandCounts& counts, DemandSource source) noexcept {
        switch (source) {
            case DemandSource::RuntimeBaseline:
                return counts.runtime_baseline;

            case DemandSource::Application:
                return counts.application;

            case DemandSource::FoxgloveSubscriber:
                return counts.foxglove_subscriber;

            case DemandSource::InternalDependent:
                return counts.internal_dependent;
        }

        std::terminate();
    }

    const std::size_t& DependencyResolver::count_for(const DemandCounts& counts, DemandSource source) noexcept {
        switch (source) {
            case DemandSource::RuntimeBaseline:
                return counts.runtime_baseline;

            case DemandSource::Application:
                return counts.application;

            case DemandSource::FoxgloveSubscriber:
                return counts.foxglove_subscriber;

            case DemandSource::InternalDependent:
                return counts.internal_dependent;
        }

        std::terminate();
    }

    std::size_t DependencyResolver::total(const DemandCounts& counts) noexcept {
        return counts.runtime_baseline +
               counts.application +
               counts.foxglove_subscriber +
               counts.internal_dependent;
    }

}