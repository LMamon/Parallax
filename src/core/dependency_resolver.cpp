#include <parallax/core/dependency_resolver.hpp>

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

            /**
             * Graph::finalize() already establishes+validated these
             * producer edges. walking dependencies before appending the current
             * producer gives dependency-first top.order
            */
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
        ++count_for(counts, source);
    }

    void DependencyResolver::release(ProductId product, DemandSource source) {
        std::lock_guard<std::mutex> lock(demand_mutex_);
        const auto it = demand_.find(product);
        if (it == demand_.end()) return;

        auto& count = count_for(it->second, source);
        if (count != 0) --count;

        if (it->second.application == 0 &&
            it->second.foxglove_subscriber == 0 &&
            it->second.internal_dependent == 0) {
                
                demand_.erase(it);
        }
    }

    std::size_t DependencyResolver::demand(ProductId product, DemandSource source) const noexcept {
        std::lock_guard<std::mutex> lock(demand_mutex_);
        const auto it = demand_.find(product);
        if (it == demand_.end()) {
            return 0;
        }

        return count_for(it->second, source);
    }

    std::size_t DependencyResolver::total_demand(ProductId product) const noexcept {
        std::lock_guard<std::mutex> lock(demand_mutex_);
        const auto it = demand_.find(product);
        if (it == demand_.end()) return 0;

        const auto& counts = it->second;

        return counts.application + counts.foxglove_subscriber + counts.internal_dependent;
    }

    std::size_t& DependencyResolver::count_for(DemandCounts& counts, DemandSource source) noexcept{
        switch (source) {
            case DemandSource::Application:
                return counts.application;
            
            case DemandSource::FoxgloveSubscriber:
                return counts.foxglove_subscriber;
            
            case DemandSource::InternalDependent:
                return counts.internal_dependent;
                
            default:
                break;
        }
        return counts.application;
    }

    const std::size_t& DependencyResolver::count_for(const DemandCounts& counts, DemandSource source) noexcept {
        switch (source) {
        case DemandSource::Application:
            return counts.application;

        case DemandSource::FoxgloveSubscriber:
            return counts.foxglove_subscriber;

        case DemandSource::InternalDependent:
            return counts.internal_dependent;

        default:
            break;
        }

        return counts.application;
    }
}