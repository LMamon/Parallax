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
}