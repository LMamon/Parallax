#include <parallax/core/graph.hpp>

#include <stdexcept>
#include <unordered_map>

namespace parallax::core {
    void Graph::register_producer(Producer& producer) {
        // a producer is registered only once. duplicate producer registration would
        // otherwise duplicate graph nodes even if product mappins agree
        for (Producer* registered : producers_) {
            if (registered == &producer) {
                throw std::logic_error("producer already registered");
            }
        }

        // validate everyh output before mutating the graph so registration is atomic
        // from the callers perspective
        for (ProductId output : producer.outputs()) {
            if (product_producers_.find(output) != product_producers_.end()) {
                throw std::logic_error("product already has a registered producer");
            }
        }

        producers_.push_back(&producer);

        for (ProductId output : producer.outputs()) {
            product_producers_.emplace(output, &producer);
        }
    }

    void Graph::finalize() {
        dependencies_.clear();

        /**
         * convert product dependencies into producer dependencies. inputs with no
         * registered product are left unresolved here; migration later may 
         * legitimately expose products before every source producer exists.
         */
        for (Producer* producer : producers_) {
            auto& dependencies = dependencies_[producer];

            for (ProductId input : producer->inputs()) {
                Producer* dependency = producer_for(input);

                if (dependency == nullptr) continue;

                // multiple input products may come from the same producer retain producer edges only once
                bool already_present = false;
                for (Producer* existing : dependencies) {
                    if (existing == dependency) {
                        already_present = true;
                        break;
                    }
                }
                if (!already_present) dependencies.push_back(dependency);
            }
        } 
        verify_acyclic();
    }

    Producer* Graph::producer_for(ProductId product) const noexcept {
        const auto it = product_producers_.find(product);
        if (it == product_producers_.end()) return nullptr;

        return it->second;
    }

    const Graph::ProducerList& Graph::dependencies_of(const Producer& producer) const noexcept {
        const auto it = dependencies_.find(&producer);
        if (it != dependencies_.end()) return it->second;

        static const ProducerList empty;
        return empty;
    }

    void Graph::verify_acyclic() const {
        enum class VisitState {
            Unvisited, Visiting, Visited
        };

        std::unordered_map<const Producer*, VisitState> states;

        for (Producer* producer : producers_) {
            states.emplace(producer, VisitState::Unvisited);
        }

        const auto visit = [&](const auto& self, const Producer* producer) -> void {
            VisitState& state = states[producer];

            if (state == VisitState::Visiting) {
                throw std::logic_error("dependency graph contains a cycle");
            }

            if (state == VisitState::Visited) return;

            state = VisitState::Visiting;
            const auto dependencies = dependencies_.find(producer);
            if (dependencies != dependencies_.end()) {
                for (const Producer* dependency : dependencies->second) {
                    self(self, dependency);
                }
            }
            state = VisitState::Visited;
        };

        for (Producer* producer : producers_) {
            visit(visit, producer);
        }
    }
}