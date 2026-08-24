#include <parallax/core/graph.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace parallax::core {
    namespace {
        // minimal graph-only producer. these tests are for verifying dependency 
        // declaration and registration semantics not execution
        class TestProducer final : public Producer {
            public: 
                TestProducer(std::string_view name, std::vector<ProductId> inputs, std::vector<ProductId> outputs) :
                                                                                name_(name),
                                                                                inputs_(std::move(inputs)),
                                                                                outputs_(std::move(outputs)) {}

                [[nodiscard]] std::string_view name() const noexcept override {
                    return name_;
                }

                [[nodiscard]] const std::vector<ProductId>& inputs() const noexcept override {
                    return inputs_;
                }

                [[nodiscard]] const std::vector<ProductId>& outputs() const noexcept override {
                    return outputs_;
                }

                [[nodiscard]] ExecutionPolicy execution_policy() const noexcept override {
                    return {};
                }

                SubmitResult submit() override {
                    return SubmitResult::NoWork;
                }

            private:
                std::string_view name_;
                std::vector<ProductId> inputs_;
                std::vector<ProductId> outputs_;
        };

        TEST(GraphTest, ProductLookupReturnsRegisteredProducer) {
            Graph graph;
            TestProducer camera{
                "camera", {}, {ProductId::RgbLeft}
            };

            graph.register_producer(camera);
            graph.finalize();

            EXPECT_EQ(graph.producer_for(ProductId::RgbLeft), &camera);
            EXPECT_EQ(graph.producer_for(ProductId::Depth), nullptr);
        }

        TEST(GraphTest, DependenciesAreDerivedFromProductInputs) {
            Graph graph;

            TestProducer camera{
                "camera", {}, {ProductId::RgbLeft}
            };
            TestProducer rectifier{
                "rectifier", {ProductId::RgbLeft}, {ProductId::RectifiedGray}
            };
            TestProducer stereo{
                "stereo", {ProductId::RectifiedGray}, {ProductId::Disparity, ProductId::Confidence}
            };

            graph.register_producer(camera);
            graph.register_producer(rectifier);
            graph.register_producer(stereo);
            graph.finalize();

            const auto& rectifier_dependencies = graph.dependencies_of(rectifier);

            ASSERT_EQ(rectifier_dependencies.size(), 1);
            EXPECT_EQ(rectifier_dependencies[0], &camera);
            
            const auto & stereo_dependencies = graph.dependencies_of(stereo);

            ASSERT_EQ(stereo_dependencies.size(), 1);
            EXPECT_EQ(stereo_dependencies[0], &rectifier);
        }

        TEST(GraphTest, DuplicateProductProducerIsRejected) {
            Graph graph;

            TestProducer first{"first", {}, {ProductId::Depth}};
            TestProducer second{"second", {}, {ProductId::Depth}};

            graph.register_producer(first);

            EXPECT_THROW(graph.register_producer(second), std::logic_error);
        }

        TEST(GraphTest, DependencyCycleIsRejected) {
            Graph graph;

            TestProducer first{"first", {ProductId::Depth}, {ProductId::Pose}};
            TestProducer second{"second", {ProductId::Pose}, {ProductId::Depth}};

            graph.register_producer(first);
            graph.register_producer(second);

            EXPECT_THROW(graph.finalize(), std::logic_error);
        }
    }
}