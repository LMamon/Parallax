#include <parallax/core/graph.hpp>
#include <parallax/core/dependency_resolver.hpp>
#include <parallax/core/execution_gate.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/core/history_configuration.hpp>


#include <chrono>
#include <memory>
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
                TestProducer(std::string_view name,
                             std::vector<ProductId> inputs,
                             std::vector<ProductId> outputs,
                             std::vector<OrderedInputRequirement> ordered_inputs = {}) :
                                name_(name),
                                inputs_(std::move(inputs)),
                                outputs_(std::move(outputs)),
                                ordered_inputs_(std::move(ordered_inputs)) {}

                [[nodiscard]] std::string_view name() const noexcept override { return name_; }
                [[nodiscard]] const std::vector<ProductId>& inputs() const noexcept override { return inputs_; }
                [[nodiscard]] const std::vector<ProductId>& outputs() const noexcept override { return outputs_; }
                [[nodiscard]] ExecutionPolicy execution_policy() const noexcept override { return {}; }

                SubmitResult submit(ExecutionContext& context) override {
                    (void)context;
                    ++submit_count_;
                    return SubmitResult::NoWork;
                }

                [[nodiscard]] const std::vector<OrderedInputRequirement>& ordered_inputs() const noexcept override {
                    return ordered_inputs_;
                }

                [[nodiscard]] int submit_count() const noexcept { return submit_count_; }

            private:
                std::string_view name_;
                std::vector<ProductId> inputs_;
                std::vector<ProductId> outputs_;
                
                int submit_count_ = 0;
                std::vector<OrderedInputRequirement> ordered_inputs_;      
        };

        void publish_test_product(
            ProductStore& store,
            ProductId id,
            SourceObservation observation) {

            ProductMetadata metadata{};
            metadata.observation = observation;
            metadata.timestamp = std::chrono::steady_clock::now();
            metadata.production_timestamp = metadata.timestamp;
            metadata.valid = true;

            store.publish(make_product<int>(id, metadata, std::make_shared<const int>(1)));
        }

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

            TestProducer camera{ "camera", {}, {ProductId::RgbLeft} };
            TestProducer rectifier{ "rectifier", {ProductId::RgbLeft}, {ProductId::RectifiedGray} };
            TestProducer stereo{ "stereo", {ProductId::RectifiedGray}, {ProductId::Disparity, ProductId::Confidence} };

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

        TEST(DependencyResolverTest, ResolvesSingleProductDependencyFirst) {
            Graph graph;

            TestProducer camera{ "camera", {}, {ProductId::RgbLeft} };
            TestProducer rectifier{ "rectifier", {ProductId::RgbLeft}, {ProductId::RectifiedGray} };
            TestProducer stereo{ "stereo", {ProductId::RectifiedGray}, {ProductId::Disparity, ProductId::Confidence} };
            TestProducer depth{ "depth", {ProductId::Disparity}, {ProductId::Depth} };

            graph.register_producer(camera);
            graph.register_producer(rectifier);
            graph.register_producer(stereo);
            graph.register_producer(depth);
            graph.finalize();

            DependencyResolver resolver{graph};

            const auto resolved = resolver.resolve(ProductId::Depth);

            ASSERT_EQ(resolved.size(), 4);
            EXPECT_EQ(resolved[0], &camera);
            EXPECT_EQ(resolved[1], &rectifier);
            EXPECT_EQ(resolved[2], &stereo);
            EXPECT_EQ(resolved[3], &depth);
        }

        TEST(DependencyResolverTest, ExcludesUnrelatedBranch) {
            Graph graph;

            TestProducer camera{ "camera", {}, {ProductId::RgbLeft} };
            TestProducer rectifier{ "rectifier", {ProductId::RgbLeft}, {ProductId::RectifiedGray} };
            TestProducer stereo{ "stereo", {ProductId::RectifiedGray}, {ProductId::Disparity} };
            TestProducer detector{ "detector", {ProductId::RgbLeft}, {ProductId::Detection} };

            graph.register_producer(camera);
            graph.register_producer(rectifier);
            graph.register_producer(stereo);
            graph.register_producer(detector);
            graph.finalize();

            DependencyResolver resolver{graph};

            const auto resolved = resolver.resolve(ProductId::Detection);

            ASSERT_EQ(resolved.size(), 2);
            EXPECT_EQ(resolved[0], &camera);
            EXPECT_EQ(resolved[1], &detector);
        }

        TEST(DependencyResolverTest, SharedDependenciesAppearOnce) {
            Graph graph;

            TestProducer camera{ "camera", {}, {ProductId::RgbLeft} };
            TestProducer rectifier{"rectifier", {ProductId::RgbLeft}, {ProductId::RectifiedGray} };
            TestProducer stereo{ "stereo", {ProductId::RectifiedGray}, {ProductId::Disparity} };
            TestProducer detector{ "detector", {ProductId::RgbLeft}, {ProductId::Detection} };

            graph.register_producer(camera);
            graph.register_producer(rectifier);
            graph.register_producer(stereo);
            graph.register_producer(detector);
            graph.finalize();

            DependencyResolver resolver{graph};

            const auto resolved = resolver.resolve(std::vector<ProductId>{ProductId::Disparity,
                                                                          ProductId::Detection});

            ASSERT_EQ(resolved.size(), 4);

            EXPECT_EQ(resolved[0], &camera);
            EXPECT_EQ(resolved[1], &rectifier);
            EXPECT_EQ(resolved[2], &stereo);
            EXPECT_EQ(resolved[3], &detector);
        }

        TEST(DependencyResolverTest, ResolutionDoesNotExecuteProducers) {
            Graph graph;

            TestProducer camera{ "camera", {}, {ProductId::RgbLeft} };

            graph.register_producer(camera);
            graph.finalize();

            DependencyResolver resolver{graph};

            const auto resolved = resolver.resolve(ProductId::RgbLeft);

            ASSERT_EQ(resolved.size(), 1);
            EXPECT_EQ(resolved[0], &camera);
            EXPECT_EQ(camera.submit_count(), 0);

            // TestProducer::submit() is never required for graph resolution.
            // Resolution describes work; execution remains a later runtime concern.
        }

        TEST(DependencyResolverTest, DemandSourcesAreCountedIndependently) {
            Graph graph;
            DependencyResolver resolver{graph};

            resolver.acquire(ProductId::Depth, DemandSource::Application);
            resolver.acquire(ProductId::Depth, DemandSource::FoxgloveSubscriber);
            resolver.acquire(ProductId::Depth, DemandSource::InternalDependent);

            EXPECT_EQ(resolver.demand(ProductId::Depth, DemandSource::Application), 1);
            EXPECT_EQ(resolver.demand(ProductId::Depth, DemandSource::FoxgloveSubscriber), 1);
            EXPECT_EQ(resolver.demand(ProductId::Depth, DemandSource::InternalDependent), 1);

            EXPECT_EQ(resolver.total_demand(ProductId::Depth), 3);
            EXPECT_TRUE(resolver.demanded(ProductId::Depth));
        }

        TEST(DependencyResolverTest, DemandIsReferenceCounted) {
            Graph graph;
            DependencyResolver resolver{graph};

            resolver.acquire(ProductId::Depth, DemandSource::FoxgloveSubscriber);
            resolver.acquire(ProductId::Depth, DemandSource::FoxgloveSubscriber);

            EXPECT_EQ(resolver.demand(ProductId::Depth, DemandSource::FoxgloveSubscriber), 2);
            EXPECT_EQ(resolver.total_demand(ProductId::Depth), 2);

            resolver.release(ProductId::Depth, DemandSource::FoxgloveSubscriber);

            EXPECT_EQ(resolver.demand(ProductId::Depth, DemandSource::FoxgloveSubscriber), 1);
            EXPECT_TRUE(resolver.demanded(ProductId::Depth));
        }

        TEST(DependencyResolverTest, ProductStopsBeingDemandedAtZeroReferences) {
            Graph graph;
            DependencyResolver resolver{graph};

            resolver.acquire(ProductId::Detection, DemandSource::Application);

            ASSERT_TRUE(resolver.demanded(ProductId::Detection));

            resolver.release(ProductId::Detection, DemandSource::Application);

            EXPECT_EQ(resolver.total_demand(ProductId::Detection), 0);
            EXPECT_FALSE(resolver.demanded(ProductId::Detection));

            // Extra release is intentionally harmless.
            resolver.release(ProductId::Detection, DemandSource::Application);
            EXPECT_EQ(resolver.total_demand(ProductId::Detection), 0);
        }

        TEST(DependencyResolverTest, PersistentDemandRemainsUntilExplicitRelease) {
            Graph graph;
            DependencyResolver resolver{graph};

            // A stateful command such as "track this target" is represented by
            // application demand that persists across resolver operations until the
            // command is explicitly cancelled.
            resolver.acquire(ProductId::Track2D, DemandSource::Application);

            EXPECT_TRUE(resolver.demanded(ProductId::Track2D));

            // Resolving other work does not consume or clear persistent demand.
            const auto resolved = resolver.resolve(ProductId::Depth);
            EXPECT_TRUE(resolved.empty());

            EXPECT_EQ(resolver.demand(ProductId::Track2D, DemandSource::Application), 1);
            EXPECT_TRUE(resolver.demanded(ProductId::Track2D));

            resolver.release(ProductId::Track2D, DemandSource::Application);

            EXPECT_FALSE(resolver.demanded(ProductId::Track2D));
        }

        TEST(DependencyResolverTest, ResolvesLidarIndependentlyFromCameraBranch) {
            TestProducer camera{"camera", {}, {ProductId::RgbLeft}};
            TestProducer rectifier{"rectifier", {ProductId::RgbLeft}, {ProductId::RectifiedGray}};
            TestProducer stereo{"stereo", {ProductId::RectifiedGray}, {ProductId::Disparity}};
            TestProducer lidar{"rplidar", {}, {ProductId::LidarScan}};

            Graph graph;
            graph.register_producer(camera);
            graph.register_producer(rectifier);
            graph.register_producer(stereo);
            graph.register_producer(lidar);
            graph.finalize();

            DependencyResolver resolver{graph};

            const auto resolved = resolver.resolve(ProductId::LidarScan);

            ASSERT_EQ(resolved.size(), 1);
            EXPECT_EQ(resolved[0], &lidar);
        }

        TEST(ExecutionGateTest, CompatibleInputsProduceOneObservation) {
            ProductStore store;

            const SourceObservation observation{SourceId::StereoCamera, 42};

            publish_test_product(store, ProductId::Pose, observation);
            publish_test_product(store, ProductId::Depth, observation);

            TestProducer producer{"marker-depth", {ProductId::Pose, ProductId::Depth}, {ProductId::MarkerDepth}};

            const auto result = input_observation(producer, store);

            ASSERT_TRUE(result.has_value());
            EXPECT_EQ(*result, observation);
        }

        TEST(ExecutionGateTest, IncompatibleInputsAreRejected) {
            ProductStore store;

            publish_test_product(store, ProductId::Pose, {SourceId::StereoCamera, 42});
            publish_test_product(store, ProductId::Depth, {SourceId::StereoCamera, 43});

            TestProducer producer{"marker-depth", {ProductId::Pose, ProductId::Depth}, {ProductId::MarkerDepth}};
            EXPECT_FALSE(input_observation(producer, store).has_value());
        }

        TEST(ExecutionGateTest, UnthrottledProducerIsAlwaysDue) {
            ExecutionPolicy policy{};
            policy.target_hz = 0.0;

            ProducerExecutionState state{};
            state.has_last_submission = true;

            const auto now = std::chrono::steady_clock::now();
            state.last_submission = now;

            InputObservation input{SourceObservation{SourceId::StereoCamera, 1}, now};

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::Submit);
        }

        TEST(ExecutionGateTest, TargetRateRejectsEarlyExecution) {
            ExecutionPolicy policy{};
            policy.target_hz = 10.0; // 100 ms period

            ProducerExecutionState state{};
            state.has_last_submission = true;

            const auto now = std::chrono::steady_clock::now();
            state.last_submission = now - std::chrono::milliseconds(50);

            InputObservation input{SourceObservation{SourceId::StereoCamera, 1}, now};

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::RateLimited);
        }

        TEST(ExecutionGateTest, TargetRateAcceptsDueExecution) {
            ExecutionPolicy policy{};
            policy.target_hz = 10.0; // 100 ms period

            ProducerExecutionState state{};
            state.has_last_submission = true;

            const auto now = std::chrono::steady_clock::now();
            state.last_submission = now - std::chrono::milliseconds(100);

            InputObservation input{SourceObservation{SourceId::StereoCamera, 1}, now};

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::Submit);
        }

        TEST(ExecutionGateTest, FreshnessDisabledAcceptsOldInput) {
            ExecutionPolicy policy{};
            policy.max_input_age_ms = 0.0;

            ProducerExecutionState state{};

            const auto now = std::chrono::steady_clock::now();

            InputObservation input{};
            input.observation = SourceObservation{SourceId::StereoCamera, 1};
            input.timestamp = now - std::chrono::seconds(10);

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::Submit);
        }

        TEST(ExecutionGateTest, FreshInputIsAccepted) {
            ExecutionPolicy policy{};
            policy.max_input_age_ms = 100.0;

            ProducerExecutionState state{};

            const auto now = std::chrono::steady_clock::now();

            InputObservation input{};
            input.observation = SourceObservation{SourceId::StereoCamera, 1};
            input.timestamp = now - std::chrono::milliseconds(50);

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::Submit);
        }

        TEST(ExecutionGateTest, StaleInputIsRejected) {
            ExecutionPolicy policy{};
            policy.max_input_age_ms = 100.0;

            ProducerExecutionState state{};

            const auto now = std::chrono::steady_clock::now();

            InputObservation input{};
            input.observation = SourceObservation{SourceId::StereoCamera, 1};
            input.timestamp = now - std::chrono::milliseconds(101);

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::StaleInput);
        }

        TEST(ExecutionGateTest, FutureTimestampIsRejected) {
            ExecutionPolicy policy{};
            policy.max_input_age_ms = 100.0;

            ProducerExecutionState state{};

            const auto now = std::chrono::steady_clock::now();

            InputObservation input{};
            input.observation = SourceObservation{SourceId::StereoCamera, 1};
            input.timestamp = now + std::chrono::milliseconds(1);

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::StaleInput);
        }

        TEST(ExecutionGateTest, SupersedeStillRejectsConsumedObservation) {
            ExecutionPolicy policy{};
            policy.drop_policy = DropPolicy::Supersede;

            ProducerExecutionState state{};

            const auto now = std::chrono::steady_clock::now();

            InputObservation input{};
            SourceObservation{SourceId::StereoCamera, 42};
            input.timestamp = now;

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::Submit);
            record_submission(state, policy, input, now);
            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::Superseded);
        }

        TEST(ExecutionGateTest, SubmissionDecisionReportsRateLimit) {
            ExecutionPolicy policy{};
            policy.target_hz = 10.0;

            ProducerExecutionState state{};
            const auto now = std::chrono::steady_clock::now();

            state.has_last_submission = true;
            state.last_submission = now - std::chrono::milliseconds(20);

            InputObservation input{SourceObservation{SourceId::StereoCamera, 1}, now};

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::RateLimited);
        }

        TEST(ExecutionGateTest, SubmissionDecisionReportsStaleInput) {
            ExecutionPolicy policy{};
            policy.max_input_age_ms = 50.0;

            ProducerExecutionState state{};
            const auto now = std::chrono::steady_clock::now();

            InputObservation input{SourceObservation{SourceId::StereoCamera, 1}, now - std::chrono::milliseconds(100)};

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::StaleInput);
        }

        TEST(ExecutionGateTest, SubmissionDecisionReportsSupersededInput) {
            ExecutionPolicy policy{};
            policy.drop_policy = DropPolicy::Supersede;

            ProducerExecutionState state{};
            state.has_last_observation = true;
            state.last_observation = SourceObservation{SourceId::StereoCamera, 7};

            const auto now = std::chrono::steady_clock::now();

            InputObservation input{
                SourceObservation{SourceId::StereoCamera, 7},
                now
            };

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::Superseded);
        }

        TEST(ExecutionGateTest, SubmissionDecisionAcceptsRunnableInput) {
            ExecutionPolicy policy{};
            policy.target_hz = 30.0;
            policy.max_input_age_ms = 100.0;
            policy.drop_policy = DropPolicy::Supersede;

            ProducerExecutionState state{};
            const auto now = std::chrono::steady_clock::now();

            InputObservation input{
                SourceObservation{SourceId::StereoCamera, 8},
                now - std::chrono::milliseconds(5)
            };

            EXPECT_EQ(submission_decision(policy, state, input, now), SubmissionDecision::Submit);
        }

        TEST(GraphTest, OrderedInputMustBeDeclaredDependency) {
            Graph graph;

            TestProducer producer{"ordered", {ProductId::Depth}, {ProductId::Track3D}, {{ProductId::LidarScan, 4}}};

            EXPECT_THROW(graph.register_producer(producer), std::logic_error);
        }

        TEST(GraphTest, OrderedRequirementsConfigureBoundedHistory) {
            Graph graph;
            ProductStore store;

            TestProducer producer{"ordered", {ProductId::Depth}, {ProductId::Track3D}, {{ProductId::Depth, 4}}};

            graph.register_producer(producer);
            graph.finalize();

            configure_ordered_history(graph, store);
            EXPECT_EQ(store.history_capacity(ProductId::Depth), 4);
        }

        TEST(GraphTest, LargestOrderedHistoryRequirementWins) {
            Graph graph;
            ProductStore store;

            TestProducer first{"first", {ProductId::Depth}, {ProductId::Track2D}, {{ProductId::Depth, 4}}};
            TestProducer second{"second", {ProductId::Depth}, {ProductId::Track3D}, {{ProductId::Depth, 12}}};

            graph.register_producer(first);
            graph.register_producer(second);
            graph.finalize();

            configure_ordered_history(graph, store);

            EXPECT_EQ(store.history_capacity(ProductId::Depth), 12);
        }

        TEST(GraphTest, OrderedConsumerCanAdvanceThroughRetainedObservations) {
            Graph graph;
            ProductStore store;

            TestProducer source{"source", {}, {ProductId::Depth}};

            TestProducer ordered_consumer{"ordered_consumer",
                                        {ProductId::Depth},
                                        {ProductId::Projection},
                                        {OrderedInputRequirement{ProductId::Depth, 4}}};

            graph.register_producer(source);
            graph.register_producer(ordered_consumer);
            graph.finalize();

            configure_ordered_history(graph, store);

            EXPECT_EQ(store.history_capacity(ProductId::Depth), 4);

            publish_test_product(store, ProductId::Depth, SourceObservation{SourceId::StereoCamera, 10});
            publish_test_product(store, ProductId::Depth, SourceObservation{SourceId::StereoCamera, 11});
            publish_test_product(store, ProductId::Depth, SourceObservation{SourceId::StereoCamera, 12});

            const SourceObservation consumed{SourceId::StereoCamera, 10};

            const auto next =store.next_after<int>(ProductId::Depth, consumed);

            ASSERT_NE(next, nullptr);
            EXPECT_EQ(next->metadata.observation, (SourceObservation{SourceId::StereoCamera, 11}));
        }
    }
}