#include <gtest/gtest.h>

#include <parallax/core/execution_policy.hpp>
#include <parallax/core/product_id.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/localization/cuvslam_localizer.hpp>
#include <parallax/localization/cuvslam_producer.hpp>

#include <algorithm>

namespace {

    TEST(CuVslamProducerTest, DeclaresRectifiedGrayAsOrderedInput) {
        parallax::core::ProductStore store;
        parallax::localization::CuVslamLocalizer localizer;

        parallax::localization::CuVslamProducer producer(localizer, store);

        ASSERT_EQ(producer.inputs().size(), 1U);
        EXPECT_EQ(producer.inputs()[0], parallax::core::ProductId::RectifiedGray);

        ASSERT_EQ(producer.ordered_inputs().size(), 1U);
        EXPECT_EQ(producer.ordered_inputs()[0].product, parallax::core::ProductId::RectifiedGray);
        EXPECT_EQ(producer.ordered_inputs()[0].history_capacity, parallax::localization::CuVslamProducer::InputHistoryCapacity);
    }

    TEST(CuVslamProducerTest, UsesStatefulBlockingPolicy) {
        parallax::core::ProductStore store;
        parallax::localization::CuVslamLocalizer localizer;

        parallax::localization::CuVslamProducer producer(localizer, store);

        const auto policy = producer.execution_policy();

        EXPECT_EQ(policy.target_hz, 0.0);
        EXPECT_EQ(policy.max_input_age_ms, 0.0);
        EXPECT_EQ(policy.drop_policy, parallax::core::DropPolicy::Block);
        EXPECT_EQ(policy.affinity, parallax::core::ResourceAffinity::Gpu);
        EXPECT_TRUE(policy.stateful);
    }

    TEST(CuVslamProducerTest, PublishesLocalizationProducts) {
        parallax::core::ProductStore store;
        parallax::localization::CuVslamLocalizer localizer;

        parallax::localization::CuVslamProducer producer(localizer, store);

        const auto& outputs = producer.outputs();

        EXPECT_NE(std::find(outputs.begin(), outputs.end(), parallax::core::ProductId::LocalizationOdometry), outputs.end());
        EXPECT_NE(std::find(outputs.begin(), outputs.end(), parallax::core::ProductId::LocalizationTrajectory), outputs.end());
        EXPECT_NE(std::find(outputs.begin(), outputs.end(), parallax::core::ProductId::LocalizationState), outputs.end());
        EXPECT_NE(std::find(outputs.begin(), outputs.end(), parallax::core::ProductId::LocalizationPose), outputs.end());
    }
}