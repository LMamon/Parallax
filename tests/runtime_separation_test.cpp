#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#ifndef PARALLAX_SOURCE_DIR
#error "PARALLAX_SOURCE_DIR must be defined for runtime separation test"
#endif

namespace {

std::string readSource(const char* relative_path) {
    const std::string path = std::string(PARALLAX_SOURCE_DIR) + "/" + relative_path;
    std::ifstream input(path);
    EXPECT_TRUE(input.good()) << "unable to read " << path;
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

}  // namespace

TEST(RuntimeSeparationTest, ExecutionUnitDoesNotOwnBootstrapOrControlPlane) {
    const std::string execution = readSource("src/core/runtime_execution.cpp");

    EXPECT_EQ(execution.find("Runtime::initialize"), std::string::npos);
    EXPECT_EQ(execution.find("StereoCamera>("), std::string::npos);
    EXPECT_EQ(execution.find("register_producer"), std::string::npos);
    EXPECT_EQ(execution.find("graph_.finalize"), std::string::npos);
    EXPECT_EQ(execution.find("foxglove_.initialize"), std::string::npos);
}

TEST(RuntimeSeparationTest, BootstrapOwnsCompositionWithoutExecutionLoop) {
    const std::string bootstrap = readSource("src/core/runtime_bootstrap.cpp");

    EXPECT_NE(bootstrap.find("Runtime::initialize"), std::string::npos);
    EXPECT_NE(bootstrap.find("register_producer"), std::string::npos);
    EXPECT_NE(bootstrap.find("graph_.finalize"), std::string::npos);
    EXPECT_NE(bootstrap.find("DemandSource::RuntimeBaseline"), std::string::npos);
    EXPECT_NE(bootstrap.find("foxglove_.initialize"), std::string::npos);

    EXPECT_EQ(bootstrap.find("Runtime::run("), std::string::npos);
    EXPECT_EQ(bootstrap.find("submission_decision"), std::string::npos);
}

TEST(RuntimeSeparationTest, ExistingExecutionSemanticsRemainInExecutionUnit) {
    const std::string execution = readSource("src/core/runtime_execution.cpp");

    EXPECT_NE(execution.find("active_demand"), std::string::npos);
    EXPECT_NE(execution.find("resolver_.resolve"), std::string::npos);
    EXPECT_NE(execution.find("ProductId::LidarScan"), std::string::npos);
    EXPECT_NE(execution.find("submission_decision"), std::string::npos);
    EXPECT_NE(execution.find("producer->submit(context_)"), std::string::npos);
}

TEST(RuntimeSeparationTest, LocalizationUsesDedicatedOrderedWorker) {
    const std::string execution = readSource("src/core/runtime_execution.cpp");
    const std::string workers = readSource("src/core/runtime_workers.cpp");
    const std::string shutdown = readSource("src/core/runtime_shutdown.cpp");
    EXPECT_NE(execution.find("execution_plan.erase"), std::string::npos);
    EXPECT_NE(execution.find("cuvslam_producer_.get()"), std::string::npos);
    EXPECT_NE(execution.find("std::thread(&Runtime::runLocalization"), std::string::npos);
    EXPECT_NE(workers.find("Runtime::runLocalization"), std::string::npos);
    EXPECT_NE(workers.find("cuvslam_producer_->submit(context_)"), std::string::npos);
    EXPECT_NE(shutdown.find("localization_thread_.joinable()"), std::string::npos);
}

TEST(RuntimeSeparationTest, HostWaitReleasesCompletionRegistryBeforeSync) {
    const std::string context = readSource("src/core/execution_context.cpp");
    const auto begin = context.find("ExecutionContext::waitForHost");
    ASSERT_NE(begin, std::string::npos);
    const auto end = context.find("void ExecutionContext::shutdown", begin);
    ASSERT_NE(end, std::string::npos);
    const std::string body = context.substr(begin, end - begin);
    const auto lease_comment = body.find("// The ticket keeps this completion slot leased");
    ASSERT_NE(lease_comment, std::string::npos);
    const auto vpi_sync = body.find("vpiEventSync");
    const auto cuda_sync = body.find("cudaEventSynchronize");
    ASSERT_NE(vpi_sync, std::string::npos);
    ASSERT_NE(cuda_sync, std::string::npos);
    EXPECT_GT(vpi_sync, lease_comment);
    EXPECT_GT(cuda_sync, lease_comment);
}
