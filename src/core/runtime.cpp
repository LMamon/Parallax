#include <parallax/core/runtime.hpp>
#include <parallax/core/pipeline.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>

#include <vector>
#include <iostream>

namespace parallax::core {
    Runtime::~Runtime() { shutdown(); }

    bool Runtime::initialize(
        const std::filesystem::path& camera_config_path,
        const std::filesystem::path& calibration_directory) {

        if (initialized_) return true;

        if (!config_.loadFromFile(camera_config_path)) {
            std::cerr << "Runtime: failed to load camera config\n";
            return false;
        }

        camera_ = std::make_unique<parallax::camera::StereoCamera>(config_);

        if (!camera_->initialize()) {
            std::cerr << "Runtime: failed to initialize camera\n";
            shutdown();
            return false;
        }

        lidar_ = std::make_unique<parallax::lidar::Rplidar>();

        if (!lidar_->initialize()) {
            std::cerr << "Runtime: failed to initialize RPLIDAR\n";
            shutdown();
            return false;
        }
        /**
         * Pipeline still owns the proven processing resources during Phase 5:
         * ISP allocations, VPI stream, rectifier, matcher, depth storage, and pose
         * estimator. Runtime now takes over orchestration through graph producers.
         */
        if (!pipeline_.initialize(config_, calibration_directory)) {
            std::cerr << "Runtime: failed to initialize processing pipeline\n";
            shutdown();
            return false;
        }

        /**
         * Construct the complete producer set before registering or finalizing the
         * graph. Graph stores non-owning Producer pointers, so Runtime owns every
         * producer for the lifetime of the graph.
         */
        camera_producer_ = std::make_unique<parallax::camera::CameraProducer>(*camera_, product_store_);
        lidar_producer_ = std::make_unique<parallax::lidar::RplidarSourceProducer>(*lidar_, product_store_);
        isp_producer_ = std::make_unique<parallax::isp::IspProducer>(pipeline_.isp(), product_store_);

        rectification_producer_ = std::make_unique<parallax::stereo::RectificationProducer>(
                                                   pipeline_.rectifier(),
                                                   pipeline_.calibration(),
                                                   product_store_);

        stereo_producer_ = std::make_unique<parallax::stereo::StereoProducer>(pipeline_.matcher(), product_store_);

        depth_producer_ = std::make_unique<parallax::stereo::DepthProducer>(
                                           pipeline_.calibration(),
                                           pipeline_.depth(),
                                           pipeline_.vpiStream(),
                                           product_store_);

        charuco_pose_producer_ = std::make_unique<parallax::pose::CharucoPoseProducer>(
                                                  pipeline_.charucoPose(),
                                                  pipeline_.calibration(),
                                                  product_store_);

        marker_depth_producer_ = std::make_unique<parallax::pose::MarkerDepthPoducer>(product_store_);

        /**
         * Registration describes the complete concrete dependency graph.
         * Finalization happens exactly once, after every producer is present.
         */
        graph_.register_producer(*camera_producer_);
        graph_.register_producer(*isp_producer_);
        graph_.register_producer(*rectification_producer_);
        graph_.register_producer(*stereo_producer_);
        graph_.register_producer(*depth_producer_);
        graph_.register_producer(*charuco_pose_producer_);
        graph_.register_producer(*marker_depth_producer_);
        graph_.register_producer(*lidar_producer_);

        graph_.finalize();

        if (!foxglove_.initialize()) {
            std::cerr << "Runtime: failed to initialize Foxglove server\n";
            shutdown();
            return false;
        }

        const auto& rgb = pipeline_.rgb();

        if (!publisher_.initialize(rgb.width, rgb.height, config_.frame_rate)) {
            std::cerr << "Runtime: failed to initialize visualization publisher\n";

            shutdown();
            return false;
        }

        if (!publisher_.publishLeftCalibration(pipeline_.calibration())) {
            std::cerr << "Runtime: failed to publish left camera calibration\n";
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    void Runtime::run(const volatile std::sig_atomic_t& stop_requested) {
        if (!initialized_) return;
        running_.store(true);
        int failed_frames = 0;

        /**
         * The normal runtime currently requests the products needed by the existing
         * Foxglove/demo surface.
         *
         * MarkerDepth pulls the pose and stereo-depth branches together.
         * Confidence is requested separately because MarkerDepth needs Disparity but
         * does not semantically depend on Confidence.
         *
         * DependencyResolver deduplicates shared producers and returns them in
         * dependency-first order.
         */
        const std::vector<ProductId> requested_products{
            ProductId::MarkerDepth,
            ProductId::Confidence
        };

        const auto execution_plan = resolver_.resolve(requested_products);

        while (running_.load() && !stop_requested) {
            bool frame_failed = false;

            for (Producer* producer : execution_plan) {
                if (producer == nullptr) {
                    frame_failed = true;
                    break;
                }

                const SubmitResult result = producer->submit();

                if (result == SubmitResult::Failed) {
                    std::cerr << "Runtime: producer failed: " << producer->name() << '\n';
                    frame_failed = true;
                    break;
                }

                if (result == SubmitResult::NoWork) {
                    /**
                     * A source may transiently fail to provide a frame. Downstream
                     * producers must not execute against stale products from the
                     * previous iteration.
                     */
                    frame_failed = true;
                    break;
                }
            }

            if (frame_failed) {
                if (++failed_frames >= 10) {
                    std::cerr
                        << "Runtime: graph failed to produce a valid frame\n";
                    break;
                }
                continue;
            }

            failed_frames = 0;

            const auto marker = product_store_.latest<parallax::pose::CharucoPoseResult>(ProductId::MarkerDepth);
            const auto stereo = product_store_.latest<parallax::isp::StereoMatchFrame>(ProductId::Disparity);
            const auto depth = product_store_.latest<parallax::isp::DepthFrame>(ProductId::Depth);

            if (!marker || !marker->valid() ||
                !stereo || !stereo->valid() ||
                !depth || !depth->valid()) {

                std::cerr << "Runtime: graph completed without required products\n";
                break;
            }

            /**
             * Rectified RGB is still the established visualization surface.
             * StereoRectifier updates this buffer during the graph-driven
             * RectificationProducer submission, so visualization can keep using the
             * existing Publisher API without changing image semantics in this cutover.
             */
            publisher_.publishLeftImage(pipeline_.rgb(), *marker->payload, marker->metadata.timestamp);
            publisher_.publishDisparity(*stereo->payload);
            publisher_.publishConfidence(*stereo->payload);
            publisher_.publishDepth(*depth->payload);

            // processCommands();
            // dispatch(products);
        }
        running_.store(false);
    }

    void Runtime::stop() noexcept { running_.store(false); }

    void Runtime::shutdown() {
        stop();

        publisher_.shutdown();
        foxglove_.shutdown();

        /**
         * Drop published handles before destroying the hardware and processing
         * resources they reference. Producers are destroyed before their backing
         * devices because Graph stores only non-owning producer pointers.
         */
        product_store_.clear();

        marker_depth_producer_.reset();
        charuco_pose_producer_.reset();
        depth_producer_.reset();
        stereo_producer_.reset();
        rectification_producer_.reset();
        isp_producer_.reset();
        lidar_producer_.reset();
        camera_producer_.reset();

        pipeline_.shutdown();

        if (lidar_) {
            lidar_->shutdown();
            lidar_.reset();
        }

        if (camera_) {
            camera_->shutdown();
            camera_.reset();
        }

        initialized_ = false;
    }
}