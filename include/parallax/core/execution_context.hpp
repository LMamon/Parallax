#pragma once

#include <parallax/core/product_store.hpp>
#include <parallax/vpi/stream.hpp>
#include <parallax/core/completion.hpp>


#include <cuda_runtime.h>

#include <vpi/Event.h>

#include <array>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>


namespace nvinfer1 {
    class IExecutionContext;
}

namespace parallax::core {

    /**
     * Small bounded worker pool for CPU-side producer work.
     *
     * The executor intentionally rejects work once its queue is full rather than
     * allowing latency to grow without bound. Detailed scheduling/drop policy
     * remains a Phase 9 responsibility.
     */
    class BoundedCpuExecutor {
        public:
            BoundedCpuExecutor() = default;
            ~BoundedCpuExecutor();

            BoundedCpuExecutor(const BoundedCpuExecutor&) = delete;
            BoundedCpuExecutor& operator=(const BoundedCpuExecutor&) = delete;

            BoundedCpuExecutor(BoundedCpuExecutor&&) = delete;
            BoundedCpuExecutor& operator=(BoundedCpuExecutor&&) = delete;

            bool initialize(std::size_t worker_count, std::size_t queue_capacity);
            void shutdown() noexcept;

            /**
             * Queue one unit of CPU work.
             * Returns false when the executor is stopped or its bounded queue is full.
             */
            bool submit(std::function<void()> task);

            [[nodiscard]] bool initialized() const noexcept {
                return initialized_;
            }

        private:
            void workerLoop();

            std::vector<std::thread> workers_;
            std::queue<std::function<void()>> tasks_;

            std::mutex mutex_;
            std::condition_variable work_available_;

            std::size_t queue_capacity_ = 0;
            bool stopping_ = false;
            bool initialized_ = false;
    };


    /**
     * Runtime-scoped owner for resources shared across graph execution.
     *
     * ExecutionContext is intentionally separate from Producer and from the
     * algorithm implementations. Producers describe graph work; algorithm classes
     * continue to own algorithm-specific payloads/resources; this context owns
     * infrastructure shared by multiple producer families.
     * 
     * Execution lanes are named by independent work domain rather than by class.
     * Multiple producers may therefore share a lane when their work is ordered by
     * dependency, while independent branches remain able to overlap.
     
     * this object grows incrementally:
     *   - completed product storage and common timing utilities
     *   - named VPI execution lanes
     *   - CUDA/TensorRT execution resources
     *   - bounded CPU execution resources
     *
     * Context lifetime is intended to match Runtime lifetime.
     */
    class ExecutionContext {
        public:
            using Clock = std::chrono::steady_clock;
            using TimePoint = Clock::time_point;

            ExecutionContext() = default;
            ~ExecutionContext();

            ExecutionContext(const ExecutionContext&) = delete;
            ExecutionContext& operator=(const ExecutionContext&) = delete;

            ExecutionContext(ExecutionContext&&) = delete;
            ExecutionContext& operator=(ExecutionContext&&) = delete;

            /**
             * Create the shared execution resources owned by this context.
             *
             * This currently initializes only VPI lanes. Later Phase 6 work adds
             * neural/CUDA and CPU execution resources behind the same lifecycle.
             */
            bool initialize();
            bool drain() noexcept;

            /**
             * Release execution resources in deterministic order.
             *
             * Individual algorithm payloads remain owned by their algorithm
             * classes; only shared execution infrastructure is released here.
             */
            void shutdown() noexcept;

            [[nodiscard]] ProductStore& products() noexcept {
                return product_store_;
            }

            [[nodiscard]] const ProductStore& products() const noexcept {
                return product_store_;
            }

            [[nodiscard]] static TimePoint now() noexcept {
                return Clock::now();
            }

            /**
             * Independent image work receives separate left/right lanes so future
             * preprocessing can overlap when no graph dependency orders the eyes.
             */
            [[nodiscard]] parallax::vpi::Stream& imageLeftLane() noexcept {
                return image_left_lane_;
            }

            [[nodiscard]] parallax::vpi::Stream& imageRightLane() noexcept {
                return image_right_lane_;
            }

            /**
             * Shared image preprocessing that is not intrinsically left- or
             * right-eye work uses this lane rather than acquiring a class-specific
             * stream.
             */
            [[nodiscard]] parallax::vpi::Stream& preprocessLane() noexcept {
                return preprocess_lane_;
            }

            /**
             * Stereo receives a dedicated lane because VIC format conversion and
             * OFA disparity form an ordered accelerator chain.
             */

            [[nodiscard]] parallax::vpi::Stream& stereoLane() noexcept {
                return stereo_lane_;
            }

            /**
             * Conventional pose / geometry work may execute independently from
             * stereo once its required image product is complete.
             */

            [[nodiscard]] parallax::vpi::Stream& conventionalPoseLane() noexcept {
                return conventional_pose_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& imageLeftLane() const noexcept {
                return image_left_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& imageRightLane() const noexcept {
                return image_right_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& preprocessLane() const noexcept {
                return preprocess_lane_;
            }

            [[nodiscard]] const parallax::vpi::Stream& conventionalPoseLane() const noexcept {
                return conventional_pose_lane_;
            }

            [[nodiscard]] bool initialized() const noexcept {
                return initialized_;
            }

            /**
             * Dedicated CUDA lane for neural inference/preprocessing.
             * TensorRT model/engine ownership remains with the eventual perception
             * adapter; this context owns only shared execution infrastructure.
             */
            [[nodiscard]] cudaStream_t neuralCudaLane() const noexcept {
                return neural_cuda_lane_;
            }

            /**
             * Optional active TensorRT execution-context handle.
             * Non-owning: the model adapter remains responsible for creating and
             * destroying its TensorRT execution context.
             */
            [[nodiscard]] nvinfer1::IExecutionContext* neuralTensorRtContext() const noexcept {
                return neural_tensorrt_context_;
            }

            void setNeuralTensorRtContext(nvinfer1::IExecutionContext* context) noexcept {
                neural_tensorrt_context_ = context;
            }

            [[nodiscard]] BoundedCpuExecutor& cpuExecutor() noexcept {
                return cpu_executor_;
            }

            [[nodiscard]] const BoundedCpuExecutor& cpuExecutor() const noexcept {
                return cpu_executor_;
            }

            /**
             * Record completion for one specific submitted generation.
             *
             * Native event storage remains owned by ExecutionContext. The returned
             * handle merely leases the selected slot so it cannot be re-recorded
             * while a published Product still references that generation.
             */
            [[nodiscard]] CompletionHandle recordVpiCompletion(VPIStream stream) noexcept;

            [[nodiscard]] CompletionHandle recordCudaCompletion(cudaStream_t stream) noexcept;

            /**
             * Establish an accelerator dependency without blocking the host.
             *
             * VPI lanes may consume both VPI and CUDA completion dependencies because
             * the wrapped lane exposes the CUDA stream used by CUDA submissions.
             */
            [[nodiscard]] bool waitFor(const CompletionHandle& completion,parallax::vpi::Stream& consumer_lane) noexcept;

            /**
             * CUDA-only consumers can directly depend on CUDA completion.
             *
             * A VPI completion cannot be translated into a generic CUDA wait here;
             * consumers of VPI work should use the VPI-lane overload so the dependency
             * remains expressed through VPI.
             */
            [[nodiscard]] bool waitFor(const CompletionHandle& completion, cudaStream_t consumer_stream) noexcept;

            /**
             * Host synchronization is allowed only at an actual CPU observation
             * boundary, recording boundary, explicit API boundary, or shutdown.
             */
            [[nodiscard]] bool waitForHost(const CompletionHandle& completion) noexcept;

        private:
            ProductStore product_store_;

            // These lanes represent execution independence, not producer identity.
            // Event dependencies between them are added later rather than forcing
            // unrelated work through one globally serialized VPI stream.
            parallax::vpi::Stream image_left_lane_;
            parallax::vpi::Stream image_right_lane_;
            parallax::vpi::Stream preprocess_lane_;
            parallax::vpi::Stream stereo_lane_;
            parallax::vpi::Stream conventional_pose_lane_;
            
            // Neural inference gets an independent CUDA lane so lower-rate inference
            // cannot serialize the camera/stereo accelerator path.
            cudaStream_t neural_cuda_lane_ = nullptr;

            // TensorRT execution contexts are model-specific and therefore remain owned
            // by their adapter. ExecutionContext only carries the active non-owning handle.
            nvinfer1::IExecutionContext* neural_tensorrt_context_ = nullptr;

            // CPU work is bounded so a slow conventional consumer cannot create an ever-growing real-time backlog.
            BoundedCpuExecutor cpu_executor_;

            static constexpr std::size_t kCompletionSlotsPerBackend = 16;

            struct VpiCompletionSlot {
                VPIEvent event = nullptr;

                // A live ticket means some published Product generation still depends
                // on this event, so the slot cannot yet be re-recorded.
                std::weak_ptr<const CompletionTicket> lease{};

                std::uint64_t generation = 0;
            };

            struct CudaCompletionSlot {
                cudaEvent_t event = nullptr;
                std::weak_ptr<const CompletionTicket> lease{};
                std::uint64_t generation = 0;
            };

            std::array<VpiCompletionSlot, kCompletionSlotsPerBackend> vpi_completion_slots_{};
            std::array<CudaCompletionSlot, kCompletionSlotsPerBackend> cuda_completion_slots_{};

            std::mutex completion_mutex_;

            bool initialized_ = false;      
    };
}