#include <parallax/core/execution_context.hpp>

#include <algorithm>
#include <cstdint>

namespace parallax::core {

    BoundedCpuExecutor::~BoundedCpuExecutor() {
        shutdown();
    }

    bool BoundedCpuExecutor::initialize(std::size_t worker_count,
                                        std::size_t queue_capacity) {
        if (initialized_) return true;
        if (worker_count == 0 || queue_capacity == 0) {
            return false;
        }

        queue_capacity_ = queue_capacity;
        stopping_ = false;

        try {
            workers_.reserve(worker_count);

            for (std::size_t i = 0; i < worker_count; ++i) {
                workers_.emplace_back(&BoundedCpuExecutor::workerLoop, this);
            }
        } catch (...) {
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    bool BoundedCpuExecutor::submit(std::function<void()> task) {
        if (!task) return false;
        std::lock_guard<std::mutex> lock(mutex_);

        if (!initialized_ || stopping_ ||
            tasks_.size() >= queue_capacity_) {
            return false;
        }

        tasks_.push(std::move(task));

        work_available_.notify_one();
        return true;
    }

    void BoundedCpuExecutor::workerLoop() {
        for (;;) {
            std::function<void()> task; {
                std::unique_lock<std::mutex> lock(mutex_);

                work_available_.wait(lock, [this] {
                    return stopping_ || !tasks_.empty();
                });

                if (stopping_ && tasks_.empty()) {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    void BoundedCpuExecutor::shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (!initialized_ && workers_.empty()) {
                return;
            }

            stopping_ = true;
        }

        work_available_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        workers_.clear();

        {
            std::lock_guard<std::mutex> lock(mutex_);

            while (!tasks_.empty()) {
                tasks_.pop();
            }

            queue_capacity_ = 0;
            stopping_ = false;
            initialized_ = false;
        }
    }

    ExecutionContext::~ExecutionContext() { shutdown(); }

    bool ExecutionContext::initialize() {
        if (initialized_) { return true; }

        /**
         * Image lanes support CUDA/VIC work. They intentionally do not expose OFA
         * because OFA belongs to the stereo execution domain.
         */
        constexpr std::uint64_t image_backends = VPI_BACKEND_CUDA | VPI_BACKEND_VIC;

        /**
         * Stereo needs VIC for pitch/block-linear conversion and OFA for disparity.
         * CUDA remains available for CUDA work sharing the wrapped stream.
         */
        constexpr std::uint64_t stereo_backends = VPI_BACKEND_CUDA | VPI_BACKEND_VIC | VPI_BACKEND_OFA;

        /**
         * Conventional pose currently contains CPU/OpenCV work, but its execution
         * lane is reserved independently so VPI/CUDA preprocessing can later feed
         * it without serializing stereo.
         */
        constexpr std::uint64_t pose_backends = VPI_BACKEND_CUDA | VPI_BACKEND_VIC;

        if (!image_left_lane_.initialize(image_backends) ||
            !image_right_lane_.initialize(image_backends) ||
            !preprocess_lane_.initialize(image_backends) ||
            !stereo_lane_.initialize(stereo_backends) ||
            !conventional_pose_lane_.initialize(pose_backends)) {

            shutdown();
            return false;
        }

        /**
        * Start conservatively with two CPU workers and a small bounded backlog.
        *
        * These are execution-capacity defaults, not producer scheduling policy.
        * Target rates, priority, freshness and supersede later behavior
        */

        if (cudaStreamCreate(&neural_cuda_lane_) != cudaSuccess) {
            shutdown();
            return false;
        }

        if (cudaEventCreateWithFlags(&isp_complete_,
                                    cudaEventDisableTiming) != cudaSuccess) {
            shutdown();
            return false;
        }

        if (vpiEventCreate(0, &preprocess_complete_) != VPI_SUCCESS) {
            shutdown();
            return false;
        }

        if (vpiEventCreate(0, &stereo_complete_) != VPI_SUCCESS) {
            shutdown();
            return false;
        }

        constexpr std::size_t cpu_worker_count = 2;
        constexpr std::size_t cpu_queue_capacity = 8;

        if (!cpu_executor_.initialize(cpu_worker_count, cpu_queue_capacity)) {
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    void ExecutionContext::shutdown() noexcept {
        cpu_executor_.shutdown();
        neural_tensorrt_context_ = nullptr;

        /**
         * Completion events are owned by the execution context.
         *
         * They do not own product buffers or execution lanes; they only represent
         * completion points recorded on those lanes. Destroy them before destroying
         * the streams that will eventually record/wait on them.
         */
        if (stereo_complete_ != nullptr) {
            vpiEventDestroy(stereo_complete_);
            stereo_complete_ = nullptr;
        }

        if (preprocess_complete_ != nullptr) {
            vpiEventDestroy(preprocess_complete_);
            preprocess_complete_ = nullptr;
        }

        if (isp_complete_ != nullptr) {
            cudaEventDestroy(isp_complete_);
            isp_complete_ = nullptr;
        }

        if (neural_cuda_lane_ != nullptr) {
            cudaStreamDestroy(neural_cuda_lane_);
            neural_cuda_lane_ = nullptr;
        }

        /**
         * Explicit draining is added once producers actually submit work to these
         * lanes. For this commit the context establishes completion-handle
         * ownership and lifetime only.
         */
        conventional_pose_lane_.shutdown();
        stereo_lane_.shutdown();
        preprocess_lane_.shutdown();
        image_right_lane_.shutdown();
        image_left_lane_.shutdown();

        initialized_ = false;
    }

} // namespace parallax::core