#include <parallax/camera/camera_producer.hpp>

#include <chrono>
#include <memory>

namespace parallax::camera {
    CameraProducer::CameraProducer(StereoCamera& camera, parallax::core::ProductStore& store) :
                                                        camera_(camera),
                                                        store_(store) {}
    
    std::string_view CameraProducer::name() const noexcept {
        return "camera";
    }

    const std::vector<parallax::core::ProductId>&CameraProducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<parallax::core::ProductId>&CameraProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy CameraProducer::execution_policy() const noexcept {
        // V4L2 capture is host-driven. The image data may feed accelerator work
        // downstream, but capture itself does not belong to a VPI/CUDA lane.
        return {parallax::core::ResourceAffinity::Cpu, false};
    }

    parallax::core::SubmitResult CameraProducer::submit() {
        parallax::camera::RawFrame frame{};
        if (!camera_.capture(frame)) {
            return parallax::core::SubmitResult::Failed;
        }

        /**
         * RawFrame itself is a small descriptor. Its data pointer still refers to
         * the V4L2 buffer owned by StereoCamera.
         *
         * Attach release() to the shared payload lifetime so the capture buffer
         * remains valid while downstream products still reference this source frame.
         * Replacing/clearing the ProductStore entry eventually drops this reference
         * and returns the V4L2 buffer to the camera.
         */
        auto payload = std::shared_ptr<const parallax::camera::RawFrame>(
            new parallax::camera::RawFrame(frame),
                [this](const parallax::camera::RawFrame* stored) {
                    camera_.release(*stored);
                    delete stored;
                }
        );

        parallax::core::ProductMetadata metadata{};
        metadata.sequence = next_sequence_++;

        // preserve+use monotonic V4L2 timestamps on the active capture path.
        metadata.timestamp = std::chrono::steady_clock::time_point{frame.timestamp};
        metadata.valid = true;

        store_.publish(parallax::core::make_product(parallax::core::ProductId::RawStereo,
                                                    metadata,
                                                    std::move(payload)));

        return parallax::core::SubmitResult::Submitted;
    }
}