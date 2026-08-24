#include <parallax/camera/camera_producer.hpp>

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
        /**
         * Capture is staying in Runtime::run() for now during migration
         * calling StereoCamera::capture() here before switching to graph execution would 
         * create a second consumer of the same camera stream+change the capture/release lifecycle
         * 
         * final graph execution will also pair every captured V4L2 buffer wih StereoCamera::release()
         * only after dependent work no longer needs it. that lifetime transition belongs with the 
         * Runtime cutover rather than being hidden inside this first structural producer commit.
         */
        return parallax::core::SubmitResult::NoWork;
    }
}