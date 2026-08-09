#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>
#include <parallax/camera/stereo_camera.hpp>
#include <parallax/camera/arducam_controls.hpp>

#include <parallax/isp/isp.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <vector>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <filesystem>

namespace {

    volatile std::sig_atomic_t running = 1;
    void signalHandler(int) {
        running = 0;
    }
} // namespace

int main() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    parallax::camera::CameraConfig config;

    if (!config.loadFromFile("config/camera/stereo.yaml")) {
        std::cerr << "Failed to load camera configuration\n";
        return EXIT_FAILURE;
    }

    parallax::camera::StereoCamera camera(config);

    if (!camera.initialize()) {
        std::cerr << "Failed to initialize camera\n";
        return EXIT_FAILURE;
    }

    parallax::isp::ISP isp;

    if (!isp.initialize(config)) {
        std::cerr << "Failed to initialize ISP\n";
        return EXIT_FAILURE;
    }

    parallax::camera::RawFrame raw_frame{};
    // int failed_frames = 0;

    // while (running) {
    //     if (!camera.capture(raw_frame)) {
    //         if (++failed_frames >= 100) {
    //             std::cerr << "Camera failed to produce a valid frame\n";
    //             break;
    //         }
    //         continue;
    //     }
    //     failed_frames = 0;
    //     // Inspect the raw frame directly from V4L2

    //     const auto* p = static_cast<const std::uint16_t*>(raw_frame.data);
    //     std::uint16_t mn = 65535;
    //     std::uint16_t mx = 0;

    //     for (size_t i = 0;
    //         i < static_cast<size_t>(raw_frame.width) * raw_frame.height;
    //         ++i)
    //     {
    //         mn = std::min(mn, p[i]);
    //         mx = std::max(mx, p[i]);
    //     }

    //     if (!isp.process(raw_frame)) {
    //         std::cerr << "ISP processing failed\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     std::cout << "CPU raw range: " << mn << " - " << mx << '\n';
    //     std::cout << std::hex << p[1000] << std::dec << '\n';
        
    //     if (!isp.synchronize()) {
    //         std::cerr << "ISP synchronization failed\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     const std::size_t raw_pitch =
    //         static_cast<std::size_t>(raw_frame.width) *
    //         sizeof(std::uint16_t);

    //     std::vector<std::uint16_t> raw_host(
    //         static_cast<std::size_t>(raw_frame.width) *
    //         static_cast<std::size_t>(raw_frame.height)
    //     );

    //     if (!isp.downloadRaw(raw_host.data(), raw_pitch)) {
    //         std::cerr << "Failed to download raw Bayer frame\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     if (!isp.synchronize()) {
    //         std::cerr << "Failed to synchronize raw download\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     std::cout
    //         << "GPU copy matches CPU: "
    //         << (std::memcmp(
    //                 raw_frame.data,
    //                 raw_host.data(),
    //                 raw_frame.bytes
    //             ) == 0)
    //         << '\n';

    //     cv::Mat raw_image(
    //         static_cast<int>(raw_frame.height),
    //         static_cast<int>(raw_frame.width),
    //         CV_16UC1,
    //         raw_host.data(),
    //         raw_pitch
    //     );

    //     if (!cv::imwrite("gpu_raw.png", raw_image)) {
    //         std::cerr << "Failed to save gpu_raw.png\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     if (!camera.release(raw_frame)) {
    //         std::cerr << "Failed to release camera frame\n";
    //         break;
    //     }

    //     std::cout << "Saved gpu_raw.png\n";
    //     break;
    // } 1
    int failed_frames = 0;

    const std::filesystem::path dataset_root = "calibration_dataset";
    const std::string calibration_side = "left_side";

    const int position = 1;

    const std::filesystem::path left_dir = dataset_root / calibration_side / "left";
    const std::filesystem::path right_dir = dataset_root / calibration_side / "right";

    std::filesystem::create_directories(left_dir);
    std::filesystem::create_directories(right_dir);

    while (running) {
        if (!camera.capture(raw_frame)) {
            if (++failed_frames >= 10) {
                std::cerr << "Camera failed to produce a valid frame\n";
                break;
            }
            continue;
        }
        const auto* raw = static_cast<const std::uint16_t*>(raw_frame.data);

        const std::size_t pixel_count =
            static_cast<std::size_t>(raw_frame.width) *
            raw_frame.height;

        uint16_t minimum = UINT16_MAX;
        uint16_t maximum = 0;
        uint64_t sum = 0;

        for (std::size_t i = 0; i < pixel_count; ++i) {
            minimum = std::min(minimum, raw[i]);
            maximum = std::max(maximum, raw[i]);
            sum += raw[i];
        }

        std::cout
            << "Raw mean: " << static_cast<double>(sum) / pixel_count
            << "\nRaw min: " << minimum
            << "\nRaw max: " << maximum
            << std::endl;

        failed_frames = 0;

        if (!isp.process(raw_frame)) {
            std::cerr << "ISP processing failed\n";
            camera.release(raw_frame);
            break;
        }

        if (!isp.synchronize()) {
            std::cerr << "ISP synchronization failed\n";
            camera.release(raw_frame);
            break;
        }

        const parallax::isp::StereoRgbFrame& rgb = isp.output();

        const std::size_t host_pitch = static_cast<std::size_t>(rgb.width) *
                                        parallax::isp::StereoRgbFrame::Channels;

        const std::size_t host_size = host_pitch * static_cast<std::size_t>(rgb.height);

        std::vector<std::uint8_t> left_host(host_size);
        std::vector<std::uint8_t> right_host(host_size);

        if (!rgb.left.downloadAsync(
                left_host.data(),
                host_pitch,
                nullptr)) {
            std::cerr << "Failed to download left RGB frame\n";
            camera.release(raw_frame);
            break;
        }
        if (!rgb.right.downloadAsync(
                right_host.data(),
                host_pitch,
                nullptr)) {
            std::cerr << "Failed to download right RGB frame\n";
            camera.release(raw_frame);
            break;
        }

        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "Failed to synchronize RGB download\n";
            camera.release(raw_frame);
            break;
        }

        cv::Mat left_rgb(static_cast<int>(rgb.height),
                        static_cast<int>(rgb.width),
                        CV_8UC3,
                        left_host.data(),
                        host_pitch);

        cv::Mat right_rgb(static_cast<int>(rgb.height),
                        static_cast<int>(rgb.width),
                        CV_8UC3,
                        right_host.data(),
                        host_pitch);    

        cv::Mat left_bgr;
        cv::Mat right_bgr;
        cv::cvtColor(left_rgb, left_bgr, cv::COLOR_RGB2BGR);
        cv::cvtColor(right_rgb, right_bgr, cv::COLOR_RGB2BGR);

        int shot = 0;
        std::string filename;

        while (true) {
            filename = "image" + std::to_string(position) +
                        "." + std::to_string(shot) +
                        ".png";
        
            const bool left_exists = std::filesystem::exists(left_dir / filename);
            const bool right_exists = std::filesystem::exists(right_dir / filename);

            if (!left_exists && !right_exists) break;
            ++shot;
        }

        const std::filesystem::path left_path = left_dir / filename;
        const std::filesystem::path right_path = right_dir / filename;
        
        if (!cv::imwrite(left_path.string(), left_bgr)) {
            std::cerr << "Failed to save" << left_path << '\n';
            camera.release(raw_frame);
            break;
        }
        if (!cv::imwrite(right_path.string(), right_bgr)) {
            std::cerr << "Failed to save " << right_path << '\n';
            camera.release(raw_frame);
            break;
        }

        if (!camera.release(raw_frame)) {
            std::cerr << "Failed to release camera frame\n";
            break;
        }

        std::cout
            << "Saved stereo pair: " << filename
            << "\n  left:  " << left_path
            << "\n  right: " << right_path
            << "\nexposure=" << config.exposure
            << ", analogue_gain=" << config.analogue_gain
            << ", frame_rate=" << config.frame_rate 
            << '\n';

        // Temporary: save one frame and exit.
        // Remove this when the live pipeline is ready.
        break;
    }
    // int failed_frames = 0;

    // while (running) {
    //     if (!camera.capture(raw_frame)) {
    //         if (++failed_frames >= 100) {
    //             std::cerr << "Camera failed to produce a valid frame\n";
    //             break;
    //         }
    //         continue;
    //     }

    //     failed_frames = 0;

    //     const auto* p = static_cast<const std::uint16_t*>(raw_frame.data);
    //     for (int y = 0; y < 4; ++y) {
    //         for (int x = 0; x < 8; ++x) {
    //             std::cout
    //                 << std::hex
    //                 << std::setw(4)
    //                 << std::setfill('0')
    //                 << p[y * raw_frame.width + x]
    //                 << " ";
    //         }
    //         std::cout << '\n';
    //     }

    //     uint16_t mn = 65535;
    //     uint16_t mx = 0;
    //     size_t min_idx = 0;
    //     size_t max_idx = 0;
    //     uint64_t sum = 0;

    //     const size_t N = static_cast<size_t>(raw_frame.width) * raw_frame.height;

    //     for (size_t i = 0; i < N; ++i) {
    //         if (p[i] < mn) {
    //             mn = p[i];
    //             min_idx = i;
    //         }

    //         if (p[i] > mx) {
    //             mx = p[i];
    //             max_idx = i;
    //         }
    //         sum += p[i];
    //     }

    //     double mean = static_cast<double>(sum) / N;
    //     std::cout << "mean=" << mean << '\n';
    //     std::cout
    //         << "min=" << mn
    //         << " at "
    //         << min_idx
    //         << "\n";

    //     std::cout
    //         << "max=" << mx
    //         << " at "
    //         << max_idx
    //         << "\n";

    //     if (!isp.process(raw_frame)) {
    //         std::cerr << "ISP processing failed\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     if (!isp.synchronize()) {
    //         std::cerr << "ISP synchronization failed\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     const parallax::isp::StereoRgbFrame& rgb = isp.output();

    //     const std::size_t host_pitch = static_cast<std::size_t>(rgb.width) *parallax::isp::StereoRgbFrame::Channels;

    //     std::vector<std::uint8_t> left_host(host_pitch * static_cast<std::size_t>(rgb.height));

    //     if (!rgb.left.downloadAsync(
    //             left_host.data(),
    //             host_pitch,
    //             nullptr)) {
    //         std::cerr << "Failed to download left RGB frame\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     /*
    //     * downloadAsync() enqueues the copy. Synchronize before reading the
    //     * host buffer.
    //     */
    //     if (cudaDeviceSynchronize() != cudaSuccess) {
    //         std::cerr << "Failed to synchronize RGB download\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     cv::Mat left_rgb(
    //         static_cast<int>(rgb.height),
    //         static_cast<int>(rgb.width),
    //         CV_8UC3,
    //         left_host.data(),
    //         host_pitch
    //     );

    //     /*
    //     * OpenCV imwrite expects BGR channel ordering.
    //     */
    //     cv::Mat left_bgr;
    //     cv::cvtColor(left_rgb, left_bgr, cv::COLOR_RGB2BGR);

    //     if (!cv::imwrite("left.png", left_bgr)) {
    //         std::cerr << "Failed to save left.png\n";
    //         camera.release(raw_frame);
    //         break;
    //     }

    //     if (!camera.release(raw_frame)) {
    //         std::cerr << "Failed to release camera frame\n";
    //         break;
    //     }

    //     std::cout << "Saved left.png\n";
    //     break;
    // } 2

    // while (running) {
    //     if (!camera.capture(raw_frame)) continue;
    //     if (!isp.process(raw_frame)) {
    //         std::cerr << "ISP processing failed\n";

    //         if (!camera.release(raw_frame)) {
    //             std::cerr << "Failed to release camera frame\n";
    //             break;
    //         }
    //         continue;
    //     }

    //     /*
    //      * The upload and demosaic kernel are asynchronous.
    //      *
    //      * The camera frame cannot be returned to V4L2 until the GPU has
    //      * finished reading its MMAP-backed data.
    //      */
    //     if (!isp.synchronize()) {
    //         std::cerr << "ISP synchronization failed\n";

    //         camera.release(raw_frame);
    //         break;
    //     }

    //     if (!camera.release(raw_frame)) {
    //         std::cerr << "Failed to release camera frame\n";
    //         break;
    //     }

    //     const parallax::isp::StereoRgbFrame& rgb = isp.output();

    //     /*
    //      * The RGB images remain resident on the GPU:
    //      *
    //      * rgb.left.data()
    //      * rgb.right.data()
    //      *
    //      * Next:
    //      * stereo rectification consumes these GPU buffers directly.
    //      */
    //     (void)rgb;
    // }

    isp.shutdown();
    camera.shutdown();

    return EXIT_SUCCESS;
}