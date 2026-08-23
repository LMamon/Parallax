#include <parallax/visualization/video_encoder.hpp>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <cstring>
#include <iostream>
#include <string>

namespace parallax::visualization {
    VideoEncoder::~VideoEncoder() { shutdown(); }

    bool VideoEncoder::initialize(std::uint32_t width, std::uint32_t height, std::uint32_t fps) {
        if (initialized_) return true;
        
        if (width == 0 || height == 0 || fps == 0) {
            std::cerr << "Invalid video encoder dimensions/FPS\n";
            return false;
        }

        gst_init(nullptr, nullptr);

        width_ = width;
        height_ = height;
        fps_ = fps;

        const std::string pipeline_description = "appsrc name=source is-live=true format=time "
                                                "caps=video/x-raw,format=RGB,width=" + std::to_string(width_) +
                                                ",height=" + std::to_string(height_) +
                                                ",framerate=" + std::to_string(fps_) + "/1 "
                                                "! videoconvert "
                                                "! x264enc tune=zerolatency speed-preset=ultrafast "
                                                "key-int-max=" + std::to_string(fps_) + " "
                                                "! h264parse "
                                                "! video/x-h264,stream-format=byte-stream,alignment=au "
                                                "! appsink name=sink sync=false max-buffers=1 drop=true";

        GError* error = nullptr;
        pipeline_ = gst_parse_launch(pipeline_description.c_str(), &error);

        if (pipeline_ == nullptr) {
            std::cerr << "Failed to create GStreamer pipeline";

            if (error != nullptr) {
                std::cerr << ": " << error->message;
                g_error_free(error);
            }

            std::cerr << "\n";
            shutdown();
            return false;
        }

        appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "source");
        appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");

        if (appsrc_ == nullptr || appsink_ == nullptr) {
            std::cerr << "Failed to retrieve GStreamer appsrc/appsink\n";
            shutdown();
            return false;
        }

        const GstStateChangeReturn state_result = gst_element_set_state(pipeline_, GST_STATE_PLAYING);

        if (state_result == GST_STATE_CHANGE_FAILURE) {
            std::cerr << "Failed to start GStreamer encoder pipeline\n";
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }


    bool VideoEncoder::encode(const std::uint8_t* rgb, std::size_t size, std::vector<std::byte>& encoded) {
        if (!initialized_ || rgb == nullptr) { return false; }
        const std::size_t expected = static_cast<std::size_t>(width_) * height_ * 3;

        if (size != expected) {
            std::cerr << "Unexpected RGB frame size\n";
            return false;
        }

        GstBuffer* input = gst_buffer_new_allocate(nullptr, size, nullptr);

        if (input == nullptr) {
            std::cerr << "Failed to allocate GStreamer input buffer\n";
            return false;
        }

        GstMapInfo input_map{};

        if (!gst_buffer_map(input, &input_map, GST_MAP_WRITE)) {
            gst_buffer_unref(input);
            return false;
        }

        std::memcpy(input_map.data, rgb, size);
        gst_buffer_unmap(input, &input_map);

        const GstClockTime duration = gst_util_uint64_scale_int(1, GST_SECOND, static_cast<int>(fps_));
        GST_BUFFER_PTS(input) = frame_index_ * duration;

        GST_BUFFER_DTS(input) = GST_CLOCK_TIME_NONE;
        GST_BUFFER_DURATION(input) = duration;

        ++frame_index_;

        const GstFlowReturn push_result = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), input);
        // appsrc owns input after push_buffer()
        if (push_result != GST_FLOW_OK) {
            std::cerr << "Failed to push RGB frame into encoder\n";
            return false;
        }

        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink_));

        if (sample == nullptr) {
            std::cerr << "failed to pull encoded H.264 frame\n";
            return false;
        }

        GstBuffer* output = gst_sample_get_buffer(sample);

        GstMapInfo output_map{};
        if (!gst_buffer_map(output, &output_map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            return false;
        }

        encoded.resize(output_map.size);

        std::memcpy(encoded.data(), output_map.data, output_map.size);
        gst_buffer_unmap(output, &output_map);
        gst_sample_unref(sample);

        return true;
    }

    void VideoEncoder::shutdown() {
        if (pipeline_ != nullptr) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        }

        if (appsrc_ != nullptr) {
            gst_object_unref(appsrc_);
            appsrc_ = nullptr;
        }

        if (appsink_ != nullptr) {
            gst_object_unref(appsink_);
            appsink_ = nullptr;
        }

        if (pipeline_ != nullptr) {
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }

        width_ = 0;
        height_ = 0;
        fps_ = 0;
        frame_index_ = 0;

        initialized_ = false;
    }
}