#pragma once
#include <parallax/core/completion.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/efficientvit_sam.hpp>
#include <parallax/perception/nanoowl_bridge.hpp>
#include <parallax/tracking/dcf_tracker.hpp>
#include <interfaces/msg/detection_set.hpp>
#include <interfaces/msg/perception_state.hpp>
#include <interfaces/msg/track2_d.hpp>
#include <interfaces/srv/clear_prompt.hpp>
#include <interfaces/srv/set_prompt.hpp>
#include <interfaces/srv/start_tracking.hpp>
#include <interfaces/srv/stop_tracking.hpp>

#include <rclcpp/rclcpp.hpp>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace parallax::perception {
    class PerceptionRuntime final {
        public:
            struct Frame {
                std::shared_ptr<const isp::RectifiedStereoFrame> rgb; 
                std::shared_ptr<const void> owner; 
                
                core::CompletionHandle ready{}; 
                std_msgs::msg::Header header{}; 
                
                uint64_t sequence = 0; 
                
                bool valid()const{
                    return rgb&&owner&&ready.valid()&&sequence;
                }
            };

            PerceptionRuntime(rclcpp::Node&,core::ExecutionContext&); 
            ~PerceptionRuntime(); 
            
            void start(); 
            void stop(); 
            void submit(Frame);
        
        private:
            enum class State:uint8_t{Tentative, Tracking, Lost, Reacquiring}; 
            
            void loop(); 
            void process(Frame); 
            bool detector(); 
            bool segmenter(); 
            void detect(const Frame&,bool); 
            void track(const Frame&); 
            void pubState();

            rclcpp::Node& n_; 
            core::ExecutionContext& ctx_; 
            NanoOwlBridge owl_; 
            EfficientVitSam sam_; 
            tracking::DcfTracker dcf_;

            std::filesystem::path owl_engine_,sam_enc_,sam_dec_; 
            std::mutex m_; 
            std::condition_variable cv_; 
            std::optional<Frame> pending_; 
            std::thread worker_; 
            std::atomic<bool> running_{false};

            std::string prompt_, target_; 
            uint64_t prompt_rev_ = 0, target_rev_ = 0, track_id_ = 0; 
            bool want_mask_ = false, want_track_ = false; 
            State state_ = State::Lost; 
            unsigned lost_ = 0;

            rclcpp::Publisher<interfaces::msg::DetectionSet>::SharedPtr det_pub_; 
            rclcpp::Publisher<interfaces::msg::Track2D>::SharedPtr track_pub_; 
            rclcpp::Publisher<interfaces::msg::PerceptionState>::SharedPtr state_pub_;

            rclcpp::Service<interfaces::srv::SetPrompt>::SharedPtr set_; 
            rclcpp::Service<interfaces::srv::ClearPrompt>::SharedPtr clear_; 
            rclcpp::Service<interfaces::srv::StartTracking>::SharedPtr start_track_; 
            rclcpp::Service<interfaces::srv::StopTracking>::SharedPtr stop_track_;
    };
}
