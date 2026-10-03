#include <parallax/perception/perception_runtime.hpp>

#include <algorithm>

namespace parallax::perception {

    using SetPrompt = interfaces::srv::SetPrompt;
    using ClearPrompt = interfaces::srv::ClearPrompt;
    using StartTracking = interfaces::srv::StartTracking;
    using StopTracking = interfaces::srv::StopTracking;

    PerceptionRuntime::PerceptionRuntime(rclcpp::Node& n,core::ExecutionContext& c) : n_(n), ctx_(c) {
        owl_engine_ = n_.declare_parameter<std::string>("perception.nanoowl_engine");
        sam_enc_ = n_.declare_parameter<std::string>("perception.sam_encoder_engine");
        sam_dec_ = n_.declare_parameter<std::string>("perception.sam_decoder_engine");
        
        auto q = rclcpp::SensorDataQoS().keep_last(1); 
        det_pub_ = n_.create_publisher<interfaces::msg::DetectionSet>("/perception/detections", q); 
        track_pub_ = n_.create_publisher<interfaces::msg::Track2D>("/perception/track", q); 
        state_pub_ = n_.create_publisher<interfaces::msg::PerceptionState>("/perception/state", 
                                                                                    rclcpp::QoS(1).transient_local().reliable());
        
        set_ = n_.create_service<interfaces::srv::SetPrompt>("/perception/set_prompt", 
                                                                        [this](SetPrompt::Request::SharedPtr a, SetPrompt::Response::SharedPtr b) {
                                                                            if (a->prompt.empty()) {
                                                                                b->accepted = false;
                                                                                b->message = "empty prompt";
                                                                                return;
                                                                            } {
                                                                                std::lock_guard l(m_);
                                                                                prompt_ = a->prompt; 
                                                                                want_mask_ = a->segment;
                                                                                b->revision = ++prompt_rev_;
                                                                            } 
                                                                            b->accepted = true;
                                                                            b->message = "accepted";
                                                                            pubState();
                                                                        }
                                                                    );

        clear_ = n_.create_service<interfaces::srv::ClearPrompt>("/perception/clear_prompt",
                                                                            [this](ClearPrompt::Request::SharedPtr a, ClearPrompt::Response::SharedPtr b) {
                                                                                {
                                                                                    std::lock_guard l(m_);
                                                                                    prompt_.clear();
                                                                                    want_mask_ = false;
                                                                                    b->revision = ++prompt_rev_;
                                                                                } 
                                                                                b->accepted = true;
                                                                                b->message = "prompt cleared; tracker unchanged";
                                                                                pubState();
                                                                            }
                                                                        );

        start_track_ = n_.create_service<interfaces::srv::StartTracking>("/tracking/start",
                                                                                    [this](StartTracking::Request::SharedPtr a, StartTracking::Response::SharedPtr b) {
                                                                                        if (a->target.empty()) {
                                                                                            b->accepted = false;
                                                                                            b->message = "empty target";
                                                                                            return;
                                                                                        } {
                                                                                            std::lock_guard l(m_);
                                                                                            target_ = a->target;
                                                                                            want_track_ = true;
                                                                                            state_ = State::Tentative;
                                                                                            lost_ = 0;
                                                                                            b->revision = ++target_rev_;
                                                                                            ++track_id_;
                                                                                        } 
                                                                                        dcf_.reset();
                                                                                        b->accepted = true;
                                                                                        b->message = "acquisition requested";
                                                                                        pubState();
                                                                                    }
                                                                                );

        stop_track_ = n_.create_service<interfaces::srv::StopTracking>("/tracking/stop",
                                                                                    [this](StopTracking::Request::SharedPtr a, StopTracking::Response::SharedPtr b) {
                                                                                        {
                                                                                            std::lock_guard l(m_);
                                                                                            target_.clear();
                                                                                            want_track_ = false;
                                                                                            state_ = State::Lost;
                                                                                            lost_ = 0;
                                                                                        } 
                                                                                        dcf_.reset();
                                                                                        b->accepted = true;
                                                                                        b->message = "stopped";
                                                                                        pubState();
                                                                                    }
                                                                                ); 
    }
        
    PerceptionRuntime::~PerceptionRuntime(){
        stop();
    } 
        
    void PerceptionRuntime::start() {
        if (running_.exchange(true)) return;worker_ = std::thread(&PerceptionRuntime::loop, this);
        pubState();
    } 
    
    void PerceptionRuntime::stop() {
        if (!running_.exchange(false)) return;
        
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        
        dcf_.reset();
        sam_.shutdown();
        owl_.shutdown();
    }

    void PerceptionRuntime::submit(Frame f) {
        if (!running_ || !f.valid()) return;
        {
            std::lock_guard l(m_);
            pending_ = std::move(f);
        }
        cv_.notify_one();
    }
    
    void PerceptionRuntime::loop() {
        while (running_ && rclcpp::ok()) {
            Frame f;
            {
                std::unique_lock l(m_);
                cv_.wait(l, [&]{
                    return !running_ || pending_;
                });
                
                if (!running_) break;
                f = std::move(*pending_);
                pending_.reset();
            } 
            process(std::move(f));
        }}
    
    bool PerceptionRuntime::detector() {
        return owl_.initialized() || owl_.initialize(owl_engine_);
    } 
    
    bool PerceptionRuntime::segmenter() {
        return sam_.initialized() || sam_.initialize(sam_enc_, sam_dec_);
    }
    
    void PerceptionRuntime::process(Frame f) {
        if (!ctx_.waitForHost(f.ready)) return;
        bool wt, need;
        {
            std::lock_guard l(m_);
            wt = want_track_;
            need = !prompt_.empty() || (want_track_ && !dcf_.initialized());
        }
        if (wt && dcf_.initialized()) track(f);
        if (need) detect(f, wt && !dcf_.initialized());
    }
    
    void PerceptionRuntime::detect(const Frame& f, bool reacq) {
        std::string q;
        uint64_t rev;
        bool mask;
        {
            std::lock_guard l(m_);
            q = reacq ? target_ : prompt_;
            rev = reacq ? target_rev_ : prompt_rev_;
            mask = want_mask_ && !reacq;
        }
        if (q.empty() || !detector() || !owl_.setQuery(q, rev)) return;
        
        DetectionSet d;
        auto s = ctx_.neuralCudaLane();
        
        if (!owl_.predict(*f.rgb, s, d)) return;
        
        interfaces::msg::DetectionSet m;
        m.header = f.header;
        m.query = d.query;
        m.query_revision = d.query_revision;
        m.observation_sequence = f.sequence;
        
        for (size_t i = 0; i < d.boxes.size(); ++i) {
            interfaces::msg::Detection2D x;
            x.x = d.boxes[i].x;
            x.y = d.boxes[i].y;
            x.width = d.boxes[i].width;
            x.height = d.boxes[i].height;
            x.confidence = d.scores[i];
            x.label_index = d.labels[i];
            m.detections.push_back(x);
        }
        
        det_pub_->publish(m);
        
        if (d.empty()) return;
        size_t k = std::distance(d.scores.begin(), std::max_element(d.scores.begin(), d.scores.end()));
        bool acquire;
        {
            std::lock_guard l(m_);
            acquire = want_track_ && !dcf_.initialized() && q == target_ && rev == target_rev_;
        }

        if (acquire && dcf_.initialize(f.rgb->left, d.boxes[k])) {
            {
                std::lock_guard l(m_);
                state_ = State::Tracking;
                lost_ = 0;
            } 
            pubState();
        } 
        
        if (mask && segmenter()) {
            EfficientVitSamResult result;
            (void)sam_.segment(*f.rgb, d.boxes[k], s, result);
            /* authoritative CUDA mask stays in-process; spatial association attaches here */
        }}
    
    void PerceptionRuntime::track(const Frame& f) {
        auto r = dcf_.update(f.rgb->left);
        if (r.tracked) {
            interfaces::msg::Track2D m;
            m.header = f.header;
            {
                std::lock_guard l(m_);
                state_ = State::Tracking;
                lost_ = 0;
                m.target = target_;
                m.target_revision = target_rev_;
                m.track_id = track_id_;
                m.lifecycle = 1;
            } 
            
            m.observation_sequence = f.sequence;
            m.x = r.box.x;
            m.y = r.box.y;
            m.width = r.box.width;
            m.height = r.box.height;
            m.quality = r.response;
            track_pub_->publish(m);
            return;
        } 
        
        bool reset = false;
        {
            std::lock_guard l(m_);
            state_ = State::Lost;
            
            if (++lost_ >= 3) {
                state_ = State::Reacquiring;
                reset = true;
            }
        } 
            
        if (reset) dcf_.reset();
        pubState();
    }
    
    void PerceptionRuntime::pubState() {
        interfaces::msg::PerceptionState x;
        x.header.stamp = n_.now();
        
        std::lock_guard l(m_);
        x.prompt = prompt_;
        x.prompt_revision = prompt_rev_;
        x.detector_active = owl_.initialized();
        x.segmenter_active = sam_.initialized();
        x.tracking_requested = want_track_;
        x.tracking_target = target_;
        x.tracking_revision = target_rev_;
        x.tracker_lifecycle = static_cast<uint8_t>(state_);
        state_pub_->publish(x);
    }
}
