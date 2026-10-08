#include <rclcpp/rclcpp.hpp>

#include <interfaces/msg/detection_set.hpp>
#include <interfaces/msg/track2_d.hpp>

#include <visualization/msg/color.hpp>
#include <visualization/msg/image_annotations.hpp>
#include <visualization/msg/key_value_pair.hpp>
#include <visualization/msg/point2.hpp>
#include <visualization/msg/points_annotation.hpp>
#include <visualization/msg/text_annotation.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace viz = visualization::msg;

namespace {

viz::Color color(double r, double g, double b, double a) {
    viz::Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    c.a = a;
    return c;
}

viz::Point2 point(double x, double y) {
    viz::Point2 p;
    p.x = x;
    p.y = y;
    return p;
}

void addMetadata(viz::ImageAnnotations& msg,
                 const std::string& key,
                 const std::string& value) {
    viz::KeyValuePair kv;
    kv.key = key;
    kv.value = value;
    msg.metadata.push_back(std::move(kv));
}

bool validBox(float x, float y, float width, float height) {
    return std::isfinite(x) &&
           std::isfinite(y) &&
           std::isfinite(width) &&
           std::isfinite(height) &&
           width > 0.0F &&
           height > 0.0F;
}

class VisualizationNode final : public rclcpp::Node {
public:
    VisualizationNode() : Node("visualization") {
        const auto qos = rclcpp::SensorDataQoS().keep_last(1);

        detection_pub_ =
            create_publisher<viz::ImageAnnotations>(
                "/perception/annotations", qos);

        track_pub_ =
            create_publisher<viz::ImageAnnotations>(
                "/tracking/annotations", qos);

        detection_sub_ =
            create_subscription<interfaces::msg::DetectionSet>(
                "/perception/detections", qos,
                [this](interfaces::msg::DetectionSet::ConstSharedPtr msg) {
                    publishDetections(*msg);
                });

        track_sub_ =
            create_subscription<interfaces::msg::Track2D>(
                "/perception/track", qos,
                [this](interfaces::msg::Track2D::ConstSharedPtr msg) {
                    publishTrack(*msg);
                });
    }

private:
    void publishDetections(const interfaces::msg::DetectionSet& in) {
        viz::ImageAnnotations out;
        out.timestamp = in.header.stamp;

        addMetadata(out, "image_space", "rectified_left");
        addMetadata(out, "source_sequence",
                    std::to_string(in.observation_sequence));
        addMetadata(out, "query_revision",
                    std::to_string(in.query_revision));

        for (const auto& d : in.detections) {
            if (!validBox(d.x, d.y, d.width, d.height)) {
                continue;
            }

            const double x0 = d.x;
            const double y0 = d.y;
            const double x1 = d.x + d.width;
            const double y1 = d.y + d.height;

            viz::PointsAnnotation box;
            box.timestamp = in.header.stamp;
            box.type = viz::PointsAnnotation::LINE_LOOP;
            box.thickness = 3.0;
            box.points = {
                point(x0, y0),
                point(x1, y0),
                point(x1, y1),
                point(x0, y1),
            };
            box.outline_color = color(0.0, 1.0, 0.0, 1.0);
            box.fill_color = color(0.0, 0.0, 0.0, 0.0);
            out.points.push_back(std::move(box));

            viz::TextAnnotation label;
            label.timestamp = in.header.stamp;
            label.position = point(x0, std::max(18.0, y0 - 4.0));
            label.font_size = 18.0;

            std::ostringstream text;
            text << in.query << ' '
                 << std::fixed << std::setprecision(2)
                 << d.confidence;
            label.text = text.str();
            label.text_color = color(1.0, 1.0, 1.0, 1.0);
            label.background_color = color(0.0, 0.0, 0.0, 0.65);
            out.texts.push_back(std::move(label));
        }

        // Empty messages deliberately clear stale annotations.
        detection_pub_->publish(out);
    }

    void publishTrack(const interfaces::msg::Track2D& in) {
        if (!validBox(in.x, in.y, in.width, in.height)) {
            return;
        }

        viz::ImageAnnotations out;
        out.timestamp = in.header.stamp;

        addMetadata(out, "image_space", "rectified_left");
        addMetadata(out, "source_sequence",
                    std::to_string(in.observation_sequence));

        const double x0 = in.x;
        const double y0 = in.y;
        const double x1 = in.x + in.width;
        const double y1 = in.y + in.height;

        viz::PointsAnnotation box;
        box.timestamp = in.header.stamp;
        box.type = viz::PointsAnnotation::LINE_LOOP;
        box.thickness = 3.0;
        box.points = {
            point(x0, y0),
            point(x1, y0),
            point(x1, y1),
            point(x0, y1),
        };
        box.outline_color = color(1.0, 0.7, 0.0, 1.0);
        box.fill_color = color(0.0, 0.0, 0.0, 0.0);
        out.points.push_back(std::move(box));

        viz::TextAnnotation label;
        label.timestamp = in.header.stamp;
        label.position = point(x0, std::max(18.0, y0 - 4.0));
        label.font_size = 18.0;

        std::ostringstream text;
        text << "track " << in.track_id
             << ' ' << in.target
             << ' ' << std::fixed << std::setprecision(2)
             << in.quality;
        label.text = text.str();
        label.text_color = color(1.0, 1.0, 1.0, 1.0);
        label.background_color = color(0.0, 0.0, 0.0, 0.65);
        out.texts.push_back(std::move(label));

        track_pub_->publish(out);
    }

    rclcpp::Publisher<viz::ImageAnnotations>::SharedPtr detection_pub_;
    rclcpp::Publisher<viz::ImageAnnotations>::SharedPtr track_pub_;

    rclcpp::Subscription<interfaces::msg::DetectionSet>::SharedPtr
        detection_sub_;
    rclcpp::Subscription<interfaces::msg::Track2D>::SharedPtr
        track_sub_;
};

}  // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VisualizationNode>());
    rclcpp::shutdown();
    return 0;
}
