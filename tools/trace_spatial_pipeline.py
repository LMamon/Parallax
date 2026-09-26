#!/usr/bin/env python3
import collections
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus

WINDOW_S = 5.0
NOMINAL_MS = 1000.0 / 37.75


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


class Stream:
    def __init__(self):
        self.count = 0
        self.prev = None
        self.deltas_ms = []
        self.gaps = 0
        self.stamps = collections.deque(maxlen=512)

    def add(self, msg):
        ns = stamp_ns(msg)
        self.count += 1
        self.stamps.append(ns)
        if self.prev is not None:
            dt = (ns - self.prev) / 1e6
            self.deltas_ms.append(dt)
            generations = max(1, int(round(dt / NOMINAL_MS)))
            self.gaps += max(0, generations - 1)
        self.prev = ns

    def stats(self, seconds):
        hz = self.count / seconds if seconds else 0.0
        mean = statistics.fmean(self.deltas_ms) if self.deltas_ms else 0.0
        maximum = max(self.deltas_ms) if self.deltas_ms else 0.0
        return hz, mean, maximum, self.gaps


class Trace(Node):
    def __init__(self):
        super().__init__("parallax_spatial_trace")
        self.streams = {
            "resize_l": Stream(),
            "resize_r": Stream(),
            "mono_l": Stream(),
            "mono_r": Stream(),
        }
        self.status_count = 0
        self.track_ms = []
        self.callback_ms = []
        self.vo_state = collections.Counter()
        self.started = time.monotonic()

        topics = {
            "resize_l": "/spatial/left/image_rect",
            "resize_r": "/spatial/right/image_rect",
            "mono_l": "/spatial/left/image_rect_mono",
            "mono_r": "/spatial/right/image_rect_mono",
        }
        for key, topic in topics.items():
            self.create_subscription(
                Image, topic,
                lambda msg, k=key: self.streams[k].add(msg),
                qos_profile_sensor_data)

        self.create_subscription(
            VisualSlamStatus, "/visual_slam/status", self.status_cb, 10)
        self.create_timer(WINDOW_S, self.report)

    def status_cb(self, msg):
        self.status_count += 1
        self.track_ms.append(msg.track_execution_time * 1000.0)
        self.callback_ms.append(msg.node_callback_execution_time * 1000.0)
        self.vo_state[int(msg.vo_state)] += 1

    @staticmethod
    def pairing(a, b):
        sa, sb = set(a.stamps), set(b.stamps)
        return len(sa & sb), len(sa - sb), len(sb - sa)

    def report(self):
        now = time.monotonic()
        seconds = now - self.started
        parts = []
        for name in ("resize_l", "resize_r", "mono_l", "mono_r"):
            hz, mean, maximum, gaps = self.streams[name].stats(seconds)
            parts.append(
                f"{name}={hz:.2f}Hz dt={mean:.3f}/{maximum:.3f}ms gaps={gaps}")

        rp = self.pairing(self.streams["resize_l"], self.streams["resize_r"])
        mp = self.pairing(self.streams["mono_l"], self.streams["mono_r"])

        if self.track_ms:
            track_mean = statistics.fmean(self.track_ms)
            track_max = max(self.track_ms)
            cb_mean = statistics.fmean(self.callback_ms)
            cb_max = max(self.callback_ms)
        else:
            track_mean = track_max = cb_mean = cb_max = 0.0

        status_hz = self.status_count / seconds if seconds else 0.0

        self.get_logger().info(
            "TRACE2 " + " ".join(parts) +
            f" resize_pairs={rp[0]}/{rp[1]}/{rp[2]}" +
            f" mono_pairs={mp[0]}/{mp[1]}/{mp[2]}" +
            f" vslam={status_hz:.2f}Hz" +
            f" track={track_mean:.3f}/{track_max:.3f}ms" +
            f" callback={cb_mean:.3f}/{cb_max:.3f}ms" +
            f" vo_state={dict(self.vo_state)}"
        )

        self.streams = {k: Stream() for k in self.streams}
        self.status_count = 0
        self.track_ms.clear()
        self.callback_ms.clear()
        self.vo_state.clear()
        self.started = now


def main():
    rclpy.init()
    node = Trace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
