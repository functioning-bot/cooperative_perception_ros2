#!/usr/bin/env python3
import csv
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from vision_msgs.msg import Detection3DArray
from collections import defaultdict
from math import inf

def tkey(stamp) -> int:
    # Use integer nanoseconds as a matching key
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)

class AccProxyMatch(Node):
    def __init__(self):
        super().__init__('accuracy_proxy_match')
        self.declare_parameter('r1_topic', '/robot_1/detections_3d_map')
        self.declare_parameter('r2_topic', '/robot_2/detections_3d_map')
        self.declare_parameter('fused_topic', '/global_detections')
        self.declare_parameter('csv_path', 'metrics_accuracy_match.csv')
        self.declare_parameter('stamp_tolerance_ns', 5_000_000)  # +/- 5 ms tolerance

        self.r1 = self.get_parameter('r1_topic').value
        self.r2 = self.get_parameter('r2_topic').value
        self.fused = self.get_parameter('fused_topic').value
        self.csv_path = self.get_parameter('csv_path').value
        self.tol = int(self.get_parameter('stamp_tolerance_ns').value)

        # Buffers keyed by stamp (ns)
        self.buf_r1 = defaultdict(list)
        self.buf_r2 = defaultdict(list)
        self.buf_f  = defaultdict(list)

        self.create_subscription(Detection3DArray, self.r1, self.cb_r1, 10)
        self.create_subscription(Detection3DArray, self.r2, self.cb_r2, 10)
        self.create_subscription(Detection3DArray, self.fused, self.cb_f , 10)

        self.csv = open(self.csv_path, 'w', newline='')
        self.wr = csv.writer(self.csv)
        self.wr.writerow(['stamp_ns','r1_count','r2_count','best_single','fused_count','gain'])

        # Periodic sweep to write rows for stamps where all streams are nearby
        self.timer = self.create_timer(0.25, self.sweep)
        self.get_logger().info(f"Accuracy proxy (match-by-stamp) -> {self.csv_path}")

    def _add(self, buf, msg):
        k = tkey(msg.header.stamp)
        buf[k].append(len(msg.detections))

    def cb_r1(self, msg): self._add(self.buf_r1, msg)
    def cb_r2(self, msg): self._add(self.buf_r2, msg)
    def cb_f (self, msg): self._add(self.buf_f , msg)

    def _nearest(self, buf, k):
        # find value at key within tolerance, prefer exact; else nearest
        best_k, best_val, best_d = None, None, inf
        for kk, vals in buf.items():
            d = abs(kk - k)
            if d < best_d:
                best_d, best_k, best_val = d, kk, vals
        if best_d <= self.tol and best_val:
            return best_k, sum(best_val)  # if multiple messages share stamp, sum
        return None, None

    def sweep(self):
        # For each fused stamp, look up nearest r1 and r2 within tolerance
        done_keys = []
        for kf, vf in list(self.buf_f.items()):
            # fused count
            fused_count = sum(vf)
            # nearest matches
            k1, r1c = self._nearest(self.buf_r1, kf)
            k2, r2c = self._nearest(self.buf_r2, kf)
            if r1c is None or r2c is None:
                # Wait until we see neighbors (or widen tolerance via param)
                continue
            best = max(r1c, r2c)
            gain = fused_count - best
            self.wr.writerow([kf, r1c, r2c, best, fused_count, gain])
            done_keys.append(kf)
            # Optional pruning to keep memory bounded
            for kk in set([kf, k1, k2]):
                self.buf_f.pop(kk, None)
                self.buf_r1.pop(kk, None)
                self.buf_r2.pop(kk, None)

    def destroy_node(self):
        try:
            self.csv.flush(); self.csv.close()
        except Exception:
            pass
        return super().destroy_node()

def main():
    rclpy.init()
    n = AccProxyMatch()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
