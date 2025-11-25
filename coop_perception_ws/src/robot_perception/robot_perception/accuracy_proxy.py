#!/usr/bin/env python3
import csv
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from vision_msgs.msg import Detection3DArray

def t_to_sec(t: Time) -> float:
    s, ns = t.seconds_nanoseconds()
    return s + ns * 1e-9

class AccProxy(Node):
    def __init__(self):
        super().__init__('accuracy_proxy')

        # Declare params (so -p use_sim_time:=true works)

        self.declare_parameter('r1_topic', '/robot_1/detections_3d_map')
        self.declare_parameter('r2_topic', '/robot_2/detections_3d_map')
        self.declare_parameter('fused_topic', '/global_detections')
        self.declare_parameter('window_sec', 1.0)
        self.declare_parameter('csv_path', 'metrics_accuracy_proxy.csv')
        self.declare_parameter('tick_hz', 2.0)  # evaluation cadence

        self.r1 = self.get_parameter('r1_topic').value
        self.r2 = self.get_parameter('r2_topic').value
        self.fused = self.get_parameter('fused_topic').value
        self.window = float(self.get_parameter('window_sec').value)

        self.buf = {'r1': [], 'r2': [], 'fused': []}
        self.csv = open(self.get_parameter('csv_path').value, 'w', newline='')
        self.wr = csv.writer(self.csv)
        self.wr.writerow(['ros_time_s', 'r1_count', 'r2_count', 'best_single', 'fused_count', 'gain'])

        self.create_subscription(Detection3DArray, self.r1, lambda m: self.cb('r1', m), 10)
        self.create_subscription(Detection3DArray, self.r2, lambda m: self.cb('r2', m), 10)
        self.create_subscription(Detection3DArray, self.fused, lambda m: self.cb('fused', m), 10)

        dt = 1.0 / float(self.get_parameter('tick_hz').value)
        self.timer = self.create_timer(dt, self.tick)

        self.get_logger().info(f"Accuracy proxy on {self.r1}, {self.r2}, {self.fused} | window={self.window}s")

    def cb(self, key, msg: Detection3DArray):
        t = Time.from_msg(msg.header.stamp)           # SIM TIME stamp from message
        self.buf[key].append((t, len(msg.detections)))

    def _prune_before(self, lo: Time):
        for k in self.buf:
            self.buf[k] = [(t, c) for (t, c) in self.buf[k] if t >= lo]

    def tick(self):
        # Use the node's clock (respects use_sim_time if set)
        now_ros = self.get_clock().now()
        lo = now_ros - Duration(seconds=self.window)
        self._prune_before(lo)

        r1c = sum(c for (t, c) in self.buf['r1'])
        r2c = sum(c for (t, c) in self.buf['r2'])
        fsc = sum(c for (t, c) in self.buf['fused'])
        best = max(r1c, r2c)
        gain = (fsc - best)
        self.wr.writerow([f"{t_to_sec(now_ros):.3f}", r1c, r2c, best, fsc, gain])

    def destroy_node(self):
        try:
            self.csv.flush()
            self.csv.close()
        except Exception:
            pass
        return super().destroy_node()

def main():
    rclpy.init()
    n = AccProxy()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
