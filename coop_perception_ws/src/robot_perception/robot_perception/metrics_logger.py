#!/usr/bin/env python3
import csv, os, time
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import Header

def stamp_to_time(stamp):  # ROS -> float seconds
    return stamp.sec + stamp.nanosec * 1e-9

class MetricsLogger(Node):
    def __init__(self):
        super().__init__('metrics_logger')
        self.declare_parameter('fused_topic', '/global_detections')
        self.declare_parameter('csv_path', 'metrics_latency.csv')
        self.declare_parameter('warmup_msgs', 20)

        self.fused_topic = self.get_parameter('fused_topic').value
        self.csv_path = self.get_parameter('csv_path').value
        self.warmup = int(self.get_parameter('warmup_msgs').value)

        self.sub = self.create_subscription(Detection3DArray, self.fused_topic, self.cb, 10)
        self.t0_wall = time.time()
        self.count = 0

        # CSV header
        new_file = not os.path.exists(self.csv_path)
        self.csv = open(self.csv_path, 'a', newline='')
        self.wr = csv.writer(self.csv)
        if new_file:
            self.wr.writerow([
                "wall_time_s",
                "ros_now_s",
                "fused_stamp_s",
                "latency_s",
                "num_detections"
            ])
        self.get_logger().info(f"Logging latency on {self.fused_topic} -> {self.csv_path}")

    def cb(self, msg: Detection3DArray):
        # End-to-end latency: now (ROS time) minus original image stamp carried through pipeline.
        # Our pipeline preserves the original image stamp in the header of fused outputs.
        self.count += 1
        if self.count <= self.warmup:
            return

        now_ros = self.get_clock().now()
        src_stamp = Time.from_msg(msg.header.stamp)
        lat = (now_ros - src_stamp).nanoseconds * 1e-9

        self.wr.writerow([
            f"{time.time():.6f}",
            f"{now_ros.seconds_nanoseconds()[0] + now_ros.seconds_nanoseconds()[1]*1e-9:.6f}",
            f"{src_stamp.seconds_nanoseconds()[0] + src_stamp.seconds_nanoseconds()[1]*1e-9:.6f}",
            f"{lat:.6f}",
            len(msg.detections)
        ])
        # Flush occasionally to avoid data loss
        if self.count % 50 == 0:
            self.csv.flush()

    def destroy_node(self):
        try:
            self.csv.flush()
            self.csv.close()
        except Exception:
            pass
        return super().destroy_node()

def main():
    rclpy.init()
    node = MetricsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
