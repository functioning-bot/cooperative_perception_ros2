#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
import message_filters

class FuseConcat(Node):
    def __init__(self):
        super().__init__('fuse_concat')
        self.declare_parameter('in_topic_r1', '/robot_1/detections_3d_map')
        self.declare_parameter('in_topic_r2', '/robot_2/detections_3d_map')
        self.declare_parameter('out_topic', '/global_detections')

        r1 = self.get_parameter('in_topic_r1').value
        r2 = self.get_parameter('in_topic_r2').value
        out = self.get_parameter('out_topic').value

        self.pub = self.create_publisher(Detection3DArray, out, 10)

        # Sync with modest slop to tolerate minor skew
        self.sub_r1 = message_filters.Subscriber(self, Detection3DArray, r1, qos_profile=10)
        self.sub_r2 = message_filters.Subscriber(self, Detection3DArray, r2, qos_profile=10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_r1, self.sub_r2], queue_size=50, slop=0.25)
        self.ts.registerCallback(self.cb_sync)

        self.get_logger().info(f"concat: {r1} + {r2} -> {out}")

    def cb_sync(self, a: Detection3DArray, b: Detection3DArray):
        out = Detection3DArray()
        # Both inputs are already in frame 'map'
        out.header = a.header
        out.detections = list(a.detections) + list(b.detections)
        self.pub.publish(out)

def main():
    rclpy.init()
    n = FuseConcat()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
