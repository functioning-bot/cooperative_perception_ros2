#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
try:
    from tf2_geometry_msgs import do_transform_pose
except Exception:
    from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


class ToMapRepublisher(Node):
    def __init__(self):
        super().__init__('to_map_republisher')
        self.declare_parameter('in_topic', '/robot_1/detections_3d')
        self.declare_parameter('out_topic', '/robot_1/detections_3d_map')
        self.declare_parameter('target_frame', 'map')

        self.in_topic = self.get_parameter('in_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.target = self.get_parameter('target_frame').value

        self.pub = self.create_publisher(Detection3DArray, self.out_topic, 10)
        self.sub = self.create_subscription(Detection3DArray, self.in_topic, self.cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # simple throttle state (no warn_throttle in rclpy logger)
        self._last_warn_ns = 0
        self._warn_period_ns = int(2.0 * 1e9)

        self.get_logger().info(f"to-map: {self.in_topic} -> {self.out_topic} [target={self.target}]")

    def _maybe_warn_tf(self, msg: str):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_warn_ns >= self._warn_period_ns:
            self.get_logger().warning(msg)
            self._last_warn_ns = now_ns

    def _transform_pose(self, pose, src_frame, stamp):
        """Wrap Pose as PoseStamped, transform, return Pose."""
        ps = PoseStamped()
        ps.header.frame_id = src_frame
        ps.header.stamp = stamp
        ps.pose = pose
        tf = self.tf_buffer.lookup_transform(self.target, src_frame, stamp, timeout=Duration(seconds=0.05))
        ps_out = do_transform_pose(ps, tf)
        return ps_out.pose

    def cb(self, msg: Detection3DArray):
        out = Detection3DArray()
        out.header = msg.header
        out.header.frame_id = self.target  # force target frame

        for det in msg.detections:
            d = Detection3D()
            d.header = out.header
            d.results = det.results
            d.bbox = BoundingBox3D()

            pose = det.bbox.center
            if msg.header.frame_id != self.target:
                try:
                    pose = self._transform_pose(det.bbox.center, msg.header.frame_id, msg.header.stamp)
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    self._maybe_warn_tf(f"TF {msg.header.frame_id}->{self.target} missing ({e}); passing through")

            d.bbox.center = pose
            d.bbox.size = det.bbox.size
            out.detections.append(d)

        self.pub.publish(out)


def main():
    rclpy.init()
    node = ToMapRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
