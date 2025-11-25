#!/usr/bin/env python3
import math
from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import Pose, PoseStamped
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, BoundingBox3D
from visualization_msgs.msg import Marker, MarkerArray

def make_qos_reliable(depth=10):
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth
    )

class AwarenessBridge(Node):
    """
    Subscribes to global fused detections (map frame) and republishes per-robot
    local awareness topics in each robot's base_link frame with optional filters.
    """

    def __init__(self):
        super().__init__('awareness_bridge')

        # Parameters
        self.declare_parameter('global_topic', '/global_detections')
        self.declare_parameter('robots', ['robot_1', 'robot_2'])           # namespace stems
        self.declare_parameter('base_links', ['robot_1', 'robot_2'])
        self.declare_parameter('class_whitelist', ['person', 'car'])        # [] = allow all
        self.declare_parameter('max_distance_m', 12.0)                      # 0 = no limit
        self.declare_parameter('fov_deg', 180.0)                            # simple symmetric FOV; 360 to disable
        self.declare_parameter('publish_rate_hz', 20.0)                     # throttle pub (min period)
        self.declare_parameter('marker_alpha', 0.65)

        self.global_topic  = self.get_parameter('global_topic').value
        self.robots        = list(self.get_parameter('robots').value)
        self.base_links    = list(self.get_parameter('base_links').value)
        self.cls_whitelist = set(self.get_parameter('class_whitelist').value)
        self.max_dist      = float(self.get_parameter('max_distance_m').value)
        self.fov_deg       = float(self.get_parameter('fov_deg').value)
        self.pub_rate_hz   = float(self.get_parameter('publish_rate_hz').value)
        self.marker_alpha  = float(self.get_parameter('marker_alpha').value)

        assert len(self.robots) == len(self.base_links), "robots and base_links must be same length"

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # IO
        self.sub_global = self.create_subscription(
            Detection3DArray, self.global_topic, self.on_global, make_qos_reliable(10)
        )

        self.pub_local: Dict[str, any] = {}
        self.pub_markers: Dict[str, any] = {}
        for i, robot in enumerate(self.robots):
            ns = f'/{robot}'
            self.pub_local[robot] = self.create_publisher(
                Detection3DArray, f'{ns}/awareness', 10
            )
            self.pub_markers[robot] = self.create_publisher(
                MarkerArray, f'{ns}/awareness_markers', 10
            )

        self.last_pub_sec: Dict[str, float] = {r: 0.0 for r in self.robots}
        self.min_period = 1.0 / max(self.pub_rate_hz, 1e-3)

        self.get_logger().info(
            f'AwarenessBridge: sub={self.global_topic} -> pubs {[f"/{r}/awareness" for r in self.robots]}'
        )

    # ---------- helpers ----------
    @staticmethod
    def pose_to_xy(p: Pose):
        return float(p.position.x), float(p.position.y)

    def in_fov(self, x: float, y: float, fov_deg: float) -> bool:
        """ Simple forward-FOV test: +X forward; if fov>=360 allow all. """
        if fov_deg >= 359.0:
            return True
        # angle from forward (+X) in robot frame
        ang = math.degrees(math.atan2(y, x))  # [-180,180]
        return abs(ang) <= (fov_deg * 0.5) and x > 0.0  # only in front half-space

    def within_dist(self, x: float, y: float, max_d: float) -> bool:
        if max_d <= 0.0:
            return True
        return math.hypot(x, y) <= max_d

    # ---------- main callback ----------
    def on_global(self, msg: Detection3DArray):
        # For each robot, build a local array
        for robot, base_link in zip(self.robots, self.base_links):
            local = Detection3DArray()
            local.header = msg.header
            local.header.frame_id = base_link  # output frame is robot base_link

            markers = MarkerArray()
            m_id = 0

            for det in msg.detections:
                if det.results:
                    label = det.results[0].hypothesis.class_id
                else:
                    label = 'object'

                if self.cls_whitelist and (label not in self.cls_whitelist):
                    continue

                # Transform detection pose (center) map->base_link at the message stamp
                src_pose = PoseStamped()
                src_pose.header = det.header  # contains map frame + time
                src_pose.pose   = det.bbox.center  # center is a Pose

                try:
                    tf = self.tf_buffer.lookup_transform(
                        base_link, src_pose.header.frame_id,
                        rclpy.time.Time.from_msg(src_pose.header.stamp),
                        timeout=Duration(seconds=0.05)
                    )
                except (LookupException, ConnectivityException, ExtrapolationException):
                    # skip if TF not available
                    continue

                # Manual transform of pose using tf (no geometry msgs convenience here)
                # Extract translation/rotation from tf
                t = tf.transform.translation
                q = tf.transform.rotation
                # Pose composition: base_T_map * map_P = base_P
                # Implement with minimal dependencies (small helper):
                bx, by, bz = self._transform_point(src_pose.pose.position.x,
                                                   src_pose.pose.position.y,
                                                   src_pose.pose.position.z,
                                                   t, q)

                # Filters in base_link
                if not self.within_dist(bx, by, self.max_dist):
                    continue
                if not self.in_fov(bx, by, self.fov_deg):
                    continue

                # Build local Detection3D
                det_local = Detection3D()
                det_local.header = local.header

                # copy hypothesis, keep score
                if det.results:
                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis = det.results[0].hypothesis
                    hyp.pose.pose.position.x = bx
                    hyp.pose.pose.position.y = by
                    hyp.pose.pose.position.z = bz
                    hyp.pose.pose.orientation.w = 1.0
                    det_local.results.append(hyp)

                # bbox: reuse metric size; center in base_link
                det_local.bbox = BoundingBox3D()
                det_local.bbox.center.position.x = bx
                det_local.bbox.center.position.y = by
                det_local.bbox.center.position.z = bz
                det_local.bbox.center.orientation.w = 1.0
                det_local.bbox.size = det.bbox.size  # copy sizes (already metric)

                local.detections.append(det_local)

                # Marker
                cube = Marker()
                cube.header = local.header
                cube.ns = 'awareness'
                cube.id = m_id; m_id += 1
                cube.type = Marker.CUBE
                cube.action = Marker.ADD
                cube.pose = det_local.bbox.center
                cube.scale = det_local.bbox.size
                cube.color.r = 0.2; cube.color.g = 0.7; cube.color.b = 1.0; cube.color.a = self.marker_alpha
                cube.lifetime = Duration(seconds=self.min_period*2).to_msg()

                text = Marker()
                text.header = local.header
                text.ns = 'awareness_text'
                text.id = m_id; m_id += 1
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose = det_local.bbox.center
                text.pose.position.z += max(0.4, det_local.bbox.size.z * 0.6)
                text.scale.z = 0.35
                text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 0.95
                txt = det.results[0].hypothesis.class_id if det.results else 'obj'
                text.text = f"{txt}"
                text.lifetime = Duration(seconds=self.min_period*2).to_msg()

                markers.markers.append(cube)
                markers.markers.append(text)

            # throttle publish per robot
            now = self.get_clock().now().seconds_nanoseconds()[0] + \
                  self.get_clock().now().seconds_nanoseconds()[1]*1e-9
            if (now - self.last_pub_sec[robot]) >= self.min_period:
                self.pub_local[robot].publish(local)
                if markers.markers:
                    self.pub_markers[robot].publish(markers)
                self.last_pub_sec[robot] = now

    # ---- small math util: rotate+translate (q*x + t) ----
    @staticmethod
    def _transform_point(x, y, z, t, q):
        # quaternion to rotation matrix
        import numpy as np
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
        ], dtype=float)
        v = np.array([x, y, z], dtype=float)
        tv = R.dot(v) + np.array([t.x, t.y, t.z], dtype=float)
        return float(tv[0]), float(tv[1]), float(tv[2])

def main():
    rclpy.init()
    node = AwarenessBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
