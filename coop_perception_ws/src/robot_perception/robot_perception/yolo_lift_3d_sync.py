#!/usr/bin/env python3
# File: robot_perception/yolo_lift_3d_sync.py
# Sync det+depth with ApproximateTimeSynchronizer; cache latest CameraInfo.
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import (
    Detection2DArray, Detection3DArray, Detection3D,
    ObjectHypothesisWithPose, BoundingBox3D
)
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

import message_filters
from tf2_ros import (
    Buffer, TransformListener,
    LookupException, ConnectivityException, ExtrapolationException
)
try:
    from tf2_geometry_msgs import do_transform_pose  # expects geometry_msgs/Pose
except Exception:
    from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


# ---------- Rotation utilities ----------
def _euler_rpy_deg_to_R(roll_x_deg: float, pitch_y_deg: float, yaw_z_deg: float):
    rx = math.radians(roll_x_deg)
    ry = math.radians(pitch_y_deg)
    rz = math.radians(yaw_z_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # R = Rz * Ry * Rx
    Rz = np.array([[cz, -sz, 0.0],
                   [sz,  cz, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    Ry = np.array([[ cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cx, -sx],
                   [0.0,  sx,  cx]], dtype=float)
    return (Rz @ Ry @ Rx).astype(float)

def _apply_R_tuple(p, R: np.ndarray):
    v = np.array([p[0], p[1], p[2]], dtype=float)
    tv = R.dot(v)
    return float(tv[0]), float(tv[1]), float(tv[2])


def deproject_pixel(u: float, v: float, depth: float, fx: float, fy: float, cx: float, cy: float):
    Z = float(depth)
    if not math.isfinite(Z) or Z <= 0.0:
        return None
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    # (X,Y,Z) are in optical convention (X right, Y down, Z forward)
    return X, Y, Z


class Lift2Dto3DSync(Node):
    def __init__(self):
        super().__init__('lift_2d_to_3d')

        # ---- Parameters ----
        self.declare_parameter('det_topic', '/robot_1/detections_2d')
        self.declare_parameter('depth_topic', '/robot_1/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/robot_1/camera/camera_info')
        self.declare_parameter('out_topic', '/robot_1/detections_3d')
        self.declare_parameter('marker_topic', '/robot_1/detections_markers')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('sample', 'grid')          # 'center' | 'grid'
        self.declare_parameter('grid_step', 3)            # px
        self.declare_parameter('min_depth', 0.4)          # m
        self.declare_parameter('max_depth', 30.0)         # m
        self.declare_parameter('box_depth_percentile', 50.0)

        # Orientation handling (replaces hardcoded _rot_optical_to_mount_xyz)
        # 'auto' -> if header frame contains 'optical', treat as optical; else as mount
        # 'optical' -> no rotation (point already in header frame)
        # 'mount'   -> rotate optical->mount using opt2mount_rpy_deg
        self.declare_parameter('orientation_mode', 'auto')
        self.declare_parameter('opt2mount_rpy_deg', [90.0, 0.0, 270.0])  # Rx=+90°, Rz=+270° by default

        self.det_topic    = self.get_parameter('det_topic').value
        self.depth_topic  = self.get_parameter('depth_topic').value
        self.cinfo_topic  = self.get_parameter('camera_info_topic').value
        self.out_topic    = self.get_parameter('out_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.sample_mode  = self.get_parameter('sample').value
        self.grid_step    = int(self.get_parameter('grid_step').value)
        self.min_depth    = float(self.get_parameter('min_depth').value)
        self.max_depth    = float(self.get_parameter('max_depth').value)
        self.box_pct      = float(self.get_parameter('box_depth_percentile').value)

        self.orientation_mode = str(self.get_parameter('orientation_mode').value)
        rpy = list(self.get_parameter('opt2mount_rpy_deg').value)
        if len(rpy) != 3:
            rpy = [90.0, 0.0, 270.0]
        self._R_opt2mount = _euler_rpy_deg_to_R(rpy[0], rpy[1], rpy[2])

        # ---- QoS ----
        self.qos_sensor = QoSProfile(
            depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.qos_rel = QoSProfile(
            depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # ---- IO ----
        self.bridge = CvBridge()
        self.pub_3d = self.create_publisher(Detection3DArray, self.out_topic, 10)
        self.pub_mk = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # cache latest CameraInfo (do NOT include it in message_filters)
        self.last_cinfo: CameraInfo | None = None
        self.create_subscription(CameraInfo, self.cinfo_topic, self._cinfo_cb, self.qos_rel)

        # message_filters only for det + depth (their stamps must align)
        self.sub_det   = message_filters.Subscriber(self, Detection2DArray, self.det_topic, qos_profile=self.qos_rel)
        self.sub_depth = message_filters.Subscriber(self, Image,            self.depth_topic, qos_profile=self.qos_sensor)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_det, self.sub_depth], queue_size=20, slop=0.20
        )
        self.ts.registerCallback(self.cb_sync)

        # TF
        self.tf_buffer = Buffer()
               # noqa: E121
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f"Lift(sync det+depth; cached cinfo): det={self.det_topic} depth={self.depth_topic} cinfo={self.cinfo_topic}"
        )
        self.get_logger().info(
            f"Outputs: {self.out_topic}, {self.marker_topic} | target frame={self.global_frame} | "
            f"orientation_mode={self.orientation_mode} | opt2mount_rpy_deg={rpy}"
        )

    # cache CameraInfo (intrinsics are static; timestamp may be zero in some sims)
    def _cinfo_cb(self, msg: CameraInfo):
        self.last_cinfo = msg

    # synchronized on det+depth only
    def cb_sync(self, det_msg: Detection2DArray, depth_msg: Image):
        if self.last_cinfo is None:
            return
        cinfo = self.last_cinfo

        # Intrinsics
        K = list(cinfo.k)
        fx, fy, cx, cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])

        # Depth image
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warning(f"depth conversion failed: {e}")
            return

        h, w = depth.shape[:2]
        out = Detection3DArray()
        out.header = det_msg.header  # start with camera frame/time
        markers = MarkerArray()
        m_id = 0

        # Determine orientation handling
        header_is_optical = ('optical' in (det_msg.header.frame_id or '').lower())
        use_mode = self.orientation_mode.lower()
        if use_mode == 'auto':
            treat_as_mount = not header_is_optical
        elif use_mode == 'mount':
            treat_as_mount = True
        else:  # 'optical'
            treat_as_mount = False

        for det in det_msg.detections:
            # bbox center (support multiple vision_msgs variants)
            c = det.bbox.center
            if   hasattr(c, 'x') and hasattr(c, 'y'):
                cxp, cyp = float(c.x), float(c.y)
            elif hasattr(c, 'position') and hasattr(c.position, 'x'):
                cxp, cyp = float(c.position.x), float(c.position.y)
            elif hasattr(getattr(c, 'pose', None), 'position'):
                cxp, cyp = float(c.pose.position.x), float(c.pose.position.y)
            else:
                continue

            bw = float(det.bbox.size_x)
            bh = float(det.bbox.size_y)
            x1 = max(0, int(round(cxp - bw/2)))
            y1 = max(0, int(round(cyp - bh/2)))
            x2 = min(w-1, int(round(cxp + bw/2)))
            y2 = min(h-1, int(round(cyp + bh/2)))

            # collect depth samples
            Z_samples = []
            if self.sample_mode == 'center':
                u, v = int(round(cxp)), int(round(cyp))
                if 0 <= v < h and 0 <= u < w:
                    z = float(depth[v, u])
                    if self.min_depth < z < self.max_depth and math.isfinite(z):
                        Z_samples.append(z)
            else:
                step = max(1, self.grid_step)
                for v in range(y1, y2 + 1, step):
                    for u in range(x1, x2 + 1, step):
                        z = float(depth[v, u])
                        if self.min_depth < z < self.max_depth and math.isfinite(z):
                            Z_samples.append(z)

            if not Z_samples:
                continue

            Z = float(np.percentile(np.array(Z_samples, dtype=np.float32), self.box_pct))
            xyz_cam_optical = deproject_pixel(cxp, cyp, Z, fx, fy, cx, cy)
            if xyz_cam_optical is None:
                continue
            Xo, Yo, Zo = xyz_cam_optical  # optical convention

            # Rotate to header mount frame if needed
            if treat_as_mount:
                Xc, Yc, Zc = _apply_R_tuple((Xo, Yo, Zo), self._R_opt2mount)
            else:
                Xc, Yc, Zc = Xo, Yo, Zo

            # Pose in the (header) camera frame
            pose_cam = PoseStamped()
            pose_cam.header = det_msg.header
            pose_cam.pose.position.x = Xc
            pose_cam.pose.position.y = Yc
            pose_cam.pose.position.z = Zc
            pose_cam.pose.orientation.x = 0.0
            pose_cam.pose.orientation.y = 0.0
            pose_cam.pose.orientation.z = 0.0
            pose_cam.pose.orientation.w = 1.0

            # Transform to global frame if TF available
            out_frame = det_msg.header.frame_id
            pose_out = Pose()
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.global_frame, pose_cam.header.frame_id,
                    rclpy.time.Time.from_msg(pose_cam.header.stamp),
                    timeout=Duration(seconds=0.05)
                )
                pose_out = do_transform_pose(pose_cam.pose, tf)  # Pose, not PoseStamped
                out_frame = self.global_frame
            except (LookupException, ConnectivityException, ExtrapolationException):
                self.get_logger().warning(f"TF to {self.global_frame} failed; keeping camera frame")
                pose_out = pose_cam.pose

            # Metric size estimate from pixel bbox at depth Z
            sx = max(0.01, (bw / fx) * Z)
            sy = max(0.01, (bh / fy) * Z)
            sz = 0.5

            # Build Detection3D
            det3 = Detection3D()
            det3.header = det_msg.header
            det3.header.frame_id = out_frame

            label, score = 'object', 1.0
            if det.results:
                label = det.results[0].hypothesis.class_id
                score = float(det.results[0].hypothesis.score)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = score
            hyp.pose.pose = pose_out
            det3.results.append(hyp)

            det3.bbox = BoundingBox3D()
            det3.bbox.center = pose_out
            det3.bbox.size.x = sx
            det3.bbox.size.y = sy
            det3.bbox.size.z = sz

            out.detections.append(det3)

            # RViz markers (cube + text)
            cube = Marker()
            cube.header.frame_id = out_frame
            cube.header.stamp = det_msg.header.stamp
            cube.ns = "lift3d"
            cube.id = m_id; m_id += 1
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose = pose_out
            cube.scale.x = sx; cube.scale.y = sy; cube.scale.z = sz
            cube.color.r = 0.0; cube.color.g = 1.0; cube.color.b = 0.0; cube.color.a = 0.6
            cube.lifetime = Duration(seconds=0.2).to_msg()

            text = Marker()
            text.header.frame_id = out_frame
            text.header.stamp = det_msg.header.stamp
            text.ns = "lift3d_text"
            text.id = m_id; m_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = pose_out
            text.pose.position.z += sz * 0.6
            text.scale.z = 0.3
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 0.9
            text.text = f"{label}"
            text.lifetime = Duration(seconds=0.2).to_msg()

            markers.markers.append(cube)
            markers.markers.append(text)

        self.pub_3d.publish(out)
        if markers.markers:
            self.pub_mk.publish(markers)


def main():
    rclpy.init()
    node = Lift2Dto3DSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
