#!/usr/bin/env python3
# File: robot_perception/yolo_lift_3d_sync.py
# Sync det+depth with ApproximateTimeSynchronizer; cache latest CameraInfo.

import math
from typing import Optional

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


# ---------- Frame-conversion helpers ----------
def _rot_optical_to_mount_xyz(): 
    """ Rotation matrix R to map optical camera coords (Z forward, X right, Y down) into a typical mount frame like camera_link (Z up, X forward, Y left). This is the inverse of REP-103 optical transform: R = Rz(+270°) * Rx(+90°). """ 
    # +270° about Z (i.e., +90° + 180°), then +90° about X 
    cz, sz = 0.0, -1.0 # cos(270)=0, sin(270)=-1 
    cx, sx = 0.0, 1.0 
    Rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]] 
    Rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]] # R = Rz * Rx
    #return [[sum(Rz[i][k]*Rx[k][j] for k in range(3)) for j in range(3)] for i in range(3)] 
    return [ [ 0.0, 0.0, -1.0], # X_m = -Z_o 
    [1.0, 0.0, 0.0], # Y_m = X_o 
    [ 0.0, -1.0, 0.0], # Z_m = -Y_o 
    ]

def _apply_R(p, R):
    x, y, z = p
    return (
        R[0][0] * x + R[0][1] * y + R[0][2] * z,
        R[1][0] * x + R[1][1] * y + R[1][2] * z,
        R[2][0] * x + R[2][1] * y + R[2][2] * z,
    )


def deproject_pixel(u: float, v: float, depth: float, fx: float, fy: float, cx: float, cy: float):
    Z = float(depth)
    if not math.isfinite(Z) or Z <= 0.0:
        return None
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    # NOTE: (X, Y, Z) here are in *optical* convention (X right, Y down, Z forward)
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

        # NEW: headers use a mount frame (e.g., camera_link), not optical → rotate points
        self.declare_parameter('input_frame_is_mount', True)

        self.det_topic = self.get_parameter('det_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.cinfo_topic = self.get_parameter('camera_info_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.global_frame = self.get_parameter('global_frame').value
        self.sample_mode = self.get_parameter('sample').value
        self.grid_step = int(self.get_parameter('grid_step').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.box_pct = float(self.get_parameter('box_depth_percentile').value)
        self.input_frame_is_mount = bool(self.get_parameter('input_frame_is_mount').value)

        # Precompute rotation (optical -> mount)
        self._R_opt2mount = _rot_optical_to_mount_xyz()

        # ---- QoS ----
        self.qos_sensor = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.qos_rel = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # ---- IO ----
        self.bridge = CvBridge()
        self.pub_3d = self.create_publisher(Detection3DArray, self.out_topic, 10)
        self.pub_mk = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # Cache latest CameraInfo (intrinsics are static; timestamp may be zero in some sims)
        self.last_cinfo: Optional[CameraInfo] = None
        self.create_subscription(CameraInfo, self.cinfo_topic, self._cinfo_cb, self.qos_rel)

        # message_filters only for det + depth (their stamps must align)
        self.sub_det = message_filters.Subscriber(self, Detection2DArray, self.det_topic, qos_profile=self.qos_rel)
        self.sub_depth = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=self.qos_sensor)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_det, self.sub_depth], queue_size=20, slop=0.20
        )
        self.ts.registerCallback(self.cb_sync)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f"Lift(sync det+depth; cached cinfo): det={self.det_topic} depth={self.depth_topic} cinfo={self.cinfo_topic}"
        )
        self.get_logger().info(
            f"Outputs: {self.out_topic}, {self.marker_topic} | target frame={self.global_frame} | input_frame_is_mount={self.input_frame_is_mount}"
        )

    # Cache CameraInfo (intrinsics are static; timestamp may be zero in some sims)
    def _cinfo_cb(self, msg: CameraInfo):
        self.last_cinfo = msg

    # Synchronized on det+depth only
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

        for det in det_msg.detections:
            # bbox center (support multiple vision_msgs field variants)
            c = det.bbox.center
            if hasattr(c, 'x') and hasattr(c, 'y'):
                cxp, cyp = float(c.x), float(c.y)
            elif hasattr(c, 'position') and hasattr(c.position, 'x'):
                cxp, cyp = float(c.position.x), float(c.position.y)
            elif hasattr(getattr(c, 'pose', None), 'position'):
                cxp, cyp = float(c.pose.position.x), float(c.pose.position.y)
            else:
                continue

            bw = float(det.bbox.size_x)
            bh = float(det.bbox.size_y)
            x1 = max(0, int(round(cxp - bw / 2)))
            y1 = max(0, int(round(cyp - bh / 2)))
            x2 = min(w - 1, int(round(cxp + bw / 2)))
            y2 = min(h - 1, int(round(cyp + bh / 2)))

            # Collect depth samples
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
            Xc, Yc, Zc = xyz_cam_optical  # optical convention

            # If headers use mount frame (camera_link), rotate optical->mount here.
            if self.input_frame_is_mount:
                Xc, Yc, Zc = _apply_R((Xc, Yc, Zc), self._R_opt2mount)

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
                    self.global_frame,
                    pose_cam.header.frame_id,
                    rclpy.time.Time.from_msg(pose_cam.header.stamp),
                    timeout=Duration(seconds=0.05)
                )
                # NOTE: do_transform_pose takes a Pose, not PoseStamped
                pose_out = do_transform_pose(pose_cam.pose, tf)
                out_frame = self.global_frame
            except (LookupException, ConnectivityException, ExtrapolationException):
                self.get_logger().warning(f"TF to {self.global_frame} failed; keeping camera frame")
                pose_out = pose_cam.pose

            # Metric size estimate from pixel bbox at depth Z (uses Z only)
            sx = max(0.01, (bw / fx) * Z)
            sy = max(0.01, (bh / fy) * Z)
            sz = 0.5  # simple constant thickness

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
            cube.id = m_id
            m_id += 1
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose = pose_out
            cube.scale.x = sx
            cube.scale.y = sy
            cube.scale.z = sz
            cube.color.r = 0.0
            cube.color.g = 1.0
            cube.color.b = 0.0
            cube.color.a = 0.6
            cube.lifetime = Duration(seconds=0.2).to_msg()

            text = Marker()
            text.header.frame_id = out_frame
            text.header.stamp = det_msg.header.stamp
            text.ns = "lift3d_text"
            text.id = m_id
            m_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = pose_out
            text.pose.position.z += sz * 0.6
            text.scale.z = 0.3
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.9
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
