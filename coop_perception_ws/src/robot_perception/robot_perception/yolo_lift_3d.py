#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import (Detection2DArray, Detection2D, Detection3DArray,
                             Detection3D, ObjectHypothesisWithPose, BoundingBox3D)
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from tf_transformations import quaternion_from_euler

def deproject_pixel(u, v, depth, fx, fy, cx, cy):
    """Return XYZ in camera frame (meters) from pixel (u,v) and depth Z."""
    Z = float(depth)
    if not np.isfinite(Z) or Z <= 0.0:
        return None
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return X, Y, Z

class Lift2Dto3D(Node):
    def __init__(self):
        super().__init__('lift_2d_to_3d')

        # -------- parameters --------
        self.declare_parameter('det_topic', '/robot_1/detections_2d')
        self.declare_parameter('depth_topic', '/robot_1/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/robot_1/camera/camera_info')
        self.declare_parameter('out_topic', '/robot_1/detections_3d')
        self.declare_parameter('marker_topic', '/robot_1/detections_markers')
        self.declare_parameter('global_frame', 'map')             # target frame for output
        self.declare_parameter('sample', 'center')                # 'center' or 'grid'
        self.declare_parameter('grid_step', 3)                    # when sample='grid', stride in pixels
        self.declare_parameter('min_depth', 0.2)                  # meters
        self.declare_parameter('max_depth', 50.0)                 # meters
        self.declare_parameter('box_depth_percentile', 50.0)      # percentile of valid samples in bbox

        self.det_topic   = self.get_parameter('det_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.cinfo_topic = self.get_parameter('camera_info_topic').value
        self.out_topic   = self.get_parameter('out_topic').value
        self.marker_topic= self.get_parameter('marker_topic').value
        self.global_frame= self.get_parameter('global_frame').value
        self.sample_mode = self.get_parameter('sample').value
        self.grid_step   = int(self.get_parameter('grid_step').value)
        self.min_depth   = float(self.get_parameter('min_depth').value)
        self.max_depth   = float(self.get_parameter('max_depth').value)
        self.box_pct     = float(self.get_parameter('box_depth_percentile').value)

        # -------- ROS I/O --------
        qos_sensor = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST)
        self.bridge = CvBridge()

        self.sub_det   = message_filters.Subscriber(self, Detection2DArray, self.det_topic, qos_profile=10)
        self.sub_depth = message_filters.Subscriber(self, Image,            self.depth_topic, qos_profile=qos_sensor)
        self.sub_cinfo = message_filters.Subscriber(self, CameraInfo,       self.cinfo_topic, qos_profile=10)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_det, self.sub_depth, self.sub_cinfo], queue_size=10, slop=0.08
        )
        self.ts.registerCallback(self.cb_sync)

        self.pub_3d  = self.create_publisher(Detection3DArray, self.out_topic, 10)
        self.pub_mk  = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"Lift node: det={self.det_topic} depth={self.depth_topic} cinfo={self.cinfo_topic}")
        self.get_logger().info(f"Outputs: {self.out_topic}, {self.marker_topic} | target frame={self.global_frame}")

    def cb_sync(self, det_msg: Detection2DArray, depth_msg: Image, cinfo: CameraInfo):
        # intrinsics
        K = cinfo.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]

        # depth image
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f"depth conversion failed: {e}")
            return

        h, w = depth.shape[:2]
        out = Detection3DArray()
        out.header = det_msg.header  # copy stamp + frame from camera stream

        markers = MarkerArray()
        m_id = 0

        for det in det_msg.detections:
            # bbox center + size in pixels
            # compatible center extraction across variants
            cxp = None; cyp = None
            c = det.bbox.center
            if hasattr(c, 'x') and hasattr(c, 'y'):
                cxp, cyp = float(c.x), float(c.y)
            elif hasattr(c, 'position'):
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

            Z_samples = []

            if self.sample_mode == 'center':
                u, v = int(round(cxp)), int(round(cyp))
                if 0 <= v < h and 0 <= u < w:
                    z = float(depth[v, u])
                    if self.min_depth < z < self.max_depth and math.isfinite(z):
                        Z_samples.append(z)

            else:  # grid sampling inside bbox
                step = max(1, self.grid_step)
                for v in range(y1, y2+1, step):
                    for u in range(x1, x2+1, step):
                        z = float(depth[v, u])
                        if self.min_depth < z < self.max_depth and math.isfinite(z):
                            Z_samples.append(z)

            if not Z_samples:
                # skip if no good depth
                continue

            # pick a robust Z
            Z = float(np.percentile(np.array(Z_samples, dtype=np.float32), self.box_pct))
            xyz_cam = deproject_pixel(cxp, cyp, Z, fx, fy, cx, cy)
            if xyz_cam is None:
                continue
            Xc, Yc, Zc = xyz_cam

            # pose in camera frame
            pose_cam = PoseStamped()
            pose_cam.header = det_msg.header  # frame_id is camera_link
            pose_cam.pose.position.x = Xc
            pose_cam.pose.position.y = Yc
            pose_cam.pose.position.z = Zc
            # orientation: align with camera (no rotation)
            q = quaternion_from_euler(0.0, 0.0, 0.0)
            pose_cam.pose.orientation.x = q[0]
            pose_cam.pose.orientation.y = q[1]
            pose_cam.pose.orientation.z = q[2]
            pose_cam.pose.orientation.w = q[3]

            # transform to global frame if available
            pose_out = Pose()
            target_frame = self.global_frame
            try:
                tf_pose = self.tf_buffer.transform(pose_cam, target_frame, timeout=rclpy.duration.Duration(seconds=0.05))
                pose_out = tf_pose.pose
                out_frame = target_frame
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                # fallback: keep camera frame
                self.get_logger().warn_throttle(2.0, f"TF to {target_frame} failed ({e}); keeping camera frame")
                pose_out = pose_cam.pose
                out_frame = det_msg.header.frame_id

            # approximate 3D box size (project pixel size to meters at depth Z)
            sx = (bw / fx) * Z
            sy = (bh / fy) * Z
            sz = max(0.3, min(2.0, 0.5))  # simple constant thickness

            det3 = Detection3D()
            det3.header = det_msg.header
            det3.header.frame_id = out_frame
            det3.results = []
            # copy best hypothesis label/score, if any
            label = 'object'; score = 1.0
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
            det3.bbox.size.x = max(0.01, sx)
            det3.bbox.size.y = max(0.01, sy)
            det3.bbox.size.z = sz

            out.detections.append(det3)

            # markers (cube + text)
            cube = Marker()
            cube.header.frame_id = out_frame
            cube.header.stamp = det_msg.header.stamp
            cube.ns = "lift3d"
            cube.id = m_id; m_id += 1
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose = pose_out
            cube.scale.x = det3.bbox.size.x
            cube.scale.y = det3.bbox.size.y
            cube.scale.z = det3.bbox.size.z
            cube.color.r = 0.0; cube.color.g = 1.0; cube.color.b = 0.0; cube.color.a = 0.6
            cube.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            markers.markers.append(cube)

            text = Marker()
            text.header.frame_id = out_frame
            text.header.stamp = det_msg.header.stamp
            text.ns = "lift3d_text"
            text.id = m_id; m_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = pose_out
            text.pose.position.z += det3.bbox.size.z * 0.6
            text.scale.z = 0.3
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 1.0; text.color.a = 0.9
            text.text = f"{label}"
            text.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            markers.markers.append(text)

        # publish
        self.pub_3d.publish(out)
        if markers.markers:
            self.pub_mk.publish(markers)

def main():
    rclpy.init()
    node = Lift2Dto3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
