#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
import numpy as np

# Ultralytics YOLO
from ultralytics import YOLO

class Yolo2DDetector(Node):
    def __init__(self):
        super().__init__('yolo2d_detector')

        # ---- Parameters ----
        self.declare_parameter('image_topic', '/robot_1/camera/image_raw')
        self.declare_parameter('detections_topic', '/robot_1/detections_2d')
        self.declare_parameter('annotated_topic', '/robot_1/camera/annotated')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_thresh', 0.25)
        self.declare_parameter('allowed_classes', ['person', 'car'])  # string[] param
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('imgsz', 640)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.dets_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self.ann_topic = self.get_parameter('annotated_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf_thresh').value)
        self.allowed = [str(s) for s in self.get_parameter('allowed_classes').value]
        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        # ---- Model ----
        self.model = YOLO(self.model_path)
        names = self.model.names  # id->label
        self.id2name = {int(k): v for k, v in names.items()} if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
        self.allowed_ids = {i for i, n in self.id2name.items() if n in self.allowed}

        # ---- ROS I/O ----
        self.bridge = CvBridge()
        img_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, img_qos)
        self.pub_dets = self.create_publisher(Detection2DArray, self.dets_topic, 10)
        self.pub_ann = self.create_publisher(Image, self.ann_topic, 10) if self.publish_annotated else None

        self.get_logger().info(f"YOLO model: {self.model_path} | allowed={sorted(self.allowed)} | conf={self.conf}")
        self.get_logger().info(f"Sub: {self.image_topic} | Pub: {self.dets_topic}" + (f", {self.ann_topic}" if self.publish_annotated else ""))

    def image_cb(self, msg: Image):
        # Convert to OpenCV image
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge conversion failed: {e}")
            return

        # Inference
        try:
            results = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False, device=0 if self._has_cuda() else 'cpu')
        except Exception as e:
            self.get_logger().error(f"YOLO inference error: {e}")
            return
        if not results:
            return

        det_array = Detection2DArray()
        det_array.header = msg.header  # copy stamp & frame_id (camera_link)

        annotated = img.copy() if self.publish_annotated else None

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes

            # loop detections
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
                if self.allowed_ids and cls_id not in self.allowed_ids:
                    continue
                score = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0

                # xywh in pixels (center x,y,width,height), guaranteed in image shape
                if hasattr(boxes, "xywh") and boxes.xywh is not None:
                    cx, cy, bw, bh = [float(v) for v in boxes.xywh[i].tolist()]
                else:
                    # fallback: compute from xyxy
                    x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    bw, bh = (x2 - x1), (y2 - y1)

                det = Detection2D()
                det.header = msg.header
                det.bbox = BoundingBox2D()

                # robust center assignment across message variants
                c = det.bbox.center
                try:
                    if hasattr(c, "x") and hasattr(c, "y"):
                        c.x = cx
                        c.y = cy
                        if hasattr(c, "theta"):
                            c.theta = 0.0
                    elif hasattr(c, "position") and hasattr(c.position, "x"):
                        c.position.x = cx
                        c.position.y = cy
                        if hasattr(c, "theta"):
                            c.theta = 0.0
                    elif hasattr(c, "pose") and hasattr(c.pose, "position"):
                        c.pose.position.x = cx
                        c.pose.position.y = cy
                        # theta field may not exist here
                except Exception as e:
                    self.get_logger().warn(f"Could not set bbox center: {e}")

                det.bbox.size_x = bw
                det.bbox.size_y = bh

                hyp = ObjectHypothesisWithPose()
                # class_id as label string (e.g., "person")
                hyp.hypothesis.class_id = self.id2name.get(cls_id, str(cls_id))
                hyp.hypothesis.score = score
                det.results.append(hyp)

                det_array.detections.append(det)

                # annotate image
                if annotated is not None:
                    x1 = int(cx - bw/2); y1 = int(cy - bh/2)
                    x2 = int(cx + bw/2); y2 = int(cy + bh/2)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(annotated.shape[1]-1, x2); y2 = min(annotated.shape[0]-1, y2)
                    label = f"{self.id2name.get(cls_id, cls_id)} {score:.2f}"
                    try:
                        import cv2
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                    except Exception:
                        pass

        # publish
        self.pub_dets.publish(det_array)
        if self.publish_annotated and annotated is not None:
            try:
                ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                ann_msg.header = msg.header
                self.pub_ann.publish(ann_msg)
            except Exception as e:
                self.get_logger().warn(f"Annotated publish failed: {e}")

    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

def main():
    rclpy.init()
    node = Yolo2DDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
