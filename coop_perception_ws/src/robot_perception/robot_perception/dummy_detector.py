#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

class DummyDetector(Node):
    def __init__(self):
        super().__init__('dummy_detector')
        self.declare_parameter('image_topic', '/robot_1/camera/image_raw')
        self.declare_parameter('detections_topic', '/robot_1/detections_2d')
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value

        img_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.pub = self.create_publisher(Detection2DArray, detections_topic, 10)
        self.sub = self.create_subscription(Image, image_topic, self.image_cb, img_qos)
        self.get_logger().info(f'Listening on {image_topic}, publishing {detections_topic}')

    def image_cb(self, msg: Image):
        w, h = msg.width, msg.height
        box_w, box_h = max(40, w//6), max(40, h//6)
        cx, cy = w/2.0, h/2.0

        det = Detection2D()
        det.header = msg.header  # copy stamp & frame_id
        det.bbox = BoundingBox2D()

        # ---- robust center assignment across message variants ----
        c = det.bbox.center
        try:
            # common cases: Pose2D or Point
            if hasattr(c, "x") and hasattr(c, "y"):
                c.x = float(cx)
                c.y = float(cy)
            elif hasattr(c, "position") and hasattr(c.position, "x"):
                c.position.x = float(cx)
                c.position.y = float(cy)
            elif hasattr(c, "pose") and hasattr(c.pose, "position"):
                c.pose.position.x = float(cx)
                c.pose.position.y = float(cy)
            else:
                raise AttributeError("Unsupported center type")
        except Exception as e:
            # log once then continue (box will still have size set below)
            self.get_logger().warn(f"Could not set bbox center: {e}")

        det.bbox.size_x = float(box_w)
        det.bbox.size_y = float(box_h)
        # ----------------------------------------------------------

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = 'dummy'
        hyp.hypothesis.score = 0.99
        det.results.append(hyp)

        out = Detection2DArray()
        out.header = msg.header
        out.detections.append(det)
        self.pub.publish(out)

def main():
    rclpy.init()
    node = DummyDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
