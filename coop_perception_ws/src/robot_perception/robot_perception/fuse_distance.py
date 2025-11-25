#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose
import message_filters

def dist(a: Pose, b: Pose) -> float:
    dx = a.position.x - b.position.x
    dy = a.position.y - b.position.y
    dz = a.position.z - b.position.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

class FuseDistance(Node):
    def __init__(self):
        super().__init__('fuse_distance')
        self.declare_parameter('in_topic_r1', '/robot_1/detections_3d_map')
        self.declare_parameter('in_topic_r2', '/robot_2/detections_3d_map')
        self.declare_parameter('out_topic', '/global_detections')
        self.declare_parameter('merge_dist', 0.75)     # meters
        self.declare_parameter('avg_scores', True)

        r1 = self.get_parameter('in_topic_r1').value
        r2 = self.get_parameter('in_topic_r2').value
        out = self.get_parameter('out_topic').value
        self.merge_dist = float(self.get_parameter('merge_dist').value)
        self.avg_scores = bool(self.get_parameter('avg_scores').value)

        self.pub = self.create_publisher(Detection3DArray, out, 10)
        self.sub_r1 = message_filters.Subscriber(self, Detection3DArray, r1, qos_profile=10)
        self.sub_r2 = message_filters.Subscriber(self, Detection3DArray, r2, qos_profile=10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_r1, self.sub_r2], 50, 0.25)
        self.ts.registerCallback(self.cb_sync)
        self.get_logger().info(f"dist-merge: {r1} + {r2} -> {out} (thr={self.merge_dist} m)")

    def cb_sync(self, A: Detection3DArray, B: Detection3DArray):
        out = Detection3DArray()
        out.header = A.header  # both are 'map'
        used = set()

        for da in A.detections:
            j_best, d_best = -1, 1e9
            for j, db in enumerate(B.detections):
                if j in used:
                    continue
                dcur = dist(da.bbox.center, db.bbox.center)
                if dcur < d_best:
                    d_best, j_best = dcur, j
            if j_best >= 0 and d_best <= self.merge_dist:
                out.detections.append(self._avg_pair(da, B.detections[j_best], out.header.stamp))
                used.add(j_best)
            else:
                out.detections.append(da)

        for j, db in enumerate(B.detections):
            if j not in used:
                out.detections.append(db)

        self.pub.publish(out)

    def _avg_pair(self, d1: Detection3D, d2: Detection3D, stamp):
        d = Detection3D()
        d.header = d1.header
        d.header.stamp = stamp

        p1, p2 = d1.bbox.center.position, d2.bbox.center.position
        pose = Pose()
        pose.position.x = 0.5*(p1.x + p2.x)
        pose.position.y = 0.5*(p1.y + p2.y)
        pose.position.z = 0.5*(p1.z + p2.z)
        pose.orientation = d1.bbox.center.orientation

        d.bbox = BoundingBox3D()
        d.bbox.center = pose
        d.bbox.size.x = 0.5*(d1.bbox.size.x + d2.bbox.size.x)
        d.bbox.size.y = 0.5*(d1.bbox.size.y + d2.bbox.size.y)
        d.bbox.size.z = 0.5*(d1.bbox.size.z + d2.bbox.size.z)

        lab = d1.results[0].hypothesis.class_id if d1.results else 'object'
        s1  = float(d1.results[0].hypothesis.score) if d1.results else 1.0
        s2  = float(d2.results[0].hypothesis.score) if d2.results else 1.0
        score = 0.5*(s1 + s2) if self.avg_scores else max(s1, s2)

        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = lab
        hyp.hypothesis.score = score
        hyp.pose.pose = pose
        d.results.append(hyp)
        return d

def main():
    rclpy.init()
    n = FuseDistance()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
