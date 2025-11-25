#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose
import message_filters

def euclid(a, b):
    dx=a.position.x-b.position.x; dy=a.position.y-b.position.y; dz=a.position.z-b.position.z
    return math.sqrt(dx*dx+dy*dy+dz*dz)

def maha2(a, b, sx, sy, sz):
    dx=(a.position.x-b.position.x)/max(sx,1e-3)
    dy=(a.position.y-b.position.y)/max(sy,1e-3)
    dz=(a.position.z-b.position.z)/max(sz,1e-3)
    return dx*dx+dy*dy+dz*dz

def mean_pose(pa, pb, wa=0.5):
    from geometry_msgs.msg import Pose
    wb=1.0-wa
    p=Pose()
    p.position.x=wa*pa.position.x+wb*pb.position.x
    p.position.y=wa*pa.position.y+wb*pb.position.y
    p.position.z=wa*pa.position.z+wb*pb.position.z
    p.orientation=pa.orientation
    return p

class FuseRobust(Node):
    def __init__(self):
        super().__init__('fuse_merge_robust')
        # params
        self.declare_parameter('in_topic_r1','/robot_1/detections_3d_map')
        self.declare_parameter('in_topic_r2','/robot_2/detections_3d_map')
        self.declare_parameter('out_topic','/global_detections')
        self.declare_parameter('merge_dist',1.0)
        self.declare_parameter('maha_gate',9.0)          # ~3σ
        self.declare_parameter('size_sigma_frac',0.35)
        self.declare_parameter('class_gate',True)
        self.declare_parameter('publish_rate_hz',20.0)
        self.declare_parameter('max_age_sec',0.25)
        self.declare_parameter('score_avg',True)

        r1=self.get_parameter('in_topic_r1').value
        r2=self.get_parameter('in_topic_r2').value
        out=self.get_parameter('out_topic').value
        self.merge_dist=float(self.get_parameter('merge_dist').value)
        self.maha_gate=float(self.get_parameter('maha_gate').value)
        self.size_sigma_frac=float(self.get_parameter('size_sigma_frac').value)
        self.class_gate=bool(self.get_parameter('class_gate').value)
        self.score_avg=bool(self.get_parameter('score_avg').value)

        self.pub=self.create_publisher(Detection3DArray,out,10)

        self.sub_r1=message_filters.Subscriber(self,Detection3DArray,r1,qos_profile=10)
        self.sub_r2=message_filters.Subscriber(self,Detection3DArray,r2,qos_profile=10)
        self.sub_r1.registerCallback(self._cb_r1)
        self.sub_r2.registerCallback(self._cb_r2)

        self.latest_r1=None; self.t_r1=None
        self.latest_r2=None; self.t_r2=None
        self.max_age=Duration(seconds=float(self.get_parameter('max_age_sec').value))

        self.timer=self.create_timer(1.0/float(self.get_parameter('publish_rate_hz').value), self._tick)
        self.get_logger().info(
            f"robust merge: {r1} + {r2} -> {out} | d<{self.merge_dist} m, "
            f"maha<{self.maha_gate}, σ≈{self.size_sigma_frac}·bbox, rate={1.0/self.timer.timer_period_ns*1e9:.1f} Hz"
        )

    def _cb_r1(self,msg): self.latest_r1=msg; self.t_r1=self.get_clock().now()
    def _cb_r2(self,msg): self.latest_r2=msg; self.t_r2=self.get_clock().now()
    def _fresh(self, t):  return (t is not None) and ((self.get_clock().now()-t) <= self.max_age)

    def _label_and_score(self, det: Detection3D):
        if det.results:
            return det.results[0].hypothesis.class_id, float(det.results[0].hypothesis.score)
        return 'object', 1.0

    def _sigma_from_bbox(self, det: Detection3D):
        sx=max(det.bbox.size.x*self.size_sigma_frac, 0.15)
        sy=max(det.bbox.size.y*self.size_sigma_frac, 0.15)
        sz=max(det.bbox.size.z*self.size_sigma_frac, 0.25)
        return sx, sy, sz

    def _avg_det(self, a: Detection3D, b: Detection3D):
        la,sa=self._label_and_score(a)
        lb,sb=self._label_and_score(b)
        w = sa/(sa+sb+1e-6)
        pose = mean_pose(a.bbox.center, b.bbox.center, wa=w)
        d=Detection3D()
        d.header=a.header
        d.header.frame_id='map'
        d.bbox=BoundingBox3D()
        d.bbox.center=pose
        d.bbox.size.x = 0.5*(a.bbox.size.x+b.bbox.size.x)
        d.bbox.size.y = 0.5*(a.bbox.size.y+b.bbox.size.y)
        d.bbox.size.z = 0.5*(a.bbox.size.z+b.bbox.size.z)
        lab = la if la==lb else la
        sc  = (0.5*(sa+sb) if self.score_avg else max(sa,sb))
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = lab
        hyp.hypothesis.score    = sc
        hyp.pose.pose = pose
        d.results.append(hyp)
        return d

    def _tick(self):
        out=Detection3DArray()
        out.header.stamp = (
            self.latest_r1.header.stamp if (self.latest_r1 and self._fresh(self.t_r1)) else
            (self.latest_r2.header.stamp if (self.latest_r2 and self._fresh(self.t_r2)) else Time().to_msg())
        )
        out.header.frame_id='map'

        A = self.latest_r1.detections if (self.latest_r1 and self._fresh(self.t_r1)) else []
        B = self.latest_r2.detections if (self.latest_r2 and self._fresh(self.t_r2)) else []

        used=set()
        for a in A:
            la,sa=self._label_and_score(a)
            best_j=-1; best_m=1e9
            for j,b in enumerate(B):
                if j in used: continue
                lb,sb=self._label_and_score(b)
                if self.class_gate and (la!=lb): continue
                # quick Euclid gate
                if euclid(a.bbox.center, b.bbox.center) > self.merge_dist:
                    continue
                # Mahalanobis-style gate using bbox-derived sigmas
                sxa,sya,sza=self._sigma_from_bbox(a)
                sxb,syb,szb=self._sigma_from_bbox(b)
                sx=(sxa+sxb)*0.5; sy=(sya+syb)*0.5; sz=(sza+szb)*0.5
                m2=maha2(a.bbox.center, b.bbox.center, sx, sy, sz)
                if m2 < self.maha_gate and m2 < best_m:
                    best_m=m2; best_j=j
            if best_j>=0:
                out.detections.append(self._avg_det(a, B[best_j]))
                used.add(best_j)
            else:
                out.detections.append(a)

        for j,b in enumerate(B):
            if j not in used:
                out.detections.append(b)

        self.pub.publish(out)

def main():
    rclpy.init()
    node = FuseRobust()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
