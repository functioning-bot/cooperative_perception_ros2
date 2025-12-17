#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from vision_msgs.msg import Detection3DArray
import numpy as np
import csv
import os

# Helper to convert ROS Time to float seconds
def to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

class DataInspector(Node):
    def __init__(self):
        super().__init__('data_inspector')
        
        # CSV Setup
        self.csv_file = 'inspection_log.csv'
        self.f = open(self.csv_file, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow([
            'Frame', 'Timestamp', 
            'R1_Count', 'R1_First_X', 'R1_First_Y', 
            'R2_Count', 'R2_First_X', 'R2_First_Y', 
            'Fused_Count', 'Fused_First_X', 'Fused_First_Y',
            'Fusion_Latency_ms'
        ])
        
        # Subscribers
        self.sub_r1 = message_filters.Subscriber(self, Detection3DArray, '/robot_1/detections_3d_map')
        self.sub_r2 = message_filters.Subscriber(self, Detection3DArray, '/robot_2/detections_3d_map')
        self.sub_fuse = message_filters.Subscriber(self, Detection3DArray, '/global_detections')

        # Sync Policy: Wait for all 3 to arrive within 0.1s of each other
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_r1, self.sub_r2, self.sub_fuse], 
            queue_size=20, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        
        self.frame_count = 0
        print(f"🔍 INSPECTOR RUNNING. Logging to {self.csv_file}...")
        print(f"Waiting for synchronized messages...")

    def get_first_pos(self, msg):
        """Returns string '(x, y, z)' of first detection, or 'None'"""
        if not msg.detections:
            return "None", 0.0, 0.0
        pos = msg.detections[0].bbox.center.position
        return f"({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})", pos.x, pos.y

    def callback(self, r1, r2, fused):
        self.frame_count += 1
        
        # 1. Timestamps
        t_r1 = to_sec(r1.header.stamp)
        t_fuse = to_sec(fused.header.stamp)
        latency_ms = (t_fuse - t_r1) * 1000.0  # Approx internal processing time
        
        # 2. Extract Data
        r1_str, r1_x, r1_y = self.get_first_pos(r1)
        r2_str, r2_x, r2_y = self.get_first_pos(r2)
        fuse_str, fuse_x, fuse_y = self.get_first_pos(fused)
        
        # 3. Print to Console (Human Readable)
        print(f"\n--- [Frame {self.frame_count}] Time: {t_r1:.2f} ---")
        print(f"   Robot 1: Found {len(r1.detections)}. First at: {r1_str}")
        print(f"   Robot 2: Found {len(r2.detections)}. First at: {r2_str}")
        print(f"   FUSED  : Found {len(fused.detections)}. First at: {fuse_str}")
        print(f"   > Sync Latency: {latency_ms:.1f} ms")
        
        # 4. Check Alignment (Are they seeing the same thing?)
        if len(r1.detections) > 0 and len(r2.detections) > 0:
            dist = np.sqrt((r1_x - r2_x)**2 + (r1_y - r2_y)**2)
            print(f"   > R1-R2 Distance: {dist:.2f}m")
            if dist < 1.0:
                 print("     ✅ OVERLAP DETECTED (Should fuse to 1)")
            else:
                 print("     ❌ FAR APART (Should keep 2)")

        # 5. Save to CSV
        self.writer.writerow([
            self.frame_count, f"{t_r1:.3f}",
            len(r1.detections), r1_x, r1_y,
            len(r2.detections), r2_x, r2_y,
            len(fused.detections), fuse_x, fuse_y,
            latency_ms
        ])

def main():
    rclpy.init()
    node = DataInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.f.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()