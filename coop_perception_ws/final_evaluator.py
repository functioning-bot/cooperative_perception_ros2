#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from vision_msgs.msg import Detection3DArray
import numpy as np
import time
import sys
import os
import csv
from datetime import datetime

# --- CONFIGURATION ---
MATCH_DIST = 1.0            
JITTER_CLASS = 'person'     
OUTPUT_FILE = "results_summary.csv"
# ---------------------

def get_centroids(msg, filter_class=None):
    points = []
    if not msg.detections: return points
    for det in msg.detections:
        if filter_class:
            cid = det.results[0].hypothesis.class_id if det.results else "unknown"
            if cid != filter_class: continue
        pos = det.bbox.center.position
        points.append(np.array([pos.x, pos.y, pos.z]))
    return points

class FinalEvaluator(Node):
    def __init__(self, scene_name, model_name):
        super().__init__('final_evaluator')
        self.scene = scene_name
        self.model = model_name
        
        # METRICS
        self.total_frames = 0
        self.dup_opps = 0; self.dup_success = 0
        self.r1_only_frames = 0; self.r1_saved = 0
        self.latencies = []
        self.jitter_diffs = []; self.last_person_pos = None 

        # TRUE FPS COUNTERS
        self.r1_msg_count = 0
        self.start_time = None
        self.last_msg_time = None

        # 1. Input Subscriber
        self.sub_r1_raw = self.create_subscription(
            Detection3DArray, '/robot_1/detections_3d_map', self.input_fps_cb, 10
        )

        # 2. Synced Subscribers
        self.sub_r1 = message_filters.Subscriber(self, Detection3DArray, '/robot_1/detections_3d_map')
        self.sub_r2 = message_filters.Subscriber(self, Detection3DArray, '/robot_2/detections_3d_map')
        self.sub_fuse = message_filters.Subscriber(self, Detection3DArray, '/global_detections')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_r1, self.sub_r2, self.sub_fuse], queue_size=50, slop=0.2
        )
        self.ts.registerCallback(self.sync_callback)
        
        print(f"✅ LOGGING DATA FOR: Scene={self.scene} | Model={self.model}")

    def input_fps_cb(self, msg):
        t = time.time()
        if self.start_time is None: self.start_time = t
        self.r1_msg_count += 1
        self.last_msg_time = t

    def sync_callback(self, r1, r2, fused):
        self.total_frames += 1
        pts_r1 = get_centroids(r1)
        pts_r2 = get_centroids(r2)
        pts_fuse = get_centroids(fused)
        
        # LATENCY
        t_in = r1.header.stamp.sec + r1.header.stamp.nanosec * 1e-9
        t_out = fused.header.stamp.sec + fused.header.stamp.nanosec * 1e-9
        self.latencies.append(abs(t_out - t_in) * 1000.0)

        # DUP SUPPRESSION
        for p1 in pts_r1:
            for p2 in pts_r2:
                if np.linalg.norm(p1 - p2) < MATCH_DIST:
                    self.dup_opps += 1
                    mid = (p1 + p2) / 2.0
                    if sum(1 for pf in pts_fuse if np.linalg.norm(pf - mid) < MATCH_DIST) == 1:
                        self.dup_success += 1

        # RECALL GAIN
        for p1 in pts_r1:
            if not any(np.linalg.norm(p1 - p2) < MATCH_DIST for p2 in pts_r2):
                self.r1_only_frames += 1
                if any(np.linalg.norm(p1 - pf) < MATCH_DIST for pf in pts_fuse):
                    self.r1_saved += 1

        # JITTER
        person_pts = get_centroids(fused, JITTER_CLASS)
        if person_pts:
            curr = person_pts[0]
            if self.last_person_pos is not None:
                dist = np.linalg.norm(curr - self.last_person_pos)
                self.jitter_diffs.append(dist)
            self.last_person_pos = curr
        
        if self.total_frames % 10 == 0: print(f"Synced {self.total_frames} frames...", end='\r')

    def save_results(self):
        # CALCULATIONS
        fps = 0.0
        if self.start_time and self.last_msg_time:
            dur = self.last_msg_time - self.start_time
            if dur > 0: fps = self.r1_msg_count / dur
        
        dup = (self.dup_success/self.dup_opps*100) if self.dup_opps > 0 else 0.0
        rec = (self.r1_saved/self.r1_only_frames*100) if self.r1_only_frames > 0 else 0.0
        jit = (np.mean(self.jitter_diffs)*100) if self.jitter_diffs else 0.0
        lat_avg = np.mean(self.latencies) if self.latencies else 0.0

        # PRINT TO CONSOLE
        print("\n" + "="*50)
        print(f"   SAVING RESULT FOR {self.scene} / {self.model}")
        print("="*50)
        print(f"FPS: {fps:.2f} | Jitter: {jit:.2f}cm | Dup: {dup:.1f}% | Latency: {lat_avg:.1f}ms")

        # APPEND TO CSV
        file_exists = os.path.isfile(OUTPUT_FILE)
        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                writer.writerow(['Timestamp', 'Scene', 'Model', 'FPS_Hz', 'Latency_ms', 'Jitter_cm', 'Recall_Gain_Pct', 'Dup_Suppression_Pct'])
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.scene,
                self.model,
                f"{fps:.4f}",
                f"{lat_avg:.2f}",
                f"{jit:.2f}",
                f"{rec:.2f}",
                f"{dup:.2f}"
            ])
        print(f"✅ Saved to {OUTPUT_FILE}\n")

def main():
    rclpy.init()
    # Get Args manually (simple way)
    if len(sys.argv) < 3:
        print("Usage: python3 final_evaluator.py [SCENE_NAME] [MODEL_NAME]")
        return

    scene = sys.argv[1]
    model = sys.argv[2]

    node = FinalEvaluator(scene, model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_results()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# import message_filters
# from vision_msgs.msg import Detection3DArray
# import numpy as np
# import time

# # --- CONFIGURATION ---
# MATCH_DIST = 1.0            
# JITTER_CLASS = 'person'     
# # ---------------------

# def get_centroids(msg, filter_class=None):
#     points = []
#     if not msg.detections: return points
#     for det in msg.detections:
#         if filter_class:
#             cid = det.results[0].hypothesis.class_id if det.results else "unknown"
#             if cid != filter_class: continue
#         pos = det.bbox.center.position
#         points.append(np.array([pos.x, pos.y, pos.z]))
#     return points

# class FinalEvaluator(Node):
#     def __init__(self):
#         super().__init__('final_evaluator')
        
#         # METRICS
#         self.total_frames = 0
#         self.dup_opps = 0; self.dup_success = 0
#         self.r1_only_frames = 0; self.r1_saved = 0
#         self.latencies = []
#         self.jitter_diffs = []; self.last_person_pos = None 

#         # TRUE FPS COUNTERS (Measuring the INPUT, which is the bottleneck)
#         self.r1_msg_count = 0
#         self.start_time = None
#         self.last_msg_time = None

#         # 1. Input Subscriber (For True FPS)
#         self.sub_r1_raw = self.create_subscription(
#             Detection3DArray, '/robot_1/detections_3d_map', self.input_fps_cb, 10
#         )

#         # 2. Synced Subscribers (For Quality Metrics)
#         self.sub_r1 = message_filters.Subscriber(self, Detection3DArray, '/robot_1/detections_3d_map')
#         self.sub_r2 = message_filters.Subscriber(self, Detection3DArray, '/robot_2/detections_3d_map')
#         self.sub_fuse = message_filters.Subscriber(self, Detection3DArray, '/global_detections')

#         self.ts = message_filters.ApproximateTimeSynchronizer(
#             [self.sub_r1, self.sub_r2, self.sub_fuse], queue_size=50, slop=0.2
#         )
#         self.ts.registerCallback(self.sync_callback)
        
#         print(f"✅ EVALUATOR READY. Measuring TRUE Throughput (Input Rate)...")

#     def input_fps_cb(self, msg):
#         """Measures the rate of the upstream pipeline (YOLO+Lift+TF)"""
#         t = time.time()
#         if self.start_time is None: self.start_time = t
#         self.r1_msg_count += 1
#         self.last_msg_time = t

#     def sync_callback(self, r1, r2, fused):
#         self.total_frames += 1
#         pts_r1 = get_centroids(r1)
#         pts_r2 = get_centroids(r2)
#         pts_fuse = get_centroids(fused)
        
#         # 1. LATENCY (Sync overhead)
#         t_in = r1.header.stamp.sec + r1.header.stamp.nanosec * 1e-9
#         t_out = fused.header.stamp.sec + fused.header.stamp.nanosec * 1e-9
#         self.latencies.append(abs(t_out - t_in) * 1000.0)

#         # 2. DUP SUPPRESSION
#         for p1 in pts_r1:
#             for p2 in pts_r2:
#                 if np.linalg.norm(p1 - p2) < MATCH_DIST:
#                     self.dup_opps += 1
#                     mid = (p1 + p2) / 2.0
#                     if sum(1 for pf in pts_fuse if np.linalg.norm(pf - mid) < MATCH_DIST) == 1:
#                         self.dup_success += 1

#         # 3. RECALL GAIN
#         for p1 in pts_r1:
#             if not any(np.linalg.norm(p1 - p2) < MATCH_DIST for p2 in pts_r2):
#                 self.r1_only_frames += 1
#                 if any(np.linalg.norm(p1 - pf) < MATCH_DIST for pf in pts_fuse):
#                     self.r1_saved += 1

#         # 4. JITTER
#         person_pts = get_centroids(fused, JITTER_CLASS)
#         if person_pts:
#             curr = person_pts[0]
#             if self.last_person_pos is not None:
#                 dist = np.linalg.norm(curr - self.last_person_pos)
#                 self.jitter_diffs.append(dist)
#             self.last_person_pos = curr

#         if self.total_frames % 10 == 0: print(f"Synced {self.total_frames} frames...", end='\r')

#     def print_report(self):
#         print("\n" + "="*50)
#         print(f"   📋 TRUE PERFORMANCE SUMMARY")
#         print("="*50)
        
#         # TRUE FPS Calculation
#         fps = 0.0
#         if self.start_time and self.last_msg_time:
#             dur = self.last_msg_time - self.start_time
#             if dur > 0: fps = self.r1_msg_count / dur
        
#         # Metrics Calculation
#         dup = (self.dup_success/self.dup_opps*100) if self.dup_opps > 0 else 0.0
#         rec = (self.r1_saved/self.r1_only_frames*100) if self.r1_only_frames > 0 else 0.0
#         jit = (np.mean(self.jitter_diffs)*100) if self.jitter_diffs else 0.0
#         lat_avg = np.mean(self.latencies) if self.latencies else 0.0

#         print(f"1. TRUE THROUGHPUT:    {fps:.2f} Hz (Processing Speed)")
#         print(f"   (Output is timer-fixed at 20Hz, ignoring that.)")
#         print(f"2. LATENCY (Sync):     {lat_avg:.1f} ms")
#         print(f"3. DUP SUPPRESSION:    {dup:.1f}%")
#         print(f"4. RECALL GAIN:        {rec:.1f}%")
#         print(f"5. JITTER:             {jit:.2f} cm")
#         print("="*50 + "\n")

# def main():
#     rclpy.init()
#     node = FinalEvaluator()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.print_report()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()