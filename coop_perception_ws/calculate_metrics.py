#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
from vision_msgs.msg import Detection3DArray
import numpy as np

# --- CONFIGURATION FROM YOUR LOGS ---
MATCH_DIST = 1.0           # Meters. Your logs show overlaps are ~0.11m to 0.35m, so 1.0 is safe.
JITTER_ROI = [-7.0, -5.0]  # X-range to identify the "Person" for jitter calc (based on your logs: -6.19)
# ------------------------------------

def get_centroids(msg):
    """Returns list of numpy points (x, y, z) for all detections in message."""
    points = []
    if not msg.detections:
        return points
    for det in msg.detections:
        pos = det.bbox.center.position
        points.append(np.array([pos.x, pos.y, pos.z]))
    return points

class MetricCalculator(Node):
    def __init__(self):
        super().__init__('metric_calculator')
        
        # 1. ACCURACY COUNTERS
        self.total_frames = 0
        self.dup_opportunities = 0  # Times R1 & R2 saw the same object
        self.dup_successes = 0      # Times Fusion correctly output 1 object
        
        # 2. RECALL COUNTERS
        self.frames_r1_only = 0     # Times R2 missed object but R1 saw it
        self.frames_r1_saved = 0    # Times Fusion kept that object
        
        # 3. JITTER DATA
        self.person_positions = []  # Stores X,Y of the person
        
        # 4. LATENCY DATA
        self.latencies = []

        # 5. SETUP SUBSCRIBERS
        self.sub_r1 = message_filters.Subscriber(self, Detection3DArray, '/robot_1/detections_3d_map')
        self.sub_r2 = message_filters.Subscriber(self, Detection3DArray, '/robot_2/detections_3d_map')
        self.sub_fuse = message_filters.Subscriber(self, Detection3DArray, '/global_detections')

        # Sync Policy: 0.15s window (covers your 16ms to 83ms jitter)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_r1, self.sub_r2, self.sub_fuse], 
            queue_size=50, slop=0.15
        )
        self.ts.registerCallback(self.callback)
        
        print(f"[{self.get_name()}] Ready. Waiting for bag replay...")

    def callback(self, r1, r2, fused):
        self.total_frames += 1
        
        # --- A. LATENCY (End-to-End Proxy) ---
        # Note: True E2E requires Image timestamp. This measures Sync overhead.
        t_in = r1.header.stamp.sec + r1.header.stamp.nanosec * 1e-9
        t_out = fused.header.stamp.sec + fused.header.stamp.nanosec * 1e-9
        lat = abs(t_out - t_in) * 1000.0 # ms
        self.latencies.append(lat)

        pts_r1 = get_centroids(r1)
        pts_r2 = get_centroids(r2)
        pts_fuse = get_centroids(fused)
        
        # --- B. DUPLICATE SUPPRESSION ---
        # Check every R1 detection against every R2 detection
        for p1 in pts_r1:
            for p2 in pts_r2:
                # If inputs are close (Overlap exists)
                if np.linalg.norm(p1 - p2) < MATCH_DIST:
                    self.dup_opportunities += 1
                    
                    # Does Fusion have exactly ONE object here?
                    midpoint = (p1 + p2) / 2.0
                    matches = 0
                    for pf in pts_fuse:
                        if np.linalg.norm(pf - midpoint) < MATCH_DIST:
                            matches += 1
                    
                    if matches == 1:
                        self.dup_successes += 1

        # --- C. RECALL GAIN (Scenario: R2 is blind) ---
        # If R1 sees something but R2 sees NOTHING nearby
        for p1 in pts_r1:
            has_r2_match = any(np.linalg.norm(p1 - p2) < MATCH_DIST for p2 in pts_r2)
            
            if not has_r2_match:
                self.frames_r1_only += 1
                # Did fusion keep it?
                has_fuse_match = any(np.linalg.norm(p1 - pf) < MATCH_DIST for pf in pts_fuse)
                if has_fuse_match:
                    self.frames_r1_saved += 1

        # --- D. JITTER (Track the specific "Person" object) ---
        # We filter by X position to find the person (who is around x=-6.0 in your logs)
        for pf in pts_fuse:
            if JITTER_ROI[0] < pf[0] < JITTER_ROI[1]:
                self.person_positions.append(pf)

        if self.total_frames % 50 == 0:
            print(f"Processed {self.total_frames} frames...", end='\r')

    def print_report(self):
        print("\n\n" + "="*45)
        print(f"   📊 FINAL PERFORMANCE METRICS ")
        print("="*45)
        
        # 1. Duplicate Suppression
        if self.dup_opportunities > 0:
            dup_rate = (self.dup_successes / self.dup_opportunities) * 100.0
        else: 
            dup_rate = 0.0
        print(f"1. Duplicate Suppression: {dup_rate:.2f}%")
        print(f"   (Correctly merged {self.dup_successes} of {self.dup_opportunities} overlaps)")

        # 2. Recall Gain
        if self.frames_r1_only > 0:
            recall_rate = (self.frames_r1_saved / self.frames_r1_only) * 100.0
        else:
            recall_rate = 0.0
        print(f"2. Fusion Recall Gain:    {recall_rate:.2f}%")
        print(f"   (Maintained {self.frames_r1_saved} objects that R2 missed)")

        # # 3. Jitter
        # if len(self.person_positions) > 10:
        #     arr = np.array(self.person_positions)
        #     std_x = np.std(arr[:, 0]) * 100.0 # cm
        #     std_y = np.std(arr[:, 1]) * 100.0 # cm
        #     mean_jitter = (std_x + std_y) / 2.0
        #     print(f"3. Positional Jitter:     {mean_jitter:.2f} cm")
        # else:
        #     print(f"3. Positional Jitter:     N/A (Not enough samples in ROI {JITTER_ROI})")

        # 3. Jitter (Sliding Window of 5 frames to ignore walking motion)
        if len(self.person_positions) > 10:
            arr = np.array(self.person_positions)
            
            # Calculate distance between consecutive frames (velocity + noise)
            # We assume constant velocity, so 'acceleration' (diff of diff) is mostly noise
            diffs = np.diff(arr, axis=0) # (x2-x1, y2-y1, z2-z1)
            dist_per_frame = np.linalg.norm(diffs, axis=1)
            
            # The standard deviation of the *step size* approximates jitter for a moving object
            jitter_metric = np.std(dist_per_frame) * 100.0
            
            print(f"3. Positional Jitter:     {jitter_metric:.2f} cm (Frame-to-Frame Stability)")
        else:
            print(f"3. Positional Jitter:     N/A")

        # 4. Latency
        if self.latencies:
            avg_lat = np.mean(self.latencies)
            p95_lat = np.percentile(self.latencies, 95)
            print(f"4. Fusion Latency:        Avg: {avg_lat:.1f} ms | p95: {p95_lat:.1f} ms")
        
        print("="*45 + "\n")

def main():
    rclpy.init()
    node = MetricCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.print_report()
    rclpy.shutdown()

if __name__ == '__main__':
    main()