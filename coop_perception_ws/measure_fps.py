#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
import time

class FpsMeter(Node):
    def __init__(self):
        super().__init__('fps_meter')
        
        self.count = 0
        self.start_time = None
        self.last_msg_time = None
        
        # Subscribe to the final output
        self.sub = self.create_subscription(
            Detection3DArray, 
            '/global_detections', 
            self.callback, 
            10
        )
        
        print("📉 FPS METER RUNNING. Waiting for data...")

    def callback(self, msg):
        current_time = time.time()
        
        # Initialize timer on first message
        if self.start_time is None:
            self.start_time = current_time
            print("   (Data stream started...)")
        
        self.count += 1
        self.last_msg_time = current_time
        
        # Simple live update every 10 frames
        if self.count % 10 == 0:
            elapsed = current_time - self.start_time
            current_fps = self.count / elapsed
            print(f"   Measuring... Current Avg: {current_fps:.2f} Hz", end='\r')

    def print_report(self):
        print("\n\n" + "="*30)
        print("   FPS PERFORMANCE REPORT")
        print("="*30)
        
        if self.start_time is None or self.count == 0:
            print("No messages received!")
        else:
            # Calculate final stats
            total_duration = self.last_msg_time - self.start_time
            if total_duration > 0:
                avg_fps = self.count / total_duration
                print(f"Total Messages: {self.count}")
                print(f"Duration:       {total_duration:.2f} sec")
                print(f"Average Rate:   {avg_fps:.2f} Hz")
            else:
                print("Duration too short to measure.")
        
        print("="*30 + "\n")

def main():
    rclpy.init()
    node = FpsMeter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.print_report()
    rclpy.shutdown()

if __name__ == '__main__':
    main()