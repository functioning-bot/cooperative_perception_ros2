#!/bin/bash

# 1. INPUT: Model Name (default to yolov8n.pt if empty)
MODEL=${1:-yolov8n.pt}
echo "=========================================================="
echo " 🚀 LAUNCHING PIPELINE WITH MODEL: $MODEL"
echo "=========================================================="

# 2. SETUP: cleanup on exit
trap "kill 0" EXIT

# 3. COMMON ARGS (Injecting Sim Time for correct metrics!)
ARGS="--ros-args -p use_sim_time:=true"

# --- ROBOT 1 ---
echo "[1/7] R1 YOLO ($MODEL)..."
python3 -m robot_perception.yolo2d_detector $ARGS \
  -p image_topic:=/robot_1/camera/image_raw \
  -p detections_topic:=/robot_1/detections_2d \
  -p annotated_topic:=/robot_1/camera/annotated \
  -p model_path:=$MODEL \
  -p conf_thresh:=0.25 -p allowed_classes:="['person','car']" \
  -p publish_annotated:=false -p imgsz:=640 &

echo "[2/7] R1 Lift..."
python3 -m robot_perception.yolo_lift_3d_sync_backup $ARGS \
  -p det_topic:=/robot_1/detections_2d \
  -p depth_topic:=/robot_1/camera/depth/image_rect_raw \
  -p camera_info_topic:=/robot_1/camera/camera_info \
  -p out_topic:=/robot_1/detections_3d \
  -p marker_topic:=/robot_1/detections_markers \
  -p global_frame:=map -p input_frame_is_mount:=true \
  -p sample:=grid -p grid_step:=3 -p box_depth_percentile:=20.0 \
  -p min_depth:=0.4 -p max_depth:=30.0 &

echo "[3/7] R1 Map Republish..."
ros2 run robot_perception to_map_republisher $ARGS \
  -p in_topic:=/robot_1/detections_3d \
  -p out_topic:=/robot_1/detections_3d_map \
  -p target_frame:=map &

# --- ROBOT 2 ---
echo "[4/7] R2 YOLO ($MODEL)..."
python3 -m robot_perception.yolo2d_detector $ARGS \
  -p image_topic:=/robot_2/camera/image_raw \
  -p detections_topic:=/robot_2/detections_2d \
  -p annotated_topic:=/robot_2/camera/annotated \
  -p model_path:=$MODEL \
  -p conf_thresh:=0.25 -p allowed_classes:="['person','car']" \
  -p publish_annotated:=false -p imgsz:=640 &

echo "[5/7] R2 Lift..."
python3 -m robot_perception.yolo_lift_3d_sync_backup $ARGS \
  -p det_topic:=/robot_2/detections_2d \
  -p depth_topic:=/robot_2/camera/depth/image_rect_raw \
  -p camera_info_topic:=/robot_2/camera/camera_info \
  -p out_topic:=/robot_2/detections_3d \
  -p marker_topic:=/robot_2/detections_markers \
  -p global_frame:=map -p input_frame_is_mount:=true \
  -p sample:=grid -p grid_step:=3 -p box_depth_percentile:=20.0 \
  -p min_depth:=0.4 -p max_depth:=30.0 &

echo "[6/7] R2 Map Republish..."
ros2 run robot_perception to_map_republisher $ARGS \
  -p in_topic:=/robot_2/detections_3d \
  -p out_topic:=/robot_2/detections_3d_map \
  -p target_frame:=map &

# --- FUSION ---
echo "[7/7] Fusion Node..."
ros2 run robot_perception fuse_merge_robust $ARGS \
  -p in_topic_r1:=/robot_1/detections_3d_map \
  -p in_topic_r2:=/robot_2/detections_3d_map \
  -p out_topic:=/global_detections \
  -p merge_dist:=1.2 -p maha_gate:=12.0 \
  -p size_sigma_frac:=0.35 -p class_gate:=true \
  -p publish_rate_hz:=20.0 -p max_age_sec:=0.25 &

echo "✅ SYSTEM RUNNING. Ready for bag replay."
wait