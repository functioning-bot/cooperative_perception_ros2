# Cooperative Perception with Multi-Robot Communication
## Abstract

This project addresses the critical safety issue of "blind spots" in autonomous mobile robotics. By implementing a decentralized Cooperative Perception System, two independent robots share processed vision data to construct a unified 3D map of the environment.

The system utilizes a Late Fusion architecture, where object detection (YOLO) and 3D lifting occur locally on each robot, and only lightweight coordinates are transmitted to a central fusion node. This approach minimizes bandwidth usage while allowing robots to "see around corners." The system was validated across Native Linux, VM, and Docker environments, achieving **100% duplicate suppression** and **real-time performance (24 Hz)** on GPU-accelerated hardware.

---

## System Architecture

The pipeline consists of four modular ROS 2 nodes:

1. **Camera Node (Isaac Sim):** Publishes RGB and Depth images.
2. **Detection Node:** Runs **YOLO (v8/v10/v11)** to generate 2D bounding boxes.
3. **3D Lifting Node:** Converts 2D boxes + Depth maps into 3D centroids.
4. **Fusion Node:** Synchronizes streams from Robot 1 & 2 and applies **Mahalanobis Distance** logic to merge duplicate detections into a global object list.

---

## Performance & Results

The system was benchmarked using a "Record and Replay" methodology on three hardware configurations.

### 1. Scalability Analysis

| Environment | Hardware Specs | Throughput (FPS) | Latency | Status |
| --- | --- | --- | --- | --- |
| **Linux (Native)** | **RTX 4090 (24GB VRAM)** | **24.1 Hz** | **1.5 ms** | Real-Time |
| **VM (Dev)** | 2 Cores, 8GB RAM | 4.8 Hz | 37.1 ms | Compute Bound |
| **Docker** | Constrained Network | 2.9 Hz | 43.2 ms | I/O Bound |

### 2. Model Evaluation

| Model | Duplicate Suppression | Recall Gain (Blind Spots) | Stability (Jitter) |
| --- | --- | --- | --- |
| **YOLOv8n** | **99.6%** | **85.1%** | 23.3 cm |
| **YOLOv10n** | 98.9% | 68.0% | 20.4 cm |
| **YOLOv11n** | **99.6%** | 68.8% | **18.4 cm** |

 YOLOv11n offers the best positional stability, while YOLOv8n provides the best awareness of hidden objects.


### Visual Evidence

**[Click Here to View Full Results and Videos (Google Drive)](https://drive.google.com/drive/folders/1yY5eTsWqYh0wDqdKsR_nPCLMyA9j9V_7?usp=sharing)**

**[Video Demo](https://www.youtube.com/playlist?list=PLNL_3xUJrITRxopcRANItIr8wRx1tay1N)**

---

## Installation & Setup

You can run this project using **Docker** (recommended for reproducibility) or **Native Linux**.

### Option A: Docker (Reproducibility Mode)

This method ensures the environment matches the submitted report exactly.

**1. Pull & Start Container**

```bash
docker pull ros:jazzy
docker run -it --rm --name ros2-jazzy-dev -v "$(pwd)":/ws ros:jazzy bash

```

**2. Install Dependencies (Inside Container)**

```bash
apt-get update && apt-get install -y python3-venv curl gnupg lsb-release git build-essential cmake
cd /ws

# Set up Virtual Environment
python3 -m venv functioning_bot
source functioning_bot/bin/activate

# Install ROS 2 Packages
sudo apt install -y ros-jazzy-desktop ros-jazzy-rviz2 python3-colcon-common-extensions python3-rosdep
sudo apt install -y ros-jazzy-cv-bridge ros-jazzy-vision-msgs ros-jazzy-tf2-ros ros-jazzy-tf-transformations

# Install Python Libraries
pip install "numpy==1.26.4" opencv-python ultralytics onnx onnxruntime

```

**3. Build Workspace**

```bash
touch bags/COLCON_IGNORE logs/COLCON_IGNORE functioning_bot/COLCON_IGNORE
colcon build
source install/setup.bash

```

### Option B: Native Linux (GPU Accelerated)

*Requires Ubuntu 24.04, ROS 2 Jazzy, and NVIDIA Drivers.*

```bash
# Create venv and install requirements
python3 -m venv venv
source venv/bin/activate
pip install "numpy==1.26.4" opencv-python ultralytics

# Build
colcon build --symlink-install
source install/setup.bash

```

---

## How to Run

### 1. Launch the System

This script launches the Simulation Bridge, Detectors, and Fusion Node.

```bash
# Usage: ./run_experiment.sh [model_name]
./run_experiment.sh yolov11n.pt

```

### 2. Replay Validation Data

In a new terminal, inject the sensor data:

```bash
ros2 bag play bags/v8/raw/Scene_1

```

### 3. Visualization

Open RViz2 to see the fusion in action:

* **Green Marker:** Robot 1 View
* **Green Marker:** Robot 2 View
* **Blue Marker:** **Fused/Global Output**

---

## Project Structure

```text
.
├── src/
│   ├── robot_perception/
├── run_experiment.sh              # Main entry point
├── final_evaluator.py             # Metrics calculation
├── resource_logger_gpu.sh         # Hardware profiling
├── bags/                              # ROS 2 Bag files (Data)
└── results/                           # CSV Logs and Plots

```
