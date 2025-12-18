#!/bin/bash
# Usage: ./resource_logger_gpu.sh [output_filename.csv]

OUT_FILE=${1:-resource_log_gpu.csv}
mkdir -p $(dirname "$OUT_FILE")

# Write Header
echo "Timestamp,CPU_Percent,RAM_Used_MB,GPU_Util_Percent,GPU_Mem_MB" > $OUT_FILE
echo "LOGGING GPU + SYSTEM RESOURCES TO: $OUT_FILE"

while true; do
    # 1. Get CPU Usage (100 - Idle)
    CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    
    # 2. Get RAM Usage
    RAM=$(free -m | awk '/Mem:/ { print $3 }')

    # 3. Get GPU Stats (Utilization % and Memory Used MB)
    # Returns specific values like "45, 1024"
    GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
    
    # 4. Save to CSV
    echo "$(date +%s),$CPU,$RAM,$GPU_STATS" >> $OUT_FILE
    
    # Log every 1 second
    sleep 1
done