#!/bin/bash

# 1. Output File from Argument
OUT_FILE=$1

if [ -z "$OUT_FILE" ]; then
    echo "Usage: ./resource_logger.sh [OUTPUT_FILENAME]"
    exit 1
fi

# Create directory if missing
mkdir -p $(dirname "$OUT_FILE")

echo "Timestamp,CPU_Percent,RAM_Used_MB" > $OUT_FILE

echo "📊 LOGGING RESOURCES TO: $OUT_FILE"
echo "   (Press Ctrl+C to stop)"

while true; do
    # CPU Usage (User + Sys)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    
    # RAM Usage (MB)
    RAM_USAGE=$(free -m | awk '/Mem:/ { print $3 }')
    
    TS=$(date +%s)
    
    echo "$TS,$CPU_USAGE,$RAM_USAGE" >> $OUT_FILE
    sleep 1
done
# #!/bin/bash

# # Output File
# LOG_FILE="resource_log.csv"
# echo "Timestamp,CPU_Percent,RAM_Used_MB" > $LOG_FILE

# echo "📊 MONITORING CPU & RAM..."
# echo "   (Press Ctrl+C to stop logging)"

# while true; do
#     # 1. Get CPU Usage (User + System from top command)
#     # top -bn1 runs once. grep "Cpu" finds the line. awk adds user+sys.
#     CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')

#     # 2. Get RAM Usage (Megabytes)
#     RAM_USAGE=$(free -m | awk '/Mem:/ { print $3 }')

#     # 3. Get Current Time
#     TS=$(date +%s)

#     # 4. Save to CSV and Print
#     echo "$TS,$CPU_USAGE,$RAM_USAGE" >> $LOG_FILE
#     echo "   CPU: $CPU_USAGE% | RAM: ${RAM_USAGE}MB"
    
#     sleep 1
# done