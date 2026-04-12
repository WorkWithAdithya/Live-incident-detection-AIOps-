#!/bin/bash

# Detect number of CPU cores
CORES=$(nproc)

echo "Detected CPU cores: $CORES"
echo "Starting gradual stress test..."

DURATION=20

for LEVEL in 10 20 30 40 50 60 70 80 90
do
    echo "--------------------------------"
    echo "Applying ${LEVEL}% stress for ${DURATION} seconds"

    # Calculate workers based on percentage
    CPU_WORKERS=$((CORES * LEVEL / 100))
    MEM_WORKERS=$((CORES * LEVEL / 200))
    DISK_WORKERS=$((CORES * LEVEL / 200))

    # Ensure minimum 1 worker
    [ $CPU_WORKERS -lt 1 ] && CPU_WORKERS=1
    [ $MEM_WORKERS -lt 1 ] && MEM_WORKERS=1
    [ $DISK_WORKERS -lt 1 ] && DISK_WORKERS=1

    stress-ng \
        --cpu $CPU_WORKERS \
        --vm $MEM_WORKERS --vm-bytes 512M \
        --hdd $DISK_WORKERS --hdd-bytes 512M \
        --timeout ${DURATION}s \
        --metrics-brief

done

echo "--------------------------------"
echo "Stress test completed!"

