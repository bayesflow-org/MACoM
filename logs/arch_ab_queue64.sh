#!/bin/bash
# Start the embed64 control arm on GPU 1 once the design-specific fine-tune (pid 2552804) exits.
while kill -0 2552804 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') fine-tune exited; starting embed64 control on GPU 1"
CUDA_VISIBLE_DEVICES=1 exec taskset -c 24-47 /home/jerry/Documents/research/projects/CogFormer-2/logs/arch_ab.sh 64
