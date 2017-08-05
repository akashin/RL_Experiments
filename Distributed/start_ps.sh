#!/bin/bash

PS_PORT=6000
WORKER_PORT=6100
INDEX=$1
PORT=$(($PS_PORT + $INDEX))

python3 main.py \
    --ps_hosts=localhost:$PORT \
    --job_name=ps \
    --task_index=$INDEX \
    --n_workers=100 \
    --workers_start_port=$WORKER_PORT
