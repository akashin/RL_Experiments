#!/bin/bash

PS_PORT=6000
WORKER_PORT=6100
INDEX=$1
PORT=$(($WORKER_PORT + $INDEX))

python3 main.py \
    --ps_hosts=localhost:$PS_PORT \
    --job_name=worker \
    --task_index=$INDEX \
    --n_workers=100 \
    --workers_start_port=$WORKER_PORT
