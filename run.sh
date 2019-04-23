#!/bin/bash
python a2c.py --job_name "ps" --task_index 0 --num_workers 2 &
python a2c.py --job_name "worker" --task_index 0 --num_workers 2 &
python a2c.py --job_name "worker" --task_index 1 --num_workers 2 &
