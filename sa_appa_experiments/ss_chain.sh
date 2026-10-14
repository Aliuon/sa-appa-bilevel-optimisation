#!/bin/bash
# main strong Stackelberg runs (resumable), then the side experiments
cd "$(dirname "$0")"
python3 run_strong_stackelberg.py --workers 2 >> results/strong_stackelberg/run.log 2>&1
python3 run_ss_extras.py --workers 2 >> results/strong_stackelberg/extras.log 2>&1
