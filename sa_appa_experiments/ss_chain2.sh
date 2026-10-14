#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f ss_chain.sh >/dev/null; do sleep 30; done
python3 run_strong_stackelberg.py --workers 2 >> results/strong_stackelberg/run.log 2>&1
python3 run_ss_extras.py --workers 2 >> results/strong_stackelberg/extras.log 2>&1
