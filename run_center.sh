#!/bin/bash
# 13 rooms on one GPU at the same time: Isaac Sim robot endoscope (+ its TensorRT detector) AND DeepStream with 12 rooms
# (tiled wall + OSD + H.264). GPU monitor for the whole window. Log: out/center_run.txt
cd ~/robot-endo
[ -f out/detector.pid ] && kill $(cat out/detector.pid) 2>/dev/null; sleep 1
ENGINE=$HOME/robot-endo/out/polyp_sim.engine setsid nohup ~/medai-colon/.venv/bin/python detector_server.py > out/detector.log 2>&1 < /dev/null & echo $! > out/detector.pid
for i in $(seq 1 30); do grep -q "detector ready" out/detector.log && break; sleep 2; done
docker run --rm -u 0 -v ~/robot-endo/out:/o --entrypoint bash nvcr.io/nvidia/isaac-sim:6.2.0.1 -c "rm -rf /o/center_robot && chmod -R 777 /o" 2>/dev/null
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > ~/endo-center/out/gpu_center.csv 2>/dev/null &
MON=$!
T0=$(date +%s.%N)
./run_isaac.sh endo_sim.py center_robot 0.010 1 > out/center_robot.log 2>&1 &
ROBOT=$!
sleep 25                                   # let Isaac start rendering before the 12 rooms join
cd ~/endo-center && docker run --rm --gpus all -v ~/endo-center:/work live3d:ds9.1 python3 /work/center_ds.py > out/wall.log 2>&1
T1=$(date +%s.%N)
wait $ROBOT
kill $MON
echo "peak VRAM MiB: $(sort -n out/gpu_center.csv | tail -1 | cut -d, -f1) | mean util: $(awk -F, '$2>30{s+=$2;n++} END{if(n) print int(s/n)}' out/gpu_center.csv)%"
grep -E "total_fps" out/wall.log
grep SUMMARY ~/robot-endo/out/center_robot.log | cut -c1-260
python3 - <<'PY'
import json, os
L = [json.loads(l) for l in open(os.path.expanduser("~/robot-endo/out/center_robot/run.jsonl"))]
print("robot frames", len(L), "| mean detector ms", round(sum(x["ms"] for x in L) / len(L), 1))
PY
echo "CENTER DONE"
