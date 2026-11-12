#!/bin/bash
# DeepStream 12-room run (bench or wall) with a VRAM/utilisation monitor; log out/<mode>.log, GPU samples out/gpu_<mode>.csv
cd ~/endo-center; MODE=${1:-bench}; shift
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > out/gpu_$MODE.csv 2>/dev/null &
MON=$!
docker run --rm --gpus all -v ~/endo-center:/work live3d:ds9.1 python3 /work/center_ds.py $([ $MODE = bench ] && echo --bench) "$@" > out/$MODE.log 2>&1
kill $MON
echo "peak VRAM MiB: $(sort -n out/gpu_$MODE.csv | tail -1 | cut -d, -f1) | mean util: $(awk -F, '$2>30{s+=$2;n++} END{if(n) print int(s/n)}' out/gpu_$MODE.csv)%"
grep -E "total_fps" out/$MODE.log
echo "RUN DONE $MODE"
