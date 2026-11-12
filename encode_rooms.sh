#!/bin/bash
# 12 "endoscopy rooms": a 60 s window of each REAL-Colon procedure (from 5 s before its first polyp) -> H.264 MP4,
# encoded on the GPU inside the DeepStream 9.1 container. Output: ~/endo-center/rooms/<video>.mp4
cd /work
mkdir -p rooms
while read v start; do
  [ -f rooms/$v.h264 ] && continue
  gst-launch-1.0 -q multifilesrc location=/frames/$v/${v}_frames/${v}_%d.jpg index=$start stop-index=$((start + 1799)) \
    caps="image/jpeg,framerate=30/1" ! jpegparse ! nvv4l2decoder ! nvvideoconvert ! "video/x-raw(memory:NVMM),format=I420" \
    ! nvv4l2h264enc bitrate=8000000 iframeinterval=30 ! h264parse ! "video/x-h264,stream-format=byte-stream" ! filesink location=rooms/$v.h264 2>&1 | tail -2
  echo "$v $(stat -c %s rooms/$v.h264)"
done < rooms.txt
echo ENCODE DONE
