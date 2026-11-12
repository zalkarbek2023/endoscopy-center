# One GPU, a whole endoscopy department

**12 real colonoscopy rooms through one NVIDIA DeepStream pipeline, plus a robot endoscope in NVIDIA Isaac Sim,
at the same time on one RTX 5060 Ti (16 GB).**

| Measured | |
|---|---|
| 12 rooms, tiled wall + OSD + H.264, with Isaac Sim running alongside | **30.5 fps per room** (real time) |
| 12 rooms, throughput only (no display) | 46.4 fps per room, 557 fps total |
| Whole GPU while all 13 rooms ran | **4.7 GB VRAM**, 94 % busy |
| Robot room (Isaac Sim + its TensorRT detector) in the same run | 3 of 3 polyps found, 0 false alarms |
| 12 min of real colonoscopy (12 × 60 s) | 12 polyps on screen, **10 flagged**, 2 false alarm episodes |

> Research demo, not a medical device. Video: REAL-Colon (CC BY 4.0).
> 7 of the 12 videos were in the detector's training set, so the accuracy row is not a fair test: this project is about
> throughput. On the 3 videos the detector never saw, 2 of 3 polyps were flagged.

## Pipeline (`center_ds.py`, DeepStream 9.1 PyServiceMaker)

```
12 x H.264 room streams (filesrc -> h264parse -> nvv4l2decoder, NVDEC)
  -> nvstreammux, batch 12, 1352x1080
  -> nvinfer: Ultralytics YOLO26m polyp detector, TensorRT FP16, dynamic batch 1-12, 960 px, NMS-free (C++ parser)
     interval=2: the detector runs on every 3rd frame, the tracker carries boxes in between
  -> nvtracker NvDCF (one id per polyp per room) -> probe: every box -> center_ai_*.csv
  -> nvmultistreamtiler 4x3 -> nvosdbin -> nvv4l2h264enc -> center_wall.mp4
Room 13: Isaac Sim 6.2 robot endoscope (github.com/zalkarbek2023/isaac-endoscope) on the same GPU
```

Rooms are 60 s windows of 12 REAL-Colon procedures, starting 5 s before each one's first polyp (`rooms.txt`), encoded
on the GPU by `encode_rooms.sh`.

Scoring (`summarize.py`): a polyp is flagged if an AI box (conf ≥ 0.45) overlaps its expert box (IoU ≥ 0.2) on ≥ 3
frames. An alarm episode means ≥ 3 of the last 5 detector passes are ≥ 0.45; it is false if none of its boxes sits on a
polyp.

## Lessons

- A static-batch-12 engine failed with "Failed to queue input batch" because the muxer sends partial batches. Export
  with a dynamic batch and set `infer-dims=3;960;960` in the nvinfer config.
- With `interval=2`, confident boxes exist only on detector frames. Alarm rules must count detector passes, not frames.
- Encode the rooms with NVENC and decode them with NVDEC; JPEG sequences would make the CPU the bottleneck.

## Run

```bash
./encode_rooms.sh                                  # in the DeepStream 9.1 container: 12 rooms -> rooms/*.h264
python export_b12.py                               # YOLO26 -> end2end ONNX, dynamic batch up to 12
./run_bench.sh bench                               # throughput + GPU monitor
./run_center.sh                                    # 12 rooms + the Isaac Sim robot at the same time
python summarize.py                                # score vs expert boxes -> results/center_data.json
cd video && npm i && npx remotion render src/index.ts Center out/center.mp4 --gl=angle
```

## Credits

- NVIDIA DeepStream 9.1 (GStreamer), TensorRT, Isaac Sim 6.2. Ultralytics YOLO26 (AGPL-3.0). Remotion.
- REAL-Colon: Biffi et al., *Scientific Data* 2024, CC BY 4.0.

Zalkarbek Arapbaev, physician and CV engineer.
