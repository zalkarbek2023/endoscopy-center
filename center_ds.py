#!/usr/bin/env python3
"""Endoscopy center on one GPU: 12 colonoscopy rooms through ONE DeepStream 9.1 pipeline (PyServiceMaker).

12 x H.264 (filesrc -> h264parse -> nvv4l2decoder, NVDEC) -> nvstreammux (batch 12) -> nvinfer: YOLO26m polyp
detector, TensorRT FP16, batch 12, NMS-free (native C++ parser) -> nvtracker NvDCF (an id per polyp per room) ->
probe: one row per box -> nvmultistreamtiler 4x3 -> nvosdbin -> H.264 MP4 (the DeepStream wall).
--bench: no tiler/OSD/encode, fakesink, as fast as possible (throughput).
Outputs in --out: center_ai.csv (room, frame, track, conf, box), center_stats.json, center_wall.mp4.
Usage (DeepStream container): python3 /work/center_ds.py [--bench] [--interval N]
"""
import argparse
import csv
import json
import time
from pathlib import Path

from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

ROOMS = [l.split()[0] for l in Path("/work/rooms.txt").read_text().split("\n") if l.strip()]
TRACKER = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
CFG = """[property]
gpu-id=0
onnx-file=/work/models/polyp_b12.onnx
model-engine-file=/work/models/polyp_b12.onnx_b12_gpu0_fp16.engine
batch-size=12
infer-dims=3;960;960
network-mode=2
network-type=0
num-detected-classes=1
custom-lib-path=/work/libyolo26.so
parse-bbox-func-name=NvDsInferParseYolo26
cluster-mode=4
net-scale-factor=0.0039215697906911373
model-color-format=0
maintain-aspect-ratio=1
symmetric-padding=1
process-mode=1
gie-unique-id=1
interval={interval}
workspace-size=2048

[class-attrs-all]
pre-cluster-threshold=0.25
topk=50
"""


class Rows(BatchMetadataOperator):
    def __init__(self, writer, stamps):
        super().__init__()
        self.w, self.stamps = writer, stamps

    def handle_metadata(self, batch_meta):
        n = 0
        for fm in batch_meta.frame_items:
            n += 1
            room = getattr(fm, "pad_index", getattr(fm, "source_id", 0))
            for o in fm.object_items:
                r = o.rect_params
                tid = int(o.object_id) if o.object_id < 2 ** 63 else -1
                self.w.writerow([room, fm.frame_number, tid, round(float(o.confidence), 3), round(r.left), round(r.top),
                                 round(r.left + r.width), round(r.top + r.height)])
        self.stamps.append((time.perf_counter(), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--interval", type=int, default=2)
    ap.add_argument("--out", default="/work/out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = out / "polyp_b12.txt"; cfg.write_text(CFG.format(interval=a.interval))
    tag = "bench" if a.bench else "wall"
    f = open(out / f"center_ai_{tag}.csv", "w", newline="")
    w = csv.writer(f); w.writerow(["room", "frame", "track_id", "conf", "x1", "y1", "x2", "y2"])
    stamps = []

    p = Pipeline("center")
    p.add("nvstreammux", "mux", {"batch-size": 12, "width": 1352, "height": 1080, "batched-push-timeout": 40000})
    for i, v in enumerate(ROOMS):
        (p.add("filesrc", f"src{i}", {"location": f"/work/rooms/{v}.h264"}).add("h264parse", f"parse{i}")
         .add("nvv4l2decoder", f"dec{i}"))
        p.link(f"src{i}", f"parse{i}", f"dec{i}").link((f"dec{i}", "mux"), ("", "sink_%u"))
    (p.add("nvinfer", "polyp", {"config-file-path": str(cfg)})
     .add("nvtracker", "tracker", {"ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                                   "ll-config-file": TRACKER, "tracker-width": 640, "tracker-height": 512}))
    if a.bench:
        p.add("fakesink", "sink", {"sync": False}).link("mux", "polyp", "tracker", "sink")
    else:
        (p.add("nvmultistreamtiler", "tiler", {"rows": 3, "columns": 4, "width": 1920, "height": 1152})
         .add("nvosdbin", "osd").add("nvvideoconvert", "conv")
         .add("capsfilter", "caps", {"caps": "video/x-raw(memory:NVMM), format=I420"})
         .add("nvv4l2h264enc", "enc", {"bitrate": 16000000}).add("h264parse", "p2").add("qtmux", "mp4")
         .add("filesink", "sink", {"location": str(out / "center_wall.mp4"), "sync": False})
         .link("mux", "polyp", "tracker", "tiler", "osd", "conv", "caps", "enc", "p2", "mp4", "sink"))
    p.attach("tracker", Probe("rows", Rows(w, stamps)))
    t0 = time.perf_counter()
    p.start().wait()
    f.close()
    frames = sum(n for _, n in stamps)
    span = stamps[-1][0] - stamps[0][0] if len(stamps) > 1 else 1e-9
    stats = {"mode": tag, "rooms": len(ROOMS), "batches": len(stamps), "frames": frames, "seconds": round(span, 2),
             "total_fps": round(frames / span, 1), "fps_per_room": round(frames / span / len(ROOMS), 1),
             "wall_clock_s": round(time.perf_counter() - t0, 1), "interval": a.interval,
             "model": "YOLO26m polyp, TensorRT FP16, batch 12, 960", "gpu": "RTX 5060 Ti 16 GB", "deepstream": "9.1"}
    print(json.dumps(stats), flush=True)
    (out / f"center_stats_{tag}.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
