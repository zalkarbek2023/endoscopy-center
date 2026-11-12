"""polyp_v2 (YOLO26m, 960) -> NMS-free end2end ONNX with a static batch of 12, for one DeepStream nvinfer serving 12 rooms."""
import shutil
from pathlib import Path
from ultralytics import YOLO
H = Path.home()
p = YOLO(str(H / "medai-colon/runs/detect/polyp_v2/weights/best.pt")).export(format="onnx", imgsz=960, end2end=True, batch=12, opset=17, simplify=True, device="cpu")
(H / "endo-center/models").mkdir(parents=True, exist_ok=True)
shutil.copy(p, H / "endo-center/models/polyp_b12.onnx")
import onnx
m = onnx.load(H / "endo-center/models/polyp_b12.onnx")
print([(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input], [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
