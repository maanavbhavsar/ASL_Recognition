import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_YOLOV5_REPO = _PROJECT_ROOT / "yolov5_repo"
if not _YOLOV5_REPO.exists():
    raise RuntimeError(
        "YOLOv5 repo not found. Run setup first:\n"
        "  Windows: .\\setup.ps1\n"
        "  Linux/Mac: bash setup.sh"
    )
sys.path.insert(0, str(_YOLOV5_REPO))

import cv2
import torch

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

img_size = (448, 448)

# Use local weights if present; DetectMultiBackend auto-downloads from GitHub if missing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = str(_PROJECT_ROOT / "weights" / "yolov5s.pt")
model = DetectMultiBackend(weights=weights_path, device=str(device))
model.eval()

img_path = Path('Project Milestone - ASL Detection/Images/test1.jpg')
img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")
img = cv2.resize(img, img_size)

img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

with torch.no_grad():
    pred = model(img_tensor)[0]
    preds = non_max_suppression(pred, 0.25, 0.45, agnostic=False)
    pred = preds[0] if len(preds) > 0 else pred.new_empty((0, 6))

annotator = Annotator(img, line_width=1, example=str(model.names))
if len(pred) > 0:
    pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], img.shape[:2]).round()

    for *xyxy, conf, cls in pred.tolist():
        label = model.names[int(cls)]
        annotator.box_label(xyxy, label, color=colors(int(cls), True))

result_img = annotator.result()
save_dir = Path('results')
save_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(save_dir / 'result.jpg', result_img)