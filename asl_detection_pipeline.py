"""
Combined YOLOv5 + I3D ASL Detection Pipeline

Uses YOLOv5 for spatial gesture/person detection (ROI cropping) and I3D for
temporal sign classification. Run from project root; requires YOLOv5 repo
structure (models/, utils/).

Expected weights (place in weights/):
  - yolov5s.pt      : YOLOv5 pretrained (e.g. from ultralytics/yolov5)
  - rgb_imagenet.pt : I3D pretrained on Kinetics (included)
  - nslt_2000_best.pt : I3D fine-tuned on WLASL (upload when ready)
"""

import argparse
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
import numpy as np
import torch

# YOLOv5 imports (from yolov5_repo after setup)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes

# Project imports
from pytorch_i3d import InceptionI3d

# -----------------------------------------------------------------------------
# Config - Update weight paths when you upload them
# -----------------------------------------------------------------------------
WEIGHTS_DIR = Path("weights")
YOLOV5_WEIGHTS = WEIGHTS_DIR / "yolov5s.pt"
I3D_PRETRAINED = WEIGHTS_DIR / "rgb_imagenet.pt"
I3D_ASL_WEIGHTS = WEIGHTS_DIR / "nslt_2000_best.pt"
ASL_CLASS_LIST = Path("wlasl_class_list.txt")

I3D_NUM_CLASSES = 2000
I3D_FRAME_COUNT = 64
I3D_INPUT_SIZE = 224
YOLOV5_IMG_SIZE = 640
COCO_PERSON_CLASS_ID = 0


def load_asl_classes(class_list_path: Path) -> dict[int, str]:
    """Load ASL sign class names from wlasl_class_list.txt (format: index\\tname)."""
    classes = {}
    if not class_list_path.exists():
        raise FileNotFoundError(f"ASL class list not found: {class_list_path}")
    with open(class_list_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                idx, name = int(parts[0]), parts[1]
                classes[idx] = name
    return classes


def load_yolov5(weights_path: Path, device: torch.device):
    """Load YOLOv5 model for ROI detection."""
    model = DetectMultiBackend(weights=str(weights_path), device=str(device))
    model.eval()
    return model


def _load_state_dict(model: torch.nn.Module, path: Path, device: torch.device, strict: bool = True):
    """Load state dict, handling DataParallel 'module.' prefix."""
    state = torch.load(path, map_location=device)
    if state and list(state.keys())[0].startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)


def load_i3d(
    pretrained_path: Path,
    asl_weights_path: Path,
    num_classes: int,
    device: torch.device,
):
    """Load I3D model with ASL-trained head."""
    i3d = InceptionI3d(400, in_channels=3)
    _load_state_dict(i3d, pretrained_path, device, strict=True)
    i3d.replace_logits(num_classes)
    if asl_weights_path.exists():
        _load_state_dict(i3d, asl_weights_path, device, strict=False)
    i3d.to(device)
    i3d.eval()
    return i3d


def get_roi_from_yolov5(
    model,
    frame: np.ndarray,
    img_size: tuple[int, int],
    device: torch.device,
    target_class: int = COCO_PERSON_CLASS_ID,
):
    """
    Run YOLOv5 on a frame and return the largest bbox for target_class.
    Returns (x1, y1, x2, y2) or None if no detection.
    """
    h, w = frame.shape[:2]
    img_resized = cv2.resize(frame, img_size)
    img_tensor = (
        torch.from_numpy(img_resized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
        / 255.0
    )

    with torch.no_grad():
        pred = model(img_tensor)[0]
        preds = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

    if len(preds) == 0:
        return None

    pred = preds[0]
    if len(pred) == 0:
        return None

    pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], frame.shape[:2]).round()

    # Filter by target class (person) and take largest bbox by area
    mask = pred[:, 5] == target_class
    person_preds = pred[mask]
    if len(person_preds) == 0:
        return None

    areas = (person_preds[:, 2] - person_preds[:, 0]) * (
        person_preds[:, 3] - person_preds[:, 1]
    )
    best_idx = areas.argmax().item()
    xyxy = person_preds[best_idx, :4].cpu().numpy()
    return tuple(map(int, xyxy))


def read_video_frames(video_path: Path, num_frames: int = I3D_FRAME_COUNT) -> np.ndarray:
    """
    Read video and return array of frames (T, H, W, C).
    Pads by repeating last frame if video is shorter than num_frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total > 0:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = cap.read()
            if ret:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        while len(frames) < num_frames:
            ret, img = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    # Pad by repeating last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.array(frames[:num_frames], dtype=np.float32)


def crop_frames_to_roi(
    frames: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    target_size: int = I3D_INPUT_SIZE,
) -> np.ndarray:
    """Crop frames to ROI (or center crop if None) and resize to target_size."""
    t, h, w, c = frames.shape

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            roi = None

    if roi is None:
        # Center crop
        crop_size = min(h, w)
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        frames = frames[:, top : top + crop_size, left : left + crop_size, :]
    else:
        x1, y1, x2, y2 = roi
        frames = frames[:, y1:y2, x1:x2, :]

    # Resize each frame
    out = np.zeros((t, target_size, target_size, c), dtype=np.float32)
    for i in range(t):
        out[i] = cv2.resize(frames[i], (target_size, target_size))

    return out


def prepare_i3d_input(frames: np.ndarray) -> torch.Tensor:
    """Normalize frames and convert to I3D input format (B, C, T, H, W)."""
    # Normalize: (img/255)*2 - 1
    frames = (frames / 255.0) * 2.0 - 1.0
    # (T, H, W, C) -> (C, T, H, W)
    tensor = torch.from_numpy(frames.transpose(3, 0, 1, 2)).float()
    tensor = tensor.unsqueeze(0)  # (1, C, T, H, W)
    return tensor


def predict_sign(
    i3d,
    video_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run I3D and return (probs, top5_indices)."""
    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        logits = i3d(video_tensor, pretrained=False)
        # logits: (B, C, T) -> pool over time
        logits = logits.mean(dim=2)
        probs = torch.softmax(logits, dim=1)
        top5_probs, top5_indices = probs[0].topk(5)

    return top5_probs.cpu(), top5_indices.cpu()


def run_pipeline(
    video_path: Path,
    yolov5_model,
    i3d_model,
    asl_classes: dict[int, str],
    device: torch.device,
) -> list[tuple[str, float]]:
    """
    Run full YOLOv5 + I3D pipeline on a video.
    Returns list of (sign_name, probability) for top-5 predictions.
    """
    # 1. Read video frames
    frames = read_video_frames(video_path)

    # 2. YOLOv5 ROI on middle frame (frames are RGB float 0-255)
    mid_idx = len(frames) // 2
    frame_uint8 = np.clip(frames[mid_idx], 0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    roi = get_roi_from_yolov5(
        yolov5_model, frame_bgr, (YOLOV5_IMG_SIZE, YOLOV5_IMG_SIZE), device
    )

    # 3. Crop and resize for I3D
    frames = crop_frames_to_roi(frames, roi)
    video_tensor = prepare_i3d_input(frames)

    # 4. I3D inference
    top5_probs, top5_indices = predict_sign(i3d_model, video_tensor, device)

    # 5. Map to sign names
    results = []
    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        sign_name = asl_classes.get(idx, f"class_{idx}")
        results.append((sign_name, prob))

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 + I3D ASL Detection Pipeline")
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=Path("videos/25871.mp4"),
        help="Path to input video",
    )
    parser.add_argument(
        "--yolov5-weights",
        type=Path,
        default=YOLOV5_WEIGHTS,
        help="YOLOv5 weights path",
    )
    parser.add_argument(
        "--i3d-pretrained",
        type=Path,
        default=I3D_PRETRAINED,
        help="I3D pretrained (Kinetics) weights",
    )
    parser.add_argument(
        "--i3d-asl-weights",
        type=Path,
        default=I3D_ASL_WEIGHTS,
        help="I3D ASL-fine-tuned weights",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=ASL_CLASS_LIST,
        help="ASL class list (wlasl_class_list.txt)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=I3D_NUM_CLASSES,
        help="Number of ASL sign classes",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading YOLOv5...")
    if not args.yolov5_weights.exists():
        raise FileNotFoundError(
            f"YOLOv5 weights not found: {args.yolov5_weights}. "
            "Upload yolov5s.pt to weights/"
        )
    yolov5_model = load_yolov5(args.yolov5_weights, device)

    print("Loading I3D...")
    if not args.i3d_pretrained.exists():
        raise FileNotFoundError(
            f"I3D pretrained weights not found: {args.i3d_pretrained}"
        )
    i3d_model = load_i3d(
        args.i3d_pretrained,
        args.i3d_asl_weights,
        args.num_classes,
        device,
    )

    print("Loading ASL classes...")
    asl_classes = load_asl_classes(args.classes)

    # Run pipeline
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    print(f"\nProcessing: {args.video}")
    results = run_pipeline(
        args.video, yolov5_model, i3d_model, asl_classes, device
    )

    print("\nTop-5 ASL predictions:")
    for i, (sign, prob) in enumerate(results, 1):
        print(f"  {i}. {sign}: {prob:.2%}")


if __name__ == "__main__":
    main()
