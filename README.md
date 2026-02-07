# American Sign Language Detection

ASL sign recognition combining **YOLOv5** for spatial person/gesture detection (ROI) and **I3D** (Inflated 3D ConvNet) for temporal sign classification on the WLASL dataset.

## Project Structure

| File | Description |
|------|-------------|
| `yolov5.py` | Standalone YOLOv5 object detection demo |
| `pytorch_i3d.py` | I3D model definition |
| `asl_detection_pipeline.py` | Combined YOLOv5 + I3D pipeline for video sign recognition |
| `train_i3d.py` | I3D training on WLASL |
| `nslt_dataset_all.py` | WLASL dataset loader |
| `videotransforms.py` | Video augmentation transforms |
| `nslt_100.json`, `nslt_300.json`, `nslt_1000.json`, `nslt_2000.json` | Train/val/test splits |
| `wlasl_class_list.txt` | ASL sign class names (2000 classes) |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- OpenCV

## Setup

### 1. Run setup script

Installs dependencies and clones YOLOv5 into `yolov5_repo/`.

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux / macOS:**
```bash
bash setup.sh
```

### 2. Weights

Place weights in the `weights/` folder:

| Weight | Description | Source |
|--------|-------------|--------|
| `yolov5s.pt` | YOLOv5 pretrained | Auto-downloads on first run of `yolov5.py`; or [YOLOv5 releases](https://github.com/ultralytics/yolov5/releases) |
| `rgb_imagenet.pt` | I3D pretrained on Kinetics | Obtain from I3D/Kinetics checkpoints; required for pipeline and training |
| `nslt_2000_best.pt` | I3D fine-tuned on WLASL | Optional; train with `train_i3d.py` or obtain separately |

### 3. Dataset (for training and full pipeline)

The dataset is **not included** in this repo. To run training or the full pipeline:

1. Download the [WLASL dataset](https://github.com/dxli94/WLASL) (videos and metadata)
2. Place videos as `{vid_id}.mp4` in your video root (e.g. `./videos/` or `./`)
3. `WLASL_v0.3.json` comes with the WLASL dataset; place it in the project root if needed

The included `nslt_*.json` files define train/val/test splits for 100, 300, 1000, and 2000 sign classes.

## Usage

### YOLOv5 standalone detection

```bash
python yolov5.py
```

Uses `Project Milestone - ASL Detection/Images/test1.jpg` and writes `results/result.jpg`. YOLOv5 weights auto-download on first run.

### Combined YOLOv5 + I3D pipeline

```bash
python asl_detection_pipeline.py path/to/video.mp4
```

Requires `weights/yolov5s.pt` and `weights/rgb_imagenet.pt`. Optional args: `--yolov5-weights`, `--i3d-pretrained`, `--i3d-asl-weights`, `--classes`, `--num-classes`.

### Training I3D

```python
from train_i3d import run

run(
    mode='rgb',
    root='./',                    # Video root (videos named {vid_id}.mp4)
    train_split='./nslt_2000.json',
    save_model='checkpoints/',
    weights='weights/rgb_imagenet.pt',
)
```

Requires WLASL videos and `weights/rgb_imagenet.pt`.

## License

MIT
