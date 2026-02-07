#!/bin/bash
# ASL Project Setup
# Installs dependencies and clones YOLOv5. Run from project root.
# You provide: dataset, weights (yolov5s.pt, rgb_imagenet.pt, nslt_2000_best.pt)

set -e
cd "$(dirname "$0")"

echo "Installing Python dependencies..."
python -m pip install -r requirements.txt

YOLOV5_DIR="yolov5_repo"
if [ ! -d "$YOLOV5_DIR" ]; then
    echo "Cloning YOLOv5 into yolov5_repo/..."
    git clone https://github.com/ultralytics/yolov5.git "$YOLOV5_DIR"
    python -m pip install -r "$YOLOV5_DIR/requirements.txt"
else
    echo "yolov5_repo/ already exists. Skipping clone."
fi

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. Place weights in weights/: yolov5s.pt, rgb_imagenet.pt (nslt_2000_best.pt optional)"
echo "  2. Place WLASL videos as {vid_id}.mp4 in your video root"
echo "  3. Run: python yolov5.py  (YOLOv5 demo)"
echo "  4. Run: python asl_detection_pipeline.py path/to/video.mp4  (full pipeline)"
