# ASL Project Setup
# Installs dependencies and clones YOLOv5. Run from project root.
# You provide: dataset, weights (yolov5s.pt, rgb_imagenet.pt, nslt_2000_best.pt)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

$Yolov5Dir = Join-Path $ProjectRoot "yolov5_repo"
if (-not (Test-Path $Yolov5Dir)) {
    Write-Host "Cloning YOLOv5 into yolov5_repo/..." -ForegroundColor Cyan
    git clone https://github.com/ultralytics/yolov5.git $Yolov5Dir
    python -m pip install -r (Join-Path $Yolov5Dir "requirements.txt")
} else {
    Write-Host "yolov5_repo/ already exists. Skipping clone." -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Place weights in weights/: yolov5s.pt, rgb_imagenet.pt (nslt_2000_best.pt optional)"
Write-Host "  2. Place WLASL videos as {vid_id}.mp4 in your video root"
Write-Host "  3. Run: python yolov5.py  (YOLOv5 demo)"
Write-Host "  4. Run: python asl_detection_pipeline.py path/to/video.mp4  (full pipeline)"
