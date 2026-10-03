# รันจากโฟลเดอร์รากของ repo หลัง activate .venv แล้ว:  .\run_experiments.ps1
# หยุดทันทีถ้าขั้นไหนพัง
$ErrorActionPreference = "Stop"

Write-Host "== 1/5 เตรียมข้อมูล HAM10000 ==" -ForegroundColor Cyan
python src/data/prepare_ham10000.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== 2/5 รอบ 1: ResNet18 baseline ==" -ForegroundColor Cyan
python src/models/train.py --arch resnet18 --epochs 15
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== 3/5 รอบ 2: EfficientNet-B0 ==" -ForegroundColor Cyan
python src/models/train.py --arch efficientnet_b0 --epochs 15
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== 4/5 รอบ 3: EfficientNet-B0 + weighted loss ==" -ForegroundColor Cyan
python src/models/train.py --arch efficientnet_b0 --weighted-loss --epochs 20
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== 5/5 ตรวจ gate และเลือกโมเดล (ยังไม่แตะ test) ==" -ForegroundColor Cyan
python src/models/select_model.py
Write-Host "`nเสร็จ — ส่งผลบนจอ + โฟลเดอร์ runs/ ให้เซียดู ก่อนรัน --final-eval" -ForegroundColor Green
