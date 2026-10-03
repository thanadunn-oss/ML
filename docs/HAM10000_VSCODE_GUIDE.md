# เทรนโมเดลด้วย HAM10000 บน VS Code (Windows + GPU)

ทดสอบเครื่อง: RTX 4050 Laptop 6 GB · ทุกคำสั่งรันใน Terminal ของ VS Code ที่โฟลเดอร์รากของ repo

## 1. ติดตั้ง (ครั้งเดียว)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pandas pillow numpy
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

ต้องได้ `True` และชื่อการ์ดจอ — ถ้าได้ `False` แปลว่าลง PyTorch รุ่น CPU ให้ดูคำสั่งที่ถูกต้องจาก https://pytorch.org/get-started/locally/

ถ้า activate แล้วขึ้น "running scripts is disabled": `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## 2. ดาวน์โหลดข้อมูล

1. https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 → Download (ต้องล็อกอิน)
2. แตก zip ไว้ที่ `data/ham10000/` ให้ได้ประมาณนี้ (ซ้อนโฟลเดอร์ได้ สคริปต์ค้นหาเอง):

```
data/ham10000/HAM10000_metadata.csv
data/ham10000/HAM10000_images_part_1/*.jpg
data/ham10000/HAM10000_images_part_2/*.jpg
```

โฟลเดอร์นี้ถูก `.gitignore` กันไว้ (ใหญ่ ~2.5 GB) ไม่ขึ้น Git

## 3. รันทั้งหมดด้วยคำสั่งเดียว

```powershell
.\run_experiments.ps1
```

ทำ 5 ขั้น: เตรียมข้อมูล → ResNet18 → EfficientNet-B0 → EfficientNet-B0 + weighted loss → ตรวจ gate และเลือกโมเดล

หรือรันทีละขั้น:

```powershell
python src/data/prepare_ham10000.py
python src/models/train.py --arch resnet18 --epochs 15
python src/models/train.py --arch efficientnet_b0 --epochs 15
python src/models/train.py --arch efficientnet_b0 --weighted-loss --epochs 20
python src/models/select_model.py
```

ขั้นเตรียมข้อมูลต้องขึ้น: ใช้ ~10,015 ภาพ, `lesion รั่วข้ามชุด 0`

## 4. หลังเลือกโมเดลแล้วเท่านั้น — วัด test ครั้งเดียว

```powershell
python src/models/train.py --final-eval runs/<run_id ที่ select_model เลือก>
```

## ข้อมูลและการแบ่งชุด

- HAM10000: 10,015 ภาพ 7 คลาส — nevus, melanoma, keratosis (bkl), bcc, akiec, dermatofibroma, vascular
- แบ่ง ~71/14/14 ตาม **lesion_id** (รอยโรคเดียวกันอยู่ชุดเดียวกัน กันข้อมูลรั่ว) และรักษาสัดส่วนคลาส, seed 42

## เกณฑ์ (configs/gate_criteria.json)

| ประเภท | เกณฑ์ | ใช้ทำอะไร |
|---|---|---|
| **Optimizing** | val Macro F1 (สูงสุด) | เลือกโมเดลที่ดีที่สุด *ในกลุ่มที่ผ่าน gate* |
| **Gate 1** | melanoma recall (val) ≥ 0.50 และ precision ≥ 0.25 | ความปลอดภัย — คลาสอันตรายที่สุดต้องไม่หลุดมาก (precision กันการโกงด้วยการทายทุกภาพเป็น melanoma) |
| **Gate 2** | เวลาทำนาย CPU p95 ≤ 200 ms | ใช้งานจริงได้ ผู้ใช้ไม่ต้องรอนาน |
| Validity | ลำดับคลาส + preprocessing ตรงกับ API | กันผลผิดทั้งระบบ |

ตัวเลข 0.50 / 0.25 และ 200 ms เป็นค่าร่าง — ทบทวนกับติวหลังเห็นผลเทรนรอบแรก

## ปัญหาที่อาจเจอ

| อาการ | แก้ |
|---|---|
| `CUDA out of memory` | ปิดโปรแกรมที่ใช้ GPU หรือเพิ่ม `--batch-size 16` |
| ช้ามาก / GPU ใช้น้อย | ลด/เพิ่ม `--num-workers` (ค่าเริ่มต้น 4) |
| error เกี่ยวกับ DataLoader worker บน Windows | ใส่ `--num-workers 0` |
| ปิด mixed precision | `--no-amp` |

| `DLL load failed ... Application Control policy has blocked this file` | Windows Smart App Control บล็อก scipy — โค้ดชุดนี้ไม่ใช้ scikit-learn/scipy แล้ว ถ้ายังเจอให้ `pip uninstall scikit-learn scipy -y` |
