# ส่งมอบโมเดลรุ่นแรก (model v1) — จากคนที่ 3

## ไฟล์ที่ส่ง

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `runs/20261003-104203-resnet18/model.pt` | น้ำหนักโมเดล (PyTorch state_dict ของ torchvision resnet18, หัว 5 คลาส) |
| `runs/20261003-104203-resnet18/serving_config.json` | ลำดับคลาส ขนาดภาพ mean/std arch — API ต้องใช้ไฟล์นี้ |
| `runs/20261003-104203-resnet18/metrics.json` | ผลประเมินของ run นี้ |
| `class_mapping.json` | mapping iddx_3 → คลาส (label_schema_version v1-draft) |
| `splits.csv` | การแบ่งชุด seed 42 |
| `train.py` | โค้ดฝึก/ประเมิน และ preprocessing (`build_transforms`) |
| `experiment_log.md` | ผล 3 รอบ เหตุผลการเลือก และข้อจำกัด |

## สำหรับคนที่ 4 (Tracking / Registry)

ค่าที่ควรบันทึกใน MLflow สำหรับ run นี้:

- params: arch=resnet18, epochs=15, lr=1e-4, batch_size=32, weighted_loss=False, seed=42, img_size=224
- versions: label_schema_version=v1-draft, preprocessing_version=v1, split_version=splits.csv seed 42
- metric ที่ใช้ตัดสิน: **best_val_macro_f1 = 0.4142** (ใช้ค่านี้เป็น gate ห้ามใช้ test_macro_f1)
- artifacts: model.pt, serving_config.json, class_mapping.json, metrics.json
- code version: commit ของ train.py ใน repo กลุ่ม
- สถานะที่เสนอ: candidate รุ่นแรก — รอบ 3 (EfficientNet-B0 + weighted loss) จะเทรนใหม่แล้วเสนอเป็นรุ่นถัดไป

รอบ 2–3 ไม่มีไฟล์โมเดล (สูญหายกับ Colab) แต่บันทึกผลเป็น run ที่ไม่มี artifact ได้จาก experiment_log.md

## สำหรับคนที่ 5 (API) — วิธีโหลดและทำนาย

```python
import json, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

cfg = json.load(open("serving_config.json"))
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(cfg["classes"]))
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()  # สำคัญ: ปิด dropout/BN แบบเทรน และห้ามใช้ augmentation ตอนให้บริการ

# ต้องตรงกับ eval_tf ใน train.py ทุกขั้น (ป้องกัน training-serving skew)
tf = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(cfg["mean"], cfg["std"]),
])

img = Image.open("ISIC_xxx.jpg").convert("RGB")
with torch.no_grad():
    probs = torch.softmax(model(tf(img).unsqueeze(0)), dim=1)[0]
result = {c: float(p) for c, p in zip(cfg["classes"], probs)}
```

- ลำดับคลาสต้องอ่านจาก `serving_config.json` ห้ามเขียนลำดับเองในโค้ด
- ยังไม่มีเกณฑ์ส่งทบทวน (threshold) — คนที่ 3 จะกำหนดจาก val และส่งตามมา
- ภาพตัวอย่างพร้อมผลทำนายอ้างอิงสำหรับตรวจ API (T07) จะส่งตามมา

## ข้อควรระวัง

- เป็นต้นแบบเพื่อการศึกษา ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์
- val Macro F1 ประมาณ 0.41 — melanoma recall ต่ำ (มักถูกทายเป็น nevus)
