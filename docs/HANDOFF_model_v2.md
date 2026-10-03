# ส่งมอบโมเดล v2 — จากคนที่ 3

| รายการ | ค่า |
|---|---|
| run_id | 20261003-132142-efficientnet_b0 |
| arch | efficientnet_b0 (torchvision) + weighted CrossEntropy, 25 epochs |
| **best val Macro F1** | **0.4682** ← ใช้ค่านี้เป็น gate (ห้ามใช้ test_macro_f1 ใน metrics.json) |
| params / ขนาดไฟล์ | 4.0M / 16.3 MB |
| CPU latency | ~25 ms/ภาพ (ไม่รวม preprocessing/HTTP) |
| label_schema_version | v1-draft |
| preprocessing_version | v1 (Resize 224, Normalize ImageNet) |
| เทรนบน | Colab CPU, seed 42 |
| แทนที่ | v1 = 20261003-104203-resnet18 (val 0.4142) |

## ไฟล์

- `model.pt` — state_dict ของ efficientnet_b0 หัว 5 คลาส
- `serving_config.json` — ลำดับคลาส ขนาดภาพ mean/std arch (API ต้องอ่านจากไฟล์นี้)
- `class_mapping.json`, `metrics.json`

## สำหรับคนที่ 4 (MLflow)

- params: arch=efficientnet_b0, epochs=25, lr=1e-4, batch_size=32, weighted_loss=True, seed=42, img_size=224, device=cpu
- metric สำหรับ gate: best_val_macro_f1 = 0.4682
- artifacts: ทั้ง 4 ไฟล์ข้างบน
- เสนอเป็นรุ่นถัดจาก v1 → เหมาะใช้สาธิตการเปลี่ยนรุ่น/ย้อนรุ่น (v1 ยังเก็บไว้)

## สำหรับคนที่ 5 (API) — โหลดโมเดล

```python
import json, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

cfg = json.load(open("serving_config.json"))
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(cfg["classes"]))  # หัวต่างจาก ResNet18 (fc)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

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

ถ้าจะรองรับทั้ง v1/v2 ให้เลือกวิธีสร้างโมเดลจาก `cfg["arch"]` (resnet18 → `fc`, efficientnet_b0 → `classifier[1]`)

## เกณฑ์ส่งทบทวน

ถ้า `max(probs) < 0.6` → ส่งคืน `needs_review: true` (เลือกจาก val: ส่งทบทวน ~35%)

ตรวจว่า API ตอบตรงกับโมเดล (T07): ใช้ `reference_predictions.json` — ส่ง 6 ภาพในไฟล์ (อยู่ใน `data/model_images/`) เข้า API แล้วเทียบ `expected_pred`, `needs_review` และความน่าจะเป็น (ต่างได้ไม่เกิน 0.001)

## ข้อควรระวัง

ต้นแบบเพื่อการศึกษา ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์ — melanoma recall บน val 0.42 และ nevus ถูกทายเป็น melanoma 17/66
