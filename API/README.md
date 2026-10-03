# Skin Lesion API + User Interface

โปรเจกต์นี้ใช้ `artifacts/model-v2/` (EfficientNet-B0) เป็นโมเดลเริ่มต้น และให้ API กับหน้าจออัปโหลดภาพในตัว

## เริ่มใช้งาน

1. สร้าง virtual environment และติดตั้ง `pip install -r requirements.txt`
2. รัน `uvicorn app:app --reload`
3. เปิด `http://127.0.0.1:8000`

API: `GET /health`, `GET /ready`, `POST /predict`, `POST /feedback`, `GET /reviews` และเอกสารทดสอบที่ `/docs`

## เปลี่ยนเป็นโมเดลรุ่นใหม่

วาง `model.pt` และ `serving_config.json` ในโฟลเดอร์รุ่นใหม่ พร้อม `class_mapping.json` ที่เรียง classes เหมือนกัน แล้วตั้งค่า:

```powershell
$env:MODEL_DIR = 'C:\path\to\new-model'
$env:CLASS_MAPPING_PATH = 'C:\path\to\class_mapping.json'
uvicorn app:app
```

ตั้งค่า `REVIEW_THRESHOLD` ได้ (ค่าเริ่มต้น `0.70`) และโมเดล baseline จะส่งผล melanoma เข้าตรวจทานเสมอผ่าน `ALWAYS_REVIEW_CLASSES=melanoma`.
