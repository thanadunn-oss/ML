# Skin Lesion API + User Interface

ให้บริการโมเดล **v3 (HAM10000, EfficientNet-B0, 7 คลาส)** ผ่าน FastAPI พร้อมหน้าจออัปโหลดภาพในตัว

## เริ่มใช้งาน

1. ติดตั้ง: `pip install -r requirements.txt`
2. **วาง `model.pt`** ใน `artifacts/model-v3/` — ไฟล์ไม่อยู่ใน Git (ใหญ่ ~16 MB) เอามาจาก `model_v3_ham10000.zip` ที่คนที่ 3 ส่งให้ หรือจาก MLflow Registry (run_id ใน `artifacts/model-v3/RUN_ID.txt`)
3. รัน `uvicorn app:app --reload` แล้วเปิด `http://127.0.0.1:8000`

ตรวจว่าพร้อม: `GET /ready` ต้องได้ `run_id = 20261004-010030-efficientnet_b0` และ 7 คลาส

API: `GET /health`, `GET /ready`, `POST /predict`, `POST /feedback`, `GET /reviews`, เอกสารที่ `/docs`

## ตรวจว่า API ตอบตรงกับโมเดล (T07)

```bash
python tests/check_reference.py --ref ../docs/reference_predictions.json
```

ส่งภาพอ้างอิงจาก `docs/reference_images/` เข้า `/predict` แล้วเทียบคลาสและความน่าจะเป็นกับผลจากโมเดลโดยตรง (ต่างได้ไม่เกิน 0.001) และทดสอบไฟล์เสีย (400) / ชนิดไฟล์ผิด (415)

## เกณฑ์ส่งทบทวน

- `REVIEW_THRESHOLD` ค่าเริ่มต้น **0.80** — ความน่าจะเป็นสูงสุดต่ำกว่านี้ → `review_required: true` (~20% ของภาพบน validation, เลือกโดยคนที่ 3 ดู `docs/MODEL_CARD.md`)
- `ALWAYS_REVIEW_CLASSES` ค่าเริ่มต้น **ปิด** — กฎ "ทายว่า melanoma → ทบทวน" ไม่ลด melanoma ที่หลุด (ที่หลุดคือถูกทายเป็นคลาสอื่น) แต่เพิ่มภาระตรวจ

## เปลี่ยนเป็นโมเดลรุ่นใหม่ / ย้อนรุ่น

วาง `model.pt`, `serving_config.json`, `class_mapping.json` (ลำดับคลาสต้องตรงกัน) และ `RUN_ID.txt` ในโฟลเดอร์รุ่นใหม่ แล้ว:

```powershell
$env:MODEL_DIR = 'C:\path\to\model-folder'
uvicorn app:app
```

ต้อง **เริ่มบริการใหม่** หลังเปลี่ยนรุ่น — การเปลี่ยน alias ใน Registry ไม่ทำให้โมเดลที่โหลดค้างในหน่วยความจำเปลี่ยนตาม

ต้นแบบเพื่อการศึกษา ใช้กับภาพ dermoscopy เท่านั้น ไม่ใช่เครื่องมือวินิจฉัย · ข้อมูล HAM10000 เป็น CC BY-NC (ห้ามใช้เชิงพาณิชย์)
