# ส่งมอบโมเดล v3 (HAM10000, 7 คลาส) — จากคนที่ 3

**แทนที่ v1/v2 ทั้งหมด** (v1/v2 เทรนด้วย ISIC 2024 5 คลาส ข้อมูลไม่ถึงเกณฑ์ เลิกใช้แล้ว)

| รายการ | ค่า |
|---|---|
| run_id | 20261004-010030-efficientnet_b0 |
| arch | efficientnet_b0 (torchvision) + weighted CrossEntropy, 20 epochs |
| classes (ลำดับนี้เท่านั้น) | nevus, melanoma, keratosis, bcc, akiec, dermatofibroma, vascular |
| **val Macro F1** (optimizing) | **0.7550** |
| test Macro F1 (วัดครั้งเดียว) | 0.6976 |
| Gate 1 melanoma recall / precision (val) | 0.642 / 0.564 ✓ |
| Gate 2 CPU p95 | 39.9 ms ✓ |
| label_schema_version / preprocessing_version | ham10000-v1 / v1 |

## ไฟล์ในชุด (model_v3_ham10000.zip)

- `model.pt` — state_dict ของ efficientnet_b0 หัว 7 คลาส (~16 MB)
- `serving_config.json` — ลำดับคลาส ขนาดภาพ mean/std arch (API ต้องอ่านจากไฟล์นี้)
- `class_mapping.json`, `metrics.json` (val), `test_metrics.json`, `gate_result.json`

## สำหรับฟง (Pipeline / MLflow / Registry)

- ลงทะเบียน 3 run ของ HAM10000 จาก `metrics.json` ของแต่ละ run (ทุกตัวผ่าน gate)
- **รุ่นที่ใช้งาน = run_id ข้างบน** · รุ่นก่อนหน้าสำหรับสาธิต rollback ใช้ 20261004-004918-efficientnet_b0 (val 0.7450)
- gate: `python src/models/gate.py --candidate runs/<new> --active runs/<current>` (exit 0 = ผ่าน)
- เลือกจากทุก run: `python src/models/select_model.py` → `runs/selection.json`
- metric ที่ใช้ตัดสินคือ `best_val_macro_f1` เท่านั้น ห้ามใช้ test

## สำหรับโฟน (API)

1. วาง `model.pt` + `serving_config.json` + `class_mapping.json` ใน `API/artifacts/model-v3/` แล้วตั้ง `MODEL_DIR`
2. **คลาสเปลี่ยนเป็น 7 คลาสและลำดับใหม่** — API อ่านจาก `serving_config.json` อยู่แล้วจึงไม่ต้องแก้โค้ด แต่ `class_mapping.json` ต้องเป็นไฟล์ใหม่นี้ (API ตรวจว่าลำดับตรงกัน)
3. ตรวจ T07 ด้วย `docs/reference_predictions.json` + `docs/reference_images/` — ความน่าจะเป็นต่างได้ไม่เกิน 0.001
4. เกณฑ์ส่งทบทวน: **REVIEW_THRESHOLD = 0.8** (คนที่ 3 เลือกจาก val: ส่งทบทวน ~20%) และ **ปิด ALWAYS_REVIEW_CLASSES** — ตารางเต็มใน `docs/review_threshold_sweep.json`

## ข้อควรระวัง

- ใช้กับภาพ dermoscopy เท่านั้น · ไม่ใช่เครื่องมือวินิจฉัย
- ข้อมูล CC BY-NC — ห้ามใช้เชิงพาณิชย์
