# ส่วนโมเดล (คนที่ 3) — วิธีรันบนเครื่อง

รันทุกคำสั่งจาก **โฟลเดอร์รากของ repo**

## 1. ติดตั้ง

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. ภาพ

ภาพ 1,068 ภาพ (~4 MB) อยู่ใน repo แล้วที่ `data/model_images/` — `git clone` แล้วใช้ได้ทันที

ที่มาของภาพ: ISIC 2024 Challenge training images คัดเฉพาะ isic_id ใน `data/needed_image_ids.csv` (สัญญาอนุญาต CC-BY ตาม `attribution` ใน metadata)

ตรวจว่าได้ข้อมูลชุดเดียวกับในรายงาน:

```bash
python -c "import os;print(len(os.listdir('data/model_images')))"   # ต้องได้ 1068
# Windows PowerShell:  Get-FileHash data/splits.csv -Algorithm SHA256
# Mac/Linux:           sha256sum data/splits.csv
```

| ไฟล์ | SHA256 |
|---|---|
| data/splits.csv | `34b4852299efee04c5dd4b176677761af9cb5a904c72ee046e98001d392705f8` |
| configs/class_mapping.json | `9add8cab7877d660e48a5de19f4b8188658eb3d6c33e897db59e1869306ebdd2` |

ถ้า checksum ไม่ตรง แปลว่าข้อมูลคนละชุดกัน ผลจะเทียบกับรายงานไม่ได้

## 3. เทรนและประเมิน

```bash
python src/models/train.py --arch resnet18 --epochs 15
python src/models/train.py --arch efficientnet_b0 --epochs 15
python src/models/train.py --arch efficientnet_b0 --weighted-loss --epochs 25
```

ตอนเทรนแสดงเฉพาะผล val ผลแต่ละ run อยู่ใน `runs/<run_id>/`

ประเมิน test ครั้งเดียวกับโมเดลที่เลือกจาก val แล้วเท่านั้น:

```bash
python src/models/train.py --final-eval runs/<run_id>
```

บนเครื่องที่ไม่มี GPU จะช้า (~3 นาที/epoch) แนะนำเทรนบน Colab/Kaggle แล้วใช้เครื่องตัวเองสำหรับทดสอบโหลดโมเดล

## 4. สร้าง class mapping และ split ใหม่ (ปกติไม่ต้องรัน — ไฟล์อยู่ใน repo แล้ว)

```bash
python src/data/prepare_data.py
```

ผลต้องได้ `data/splits.csv` เหมือนเดิมทุกครั้ง (seed 42) — ตรวจด้วย checksum ด้านบน

## 5. เกณฑ์ว่า "ทำซ้ำได้" (reproducibility)

ใช้ seed 42 เหมือนกัน แต่ผลจะ **ไม่เท่ากันเป๊ะทุกตำแหน่งทศนิยม** เพราะ GPU (cuDNN) มีการคำนวณบางส่วนที่ไม่ deterministic และต่างรุ่นฮาร์ดแวร์ก็ต่างกัน

จากการรันซ้ำจริงของกลุ่ม (ดู `docs/experiment_log.md`):

| arch | ครั้งที่ 1 val Macro F1 | ครั้งที่ 2 val Macro F1 |
|---|---|---|
| resnet18 | 0.414 | 0.422 |
| efficientnet_b0 | 0.389 | 0.408 |

**เกณฑ์ที่ยอมรับ:** best val Macro F1 ต่างจากค่าในรายงานไม่เกิน **±0.03**

สิ่งที่ต้อง **ตรงกันทุกครั้ง** (ถ้าไม่ตรง = มีอะไรผิด):

- `data/splits.csv` และ `configs/class_mapping.json` (checksum)
- จำนวนภาพ train / val / test = 674 / 144 / 145
- ลำดับคลาสใน `serving_config.json` = nevus, bcc, melanoma, keratosis, scc

## 6. ใช้โมเดลที่เทรนแล้วโดยไม่ต้องเทรนเอง

ดาวน์โหลด `model.pt` + `serving_config.json` จาก MLflow Registry ของกลุ่ม (คนที่ 4) แล้วโหลดตามตัวอย่างใน `docs/HANDOFF_model_v1.md`

## 7. ด่านตรวจคุณภาพ (gate) ก่อนอนุมัติโมเดล

```bash
python src/models/gate.py --candidate runs/<new> --active runs/<current>
```

- เกณฑ์อยู่ใน `configs/gate_criteria.json` (val Macro F1 ≥ 0.40, ไม่แย่กว่ารุ่นที่ใช้อยู่, ทุกคลาส F1 > 0, ลำดับคลาส/preprocessing ตรงกับ API)
- exit code 0 = ผ่าน, 1 = ไม่ผ่าน → ใช้เป็นเงื่อนไขใน Prefect DAG ได้ตรงๆ หรือ `from gate import check_gate`
- ผลบันทึกใน `runs/<run_id>/gate_result.json` ทุกครั้ง (หลักฐาน T09) ตัวอย่างดู `docs/gate_evidence.md`

run ที่สร้างจาก train.py รุ่นแรกไม่มีผลรายคลาสของ val → ประเมินใหม่ก่อนเข้า gate (ไม่ต้องเทรน):

```bash
python src/models/train.py --eval-val runs/<run_id>
```
