# Pipeline: Prefect DAG + MLflow Registry + Rollback

```
1-validate-data ─► 2-train ─► 3-gate ─► 4-log-and-register ─► 5-approve ─► 6-deploy-and-verify
   (ข้อมูลเสีย = หยุด)  (retry 1 ครั้ง)   (ไม่ผ่าน = rejected,     (champion/previous)   (วาง bundle ให้ API
                                         champion เดิมใช้ต่อ)                          + ตรวจโหลดได้)
```

## ติดตั้ง (ครั้งเดียว)

```powershell
pip install -r requirements-pipeline.txt
```

ใช้ `mlflow-skinny` (ไม่ต้องพึ่ง scipy ที่ Windows บล็อก) · MLflow เก็บใน `mlflow.db` (SQLite) ไม่ต้องเปิดเซิร์ฟเวอร์

## คำสั่ง

| งาน | คำสั่ง |
|---|---|
| รันครบทั้ง DAG ด้วยคำสั่งเดียว | `python src/pipelines/flow.py --arch efficientnet_b0 --weighted-loss --epochs 20` |
| ลงทะเบียน run ที่เทรนไว้แล้ว | `python src/pipelines/flow.py --from-run runs/<run_id>` |
| เทรนใหม่จากเหตุการณ์ Monitoring | `python src/pipelines/flow.py --weighted-loss --epochs 20 --reason "drift alert ..."` |
| ดูทะเบียนทุกรุ่น | `python src/pipelines/status.py` |
| ย้อนรุ่น | `python src/pipelines/rollback.py` (หรือ `--to-version N`) |
| เปิดหน้าเว็บ MLflow (ถ้าต้องการ) | `mlflow ui --backend-store-uri sqlite:///mlflow.db` (ต้องใช้ mlflow ตัวเต็ม) |

ส่งค่าให้ train.py ได้ด้วย `--train-args "--num-workers 0 --batch-size 16"`

## กติกา

- **Gate** ใช้ `src/models/gate.py` + `configs/gate_criteria.json` (optimizing = val Macro F1, Gate 1 = melanoma recall/precision, Gate 2 = CPU latency)
- **อนุมัติ** เมื่อผ่าน gate **และ** val Macro F1 ดีกว่า champion ปัจจุบัน
- ทุก run ลงทะเบียนเสมอ (ทั้งผ่าน/ไม่ผ่าน) — status: candidate / champion / archived / rejected / rolled_back
- **บันทึกครบ 6 รายการ:** พารามิเตอร์, ตัวชี้วัด (val), ไฟล์ผลลัพธ์ (bundle), code commit, data (split sha256 + label/preprocessing version), สภาพแวดล้อม (device)
- **หลังเปลี่ยนรุ่นหรือย้อนรุ่น ต้องเริ่ม API ใหม่** — การเปลี่ยน alias ไม่เปลี่ยนโมเดลที่โหลดค้างในหน่วยความจำ
- ไม่มีวงวน: Monitoring เรียก flow ใหม่ทั้งรอบ

## ผลทดสอบระหว่างพัฒนา (ข้อมูลจำลอง)

| สถานการณ์ | ผล |
|---|---|
| ข้อมูลไม่ถึง 10,000 ภาพ | หยุดที่ 1-validate-data (`DataError`) ไม่เทรน |
| เทรนล้มเหลว | retry 1 ครั้ง แล้วหยุดพร้อม error |
| ไม่ผ่าน gate | ลงทะเบียนเป็น `rejected` champion เดิมไม่เปลี่ยน |
| ผ่าน gate แต่ไม่ดีกว่า champion | ลงทะเบียนเป็น `candidate` ไม่ promote |
| ผ่าน gate และดีกว่า | promote เป็น champion, รุ่นเดิมเป็น previous, วางให้ API |
| rollback | champion ↔ previous สลับ และวาง bundle รุ่นที่ย้อนไปให้ API ทันที |

ครั้งแรกที่รัน Prefect อาจขึ้น `database is locked` หนึ่งครั้งตอนสร้างฐานข้อมูลของ Prefect เอง — flow ยังทำงานจนจบ
