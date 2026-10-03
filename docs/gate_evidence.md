# หลักฐานการทดสอบ Gate (T09) — HAM10000

เกณฑ์: `configs/gate_criteria.json` · optimizing metric แยกจาก gate · ใช้ค่า validation เท่านั้น

| ประเภท | เกณฑ์ |
|---|---|
| Optimizing | val Macro F1 สูงสุด (เลือกเฉพาะในกลุ่มที่ผ่าน gate) |
| Gate 1 | melanoma recall ≥ 0.50 และ precision ≥ 0.25 |
| Gate 2 | CPU latency p95 ≤ 200 ms (รวมเปิด+เตรียมภาพ) |

## ผลจริง (4 ต.ค. 2569, `python src/models/select_model.py`)

| run | val Macro F1 | Gate 1 (mel R / P) | Gate 2 (p95) | ผล |
|---|---|---|---|---|
| 20261004-004133-resnet18 | 0.7236 | 0.572 / 0.591 ✓ | 34.4 ms ✓ | ผ่าน |
| 20261004-004918-efficientnet_b0 | 0.7450 | 0.579 / 0.681 ✓ | 38.3 ms ✓ | ผ่าน |
| 20261004-010030-efficientnet_b0 | **0.7550** | 0.642 / 0.564 ✓ | 39.9 ms ✓ | ผ่าน → **ถูกเลือก** |

## กรณีที่ต้องถูกปฏิเสธ (ทดสอบระหว่างพัฒนา)

| กรณี | ผล | เหตุผล |
|---|---|---|
| โมเดลไม่ได้ pretrained เทรน 1 epoch — ทายทุกภาพเป็น melanoma | **REJECTED** | melanoma recall 1.000 แต่ precision 0.138 < 0.25 |
| ไม่มีโมเดลใดผ่าน gate | `select_model.py` ตอบ "ไม่มีโมเดลผ่าน gate — ห้ามนำขึ้นใช้งาน" | — |

กรณีแรกคือเหตุผลที่ Gate 1 ต้องมี precision: ถ้าใช้ recall อย่างเดียว โมเดลที่ไร้ประโยชน์จะผ่านได้

ผลละเอียดของแต่ละ run อยู่ใน `runs/<run_id>/gate_result.json`
