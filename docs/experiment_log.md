# Experiment Log — คนที่ 3 (โมเดล การทดลอง และการประเมิน)

บันทึกจากผลรันจริงบน Google Colab วันที่ 3 ต.ค. 2569

## ข้อมูลที่ใช้ร่วมกันทุกรอบ

| รายการ | ค่า |
|---|---|
| dataset | ISIC 2024 Challenge — เฉพาะภาพที่มี iddx_3 (1,068 ภาพ) |
| label_schema_version | v1-draft (5 คลาส: nevus, bcc, melanoma, keratosis, scc) — รอคนที่ 2 ยืนยัน |
| split_version | splits.csv, seed 42, 70/15/15 stratified ระดับภาพ (ไม่มี patient_id) |
| จำนวนภาพ train / val / test | 674 / 144 / 145 (รวม 963, ตัดออก 105) |
| preprocessing_version | v1 — Resize 224x224, Normalize ด้วยค่า ImageNet |
| augmentation (train เท่านั้น) | flip แนวนอน/แนวตั้ง, rotation 30°, ColorJitter 0.1 |
| optimizer | AdamW lr 1e-4, weight_decay 1e-4, batch 32 |
| pretrained | ImageNet (torchvision IMAGENET1K_V1) |
| เกณฑ์เลือกโมเดล | best val Macro F1 (กำหนดก่อนเริ่มทดลอง) |
| hardware | Colab T4 GPU |

## ผลแต่ละรอบ

| รอบ | run_id | arch | การเปลี่ยนแปลง | epochs | best val Macro F1 (epoch) | ไฟล์โมเดล |
|---|---|---|---|---|---|---|
| 1 | 20261003-104203-resnet18 | resnet18 | baseline | 15 | 0.4142 (ep 6) | มี (runs.zip) |
| 2 | 20261003-104759-efficientnet_b0 | efficientnet_b0 | เปลี่ยนสถาปัตยกรรม | 15 | 0.3894 (ep 13) | สูญหายกับ Colab session |
| 3 | 20261003-105501-efficientnet_b0 | efficientnet_b0 | + weighted CrossEntropy | 25 | 0.4168 (ep 8) | สูญหายกับ Colab session |

### ผลรายคลาสบน test (เขียนไว้เพื่อความโปร่งใส — ไม่ได้ใช้เลือกโมเดล)

รอบ 1–3 รันด้วย train.py เวอร์ชันแรกที่คำนวณ test อัตโนมัติ จึงเห็นค่า test แล้ว
หลังจากพบปัญหานี้ได้แก้ train.py ให้ประเมิน test เฉพาะผ่าน `--final-eval` ครั้งเดียว

| คลาส (n test) | รอบ 1 F1 (P/R) | รอบ 2 F1 (P/R) | รอบ 3 F1 (P/R) |
|---|---|---|---|
| nevus (67) | 0.81 (0.82/0.81) | 0.85 (0.82/0.88) | 0.71 (0.80/0.64) |
| bcc (25) | 0.54 (0.48/0.60) | 0.71 (0.65/0.80) | 0.62 (0.53/0.76) |
| melanoma (23) | 0.24 (0.40/0.17) | 0.42 (0.45/0.39) | 0.30 (0.29/0.30) |
| keratosis (19) | 0.24 (0.23/0.26) | 0.41 (0.47/0.37) | 0.24 (0.27/0.21) |
| scc (11) | 0.30 (0.25/0.36) | 0.44 (0.57/0.36) | 0.37 (0.31/0.45) |
| **Macro F1** | 0.426 | 0.568 | 0.447 |

Confusion matrix รอบ 1 บน test (rows = จริง, cols = ทาย; ลำดับ nevus, bcc, melanoma, keratosis, scc)

```
nevus      [54, 4, 3, 5, 1]
bcc        [0, 15, 1, 4, 5]
melanoma   [11, 2, 4, 6, 0]
keratosis  [1, 5, 2, 5, 6]
scc        [0, 5, 0, 2, 4]
```

## การตัดสินใจ

- ตามเกณฑ์ val รอบ 3 สูงสุด (0.417) แต่ต่างจากรอบ 1 (0.414) เพียง 0.003 ซึ่งอยู่ในระดับความแกว่งของ val ที่มี 144 ภาพ จึงถือว่าเสมอกัน
- ไฟล์โมเดลรอบ 2–3 สูญหาย จึง **ส่งรอบ 1 (ResNet18) เป็นโมเดลรุ่นแรกให้ Registry/API** และจะเทรนรอบ 3 ใหม่ด้วย train.py ตัวใหม่ เมื่อเลือกได้ด้วย val แล้วค่อยเปลี่ยนรุ่นผ่าน Registry

## ข้อค้นพบ (Error analysis เบื้องต้น)

1. melanoma ถูกทายเป็น nevus 11/23 ภาพในรอบ 1 — เป็นรอยโรคเซลล์เม็ดสีทั้งคู่ และ nevus เป็นคลาสใหญ่สุด
2. weighted loss เพิ่ม recall ของ melanoma (0.17 → 0.30) และ scc (0.36 → 0.45) แต่ nevus recall ลด (0.81 → 0.64)
3. keratosis กับ scc แยกกันไม่ออก — น่าจะเพราะรวม actinic keratosis (รอยโรคก่อนมะเร็ง SCC) ไว้ใน keratosis
4. ทุกรอบ val Macro F1 ตันที่ประมาณ 0.41 แม้ train loss ลดต่อ → ข้อจำกัดหลักคือข้อมูล (จำนวนน้อย, ภาพ 100–215 px, การจัดกลุ่มคลาส)

## ข้อจำกัด

- แบ่งชุดระดับภาพ เพราะไฟล์ที่ได้ไม่มี patient_id → อาจมีผู้ป่วยคนเดียวกันข้ามชุด
- val/test เล็ก (คลาส scc มี 11 ภาพ) → ค่ารายคลาสแกว่งมาก
- ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์
