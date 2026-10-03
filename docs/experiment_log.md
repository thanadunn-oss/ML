# Experiment Log — คนที่ 3 (โมเดล การทดลอง และการประเมิน)

## ชุดทดลองหลัก: HAM10000 (4 ต.ค. 2569)

ข้อมูล HAM10000 10,015 ภาพ 7 คลาส แบ่งตาม lesion_id (train 7,144 / val 1,436 / test 1,435), seed 42
เทรนบน RTX 4050 Laptop (CUDA, mixed precision) · preprocessing v1 · label_schema ham10000-v1

| รอบ | run_id | arch | การเปลี่ยนแปลง | epochs | best val Macro F1 (epoch) | melanoma R / P (val) | CPU p95 | gate |
|---|---|---|---|---|---|---|---|---|
| 1 | 20261004-004133-resnet18 | resnet18 | baseline | 15 | 0.7236 (12) | 0.57 / 0.59 | 34 ms | ผ่าน |
| 2 | 20261004-004918-efficientnet_b0 | efficientnet_b0 | เปลี่ยนสถาปัตยกรรม | 15 | 0.7450 (10) | 0.58 / 0.68 | 38 ms | ผ่าน |
| 3 | 20261004-010030-efficientnet_b0 | efficientnet_b0 | + weighted loss | 20 | **0.7550 (18)** | **0.64** / 0.56 | 40 ms | ผ่าน |

**เลือก:** รอบ 3 — ทุกรอบผ่าน gate จึงตัดสินด้วย optimizing metric (val Macro F1) · test ครั้งเดียว = **0.698**

### F1 รายคลาสบน val

| คลาส | รอบ 1 | รอบ 2 | รอบ 3 |
|---|---|---|---|
| nevus | 0.92 | 0.93 | 0.92 |
| melanoma | 0.58 | 0.63 | 0.60 |
| keratosis | 0.70 | 0.73 | 0.69 |
| bcc | 0.77 | 0.74 | 0.78 |
| akiec | 0.60 | 0.66 | 0.64 |
| dermatofibroma | 0.73 | 0.60 | 0.78 |
| vascular | 0.76 | 0.93 | 0.89 |

### ข้อค้นพบ (val)

1. **weighted loss เปลี่ยนสมดุลตามที่คาด:** melanoma recall 0.58 → 0.64 และ melanoma ที่หลุดเป็น nevus 55 → 33 ภาพ แต่ nevus ถูกทายเป็น melanoma 29 → 64 ภาพ — สำหรับระบบที่มีคนตรวจทาน เตือนเกินยอมรับได้มากกว่าปล่อย melanoma หลุด
2. คลาสเล็กได้ประโยชน์จาก weighted loss ชัดเจน: dermatofibroma recall 0.53 → 0.82, akiec 0.65 → 0.73
3. **รอบ 2 กับ 3 ต่างกันเพียง 0.010** — ใกล้กันมาก จุดต่างที่มีความหมายคือ melanoma recall
4. ทุกรอบ overfit หลัง ~epoch 5–10 (train loss ลดต่อ, val แกว่ง) → การเลือก best epoch จาก val จำเป็น
5. ข้อผิดพลาดหลักยังเป็น melanoma ↔ nevus เหมือนชุด ISIC 2024 → เป็นความยากของโจทย์ ไม่ใช่เพราะข้อมูลน้อยอย่างเดียว

---

## ชุดทดลองแรก: ISIC 2024 (3 ต.ค. 2569) — เลิกใช้ เพราะข้อมูลไม่ถึงเกณฑ์

ISIC 2024 มี 401,059 ภาพ ทุกภาพมี label มะเร็ง/ไม่ใช่ แต่ label ชนิดโรค (iddx_3) มีเพียง 1,065 ภาพ → ใช้ได้ 963 ภาพ 5 คลาส ต่ำกว่าเกณฑ์ 10,000 ภาพที่กลุ่มกำหนด จึงเปลี่ยนเป็น HAM10000
ผลด้านล่างเก็บไว้เป็นหลักฐานการตัดสินใจ (best val Macro F1 สูงสุด 0.47, test 0.51)

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

## รันซ้ำบน Colab (ชุดที่ 2) — 3 ต.ค. 2569

รันด้วย train.py เวอร์ชันแรก (ยังพิมพ์ test) บน **CPU** (Colab ไม่ได้จัด GPU ให้, ~180–210 วินาที/epoch) ค่าอื่นเหมือนชุดแรกทุกอย่าง (seed 42)
ไฟล์โมเดลครบทั้ง 3 รอบ (runs_rerun.zip) — ผลด้านล่างคำนวณซ้ำบน val จาก model.pt ได้ค่าตรงกับ metrics.json

| รอบ | run_id | arch | การเปลี่ยนแปลง | best val Macro F1 | params | ขนาดไฟล์ | CPU latency/ภาพ |
|---|---|---|---|---|---|---|---|
| 1 | 20261003-113735-resnet18 | resnet18 | baseline | 0.4218 | 11.2M | 44.8 MB | ~30 ms |
| 2 | 20261003-123032-efficientnet_b0 | efficientnet_b0 | เปลี่ยนสถาปัตยกรรม | 0.4080 | 4.0M | 16.3 MB | ~21 ms |
| 3 | 20261003-132142-efficientnet_b0 | efficientnet_b0 | + weighted loss, 25 epochs | **0.4682** | 4.0M | 16.3 MB | ~25 ms |

latency วัดบน CPU 2 cores ภาพเดียว หลัง warm-up เฉลี่ย 30 ครั้ง (ยังไม่รวม preprocessing/HTTP)

### ความแกว่งระหว่างชุดที่ 1 (GPU) กับชุดที่ 2 (CPU) ที่ seed เดียวกัน

| รอบ | val ชุด 1 | val ชุด 2 | ต่าง |
|---|---|---|---|
| 1 resnet18 | 0.414 | 0.422 | +0.008 |
| 2 efficientnet_b0 | 0.389 | 0.408 | +0.019 |
| 3 efficientnet_b0 + wl | 0.417 | 0.468 | +0.051 |

ผลแกว่งได้ถึง ~0.05 เมื่อเปลี่ยนฮาร์ดแวร์ → ต้องตีความความต่างเล็กๆ ระหว่างรอบด้วยความระมัดระวัง

### Val confusion matrix (rows = จริง, cols = ทาย; nevus, bcc, melanoma, keratosis, scc)

รอบ 1 resnet18
```
nevus      [40, 5, 13, 8, 0]
bcc        [2, 14, 1, 5, 2]
melanoma   [6, 4, 10, 4, 0]
keratosis  [5, 3, 1, 7, 3]
scc        [1, 5, 1, 2, 2]
```
รอบ 2 efficientnet_b0
```
nevus      [59, 3, 4, 0, 0]
bcc        [5, 14, 1, 1, 3]
melanoma   [12, 5, 4, 2, 1]
keratosis  [7, 3, 3, 3, 3]
scc        [1, 3, 0, 4, 3]
```
รอบ 3 efficientnet_b0 + weighted loss
```
nevus      [45, 0, 17, 2, 2]
bcc        [2, 8, 7, 4, 3]
melanoma   [7, 1, 10, 5, 1]
keratosis  [3, 1, 4, 9, 2]
scc        [1, 1, 0, 4, 5]
```

ข้อสังเกตจาก val:
- รอบ 2 (ไม่ถ่วงน้ำหนัก) เอนไปทาง nevus มาก: melanoma ถูกทายเป็น nevus 12/24, melanoma recall 0.17
- รอบ 3 (weighted loss) melanoma recall ขึ้นเป็น 0.42 และ keratosis/scc ดีขึ้น แต่แลกกับ nevus ถูกทายเป็น melanoma 17/66 (false alarm) และ bcc recall ลดเหลือ 0.33
- keratosis ↔ scc ยังสับสนกันทุกรอบ สอดคล้องกับประเด็นการรวม actinic keratosis

### การตัดสินใจ (ชุดที่ 2)

ตามเกณฑ์ best val Macro F1: **รอบ 3 (20261003-132142-efficientnet_b0)** — สูงกว่ารอบ 1 อยู่ 0.046 และยังเล็กกว่า (4.0M vs 11.2M params) และเร็วกว่าบน CPU
เสนอเป็น **model v2** แทน v1 (ResNet18 ชุดแรก) ผ่าน Registry ของคนที่ 4

## ข้อค้นพบ (Error analysis เบื้องต้น — จากชุดที่ 1)

1. melanoma ถูกทายเป็น nevus 11/23 ภาพในรอบ 1 — เป็นรอยโรคเซลล์เม็ดสีทั้งคู่ และ nevus เป็นคลาสใหญ่สุด
2. weighted loss เพิ่ม recall ของ melanoma (0.17 → 0.30) และ scc (0.36 → 0.45) แต่ nevus recall ลด (0.81 → 0.64)
3. keratosis กับ scc แยกกันไม่ออก — น่าจะเพราะรวม actinic keratosis (รอยโรคก่อนมะเร็ง SCC) ไว้ใน keratosis
4. ทุกรอบ val Macro F1 ตันที่ประมาณ 0.41 แม้ train loss ลดต่อ → ข้อจำกัดหลักคือข้อมูล (จำนวนน้อย, ภาพ 100–215 px, การจัดกลุ่มคลาส)

## ข้อจำกัด

- แบ่งชุดระดับภาพ เพราะไฟล์ที่ได้ไม่มี patient_id → อาจมีผู้ป่วยคนเดียวกันข้ามชุด
- val/test เล็ก (คลาส scc มี 11 ภาพ) → ค่ารายคลาสแกว่งมาก
- ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์
