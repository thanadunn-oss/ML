"""
src/data/prepare_data.py — (ร่างโดยคนที่ 3 ส่งให้คนที่ 2 เป็นเจ้าของ) สร้าง class mapping + แบ่งชุด train/val/test (ร่างสำหรับคนที่ 2 ตรวจ)

เหตุผลหลักของการออกแบบ:
- ใช้ iddx_3 เพราะเป็นระดับที่ "บอกชนิดรอยโรค" ได้จริง (iddx_1 มีแค่ Benign/Malignant)
- รวมชนิดย่อยเป็นกลุ่มใหญ่ เพราะหลายชนิดมีภาพไม่ถึง 20 ภาพ ซึ่งวัด F1 รายคลาสไม่ได้อย่างมีความหมาย
- ตัดกลุ่ม "Atypical / Indeterminate" ออก เพราะฉลากเองยังไม่แน่นอน -> ถ้าใส่เข้าไปจะสอนโมเดลด้วยคำตอบกำกวม
- ไม่มี patient_id ในไฟล์ที่ได้รับ และ lesion_id ไม่ซ้ำกันเลย -> แบ่งระดับภาพได้เท่านั้น (ต้องเขียนเป็นข้อจำกัด)
"""
import json
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # ล็อก seed เพื่อให้แบ่งชุดซ้ำได้เหมือนเดิมทุกครั้ง (เกณฑ์ T02)
LABEL_SCHEMA_VERSION = "v1-draft"

# mapping จาก iddx_3 -> คลาสเป้าหมาย; ตัวที่ไม่อยู่ใน dict = ตัดออก
IDDX3_TO_CLASS = {
    "Nevus": "nevus",
    "Basal cell carcinoma": "bcc",
    "Melanoma in situ": "melanoma",
    "Melanoma Invasive": "melanoma",
    "Melanoma, NOS": "melanoma",
    "Melanoma metastasis": "melanoma",
    "Seborrheic keratosis": "keratosis",
    "Solar or actinic keratosis": "keratosis",   # หมายเหตุ: AK เป็นรอยโรคก่อนมะเร็ง ควรให้คนที่ 2 ยืนยันว่ารวมกลุ่มนี้ได้
    "Solar lentigo": "keratosis",
    "Lichen planus like keratosis": "keratosis",
    "Lentigo NOS": "keratosis",
    "Pigmented benign keratosis": "keratosis",
    "Squamous cell carcinoma in situ": "scc",
    "Squamous cell carcinoma, Invasive": "scc",
    "Squamous cell carcinoma, NOS": "scc",
}
# ลำดับคลาสต้องคงที่ เพราะ index นี้คือ output ของโมเดลและ API ต้องใช้ชุดเดียวกัน
CLASSES = ["nevus", "bcc", "melanoma", "keratosis", "scc"]


def main():
    df = pd.read_csv("data/needed_image_ids.csv")
    df["label"] = df["iddx_3"].map(IDDX3_TO_CLASS)

    dropped = df[df["label"].isna()]
    kept = df[df["label"].notna()].copy()
    kept["label_idx"] = kept["label"].map({c: i for i, c in enumerate(CLASSES)})

    # stratify เพื่อให้ทุกชุดมีสัดส่วนคลาสใกล้กัน — สำคัญมากเมื่อคลาสเล็กมีแค่ ~70 ภาพ
    train, temp = train_test_split(kept, test_size=0.30, stratify=kept["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        kept.loc[part.index, "split"] = name

    kept[["isic_id", "iddx_3", "label", "label_idx", "split"]].to_csv("data/splits.csv", index=False)
    dropped[["isic_id", "iddx_2", "iddx_3"]].to_csv("data/dropped_images.csv", index=False)

    mapping = {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "classes": CLASSES,
        "iddx3_to_class": IDDX3_TO_CLASS,
        "split_seed": SEED,
        "split_ratio": "70/15/15 stratified, image-level (no patient_id available)",
    }
    json.dump(mapping, open("configs/class_mapping.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("kept:", len(kept), "dropped:", len(dropped))
    print(pd.crosstab(kept["label"], kept["split"]).loc[CLASSES])
    print("\ndropped by iddx_3:\n", dropped["iddx_3"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
