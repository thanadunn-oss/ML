"""
src/data/prepare_ham10000.py — เตรียมข้อมูล HAM10000 (10,015 ภาพ, 7 คลาส) ให้ train.py ใช้

รันจากโฟลเดอร์รากของ repo:
    python src/data/prepare_ham10000.py --ham-dir data/ham10000

ผลลัพธ์:
    data/splits.csv            — ภาพละ 1 แถว: isic_id, path, lesion_id, dx, label, label_idx, split
    configs/class_mapping.json — ลำดับคลาส (API/gate ใช้ไฟล์นี้)
    data/data_report.json      — จำนวนภาพต่อคลาส/ชุด, ภาพที่หาไม่เจอ, ผลตรวจการรั่วไหล

เหตุผลของการออกแบบ:
- ใช้ครบ 7 คลาส เพราะทีมกำหนดให้เทรนด้วย ~10,000 ภาพ ถ้าตัด df/vasc ทิ้งจะเหลือ ~9,758
- แบ่งชุดตาม lesion_id (รอยโรคเดียวกันถ่ายหลายภาพ) → ภาพของรอยโรคเดียวกันอยู่ชุดเดียวกันเสมอ
  ถ้าแบ่งตามภาพ โมเดลจะ "เคยเห็น" รอยโรคใน test มาแล้วตอนเทรน ผลจะดีเกินจริง (data leakage)
- stratify ตาม dx ให้ทุกชุดมีสัดส่วนคลาสใกล้กัน — สำคัญกับคลาสเล็กอย่าง df (~115 ภาพ)
- หาไฟล์ภาพแบบค้นทุกโฟลเดอร์ย่อย เพราะ zip จาก Kaggle แตกออกมาเป็น part_1/part_2 และบางครั้งซ้อนสองชั้น
"""
import argparse, json, os
from pathlib import Path
import pandas as pd
import numpy as np  # ไม่ใช้ sklearn: บน Windows ที่เปิด Smart App Control ไฟล์ DLL ของ scipy ถูกบล็อก

SEED = 42
LABEL_SCHEMA_VERSION = "ham10000-v1"
# ลำดับนี้คือ index ที่โมเดลตอบออกมา — ห้ามสลับหลังจากเทรนแล้ว
DX_TO_CLASS = {
    "nv": "nevus",
    "mel": "melanoma",
    "bkl": "keratosis",          # benign keratosis-like: seborrheic keratosis, solar lentigo, LPLK
    "bcc": "bcc",
    "akiec": "akiec",            # actinic keratosis + intraepithelial carcinoma (Bowen's)
    "df": "dermatofibroma",
    "vasc": "vascular",
}
CLASSES = list(DX_TO_CLASS.values())


def grouped_stratified_split(df, val_frac=1/7, test_frac=1/7, seed=SEED):
    """แบ่ง lesion (ไม่ใช่ภาพ) ลงชุด train/val/test แยกตาม dx
    - ทำทีละ dx → ทุกชุดมีสัดส่วนคลาสใกล้กัน (stratified)
    - หน่วยที่สุ่มคือ lesion_id → ภาพของรอยโรคเดียวกันอยู่ชุดเดียวกันเสมอ (grouped, กันข้อมูลรั่ว)
    - ไล่ใส่ lesion ที่สุ่มลำดับแล้วลง test จนครบ ~test_frac ของจำนวนภาพ dx นั้น แล้วใส่ val ต่อ ที่เหลือเป็น train
    """
    rng = np.random.default_rng(seed)
    les = df.groupby("lesion_id").agg(dx=("dx", "first"), n=("image_id", "size")).reset_index()
    split_of = {}
    for dx, g in les.groupby("dx"):
        g = g.iloc[rng.permutation(len(g))]
        total = g["n"].sum()
        acc_test = acc_val = 0
        for lid, n in zip(g["lesion_id"], g["n"]):
            if acc_test < total * test_frac:
                split_of[lid] = "test"; acc_test += n
            elif acc_val < total * val_frac:
                split_of[lid] = "val"; acc_val += n
            else:
                split_of[lid] = "train"
    return df["lesion_id"].map(split_of)


def find_images(ham_dir):
    paths = {}
    for p in Path(ham_dir).rglob("*.jpg"):
        paths.setdefault(p.stem, p)  # มีไฟล์ซ้ำในหลายโฟลเดอร์ได้ → เก็บอันแรก
    return paths


def find_metadata(ham_dir):
    # Kaggle ให้ .csv ส่วน Harvard Dataverse ให้ .tab (คั่นด้วย tab) → รองรับทั้งสองแบบ
    hits = [p for p in Path(ham_dir).rglob("HAM10000_metadata*") if p.suffix.lower() in (".csv", ".tab", ".txt", "")]
    if not hits:
        raise SystemExit(f"หา HAM10000_metadata ใน {ham_dir} ไม่เจอ — แตกไฟล์ไว้ในโฟลเดอร์นี้หรือยัง?")
    return hits[0]


def read_metadata(path):
    df = pd.read_csv(path, sep=None, engine="python")  # sep=None → ให้ pandas เดาเองว่าคั่นด้วย , หรือ tab
    df.columns = [c.strip().strip('"') for c in df.columns]
    for c in ("lesion_id", "image_id", "dx"):
        df[c] = df[c].astype(str).str.strip().str.strip('"')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ham-dir", default="data/ham10000")
    ap.add_argument("--min-images", type=int, default=10000)
    a = ap.parse_args()

    meta = read_metadata(find_metadata(a.ham_dir))
    imgs = find_images(a.ham_dir)
    meta["exists"] = meta["image_id"].isin(imgs)
    missing = meta.loc[~meta["exists"], "image_id"].tolist()
    unknown_dx = sorted(set(meta["dx"]) - set(DX_TO_CLASS))
    df = meta[meta["exists"] & meta["dx"].isin(DX_TO_CLASS)].copy()

    if len(df) < a.min_images:
        print(f"⚠️  ใช้ได้ {len(df)} ภาพ น้อยกว่าเกณฑ์ {a.min_images} — ตรวจว่าแตกไฟล์ภาพครบทั้ง part_1 และ part_2")

    df["isic_id"] = df["image_id"]
    df["path"] = [imgs[i].as_posix() for i in df["image_id"]]
    df["label"] = df["dx"].map(DX_TO_CLASS)
    df["label_idx"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})

    # แบ่งชุดแบบ group (lesion_id) + stratified (dx): ~71/14/14
    df["split"] = grouped_stratified_split(df)

    # ตรวจการรั่วไหล: lesion_id เดียวกันต้องไม่อยู่หลายชุด
    leak = df.groupby("lesion_id")["split"].nunique()
    n_leak = int((leak > 1).sum())
    assert n_leak == 0, f"พบ lesion_id ข้ามชุด {n_leak} รายการ"

    os.makedirs("data", exist_ok=True); os.makedirs("configs", exist_ok=True)
    cols = ["isic_id", "path", "lesion_id", "dx", "label", "label_idx", "split"]
    df[cols].to_csv("data/splits.csv", index=False)
    json.dump({
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "dataset": "HAM10000",
        "classes": CLASSES,
        "dx_to_class": DX_TO_CLASS,
        "split_seed": SEED,
        "split_method": "grouped by lesion_id, stratified by dx (numpy, seed 42), ~71/14/14",
    }, open("configs/class_mapping.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    table = pd.crosstab(df["label"], df["split"]).reindex(CLASSES)[["train", "val", "test"]]
    report = {
        "total_used": int(len(df)),
        "metadata_rows": int(len(meta)),
        "missing_images": missing[:50], "missing_count": len(missing),
        "unknown_dx": unknown_dx,
        "lesions": int(df["lesion_id"].nunique()),
        "lesion_leak_across_splits": n_leak,
        "per_class_split": table.to_dict(orient="index"),
        "split_totals": df["split"].value_counts().to_dict(),
    }
    json.dump(report, open("data/data_report.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"ใช้ {len(df)} ภาพ จาก {df['lesion_id'].nunique()} รอยโรค | หาไม่เจอ {len(missing)} | lesion รั่วข้ามชุด {n_leak}")
    print(table.to_string())
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
