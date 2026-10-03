"""
src/models/train.py (รันจากโฟลเดอร์รากของ repo) — Baseline รอบที่ 1: ResNet18 + transfer learning

ตัวอย่างการรัน (บน Colab/เครื่องที่ดาวน์โหลด pretrained weights ได้):
    python src/models/train.py --arch resnet18 --epochs 15
รอบ 2:  python src/models/train.py --arch efficientnet_b0 --epochs 15
รอบ 3:  python src/models/train.py --arch efficientnet_b0 --weighted-loss --epochs 25
เลือกโมเดลจาก val แล้วเท่านั้น จึงประเมิน test ครั้งเดียว:
        python src/models/train.py --final-eval runs/<run_id>
"""
import argparse, json, os, sys, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from metrics import per_class_prf, macro_f1  # numpy ล้วน (ไม่พึ่ง sklearn/scipy)

PREPROCESSING_VERSION = "v1"
IMG_SIZE = 224          # ภาพจริงแค่ ~100-215 px แต่ขยายเป็น 224 เพราะ pretrained weights เรียนมาที่ขนาดนี้
                        # ถ้าย่อลงไปอีกจะเสียรายละเอียดขอบ/สี ซึ่งเป็นสิ่งที่แยกรอยโรคได้
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ต้องใช้สถิติ ImageNet ให้ตรงกับ weights


def set_seed(seed):
    # ล็อกทุกแหล่งสุ่ม เพื่อให้ผลรันซ้ำได้ใกล้เคียงกัน (CPU/GPU ต่างรุ่นอาจต่างเล็กน้อย)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_transforms():
    # แยก transform ตอนเทรน กับตอนประเมิน/ให้บริการ ชัดเจน
    # -> eval_tf ชุดนี้คือสิ่งที่คนที่ 5 ต้องใช้ใน API เหมือนกันทุกตัวอักษร (ป้องกัน training-serving skew)
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # augmentation เฉพาะตอนเทรน: รอยโรคบนผิวไม่มีทิศ "ด้านบน" จึงพลิก/หมุนได้โดยไม่เปลี่ยนความหมาย
    # color jitter ใส่แค่น้อยๆ เพราะสีเป็นข้อมูลสำคัญในการแยก melanoma
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


class LesionDataset(Dataset):
    def __init__(self, df, img_dir, tf):
        self.df, self.img_dir, self.tf = df.reset_index(drop=True), img_dir, tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # HAM10000: splits.csv มีคอลัมน์ path (ภาพอยู่หลายโฟลเดอร์) / ชุด ISIC 2024 เดิม: ใช้ img_dir/<isic_id>.jpg
        p = r.path if "path" in self.df.columns and isinstance(r.path, str) else os.path.join(self.img_dir, f"{r.isic_id}.jpg")
        img = Image.open(p).convert("RGB")
        return self.tf(img), int(r.label_idx)


def build_model(arch, n_classes, pretrained):
    if arch == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)  # เปลี่ยนหัวจาก 1000 คลาส ImageNet เป็นคลาสของเรา
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()  # ปิด dropout/BN แบบเทรน — ถ้าลืม ผลประเมินจะไม่ตรงกับ API
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)  # lr ต่ำ เพราะ fine-tune ทั้งโมเดลจาก weights ที่ดีอยู่แล้ว
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true", help="ปิด mixed precision บน GPU")
    ap.add_argument("--img-dir", default="data/model_images")  # ภาพไม่อยู่ใน Git (ใหญ่) ต้องวางเองตาม README
    ap.add_argument("--mapping", default="configs/class_mapping.json")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--final-eval", metavar="RUN_DIR", help="ประเมิน test ครั้งเดียวกับโมเดลที่เลือกแล้ว (ไม่เทรน)")
    ap.add_argument("--eval-val", metavar="RUN_DIR", help="ประเมิน val ใหม่จาก model.pt แล้วเขียน metrics.json รูปแบบปัจจุบัน (ไม่เทรน)")
    a = ap.parse_args()
    if a.eval_val:
        return eval_val(a.eval_val, a.img_dir, a.splits)
    if a.final_eval:
        return final_eval(a.final_eval, a.img_dir, a.splits)

    set_seed(a.seed)
    torch.backends.cudnn.benchmark = True  # ขนาดภาพคงที่ 224 → ให้ cuDNN เลือกอัลกอริทึมที่เร็วที่สุด
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapping = json.load(open(a.mapping, encoding="utf-8"))
    classes = mapping["classes"]
    df = pd.read_csv(a.splits)
    tr, va = (df[df.split == s] for s in ["train", "val"])

    train_tf, eval_tf = build_transforms()
    # pin_memory + หลาย worker: ภาพ HAM10000 ใหญ่ (600x450) การอ่าน/ย่อภาพบน CPU มักช้ากว่า GPU → ใช้หลาย worker ช่วย
    dl = lambda d, tf, sh: DataLoader(LesionDataset(d, a.img_dir, tf), batch_size=a.batch_size, shuffle=sh,
                                      num_workers=a.num_workers, pin_memory=(device == "cuda"),
                                      persistent_workers=a.num_workers > 0)
    tr_dl, va_dl = dl(tr, train_tf, True), dl(va, eval_tf, False)  # ไม่สร้าง test loader ตอนเทรนเลย

    model = build_model(a.arch, len(classes), not a.no_pretrained).to(device)

    weight = None
    if a.weighted_loss:
        # น้ำหนัก = ผกผันกับจำนวนภาพต่อคลาส -> คลาสเล็ก (scc) ทายผิดแล้วโดนลงโทษมากขึ้น
        counts = tr.label_idx.value_counts().sort_index().values
        weight = torch.tensor(len(tr) / (len(classes) * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    use_amp = device == "cuda" and not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{a.arch}"
    out = os.path.join(a.out, run_id); os.makedirs(out, exist_ok=True)
    best_f1, history = -1, []

    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); total = 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            # AMP (mixed precision) บน GPU: เร็วขึ้นและใช้หน่วยความจำน้อยลง เหมาะกับ RTX 4050 (6 GB)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * len(y)
        yv, pv = predict(model, va_dl, device)
        # เลือก "รอบที่ดีที่สุด" จาก Macro F1 บน validation เท่านั้น — ห้ามแตะ test ระหว่างนี้
        val_f1 = macro_f1(yv, pv, len(classes))
        history.append({"epoch": ep, "train_loss": total / len(tr), "val_macro_f1": val_f1})
        print(f"ep{ep:02d} loss={total/len(tr):.4f} val_macroF1={val_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))

    # ประเมินบน VAL เท่านั้น ด้วยโมเดลที่ดีที่สุด -> ใช้วิเคราะห์ข้อผิดพลาดและเลือกโมเดล
    # test ถูกแยกไปไว้ในคำสั่ง --final-eval เพื่อไม่ให้เห็นค่า test ระหว่างเลือกโมเดล
    model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=device))
    yv, pv = predict(model, va_dl, device)
    result = {
        "run_id": run_id,
        "config": vars(a) | {"img_size": IMG_SIZE, "preprocessing_version": PREPROCESSING_VERSION,
                             "label_schema_version": mapping["label_schema_version"], "device": device},
        "split_evaluated": "val",
        "best_val_macro_f1": best_f1,
        **report(yv, pv, classes),
        "history": history,
    }
    json.dump(result, open(os.path.join(out, "metrics.json"), "w"), indent=2, default=float)
    json.dump({"classes": classes, "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
               "preprocessing_version": PREPROCESSING_VERSION, "arch": a.arch},
              open(os.path.join(out, "serving_config.json"), "w"), indent=2)  # ส่งให้คนที่ 5 ใช้คู่กับ model.pt
    print(json.dumps({"run_id": run_id, "best_val_macro_f1": best_f1}, default=float))
    print_report(result, "VAL")


def report(y, p, classes):
    pr, rc, f, s, cm = per_class_prf(y, p, len(classes))
    return {
        "macro_f1": float(f.mean()),
        "per_class": {c: {"precision": pr[i], "recall": rc[i], "f1": f[i], "support": int(s[i])} for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
    }


def print_report(r, name):
    print(f"[{name}] macro_f1={r['macro_f1']:.4f}")
    for c, v in r["per_class"].items():
        print(f"  {c:10s} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} n={v['support']}")
    print("  confusion (rows=true, cols=pred):", r["confusion_matrix"]["labels"])
    for c, row in zip(r["confusion_matrix"]["labels"], r["confusion_matrix"]["matrix"]):
        print(f"  {c:10s} {row}")


def eval_val(run_dir, img_dir, splits_path):
    """ใช้กับ run เก่า (train.py รุ่นแรกเก็บผลรายคลาสของ test ไว้) ให้มีผลรายคลาสของ val สำหรับ gate
    เก็บ config/history เดิมไว้ เปลี่ยนเฉพาะส่วนผลประเมิน และย้ายค่า test เดิมไปเก็บแยกเพื่อความโปร่งใส"""
    cfg = json.load(open(os.path.join(run_dir, "serving_config.json")))
    classes = cfg["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["arch"], len(classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    _, eval_tf = build_transforms()
    df = pd.read_csv(splits_path)
    va_dl = DataLoader(LesionDataset(df[df.split == "val"], img_dir, eval_tf), batch_size=32, num_workers=2)
    yv, pv = predict(model, va_dl, device)
    path = os.path.join(run_dir, "metrics.json")
    old = json.load(open(path))
    new = {k: old[k] for k in ["run_id", "config", "history"] if k in old}
    new.update({"split_evaluated": "val", "best_val_macro_f1": macro_f1(yv, pv, len(classes)),
                **report(yv, pv, classes)})
    if "test_macro_f1" in old:  # ค่าที่ train.py รุ่นแรกคำนวณไว้ — เก็บไว้เป็นประวัติ ไม่ใช้ตัดสินใจ
        new["legacy_test_results_not_for_selection"] = {k: old[k] for k in ["test_macro_f1", "per_class", "confusion_matrix"] if k in old}
    json.dump(new, open(path, "w"), indent=2, default=float)
    print_report(new, "VAL (re-evaluated)")


def final_eval(run_dir, img_dir, splits_path):
    """ประเมิน test ครั้งเดียว กับโมเดลที่ 'เลือกแล้ว' เท่านั้น — ไม่เทรนใหม่ ไม่ปรับอะไรหลังจากนี้"""
    marker = os.path.join(os.path.dirname(run_dir.rstrip("/")) or ".", "FINAL_TEST_USED.txt")
    if os.path.exists(marker):
        # กันเผลอประเมิน test หลายโมเดลแล้วเลือกตัวที่ดีที่สุด (ซึ่งก็คือการใช้ test เลือกโมเดล)
        raise SystemExit(f"test ถูกใช้ไปแล้ว: {open(marker).read().strip()} — ลบไฟล์นี้เองถ้ามีเหตุผลจริงและบันทึกไว้ในรายงาน")
    cfg = json.load(open(os.path.join(run_dir, "serving_config.json")))
    classes = cfg["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["arch"], len(classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    _, eval_tf = build_transforms()
    df = pd.read_csv(splits_path)
    te_dl = DataLoader(LesionDataset(df[df.split == "test"], img_dir, eval_tf), batch_size=32, num_workers=2)
    yt, pt = predict(model, te_dl, device)
    r = {"run_dir": run_dir, "split_evaluated": "test", **report(yt, pt, classes)}
    json.dump(r, open(os.path.join(run_dir, "test_metrics.json"), "w"), indent=2, default=float)
    open(marker, "w").write(f"{run_dir} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_report(r, "TEST (final)")


if __name__ == "__main__":
    main()
