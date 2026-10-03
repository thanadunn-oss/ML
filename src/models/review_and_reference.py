"""
review_and_reference.py — (1) ตารางเกณฑ์ส่งทบทวนบน val  (2) ภาพอ้างอิงสำหรับตรวจ API

    python src/models/review_and_reference.py --run runs/<run_id ที่เลือก>

ผลลัพธ์:
    docs/review_threshold_sweep.json  — แต่ละค่า τ: % ส่งทบทวน, accuracy ของภาพที่ระบบตอบเอง, melanoma ที่หลุด
    docs/reference_predictions.json   — ภาพอ้างอิง + ผลที่ API ต้องตอบ (T07)
    docs/reference_images/*.jpg       — ภาพอ้างอิง (เล็ก ใส่ Git ได้) เพราะภาพ HAM10000 ไม่อยู่ใน Git

ใช้ val เท่านั้น (ไม่ใช้ test เลือกเกณฑ์)
"""
import argparse, json, os, shutil, sys
import numpy as np, pandas as pd, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model, build_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", default="data/splits.csv")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.run, "serving_config.json")))
    cls = cfg["classes"]; mel = cls.index("melanoma")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(cfg["arch"], len(cls), pretrained=False).to(device)
    m.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=device)); m.eval()
    _, tf = build_transforms()
    va = pd.read_csv(a.splits).query("split == 'val'").reset_index(drop=True)

    P = []
    with torch.inference_mode():
        for i in range(0, len(va), 64):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in va.path[i:i + 64]]).to(device)
            P.append(torch.softmax(m(x), 1).cpu().numpy())
    P = np.concatenate(P); y = va.label_idx.values; pred = P.argmax(1); conf = P.max(1)

    # (1) sweep เกณฑ์: max prob < τ → ส่งทบทวน
    sweep = []
    for tau in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        auto = conf >= tau
        sweep.append({
            "tau": tau,
            "review_frac": round(float(1 - auto.mean()), 3),
            "auto_accuracy": round(float((pred[auto] == y[auto]).mean()), 3) if auto.any() else None,
            "melanoma_missed_without_review": int(((y == mel) & auto & (pred != mel)).sum()),
            "melanoma_total": int((y == mel).sum()),
        })
    json.dump({"run": a.run, "split": "val", "rule": "max probability < tau → needs_review", "sweep": sweep},
              open("docs/review_threshold_sweep.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{'tau':>5} {'ส่งทบทวน':>9} {'acc(ตอบเอง)':>12} {'melanoma หลุด':>14}")
    for s in sweep:
        print(f"{s['tau']:5.1f} {s['review_frac']*100:8.1f}% {s['auto_accuracy'] or 0:12.3f} "
              f"{s['melanoma_missed_without_review']:>7}/{s['melanoma_total']}")

    # (2) ภาพอ้างอิง: คลาสละ 1 ภาพที่ทายถูกมั่นใจสุด + 1 ภาพที่ความมั่นใจต่ำ
    os.makedirs("docs/reference_images", exist_ok=True)
    picks = []
    for k, c in enumerate(cls):
        idx = np.where((y == k) & (pred == k))[0]
        if len(idx): picks.append(int(idx[np.argmax(conf[idx])]))
    low = np.where(conf < 0.6)[0]
    if len(low): picks.append(int(low[0]))
    # ค่าที่คาดหวังต้องคำนวณแบบเดียวกับ API: CPU, ทีละ 1 ภาพ
    # (GPU/batch ปัดเศษต่างจาก CPU เล็กน้อย ภาพที่โมเดลไม่มั่นใจอาจต่างเกิน 0.001 ได้ทั้งที่ API ถูกต้อง)
    cpu = build_model(cfg["arch"], len(cls), pretrained=False)
    cpu.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location="cpu")); cpu.eval()
    cases = []
    for i in picks:
        r = va.iloc[i]; dst = f"docs/reference_images/{r.isic_id}.jpg"
        shutil.copyfile(r.path, dst)
        with torch.inference_mode():
            pc = torch.softmax(cpu(tf(Image.open(dst).convert("RGB")).unsqueeze(0)), 1)[0].numpy()
        cases.append({"isic_id": r.isic_id, "image": dst, "true_label": r.label,
                      "expected_pred": cls[int(pc.argmax())],
                      "expected_probs": {c: round(float(p), 6) for c, p in zip(cls, pc)}})
    json.dump({"model_run_id": os.path.basename(a.run.rstrip("/\\")),
               "tolerance": "ความน่าจะเป็นต่างได้ไม่เกิน 0.001 ต่อคลาส",
               "computed_on": "CPU, ทีละ 1 ภาพ (เหมือน API)",
               "note": "needs_review ขึ้นกับเกณฑ์ที่ทีมเลือก ดู docs/review_threshold_sweep.json",
               "cases": cases},
              open("docs/reference_predictions.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nภาพอ้างอิง {len(cases)} ภาพ → docs/reference_images/ และ docs/reference_predictions.json")


if __name__ == "__main__":
    main()
