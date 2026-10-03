"""
gate.py — ด่านตรวจก่อนอนุมัติโมเดล (แยก optimizing metric ออกจาก gate)

แนวคิด (optimizing vs satisficing):
- Optimizing metric = val Macro F1 → "ยิ่งสูงยิ่งดี" ใช้เลือกว่าโมเดลไหนดีที่สุด
- Gate (satisficing) = เงื่อนไขขั้นต่ำที่ "ต้องผ่าน" ก่อน ไม่ว่า Macro F1 จะสูงแค่ไหน
    Gate 1: melanoma recall บน val ≥ เกณฑ์ และ precision ≥ เกณฑ์ → ความปลอดภัย: คลาสอันตรายที่สุดต้องไม่หลุดมาก
            (Macro F1 สูงได้แม้ melanoma แย่ เพราะเฉลี่ยรวมกับคลาสอื่น)
            precision กันโมเดลที่โกงด้วยการทายทุกภาพเป็น melanoma (recall = 1.0 แต่ไร้ประโยชน์)
    Gate 2: เวลาทำนายบน CPU (p95, รวมเตรียมภาพ) ≤ เกณฑ์ → ใช้งานจริงได้ ผู้ใช้ไม่ต้องรอนาน
- Validity checks = ลำดับคลาส/preprocessing ตรงกับ API (ถ้าไม่ตรง ผลจะผิดทั้งระบบ)

ใช้:
    python src/models/gate.py --candidate runs/<run_id> [--active runs/<current>]
    exit code 0 = ผ่าน gate, 1 = ไม่ผ่าน   ·  บันทึกผลใน runs/<run_id>/gate_result.json เสมอ
"""
import argparse, json, os, sys, time
import numpy as np


def _load(run_dir):
    with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    with open(os.path.join(run_dir, "serving_config.json"), encoding="utf-8") as f:
        s = json.load(f)
    return m, s


def measure_cpu_latency(run_dir, sample_image, n=50, warmup=5):
    """วัดเวลาทำนายภาพเดียวบน CPU รวมการเปิดภาพ+เตรียมภาพ (ใกล้กับที่ API ทำจริง) คืนค่า p50/p95 เป็น ms"""
    import torch
    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import build_model, build_transforms
    cfg = json.load(open(os.path.join(run_dir, "serving_config.json")))
    model = build_model(cfg["arch"], len(cfg["classes"]), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    model.eval()
    _, tf = build_transforms()
    times = []
    with torch.inference_mode():
        for i in range(warmup + n):
            t0 = time.perf_counter()
            x = tf(Image.open(sample_image).convert("RGB")).unsqueeze(0)
            model(x)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def _sample_val_image(splits_path, img_dir):
    import pandas as pd
    df = pd.read_csv(splits_path)
    r = df[df.split == "val"].iloc[0]
    return r.path if "path" in df.columns else os.path.join(img_dir, f"{r.isic_id}.jpg")


def check_gate(candidate_dir, active_dir=None, criteria_path="configs/gate_criteria.json",
               splits_path="data/splits.csv", img_dir="data/model_images"):
    crit = json.load(open(criteria_path, encoding="utf-8"))
    g, v = crit["gates"], crit["validity_checks"]
    m, s = _load(candidate_dir)
    checks = []

    def add(kind, name, passed, detail):
        checks.append({"type": kind, "check": name, "passed": bool(passed), "detail": detail})

    # --- Gate 1: melanoma recall (val) ---
    if m.get("split_evaluated") != "val" or "melanoma" not in m.get("per_class", {}):
        add("gate", "gate1_melanoma_recall", False, "metrics.json ไม่มี recall ของ melanoma บน val → รัน train.py --eval-val ก่อน")
    else:
        rec = float(m["per_class"]["melanoma"]["recall"])
        prec = float(m["per_class"]["melanoma"]["precision"])
        # ต้องผ่านทั้งคู่: recall อย่างเดียวโกงได้ (ทายทุกภาพเป็น melanoma → recall 1.0)
        ok = rec >= g["gate1_melanoma_recall_min"] and prec >= g.get("gate1_melanoma_precision_min", 0.0)
        add("gate", "gate1_melanoma_recall", ok,
            f"melanoma recall {rec:.3f} (>= {g['gate1_melanoma_recall_min']}), "
            f"precision {prec:.3f} (>= {g.get('gate1_melanoma_precision_min', 0.0)})")

    # --- Gate 2: CPU latency p95 ---
    p50, p95 = measure_cpu_latency(candidate_dir, _sample_val_image(splits_path, img_dir))
    add("gate", "gate2_cpu_latency_p95", p95 <= g["gate2_cpu_latency_p95_ms_max"],
        f"p95 {p95:.1f} ms (p50 {p50:.1f}) ต้อง <= {g['gate2_cpu_latency_p95_ms_max']} ms")

    # --- Validity: สัญญากับ API ---
    add("validity", "class_order", s["classes"] == v["expected_classes"], f"classes = {s['classes']}")
    add("validity", "preprocessing_version", s.get("preprocessing_version") == v["expected_preprocessing_version"],
        f"preprocessing_version = {s.get('preprocessing_version')}")

    passed = all(c["passed"] for c in checks)
    opt = float(m["best_val_macro_f1"])
    # Optimizing metric: ไม่ใช่ gate — ใช้ตัดสินว่า "ควรเปลี่ยนรุ่นไหม" หลังผ่าน gate แล้ว
    comparison = None
    if active_dir:
        a_opt = float(_load(active_dir)[0]["best_val_macro_f1"])
        comparison = {"active": active_dir, "active_val_macro_f1": a_opt, "candidate_better": opt > a_opt}
    result = {
        "candidate": candidate_dir, "gate_version": crit["gate_version"],
        "passed_gates": passed,
        "optimizing_metric": {"val_macro_f1": opt},
        "vs_active": comparison,
        "recommend_promote": passed and (comparison is None or comparison["candidate_better"]),
        "checks": checks, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(result, open(os.path.join(candidate_dir, "gate_result.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    return result


def print_result(r):
    for c in r["checks"]:
        print(("PASS " if c["passed"] else "FAIL ") + f"[{c['type']}] {c['check']:24s} {c['detail']}")
    print(f"optimizing metric: val Macro F1 = {r['optimizing_metric']['val_macro_f1']:.4f}")
    if r["vs_active"]:
        print(f"  vs active {r['vs_active']['active_val_macro_f1']:.4f} → {'ดีกว่า' if r['vs_active']['candidate_better'] else 'ไม่ดีกว่า'}")
    print("GATES:", "PASSED" if r["passed_gates"] else "REJECTED", "| recommend promote:", r["recommend_promote"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--active")
    ap.add_argument("--criteria", default="configs/gate_criteria.json")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--img-dir", default="data/model_images")
    a = ap.parse_args()
    r = check_gate(a.candidate, a.active, a.criteria, a.splits, a.img_dir)
    print_result(r)
    sys.exit(0 if r["passed_gates"] else 1)


if __name__ == "__main__":
    main()
