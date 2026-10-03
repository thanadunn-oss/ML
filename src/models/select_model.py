"""
select_model.py — ตรวจ gate ทุก run แล้วเลือกโมเดลที่ดีที่สุด

    python src/models/select_model.py

ขั้นตอน: (1) ทุก run ใน runs/ ต้องผ่าน gate ก่อน  (2) ในกลุ่มที่ผ่าน เลือก val Macro F1 สูงสุด (optimizing metric)
ผล: ตารางบนจอ + runs/selection.json  ·  ไม่แตะ test — หลังเลือกแล้วค่อย train.py --final-eval กับตัวที่เลือกตัวเดียว
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import check_gate


def main():
    rows = []
    for rd in sorted(glob.glob("runs/*/")):
        rd = rd.rstrip("/\\")
        if not os.path.exists(os.path.join(rd, "model.pt")):
            continue
        r = check_gate(rd)
        g = {c["check"]: c for c in r["checks"]}
        rows.append({"run": os.path.basename(rd), "val_macro_f1": r["optimizing_metric"]["val_macro_f1"],
                     "gate1": g["gate1_melanoma_recall"], "gate2": g["gate2_cpu_latency_p95"],
                     "passed": r["passed_gates"]})
    if not rows:
        sys.exit("ไม่พบ run ที่มี model.pt ใน runs/")
    print(f"{'run':36s} {'valF1':>7s}  ผล")
    for x in rows:
        mark = lambda c: ("✓ " if c["passed"] else "✗ ") + c["detail"]
        print(f"{x['run']:36s} {x['val_macro_f1']:7.4f}  {'ผ่าน' if x['passed'] else 'ไม่ผ่าน'}")
        print(f"{'':46s}gate1: {mark(x['gate1'])}")
        print(f"{'':46s}gate2: {mark(x['gate2'])}")
    ok = [x for x in rows if x["passed"]]
    best = max(ok, key=lambda x: x["val_macro_f1"]) if ok else None
    json.dump({"rule": "ผ่าน gate ทั้งหมดก่อน แล้วเลือก val Macro F1 สูงสุด",
               "selected": best["run"] if best else None,
               "candidates": [{k: v for k, v in x.items() if k not in ("gate1", "gate2")} |
                              {"gate1": x["gate1"]["detail"], "gate2": x["gate2"]["detail"]} for x in rows]},
              open("runs/selection.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nเลือก:", best["run"] if best else "ไม่มีโมเดลผ่าน gate — ห้ามนำขึ้นใช้งาน")


if __name__ == "__main__":
    main()
