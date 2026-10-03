"""
ตรวจว่า API ตอบตรงกับโมเดล (T07) — รันจากโฟลเดอร์ API/ หลังวาง model.pt แล้ว

    python tests/check_reference.py --ref ../docs/reference_predictions.json

ส่งภาพอ้างอิงทุกภาพเข้า /predict ผ่าน TestClient (ไม่ต้องเปิดเซิร์ฟเวอร์) แล้วเทียบ
คลาสที่ทาย + ความน่าจะเป็นทุกคลาส (ต่างได้ไม่เกิน tolerance) และทดสอบกรณีผิดพลาด (ไฟล์เสีย/ชนิดผิด)
"""
import argparse, json, os, sys, tempfile
os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "test.db"))  # ไม่ปนกับฐานข้อมูลจริง
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
import app as api


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="../docs/reference_predictions.json")
    ap.add_argument("--tol", type=float, default=0.001)
    a = ap.parse_args()
    ref = json.load(open(a.ref, encoding="utf-8"))
    base = os.path.dirname(os.path.dirname(os.path.abspath(a.ref)))  # รากของ repo (path ภาพในไฟล์อ้างอิงเทียบจากราก)
    ok_all = True
    with TestClient(api.app) as c:
        ready = c.get("/ready")
        print("ready:", ready.status_code, ready.json())
        if ready.status_code != 200:
            sys.exit("โมเดลยังไม่พร้อม — วาง model.pt ใน artifacts/model-v3/ ก่อน")
        if ready.json().get("run_id") != ref["model_run_id"]:
            print(f"⚠️  run_id ของ API ({ready.json().get('run_id')}) ไม่ตรงกับไฟล์อ้างอิง ({ref['model_run_id']})"); ok_all = False
        for case in ref["cases"]:
            path = os.path.join(base, case["image"])
            with open(path, "rb") as f:
                r = c.post("/predict", files={"image": (os.path.basename(path), f, "image/jpeg")}).json()
            diff = max(abs(r["scores"][k] - v) for k, v in case["expected_probs"].items())
            ok = r["predicted_class"] == case["expected_pred"] and diff <= a.tol
            ok_all &= ok
            print(f"{'PASS' if ok else 'FAIL'} {case['isic_id']} ทาย={r['predicted_class']:15s} คาด={case['expected_pred']:15s} "
                  f"prob ต่างสูงสุด={diff:.6f} review={r['review_required']}")
        bad = c.post("/predict", files={"image": ("x.jpg", b"not an image", "image/jpeg")}).status_code
        wrong = c.post("/predict", files={"image": ("x.txt", b"hello", "text/plain")}).status_code
        print(f"{'PASS' if bad == 400 else 'FAIL'} ไฟล์เสีย → {bad} (คาด 400)")
        print(f"{'PASS' if wrong == 415 else 'FAIL'} ชนิดไฟล์ผิด → {wrong} (คาด 415)")
        ok_all &= bad == 400 and wrong == 415
    print("\nผลรวม:", "ผ่านทั้งหมด (T07)" if ok_all else "มีรายการไม่ผ่าน")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
