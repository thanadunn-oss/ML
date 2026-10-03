"""
registry.py — บันทึกการทดลองและทะเบียนโมเดลด้วย MLflow (backend: SQLite ในเครื่อง)

ทำไม SQLite: MLflow รุ่นใหม่เลิกพัฒนา backend แบบไฟล์ (./mlruns) แล้ว และ SQLite ไม่ต้องตั้งเซิร์ฟเวอร์
ทำไม mlflow-skinny: mlflow ตัวเต็มต้องใช้ scipy/scikit-learn ซึ่งถูก Windows Smart App Control บล็อก

ทะเบียนโมเดลชื่อ skin-lesion-classifier
- ทุกรุ่นมี tag: gate (passed/rejected), status (candidate/champion/archived/rejected), run_dir, val_macro_f1
- alias "champion" = รุ่นที่ให้บริการ, "previous" = รุ่นก่อนหน้า (ใช้ rollback)
"""
import json, os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "skin-lesion-ham10000"
MODEL_NAME = "skin-lesion-classifier"
BUNDLE_FILES = ["model.pt", "serving_config.json", "metrics.json", "gate_result.json"]


def client():
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def log_run(run_dir, class_mapping="configs/class_mapping.json", splits="data/splits.csv"):
    """บันทึก run ที่เทรนแล้วลง MLflow ครบ 6 รายการ: โค้ด ข้อมูล พารามิเตอร์ ตัวชี้วัด ไฟล์ผลลัพธ์ สภาพแวดล้อม"""
    c = client(); mlflow.set_experiment(EXPERIMENT)
    m = json.load(open(os.path.join(run_dir, "metrics.json"), encoding="utf-8"))
    cfg = m["config"]
    run_name = m.get("run_id", os.path.basename(run_dir.rstrip("/\\")))
    with mlflow.start_run(run_name=run_name) as r:
        # 1) พารามิเตอร์
        mlflow.log_params({k: cfg[k] for k in ["arch", "epochs", "lr", "batch_size", "weighted_loss", "seed", "img_size"] if k in cfg})
        # 2) ข้อมูล + 3) สภาพแวดล้อม + 4) โค้ด (เป็น tag)
        mlflow.set_tags({
            "train_run_id": run_name, "run_dir": run_dir,
            "label_schema_version": cfg.get("label_schema_version"), "preprocessing_version": cfg.get("preprocessing_version"),
            "device": cfg.get("device"), "code_commit": _git_commit(),
            "split_file": splits, "split_sha256": _sha256(splits),
        })
        # 5) ตัวชี้วัด (val เท่านั้น)
        mlflow.log_metric("val_macro_f1", float(m["best_val_macro_f1"]))
        for c_, v in m.get("per_class", {}).items():
            mlflow.log_metric(f"val_recall_{c_}", float(v["recall"]))
            mlflow.log_metric(f"val_precision_{c_}", float(v["precision"]))
        for h in m.get("history", []):
            mlflow.log_metric("val_macro_f1_epoch", h["val_macro_f1"], step=h["epoch"])
            mlflow.log_metric("train_loss", h["train_loss"], step=h["epoch"])
        # 6) ไฟล์ผลลัพธ์ (bundle ที่ API ใช้ได้ทันที)
        for f in BUNDLE_FILES:
            p = os.path.join(run_dir, f)
            if os.path.exists(p):
                mlflow.log_artifact(p, artifact_path="bundle")
        if os.path.exists(class_mapping):
            mlflow.log_artifact(class_mapping, artifact_path="bundle")
        return r.info.run_id


def register(mlflow_run_id, gate_result):
    """ลงทะเบียนเป็นรุ่นใหม่เสมอ (ทั้งผ่านและไม่ผ่าน gate) เพื่อให้มีประวัติ — สถานะบอกใน tag"""
    c = client()
    try:
        c.create_registered_model(MODEL_NAME)
    except Exception:
        pass  # มีอยู่แล้ว
    passed = bool(gate_result["passed_gates"])
    mv = c.create_model_version(MODEL_NAME, f"runs:/{mlflow_run_id}/bundle", mlflow_run_id, tags={
        "gate": "passed" if passed else "rejected",
        "status": "candidate" if passed else "rejected",
        "val_macro_f1": f"{gate_result['optimizing_metric']['val_macro_f1']:.4f}",
        "gate_detail": "; ".join(c_["detail"] for c_ in gate_result["checks"] if not c_["passed"]) or "all passed",
    })
    return mv.version


def get_alias(alias):
    c = client()
    try:
        return c.get_model_version_by_alias(MODEL_NAME, alias)
    except Exception:
        return None


def promote(version):
    """ตั้งรุ่นใหม่เป็น champion และเลื่อนรุ่นเดิมเป็น previous"""
    c = client()
    old = get_alias("champion")
    if old is not None and old.version != str(version):
        c.set_registered_model_alias(MODEL_NAME, "previous", old.version)
        c.set_model_version_tag(MODEL_NAME, old.version, "status", "archived")
    c.set_registered_model_alias(MODEL_NAME, "champion", version)
    c.set_model_version_tag(MODEL_NAME, version, "status", "champion")
    return old.version if old is not None else None


def download(alias_or_version, dst):
    client()
    ref = f"models:/{MODEL_NAME}@{alias_or_version}" if not str(alias_or_version).isdigit() else f"models:/{MODEL_NAME}/{alias_or_version}"
    return mlflow.artifacts.download_artifacts(ref, dst_path=dst)


def champion_run_dir():
    """โฟลเดอร์ run ของ champion ปัจจุบัน (ใช้เทียบใน gate)"""
    v = get_alias("champion")
    if v is None:
        return None
    return client().get_run(v.run_id).data.tags.get("run_dir")


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path):
    import hashlib
    if not os.path.exists(path):
        return "missing"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
