"""
flow.py — Pipeline แบบ DAG ด้วย Prefect: ตรวจข้อมูล → เทรน → gate → บันทึก/ลงทะเบียน → อนุมัติ → นำขึ้นใช้งาน

คำสั่งเดียว (รันจากรากของ repo):
    python src/pipelines/flow.py --arch efficientnet_b0 --weighted-loss --epochs 20
ใช้ run ที่เทรนไว้แล้ว (ข้ามการเทรน — ใช้ลงทะเบียน run เดิมหรือสาธิต):
    python src/pipelines/flow.py --from-run runs/<run_id>
เหตุการณ์เทรนใหม่จาก Monitoring (คนที่ 7) ส่งเหตุผลมาด้วย:
    python src/pipelines/flow.py --epochs 20 --weighted-loss --reason "data drift alert 2026-10-05"

หลักการ:
- Task ถัดไปทำงานเมื่อ Task ก่อนหน้าผ่านเท่านั้น และแสดงว่าหยุดที่ขั้นไหน
- retry เฉพาะข้อผิดพลาดชั่วคราว (เช่น เทรนล้มเพราะ GPU) — ข้อมูลเสียไม่ retry
- gate ไม่ผ่าน → ลงทะเบียนเป็น rejected แต่ไม่เปลี่ยน champion → API ใช้รุ่นเดิมต่อ
- ไม่มีวงวน: Monitoring เรียก flow ใหม่ทั้งรอบ ไม่ย้อนกลับเข้า flow เดิม
"""
import argparse, json, os, shutil, subprocess, sys
from prefect import flow, task, get_run_logger

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry  # noqa: E402
from gate import check_gate  # noqa: E402


def _log():
    """logger ของ Prefect เมื่ออยู่ใน flow; ถ้าเรียกตรง (เช่นจาก rollback.py) ใช้ logging ธรรมดา"""
    try:
        return get_run_logger()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return logging.getLogger("pipeline")


class DataError(Exception):
    """ข้อมูลไม่ผ่านการตรวจ — ไม่ retry"""


@task(name="1-validate-data")
def validate_data(min_images: int = 10000):
    log = get_run_logger()
    r = subprocess.run([sys.executable, "src/data/prepare_ham10000.py", "--min-images", str(min_images)],
                       cwd=ROOT, capture_output=True, text=True)
    log.info(r.stdout[-800:])
    if r.returncode != 0:
        raise DataError(f"prepare_ham10000 ล้มเหลว: {r.stderr[-500:]}")
    rep = json.load(open(os.path.join(ROOT, "data", "data_report.json"), encoding="utf-8"))
    problems = []
    if rep["total_used"] < min_images: problems.append(f"ภาพ {rep['total_used']} < {min_images}")
    if rep["lesion_leak_across_splits"] != 0: problems.append("lesion รั่วข้ามชุด")
    if rep["missing_count"] > 0: problems.append(f"ภาพหาย {rep['missing_count']}")
    if rep["unknown_dx"]: problems.append(f"ฉลากนอกคลาส {rep['unknown_dx']}")
    if problems:
        raise DataError("ข้อมูลไม่ผ่าน: " + "; ".join(problems))
    return rep


@task(name="2-train", retries=1, retry_delay_seconds=30)  # retry ครั้งเดียว เผื่อ GPU/หน่วยความจำชั่วคราว
def train(arch: str, epochs: int, weighted_loss: bool, extra_args: str = ""):
    cmd = [sys.executable, "src/models/train.py", "--arch", arch, "--epochs", str(epochs)] + extra_args.split()
    if weighted_loss:
        cmd.append("--weighted-loss")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    get_run_logger().info(r.stdout[-1500:])
    if r.returncode != 0:
        raise RuntimeError(f"เทรนล้มเหลว: {r.stderr[-800:]}")
    run_id = next(json.loads(l)["run_id"] for l in r.stdout.splitlines() if l.startswith('{"run_id"'))
    return os.path.join("runs", run_id)


@task(name="3-gate")
def gate(run_dir: str):
    active = registry.champion_run_dir()
    if active and not os.path.exists(os.path.join(ROOT, active, "metrics.json")):
        active = None  # champion อยู่คนละเครื่อง → เทียบไม่ได้ ให้ gate ตัดสินอย่างเดียว
    res = check_gate(run_dir, active)
    log = get_run_logger()
    for c in res["checks"]:
        log.info(("PASS " if c["passed"] else "FAIL ") + c["check"] + " — " + c["detail"])
    return res


@task(name="4-log-and-register")
def log_and_register(run_dir: str, gate_result: dict, reason: str):
    mid = registry.log_run(run_dir)
    if reason:
        registry.client().set_tag(mid, "trigger_reason", reason)
    version = registry.register(mid, gate_result)
    get_run_logger().info(f"MLflow run {mid} → {registry.MODEL_NAME} version {version} ({'passed' if gate_result['passed_gates'] else 'rejected'})")
    return version


@task(name="5-approve")
def approve(version: str, gate_result: dict):
    log = get_run_logger()
    if not gate_result["recommend_promote"]:
        why = "ไม่ผ่าน gate" if not gate_result["passed_gates"] else "ผ่าน gate แต่ไม่ดีกว่ารุ่นที่ใช้อยู่ (optimizing metric)"
        log.warning(f"ไม่อนุมัติ version {version}: {why} → คง champion เดิม")
        return False
    prev = registry.promote(version)
    log.info(f"อนุมัติ version {version} เป็น champion (รุ่นก่อนหน้า: {prev})")
    return True


@task(name="6-deploy-and-verify")
def deploy(deploy_dir: str):
    """ดึง bundle ของ champion ไปวางให้ API แล้วตรวจหลังวาง (ไฟล์ครบ + โหลดได้ + ลำดับคลาสตรง)"""
    log = _log()
    v = registry.get_alias("champion")
    tmp = os.path.join(ROOT, ".deploy_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    src = registry.download("champion", tmp)
    os.makedirs(deploy_dir, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join(deploy_dir, f))
    train_run = registry.client().get_run(v.run_id).data.tags.get("train_run_id", "unknown")
    open(os.path.join(deploy_dir, "RUN_ID.txt"), "w", encoding="utf-8").write(train_run)
    shutil.rmtree(tmp, ignore_errors=True)
    verify_bundle(deploy_dir)
    log.info(f"วาง champion (version {v.version}, run {train_run}) ที่ {deploy_dir} แล้ว — เริ่ม API ใหม่เพื่อโหลดรุ่นนี้")
    return {"version": v.version, "run_id": train_run, "deploy_dir": deploy_dir}


def verify_bundle(d):
    import torch
    from train import build_model
    cfg = json.load(open(os.path.join(d, "serving_config.json")))
    mp = json.load(open(os.path.join(d, "class_mapping.json"), encoding="utf-8"))
    if cfg["classes"] != mp["classes"]:
        raise RuntimeError("ลำดับคลาสใน serving_config กับ class_mapping ไม่ตรงกัน")
    m = build_model(cfg["arch"], len(cfg["classes"]), pretrained=False)
    m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location="cpu", weights_only=True))
    m.eval()
    with torch.inference_mode():
        out = m(torch.zeros(1, 3, cfg["img_size"], cfg["img_size"]))
    if out.shape[1] != len(cfg["classes"]):
        raise RuntimeError("จำนวน output ไม่ตรงกับจำนวนคลาส")


@flow(name="skin-lesion-train-and-deploy")
def train_and_deploy(arch: str = "efficientnet_b0", epochs: int = 20, weighted_loss: bool = True,
                     from_run: str = "", reason: str = "", deploy_dir: str = "API/artifacts/model-v3",
                     skip_data_check: bool = False, train_args: str = ""):
    os.chdir(ROOT)
    if from_run:
        run_dir = from_run  # ใช้ run ที่เทรนไว้แล้ว
    else:
        if not skip_data_check:
            validate_data()
        run_dir = train(arch, epochs, weighted_loss, train_args)
    g = gate(run_dir)
    version = log_and_register(run_dir, g, reason)
    approved = approve(version, g)
    result = {"run_dir": run_dir, "registered_version": version, "approved": approved,
              "passed_gates": g["passed_gates"]}
    if approved:
        result["deployed"] = deploy(deploy_dir)
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--from-run", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--deploy-dir", default="API/artifacts/model-v3")
    ap.add_argument("--skip-data-check", action="store_true")
    ap.add_argument("--train-args", default="", help='ส่งต่อให้ train.py เช่น "--num-workers 0 --batch-size 16"')
    a = ap.parse_args()
    train_and_deploy(a.arch, a.epochs, a.weighted_loss, a.from_run, a.reason, a.deploy_dir, a.skip_data_check, a.train_args)
