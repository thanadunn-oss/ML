"""
rollback.py — ย้อนกลับไปใช้รุ่นก่อนหน้า (alias "previous") แล้ววางให้ API ใหม่

    python src/pipelines/rollback.py
    python src/pipelines/rollback.py --to-version 2    # เลือกรุ่นเอง

ใช้เมื่อตรวจหลังนำขึ้นใช้งานแล้วไม่ผ่าน · หลังรันต้องเริ่ม API ใหม่
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry
from flow import deploy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-version")
    ap.add_argument("--deploy-dir", default="API/artifacts/model-v3")
    a = ap.parse_args()
    cur = registry.get_alias("champion")
    target = a.to_version or (registry.get_alias("previous").version if registry.get_alias("previous") else None)
    if not target:
        sys.exit("ไม่มีรุ่นก่อนหน้าให้ย้อน")
    c = registry.client()
    c.set_registered_model_alias(registry.MODEL_NAME, "champion", target)
    c.set_model_version_tag(registry.MODEL_NAME, target, "status", "champion")
    if cur is not None and cur.version != str(target):
        c.set_registered_model_alias(registry.MODEL_NAME, "previous", cur.version)
        c.set_model_version_tag(registry.MODEL_NAME, cur.version, "status", "rolled_back")
    print(f"ย้อน champion: version {cur.version if cur else None} → {target}")
    print(deploy.fn(a.deploy_dir))


if __name__ == "__main__":
    main()
