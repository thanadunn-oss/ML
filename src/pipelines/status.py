"""status.py — แสดงทุกรุ่นในทะเบียน: version, สถานะ, gate, val Macro F1, alias   ·   python src/pipelines/status.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry

c = registry.client()
aliases = {}
for a in ("champion", "previous"):
    v = registry.get_alias(a)
    if v: aliases.setdefault(v.version, []).append(a)
print(f"{'ver':>4} {'status':12s} {'gate':9s} {'valF1':>7s}  run                                   alias")
for v in sorted(c.search_model_versions(f"name='{registry.MODEL_NAME}'"), key=lambda v: int(v.version)):
    t = v.tags; run = c.get_run(v.run_id).data.tags.get("train_run_id", "")
    print(f"{v.version:>4} {t.get('status',''):12s} {t.get('gate',''):9s} {t.get('val_macro_f1',''):>7s}  {run:37s} {','.join(aliases.get(v.version, []))}")
