"""
metrics.py — คำนวณ Precision / Recall / F1 / Confusion matrix ด้วย numpy ล้วน

ทำไมไม่ใช้ scikit-learn: บน Windows 11 ที่เปิด Smart App Control ไฟล์ DLL ของ scipy (ที่ sklearn ต้องใช้)
ถูกบล็อก → import sklearn ไม่ได้ จึงเขียนสูตรเองให้ผลเท่ากับ sklearn (zero_division=0)
"""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1           # แถว = คำตอบจริง, คอลัมน์ = ที่โมเดลทาย
    return cm


def per_class_prf(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    pred_count = cm.sum(axis=0)           # ทายว่าเป็นคลาสนี้กี่ครั้ง
    true_count = cm.sum(axis=1)           # มีคลาสนี้จริงกี่ภาพ (support)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_count > 0, tp / pred_count, 0.0)
        recall = np.where(true_count > 0, tp / true_count, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1, true_count, cm


def macro_f1(y_true, y_pred, n_classes):
    return float(per_class_prf(y_true, y_pred, n_classes)[2].mean())  # เฉลี่ยทุกคลาสเท่ากัน
