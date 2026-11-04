"""Đánh giá theo chuẩn RAID, lưu bảng/hình theo mã, nhật ký runs.csv. Chủ file: Người 1."""
import csv
import datetime as dt
import os

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from . import config

_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score", metricName="areaUnderROC")


def score(model, df, text_col, logodds=False):
    """Chấm điểm. text_col: biểu thức cột văn bản đưa vào mô hình (gốc / bị tấn công / đã chuẩn hóa).
    logodds=True: điểm = log-odds từ rawPrediction thay cho xác suất. Dùng cho Naive Bayes: xác suất
    của NB bão hòa đúng 1,0 (E4: 27% văn bản người trên val) làm ngưỡng FPR 5% thành 1,0 và TPR = 0."""
    out = model.transform(df.withColumn("text_in", text_col))
    if logodds:
        raw = vector_to_array("rawPrediction")
        s = raw[1] - raw[0]
    else:
        s = vector_to_array("probability")[1]
    return out.withColumn("score", s).select("label", "model", "domain", "attack", "decoding", "score")


def thr_at_fpr(pred, fpr=None):
    """Ngưỡng sao cho chỉ `fpr` văn bản người bị chấm cao hơn. Chỉ nên tính trên VALIDATION."""
    return pred.filter("label = 0").approxQuantile("score", [1 - (fpr or config.FPR_TARGET)], 0.001)[0]


def rate_above(pred, label, thr):
    """Tỷ lệ văn bản có nhãn `label` được chấm > thr. label=1 → TPR; label=0 → FPR."""
    return pred.filter(f"label = {label}").select(F.avg((F.col("score") > thr).cast("double"))).first()[0]


def auroc(pred):
    return _auc.evaluate(pred) if pred.select("label").distinct().count() == 2 else float("nan")


def metrics(pred, thr=None, boundary=0.5):
    """AUROC; TPR/FPR tại ngưỡng vận hành `thr` (lấy từ validation); FPR tại ranh giới mặc định; TPR@FPR5% kiểu RAID.
    boundary: ranh giới mặc định của bộ phân loại — 0,5 với xác suất, 0 với log-odds (tương đương p = 0,5).
    Cột kết quả vẫn tên "fpr_at_0.5" để các bảng cũ không đổi."""
    pred = pred.cache()
    res = {"auroc": auroc(pred), "fpr_at_0.5": rate_above(pred, 0, boundary),
           "tpr_at_fpr_oracle": rate_above(pred, 1, thr_at_fpr(pred)), "n": pred.count()}
    if thr is not None:
        res.update({"threshold": thr, "tpr": rate_above(pred, 1, thr), "fpr": rate_above(pred, 0, thr)})
    pred.unpersist()
    return res


def auc_by(pred, col):
    """AUROC theo từng nhóm (vd. từng LLM): toàn bộ văn bản người so với văn bản AI của nhóm đó."""
    human = pred.filter("label = 0")
    rows = []
    for v in [r[col] for r in pred.filter("label = 1").select(col).distinct().collect()]:
        sub = human.unionByName(pred.filter((F.col("label") == 1) & (F.col(col) == v)))
        rows.append({col: v, "auroc": _auc.evaluate(sub)})
    return rows


# ---- Lưu bảng / hình theo mã trong docs/figure_list.md ----
def save_table(df_pandas, code, notes=""):
    """Lưu bảng pandas thành results/<code>.csv (+ <code>_notes.md nếu có ghi chú)."""
    config.ensure_dirs()
    df_pandas.to_csv(f"{config.RESULTS_DIR}/{code}.csv", index=False)
    if notes:
        with open(f"{config.RESULTS_DIR}/{code}_notes.md", "w", encoding="utf-8") as f:
            f.write(notes)


def save_figure(fig, code):
    config.ensure_dirs()
    fig.savefig(f"{config.RESULTS_DIR}/figures/{code}.png", dpi=170, bbox_inches="tight")


def log_run(experiment, owner, setting, result, notes=""):
    """Ghi 1 dòng vào results/runs.csv trên Drive để cả nhóm xem mọi lần chạy."""
    config.ensure_dirs()
    path = f"{config.RESULTS_DIR}/runs.csv"
    keys = ["auroc", "tpr", "fpr", "threshold", "fpr_at_0.5", "tpr_at_fpr_oracle", "n"]
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "experiment", "owner", "setting", *keys, "notes"])
        vals = [round(result[k], 4) if isinstance(result.get(k), float) else result.get(k, "") for k in keys]
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), experiment, owner, setting, *vals, notes])
