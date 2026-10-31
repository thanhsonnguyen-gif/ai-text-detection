"""Pipeline Spark ML dùng chung + lưu / tải mô hình và ngưỡng. Chủ file: Người 1."""
import json
import os

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.feature import IDF, HashingTF, NGram, RegexTokenizer, VectorAssembler
from pyspark.sql import functions as F

from . import config
from .textops import normalize


def build_pipeline(clf="lr", reg_param=None, num_features=None):
    num_features = num_features or config.NUM_FEATURES
    rp = reg_param if reg_param is not None else best_reg_param()
    tok = RegexTokenizer(inputCol="text_in", outputCol="tok", pattern=r"\s+", toLowercase=True)
    bi = NGram(n=2, inputCol="tok", outputCol="bi")
    tf1 = HashingTF(inputCol="tok", outputCol="tf1", numFeatures=num_features)
    tf2 = HashingTF(inputCol="bi", outputCol="tf2", numFeatures=num_features)
    idf1 = IDF(inputCol="tf1", outputCol="x1", minDocFreq=3)
    idf2 = IDF(inputCol="tf2", outputCol="x2", minDocFreq=3)
    vec = VectorAssembler(inputCols=["x1", "x2"], outputCol="features")
    if clf == "lr":
        model = LogisticRegression(featuresCol="features", labelCol="label", weightCol="w", maxIter=30, regParam=rp)
    elif clf == "nb":
        model = NaiveBayes(featuresCol="features", labelCol="label", weightCol="w", modelType="multinomial")
    else:
        raise ValueError(clf)
    return Pipeline(stages=[tok, bi, tf1, tf2, idf1, idf2, vec, model])


def text_col(defended):
    """Cột đầu vào: văn bản gốc (baseline) hoặc đã chuẩn hóa (defended)."""
    g = F.col("generation")
    return normalize(g) if defended else g


def fit(train_df, defended, clf="lr", reg_param=None):
    return build_pipeline(clf, reg_param).fit(train_df.withColumn("text_in", text_col(defended)))


def save(model, name):
    model.write().overwrite().save(f"{config.MODEL_DIR}/{name}")


def load(name):
    return PipelineModel.load(f"{config.MODEL_DIR}/{name}")


# ---- thresholds.json: regParam đã chọn + ngưỡng vận hành (do E1 ghi) ----
def save_thresholds(**kw):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    cur = load_thresholds()
    cur.update(kw)
    with open(config.THRESHOLDS, "w") as f:
        json.dump(cur, f, indent=2)
    return cur


def load_thresholds():
    if os.path.exists(config.THRESHOLDS):
        with open(config.THRESHOLDS) as f:
            return json.load(f)
    return {}


def best_reg_param():
    return load_thresholds().get("reg_param", config.DEFAULT_REG)
