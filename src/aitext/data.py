"""Tải, lưu trữ và chia dữ liệu RAID. Chủ file: Người 1.

Các hàm prepare_* chỉ chạy MỘT LẦN (notebook 01). Mọi notebook khác chỉ gọi load_*, để cả nhóm dùng
chung một tập train / val / test.
"""
import os
import subprocess

from pyspark.sql import functions as F

from . import config

COLS = ["id", "source_id", "model", "decoding", "repetition_penalty", "attack", "domain", "generation", "label"]


def download(name="train_none"):
    """Tải CSV của RAID về ổ tạm của Colab; trả về đường dẫn file."""
    config.ensure_dirs()
    out = f"{config.RAW_DIR}/{name}.csv"
    if not os.path.exists(out):
        subprocess.run(["wget", "-q", "--show-progress", "-O", out, config.RAID_URLS[name]], check=True)
    return out


def read_raid_csv(spark, path):
    df = (spark.read.option("header", True).option("multiLine", True)
          .option("escape", '"').option("quote", '"').csv(path))
    df = df.filter(F.col("generation").isNotNull() & F.col("model").isNotNull())
    return df.withColumn("label", F.when(F.col("model") == "human", 0).otherwise(1).cast("int"))


def _group():
    return F.coalesce(F.col("source_id"), F.col("id"))


def add_split(df):
    """Gán cột split ∈ {train, val, test} theo hash của source_id.

    Mọi bản (người + AI) của cùng một văn bản gốc nằm trọn trong một tập, nên không rò rỉ chủ đề.
    """
    h_test = F.abs(F.hash(_group())) % 100
    h_val = F.abs(F.hash(F.concat_ws("|", _group(), F.lit("val")))) % 100
    return df.withColumn("split", F.when(h_test < config.TEST_PERCENT, "test")
                                   .when(h_val < config.VAL_PERCENT, "val").otherwise("train"))


def prepare_clean_split(spark, csv_path, ai_fraction=None):
    """CSV không tấn công → lấy mẫu phân tầng → chia train/val/test → ghi Parquet lên Drive."""
    ai_fraction = ai_fraction or config.AI_FRACTION
    df = read_raid_csv(spark, csv_path)
    df = df.sampleBy("label", fractions={0: 1.0, 1: ai_fraction}, seed=config.SEED)
    df = add_split(df).select(*COLS, "split").cache()
    for name, path, parts in [("train", config.SPLIT_TRAIN, 16), ("val", config.SPLIT_VAL, 4), ("test", config.SPLIT_TEST, 8)]:
        df.filter(F.col("split") == name).drop("split").repartition(parts).write.mode("overwrite").parquet(path)
    df.unpersist()


def prepare_attacked_sample(spark, csv_path, per_attack_fraction=0.02):
    """CSV đầy đủ (có tấn công) → chỉ giữ văn bản thuộc tập TEST → lấy mẫu mỗi kiểu tấn công → Parquet."""
    df = add_split(read_raid_csv(spark, csv_path)).filter(F.col("split") == "test")
    df = df.filter(F.col("attack") != "none")
    df = df.sampleBy("attack", fractions={a: per_attack_fraction for a in config.RAID_ATTACKS}, seed=config.SEED)
    df.select(*COLS).repartition(16).write.mode("overwrite").partitionBy("attack").parquet(config.ATTACKED_DIR)


def load_splits(spark):
    """Trả về (train, val, test)."""
    r = spark.read.parquet
    return r(config.SPLIT_TRAIN), r(config.SPLIT_VAL), r(config.SPLIT_TEST)


def load_attacked(spark, attack=None):
    df = spark.read.parquet(config.ATTACKED_DIR)
    return df.filter(F.col("attack") == attack) if attack else df


def with_class_weight(df):
    """Thêm cột w cân bằng lớp (văn bản người ít hơn văn bản AI nhiều lần)."""
    cnt = {r["label"]: r["count"] for r in df.groupBy("label").count().collect()}
    n = cnt.get(0, 0) + cnt.get(1, 0)
    w0, w1 = n / (2 * cnt[0]), n / (2 * cnt[1])
    return df.withColumn("w", F.when(F.col("label") == 0, w0).otherwise(w1))
