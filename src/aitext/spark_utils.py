"""Khởi tạo Spark giống nhau cho mọi notebook."""
import re
import subprocess
import sys


def ensure_pyspark():
    """Dùng PySpark có sẵn trên Colab; chỉ cài 3.5.3 khi thiếu, hoặc khi PySpark 4.x mà Java < 17."""
    jv = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
    m = re.search(r'version "(\d+)', jv)
    java_major = int(m.group(1)) if m else 0
    try:
        import pyspark
        ver = pyspark.__version__
    except ImportError:
        ver = None
    if ver is None or (ver.startswith("4") and java_major < 17):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyspark==3.5.3"], check=True)
    import pyspark
    print(f"Java {java_major} | PySpark {pyspark.__version__}")


def get_spark(app="ai-text-detection", driver_mem="8g", shuffle_partitions=32):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName(app).master("local[*]")
             .config("spark.driver.memory", driver_mem)
             .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark
