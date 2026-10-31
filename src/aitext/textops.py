"""Tấn công và chuẩn hóa văn bản.

Bản Spark (hàm native, chạy phân tán) dùng cho thí nghiệm; bản Python dùng cho demo Gradio.
Mọi người dùng chung file này để định nghĩa "tấn công" và "chuẩn hóa" luôn thống nhất.
"""
import hashlib
from pyspark.sql import functions as F

from . import config

LATIN = "aeopcxyiAEOPCXHKMTB"
CYRIL = ("аеорсхуі"
         "АЕОРСХНКМТВ")
assert len(LATIN) == len(CYRIL)
ZW_CLASS = "[​‌‍⁠﻿]"


# ---------------- Spark ----------------
def normalize(c):
    """Phòng thủ: xóa ký tự vô hình, Cyrillic giống Latin -> Latin, gom khoảng trắng, chữ thường."""
    c = F.regexp_replace(c, ZW_CLASS, "")
    c = F.translate(c, CYRIL, LATIN)
    c = F.regexp_replace(c, r"\s+", " ")
    return F.lower(F.trim(c))


def _per_word(c, fn, rate=None):
    rate = rate or config.ATTACK_RATE
    picked = lambda w: (F.abs(F.xxhash64(w)) % rate) == 0
    return F.array_join(F.transform(F.split(c, " "), lambda w: F.when(picked(w), fn(w)).otherwise(w)), " ")


def attack_homoglyph(c, rate=None):
    return _per_word(c, lambda w: F.translate(w, LATIN, CYRIL), rate)


def attack_zero_width(c, rate=None):
    return _per_word(c, lambda w: F.concat(F.substring(w, 1, 2), F.lit("​"), F.substring(w, 3, 100000)), rate)


def attack_both(c, rate=None):
    return attack_zero_width(attack_homoglyph(c, rate), rate)


# ---------------- Python (cho demo) ----------------
_L2C = str.maketrans(LATIN, CYRIL)


def _py_pick(w, rate):
    return int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % rate == 0


def py_attack(text, kind="both", rate=None):
    rate = rate or config.ATTACK_RATE
    hg = lambda w: w.translate(_L2C)
    zw = lambda w: w[:2] + "​" + w[2:]
    def per_word(s, fn):
        return " ".join(fn(w) if _py_pick(w, rate) else w for w in s.split(" "))
    if kind == "homoglyph":
        return per_word(text, hg)
    if kind == "zero_width":
        return per_word(text, zw)
    return per_word(per_word(text, hg), zw)
