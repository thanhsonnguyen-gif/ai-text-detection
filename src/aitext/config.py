"""Cấu hình dùng chung. Chủ file: Người 1 (Thực nghiệm). Người 2 muốn đổi thì mở GitHub Issue.

Quy tắc: KHÔNG hard-code đường dẫn hay tham số trong notebook. Đổi ở đây để mọi thí nghiệm dùng chung
một cấu hình và so sánh được với nhau.
"""
import os

# ---- Đường dẫn -------------------------------------------------------------
# Mỗi thành viên "Add shortcut to My Drive" cho thư mục chung AI-Text-Project,
# để đường dẫn dưới đây giống nhau trên Colab của cả hai người.
# DRIVE_ROOT = os.environ.get("AITEXT_DRIVE_ROOT", "/content/drive/MyDrive/AI-Text-Project")
def _find_drive_root():
    if os.environ.get("AITEXT_DRIVE_ROOT"):
        return os.environ["AITEXT_DRIVE_ROOT"]
    candidates = [
        "/content/drive/MyDrive/BigData_IE212_R11_VB2/AI-Text-Project",  # máy chủ thư mục
        "/content/drive/MyDrive/AI-Text-Project",                        # shortcut của teammate
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Không thấy thư mục AI-Text-Project trên Drive. "
                            "Kiểm tra đã mount Drive và đã 'Add shortcut to My Drive' chưa.")

DRIVE_ROOT = _find_drive_root()

LOCAL_TMP    = "/content/tmp"                      # ổ tạm của Colab (nhanh, mất khi ngắt phiên)
RAW_DIR      = f"{LOCAL_TMP}/raw"                  # CSV tải về để ở ổ tạm, KHÔNG để trên Drive
PARQUET_DIR  = f"{DRIVE_ROOT}/data/parquet"
SPLIT_TRAIN  = f"{PARQUET_DIR}/clean_train"        # dùng để huấn luyện
SPLIT_VAL    = f"{PARQUET_DIR}/clean_val"          # dùng để chọn tham số và ngưỡng
SPLIT_TEST   = f"{PARQUET_DIR}/clean_test"         # CHỈ dùng để báo cáo kết quả cuối
ATTACKED_DIR = f"{PARQUET_DIR}/attacked"           # mẫu bị tấn công thật của RAID (chỉ từ phía test)
MODEL_DIR    = f"{DRIVE_ROOT}/models"
THRESHOLDS   = f"{MODEL_DIR}/thresholds.json"      # regParam đã chọn + ngưỡng vận hành
RESULTS_DIR  = f"{DRIVE_ROOT}/results"             # bảng/hình đặt tên theo mã trong docs/figure_list.md

RAID_URLS = {
    "train_none": "https://dataset.raid-bench.xyz/train_none.csv",   # ~800MB, không tấn công
    "train":      "https://dataset.raid-bench.xyz/train.csv",        # ~11.8GB, có tấn công
}

# ---- Tham số thí nghiệm ----------------------------------------------------
SEED          = 42
SEEDS         = [42, 43, 44]   # số liệu cuối: lặp 3 lần, báo cáo trung bình ± độ lệch chuẩn
AI_FRACTION   = 0.15           # tỷ lệ văn bản AI lấy từ RAID (giữ 100% văn bản người)
TEST_PERCENT  = 20             # % nhóm source_id dành cho test
VAL_PERCENT   = 10             # % nhóm source_id (trong phần còn lại) dành cho validation
NUM_FEATURES  = 1 << 18        # số ô của HashingTF
REG_PARAMS    = [1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0]
DEFAULT_REG   = 1e-2           # dùng khi chưa có thresholds.json
ATTACK_RATE   = 5              # tấn công tự cài: sửa ~1/ATTACK_RATE số từ
FPR_TARGET    = 0.05           # ngưỡng vận hành: chấp nhận báo nhầm 5% văn bản người

RAID_ATTACKS = ["homoglyph", "number", "article_deletion", "insert_paragraphs",
                "perplexity_misspelling", "upper_lower", "whitespace", "zero_width_space",
                "synonym", "paraphrase", "alternative_spelling"]
OLD_GENERATORS = ["gpt2", "gpt3", "mpt", "mpt-chat", "cohere", "cohere-chat"]
NEW_GENERATORS = ["gpt4", "chatgpt", "llama-chat", "mistral", "mistral-chat"]


def ensure_dirs():
    for d in [LOCAL_TMP, RAW_DIR, PARQUET_DIR, MODEL_DIR, RESULTS_DIR, f"{RESULTS_DIR}/figures"]:
        os.makedirs(d, exist_ok=True)
