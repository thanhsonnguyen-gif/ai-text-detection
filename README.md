# Phát hiện văn bản do AI sinh ra trên quy mô lớn, có khả năng chống tấn công

Đồ án môn Công nghệ dữ liệu lớn · Dữ liệu: [RAID](https://github.com/liamdugan/raid) · Công cụ: Apache Spark (PySpark) trên Google Colab

## Phân công (chia theo chương báo cáo)

| | **Người 1: Thực nghiệm** | **Người 2: Nội dung & trình bày** |
|---|---|---|
| Chương | 4, 5 | 1, 2, 3, 6, tài liệu tham khảo; ghép báo cáo |
| Code | toàn bộ `src/aitext/` (trừ `demo.py`), notebook `01`, `E1`–`E5` | `src/aitext/demo.py`, notebook `00`, slide, video |
| Drive | ghi `data/`, `models/`, `results/` | chỉ đọc `data/`, `models/`, `results/`; chủ file `report/BaoCao.docx` |
| Hợp đồng | tạo đúng bảng/hình trong `docs/figure_list.md` | viết `docs/figure_list.md`, `docs/report_outline.md` |

## Cấu trúc

```
src/aitext/      config · spark_utils · data · textops · models · evaluate   [Người 1]   demo  [Người 2]
notebooks/       01_data_prep · E1 … E5                                      [Người 1]   00_quick_demo [Người 2]
docs/            figure_list.md (hợp đồng) · report_outline.md · meeting_notes_template.md
.github/         mẫu Issue "Yêu cầu kết quả"
results/         bảng & hình CHỐT (chép từ Drive sau mốc đóng băng 06/11)
```

Google Drive `AI-Text-Project/`:
```
data/parquet/   clean_train · clean_val · clean_test · attacked/
models/         lr_base · lr_def · thresholds.json   (regParam + ngưỡng vận hành, do E1 ghi)
results/        runs.csv · T*.csv · *_notes.md · figures/F*.png
report/         BaoCao.docx
```

## Thứ tự phụ thuộc

`01_data_prep` → `E1_baseline` (tạo mô hình + thresholds.json) → `E2` và `00_quick_demo`. `E3`, `E4`, `E5` chạy được ngay sau `01`.

## Thiết lập (mỗi người làm một lần)

1. **Drive:** Người 1 tạo thư mục `AI-Text-Project` (bên trong tạo `report/`), chia sẻ quyền **Editor** cho Người 2.
   Cả hai: chuột phải vào thư mục → **Organize → Add shortcut → My Drive**. Trong Colab đường dẫn luôn là `/content/drive/MyDrive/AI-Text-Project`.
2. **GitHub:** Người 1 tạo repo `ai-text-detection` (private), đẩy toàn bộ bộ khung lên, mời Người 2 làm Collaborator.
3. **Token cho Colab (repo private):** GitHub → Settings → Developer settings → Personal access tokens → Fine-grained token,
   chọn repo `ai-text-detection`, quyền *Contents: Read and write*. Trong Colab: biểu tượng chìa khóa (Secrets) → thêm `GH_TOKEN`, bật "Notebook access".
4. **Colab ↔ GitHub:** `File → Open notebook → GitHub`, đăng nhập GitHub, chọn repo, mở notebook của mình.
5. Sửa `OWNER` và `REPO` trong cell BOOTSTRAP đầu notebook, chạy cell đó.

## Quy tắc làm việc

1. **Một notebook một chủ.** Chỉ chủ notebook sửa và lưu notebook đó (file .ipynb gần như không gộp được).
2. **Code dùng chung chỉ nằm trong `src/aitext/`.** Không copy hàm ra notebook để sửa riêng. Người 2 cần thay đổi thì mở Issue.
3. **Không chạy lại `01_data_prep`** khi đã có dữ liệu chung, trừ khi cả hai đồng ý.
4. **Tập test chỉ dùng để báo cáo.** Chọn tham số và ngưỡng luôn trên `clean_val`.
5. **Mọi kết quả** ghi qua `evaluate.log_run(...)`; bảng/hình lưu bằng `evaluate.save_table/save_figure` với **mã trong `figure_list.md`**.
6. **Lưu notebook:** `File → Save a copy in GitHub`, commit message có mã: `E2: thêm T5.4`.
7. **Mốc đóng băng 06/11:** sau ngày này không chạy thí nghiệm mới.
