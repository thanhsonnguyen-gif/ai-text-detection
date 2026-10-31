# Danh mục bảng và hình (hợp đồng giữa Người 1 và Người 2)

- **Người 2** viết / chỉnh danh mục này (cần những bảng, hình nào để kể câu chuyện của báo cáo).
- **Người 1** duyệt (có làm kịp không), rồi tạo **đúng** các file dưới đây trên Drive: `AI-Text-Project/results/<mã>.csv`
  và `results/figures/<mã>.png`, kèm `<mã>_notes.md` (cấu hình + 3–5 câu nhận xét) nếu cần.
- Muốn thêm hoặc đổi một mục: mở GitHub Issue "Yêu cầu kết quả", không tự sửa notebook của người kia.

| Mã | Nội dung | Notebook | Chương | Hạn | Trạng thái |
|---|---|---|---|---|---|
| T4.1_* | Thống kê dữ liệu: số mẫu theo tập × nhãn, theo LLM, theo lĩnh vực, độ dài văn bản | 01 | 4.1 | 01/11 | ☐ |
| F4.1_lengths | Phân bố độ dài văn bản người / AI | 01 | 4.1 | 01/11 | ☐ |
| T4.2_scaling / F4.2_scaling | Thời gian huấn luyện theo số dòng | E5 | 4.3 | 06/11 | ☐ |
| T4.3_csv_vs_parquet | Thời gian đọc CSV và Parquet | E5 | 4.3 | 06/11 | ☐ |
| T5.1_regparam | Dò regParam trên validation | E1 | 5.1 | 03/11 | ☐ |
| T5.2_clean_mean_std | Kết quả trên test (3 seed, TB ± độ lệch chuẩn): AUROC, TPR, FPR tại ngưỡng vận hành, FPR@0,5 | E1 | 5.1 | 03/11 | ☐ |
| F5.1_auc_by_model | AUROC theo từng LLM | E1 | 5.1 | 03/11 | ☐ |
| F5.2_auc_by_domain | AUROC theo lĩnh vực | E1 | 5.1 | 03/11 | ☐ |
| T5.3_auc_by_decoding, T5.3b_fpr_by_domain | AUROC theo cách giải mã; báo nhầm theo lĩnh vực | E1 | 5.1 | 03/11 | ☐ |
| F5.3_score_hist | Phân bố điểm người / AI và ngưỡng | E1 | 5.1 | 03/11 | ☐ |
| T5.4_raid_attacks / F5.4_raid_attacks | TPR, FPR theo 11 tấn công của RAID, baseline và defended | E2 | 5.2 | 05/11 | ☐ |
| F5.5_attack_strength | TPR theo cường độ tấn công tự cài | E2 | 5.2 | 05/11 | ☐ |
| T5.5_unseen_generators | AUROC, TPR trên LLM đã thấy và chưa thấy | E3 | 5.3 | 05/11 | ☐ |
| T5.6_model_compare | LR, NB (, RoBERTa): độ chính xác, thời gian, phần cứng | E4 | 5.4 | 06/11 | ☐ |
| Ảnh chụp demo | Gradio trước / sau tấn công | 00 | 5.5 | 06/11 | ☐ |
