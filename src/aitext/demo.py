"""Giao diện Gradio. Chủ file: Người 2 (Nội dung & trình bày)."""
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from . import config
from .textops import normalize, py_attack

EXAMPLE_AI = ("Troy is an epic historical war film directed by Wolfgang Petersen. Inspired by ancient Greek "
              "mythology, the film tells a story of love, honor, ambition, and war. I enjoy this kind of movie "
              "because it combines exciting battles with stories about human emotions and difficult choices. "
              "Overall, I think Troy is an exciting and memorable film because it combines action with meaningful "
              "questions about honor, love, and the cost of war. It also reminds me that the desire for fame and "
              "glory can lead to tragic consequences. I would recommend it to anyone who enjoys epic historical movies.")


def launch(spark, model_base, model_def, thr_base, thr_def, example=EXAMPLE_AI, share=True):
    """thr_base / thr_def: ngưỡng vận hành từ thresholds.json (chọn trên validation)."""
    import gradio as gr

    def prob(model, text, defended):
        d = spark.createDataFrame([(text,)], ["generation"])
        col = normalize(F.col("generation")) if defended else F.col("generation")
        return model.transform(d.withColumn("text_in", col)).select(vector_to_array("probability")[1]).first()[0]

    def detect(text):
        pb, pdf_ = prob(model_base, text, False), prob(model_def, text, True)
        v = lambda p, t: f"**{'AI' if p > t else 'Người'}** (điểm {p:.3f}, ngưỡng {t:.3f})"
        note = (f"Kết luận theo ngưỡng vận hành (chấp nhận báo nhầm {config.FPR_TARGET:.0%} văn bản người):\n\n"
                f"- Baseline: {v(pb, thr_base)}\n- Defended: {v(pdf_, thr_def)}")
        n = len(text.split())
        if n < 80:
            note += f"\n\n⚠️ Văn bản chỉ có {n} từ; dưới khoảng 80 từ kết quả kém tin cậy."
        return {"AI": pb, "Người": 1 - pb}, {"AI": pdf_, "Người": 1 - pdf_}, note

    kinds = {"Homoglyph": "homoglyph", "Zero-width": "zero_width", "Cả hai": "both"}
    with gr.Blocks(title="AI-text detector") as ui:
        gr.Markdown("### Phát hiện văn bản do AI viết: baseline và bản có chuẩn hóa")
        inp = gr.Textbox(lines=8, label="Văn bản (tiếng Anh)", value=example)
        with gr.Row():
            kind = gr.Radio(list(kinds), value="Cả hai", label=f"Kiểu tấn công (~{100 // config.ATTACK_RATE}% số từ)")
            b_atk = gr.Button("Tấn công văn bản này")
            b_go = gr.Button("Chấm điểm", variant="primary")
        with gr.Row():
            out_b = gr.Label(label="Baseline (không phòng thủ)")
            out_d = gr.Label(label="Defended (có chuẩn hóa)")
        verdict = gr.Markdown()
        b_atk.click(lambda t, k: py_attack(t, kinds[k]), [inp, kind], inp)
        b_go.click(detect, inp, [out_b, out_d, verdict])
    ui.launch(share=share, debug=False)
    return ui
