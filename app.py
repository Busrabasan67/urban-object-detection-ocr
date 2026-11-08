"""
Yapay Zeka ile Nesne Tanıma ve OCR Web Arayüzü (Gradio)
"""

import os
import json
import csv
import tempfile
from datetime import datetime
from pathlib import Path
import cv2
import gradio as gr
import numpy as np
import pandas as pd

from src.detect import Detector

# Modeli başlat (Global tek örnek, PaddleOCR varsayılan)
print("[BİLGİ] Web arayüzü başlatılıyor, dedektör yükleniyor...")
detector = Detector(model_path="models/best.pt", conf_thresh=0.25, use_ocr=True, ocr_engine="paddle")


def predict(image: np.ndarray, conf_threshold: float, enable_ocr: bool, ocr_engine_choice: str = "PaddleOCR (Önerilen)", progress=gr.Progress()):
    """
    Kullanıcının yüklediği görseli işler.
    Görsel Gradio'dan RGB formatında gelir.
    """
    if image is None:
        return None, pd.DataFrame(), "Lütfen bir görsel yükleyin."

    progress(0.2, desc="Görsel hazırlanıyor...")
    # Gradio RGB verir, OpenCV BGR bekler
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Geçici dosyaya yazıp detector üzerinden işleyelim
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image_bgr)

    try:
        detector.conf_thresh = conf_threshold
        detector.use_ocr = enable_ocr
        target_engine = "paddle" if "paddle" in ocr_engine_choice.lower() else "easyocr"
        if detector.ocr_reader is not None and detector.ocr_reader.engine != target_engine:
            from src.ocr import OCRReader
            detector.ocr_reader = OCRReader(engine=target_engine, gpu=False)

        progress(0.5, desc="Nesneler tespit ediliyor ve tabela yazıları okunuyor...")
        detections, annotated_bgr, inf_time = detector.process_image(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    progress(1.0, desc="Tamamlandı!")

    # annotated_bgr -> RGB
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Tespit edilen görseli output/annotated_images klasörüne kaydet
    annotated_dir = Path("output/annotated_images")
    annotated_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_img_name = f"web_detected_{timestamp}.jpg"
    cv2.imwrite(str(annotated_dir / saved_img_name), annotated_bgr)

    # Sonuçları output/results.json ve output/results.csv dosyalarına kaydet
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"

    # JSON güncelle / ekle
    existing_json = []
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                existing_json = json.load(jf)
        except Exception:
            existing_json = []

    # Görsel adını web yüklemesi için güncelle
    for d in detections:
        d["image_name"] = saved_img_name

    existing_json.extend(detections)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(existing_json, jf, ensure_ascii=False, indent=4)

    # CSV güncelle / ekle
    file_exists = csv_path.exists()
    fieldnames = [
        "image_name", "class_name", "class_label_tr", "confidence",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "ocr_text",
        "detection_date", "inference_time_sec"
    ]
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            writer.writerow({
                "image_name": d.get("image_name"),
                "class_name": d.get("class_name"),
                "class_label_tr": d.get("class_label_tr"),
                "confidence": d.get("confidence"),
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2],
                "bbox_h": bbox[3],
                "ocr_text": d.get("ocr_text", ""),
                "detection_date": d.get("detection_date"),
                "inference_time_sec": d.get("inference_time_sec")
            })

    # Tablo formatında sonuç hazırla
    table_data = []
    for d in detections:
        x, y, w, h = d["bbox"]
        table_data.append({
            "Sınıf": d["class_label_tr"],
            "Güven Skoru": f"%{int(d['confidence'] * 100)}",
            "Tabela OCR Metni": d["ocr_text"] if d["ocr_text"] else "-",
            "Konum (x, y, w, h)": f"({x}, {y}, {w}, {h})"
        })

    df = pd.DataFrame(table_data)

    summary_text = (
        f"**Toplam Tespit Edilen Nesne:** {len(detections)}\n\n"
        f"**Tahmin Süresi:** {inf_time:.3f} saniye\n\n"
        f"**Güven Eşiği:** %{int(conf_threshold * 100)}\n\n"
        f"💾 *Sonuçlar `output/results.json` ve `output/results.csv` dosyalarına kaydedildi.*"
    )

    return annotated_rgb, df, summary_text


# Gradio Arayüz Tasarımı
custom_css = """
#main-container { max-width: 1200px; margin: auto; }
.output-image { min-height: 400px; }
"""

with gr.Blocks(title="Kentsel Nesne Tanıma & OCR") as demo:
    # Alt bilgi çubuğunu gizle ve sıfırlama ikonuna açıklayıcı ipucu ekle
    gr.HTML("""
    <style>
    footer, .footer { display: none !important; }
    /* Sıfırlama ikonuna fareyle gelindiğinde açıklama çıksın */
    button[aria-label="Reset value"], button[title="Reset value"] {
        cursor: pointer;
    }
    </style>
    """)

    gr.Markdown(
        """
        # Yapay Zeka ile Kentsel Nesne Tanıma ve Tabela Okuma
        Bu web uygulaması panoramik ve sokak görselleri üzerindeki kentsel nesneleri 
        (**Bina**, **Çöp Konteyneri**, **Lamba Direği**, **Rögar Kapağı**, **Tabela**) 
        tespit eder ve tabela üzerindeki metinleri **PaddleOCR** ile yüksek doğrulukla okur.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="numpy",
                sources=["upload"],
                label="Görsel Yükleyin / Sürükleyin"
            )
            with gr.Row():
                btn_clear = gr.Button("Ekranı Temizle", variant="secondary", size="sm")

            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Tespit Hassasiyeti / Güven Eşiği (Varsayılan: %25)",
               
            )
            ocr_checkbox = gr.Checkbox(value=True, label="Tabela OCR Okumasını Etkinleştir")
            ocr_engine_dropdown = gr.Dropdown(
                choices=["PaddleOCR (Önerilen)", "EasyOCR"],
                value="PaddleOCR (Önerilen)",
                label="OCR Motoru Seçimi"
            )
            btn_run = gr.Button("Nesneleri Tespit Et", variant="primary")

            summary_box = gr.Markdown("Sonuçlar burada görüntülenecek.")

        with gr.Column(scale=2):
            output_image = gr.Image(type="numpy", label="Tespit Edilmiş Görsel (Bounding Box & Etiket)")
            results_table = gr.Dataframe(
                headers=["Sınıf", "Güven Skoru", "Tabela OCR Metni", "Konum (x, y, w, h)"],
                label="Tespit Edilen Nesneler ve Tabela Yazıları",
                interactive=False
            )

    btn_run.click(
        fn=predict,
        inputs=[input_image, conf_slider, ocr_checkbox, ocr_engine_dropdown],
        outputs=[output_image, results_table, summary_box],
        show_progress_on=[output_image]
    )

    btn_clear.click(
        fn=lambda: (None, None, [], "Görsel temizlendi. Yeni bir görsel yükleyip 'Nesneleri Tespit Et' butonuna basabilirsiniz."),
        inputs=[],
        outputs=[input_image, output_image, results_table, summary_box]
    )

    # Örnek görseller ekle (Kullanıcı kolayca tıklayıp test edebilsin)
    example_imgs = [
        "data/images/val/pano_00027_jpg.rf.6401a47126072afcf67d2d64a23089f5.jpg",
        "data/images/val/pano_00401_jpg.rf.e443ace747578f541885dcaee77b8eb2.jpg",
        "data/images/val/pano_02622_jpg.rf.0c7b1c6d42227994455ab75cd7b4befa.jpg",
        "data/images/train/pano_00654_jpg.rf.6a5d722de3608611d795e30fbaed1ee1.jpg"
    ]
    existing_examples = [[p, 0.25, True, "PaddleOCR (Önerilen)"] for p in example_imgs if os.path.exists(p)]

    if existing_examples:
        gr.Examples(
            examples=existing_examples,
            inputs=[input_image, conf_slider, ocr_checkbox, ocr_engine_dropdown],
            outputs=[output_image, results_table, summary_box],
            fn=predict,
            cache_examples=False,
            label="Hızlı Test İçin Örnek Görseller"
        )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )

