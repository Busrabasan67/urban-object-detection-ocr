"""
Nesne Tespiti ve Görsel İşleme Modülü.
YOLO modelini çalıştırır, nesneleri tespit eder, tabelalara OCR uygular
ve görsel üzerine kutu ile metin çizer.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.ocr import OCRReader


# Sınıf isimleri ve görsel renk paleti (BGR)
CLASS_COLORS = {
    'bina': (255, 120, 0),          # Mavi ton
    'cop_konteyneri': (0, 165, 255), # Turuncu
    'lamba_diregi': (0, 215, 255),   # Sarı
    'rogar_kapagi': (180, 105, 255), # Pembe/Mor
    'tabela': (0, 255, 0)            # Yeşil
}

CLASS_TR_NAMES = {
    'bina': 'Bina',
    'cop_konteyneri': 'Çöp Konteyneri',
    'lamba_diregi': 'Lamba Direği',
    'rogar_kapagi': 'Rögar Kapağı',
    'tabela': 'Tabela'
}


class Detector:
    def __init__(self, model_path: str = "models/best.pt", conf_thresh: float = 0.25, use_ocr: bool = True, ocr_engine: str = "paddle"):
        """
        Dedektör sınıfını başlatır.
        """
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.use_ocr = use_ocr
        self.ocr_engine = ocr_engine

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        print(f"[BİLGİ] YOLO modeli yükleniyor: {model_path}")
        self.model = YOLO(model_path)
        self.names = self.model.names

        if self.use_ocr:
            self.ocr_reader = OCRReader(engine=self.ocr_engine, gpu=False)
        else:
            self.ocr_reader = None

    def draw_annotations(self, image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Görsel üzerine Türkçe karakterleri destekleyecek şekilde (PIL yardımıyla)
        bounding box ve etiket bilgilerini çizer. Yazı boyutunu görsel çözünürlüğüne göre
        dinamik ve net okunabilir ayarlar.
        """
        annotated = image_bgr.copy()
        h_img, w_img = annotated.shape[:2]

        # Görsel çözünürlüğüne göre dinamik font boyutu (Panoramik büyük görsellerde net okunması için)
        font_size = max(18, int(min(w_img, h_img) * 0.025))

        # Windows sistem fontu Arial (Türkçe karakterleri tam destekler)
        font = None
        font_paths = [
            "C:/Windows/Fonts/arialbd.ttf",  # Arial Bold
            "C:/Windows/Fonts/arial.ttf",    # Arial Regular
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        for fp in font_paths:
            if Path(fp).exists():
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    pass

        if font is None:
            font = ImageFont.load_default()

        # Çizimleri PIL üzerinde yapalım (Daha keskin yazı ve arka plan için)
        img_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for det in detections:
            x, y, w, h = det["bbox"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            ocr_text = det.get("ocr_text", "")

            # BGR'den RGB renge çevir
            bgr_color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            tr_label = CLASS_TR_NAMES.get(cls_name, cls_name)

            # Bounding box çizimi (kalınlık: görsel boyutuna göre)
            line_w = max(2, int(min(w_img, h_img) * 0.003))
            draw.rectangle([x, y, x + w, y + h], outline=rgb_color, width=line_w)

            # Etiket metni
            label_text = f" {tr_label} %{int(conf * 100)} "
            if ocr_text:
                label_text += f"| {ocr_text} "

            # Yazı boyutunu hesapla
            bbox_text = draw.textbbox((x, y), label_text, font=font)
            text_w = bbox_text[2] - bbox_text[0]
            text_h = bbox_text[3] - bbox_text[1]

            tag_y1 = max(0, y - text_h - 6)
            tag_y2 = tag_y1 + text_h + 6
            tag_x1 = x
            tag_x2 = x + text_w + 6

            # Yazı arka plan kutusu (dolgulu)
            draw.rectangle([tag_x1, tag_y1, tag_x2, tag_y2], fill=rgb_color)
            # Metni yaz (kontrastlı beyaz renk)
            draw.text((tag_x1 + 3, tag_y1 + 2), label_text, fill=(255, 255, 255), font=font)

        annotated = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return annotated

    def process_image(self, image_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray, float]:
        """
        Tek bir görseli okur, tespitleri yapar, OCR uygular ve çizilmiş görseli döner.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Görsel okunamadı veya desteklenmeyen format: {image_path}")

        img_h, img_w = image.shape[:2]
        start_time = time.time()

        # YOLO tahmini (1280 çözünürlük)
        results = self.model.predict(source=image, imgsz=1280, conf=self.conf_thresh, verbose=False)
        inference_time = round(time.time() - start_time, 4)

        detections = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())

                # x1, y1, x2, y2 formatından x, y, w, h formatına
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = max(0, int(x1)), max(0, int(y1))
                w = min(img_w - x, int(x2 - x1))
                h = min(img_h - y, int(y2 - y1))

                ocr_text = ""
                # Eğer sınıf tabela ise ve OCR etkinse OCR çalıştır
                if cls_name == "tabela" and self.use_ocr and self.ocr_reader is not None:
                    # Kenardaki harflerin kesilmemesi için %8 pay ekle
                    pad_w = int(w * 0.08)
                    pad_h = int(h * 0.08)
                    crop_y1 = max(0, y - pad_h)
                    crop_y2 = min(img_h, y + h + pad_h)
                    crop_x1 = max(0, x - pad_w)
                    crop_x2 = min(img_w, x + w + pad_w)

                    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop.size > 0:
                        ocr_text = self.ocr_reader.read_text_from_crop(crop)

                detection_info = {
                    "image_name": path.name,
                    "class_name": cls_name,
                    "class_label_tr": CLASS_TR_NAMES.get(cls_name, cls_name),
                    "confidence": round(conf, 4),
                    "bbox": [x, y, w, h],
                    "ocr_text": ocr_text,
                    "detection_date": now_str,
                    "inference_time_sec": inference_time
                }
                detections.append(detection_info)

        # Görsel üzerine çizim yap
        annotated_img = self.draw_annotations(image, detections)

        return detections, annotated_img, inference_time

