"""
OCR Modülü: Tabela bölgelerinden metin okuma ve temizleme işlemleri.
PaddleOCR (varsayılan) ve EasyOCR desteği ile Türkçe tabela yazılarını ayıklar.
"""

import re
import cv2
import numpy as np


class OCRReader:
    def __init__(self, engine: str = 'paddle', languages=['tr', 'en'], gpu: bool = False):
        """
        OCR okuyucusunu başlatır.
        
        Args:
            engine: 'paddle' (önerilen, varsayılan) veya 'easyocr'
            languages: EasyOCR için diller
            gpu: GPU kullanımı
        """
        self.engine = engine.lower()
        self.gpu = gpu
        self.languages = languages
        self.reader = None

        if self.engine == 'paddle':
            try:
                from paddleocr import PaddleOCR
                print("[BİLGİ] PaddleOCR motoru başlatılıyor (lang='latin', use_angle_cls=True)...")
                # PaddleOCR Latin modeli Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) ve açı düzeltmeyi tam destekler
                self.reader = PaddleOCR(
                    use_angle_cls=True,
                    lang='latin',
                    show_log=False,
                    use_gpu=gpu
                )
                print("[BAŞARILI] PaddleOCR hazır.")
            except Exception as e:
                print(f"[UYARI] PaddleOCR başlatılamadı ({e}), EasyOCR'a dönülüyor...")
                self.engine = 'easyocr'

        if self.engine == 'easyocr':
            try:
                import easyocr
                print(f"[BİLGİ] EasyOCR başlatılıyor (languages={languages})...")
                self.reader = easyocr.Reader(languages, gpu=gpu)
                print("[BAŞARILI] EasyOCR hazır.")
            except Exception as e:
                print(f"[HATA] EasyOCR başlatılamadı: {e}")
                self.reader = None

    def preprocess_image(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        OCR kalitesini artırmak için tabela görseline ön işleme uygular:
        - Boyut büyütme (Düşük çözünürlüklü tabelalar için süper-örnekleme)
        - Gri tona çevirme
        - Gürültü azaltma (Bilateral Filter: Kenarları korur)
        - Kontrast iyileştirme (CLAHE)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        img = crop_bgr.copy()
        h, w = img.shape[:2]

        # Görsel boyutunu gerekirse büyüt (minimum yükseklik/genişlik 80px olmalı)
        if h < 80 or w < 80:
            scale = max(2.0, 128.0 / max(h, w))
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Gri tona çevir
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Bilateral filtre: Harf kenarlarını korurken zemin doku gürültüsünü giderir
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Kontrast artırma (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        # 3 kanallı BGR formata geri çevir
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

    def clean_text(self, text: str) -> str:
        """
        Okunan metni temizler ve OCR karakter hatalarını düzeltir:
        - $ -> Ş dönüşümü (PaddleOCR Latin modelinde Ş bazen $ olarak çıkar)
        - Fazla boşlukları ve gereksiz özel sembolleri ayıklar
        """
        if not text:
            return ""

        # Sık rastlanan model sembol eşleşmeleri
        text = text.replace("$", "Ş")

        # Alfanumerik ve standart tabela sembollerini koru (Türkçe karakterler dahil)
        cleaned = re.sub(r"[^\w\s\-\.\,\:\/]", " ", text, flags=re.UNICODE)
        # Birden fazla boşluğu teke indir
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def read_text_from_crop(self, crop_bgr: np.ndarray, min_confidence: float = 0.25) -> str:
        """
        Kırpılmış tabela görselini alıp metin okur.
        """
        if crop_bgr is None or crop_bgr.size == 0 or self.reader is None:
            return ""

        try:
            detected_words = []

            if self.engine == 'paddle':
                # Önce orijinal görseli işle
                res = self.reader.ocr(crop_bgr, cls=True)

                # Eğer orijinalden metin çıkmazsa ön işlenmiş görseli dene
                if not res or not res[0]:
                    preprocessed = self.preprocess_image(crop_bgr)
                    if preprocessed is not None:
                        res = self.reader.ocr(preprocessed, cls=True)

                if res and res[0]:
                    for line in res[0]:
                        if not line or len(line) < 2:
                            continue
                        text, conf = line[1]
                        if conf >= min_confidence:
                            cleaned = self.clean_text(text)
                            if cleaned:
                                detected_words.append(cleaned)

            elif self.engine == 'easyocr':
                results = self.reader.readtext(crop_bgr)
                if not results:
                    preprocessed = self.preprocess_image(crop_bgr)
                    if preprocessed is not None:
                        results = self.reader.readtext(preprocessed)

                for bbox, text, conf in results:
                    if conf >= min_confidence:
                        cleaned = self.clean_text(text)
                        if cleaned:
                            detected_words.append(cleaned)

            return " ".join(detected_words).strip()

        except Exception as e:
            print(f"[OCR HATA] Metin okuma sırasında hata: {e}")
            return ""
