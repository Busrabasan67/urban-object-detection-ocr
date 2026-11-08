"""
MOSK BİLİŞİM - Yapay Zeka ile Nesne Tanıma ve Tabela OCR Uygulaması.

Kullanım:
  # Tek bir görseli işlemek için:
  python main.py --source data/images/val/ornek.jpg

  # Bir klasördeki tüm görselleri işlemek için:
  python main.py --source data/images/val/

  # OCR olmadan hızlı tespit:
  python main.py --source data/images/val/ --no-ocr
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
from src.detect import Detector


def save_results_to_json(results: List[Dict[str, Any]], output_path: str):
    """Tespit sonuçlarını JSON formatında kaydeder."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"[KAYDEDİLDİ] JSON sonuçları: {output_path}")


def save_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """Tespit sonuçlarını CSV formatında kaydeder."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "image_name",
        "class_name",
        "class_label_tr",
        "confidence",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "ocr_text",
        "detection_date",
        "inference_time_sec"
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "image_name": r.get("image_name"),
                "class_name": r.get("class_name"),
                "class_label_tr": r.get("class_label_tr"),
                "confidence": r.get("confidence"),
                "bbox_x": r.get("bbox", [0, 0, 0, 0])[0],
                "bbox_y": r.get("bbox", [0, 0, 0, 0])[1],
                "bbox_w": r.get("bbox", [0, 0, 0, 0])[2],
                "bbox_h": r.get("bbox", [0, 0, 0, 0])[3],
                "ocr_text": r.get("ocr_text", ""),
                "detection_date": r.get("detection_date"),
                "inference_time_sec": r.get("inference_time_sec")
            })
    print(f"[KAYDEDİLDİ] CSV sonuçları: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MOSK Bilişim Nesne Tanıma ve OCR Uygulaması")
    parser.add_argument("--source", type=str, default="data/images/val", help="Girdi görseli veya görsel klasörü yolu")
    parser.add_argument("--model", type=str, default="models/best.pt", help="YOLO model ağırlıkları (.pt)")
    parser.add_argument("--output", type=str, default="output", help="Çıktıların kaydedileceği klasör")
    parser.add_argument("--conf", type=float, default=0.25, help="Güven (confidence) eşik değeri")
    parser.add_argument("--no-ocr", action="store_true", help="Tabela OCR okumasını devre dışı bırakır")
    parser.add_argument("--ocr-engine", type=str, default="paddle", choices=["paddle", "easyocr"], help="Kullanılacak OCR motoru (paddle veya easyocr)")

    args = parser.parse_args()

    source_path = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dedektörü başlat
    detector = Detector(
        model_path=args.model,
        conf_thresh=args.conf,
        use_ocr=not args.no_ocr,
        ocr_engine=args.ocr_engine
    )

    # İşlenecek görselleri topla
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images_to_process = []

    if source_path.is_file():
        if source_path.suffix.lower() in valid_extensions:
            images_to_process.append(source_path)
        else:
            print(f"[UYARI] Desteklenmeyen dosya formatı: {source_path}")
            return
    elif source_path.is_dir():
        for p in source_path.iterdir():
            if p.is_file() and p.suffix.lower() in valid_extensions:
                images_to_process.append(p)
    else:
        print(f"[HATA] Kaynak yol bulunamadı: {source_path}")
        return

    if not images_to_process:
        print(f"[UYARI] İşlenecek görsel bulunamadı: {source_path}")
        return

    print(f"\n[BAŞLADI] Toplam {len(images_to_process)} görsel işlenecek...\n")

    all_detections = []
    total_time = 0.0

    for idx, img_p in enumerate(images_to_process, 1):
        print(f"[{idx}/{len(images_to_process)}] İşleniyor: {img_p.name}")
        try:
            detections, annotated_img, inf_time = detector.process_image(str(img_p))
            total_time += inf_time
            all_detections.extend(detections)

            # Çıktı görselini kaydet (yalnızca annotated_images içerisine)
            annotated_dir = output_dir / "annotated_images"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            save_img_path = annotated_dir / f"detected_{img_p.name}"
            cv2.imwrite(str(save_img_path), annotated_img)

            # Tespit özetini yazdır
            print(f"   -> {len(detections)} nesne tespit edildi ({inf_time:.3f} sn)")
            for d in detections:
                tabela_txt = f" (OCR: '{d['ocr_text']}')" if d['ocr_text'] else ""
                print(f"      - {d['class_label_tr']} (%{int(d['confidence']*100)}){tabela_txt}")

        except Exception as e:
            print(f"[HATA] {img_p.name} işlenirken hata oluştu: {e}")

    avg_time = (total_time / len(images_to_process)) if images_to_process else 0

    # Sonuçları yapısal formatlarda kaydet
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    save_results_to_json(all_detections, str(json_path))
    save_results_to_csv(all_detections, str(csv_path))
    save_results_to_json(all_detections, str(output_dir / "detections.json"))
    save_results_to_csv(all_detections, str(output_dir / "detections.csv"))

    print("\n" + "=" * 50)
    print("İŞLEM TAMAMLANDI!")
    print(f"İşlenen Görsel Sayısı : {len(images_to_process)}")
    print(f"Toplam Tespit Edilen : {len(all_detections)} nesne")
    print(f"Ortalama Tahmin Süresi : {avg_time:.3f} sn / görsel")
    print(f"Çıktı Dizini           : {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()

