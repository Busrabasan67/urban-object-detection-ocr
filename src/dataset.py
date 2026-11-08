"""
Veri seti hazırlama, doğrulama ve veri artırma (augmentation) modülü.
Roboflow veya yerel veri dizinlerini YOLO formatına uyumlu biçimde yönetir.
"""

import os
import shutil
from pathlib import Path
import yaml


def verify_dataset_structure(data_dir: str = "data") -> bool:
    """
    Veri setinin dizin yapısını ve yaml dosyasını kontrol eder.
    """
    base = Path(data_dir)
    images_train = base / "images" / "train"
    images_val = base / "images" / "val"
    labels_train = base / "labels" / "train"
    labels_val = base / "labels" / "val"
    yaml_file = base / "data.yaml"

    checks = {
        "Train Images": images_train.exists(),
        "Val Images": images_val.exists(),
        "Train Labels": labels_train.exists(),
        "Val Labels": labels_val.exists(),
        "Data YAML": yaml_file.exists(),
    }

    all_ok = True
    print("[KONTROL] Veri seti yapısı inceleniyor:")
    for name, exists in checks.items():
        status = "MEVCUT" if exists else "EKSİK"
        print(f"  - {name}: {status}")
        if not exists:
            all_ok = False

    if all_ok:
        train_count = len(list(images_train.glob("*.*")))
        val_count = len(list(images_val.glob("*.*")))
        print(f"\n[BİLGİ] Eğitim Görseli: {train_count}, Doğrulama Görseli: {val_count}")

    return all_ok


if __name__ == "__main__":
    verify_dataset_structure()

