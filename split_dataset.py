import os
import shutil
import random
import sys

random.seed(42)

SOURCE_DIR = "COVID-19_Radiography_Dataset"
TARGET_DIR = "data"

# Exact folder names based on your dataset structure
classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

splits = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

def progress_bar(current, total, prefix=""):
    percent = (current / total) * 100
    bar = "█" * int(percent // 2) + "-" * (50 - int(percent // 2))
    sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}%")
    sys.stdout.flush()


# Create target directories
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls, "images")

    if not os.path.exists(class_path):
        print(f"❌ ERROR: 'images' folder not found in: {class_path}")
        continue

    # Collect only image files
    imgs = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(imgs) == 0:
        print(f"⚠️ WARNING: No images found in {class_path}")
        continue

    random.shuffle(imgs)

    n = len(imgs)
    train_end = int(splits["train"] * n)
    val_end = train_end + int(splits["val"] * n)

    tasks = [
        ("train", imgs[:train_end]),
        ("val", imgs[train_end:val_end]),
        ("test", imgs[val_end:])
    ]

    print(f"\n📁 Processing class: {cls} ({n} images)")

    copied = 0
    total_to_copy = n

    for split_name, split_imgs in tasks:
        dst_folder = os.path.join(TARGET_DIR, split_name, cls)

        for img in split_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(dst_folder, img)

            # OVERWRITE if file already exists
            if os.path.exists(dst):
                os.remove(dst)

            shutil.copy(src, dst)

            copied += 1
            progress_bar(copied, total_to_copy, prefix=f" → Copying {cls}")

    print("\n✔ Done!")


print("\n🎉 All classes successfully split into train/val/test!")
