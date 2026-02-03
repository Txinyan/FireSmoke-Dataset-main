# #!/usr/bin/env python3
# import argparse
# import os
# import random
# import shutil
# from pathlib import Path

# IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# def parse_args():
#     ap = argparse.ArgumentParser(description="YOLO dataset subset extractor with train/val split")
#     ap.add_argument("--dataset", required=True, type=str,
#                     help="Path to original YOLO dataset (must contain images/ and labels/)")
#     ap.add_argument("--out", required=True, type=str,
#                     help="Output folder for subset dataset")
#     ap.add_argument("--num-fire", type=int, default=5500,
#                     help="Number of fire (class 0) images to keep")
#     ap.add_argument("--num-smoke", type=int, default=3000,
#                     help="Number of smoke (class 1) images to keep")
#     ap.add_argument("--split", type=float, default=0.8,
#                     help="Train split ratio (default 0.8 = 80% train, 20% val)")
#     ap.add_argument("--seed", type=int, default=42, help="Random seed")
#     return ap.parse_args()

# def read_label_file(label_path):
#     """Read YOLO txt file and return set of class IDs."""
#     class_ids = set()
#     if not label_path.exists():
#         return class_ids
#     with open(label_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 5:
#                 try:
#                     cid = int(float(parts[0]))
#                     class_ids.add(cid)
#                 except:
#                     continue
#     return class_ids

# def collect_images(labels_dir, images_dir):
#     """Return dict: {0: [image_paths], 1: [image_paths]}"""
#     class_to_imgs = {0: [], 1: []}
#     for lbl_file in labels_dir.rglob("*.txt"):
#         img_file = None
#         for ext in IMAGE_EXTS:
#             candidate = images_dir / (lbl_file.stem + ext)
#             if candidate.exists():
#                 img_file = candidate
#                 break
#         if img_file is None:
#             continue

#         class_ids = read_label_file(lbl_file)
#         for cid in class_ids:
#             if cid in class_to_imgs:
#                 class_to_imgs[cid].append((img_file, lbl_file))
#     return class_to_imgs

# def copy_subset(pairs, out_img_dir, out_lbl_dir):
#     for img_path, lbl_path in pairs:
#         shutil.copy(img_path, out_img_dir / img_path.name)
#         shutil.copy(lbl_path, out_lbl_dir / lbl_path.name)

# def main():
#     args = parse_args()
#     random.seed(args.seed)

#     dataset_dir = Path(os.path.expanduser(args.dataset))
#     images_dir = dataset_dir / "images"
#     labels_dir = dataset_dir / "labels"

#     if not images_dir.exists() or not labels_dir.exists():
#         raise FileNotFoundError("Dataset must contain 'images/' and 'labels/' directories")

#     out_dir = Path(args.out)
#     train_img_dir = out_dir / "train" / "images"
#     train_lbl_dir = out_dir / "train" / "labels"
#     val_img_dir = out_dir / "val" / "images"
#     val_lbl_dir = out_dir / "val" / "labels"

#     for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
#         d.mkdir(parents=True, exist_ok=True)

#     # Collect
#     class_to_imgs = collect_images(labels_dir, images_dir)

#     fire_imgs = class_to_imgs[0]
#     smoke_imgs = class_to_imgs[1]

#     print(f"Found {len(fire_imgs)} fire images, {len(smoke_imgs)} smoke images in dataset")

#     fire_subset = random.sample(fire_imgs, min(args.num_fire, len(fire_imgs)))
#     smoke_subset = random.sample(smoke_imgs, min(args.num_smoke, len(smoke_imgs)))

#     all_subset = fire_subset + smoke_subset
#     random.shuffle(all_subset)

#     # Split train/val
#     split_idx = int(len(all_subset) * args.split)
#     train_pairs = all_subset[:split_idx]
#     val_pairs = all_subset[split_idx:]

#     # Copy
#     copy_subset(train_pairs, train_img_dir, train_lbl_dir)
#     copy_subset(val_pairs, val_img_dir, val_lbl_dir)

#     print(f"[Done] Subset created at: {out_dir}")
#     print(f" - Train: {len(train_pairs)} images")
#     print(f" - Val:   {len(val_pairs)} images")

# if __name__ == "__main__":
#     main()


# python3 select_dataset.py --dataset "/home/txy/下载/archive/Fire and Smoke Dataset/train" --out "./subset_dataset" --num-fire 5500 --num-smoke 3000 --split 0.8 --seed 42

#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def parse_args():
    ap = argparse.ArgumentParser(description="YOLO dataset subset extractor with train/val split")
    ap.add_argument("--dataset", required=True, type=str,
                    help="Path to original YOLO dataset (must contain images/ and labels/)")
    ap.add_argument("--out", required=True, type=str,
                    help="Output folder for subset dataset")
    ap.add_argument("--num-fire", type=int, default=5500,
                    help="Number of fire (class 0) images to keep")
    ap.add_argument("--num-smoke", type=int, default=3000,
                    help="Number of smoke (class 1) images to keep")
    ap.add_argument("--split", type=float, default=0.8,
                    help="Train split ratio (default 0.8 = 80% train, 20% val)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()

def read_label_file(label_path):
    """Read YOLO txt file and return set of class IDs."""
    class_ids = set()
    if not label_path.exists():
        return class_ids
    # 排除 .rf. 文件
    if ".rf." in label_path.name:
        return class_ids
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cid = int(float(parts[0]))
                    class_ids.add(cid)
                except:
                    continue
    return class_ids

def collect_images(labels_dir, images_dir):
    """Return dict: {0: [image_paths], 1: [image_paths]}, ignoring .rf. files"""
    class_to_imgs = {0: [], 1: []}
    for lbl_file in labels_dir.rglob("*.txt"):
        if ".rf." in lbl_file.name:  # 忽略.rf.文件
            continue

        img_file = None
        for ext in IMAGE_EXTS:
            candidate = images_dir / (lbl_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            continue

        class_ids = read_label_file(lbl_file)
        for cid in class_ids:
            if cid in class_to_imgs:
                class_to_imgs[cid].append((img_file, lbl_file))
    return class_to_imgs

def copy_subset(pairs, out_img_dir, out_lbl_dir):
    for img_path, lbl_path in pairs:
        shutil.copy(img_path, out_img_dir / img_path.name)
        shutil.copy(lbl_path, out_lbl_dir / lbl_path.name)

def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_dir = Path(os.path.expanduser(args.dataset))
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Dataset must contain 'images/' and 'labels/' directories")

    out_dir = Path(args.out)
    train_img_dir = out_dir / "train" / "images"
    train_lbl_dir = out_dir / "train" / "labels"
    val_img_dir = out_dir / "val" / "images"
    val_lbl_dir = out_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect images excluding .rf. labels
    class_to_imgs = collect_images(labels_dir, images_dir)

    fire_imgs = class_to_imgs[0]
    smoke_imgs = class_to_imgs[1]

    print(f"Found {len(fire_imgs)} fire images, {len(smoke_imgs)} smoke images in dataset")

    fire_subset = random.sample(fire_imgs, min(args.num_fire, len(fire_imgs)))
    smoke_subset = random.sample(smoke_imgs, min(args.num_smoke, len(smoke_imgs)))

    all_subset = fire_subset + smoke_subset
    random.shuffle(all_subset)

    # Split train/val
    split_idx = int(len(all_subset) * args.split)
    train_pairs = all_subset[:split_idx]
    val_pairs = all_subset[split_idx:]

    # Copy files
    copy_subset(train_pairs, train_img_dir, train_lbl_dir)
    copy_subset(val_pairs, val_img_dir, val_lbl_dir)

    print(f"[Done] Subset created at: {out_dir}")
    print(f" - Train: {len(train_pairs)} images")
    print(f" - Val:   {len(val_pairs)} images")

if __name__ == "__main__":
    main()
