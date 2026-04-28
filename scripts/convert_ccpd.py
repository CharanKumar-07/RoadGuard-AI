# scripts/convert_ccpd.py

import os
import cv2
import random
import shutil
from tqdm import tqdm


def find_image_folder(base_path):
    """Find a subfolder that contains .jpg files."""
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                return root
    return None


def convert_ccpd_to_yolo(image_dir, output_images_dir, output_labels_dir):
    """Convert images from image_dir to YOLO format."""
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    print(f"Found {len(image_files)} images in {image_dir}")

    for img_file in tqdm(image_files, desc="Converting"):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_file}")
            continue
        h, w, _ = img.shape

        # Parse filename to get bounding box coordinates
        parts = img_file.split('-')
        if len(parts) < 3:
            print(f"Warning: Unexpected filename format {img_file}")
            continue

        bbox_part = parts[2]  # e.g., "154&383_386&473"
        coords = bbox_part.split('_')
        if len(coords) != 2:
            print(f"Warning: Bbox format wrong in {img_file}")
            continue
        x1y1 = coords[0].split('&')
        x2y2 = coords[1].split('&')
        if len(x1y1) != 2 or len(x2y2) != 2:
            print(f"Warning: Coordinate split failed in {img_file}")
            continue

        try:
            x1, y1 = int(x1y1[0]), int(x1y1[1])
            x2, y2 = int(x2y2[0]), int(x2y2[1])
        except ValueError:
            print(f"Warning: Non-integer coordinates in {img_file}")
            continue

        # Normalize to YOLO format
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        # Copy image to output folder
        dst_img_path = os.path.join(output_images_dir, img_file)
        cv2.imwrite(dst_img_path, img)

        # Write label file
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(output_labels_dir, label_file)
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


if __name__ == "__main__":
    # Define base CCPD folder (where you extracted the zip)
    ccpd_root = "../datasets/ccpd"  # this is the folder containing subfolders like ccpd_base, etc.

    # Automatically find the folder that contains images (usually ccpd_base)
    image_source_dir = find_image_folder(ccpd_root)
    if image_source_dir is None:
        print("ERROR: No .jpg files found in any subfolder of", ccpd_root)
        print("Please check the contents of ../datasets/ccpd")
        exit(1)

    print(f"Using images from: {image_source_dir}")

    # Get all image files
    all_images = [f for f in os.listdir(image_source_dir) if f.lower().endswith('.jpg')]
    print(f"Total images found: {len(all_images)}")

    # Shuffle and split
    random.shuffle(all_images)
    total = len(all_images)
    if total == 0:
        print("ERROR: No images found. Exiting.")
        exit(1)

    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    output_base = "../datasets/license_plate"

    for split_name, img_list in splits.items():
        print(f"\nProcessing {split_name} split ({len(img_list)} images)")

        # Create temporary folder for this split (within current directory)
        temp_split_dir = f"temp_{split_name}"
        os.makedirs(temp_split_dir, exist_ok=True)

        # Copy images to temp folder
        print(f"Copying {len(img_list)} images to {temp_split_dir}...")
        for img in tqdm(img_list, desc=f"Copying {split_name}"):
            src = os.path.join(image_source_dir, img)
            dst = os.path.join(temp_split_dir, img)
            shutil.copy2(src, dst)

        # Convert this split
        convert_ccpd_to_yolo(
            temp_split_dir,
            os.path.join(output_base, "images", split_name),
            os.path.join(output_base, "labels", split_name)
        )

        # Remove temp folder
        shutil.rmtree(temp_split_dir)

    print("\nCCPD conversion completed!")