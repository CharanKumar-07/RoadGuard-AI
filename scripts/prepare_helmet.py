# scripts/prepare_helmet.py

import os
import random
import shutil
from tqdm import tqdm

def organize_helmet_dataset(source_images, source_labels, output_dir, split_ratios=(0.8, 0.1, 0.1)):
    """
    Organize helmet dataset into YOLO folder structure.
    Assumes images and labels are in separate folders with matching filenames.
    """
    # Get all image files
    image_files = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Total images: {len(image_files)}")

    # Shuffle and split
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(split_ratios[0] * total)
    val_end = train_end + int(split_ratios[1] * total)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    # Create output directories
    for split in splits:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Copy files
    for split_name, file_list in splits.items():
        print(f"Processing {split_name} ({len(file_list)} files)")
        for img_file in tqdm(file_list):
            # Copy image
            src_img = os.path.join(source_images, img_file)
            dst_img = os.path.join(output_dir, 'images', split_name, img_file)
            shutil.copy2(src_img, dst_img)

            # Copy label (assuming same name with .txt extension)
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            src_label = os.path.join(source_labels, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_dir, 'labels', split_name, label_file)
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Warning: Label not found for {img_file}")

if __name__ == "__main__":
    # Adjust these paths according to your actual folder names
    source_images = "../datasets/helmet_raw/images"      # path to images
    source_labels = "../datasets/helmet_raw/annotations" # path to labels
    output_dir = "../datasets/helmet_detection"

    organize_helmet_dataset(source_images, source_labels, output_dir)
    print("Helmet dataset preparation completed!")