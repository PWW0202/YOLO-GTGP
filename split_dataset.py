import os
import random
import shutil
from pathlib import Path

def split_yolo_dataset(image_dir, label_dir, output_dir, split_ratio=(0.7, 0.2, 0.1), seed=42):
    """
    Splits a YOLO format dataset into training, validation, and testing sets.
    
    Args:
        image_dir (str/Path): Directory containing the source images.
        label_dir (str/Path): Directory containing the corresponding YOLO label files.
        output_dir (str/Path): Target root directory for the split dataset.
        split_ratio (tuple): Ratio for (train, val, test). Must sum to 1.0. Defaults to (0.7, 0.2, 0.1).
        seed (int): Random seed for reproducibility. Defaults to 42.
    """
    # 1. Set random seed to ensure reproducibility
    random.seed(seed)
    
    # 2. Convert to Path objects for easier manipulation
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    # 3. Define the target directory structure
    sub_dirs = ['train', 'val', 'test']
    for sub in sub_dirs:
        (output_dir / 'images' / sub).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / sub).mkdir(parents=True, exist_ok=True)
        
    # 4. Get all image files (supports common formats like jpg, jpeg, png, etc.)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    all_images = [f for f in image_dir.iterdir() if f.suffix in image_extensions]
    
    # Sort image names in ascending order to guarantee identical shuffling results across different runs with the same seed
    all_images.sort()
    
    # Shuffle the dataset
    random.shuffle(all_images)
    
    total_count = len(all_images)
    if total_count == 0:
        print(f"❌ Error: No image files found in {image_dir}!")
        return
        
    print(f"📊 Found {total_count} images. Starting split with ratio {split_ratio}...")

    # 5. Calculate split indices based on the provided ratio
    train_end = int(total_count * split_ratio[0])
    val_end = train_end + int(total_count * split_ratio[1])
    
    # 6. Distribute the data
    for i, img_path in enumerate(all_images):
        # Determine which subset the current image belongs to
        if i < train_end:
            category = 'train'
        elif i < val_end:
            category = 'val'
        else:
            category = 'test'
            
        # Find the corresponding label file
        label_path = label_dir / f"{img_path.stem}.txt"
        
        # Define target paths
        dest_img_path = output_dir / 'images' / category / img_path.name
        dest_lbl_path = output_dir / 'labels' / category / label_path.name
        
        # Copy the image file
        shutil.copy(img_path, dest_img_path)
        
        # Copy the label file (if it exists)
        if label_path.exists():
            shutil.copy(label_path, dest_lbl_path)
        else:
            print(f"⚠️ Warning: Label file {label_path.name} not found for image {img_path.name}")

    print("✅ Dataset splitting completed!")
    print(f"Training set (train):   {len(os.listdir(output_dir / 'images' / 'train'))} images")
    print(f"Validation set (val):   {len(os.listdir(output_dir / 'images' / 'val'))} images")
    print(f"Test set (test):        {len(os.listdir(output_dir / 'images' / 'test'))} images")


if __name__ == "__main__":
    # --- Please modify the following variables according to your actual paths ---
    
    # If the script and folders are in the same directory, you can leave these as is
    IMAGE_DIRECTORY = "./images" 
    LABEL_DIRECTORY = "./labels"
    OUTPUT_DIRECTORY = "./dataset" # Root directory for the newly split dataset
    
    # Split ratio (Train, Val, Test). Ensure the sum equals 1.0
    SPLIT_RATIO = (0.7, 0.2, 0.1) 
    
    # Random seed. Changing this number yields a different random split; 
    # fixing this number ensures consistent results across different runs.
    RANDOM_SEED = 42 
    
    split_yolo_dataset(IMAGE_DIRECTORY, LABEL_DIRECTORY, OUTPUT_DIRECTORY, SPLIT_RATIO, RANDOM_SEED)