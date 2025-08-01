import pandas as pd
import cv2
import numpy as np
import os
import yaml
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import argparse

class YOLODataProcessor:
    def __init__(self, parquet_path, output_dir):
        """
        Initialize YOLO data processor
        
        Args:
            parquet_path: Path to the parquet file containing annotations
            output_dir: Directory to save processed YOLO dataset
        """
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.data = pd.read_parquet(parquet_path)
        
        # Create directory structure
        self.setup_directories()
        self.imgsz = None  
        
        # Define augmentation pipelines
        self.setup_augmentations()
    
    def setup_directories(self):
        """Create YOLO dataset directory structure"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def setup_augmentations(self):
        """Setup augmentation pipelines for training data"""
        # Heavy augmentations for training
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        # Light augmentations for validation
        self.val_transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def extract_normalized_keypoints(self, annotations):
        """Extract already normalized keypoints from annotations"""
        all_keypoints = []
        for obj in annotations['objects']:
            keypoints_array = obj['keyPoints']
            normalized_points = []
            for point in keypoints_array[0]['points']:
                # Coordinates are already normalized (0-1)
                normalized_points.extend([point['x'], point['y']])
            all_keypoints.append(normalized_points)
        return all_keypoints
    
    def convert_to_pixel_coordinates(self, normalized_keypoints, image_width, image_height):
        """Convert normalized keypoints to pixel coordinates for augmentation"""
        pixel_keypoints = []
        for keypoint_list in normalized_keypoints:
            pixel_points = []
            for i in range(0, len(keypoint_list), 2):
                pixel_x = keypoint_list[i] * image_width
                pixel_y = keypoint_list[i+1] * image_height
                pixel_points.append([pixel_x, pixel_y])
            pixel_keypoints.append(pixel_points)
        return pixel_keypoints
    
    def convert_to_normalized_coordinates(self, pixel_keypoints, image_width, image_height):
        """Convert pixel keypoints back to normalized coordinates"""
        normalized_keypoints = []
        for pixel_points in pixel_keypoints:
            normalized_points = []
            for point in pixel_points:
                norm_x = point[0] / image_width
                norm_y = point[1] / image_height
                # Clamp to valid range
                norm_x = max(0, min(1, norm_x))
                norm_y = max(0, min(1, norm_y))
                normalized_points.extend([norm_x, norm_y])
            normalized_keypoints.append(normalized_points)
        return normalized_keypoints
    
    def apply_augmentation(self, image, normalized_keypoints, split):
        """Apply augmentation to image and keypoints"""
        if split == 'test':
            return image, normalized_keypoints
        
        image_height, image_width = image.shape[:2]
        if self.imgsz is None:
            self.imgsz = max(image_height, image_width)
        
        # Convert to pixel coordinates for augmentation
        pixel_keypoints = self.convert_to_pixel_coordinates(normalized_keypoints, image_width, image_height)
        
        # Flatten keypoints for albumentations
        all_keypoints = []
        keypoint_counts = []
        
        for keypoint_group in pixel_keypoints:
            all_keypoints.extend(keypoint_group)
            keypoint_counts.append(len(keypoint_group))
        
        # Apply augmentation
        try:
            if split == 'train':
                transformed = self.train_transform(image=image, keypoints=all_keypoints)
            else:  # validation
                transformed = self.val_transform(image=image, keypoints=all_keypoints)
            
            aug_image = transformed['image']
            aug_keypoints = transformed['keypoints']
            
            # Reconstruct keypoint groups
            aug_pixel_keypoints = []
            start_idx = 0
            
            for count in keypoint_counts:
                end_idx = start_idx + count
                group_keypoints = []
                for i in range(start_idx, min(end_idx, len(aug_keypoints))):
                    group_keypoints.append(aug_keypoints[i])
                
                # Fill missing keypoints if some were removed during augmentation
                while len(group_keypoints) < count:
                    if group_keypoints:
                        group_keypoints.append(group_keypoints[-1])  # Duplicate last point
                    else:
                        group_keypoints.append([0, 0])  # Default point
                
                aug_pixel_keypoints.append(group_keypoints)
                start_idx = end_idx
            
            # Convert back to normalized coordinates
            aug_height, aug_width = aug_image.shape[:2]
            aug_normalized_keypoints = self.convert_to_normalized_coordinates(
                aug_pixel_keypoints, aug_width, aug_height
            )
            
            return aug_image, aug_normalized_keypoints
            
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return image, normalized_keypoints
    
    def process_dataset(self, augmentation_factor=3):
        """Process entire dataset with augmentations"""
        print("Processing dataset...")
        
        for split in ['train', 'val']:
            split_data = self.data[self.data['partition'] == split]
            print(f"Processing {split} split: {len(split_data)} images")
            
            for idx, row in tqdm(split_data.iterrows(), total=len(split_data)):
                img_path = os.path.join(os.getcwd(), row['asset_url'])
                annotations = row['annotations'][0]
                
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract normalized keypoints directly
                normalized_keypoints = self.extract_normalized_keypoints(annotations)
                
                # Original image
                self.save_sample(image, normalized_keypoints, f"{idx}_original", split)
                
                # Apply augmentations (only for train split)
                if split == 'train':
                    for aug_idx in range(augmentation_factor):
                        try:
                            aug_image, aug_keypoints = self.apply_augmentation(
                                image, normalized_keypoints, split
                            )
                            self.save_sample(
                                aug_image, aug_keypoints, 
                                f"{idx}_aug_{aug_idx}", split
                            )
                        except Exception as e:
                            print(f"Warning: Augmentation failed for {idx}_aug_{aug_idx}: {e}")
                elif split == 'val':
                    # Light augmentations for validation
                    for aug_idx in range(1):  # Only 1 augmentation for val
                        try:
                            aug_image, aug_keypoints = self.apply_augmentation(
                                image, normalized_keypoints, split
                            )
                            self.save_sample(
                                aug_image, aug_keypoints, 
                                f"{idx}_aug_{aug_idx}", split
                            )
                        except Exception as e:
                            print(f"Warning: Augmentation failed for {idx}_aug_{aug_idx}: {e}")
    
    def save_sample(self, image, normalized_keypoints_list, filename, split):
        """Save image and corresponding YOLO label with normalized coordinates"""
        # Save image
        img_path = self.output_dir / 'images' / split / f"{filename}.jpg"
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), image_bgr)
        
        # Save label file with normalized coordinates directly
        label_path = self.output_dir / 'labels' / split / f"{filename}.txt"
        with open(label_path, 'w') as f:
            for normalized_keypoints in normalized_keypoints_list:
                if len(normalized_keypoints) >= 6:  # At least 3 points (6 coordinates)
                    # Ensure coordinates are properly formatted
                    coords_str = ' '.join([f"{coord:.6f}" for coord in normalized_keypoints])
                    f.write(f"0 {coords_str}\n")  # Class 0 for roof
    
    def create_yaml_config(self):
        """Create YOLO dataset configuration file"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes
            'names': ['roof'],
            'imgsz': self.imgsz,
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to: {yaml_path}")
        return yaml_path

if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser(description="Process dataset for YOLO")
    parser.add_argument("--parquet_path", type=str, default="/data/dataset.parquet")
    parser.add_argument("--output_dir", type=str, default="/outputs/yolo_dataset")
    args = parser.parse_args()
    PARQUET_PATH = args.parquet_path
    OUTPUT_DIR = args.output_dir
    # Process dataset
    processor = YOLODataProcessor(PARQUET_PATH, OUTPUT_DIR)
    processor.process_dataset(augmentation_factor=5)  # 5x augmentation for training
    yaml_path = processor.create_yaml_config()
    
    print("Dataset processing completed!")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print(f"YAML config: {yaml_path}")