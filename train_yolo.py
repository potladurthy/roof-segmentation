import torch
from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import yaml


class YOLOTrainer:
    def __init__(self, dataset_yaml, model_size='n'):
        """
        Initialize YOLO trainer
        
        Args:
            dataset_yaml: Path to dataset YAML configuration
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        """
        self.dataset_yaml = dataset_yaml
        self.model_size = model_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = YOLO(f'yolov8{model_size}-seg.pt')
        self.num_workers = torch.multiprocessing.cpu_count()  
    
    def train(self, epochs=100, imgsz=640, batch_size=16, workers=8):
        """
        Train YOLO segmentation model
        
        Args:
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Batch size for training
            workers: Number of data loader workers
        """        
        # Training configuration
        train_config = {
            'data': self.dataset_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': self.device,
            'workers': workers,
            'patience': 20,
            'save': True,
            'cache': True,
            'amp': True,
            'val': True,
            'lr0': 0.01,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate (lr0 * lrf)
            'momentum': 0.937,  # Momentum
            'weight_decay': 0.0005,  # Weight decay
            'warmup_epochs': 3,  # Warmup epochs
            'warmup_momentum': 0.8,  # Warmup momentum
            'warmup_bias_lr': 0.1,  # Warmup bias learning rate
        }
        
        print("Starting YOLO segmentation training...")
        print(f"Configuration: {train_config}")
        
        # Start training
        results = self.model.train(**train_config)
        
        return results
    
    def validate(self):
        """Validate the trained model"""
        print("Validating model...")
        results = self.model.val()
        return results
    
    def export_model(self, format='onnx'):
        """Export trained model to different formats"""
        print(f"Exporting model to {format}...")
        self.model.export(format=format)

def main():
    # Configuration
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model")
    parser.add_argument("--dataset_yaml", type=str, default="/outputs/yolo_dataset/dataset.yaml")
    parser.add_argument("--model_size", type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help="YOLO model size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    DATASET_YAML = args.dataset_yaml
    MODEL_SIZE = args.model_size
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    # read the image size from the dataset YAML
    with open(DATASET_YAML, 'r') as f:
        data = yaml.safe_load(f)
        IMAGE_SIZE = data.get('imgsz', 640)  # Default to 640 if not found

    # Check if dataset exists
    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset YAML not found at {DATASET_YAML}")
        print("Please run data_processing.py first to prepare the dataset.")
        return
    
    # Initialize trainer
    trainer = YOLOTrainer(DATASET_YAML, MODEL_SIZE)
    
    # Train model
    results = trainer.train(
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        workers=trainer.num_workers 
    )
    
    # Validate model
    val_results = trainer.validate()
    
    # Export model (optional)
    # trainer.export_model('onnx')
    
    print("\nTraining completed!")
    print(f"Best model saved at: runs/segment/train/weights/best.pt")
    print(f"Validation results: {val_results}")

if __name__ == "__main__":
    main()
