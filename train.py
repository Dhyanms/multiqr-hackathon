"""
Multi-QR Code Detection - Training Script
Uses YOLOv8 for maximum accuracy
"""

import os
import sys
import yaml
from pathlib import Path

print("Starting training script...")
print(f"Python version: {sys.version}")

try:
    from ultralytics import YOLO
    import torch
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

def setup_data_yaml(dataset_path):
    """
    Update data.yaml with correct paths
    """
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ Error: data.yaml not found at {data_yaml_path}")
        sys.exit(1)
    
    print(f"Loading data.yaml from: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    base_path = os.path.abspath(dataset_path)
    data['train'] = os.path.join(base_path, 'train', 'images')
    data['val'] = os.path.join(base_path, 'train', 'images')
    
    test_path = os.path.join(os.path.dirname(base_path), 'test', 'images')
    if os.path.exists(test_path):
        data['test'] = test_path
    
    if not os.path.exists(data['train']):
        print(f"❌ Error: Training images not found at {data['train']}")
        sys.exit(1)
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✓ Updated data.yaml")
    print(f"  Train path: {data['train']}")
    print(f"  Val path: {data['val']}")
    
    return data_yaml_path

def train_model(dataset_path='data/annotation.v3-train-clr.yolov8', 
                epochs=100, 
                img_size=640,
                batch_size=16,
                model_name='yolov8s.pt'):
    """
    Train YOLOv8 model for QR code detection
    """
    
    print("=" * 70)
    print("Multi-QR Code Detection - Training Pipeline")
    print("=" * 70)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔧 Device: {'GPU' if device == '0' else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  No GPU detected. Training will be slow on CPU.")
    
    print("\n📂 Setting up dataset configuration...")
    data_yaml_path = setup_data_yaml(dataset_path)
    
    print(f"\n📦 Loading YOLOv8 model: {model_name}")
    try:
        model = YOLO(model_name)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    print("\n⚙️  Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {img_size}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {dataset_path}")
    
    os.makedirs('outputs', exist_ok=True)
    
    print("\n🚀 Starting training...\n")
    print("=" * 70)
    
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project='outputs',
            name='qr_detection',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            patience=50,
            save=True,
            save_period=-1,
            cache=False,
            verbose=True,
            workers=4,
            plots=True,
            amp=True
        )
        
        print("\n" + "=" * 70)
        
        best_model_path = 'outputs/qr_detection/weights/best.pt'
        if os.path.exists(best_model_path):
            print(f"\n✅ Training completed!")
            print(f"📊 Best model saved at: {best_model_path}")
            
            print("\n🔍 Validating model...")
            metrics = model.val()
            print(f"\n📈 Validation Metrics:")
            print(f"   mAP50: {metrics.box.map50:.4f}")
            print(f"   mAP50-95: {metrics.box.map:.4f}")
            print(f"   Precision: {metrics.box.mp:.4f}")
            print(f"   Recall: {metrics.box.mr:.4f}")
        else:
            print("\n⚠️  Warning: Best model not found. Check training logs.")
        
        print("\n" + "=" * 70)
        print("Training pipeline completed!")
        print("=" * 70)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for QR Code Detection')
    parser.add_argument('--dataset', '--data', type=str, default='data/annotation.v3-train-clr.yolov8',
                        dest='dataset', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--img-size', '--img', type=int, default=640,
                        dest='img_size', help='Input image size')
    parser.add_argument('--batch-size', '--batch', type=int, default=16,
                        dest='batch_size', help='Batch size')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='YOLOv8 model variant (yolov8n/s/m/l/x)')
    
    args = parser.parse_args()
    
    print("\n🎯 Arguments:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Image Size: {args.img_size}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Model: {args.model}")
    print()
    
    train_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        model_name=args.model
    )