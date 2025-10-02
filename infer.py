"""
Multi-QR Code Detection - Inference Script
Generates submission JSON files for detection and decoding stages
"""

import os
import json
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

from src.utils.qr_decoder import decode_qr_codes
from src.utils.nms_utils import apply_custom_nms

def load_model(model_path='outputs/qr_detection/weights/best.pt'):
    """Load trained YOLOv8 model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    print(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    return model

def detect_qr_codes(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    Detect QR codes in an image
    
    Returns:
        list: List of bounding boxes in format [x_min, y_min, x_max, y_max]
    """
    results = model.predict(
        image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    bboxes = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    bboxes = apply_custom_nms(bboxes, iou_threshold=iou_threshold)
    
    return bboxes

def run_stage1_detection(model, input_dir, output_file, conf_threshold=0.25):
    """
    Stage 1: Detection only
    Generate submission_detection_1.json
    """
    print("\n" + "=" * 70)
    print("Stage 1: QR Code Detection")
    print("=" * 70)
    
    results = []
    image_files = sorted([f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n🔍 Processing {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc="Detecting QR codes"):
        img_path = os.path.join(input_dir, img_file)
        image_id = os.path.splitext(img_file)[0]
        
        bboxes = detect_qr_codes(model, img_path, conf_threshold=conf_threshold)
        
        qr_list = [{"bbox": bbox} for bbox in bboxes]
        results.append({
            "image_id": image_id,
            "qrs": qr_list
        })
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Stage 1 completed!")
    print(f"📄 Results saved to: {output_file}")
    print(f"📊 Total images processed: {len(results)}")
    print(f"📊 Total QR codes detected: {sum(len(r['qrs']) for r in results)}")
    
    return results

def run_stage2_decoding(model, input_dir, output_file, conf_threshold=0.25):
    """
    Stage 2 (Bonus): Detection + Decoding + Classification
    Generate submission_decoding_2.json
    """
    print("\n" + "=" * 70)
    print("Stage 2 (Bonus): QR Code Detection + Decoding + Classification")
    print("=" * 70)
    
    results = []
    image_files = sorted([f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n🔍 Processing {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc="Detecting & Decoding QR codes"):
        img_path = os.path.join(input_dir, img_file)
        image_id = os.path.splitext(img_file)[0]
        
        image = cv2.imread(img_path)
        
        bboxes = detect_qr_codes(model, img_path, conf_threshold=conf_threshold)
        
        qr_list = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            
            padding = 10
            y1 = max(0, y_min - padding)
            y2 = min(image.shape[0], y_max + padding)
            x1 = max(0, x_min - padding)
            x2 = min(image.shape[1], x_max + padding)
            
            qr_region = image[y1:y2, x1:x2]
            
            decoded_value, qr_type = decode_qr_codes(qr_region)
            
            qr_list.append({
                "bbox": bbox,
                "value": decoded_value if decoded_value else "DECODE_FAILED",
                "type": qr_type if qr_type else "unknown"
            })
        
        results.append({
            "image_id": image_id,
            "qrs": qr_list
        })
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Stage 2 completed!")
    print(f"📄 Results saved to: {output_file}")
    print(f"📊 Total images processed: {len(results)}")
    print(f"📊 Total QR codes detected: {sum(len(r['qrs']) for r in results)}")
    decoded_count = sum(1 for r in results for qr in r['qrs'] if qr.get('value') != 'DECODE_FAILED')
    print(f"📊 Successfully decoded: {decoded_count}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Multi-QR Code Detection Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing test images')
    parser.add_argument('--output', type=str, default='outputs/submission_detection_1.json',
                        help='Output JSON file path')
    parser.add_argument('--model', type=str, default='outputs/qr_detection/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                        help='Stage 1: Detection only, Stage 2: Detection + Decoding')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detection')
    
    args = parser.parse_args()
    
    model = load_model(args.model)
    
    if args.stage == 1:
        run_stage1_detection(model, args.input, args.output, args.conf)
    else:
        run_stage2_decoding(model, args.input, args.output, args.conf)

if __name__ == "__main__":
    main()