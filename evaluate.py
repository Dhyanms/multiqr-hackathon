"""
Multi-QR Code Detection - Evaluation Script
Evaluates detection performance against ground truth annotations
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1, box2: [x_min, y_min, x_max, y_max]
    
    Returns:
        float: IoU score
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def yolo_to_bbox(yolo_line, img_width, img_height):
    """
    Convert YOLO format to bounding box [x_min, y_min, x_max, y_max]
    
    Args:
        yolo_line: "class_id x_center y_center width height" (normalized)
        img_width, img_height: Image dimensions
    
    Returns:
        list: [x_min, y_min, x_max, y_max]
    """
    parts = yolo_line.strip().split()
    x_center = float(parts[1]) * img_width
    y_center = float(parts[2]) * img_height
    width = float(parts[3]) * img_width
    height = float(parts[4]) * img_height
    
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    return [x_min, y_min, x_max, y_max]

def load_ground_truth(label_dir, image_dir):
    """
    Load ground truth annotations from YOLO format labels
    
    Returns:
        dict: {image_id: [bbox1, bbox2, ...]}
    """
    ground_truth = {}
    
    if not os.path.exists(label_dir):
        print(f"⚠️  Label directory not found: {label_dir}")
        return ground_truth
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        image_id = os.path.splitext(label_file)[0]
        label_path = os.path.join(label_dir, label_file)
        
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(image_dir, image_id + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    bbox = yolo_to_bbox(line, img_width, img_height)
                    bboxes.append(bbox)
        
        ground_truth[image_id] = bboxes
    
    return ground_truth

def evaluate_detection(predictions_file, ground_truth, iou_threshold=0.5):
    """
    Evaluate detection performance
    
    Args:
        predictions_file: Path to submission JSON
        ground_truth: Dict of ground truth bounding boxes
        iou_threshold: IoU threshold for matching
    
    Returns:
        dict: Evaluation metrics
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    total_gt = 0
    total_pred = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred in predictions:
        image_id = pred['image_id']
        pred_boxes = [qr['bbox'] for qr in pred['qrs']]
        
        if image_id not in ground_truth:
            # No ground truth for this image
            false_positives += len(pred_boxes)
            total_pred += len(pred_boxes)
            continue
        
        gt_boxes = ground_truth[image_id]
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
        
        false_negatives += len(gt_boxes) - len(matched_gt)
    
    precision = true_positives / total_pred if total_pred > 0 else 0
    recall = true_positives / total_gt if total_gt > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'iou_threshold': iou_threshold
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate QR Code Detection')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to ground truth labels directory')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for matching (default: 0.5)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Multi-QR Code Detection - Evaluation")
    print("=" * 70)
    
    print(f"\n📂 Loading ground truth from: {args.labels}")
    ground_truth = load_ground_truth(args.labels, args.images)
    print(f"   Loaded {len(ground_truth)} images with annotations")
    
    if len(ground_truth) == 0:
        print("\n❌ No ground truth found. Cannot evaluate.")
        return
    
    print(f"\n🔍 Evaluating predictions from: {args.predictions}")
    metrics = evaluate_detection(args.predictions, ground_truth, args.iou)
    
    print("\n📊 Evaluation Results:")
    print("=" * 70)
    print(f"IoU Threshold:      {metrics['iou_threshold']:.2f}")
    print(f"Precision:          {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:             {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:           {metrics['f1_score']:.4f}")
    print("-" * 70)
    print(f"True Positives:     {metrics['true_positives']}")
    print(f"False Positives:    {metrics['false_positives']}")
    print(f"False Negatives:    {metrics['false_negatives']}")
    print(f"Total GT Boxes:     {metrics['total_gt']}")
    print(f"Total Pred Boxes:   {metrics['total_pred']}")
    print("=" * 70)
    
    output_dir = os.path.dirname(args.predictions)
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()