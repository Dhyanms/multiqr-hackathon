"""
Data Utilities
Helper functions for data processing and visualization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_detections(image_path, bboxes, output_path=None, show=False):
    """
    Visualize QR code detections on image
    
    Args:
        image_path: Path to input image
        bboxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
        output_path: Path to save visualization (optional)
        show: Whether to display image
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Draw bounding boxes
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add label
        label = f"QR {i+1}"
        cv2.putText(image, label, (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to {output_path}")
    
    # Display if requested
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"QR Detections: {len(bboxes)} codes found")
        plt.tight_layout()
        plt.show()

def visualize_detections_with_labels(image_path, detections, output_path=None, show=False):
    """
    Visualize QR code detections with decoded values
    
    Args:
        image_path: Path to input image
        detections: List of dicts with 'bbox', 'value', 'type'
        output_path: Path to save visualization
        show: Whether to display image
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Draw bounding boxes with labels
    for i, det in enumerate(detections):
        bbox = det['bbox']
        value = det.get('value', 'N/A')
        qr_type = det.get('type', 'unknown')
        
        x_min, y_min, x_max, y_max = map(int, bbox)
        
        # Color based on decode success
        color = (0, 255, 0) if value != 'DECODE_FAILED' else (0, 165, 255)
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Add labels
        label1 = f"QR {i+1}: {qr_type}"
        label2 = f"{value[:20]}..." if len(value) > 20 else value
        
        # Background for text
        (w1, h1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (w2, h2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        cv2.rectangle(image, (x_min, y_min - h1 - h2 - 20), 
                     (x_min + max(w1, w2) + 10, y_min), (0, 0, 0), -1)
        
        cv2.putText(image, label1, (x_min + 5, y_min - h2 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(image, label2, (x_min + 5, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to {output_path}")
    
    # Display if requested
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"QR Detections: {len(detections)} codes found")
        plt.tight_layout()
        plt.show()

def load_yolo_annotations(label_path, img_width, img_height):
    """
    Load YOLO format annotations and convert to bounding boxes
    
    Args:
        label_path: Path to YOLO .txt file
        img_width: Image width
        img_height: Image height
    
    Returns:
        list: Bounding boxes [[x_min, y_min, x_max, y_max], ...]
    """
    if not os.path.exists(label_path):
        return []
    
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                # class_id x_center y_center width height (normalized)
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                
                bboxes.append([x_min, y_min, x_max, y_max])
    
    return bboxes

def create_dataset_summary(dataset_path):
    """
    Create a summary of the dataset
    
    Args:
        dataset_path: Path to dataset directory
    """
    train_images = os.path.join(dataset_path, 'train', 'images')
    train_labels = os.path.join(dataset_path, 'train', 'labels')
    
    if not os.path.exists(train_images):
        print(f"Error: Train images directory not found at {train_images}")
        return
    
    image_files = [f for f in os.listdir(train_images) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(train_labels) if f.endswith('.txt')] if os.path.exists(train_labels) else []
    
    print("=" * 70)
    print("Dataset Summary")
    print("=" * 70)
    print(f"Training Images: {len(image_files)}")
    print(f"Training Labels: {len(label_files)}")
    
    if len(label_files) > 0:
        # Count total QR codes
        total_qr_codes = 0
        qr_per_image = []
        
        for label_file in label_files:
            label_path = os.path.join(train_labels, label_file)
            with open(label_path, 'r') as f:
                lines = [line for line in f if line.strip()]
                count = len(lines)
                total_qr_codes += count
                qr_per_image.append(count)
        
        print(f"\nTotal QR Codes: {total_qr_codes}")
        print(f"Avg QR per Image: {np.mean(qr_per_image):.2f}")
        print(f"Min QR per Image: {np.min(qr_per_image)}")
        print(f"Max QR per Image: {np.max(qr_per_image)}")
    
    print("=" * 70)