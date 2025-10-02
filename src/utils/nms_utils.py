"""
Non-Maximum Suppression (NMS) Utilities
Custom NMS implementation for handling overlapping QR code detections
"""

import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1, box2: [x_min, y_min, x_max, y_max]
    
    Returns:
        float: IoU value
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_box_area(box):
    """Calculate area of a bounding box"""
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)

def apply_custom_nms(bboxes, iou_threshold=0.45):
    """
    Apply custom Non-Maximum Suppression to remove overlapping detections
    
    For medicine packs, we want to keep all distinct QR codes while removing
    duplicates from the same QR code detected multiple times.
    
    Args:
        bboxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
        iou_threshold: IoU threshold for suppression
    
    Returns:
        list: Filtered bounding boxes
    """
    if len(bboxes) == 0:
        return []
    
    # Convert to numpy array
    boxes = np.array(bboxes, dtype=np.float32)
    
    # Calculate areas
    areas = np.array([calculate_box_area(box) for box in boxes])
    
    # Sort by area (larger boxes first - they're more likely to be complete QR codes)
    order = areas.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0:
        # Pick the box with largest area
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in order[1:]])
        
        # Keep boxes with IoU less than threshold
        remaining = np.where(ious < iou_threshold)[0]
        order = order[remaining + 1]
    
    return [bboxes[i] for i in keep]

def merge_close_boxes(bboxes, distance_threshold=20):
    """
    Merge bounding boxes that are very close to each other
    Useful for handling fragmented detections of the same QR code
    
    Args:
        bboxes: List of bounding boxes
        distance_threshold: Maximum distance between box centers to merge
    
    Returns:
        list: Merged bounding boxes
    """
    if len(bboxes) <= 1:
        return bboxes
    
    def get_center(box):
        x_min, y_min, x_max, y_max = box
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)
    
    def distance(center1, center2):
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def merge_two_boxes(box1, box2):
        x_min = min(box1[0], box2[0])
        y_min = min(box1[1], box2[1])
        x_max = max(box1[2], box2[2])
        y_max = max(box1[3], box2[3])
        return [x_min, y_min, x_max, y_max]
    
    merged = []
    used = set()
    
    for i in range(len(bboxes)):
        if i in used:
            continue
        
        current_box = bboxes[i]
        current_center = get_center(current_box)
        
        # Find boxes to merge with current box
        to_merge = [current_box]
        used.add(i)
        
        for j in range(i + 1, len(bboxes)):
            if j in used:
                continue
            
            other_center = get_center(bboxes[j])
            if distance(current_center, other_center) < distance_threshold:
                to_merge.append(bboxes[j])
                used.add(j)
        
        # Merge all close boxes
        if len(to_merge) == 1:
            merged.append(current_box)
        else:
            merged_box = to_merge[0]
            for box in to_merge[1:]:
                merged_box = merge_two_boxes(merged_box, box)
            merged.append(merged_box)
    
    return merged