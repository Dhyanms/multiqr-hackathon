"""
QR Code Decoder and Classifier
Decodes QR codes and classifies them by type
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import re

def preprocess_qr_region(image):
    """
    Preprocess QR code region for better decoding
    
    Args:
        image: QR code region (BGR)
    
    Returns:
        list: Preprocessed image variants
    """
    variants = []
    
    # Original
    variants.append(image.copy())
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(gray)
    
    # Binary threshold (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary)
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(sharpened)
    
    # Denoise + threshold
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, denoised_binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(denoised_binary)
    
    # Morphological operations
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph)
    variants.append(morph)
    
    return variants

def classify_qr_type(decoded_text):
    """
    Classify QR code based on decoded content
    
    Args:
        decoded_text: Decoded QR string
    
    Returns:
        str: QR type classification
    """
    if not decoded_text:
        return "unknown"
    
    text_upper = decoded_text.upper()
    
    # Batch number patterns
    batch_patterns = [
        r'\bBATCH\b', r'\bLOT\b', r'\bB\d+\b', r'\bLOT\d+\b',
        r'\bBN\d+\b', r'\bBATCH\s*[:#]?\s*[A-Z0-9]+\b'
    ]
    for pattern in batch_patterns:
        if re.search(pattern, text_upper):
            return "batch_number"
    
    # Manufacturer patterns
    mfr_patterns = [
        r'\bMFR\b', r'\bMFG\b', r'\bMANUFACTURER\b', r'\bMADE\s+BY\b',
        r'\bCOMPANY\b', r'\bPRODUCED\s+BY\b'
    ]
    for pattern in mfr_patterns:
        if re.search(pattern, text_upper):
            return "manufacturer"
    
    # Distributor patterns
    dist_patterns = [
        r'\bDIST\b', r'\bDISTRIBUTOR\b', r'\bDISTRIBUTED\s+BY\b',
        r'\bWHOLESALER\b', r'\bSUPPLIER\b'
    ]
    for pattern in dist_patterns:
        if re.search(pattern, text_upper):
            return "distributor"
    
    # Regulatory patterns
    reg_patterns = [
        r'\bFDA\b', r'\bCE\b', r'\bISO\b', r'\bREG\b',
        r'\bAPPROVED\b', r'\bCERTIFIED\b', r'\bLICENSE\b',
        r'\bREGISTRATION\b'
    ]
    for pattern in reg_patterns:
        if re.search(pattern, text_upper):
            return "regulatory"
    
    # Product code patterns
    product_patterns = [
        r'\bPROD\b', r'\bSKU\b', r'\bITEM\b', r'\bCODE\b',
        r'\bP\d+\b', r'\bPRODUCT\s+CODE\b'
    ]
    for pattern in product_patterns:
        if re.search(pattern, text_upper):
            return "product_code"
    
    # URL patterns
    if re.search(r'https?://', decoded_text, re.IGNORECASE):
        return "url"
    
    # Serial number patterns
    if re.search(r'\bSERIAL\b|\bSN\b|\bS/N\b', text_upper):
        return "serial_number"
    
    # Expiry date patterns
    if re.search(r'\bEXP\b|\bEXPIRY\b|\bEXPIRES\b|\bUSE\s+BY\b', text_upper):
        return "expiry_date"
    
    # Default classification based on content type
    if re.match(r'^[A-Z0-9]{6,}$', decoded_text):
        return "alphanumeric_code"
    elif re.match(r'^\d+$', decoded_text):
        return "numeric_code"
    
    return "general"

def decode_qr_codes(qr_region):
    """
    Decode QR code from image region with multiple preprocessing attempts
    
    Args:
        qr_region: Image region containing QR code (BGR or grayscale)
    
    Returns:
        tuple: (decoded_text, qr_type) or (None, None) if decoding fails
    """
    if qr_region is None or qr_region.size == 0:
        return None, None
    
    # Preprocess image
    variants = preprocess_qr_region(qr_region)
    
    # Try decoding with each variant
    for variant in variants:
        try:
            decoded_objects = pyzbar.decode(variant)
            
            if decoded_objects:
                # Get the first decoded QR code
                qr = decoded_objects[0]
                decoded_text = qr.data.decode('utf-8', errors='ignore')
                
                # Classify QR type
                qr_type = classify_qr_type(decoded_text)
                
                return decoded_text, qr_type
        except Exception as e:
            continue
    
    # Try with rotations if initial attempts fail
    for angle in [90, 180, 270]:
        try:
            rotated = cv2.rotate(qr_region, 
                               cv2.ROTATE_90_CLOCKWISE if angle == 90 
                               else cv2.ROTATE_180 if angle == 180 
                               else cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            decoded_objects = pyzbar.decode(rotated)
            if decoded_objects:
                qr = decoded_objects[0]
                decoded_text = qr.data.decode('utf-8', errors='ignore')
                qr_type = classify_qr_type(decoded_text)
                return decoded_text, qr_type
        except Exception as e:
            continue
    
    # Decoding failed
    return None, None

def decode_qr_batch(image_path, bboxes):
    """
    Decode multiple QR codes from an image given bounding boxes
    
    Args:
        image_path: Path to image
        bboxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...]
    
    Returns:
        list: [(decoded_text, qr_type), ...] for each bbox
    """
    image = cv2.imread(image_path)
    if image is None:
        return [(None, None)] * len(bboxes)
    
    results = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        
        # Add padding
        padding = 10
        y1 = max(0, y_min - padding)
        y2 = min(image.shape[0], y_max + padding)
        x1 = max(0, x_min - padding)
        x2 = min(image.shape[1], x_max + padding)
        
        qr_region = image[y1:y2, x1:x2]
        decoded_text, qr_type = decode_qr_codes(qr_region)
        results.append((decoded_text, qr_type))
    
    return results