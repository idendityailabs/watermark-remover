import os
import time
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import easyocr

def enhance_image(image):
    # Increase contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Additional preprocessing
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def is_likely_watermark(bbox, image_shape, text):
    """
    Determine if detected text is likely to be a watermark based on position and characteristics
    """
    height, width = image_shape[:2]
    (top_left, top_right, bottom_right, bottom_left) = bbox
    x1, y1 = map(int, top_left)
    x2, y2 = map(int, bottom_right)
    
    # Calculate text region properties
    text_width = x2 - x1
    text_height = y2 - y1
    text_area = text_width * text_height
    image_area = height * width
    
    # Check if text is in corner regions (common for watermarks)
    corner_threshold = 0.2  # 20% from edges
    in_corner = (x1 < width * corner_threshold or x2 > width * (1 - corner_threshold)) and \
                (y1 < height * corner_threshold or y2 > height * (1 - corner_threshold))
    
    # Check if text is along edges
    edge_threshold = 0.1  # 10% from edges
    on_edge = (y1 < height * edge_threshold or y2 > height * (1 - edge_threshold) or
              x1 < width * edge_threshold or x2 > width * (1 - edge_threshold))
    
    # Check text size relative to image
    rel_size = text_area / image_area
    too_large = rel_size > 0.3  # Text taking up more than 30% is likely not a watermark
    too_small = rel_size < 0.0001  # Extremely small text might be noise
    
    # Common watermark patterns
    watermark_patterns = ['copyright', '©', 'all rights reserved', 'watermark', 
                         'sample', 'preview', 'property of', 'stock']
    contains_watermark_text = any(pattern in text.lower() for pattern in watermark_patterns)
    
    # Decision logic
    if contains_watermark_text:
        return True
    if too_large or too_small:
        return False
    if in_corner or on_edge:
        return True
        
    return False

def generate_text_mask(image_path, output_mask_path):
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")

    # Create enhanced version for better detection
    enhanced_image = enhance_image(image)
    
    # Create initial mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Initialize EasyOCR with English only
    reader = easyocr.Reader(['en'])
    
    # Try detection on both original and enhanced images
    results = reader.readtext(image_path)
    results_enhanced = reader.readtext(enhanced_image)
    
    # Combine results
    all_results = results + results_enhanced

    # Draw detected regions from EasyOCR
    for (bbox, text, prob) in all_results:
        if prob > 0.3 and len(text.strip()) > 3 and is_likely_watermark(bbox, image.shape, text):
        # if prob > 0.05 and is_likely_watermark(bbox, image.shape, text):  # Check if it's likely a watermark
            (top_left, top_right, bottom_right, bottom_left) = bbox
            points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
            # Draw filled polygon
            cv2.fillPoly(mask, [points], 255)
            # Add padding around the text
            x1, y1 = map(int, top_left)
            x2, y2 = map(int, bottom_right)
            padding = 15
            cv2.rectangle(mask, 
                         (max(0, x1-padding), max(0, y1-padding)), 
                         (min(mask.shape[1], x2+padding), min(mask.shape[0], y2+padding)), 
                         255, -1)

    # Dilate the mask more aggressively
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Save the generated mask
    cv2.imwrite(output_mask_path, mask)
    return mask

def remove_text_with_inpainting(image_path, mask_path, output_path):
    # Load the original image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Multiple inpainting passes with different parameters
    radius = 20
    # First pass with NS
    cleaned_image = cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
    # Second pass with TELEA
    cleaned_image = cv2.inpaint(cleaned_image, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    # Third pass with NS again
    cleaned_image = cv2.inpaint(cleaned_image, mask, inpaintRadius=radius//2, flags=cv2.INPAINT_NS)

    # Save result
    cv2.imwrite(output_path, cleaned_image)
    return cleaned_image

def process_single_image(args):
    """
    Process a single image - used for parallel processing
    """
    input_path, output_dir = args
    try:
        # Create output filename
        output_filename = f"{Path(input_path).name}"
        mask_filename = f"mask_{Path(input_path).name}"
        output_path = os.path.join(output_dir, output_filename)
        mask_path = os.path.join(output_dir, mask_filename)
        
        # Process the image
        mask = generate_text_mask(input_path, mask_path)
        cleaned_image = remove_text_with_inpainting(input_path, mask_path, output_path)
        
        # Remove temporary mask file
        if os.path.exists(mask_path):
            os.remove(mask_path)
            
        return True, input_path
    except Exception as e:
        return False, f"Error processing {input_path}: {str(e)}"

def process_directory(input_dir, output_dir, num_workers=None):
    """
    Process all images in a directory using parallel processing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [
        f for f in Path(input_dir).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Processing {len(image_files)} images using {num_workers} workers...")
    start_time = time.time()
    
    # Prepare arguments for parallel processing
    args = [(str(f), output_dir) for f in image_files]
    
    # Process images in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_single_image, args)
    
    # Process results
    successful = 0
    failed = 0
    for success, message in results:
        if success:
            successful += 1
        else:
            failed += 1
            print(message)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Successfully processed: {successful} images")
    print(f"Failed: {failed} images")
    print(f"Average time per image: {duration/len(image_files):.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove text/watermarks from images')
    parser.add_argument('--input_dir', type=str, help='Input directory with images')
    parser.add_argument('--output_dir', type=str, help='Output directory for cleaned images')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--single', type=str, help='Process a single image instead of directory')
    
    args = parser.parse_args()
    
    if args.single:
        # Single image processing
        try:
            mask = generate_text_mask(args.single, "temp_mask.jpg")
            cleaned_image = remove_text_with_inpainting(args.single, "temp_mask.jpg", 
                          os.path.join(args.output_dir, f"cleaned_{Path(args.single).name}"))
            print(f"Successfully processed image")
            if os.path.exists("temp_mask.jpg"):
                os.remove("temp_mask.jpg")
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        # Parallel processing of directory
        process_directory(args.input_dir, args.output_dir, num_workers=args.workers)
