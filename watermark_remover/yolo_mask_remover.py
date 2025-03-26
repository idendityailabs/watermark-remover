import os
import time
import argparse
from pathlib import Path
import multiprocessing
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Owlv2VisionModel
from ultralytics import YOLO
import torchvision.transforms.functional as TVF
from watermark_remover.models import DetectorModelOwl
import torch.nn.functional as F
from tqdm import tqdm
import queue
import threading

# Global variables for GPU workers
_LOADED_MODELS = {}  # Cache models per GPU

def remove_text_with_opencv(image, mask, inpaint_radius=7, method=cv2.INPAINT_TELEA):
    """
    Replace SimpleLama with OpenCV inpainting
    
    Parameters:
    - image: PIL Image
    - mask: PIL Image (binary mask where white pixels need inpainting)
    - inpaint_radius: The radius of the neighborhood area to be inpainted (higher = smoother but slower)
    - method: cv2.INPAINT_TELEA or cv2.INPAINT_NS (TELEA is usually faster)
    
    Returns:
    - Inpainted PIL Image
    """
    # Convert PIL images to OpenCV format
    cv_image = np.array(image)
    cv_mask = np.array(mask)
    
    # Ensure mask is binary (255 for areas to inpaint)
    _, cv_mask = cv2.threshold(cv_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Inpaint
    result = cv2.inpaint(cv_image, cv_mask, inpaint_radius, method)
    
    # Convert back to PIL
    return Image.fromarray(result)

def process_batch_on_gpu(gpu_id, image_paths, output_dir, use_owl=True, inpaint_radius=7, inpaint_method=cv2.INPAINT_TELEA):
    """
    Process a batch of images on a specific GPU.
    This function will be called by a dedicated process for each GPU.
    """
    device = torch.device(f"cuda:{gpu_id}")
    print(f"GPU {gpu_id}: Starting worker. Processing {len(image_paths)} images")
    
    # Load models only once for this GPU (no SimpleLama anymore)
    if gpu_id not in _LOADED_MODELS:
        print(f"GPU {gpu_id}: Loading models")
        
        # Load OWLv2 classification model if needed
        if use_owl:
            owl_model = DetectorModelOwl("google/owlv2-base-patch16-ensemble", dropout=0.0)
            owl_model.load_state_dict(torch.load("far5y1y5-8000.pt", map_location=device))
            owl_model.eval().to(device)
        else:
            owl_model = None
        
        # Load YOLO model
        yolo_model = YOLO("yolo11x-train28-best.pt")
        yolo_model.to(device)
        
        _LOADED_MODELS[gpu_id] = (owl_model, yolo_model)
    else:
        owl_model, yolo_model = _LOADED_MODELS[gpu_id]
    
    # Process each image in the batch
    results = []
    pbar = tqdm(image_paths, desc=f"GPU {gpu_id}")
    for input_path in pbar:
        try:
            # Setup paths
            output_filename = f"{Path(input_path).name}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Open image
            image = Image.open(input_path).convert("RGB")
            
            # OWL classification (optional)
            if use_owl and owl_model:
                owl_result = owl_predict(image, owl_model, device)
            
            # YOLO detection for mask creation
            # Create in-memory mask without saving to disk
            mask = create_mask_from_yolo(image, yolo_model, device)
            
            # Use OpenCV inpainting instead of SimpleLama
            result = remove_text_with_opencv(image, mask, inpaint_radius, inpaint_method)
            result.save(output_path)
            
            results.append((True, input_path))
        except Exception as e:
            results.append((False, f"Error processing {input_path}: {str(e)}"))
            pbar.set_description(f"GPU {gpu_id}: Error - {str(e)[:30]}...")
    
    print(f"GPU {gpu_id}: Finished processing batch of {len(image_paths)} images")
    return results


def owl_predict(image: Image.Image, model, device) -> bool:
    big_side = max(image.size)
    new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
    new_image.paste(image, (0, 0))

    preped = new_image.resize((960, 960), Image.BICUBIC)
    preped = TVF.pil_to_tensor(preped).float() / 255.0
    input_image = TVF.normalize(preped, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    with torch.no_grad():  # Add this to reduce memory usage
        logits, = model(input_image.to(device).unsqueeze(0), None)
    
    probs = F.softmax(logits, dim=1)
    prediction = torch.argmax(probs.cpu(), dim=1)

    return prediction.item() == 1

def create_mask_from_yolo(image: Image.Image, yolo_model, device) -> Image.Image:
    """Create a mask from YOLO detection without saving to disk"""
    with torch.no_grad():  # Add this to reduce memory usage
        results = yolo_model(image, imgsz=1024, augment=True, iou=0.5)
    
    result = results[0]

    # Create a mask with NumPy (much faster than pixel-by-pixel)
    width, height = image.size
    mask_array = np.zeros((height, width), dtype=np.uint8)
    
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        mask_array[y1:y2, x1:x2] = 255
    
    return Image.fromarray(mask_array)

def process_directory(input_dir, output_dir, use_owl=True, inpaint_radius=7, inpaint_method=cv2.INPAINT_TELEA):
    """Process all images in a directory with a worker per GPU"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = [
        str(f) for f in Path(input_dir).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Get GPU count
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPUs")
    
    if gpu_count == 0:
        print("No GPUs found. Using CPU only (this will be very slow)")
        process_batch_on_gpu(0, image_files, output_dir, use_owl, inpaint_radius, inpaint_method)
        return
    
    # Split images across GPUs
    batches = split_list(image_files, gpu_count)
    print(f"Split {len(image_files)} images into {len(batches)} batches")
    
    # Process batches in parallel, one process per GPU
    start_time = time.time()
    
    # Create and start processes
    processes = []
    for gpu_id, batch in enumerate(batches):
        if len(batch) == 0:
            continue
        
        p = multiprocessing.Process(
            target=process_batch_on_gpu,
            args=(gpu_id, batch, output_dir, use_owl, inpaint_radius, inpaint_method)
        )
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Print summary
    print(f"\nProcessing complete!")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average time per image: {duration/len(image_files):.2f} seconds")

def split_list(lst, n):
    """Split a list into n approximately equal parts"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def single_image_processing(input_path, output_dir, use_owl=True, inpaint_radius=7, inpaint_method=cv2.INPAINT_TELEA):
    """Process a single image"""
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        gpu_id = None
    else:
        gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
    
    print(f"Processing single image on {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models (no SimpleLama anymore)
    if use_owl:
        owl_model = DetectorModelOwl("google/owlv2-base-patch16-ensemble", dropout=0.0)
        owl_model.load_state_dict(torch.load("far5y1y5-8000.pt", map_location=device))
        owl_model.eval().to(device)
    else:
        owl_model = None
    
    yolo_model = YOLO("yolo11x-train28-best.pt")
    yolo_model.to(device)
    
    try:
        # Create output filename
        output_filename = f"cleaned_{Path(input_path).name}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Process the image
        image = Image.open(input_path).convert("RGB")
        
        # OWL classification (optional)
        if use_owl and owl_model:
            owl_result = owl_predict(image, owl_model, device)
        
        # YOLO detection
        mask = create_mask_from_yolo(image, yolo_model, device)
        
        # Use OpenCV inpainting instead of SimpleLama
        result = remove_text_with_opencv(image, mask, inpaint_radius, inpaint_method)
        result.save(output_path)
        
        print(f"Successfully processed image to {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Remove text/watermarks from images')
    parser.add_argument('--input_dir', type=str, help='Input directory with images')
    parser.add_argument('--output_dir', type=str, help='Output directory for cleaned images')
    parser.add_argument('--no_owl', action='store_true', help='Skip OWL classification to speed up processing')
    parser.add_argument('--single', type=str, help='Process a single image instead of directory')
    parser.add_argument('--inpaint_radius', type=int, default=7, 
                        help='Radius for OpenCV inpainting (default: 7, higher = smoother but slower)')
    parser.add_argument('--inpaint_method', type=str, choices=['telea', 'ns'], default='telea',
                        help='OpenCV inpainting method: telea (faster) or ns (better quality)')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = "./output"
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine whether to use OWL
    use_owl = not args.no_owl
    
    # Set inpainting method
    inpaint_method = cv2.INPAINT_TELEA if args.inpaint_method == 'telea' else cv2.INPAINT_NS

    if args.single:
        # Single image processing
        single_image_processing(args.single, args.output_dir, use_owl, args.inpaint_radius, inpaint_method)
    else:
        # One worker per GPU processing
        process_directory(args.input_dir, args.output_dir, use_owl, args.inpaint_radius, inpaint_method)
