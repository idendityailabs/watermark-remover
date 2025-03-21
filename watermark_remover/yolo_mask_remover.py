import os
import time
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing
import cv2
import numpy as np
from PIL import Image
import torch
from simple_lama_inpainting import SimpleLama
from transformers import Owlv2VisionModel
from ultralytics import YOLO
import torchvision.transforms.functional as TVF
from watermark_remover.models import DetectorModelOwl
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

simple_lama = SimpleLama()

# Load OWLv2 classification model
model = DetectorModelOwl("google/owlv2-base-patch16-ensemble", dropout=0.0)
model.load_state_dict(torch.load("far5y1y5-8000.pt", map_location=device))
model.eval().to(device)

# Load YOLO model
yolo_model = YOLO("yolo11x-train28-best.pt")
yolo_model.to(device)

def owl_predict(image: Image.Image) -> bool:
    big_side = max(image.size)
    new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
    new_image.paste(image, (0, 0))

    preped = new_image.resize((960, 960), Image.BICUBIC)
    preped = TVF.pil_to_tensor(preped).float() / 255.0
    input_image = TVF.normalize(preped, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    logits, = model(input_image.to(device).unsqueeze(0), None)
    probs = F.softmax(logits, dim=1)
    prediction = torch.argmax(probs.cpu(), dim=1)

    return prediction.item() == 1

def yolo_predict(image: Image.Image, save_path: str) -> None:
    results = yolo_model(image, imgsz=1024, augment=True, iou=0.5)
    result = results[0]

    # Create a black mask the same size as the input image
    width, height = image.size
    mask = Image.new("L", (width, height), 0)

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Draw white rectangle on the mask
        for x in range(x1, x2):
            for y in range(y1, y2):
                if 0 <= x < width and 0 <= y < height:
                    mask.putpixel((x, y), 255)

    mask.save(save_path)

def remove_text_with_lama(image_path, mask_path, output_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path).convert('L')

    result = simple_lama(image, mask)
    result.save(output_path)


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
        cleaned_image = remove_text_with_lama(input_path, mask_path, output_path)
        
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

def generate_text_mask(image_path, output_mask_path):
    image = Image.open(image_path).convert("RGB")
    owl_result = owl_predict(image)
    yolo_predict(image, output_mask_path)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
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
            cleaned_image = remove_text_with_lama(args.single, "temp_mask.jpg", 
                          os.path.join(args.output_dir, f"cleaned2_{Path(args.single).name}"))
            
            print(f"Successfully processed image")
            if os.path.exists("temp_mask.jpg"):
                os.remove("temp_mask.jpg")
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        # Parallel processing of directory
        process_directory(args.input_dir, args.output_dir, num_workers=args.workers)
