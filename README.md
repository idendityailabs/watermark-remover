# Text Remover

A Python tool for removing text and watermarks from images using computer vision techniques.

## Features
- Detects and removes watermarks and overlaid text
- Preserves text that's part of the image content
- Supports parallel processing for batch operations
- Works with jpg, jpeg, png, and webp images


To start the environment:
```bash
python -m venv venv
source venv/bin/avtivate
pip install -r requirements.txt
```

## Usage For open cv masking
```bash
# Process a directory of images in parallel
python -m watermark_remover.main --input_dir /path/to/images --output_dir /path/to/output --workers numberof_workers

# Process a single image
python -m watermark_remover.main --single /path/to/image.jpg --output_dir /path/to/outputdir
```

## Usage For yolo masking
```bash
# Process a directory of images in parallel
python -m watermark_remover.yolo_mask_remover --input_dir /path/to/images --output_dir /path/to/output --workers numberof_workers

# Process a single image
python -m watermark_remover.yolo_mask_remover --single /path/to/image.jpg --output_dir /path/to/outputdir
```
