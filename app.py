#!/usr/bin/env python3
"""
ML Pipeline for CloudFactory Object Detection Models
====================================================

This script implements a complete ML pipeline for object detection models
1. Image preprocessing with augmentations 
2. Model inference on batch images (using your model.pt)
3. COCO format output generation 
4. Infrastructure teardown

Based on CloudFactory's exact inference patterns for TorchScript models.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision
import cv2
import albumentations as A
from pathlib import Path
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CloudFactoryImageProcessor:
    """
    Step 1: Image preprocessing with augmentations
    
    Why exactly like this:
    - Uses exact transforms.json format
    - Loads with A.load() as shown in documentation
    - Maintains model performance by using same preprocessing
    """
    
    def __init__(self, transforms_config_path: str = None):
        if transforms_config_path and os.path.exists(transforms_config_path):
            logger.info(f"Loading CloudFactory transforms from: {transforms_config_path}")
            self.transforms = A.load(transforms_config_path)
        else:
            logger.warning("No transforms.json found, using default resize to 224x224")
            # Default based on CloudFactory example (224x224 resize)
            self.transforms = A.Compose([
                A.Resize(height=224, width=224, interpolation=1, p=1.0)
            ])
        
        logger.info("Image processor initialized with CloudFactory transforms")
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process single image 
        
        Returns:
            original_image: Original image for COCO metadata
            processed_image: Transformed image ready for model.pt
        """
        # Load image using PIL then convert to numpy 
        pil_image = Image.open(image_path)
        original_image = np.array(pil_image)
        
        # Apply CloudFactory transforms
        transformed = self.transforms(image=original_image)
        processed_image = transformed['image']
        
        return original_image, processed_image


class CloudFactoryObjectDetector:
    """
    Step 2: Model inference using your model.pt
    
    - CloudFactory's exact inference pattern
    - Handles both Faster RCNN and FBNetv3 output formats
    
    """
    
    def __init__(self, model_path: str, class_mapping_path: str):
        # Load exactly like CloudFactory examples
        with open(class_mapping_path) as data:
            mappings = json.load(data)
        
        self.class_mapping = {item['model_idx']: item['class_name'] for item in mappings}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load TorchScript model
        self.model = torch.jit.load(model_path).to(self.device)
        #self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Classes: {list(self.class_mapping.values())}")
    
    def predict_single_image(self, processed_image: np.ndarray, 
                           nms_threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Run inference on single image (CloudFactory pattern)
        
        Args:
            processed_image: Preprocessed image from transforms
            nms_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary with boxes, scores, classes
        """
        # Convert to tensor exactly like CloudFactory examples
        x = torch.from_numpy(processed_image).to(self.device)
        
        with torch.no_grad():
            # Convert to channels first, convert to float (
            x = x.permute(2, 0, 1).float()
            
            # Run model inference
            y = self.model(x)
            
            # Handle different model output formats
            if isinstance(y, dict):
                # Faster RCNN format (CloudFactory example 1)
                pred_boxes = y['pred_boxes']
                scores = y['scores'] 
                pred_classes = y['pred_classes']
                
                # Apply NMS (CloudFactory pattern)
                to_keep = torchvision.ops.nms(pred_boxes, scores, nms_threshold)
                final_boxes = pred_boxes[to_keep]
                final_classes = pred_classes[to_keep]
                final_scores = scores[to_keep]
                
            elif isinstance(y, (tuple, list)) and len(y) >= 4:
                # FBNetv3 format (CloudFactory example 2)
                pred_boxes, pred_classes, scores, _ = y
                
                # Apply NMS (CloudFactory pattern)
                to_keep = torchvision.ops.nms(pred_boxes, scores, nms_threshold)
                final_boxes = pred_boxes[to_keep]
                final_classes = pred_classes[to_keep] 
                final_scores = scores[to_keep]
                
            else:
                raise ValueError(f"Unknown model output format: {type(y)}")
        
        return {
            'boxes': final_boxes.cpu().numpy(),
            'scores': final_scores.cpu().numpy(),
            'classes': final_classes.cpu().numpy()
        }


class COCOResultsExporter:
    """
    Step 3: COCO format output generation

    COCO format because of:
    - Standard format for object detection results
    - Compatible with evaluation tools
    - Easy to analyze and visualize
    - Industry standard for ML competitions and benchmarks
    """
    
    def __init__(self, class_mapping: Dict[int, str]):
        self.coco_output = {
            "info": {
                "description": "CloudFactory Model Inference Results",
                "version": "1.0", 
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.annotation_id = 1
        
        # Add categories from class mapping
        for class_id, class_name in class_mapping.items():
            self.coco_output["categories"].append({
                "id": class_id,
                "name": class_name,
                "supercategory": "object"
            })
        
        logger.info(f"COCO exporter initialized with {len(class_mapping)} categories")
    
    def add_image_results(self, image_id: int, image_path: str, 
                         original_image: np.ndarray, predictions: Dict[str, np.ndarray]):
        """Add single image results to COCO format"""
        height, width = original_image.shape[:2]
        
        # Add image metadata
        self.coco_output["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "height": height,
            "width": width,
            "date_captured": datetime.now().isoformat()
        })
        
        # Add annotations for this image
        boxes = predictions['boxes']
        scores = predictions['scores'] 
        classes = predictions['classes']
        
        for box, score, class_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Only add valid boxes
            if bbox_width > 0 and bbox_height > 0:
                self.coco_output["annotations"].append({
                    "id": self.annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
                    "area": float(bbox_width * bbox_height),
                    "iscrowd": 0,
                    "score": float(score)
                })
                self.annotation_id += 1
    
    def save_coco_results(self, output_path: str) -> str:
        """Save results in COCO format"""
        with open(output_path, 'w') as f:
            json.dump(self.coco_output, f, indent=2)
        
        num_images = len(self.coco_output['images'])
        num_annotations = len(self.coco_output['annotations'])
        
        logger.info(f"COCO results saved: {output_path}")
        logger.info(f"Summary: {num_images} images, {num_annotations} detections")
        
        return output_path


class PipelineInfrastructure:
    """
    Step 4: Infrastructure teardown and resource management
    
    Why this is important:
    - Cleans up GPU memory after processing
    - Removes temporary files
    - Logs final statistics
    - Ensures clean exit for automated deployment
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.temp_files = []
        
    def register_temp_file(self, file_path: str):
        """Register temporary file for cleanup"""
        self.temp_files.append(file_path)
    
    def teardown(self):
        """Clean up all resources"""
        logger.info("Starting infrastructure teardown...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
        
        # Clean temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Removed: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove {temp_file}: {e}")
        
        # Final statistics
        total_time = time.time() - self.start_time
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        logger.info("Infrastructure teardown completed")


def run_pipeline(images_dir: str, model_path: str, class_mapping_path: str, 
                output_dir: str, transforms_config: str = None, 
                nms_threshold: float = 0.5) -> str:
    """
    Main pipeline function - processes batch of images through CloudFactory model
    
    Returns:
        Path to generated COCO results file
    """
    
    # Initialize infrastructure
    infra = PipelineInfrastructure(output_dir)
    
    try:
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(images_dir).glob(f"*{ext}"))
            image_paths.extend(Path(images_dir).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in sorted(image_paths)]
        
        if not image_paths:
            raise ValueError(f"No images found in {images_dir}")
        
        logger.info("="*60)
        logger.info("STARTING CLOUDACTORY ML PIPELINE")
        logger.info("="*60)
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Step 1: Initialize image processor
        logger.info("STEP 1: Initializing image processor with CloudFactory transforms")
        processor = CloudFactoryImageProcessor(transforms_config)
        
        # Step 2: Initialize object detector
        logger.info("STEP 2: Loading CloudFactory object detection model")
        detector = CloudFactoryObjectDetector(model_path, class_mapping_path)
        
        # Step 3: Initialize COCO exporter  
        logger.info("STEP 3: Initializing COCO format exporter")
        exporter = COCOResultsExporter(detector.class_mapping)
        
        # Process each image (following CloudFactory single-image pattern)
        logger.info("Processing images (single image inference pattern)...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                logger.info(f"Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Step 1: Image preprocessing 
                original_image, processed_image = processor.process_image(image_path)
                
                # Step 2: Model inference (CloudFactory single-image pattern)
                predictions = detector.predict_single_image(processed_image, nms_threshold)
                
                # Step 3: Add to COCO format
                exporter.add_image_results(i, image_path, original_image, predictions)
                
                # Log detection count for this image
                num_detections = len(predictions['boxes'])
                if num_detections > 0:
                    logger.info(f"  → Found {num_detections} detections")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        # Save COCO results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = Path(output_dir) / f"coco_results_{timestamp}.json"
        exporter.save_coco_results(str(results_file))
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return str(results_file)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Step 4: Infrastructure teardown
        logger.info("STEP 4: Infrastructure teardown") 
        infra.teardown()


def main():
    """Main execution - simplified for your exact use case"""
    parser = argparse.ArgumentParser(description="CloudFactory ML Pipeline - Batch Image Processing")
    
    # Required arguments
    parser.add_argument('--images_dir', required=True,
                       help='Directory containing images to process')
    parser.add_argument('--model_path', required=True,
                       help='Path to your model.pt file') 
    parser.add_argument('--class_mapping_path', required=True,
                       help='Path to class_mapping.json file')
    parser.add_argument('--output_dir', default='./output',
                       help='Output directory (default: ./output)')
    
    # Optional arguments
    parser.add_argument('--transforms_config', 
                       help='Path to transforms.json (optional)')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help='NMS IoU threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate required files exist
    for path, name in [(args.model_path, "model.pt"), 
                       (args.class_mapping_path, "class_mapping.json"),
                       (args.images_dir, "images directory")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Run the pipeline
    results_path = run_pipeline(
        images_dir=args.images_dir,
        model_path=args.model_path, 
        class_mapping_path=args.class_mapping_path,
        output_dir=args.output_dir,
        transforms_config=args.transforms_config,
        nms_threshold=args.nms_threshold
    )
    
    print(f"\n SUCCESS! Results saved to: {results_path}")


if __name__ == "__main__":
    main()
