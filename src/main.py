"""
Main Computer Vision Application
Command-line interface for CV tasks
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from cv_pipeline import CVPipeline
from models.classifier import ImageClassifier
from models.detector import ObjectDetector
from models.segmentation import ImageSegmentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Computer Vision Tool')
    parser.add_argument('--task', type=str, required=True,
                       choices=['classify', 'detect', 'segment', 'pipeline'],
                       help='CV task to perform')
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--video', type=str, help='Input video path')
    parser.add_argument('--folder', type=str, help='Input folder path')
    parser.add_argument('--model', type=str, default='resnet50',
                       help='Model to use')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CV pipeline
    logger.info(f"Initializing CV pipeline for task: {args.task}")
    pipeline = CVPipeline(device=args.device)
    
    try:
        if args.image:
            # Process single image
            process_single_image(pipeline, args)
        elif args.video:
            # Process video
            process_video(pipeline, args)
        elif args.folder:
            # Process folder of images
            process_folder(pipeline, args)
        else:
            # Interactive mode
            interactive_mode(pipeline, args)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

def process_single_image(pipeline, args):
    """Process a single image"""
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return
    
    logger.info(f"Processing image: {args.image}")
    
    if args.task == 'classify':
        result = pipeline.classify_image(args.image, model_name=args.model)
        print(f"\nClassification Results:")
        print(f"Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
    elif args.task == 'detect':
        detections = pipeline.detect_objects(args.image, 
                                           model_name=args.model,
                                           confidence_threshold=args.confidence)
        print(f"\nDetection Results:")
        print(f"Found {len(detections)} objects:")
        for i, detection in enumerate(detections):
            print(f"{i+1}. {detection['class']} ({detection['confidence']:.3f})")
            print(f"   Bbox: {detection['bbox']}")
    
    elif args.task == 'segment':
        result = pipeline.segment_image(args.image, model_name=args.model)
        print(f"\nSegmentation Results:")
        print(f"Segmentation mask shape: {result['segmentation_mask'].shape}")
        print(f"Number of classes: {result['num_classes']}")
        
        # Save visualization if output specified
        if args.output:
            pipeline.visualize_segmentation(args.image, result, args.output)
            logger.info(f"Segmentation visualization saved to: {args.output}")

def process_video(pipeline, args):
    """Process video file"""
    if not os.path.exists(args.video):
        logger.error(f"Video not found: {args.video}")
        return
    
    logger.info(f"Processing video: {args.video}")
    
    cap = cv2.VideoCapture(args.video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    logger.info(f"Video info: {frame_count} frames, {fps} FPS")
    
    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame based on task
        if args.task == 'detect':
            detections = pipeline.detect_objects_frame(frame, 
                                                     model_name=args.model,
                                                     confidence_threshold=args.confidence)
            
            # Draw detections on frame
            frame = draw_detections(frame, detections)
        
        # Write frame if output specified
        if writer:
            writer.write(frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            logger.info(f"Processed {frame_idx}/{frame_count} frames")
    
    cap.release()
    if writer:
        writer.release()
        logger.info(f"Output video saved to: {args.output}")

def process_folder(pipeline, args):
    """Process folder of images"""
    if not os.path.exists(args.folder):
        logger.error(f"Folder not found: {args.folder}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(args.folder).glob(f'*{ext}'))
        image_files.extend(Path(args.folder).glob(f'*{ext.upper()}'))
    
    if not image_files:
        logger.error(f"No image files found in: {args.folder}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process in batches
    results = []
    for i in range(0, len(image_files), args.batch_size):
        batch = image_files[i:i + args.batch_size]
        batch_paths = [str(img) for img in batch]
        
        if args.task == 'classify':
            batch_results = pipeline.classify_batch(batch_paths, model_name=args.model)
        elif args.task == 'detect':
            batch_results = pipeline.detect_batch(batch_paths, 
                                                model_name=args.model,
                                                confidence_threshold=args.confidence)
        
        results.extend(batch_results)
        logger.info(f"Processed {min(i + args.batch_size, len(image_files))}/{len(image_files)} images")
    
    # Save results
    if args.output:
        save_batch_results(image_files, results, args.output, args.task)
        logger.info(f"Results saved to: {args.output}")

def interactive_mode(pipeline, args):
    """Interactive mode for CV processing"""
    logger.info("Starting interactive mode...")
    print("Computer Vision Tool - Interactive Mode")
    print("Commands: classify, detect, segment, quit")
    print()
    
    while True:
        command = input("Enter command: ").strip().lower()
        
        if command in ['quit', 'exit', 'q']:
            break
        
        if command not in ['classify', 'detect', 'segment']:
            print("Invalid command. Use: classify, detect, segment, quit")
            continue
        
        image_path = input("Enter image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        try:
            if command == 'classify':
                result = pipeline.classify_image(image_path)
                print(f"Class: {result['class']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
            elif command == 'detect':
                detections = pipeline.detect_objects(image_path)
                print(f"Found {len(detections)} objects:")
                for detection in detections:
                    print(f"- {detection['class']} ({detection['confidence']:.3f})")
                    
            elif command == 'segment':
                result = pipeline.segment_image(image_path)
                print(f"Segmentation completed")
                print(f"Mask shape: {result['segmentation_mask'].shape}")
            
            print()
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")

def draw_detections(frame, detections):
    """Draw detection boxes on frame"""
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(frame, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, 
                   (int(bbox[0]), int(bbox[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def save_batch_results(image_files, results, output_path, task):
    """Save batch processing results"""
    import pandas as pd
    
    data = []
    for img_file, result in zip(image_files, results):
        if task == 'classify':
            data.append({
                'image': str(img_file),
                'class': result['class'],
                'confidence': result['confidence']
            })
        elif task == 'detect':
            for detection in result:
                data.append({
                    'image': str(img_file),
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'bbox': str(detection['bbox'])
                })
    
    df = pd.DataFrame(data)
    
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path + '.csv', index=False)

if __name__ == "__main__":
    main()

