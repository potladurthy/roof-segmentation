import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import json
from tqdm import tqdm
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')


def mask_to_polygon(mask, original_image_shape):
    """Convert binary mask to normalized polygon points"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([], dtype=object)
    
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Use original image dimensions for normalization
    img_height, img_width = original_image_shape[:2]
    mask_height, mask_width = mask.shape
    
    # Scale factors to convert mask coordinates to image coordinates
    scale_x = img_width / mask_width
    scale_y = img_height / mask_height
    
    points = []
    for point in approx_polygon:
        x, y = point[0]
        # Scale to image coordinates then normalize
        img_x = x * scale_x
        img_y = y * scale_y
        points.append({
            'category': None, 'classLabel': None, 'confidence': None,
            'visible': None, 'x': float(img_x / img_width), 'y': float(img_y / img_height), 'z': None
        })
    
    return np.array(points, dtype=object)


def create_object(class_label, confidence, bbox, polygon_points):
    """Create object structure from YOLO predictions"""
    x, y, w, h = bbox  # YOLO provides normalized coordinates
    
    keypoints = np.array([{
        'category': None,
        'points': polygon_points,
        'type': None
    }], dtype=object)
    
    return {
        'category': None, 'classLabel': class_label, 'confidence': float(confidence),
        'height': float(h), 'keyPoints': keypoints, 'texts': None,
        'user_review': 'approved', 'width': float(w), 'x': float(x), 'y': float(y)
    }
   
def create_result(image_path, objects):
    """Create final result structure"""
    return {
        'classes': np.array([], dtype=object),
        'embeddings': np.array([], dtype=object),
        'keyPoints': np.array([], dtype=object),
        'objects': np.array(objects, dtype=object),
        'asset_url': None,
        'asset_url_model_uuid': None,
        'texts': np.array([], dtype=object),
        'type': 'prediction',
        'user_review': 'approved'
    }

def polygon_iou(pred_points, gt_points):
    """Calculate IoU between two polygons"""
    try:
        pred_coords = [(p['x'], p['y']) for p in pred_points if len(pred_points) > 0]
        gt_coords = [(p['x'], p['y']) for p in gt_points if len(gt_points) > 0]
        
        if len(pred_coords) < 3 or len(gt_coords) < 3:
            return 0.0
        
        pred_poly = Polygon(pred_coords)
        gt_poly = Polygon(gt_coords)
        
        if not pred_poly.is_valid or not gt_poly.is_valid:
            return 0.0
        
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

def calculate_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Calculate precision, recall, F1, and mIoU"""
    total_gt, total_pred, matched, total_iou = 0, 0, 0, 0.0

    for gt, pred in zip(ground_truth, predictions):

        pred_objs = pred.get('objects', [])
        gt_objs = gt.get('objects', [])

        total_pred += len(pred_objs)
        total_gt += len(gt_objs)

        gt_matched = set()
        pred_matched = set()

        # Compute IoU for every pair and match greedily
        iou_matrix = np.zeros((len(pred_objs), len(gt_objs)))
        for i, pred_obj in enumerate(pred_objs):
            pred_keypoints = pred_obj.get('keyPoints', [])
            if not pred_keypoints:
                continue
            pred_points = pred_keypoints[0].get('points', [])
            for j, gt_obj in enumerate(gt_objs):
                gt_keypoints = gt_obj.get('keyPoints', [])
                if not gt_keypoints:
                    continue
                gt_points = gt_keypoints[0].get('points', [])
                iou_matrix[i, j] = polygon_iou(pred_points, gt_points)

        # Greedy matching
        while True:
            max_iou = 0
            max_idx = (-1, -1)
            for i in range(len(pred_objs)):
                if i in pred_matched:
                    continue
                for j in range(len(gt_objs)):
                    if j in gt_matched:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_idx = (i, j)
            if max_iou >= 0.7:
                matched += 1
                total_iou += max_iou
                pred_matched.add(max_idx[0])
                gt_matched.add(max_idx[1])
            else:
                break

    precision = matched / total_pred if total_pred > 0 else 0.0
    recall = matched / total_gt if total_gt > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / matched if matched > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_iou': mean_iou,
        'total_predicted_roofs': total_pred,
        'total_ground_truth_roofs': total_gt,
        'matched_roofs': matched
    }

def visualize_predictions(image_path, results, output_dir, subset_name):
    """Visualize predictions on image and save"""
    try:
        # Read original image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        overlay = image.copy()
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            # Generate random colors for each instance
            colors = np.random.randint(0, 255, size=(len(masks), 3))
            
            for i, (mask, color) in enumerate(zip(masks, colors)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # Apply color to mask areas
                colored_mask = np.zeros_like(image)
                colored_mask[mask_resized > 0.5] = color
                
                # Blend with overlay
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                
                # Add bounding box and label
                if boxes is not None:
                    box = boxes.xyxy[i].cpu().numpy()
                    class_name = results[0].names[int(boxes.cls[i].cpu().numpy())]
                    confidence = boxes.conf[i].cpu().numpy()
                    
                    # Draw bounding box
                    cv2.rectangle(overlay, 
                                (int(box[0]), int(box[1])), 
                                (int(box[2]), int(box[3])), 
                                color.tolist(), 2)
                    
                    # Add label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(overlay, label, 
                            (int(box[0]), int(box[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            color.tolist(), 2)
        
        # Save visualization
        image_name = Path(image_path).stem
        output_path = Path(output_dir) / f"{subset_name}_visualizations"
        output_path.mkdir(exist_ok=True)
        
        output_file = output_path / f"{image_name}_pred.jpg"
        cv2.imwrite(str(output_file), overlay)
        
    except Exception as e:
        
        print(f"Error visualizing {image_path}: {e}")

def draw_polygon_from_keypoints(image, keypoints, options=None):
    """Draw polygon from keypoints similar to testing notebook"""
    if options is None:
        options = {}
    
    # Default options
    line_color = options.get('line_color', (0, 255, 0))
    line_thickness = options.get('line_thickness', 2)
    fill_polygon = options.get('fill_polygon', True)
    fill_color = options.get('fill_color', (0, 255, 0))
    fill_alpha = options.get('fill_alpha', 0.3)
    
    result_image = image.copy()
    
    if len(keypoints) < 3:
        return result_image
    
    # Convert to numpy array for OpenCV functions
    pts = np.array(keypoints, np.int32)
    
    # Fill polygon if requested
    if fill_polygon:
        overlay = result_image.copy()
        cv2.fillPoly(overlay, [pts], fill_color)
        result_image = cv2.addWeighted(result_image, 1 - fill_alpha, overlay, fill_alpha, 0)
    
    # Draw polygon outline
    cv2.polylines(result_image, [pts], True, line_color, line_thickness)
    
    return result_image

def extract_keypoints_from_objects(objects, image_width, image_height):
    """Extract keypoints from objects and convert to pixel coordinates"""
    all_keypoints = []
    for obj in objects:
        keypoints_array = obj.get('keyPoints', [])
        if not keypoints_array:
            continue
            
        # Extract points from each keypoint object
        pixel_points = []
        for point in keypoints_array[0].get('points', []):
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(point['x'] * image_width)
            pixel_y = int(point['y'] * image_height)
            pixel_points.append((pixel_x, pixel_y))
        all_keypoints.append(pixel_points)
    
    return all_keypoints

def visualize_predictions_with_polygons(image_path, prediction_result, gt_result, output_dir, subset_name):
    """Visualize predictions with polygon drawing for validation and test"""
    try:
        # Read original image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        image_height, image_width = image.shape[:2]
        
        if subset_name == "val":
            # For validation: side-by-side comparison
            # Create side-by-side image
            combined_image = np.zeros((image_height, image_width * 2, 3), dtype=np.uint8)
            
            # Left side: Ground Truth (Green)
            gt_image = image.copy()
            if gt_result and 'objects' in gt_result:
                gt_keypoints = extract_keypoints_from_objects(gt_result['objects'], image_width, image_height)
                gt_options = {
                    'line_color': (0, 255, 0),  # Green
                    'line_thickness': 3,
                    'fill_polygon': True,
                    'fill_color': (0, 255, 0),
                    'fill_alpha': 0.3
                }
                for points in gt_keypoints:
                    gt_image = draw_polygon_from_keypoints(gt_image, points, gt_options)
            
            # Right side: Predictions (Red)
            pred_image = image.copy()
            if prediction_result and 'objects' in prediction_result:
                pred_keypoints = extract_keypoints_from_objects(prediction_result['objects'], image_width, image_height)
                pred_options = {
                    'line_color': (0, 0, 255),  # Red
                    'line_thickness': 3,
                    'fill_polygon': True,
                    'fill_color': (0, 0, 255),
                    'fill_alpha': 0.3
                }
                for points in pred_keypoints:
                    pred_image = draw_polygon_from_keypoints(pred_image, points, pred_options)
            
            # Combine images
            combined_image[:, :image_width] = gt_image
            combined_image[:, image_width:] = pred_image
            
            # Add labels
            cv2.putText(combined_image, "Ground Truth", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_image, "Predictions", (image_width + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            final_image = combined_image
            
        else:
            # For test: only predictions
            final_image = image.copy()
            if prediction_result and 'objects' in prediction_result:
                pred_keypoints = extract_keypoints_from_objects(prediction_result['objects'], image_width, image_height)
                pred_options = {
                    'line_color': (0, 0, 255),  # Red
                    'line_thickness': 3,
                    'fill_polygon': True,
                    'fill_color': (0, 0, 255),
                    'fill_alpha': 0.3
                }
                for points in pred_keypoints:
                    final_image = draw_polygon_from_keypoints(final_image, points, pred_options)
        
        # Save polygon visualization
        image_name = Path(image_path).stem
        output_path = Path(output_dir) / f"{subset_name}_polygon_visualizations"
        output_path.mkdir(exist_ok=True)
        
        output_file = output_path / f"{image_name}_polygons.jpg"
        cv2.imwrite(str(output_file), final_image)
        
    except Exception as e:
        print(f"Error visualizing polygons for {image_path}: {e}")

class YOLOValidationProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def process_image(self, image_path):
        """Process single image and return formatted result"""
        predictions = self.model(str(image_path))
        objects = []
        
        if predictions and predictions[0].masks is not None:
            masks = predictions[0].masks.data.cpu().numpy()
            boxes = predictions[0].boxes
            
            # Get original image dimensions
            original_image = cv2.imread(str(image_path))
            original_shape = original_image.shape
            
            for i, mask in enumerate(masks):
                class_label = self.model.names[int(boxes.cls[i])]
                confidence = boxes.conf[i]
                
                # Use YOLO's normalized bounding box
                bbox = boxes.xywhn[i].cpu().numpy()  # normalized x_center, y_center, width, height
                
                polygon_points = mask_to_polygon(mask, original_shape)
                obj = create_object(class_label, confidence, bbox, polygon_points)
                objects.append(obj)

        return create_result(image_path, objects)

    def process_dataset_from_parquet(self, dataset_parquet_path, images_base_path, output_path, subset_name):
        """Process dataset subset from parquet file based on partition column"""
        print(f"Processing {subset_name} dataset from parquet...")
        
        # Load dataset parquet file
        df = pd.read_parquet(dataset_parquet_path)
        
        # Filter by partition
        subset_df = df[df['partition'] == subset_name]
        
        if len(subset_df) == 0:
            print(f"No data found for partition '{subset_name}'")
            return []
        
        print(f"Found {len(subset_df)} images in {subset_name} partition")
        
        results = []
        gt = []
        for _, row in tqdm(subset_df.iterrows(), desc=f"Processing {subset_name}", total=len(subset_df)):
            try:
                # Get image path from asset_url column
                image_asset_url = row.get('asset_url', '')
                if not image_asset_url:
                    continue
                image_path = image_asset_url
                if not Path(image_path).exists():
                    print(f"Image not found: {image_path}")
                    continue
                
                # Make predictions
                predictions = self.model(str(image_path))
                
                # Create formatted result
                result = self.process_image(image_path)
                results.append(result)
                
                # Get ground truth
                gt_result = None
                if len(row.get('annotations')) > 0:
                    gt_result = row['annotations'][0]
                    gt.append(gt_result)
                
                # Visualize with masks and bounding boxes (existing functionality)
                visualize_predictions(image_path, predictions, output_path, subset_name)
                
                # Visualize with polygons (new functionality)
                visualize_predictions_with_polygons(image_path, result, gt_result, output_path, subset_name)
                
            except Exception as e:
                
                print(f"Error processing row: {e}")
                continue
        
        if results:
            df_results = pd.DataFrame(results)
            output_file = Path(output_path) / f"{subset_name}_outputs.parquet"
            df_results.to_parquet(output_file, index=False)
            print(f"Saved {len(results)} outputs to {output_file}")
            print(f"Saved mask visualizations to {Path(output_path) / f'{subset_name}_visualizations'}")
            print(f"Saved polygon visualizations to {Path(output_path) / f'{subset_name}_polygon_visualizations'}")

        return results, gt

    def validate(self, predictions, ground_truth):
        """Calculate and print validation metrics"""
        metrics = calculate_metrics(predictions, ground_truth)

        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"Total Predicted Roofs: {metrics['total_predicted_roofs']}")
        print(f"Total Ground Truth Roofs: {metrics['total_ground_truth_roofs']}")
        print(f"Matched Roofs: {metrics['matched_roofs']}")
        print("="*50)
        
        return metrics

def main():
    # Configuration
    MODEL_PATH = "outputs/weights/best.pt"
    DATASET_PARQUET_PATH = "data/dataset.parquet"  # Main dataset parquet file
    IMAGES_BASE_PATH = "data/images"       # Base path where images are stored
    OUTPUT_DIR = "outputs"

    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    processor = YOLOValidationProcessor(MODEL_PATH)
    
    # Process datasets from parquet subsets
    val_predictions, val_ground_truth = processor.process_dataset_from_parquet(
        DATASET_PARQUET_PATH, IMAGES_BASE_PATH, OUTPUT_DIR, "val"
    )
    test_predictions, test_ground_truth = processor.process_dataset_from_parquet(
        DATASET_PARQUET_PATH, IMAGES_BASE_PATH, OUTPUT_DIR, "test"
    )

    # np.save(Path(OUTPUT_DIR) / "val_predictions.npy", val_predictions)
    # np.save(Path(OUTPUT_DIR) / "val_ground_truth.npy", val_ground_truth)

    # val_predictions = np.load(Path(OUTPUT_DIR) / "val_predictions.npy", allow_pickle=True)
    # val_ground_truth = np.load(Path(OUTPUT_DIR) / "val_ground_truth.npy", allow_pickle=True)

    assert len(val_predictions) == len(val_ground_truth), "Validation predictions and ground truth lengths do not match"
    assert isinstance(val_predictions[0], dict), "Validation predictions should be a list of dictionaries"
    assert isinstance(val_ground_truth[0], dict), "Validation ground truth should be a list of dictionaries"

    try:
        if val_ground_truth and val_predictions:
            metrics = processor.validate(val_predictions, val_ground_truth)
            
            # Save metrics
            with open(Path(OUTPUT_DIR) / "validation_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            print("No validation data available for metrics calculation")
            
    except Exception as e:
        print(f"Could not perform validation: {e}")

if __name__ == "__main__":
    main()