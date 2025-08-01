import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import math

class PolygonExtractor:
    def __init__(self, min_area=100, epsilon_factor=0.02):
        """
        Initialize polygon extractor
        
        Args:
            min_area: Minimum area threshold for polygons
            epsilon_factor: Factor for polygon approximation (smaller = more precise)
        """
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor
    
    def extract_contours(self, mask):
        """Extract contours from binary mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def approximate_polygon(self, contour):
        """Approximate contour to polygon using Douglas-Peucker algorithm"""
        epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx
    
    def regularize_polygon(self, polygon_points, target_sides=None):
        """
        Regularize polygon to have more regular shape
        
        Args:
            polygon_points: Array of polygon vertices
            target_sides: Target number of sides (auto-detect if None)
        """
        if len(polygon_points) < 3:
            return polygon_points
        
        # Convert to shapely polygon for easier manipulation
        try:
            poly = Polygon(polygon_points.reshape(-1, 2))
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix self-intersections
            
            # Get centroid
            centroid = poly.centroid
            cx, cy = centroid.x, centroid.y
            
            # Convert to polar coordinates relative to centroid
            points = polygon_points.reshape(-1, 2)
            angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
            distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
            
            # Sort by angle
            sorted_indices = np.argsort(angles)
            sorted_angles = angles[sorted_indices]
            sorted_distances = distances[sorted_indices]
            
            # Auto-detect number of sides if not specified
            if target_sides is None:
                target_sides = self.detect_polygon_sides(sorted_angles, sorted_distances)
            
            # Create regular polygon
            regular_angles = np.linspace(0, 2*np.pi, target_sides, endpoint=False)
            avg_distance = np.mean(sorted_distances)
            
            # Generate regular polygon points
            regular_points = []
            for angle in regular_angles:
                x = cx + avg_distance * np.cos(angle)
                y = cy + avg_distance * np.sin(angle)
                regular_points.append([x, y])
            
            return np.array(regular_points, dtype=np.int32)
        
        except Exception as e:
            print(f"Error in regularization: {e}")
            return polygon_points
    
    def detect_polygon_sides(self, angles, distances, max_sides=8):
        """Detect optimal number of sides for polygon"""
        # Try different numbers of sides and find best fit
        best_sides = 4  # Default to rectangle
        min_error = float('inf')
        
        for sides in range(3, max_sides + 1):
            # Create regular polygon with this many sides
            regular_angles = np.linspace(0, 2*np.pi, sides, endpoint=False)
            
            # Find closest regular angle for each point
            errors = []
            for angle in angles:
                closest_regular = regular_angles[np.argmin(np.abs(regular_angles - angle))]
                errors.append(abs(angle - closest_regular))
            
            avg_error = np.mean(errors)
            if avg_error < min_error:
                min_error = avg_error
                best_sides = sides
        
        return best_sides
    
    def smooth_polygon(self, polygon_points, sigma=1.0):
        """Apply Gaussian smoothing to polygon vertices"""
        if len(polygon_points) < 3:
            return polygon_points
        
        points = polygon_points.reshape(-1, 2).astype(np.float32)
        
        # Apply Gaussian filter to coordinates
        from scipy.ndimage import gaussian_filter1d
        
        # Make polygon cyclic for smoothing
        cyclic_points = np.vstack([points, points[:2]])  # Add first two points at end
        
        smoothed_x = gaussian_filter1d(cyclic_points[:, 0], sigma=sigma, mode='wrap')
        smoothed_y = gaussian_filter1d(cyclic_points[:, 1], sigma=sigma, mode='wrap')
        
        # Remove the extra points
        smoothed_points = np.column_stack([smoothed_x[:-2], smoothed_y[:-2]])
        
        return smoothed_points.astype(np.int32)
    
    def process_mask(self, mask, regularize=True, smooth=True, target_sides=None):
        """
        Process segmentation mask to extract regular polygons
        
        Args:
            mask: Binary segmentation mask
            regularize: Whether to regularize polygons
            smooth: Whether to smooth polygon vertices
            target_sides: Target number of sides for regularization
        
        Returns:
            List of polygon vertices
        """
        # Extract contours
        contours = self.extract_contours(mask)
        
        polygons = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Approximate to polygon
            polygon = self.approximate_polygon(contour)
            
            # Regularize if requested
            if regularize:
                polygon = self.regularize_polygon(polygon, target_sides)
            
            # Smooth if requested
            if smooth:
                polygon = self.smooth_polygon(polygon)
            
            polygons.append(polygon)
        
        return polygons
    
    def visualize_results(self, image, original_mask, polygons, save_path=None):
        """Visualize extraction results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Original mask
        axes[1].imshow(original_mask, cmap='gray')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Extracted polygons
        result_image = image.copy()
        for i, polygon in enumerate(polygons):
            if len(polygon) >= 3:
                # Draw filled polygon
                cv2.fillPoly(result_image, [polygon], (0, 255, 0))
                # Draw polygon outline
                cv2.polylines(result_image, [polygon], True, (0, 0, 255), 2)
                
                # Add polygon number
                centroid = np.mean(polygon.reshape(-1, 2), axis=0).astype(int)
                cv2.putText(result_image, str(i), tuple(centroid), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        axes[2].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Extracted Polygons')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return result_image

def demo_polygon_extraction():
    """Demonstration of polygon extraction"""
    from ultralytics import YOLO
    
    # Load trained model
    model = YOLO('runs/segment/train/weights/best.pt')
    
    # Load test image
    test_image_path = '/path/to/test/image.jpg'  # Update with actual test image path
    image = cv2.imread(test_image_path)
    
    # Run inference
    results = model(image)
    
    # Extract segmentation mask
    if results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()  # Get first mask
        mask = (mask * 255).astype(np.uint8)
        
        # Resize mask to match image size
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Extract polygons
        extractor = PolygonExtractor(min_area=500, epsilon_factor=0.01)
        polygons = extractor.process_mask(
            mask, 
            regularize=True, 
            smooth=True, 
            target_sides=4  # Assuming roofs are mostly rectangular
        )
        
        # Visualize results
        result_image = extractor.visualize_results(image, mask, polygons)
        
        print(f"Extracted {len(polygons)} roof polygons")
        for i, polygon in enumerate(polygons):
            print(f"Polygon {i}: {len(polygon)} vertices")
    
    else:
        print("No segmentation masks found in the results")

if __name__ == "__main__":
    demo_polygon_extraction()
