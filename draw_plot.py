import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

class KAISTEvaluator:
    def __init__(self, gt_dir, pred_file, img_list_file=None):
        """
        KAIST dataset evaluator for Miss Rate vs FPPI
        
        Args:
            gt_dir: Ground truth XML directory
            pred_file: Prediction JSON file or directory
            img_list_file: Image list file (optional)
        """
        self.gt_dir = Path(gt_dir)
        self.pred_file = pred_file
        self.img_list_file = img_list_file
        
        # Load ground truth and predictions
        self.gt_data = self.load_ground_truth()
        self.pred_data = self.load_predictions()
        
    def load_ground_truth(self):
        """Load ground truth annotations"""
        gt_data = defaultdict(list)
        
        for xml_file in self.gt_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                img_name = xml_file.stem
                
                # Extract time condition from filename (V000, V001 = day, V002, V003 = night)
                if 'V000' in img_name or 'V001' in img_name:
                    condition = 'day'
                elif 'V002' in img_name or 'V003' in img_name:
                    condition = 'night'
                else:
                    condition = 'day'  # default
                
                boxes = []
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name in ['person', 'cyclist', 'people', 'person?']:
                        bbox = obj.find('bndbox')
                        x = float(bbox.find('x').text)
                        y = float(bbox.find('y').text)
                        w = float(bbox.find('w').text)
                        h = float(bbox.find('h').text)
                        
                        # Filter reasonable boxes (KAIST reasonable subset)
                        if h >= 50 and (y + h) <= 511:  # Height >= 50px, not cut off
                            boxes.append({
                                'bbox': [x, y, w, h],
                                'class': class_name,
                                'condition': condition
                            })
                
                gt_data[img_name] = {
                    'boxes': boxes,
                    'condition': condition
                }
                
            except Exception as e:
                print(f"Error loading {xml_file}: {e}")
                
        return dict(gt_data)
    
    def load_predictions(self):
        """Load prediction results"""
        if isinstance(self.pred_file, str) and self.pred_file.endswith('.json'):
            with open(self.pred_file, 'r') as f:
                pred_data = json.load(f)
        else:
            # If it's a directory, load all JSON files
            pred_data = []
            pred_dir = Path(self.pred_file)
            for json_file in pred_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    pred_data.extend(data)
        
        # Convert to dict format
        pred_dict = defaultdict(list)
        
        if isinstance(pred_data, list):
            # COCO format
            for pred in pred_data:
                img_name = pred.get('image_id', pred.get('filename', ''))
                if isinstance(img_name, int):
                    img_name = f"I{img_name:05d}"  # Convert to KAIST format
                
                pred_dict[img_name].append({
                    'bbox': pred['bbox'],  # [x, y, w, h]
                    'score': pred['score'],
                    'category_id': pred.get('category_id', 1)
                })
        else:
            # Custom format
            for img_name, preds in pred_data.items():
                pred_dict[img_name] = preds
                
        return dict(pred_dict)
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x, y, w, h]"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_condition(self, condition='all', iou_threshold=0.5):
        """
        Evaluate for specific condition (all, day, night)
        Returns arrays for miss rate and FPPI calculation
        """
        all_predictions = []
        total_gt_boxes = 0
        total_images = 0
        
        for img_name, gt_info in self.gt_data.items():
            img_condition = gt_info['condition']
            
            # Filter by condition
            if condition != 'all' and img_condition != condition:
                continue
                
            total_images += 1
            gt_boxes = gt_info['boxes']
            total_gt_boxes += len(gt_boxes)
            
            # Get predictions for this image
            pred_boxes = self.pred_data.get(img_name, [])
            
            # Match predictions to ground truth
            gt_matched = [False] * len(gt_boxes)
            
            for pred in pred_boxes:
                pred_bbox = pred['bbox']
                pred_score = pred['score']
                
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                        
                    iou = self.calculate_iou(pred_bbox, gt_box['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Determine if it's TP or FP
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    is_tp = True
                else:
                    is_tp = False
                
                all_predictions.append({
                    'score': pred_score,
                    'is_tp': is_tp,
                    'image': img_name
                })
        
        return all_predictions, total_gt_boxes, total_images
    
    def compute_miss_rate_fppi(self, condition='all'):
        """Compute miss rate and FPPI curves"""
        predictions, total_gt, total_images = self.evaluate_condition(condition)
        
        if len(predictions) == 0:
            return np.array([1.0]), np.array([0.0])
        
        # Sort by confidence score (descending)
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate cumulative TP and FP
        tp_cumsum = 0
        fp_cumsum = 0
        miss_rates = []
        fppi_values = []
        
        # Add initial point (0 detections)
        miss_rates.append(1.0)
        fppi_values.append(0.0)
        
        for pred in predictions:
            if pred['is_tp']:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            # Calculate miss rate and FPPI
            recall = tp_cumsum / total_gt if total_gt > 0 else 0
            miss_rate = 1 - recall
            fppi = fp_cumsum / total_images if total_images > 0 else 0
            
            miss_rates.append(miss_rate)
            fppi_values.append(fppi)
        
        return np.array(miss_rates), np.array(fppi_values)
    
    def plot_miss_rate_fppi(self, save_path=None):
        """Plot Miss Rate vs FPPI for all, day, night conditions"""
        plt.figure(figsize=(12, 8))
        
        conditions = ['all', 'day', 'night']
        colors = ['blue', 'red', 'green']
        
        for condition, color in zip(conditions, colors):
            miss_rates, fppi_values = self.compute_miss_rate_fppi(condition)
            
            plt.loglog(fppi_values, miss_rates, 
                      color=color, linewidth=2, label=f'{condition.capitalize()}')
        
        # Formatting
        plt.xlim([1e-3, 1e2])
        plt.ylim([1e-2, 1])
        plt.xlabel('False Positive Per Image (FPPI)', fontsize=12)
        plt.ylabel('Miss Rate', fontsize=12)
        plt.title('Miss Rate vs FPPI - KAIST Dataset', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Add reference lines
        plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def print_performance_summary(self):
        """Print performance summary for each condition"""
        print("\n" + "="*50)
        print("KAIST Dataset Performance Summary")
        print("="*50)
        
        for condition in ['all', 'day', 'night']:
            miss_rates, fppi_values = self.compute_miss_rate_fppi(condition)
            
            # Find miss rate at FPPI = 0.1
            fppi_01_idx = np.where(fppi_values >= 0.1)[0]
            if len(fppi_01_idx) > 0:
                miss_rate_at_01 = miss_rates[fppi_01_idx[0]]
            else:
                miss_rate_at_01 = miss_rates[-1]
            
            # Find miss rate at FPPI = 1.0
            fppi_1_idx = np.where(fppi_values >= 1.0)[0]
            if len(fppi_1_idx) > 0:
                miss_rate_at_1 = miss_rates[fppi_1_idx[0]]
            else:
                miss_rate_at_1 = miss_rates[-1]
            
            print(f"\n{condition.upper()}:")
            print(f"  Miss Rate @ FPPI=0.1: {miss_rate_at_01:.3f}")
            print(f"  Miss Rate @ FPPI=1.0: {miss_rate_at_1:.3f}")
            print(f"  Total FPPI range: {fppi_values.min():.3f} - {fppi_values.max():.3f}")

def main():
    """Example usage"""
    # TODO: Update these paths according to your setup
    gt_dir = "datasets/kaist-rgbt/train/labels-xml"  # Ground truth XML directory
    pred_file = "results/predictions.json"  # Your prediction results
    
    # Check if paths exist
    if not os.path.exists(gt_dir):
        print(f"Ground truth directory not found: {gt_dir}")
        print("Please update the gt_dir path to your KAIST test labels")
        return
    
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        print("Please update the pred_file path to your model predictions")
        print("\nExpected prediction format (JSON):")
        print("""[
    {
        "image_id": "I00001",  # or just filename
        "bbox": [x, y, width, height],
        "score": 0.95,
        "category_id": 1
    },
    ...
]""")
        return
    
    # Create evaluator
    evaluator = KAISTEvaluator(gt_dir, pred_file)
    
    # Plot Miss Rate vs FPPI
    evaluator.plot_miss_rate_fppi(save_path="miss_rate_fppi_plot.png")
    
    # Print performance summary
    evaluator.print_performance_summary()

if __name__ == "__main__":
    main()