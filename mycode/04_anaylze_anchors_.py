# analyze_all_anchors.py
"""
모든 클래스에 대한 anchor 분석 및 비교
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.cluster import KMeans

def analyze_all_boxes(label_dir, image_size=(640, 640)):
    """모든 클래스의 박스를 분석"""
    all_boxes = []

    for xml_file in Path(label_dir).glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                x = float(bbox.find('x').text)
                y = float(bbox.find('y').text)
                w = float(bbox.find('w').text)
                h = float(bbox.find('h').text)

                scale_x = image_size[0] / 640
                scale_y = image_size[1] / 512

                scaled_w = w * scale_x
                scaled_h = h * scale_y

                if scaled_w > 0 and scaled_h > 0:
                    all_boxes.append([scaled_w, scaled_h])

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    return np.array(all_boxes)

def generate_optimal_anchors(boxes, n_anchors=9):
    """최적화된 anchor 생성"""

    print(f"\n📊 All boxes statistics:")
    print(f"Total boxes: {len(boxes)}")
    print(f"Width - min: {boxes[:, 0].min():.1f}, max: {boxes[:, 0].max():.1f}, median: {np.median(boxes[:, 0]):.1f}")
    print(f"Height - min: {boxes[:, 1].min():.1f}, max: {boxes[:, 1].max():.1f}, median: {np.median(boxes[:, 1]):.1f}")

    ratios = boxes[:, 0] / boxes[:, 1]
    print(f"Aspect ratio - min: {ratios.min():.2f}, max: {ratios.max():.2f}, median: {np.median(ratios):.2f}")

    kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(boxes)
    centers = kmeans.cluster_centers_

    areas = centers[:, 0] * centers[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_centers = centers[sorted_indices]

    adjusted_anchors = []
    for w, h in sorted_centers:
        if w / h > 0.7:
            h *= 1.2
            w *= 0.95
        adjusted_anchors.append([int(w), int(h)])

    anchors_p3 = adjusted_anchors[:3]
    anchors_p4 = adjusted_anchors[3:6] 
    anchors_p5 = adjusted_anchors[6:9]

    return {
        'P3': anchors_p3,
        'P4': anchors_p4,
        'P5': anchors_p5,
        'all_centers': adjusted_anchors
    }

def main():
    label_dir = "datasets/kaist-rgbt/train/labels-xml"

    if not os.path.exists(label_dir):
        print(f"Error: {label_dir} not found!")
        return

    print("🎯 Analyzing all class boxes for optimal anchor design...")
    all_boxes = analyze_all_boxes(label_dir)

    if len(all_boxes) == 0:
        print("No boxes found!")
        return

    print("\n🎯 Generating optimized anchors for all classes...")
    optimal_anchors = generate_optimal_anchors(all_boxes)

    print("\n📋 Optimized Anchors:")
    print(f"P3/8  (small):  {optimal_anchors['P3']}")
    print(f"P4/16 (medium): {optimal_anchors['P4']}")
    print(f"P5/32 (large):  {optimal_anchors['P5']}")

    return optimal_anchors

if __name__ == "__main__":
    optimal_anchors = main()

"""
P3/8  (small):  [[24, 59], [32, 86], [40, 117]]
P4/16 (medium): [[106, 76], [50, 152], [67, 209]]
P5/32 (large):  [[197, 81], [366, 74], [86, 288]]
"""