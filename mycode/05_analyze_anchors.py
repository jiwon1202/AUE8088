import os
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  # 추가!
from collections import Counter

def analyze_target_boxes_only(label_dir, classes=('person', 'cyclist', 'people', 'person?'), image_size=(640, 640)):
    """지정된 클래스 박스만 분석"""
    target_boxes = []

    for xml_file in Path(label_dir).glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text

                if class_name in classes:
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
                        target_boxes.append([scaled_w, scaled_h])

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    return np.array(target_boxes)

def generate_optimal_anchors(boxes, n_anchors=12, visualize=True):
    """클래스에 최적화된 anchor 생성 (P3~P6용 12개)"""

    print(f"\n📊 Target boxes statistics:")
    print(f"Total boxes: {len(boxes)}")
    print(f"Width - min: {boxes[:, 0].min():.1f}, max: {boxes[:, 0].max():.1f}, median: {np.median(boxes[:, 0]):.1f}")
    print(f"Height - min: {boxes[:, 1].min():.1f}, max: {boxes[:, 1].max():.1f}, median: {np.median(boxes[:, 1]):.1f}")

    ratios = boxes[:, 0] / boxes[:, 1]
    print(f"Aspect ratio - min: {ratios.min():.2f}, max: {ratios.max():.2f}, median: {np.median(ratios):.2f}")

    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(boxes)
    centers = kmeans.cluster_centers_

    # 정렬
    areas = centers[:, 0] * centers[:, 1]
    sorted_indices = np.argsort(areas)
    sorted_centers = centers[sorted_indices]

    # anchor 보정
    adjusted_anchors = []
    for w, h in sorted_centers:
        if w / h > 0.7:
            h *= 1.2
            w *= 0.95
        adjusted_anchors.append([int(w), int(h)])

    # 시각화
    if visualize:
        visualize_box_statistics(boxes, clusters, centers, sorted_indices)

    # P3~P6 분리
    anchors_p3 = adjusted_anchors[:3]
    anchors_p4 = adjusted_anchors[3:6] 
    anchors_p5 = adjusted_anchors[6:9]
    anchors_p6 = adjusted_anchors[9:12]

    return {
        'P3': anchors_p3,
        'P4': anchors_p4,
        'P5': anchors_p5,
        'P6': anchors_p6,
        'all_centers': adjusted_anchors
    }

def visualize_box_statistics(boxes, clusters, centers, sorted_indices, save_path="anchor_analysis.png"):
    """box 통계 시각화 + 저장 + C1~C12 정확히 고정 표시"""
    import matplotlib.colors as mcolors

    ratios = boxes[:, 0] / boxes[:, 1]
    areas = boxes[:, 0] * boxes[:, 1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # P3/P4/P5/P6 mapping
    layer_labels = ['P3', 'P3', 'P3', 'P4', 'P4', 'P4', 'P5', 'P5', 'P5', 'P6', 'P6', 'P6']
    cluster_to_layer = {sorted_indices[i]: layer_labels[i] for i in range(12)}

    # cluster_id → C1~C12 label 고정 mapping
    cluster_id_to_c_label = {sorted_indices[i]: f"C{i+1}" for i in range(12)}

    # base 색상 설정
    layer_base_colors = {
        'P3': mcolors.to_rgb('blue'),
        'P4': mcolors.to_rgb('green'),
        'P5': mcolors.to_rgb('orange'),
        'P6': mcolors.to_rgb('red')
    }

    # 레이어별 해당 cluster index
    layer_cluster_indices = {
        'P3': sorted_indices[0:3],
        'P4': sorted_indices[3:6],
        'P5': sorted_indices[6:9],
        'P6': sorted_indices[9:12]
    }

    # ⭐️ cluster별 색상 저장 리스트 → bar chart에도 동일 적용
    cluster_color_dict = {}

    # 1️⃣ W vs H scatter → C1~C12 순서 고정으로 출력
    for i, cluster_id in enumerate(sorted_indices):
        c_label = cluster_id_to_c_label[cluster_id]

        # base color 결정
        layer = cluster_to_layer[cluster_id]
        base_color = np.array(layer_base_colors[layer])
        layer_cluster_ids = layer_cluster_indices[layer]
        cluster_idx_in_layer = list(layer_cluster_ids).index(cluster_id)
        n_clusters_in_layer = len(layer_cluster_ids)

        # blending
        blend_factor = 0.4 + 0.3 * (cluster_idx_in_layer / (n_clusters_in_layer - 1))
        adjusted_color = base_color * (1 - blend_factor) + np.array([1.0, 1.0, 1.0]) * blend_factor
        adjusted_color = np.clip(adjusted_color, 0, 1)

        # 저장
        cluster_color_dict[cluster_id] = adjusted_color

        # 박스 scatter
        mask = clusters == cluster_id
        axs[0, 0].scatter(boxes[mask, 0], boxes[mask, 1],
                          color=adjusted_color, alpha=0.9, label=c_label)

        # Anchor center 텍스트 C1~C12
        center_x = centers[cluster_id, 0]
        center_y = centers[cluster_id, 1]
        axs[0, 0].text(center_x, center_y, c_label, fontsize=10, fontweight='bold',
                       ha='center', va='center', color='black',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    axs[0, 0].set_title("Box Width vs Height (C1~C12)")
    axs[0, 0].set_xlabel("Width")
    axs[0, 0].set_ylabel("Height")
    axs[0, 0].legend(fontsize='small', loc='best', ncol=2)

    # 2️⃣ Aspect Ratio histogram
    axs[0, 1].hist(ratios, bins=50, color='skyblue', edgecolor='black')
    axs[0, 1].set_title("Aspect Ratio Distribution (W/H)")
    axs[0, 1].set_xlabel("Aspect Ratio")
    axs[0, 1].set_ylabel("Count")

    # 3️⃣ Area histogram
    axs[1, 0].hist(areas, bins=50, color='lightgreen', edgecolor='black')
    axs[1, 0].set_title("Box Area Distribution (W x H)")
    axs[1, 0].set_xlabel("Area")
    axs[1, 0].set_ylabel("Count")

    # 4️⃣ Number of boxes per cluster → C1~C12 순서 고정
    cluster_counts = Counter(clusters)

    # C1~C12 순서 고정으로 bar 출력
    bar_colors = [cluster_color_dict[cid] for cid in sorted_indices]
    bar_heights = [cluster_counts.get(cid, 0) for cid in sorted_indices]

    axs[1, 1].bar(range(12), bar_heights, color=bar_colors, edgecolor='black')
    axs[1, 1].set_title("Number of Boxes per Cluster")
    axs[1, 1].set_xlabel("Cluster ID (C1~C12)")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].set_xticks(range(12))
    axs[1, 1].set_xticklabels([cluster_id_to_c_label[cid] for cid in sorted_indices])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)




def main():
    label_dir = "datasets/kaist-rgbt/train/labels-xml"

    if not os.path.exists(label_dir):
        print(f"Error: {label_dir} not found!")
        return

    print("🎯 Analyzing Person + Cyclist boxes for optimal anchor design (P3~P6)...")
    target_boxes = analyze_target_boxes_only(label_dir)

    if len(target_boxes) == 0:
        print("No target boxes found!")
        return

    print("\n🎯 Generating optimized anchors for person + cyclist (12 anchors)...")
    optimal_anchors = generate_optimal_anchors(target_boxes)

    print("\n📋 Optimized Anchors:")
    print(f"P3/8  (small):   {optimal_anchors['P3']}")
    print(f"P4/16 (medium):  {optimal_anchors['P4']}")
    print(f"P5/32 (large):   {optimal_anchors['P5']}")
    print(f"P6/64 (x-large): {optimal_anchors['P6']}\n\n")

    return optimal_anchors

if __name__ == "__main__":
    optimal_anchors = main()

"""
P3/8  (small):   [[22, 50], [23, 64], [28, 77]]
P4/16 (medium):  [[33, 92], [39, 108], [42, 124]]
P5/32 (large):   [[46, 142], [53, 163], [63, 190]]
P6/64 (x-large): [[71, 224], [79, 261], [92, 315]]
"""

"""
P3/8  (small):   [[24, 53], [26, 71], [35, 94]]
P4/16 (medium):  [[41, 119], [103, 76], [49, 146]]
P5/32 (large):   [[60, 185], [184, 83], [74, 239]]
P6/64 (x-large): [[264, 77], [422, 69], [90, 307]]
"""