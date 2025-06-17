import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def draw_bbox(image, bbox, score, category_id, color=(0, 255, 0), thickness=2):
    """바운딩 박스와 레이블을 이미지에 그리기"""
    x, y, w, h = map(int, bbox)
    
    # 바운딩 박스 그리기
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # 레이블 텍스트 생성 (category_id에 따라 다르게)
    if category_id == 0:
        label = f"person: {score:.3f}"
    elif category_id == 1:
        label = f"cyclist: {score:.3f}"
    elif category_id == 2:
        label = f"people: {score:.3f}"
    else:
        label = f"person?: {score:.3f}"
    
    # 텍스트 설정 개선
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # 폰트 크기 증가
    font_thickness = 2  # 폰트 두께 증가
    
    # 텍스트 배경 박스 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )
    
    # 텍스트 배경에 패딩 추가
    padding = 5
    
    # 텍스트 배경 그리기
    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding * 2),
        (x + text_width + padding, y),
        color,
        -1
    )
    
    # 텍스트 그리기 (안티앨리어싱 적용)
    cv2.putText(
        image,
        label,
        (x, y - padding),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA  # 안티앨리어싱 적용
    )
    
    return image
    
    return image

def visualize_predictions(json_path, images_dir, output_dir, conf_threshold=0.3):
    """Prediction 결과를 이미지에 시각화하여 저장"""
    
    # JSON 파일 읽기
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    # 출력 디렉토리 생성
    output_lwir_dir = os.path.join(output_dir, 'lwir')
    output_visible_dir = os.path.join(output_dir, 'visible')
    output_extra_lwir_dir = os.path.join(output_dir, 'extra', 'lwir')
    output_extra_visible_dir = os.path.join(output_dir, 'extra', 'visible')
    os.makedirs(output_lwir_dir, exist_ok=True)
    os.makedirs(output_visible_dir, exist_ok=True)
    os.makedirs(output_extra_lwir_dir, exist_ok=True)
    os.makedirs(output_extra_visible_dir, exist_ok=True)
    
    # 이미지별로 prediction 그룹화
    image_predictions = {}
    for pred in predictions:
        if pred['score'] >= conf_threshold:  # confidence threshold 적용
            image_name = pred['image_name']
            if image_name not in image_predictions:
                image_predictions[image_name] = []
            image_predictions[image_name].append(pred)
    
    # 각 이미지에 대해 처리
    non_zero_class_count = 0
    for image_name, preds in tqdm(image_predictions.items(), desc="Processing images"):
        # category_id가 0이 아닌 detection이 있는지 확인
        has_non_zero_class = any(pred['category_id'] != 0 for pred in preds)
        
        # LWIR 이미지 처리
        lwir_path = os.path.join(images_dir, 'lwir', f"{image_name}.jpg")
        if os.path.exists(lwir_path):
            lwir_img = cv2.imread(lwir_path)
            
            # 모든 prediction 그리기
            for pred in preds:
                # category_id에 따라 색상 다르게 설정
                if pred['category_id'] == 0:
                    color = (0, 255, 0)  # 녹색 (person)
                else:
                    color = (255, 0, 255)  # 보라색 (다른 클래스)
                
                lwir_img = draw_bbox(
                    lwir_img,
                    pred['bbox'],
                    pred['score'],
                    pred['category_id'],
                    color=color
                )
            
            # 저장
            output_path = os.path.join(output_lwir_dir, f"{image_name}_result.jpg")
            cv2.imwrite(output_path, lwir_img)
            
            # category_id가 0이 아닌 경우 extra 폴더에도 저장
            if has_non_zero_class:
                extra_path = os.path.join(output_extra_lwir_dir, f"{image_name}_result.jpg")
                cv2.imwrite(extra_path, lwir_img)
        
        # Visible 이미지 처리
        visible_path = os.path.join(images_dir, 'visible', f"{image_name}.jpg")
        if os.path.exists(visible_path):
            visible_img = cv2.imread(visible_path)
            
            # 모든 prediction 그리기
            for pred in preds:
                # category_id에 따라 색상 다르게 설정
                if pred['category_id'] == 0:
                    color = (0, 0, 255)  # 빨간색 (person)
                else:
                    color = (255, 255, 0)  # 노란색 (다른 클래스)
                
                visible_img = draw_bbox(
                    visible_img,
                    pred['bbox'],
                    pred['score'],
                    pred['category_id'],
                    color=color
                )
            
            # 저장
            output_path = os.path.join(output_visible_dir, f"{image_name}_result.jpg")
            cv2.imwrite(output_path, visible_img)
            
            # category_id가 0이 아닌 경우 extra 폴더에도 저장
            if has_non_zero_class:
                extra_path = os.path.join(output_extra_visible_dir, f"{image_name}_result.jpg")
                cv2.imwrite(extra_path, visible_img)
                non_zero_class_count += 1
    
    print(f"결과가 {output_dir}에 저장되었습니다.")
    print(f"처리된 이미지 수: {len(image_predictions)}")
    print(f"클래스 0이 아닌 검출이 있는 이미지 수: {non_zero_class_count}")

def create_side_by_side_visualization(json_path, images_dir, output_dir, conf_threshold=0.3):
    """LWIR과 Visible 이미지를 나란히 배치하여 시각화"""
    
    # JSON 파일 읽기
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    # 출력 디렉토리 생성
    output_combined_dir = os.path.join(output_dir, 'combined')
    output_extra_combined_dir = os.path.join(output_dir, 'extra', 'combined')
    os.makedirs(output_combined_dir, exist_ok=True)
    os.makedirs(output_extra_combined_dir, exist_ok=True)
    
    # 이미지별로 prediction 그룹화
    image_predictions = {}
    for pred in predictions:
        if pred['score'] >= conf_threshold:
            image_name = pred['image_name']
            if image_name not in image_predictions:
                image_predictions[image_name] = []
            image_predictions[image_name].append(pred)
    
    # 각 이미지에 대해 처리
    for image_name, preds in tqdm(image_predictions.items(), desc="Creating combined images"):
        lwir_path = os.path.join(images_dir, 'lwir', f"{image_name}.jpg")
        visible_path = os.path.join(images_dir, 'visible', f"{image_name}.jpg")
        
        # category_id가 0이 아닌 detection이 있는지 확인
        has_non_zero_class = any(pred['category_id'] != 0 for pred in preds)
        
        if os.path.exists(lwir_path) and os.path.exists(visible_path):
            # 이미지 읽기
            lwir_img = cv2.imread(lwir_path)
            visible_img = cv2.imread(visible_path)
            
            # prediction 그리기
            for pred in preds:
                # LWIR 이미지 색상
                if pred['category_id'] == 0:
                    lwir_color = (0, 255, 0)  # 녹색
                else:
                    lwir_color = (255, 0, 255)  # 보라색
                
                # Visible 이미지 색상
                if pred['category_id'] == 0:
                    visible_color = (0, 0, 255)  # 빨간색
                else:
                    visible_color = (255, 255, 0)  # 노란색
                
                lwir_img = draw_bbox(
                    lwir_img,
                    pred['bbox'],
                    pred['score'],
                    pred['category_id'],
                    color=lwir_color
                )
                visible_img = draw_bbox(
                    visible_img,
                    pred['bbox'],
                    pred['score'],
                    pred['category_id'],
                    color=visible_color
                )
            
            # 이미지 합치기
            combined = np.hstack([lwir_img, visible_img])
            
            # 저장
            output_path = os.path.join(output_combined_dir, f"{image_name}_combined.jpg")
            cv2.imwrite(output_path, combined)
            
            # category_id가 0이 아닌 경우 extra 폴더에도 저장
            if has_non_zero_class:
                extra_path = os.path.join(output_extra_combined_dir, f"{image_name}_combined.jpg")
                cv2.imwrite(extra_path, combined)

if __name__ == "__main__":
    # 경로 설정
    json_path = "runs/val/exp41/best_predictions.json"
    images_dir = "datasets/kaist-rgbt/test/images"
    output_dir = "result"
    
    # 개별 이미지 시각화
    visualize_predictions(json_path, images_dir, output_dir, conf_threshold=0.3)
    
    # LWIR-Visible 결합 이미지 생성 (선택사항)
    create_side_by_side_visualization(json_path, images_dir, output_dir, conf_threshold=0.3)