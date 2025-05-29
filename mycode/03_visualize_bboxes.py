import os
import cv2

# 클래스 이름 정의
class_names = {
    0: "person",
    1: "cyclist",
    2: "people",
    3: "person?"
}
# 원본 이미지 크기 (라벨링 시 사용된 크기)
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512

# 변환된 이미지 크기 (현재 사용하는 크기)
NEW_WIDTH = 640
NEW_HEIGHT = 640

# 스케일링 계수 계산
scale_x = NEW_WIDTH / ORIGINAL_WIDTH  # 1.0 (가로는 변환 없음)
scale_y = NEW_HEIGHT / ORIGINAL_HEIGHT  # 1.25 (세로 512→640으로 확대)

image_dir = 'datasets/kaist-rgbt/train/images/visible'
label_dir = 'datasets/kaist-rgbt/train/labels'
save_dir = 'visualize_640'
os.makedirs(save_dir, exist_ok=True)

for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')

    # 이미지 읽기 & 640x640으로 리사이즈
    image = cv2.imread(img_path)
    if image is None:
        print(f"이미지 읽기 실패: {img_path}")
        continue
    image = cv2.resize(image, (NEW_WIDTH, NEW_HEIGHT))  # 중요! 이미지 크기 변경
    
    if not os.path.exists(label_path):
        print(f"라벨 없음: {label_path}")
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(parts[0])
            # 원본 크기 기준 좌표 계산
            x_center_orig = float(parts[1]) * ORIGINAL_WIDTH
            y_center_orig = float(parts[2]) * ORIGINAL_HEIGHT
            bbox_width_orig = float(parts[3]) * ORIGINAL_WIDTH
            bbox_height_orig = float(parts[4]) * ORIGINAL_HEIGHT

            # 새로운 크기로 스케일링
            x_center = x_center_orig * scale_x
            y_center = y_center_orig * scale_y
            bbox_width = bbox_width_orig * scale_x
            bbox_height = bbox_height_orig * scale_y

            # 픽셀 좌표 계산 + 범위 제한
            x1 = max(0, int(x_center - bbox_width / 2))
            y1 = max(0, int(y_center - bbox_height / 2))
            x2 = min(NEW_WIDTH, int(x_center + bbox_width / 2))
            y2 = min(NEW_HEIGHT, int(y_center + bbox_height / 2))
            
            if x1 >= x2 or y1 >= y2:
                continue

            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 클래스 라벨 표시
            label = class_names.get(class_id, str(class_id))
            cv2.putText(image, label, (x1, max(y1-5, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            
        except Exception as e:
            print(f"에러 발생: {line.strip()} → {str(e)}")
            continue

    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, image)

print("✅ 라벨에 맞게 바운딩 박스 시각화 완료.")