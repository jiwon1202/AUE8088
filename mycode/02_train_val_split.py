import os
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter

label_dir = "/home/cv/jw/AUE8088/datasets/kaist-rgbt/train/labels"
image_list_file = "/home/cv/jw/AUE8088/datasets/kaist-rgbt/train-all-04.txt"
NUM_CLASSES = 4

# 전체 이미지 목록
with open(image_list_file, "r") as f:
    all_image_paths = [line.strip() for line in f]

# 각 이미지의 클래스 정보 저장
image_class_map = {}  # 이미지 경로 -> set(class_ids)
class_counts_total = Counter()
no_object_images = []

for img_path in all_image_paths:
    basename = os.path.basename(img_path)
    label_path = os.path.join(label_dir, basename.replace(".jpg", ".txt"))
    class_ids = set()

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    if 0 <= class_id < NUM_CLASSES:
                        class_ids.add(class_id)
                        class_counts_total[class_id] += 1

    if class_ids:
        image_class_map[img_path] = class_ids
    else:
        no_object_images.append(img_path)

# 객체 포함 이미지 리스트
object_images = list(image_class_map.keys())

# 이미지 단위로 train/val 분할 (객체 있는 이미지)
obj_train, obj_val = train_test_split(object_images, test_size=0.1, random_state=42)

# 객체 없는 이미지도 8:2 분할
noobj_train, noobj_val = train_test_split(no_object_images, test_size=0.1, random_state=42)

# 최종 train/val 리스트
train_list = obj_train + noobj_train
val_list = obj_val + noobj_val

# 저장
with open('/home/cv/jw/AUE8088/datasets/kaist-rgbt/train2.txt', 'w') as f:
    f.writelines('\n'.join(train_list) + '\n')
with open('/home/cv/jw/AUE8088/datasets/kaist-rgbt/val2.txt', 'w') as f:
    f.writelines('\n'.join(val_list) + '\n')

# 📊 통계 출력 (맨 아래 정리)
def count_classes(image_list, image_class_map):
    counter = Counter()
    for img in image_list:
        for cls in image_class_map.get(img, []):
            counter[cls] += 1
    return counter

train_class_counts = count_classes(train_list, image_class_map)
val_class_counts = count_classes(val_list, image_class_map)

print(f"전체 이미지 수: {len(all_image_paths)}")
print(f" - 객체 포함 이미지: {len(object_images)}")
print(f" - 객체 없는 이미지: {len(no_object_images)}")
print(f"학습 이미지 수: {len(train_list)}")
print(f"검증 이미지 수: {len(val_list)}")

print("\n[클래스별 객체 수]")
for cls in range(NUM_CLASSES):
    print(f"Class {cls}: train={train_class_counts[cls]}, val={val_class_counts[cls]}")

# ✅ 이게 출력 맨 아래로 이동!
num_noobj_train = sum(1 for img in train_list if img in noobj_train)
num_noobj_val = sum(1 for img in val_list if img in noobj_val)
print(f"\n[객체 없는 이미지]")
print(f" - 학습 이미지 중 객체 없는 이미지 수: {num_noobj_train}")
print(f" - 검증 이미지 중 객체 없는 이미지 수: {num_noobj_val}")

"""
비율:0.1
전체 이미지 수: 12538
 - 객체 포함 이미지: 5529
 - 객체 없는 이미지: 7009
학습 이미지 수: 11284
검증 이미지 수: 1254

[클래스별 객체 수]
Class 0: train=4179, val=454
Class 1: train=859, val=98
Class 2: train=999, val=128
Class 3: train=177, val=23

[객체 없는 이미지]
 - 학습 이미지 중 객체 없는 이미지 수: 6308
 - 검증 이미지 중 객체 없는 이미지 수: 701
"""