import os
from collections import Counter

folder_path = "/home/cv/jw/AUE8088/datasets/kaist-rgbt/train/labels"
class_counts = Counter()
print("총 데이터 개수:", len(os.listdir(folder_path)))

cnt = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
            Bool = True
            for line in lines:
                line = line.strip()
                if line:  # 비어 있지 않은 줄만 처리
                    if Bool: 
                        cnt += 1
                        Bool = False
                    class_id = line.split()[0]
                    class_counts[class_id] += 1

# 결과 출력
print("객체 포함된 데이터 개수:", cnt)
total = 0
for class_id in sorted(class_counts.keys()):
    print(f"Class {class_id}: {class_counts[class_id]}")
    total += class_counts[class_id]
print("Total:", total)