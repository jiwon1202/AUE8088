

# utils/dataloaders.py에 추가/수정 # !!
import random
import numpy as np
import cv2
import torch
from torchvision import transforms


class MODIFIED_LoadRGBTImagesAndLabels(Dataset):
    """RGB-T 멀티스펙트럴 데이터를 위한 동기화된 데이터로더"""
    
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, 
                 rect=False, image_weights=False, cache_images=False, single_cls=False,
                 stride=32, pad=0.0, min_items=0, prefix="", rank=-1, seed=0, modality=None):
        
        # 기존 초기화 코드...
        self.augment = augment
        self.hyp = hyp
        self.img_size = img_size
        
        # RGB-T 특화 증강 파라미터
        self.rgbt_augment_params = {
            'sync_geometric': True,      # 기하학적 변환 동기화
            'separate_photometric': True, # 색상/강도 변환 분리
            'thermal_normalization': True # Thermal 이미지 정규화
        }
        if modality == "rgbt" or modality is None:
            self.output_both_modalities = True
        else:
            self.output_both_modalities = False
            self.selected_modality = modality

    def __getitem__(self, index):
        index = self.indices[index]
        
        # 1. 이미지 로드
        img_rgb, img_thermal = self.load_rgbt_images(index)
        h0, w0 = img_rgb.shape[:2]  # 원본 크기
        r = self.img_size / max(h0, w0)  # 리사이즈 비율
        
        if r != 1:  # 리사이즈 필요한 경우
            img_rgb = cv2.resize(img_rgb, (int(w0 * r), int(h0 * r)), 
                               interpolation=cv2.INTER_LINEAR)
            img_thermal = cv2.resize(img_thermal, (int(w0 * r), int(h0 * r)), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # 2. 라벨 로드
        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], int(w0 * r), int(h0 * r), 
                                      padw=0, padh=0)
        
        # 3. 동기화된 증강 적용
        if self.augment:
            img_rgb, img_thermal, labels = self.apply_synchronized_augmentation(
                img_rgb, img_thermal, labels
            )
        
        # 4. 최종 전처리
        img_rgb, img_thermal = self.final_preprocessing(img_rgb, img_thermal)
        
        return [img_rgb, img_thermal], labels, self.im_files[index], (h0, w0), index
    
    def load_rgbt_images(self, index):
        """RGB와 Thermal 이미지를 동시에 로드"""
        # RGB 이미지 로드
        path_rgb = self.im_files[index]
        img_rgb = cv2.imread(path_rgb)
        assert img_rgb is not None, f'RGB Image Not Found {path_rgb}'
        
        # Thermal 이미지 로드 (경로 변환)
        path_thermal = path_rgb.replace('/visible/', '/lwir/')
        img_thermal = cv2.imread(path_thermal, cv2.IMREAD_GRAYSCALE)
        assert img_thermal is not None, f'Thermal Image Not Found {path_thermal}'
        
        # Thermal 이미지를 3채널로 확장 (처리 일관성을 위해)
        img_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_GRAY2BGR)
        
        return img_rgb, img_thermal
    
    def apply_synchronized_augmentation(self, img_rgb, img_thermal, labels):
        """RGB와 Thermal 이미지에 동기화된 증강 적용"""
        
        # 1. 공통 랜덤 시드 설정 (동일한 기하학적 변환 보장)
        seed = random.randint(0, 2**32 - 1)
        
        # 2. 기하학적 변환 (동기화)
        img_rgb, img_thermal, labels = self.apply_geometric_augmentation(
            img_rgb, img_thermal, labels, seed
        )
        
        # 3. 색상/강도 변환 (분리)
        img_rgb = self.apply_photometric_augmentation_rgb(img_rgb)
        img_thermal = self.apply_photometric_augmentation_thermal(img_thermal)
        
        # 4. 노이즈 추가 (선택적)
        if random.random() < self.hyp.get('noise_prob', 0.1):
            img_rgb, img_thermal = self.add_synchronized_noise(img_rgb, img_thermal)
        
        return img_rgb, img_thermal, labels
    
    def apply_geometric_augmentation(self, img_rgb, img_thermal, labels, seed):
        """동기화된 기하학적 변환"""
        
        # 동일한 시드로 변환 파라미터 생성
        random.seed(seed)
        np.random.seed(seed % (2**31))
        
        h, w = img_rgb.shape[:2]
        
        # 회전
        if random.random() < self.hyp.get('degrees_prob', 0.5):
            angle = random.uniform(-self.hyp.get('degrees', 0), self.hyp.get('degrees', 0))
            center = (w // 2, h // 2)
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            img_rgb = cv2.warpAffine(img_rgb, M_rot, (w, h), 
                                   flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            img_thermal = cv2.warpAffine(img_thermal, M_rot, (w, h), 
                                       flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            
            # 라벨 회전 적용
            if labels.size:
                labels = self.rotate_labels(labels, M_rot, w, h)
        
        # 이동 (Translation)
        if random.random() < self.hyp.get('translate_prob', 0.5):
            tx = random.uniform(-self.hyp.get('translate', 0), self.hyp.get('translate', 0)) * w
            ty = random.uniform(-self.hyp.get('translate', 0), self.hyp.get('translate', 0)) * h
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            
            img_rgb = cv2.warpAffine(img_rgb, M_trans, (w, h), 
                                   flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            img_thermal = cv2.warpAffine(img_thermal, M_trans, (w, h), 
                                       flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
            
            # 라벨 이동 적용
            if labels.size:
                labels[:, 1] += tx / w  # x center
                labels[:, 2] += ty / h  # y center
        
        # 스케일링
        if random.random() < self.hyp.get('scale_prob', 0.5):
            scale = random.uniform(1 - self.hyp.get('scale', 0), 1 + self.hyp.get('scale', 0))
            new_w, new_h = int(w * scale), int(h * scale)
            
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_thermal = cv2.resize(img_thermal, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 원본 크기로 crop 또는 pad
            img_rgb = self.resize_and_pad(img_rgb, (w, h))
            img_thermal = self.resize_and_pad(img_thermal, (w, h))
        
        # 좌우 뒤집기
        if random.random() < self.hyp.get('fliplr', 0):
            img_rgb = np.fliplr(img_rgb).copy()
            img_thermal = np.fliplr(img_thermal).copy()
            if labels.size:
                labels[:, 1] = 1 - labels[:, 1]  # x center flip
        
        # 상하 뒤집기 (보행자에게는 부적절하므로 낮은 확률)
        if random.random() < self.hyp.get('flipud', 0):
            img_rgb = np.flipud(img_rgb).copy()
            img_thermal = np.flipud(img_thermal).copy()
            if labels.size:
                labels[:, 2] = 1 - labels[:, 2]  # y center flip
        
        return img_rgb, img_thermal, labels
    
    def apply_photometric_augmentation_rgb(self, img_rgb):
        """RGB 이미지용 색상 증강"""
        
        # HSV 변환
        if random.random() < 0.5:
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
            
            # Hue 조정
            h_gain = random.uniform(-self.hyp.get('hsv_h', 0), self.hyp.get('hsv_h', 0))
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + h_gain * 180) % 180
            
            # Saturation 조정
            s_gain = random.uniform(1 - self.hyp.get('hsv_s', 0), 1 + self.hyp.get('hsv_s', 0))
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * s_gain, 0, 255)
            
            # Value 조정
            v_gain = random.uniform(1 - self.hyp.get('hsv_v', 0), 1 + self.hyp.get('hsv_v', 0))
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * v_gain, 0, 255)
            
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        return img_rgb
    
    def apply_photometric_augmentation_thermal(self, img_thermal):
        """Thermal 이미지용 강도 증강"""
        
        # Thermal 이미지는 강도 정보만 의미가 있으므로 단순한 변환 적용
        if random.random() < 0.5:
            # 밝기 조정
            brightness_gain = random.uniform(0.8, 1.2)
            img_thermal = np.clip(img_thermal * brightness_gain, 0, 255).astype(np.uint8)
            
            # 대비 조정
            contrast_gain = random.uniform(0.8, 1.2)
            img_thermal = np.clip((img_thermal - 127.5) * contrast_gain + 127.5, 0, 255).astype(np.uint8)
            
            # Thermal 특화: 히스토그램 평활화 (선택적)
            if random.random() < 0.3:
                img_thermal_gray = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
                img_thermal_eq = cv2.equalizeHist(img_thermal_gray)
                img_thermal = cv2.cvtColor(img_thermal_eq, cv2.COLOR_GRAY2BGR)
        
        return img_thermal
    
    def add_synchronized_noise(self, img_rgb, img_thermal):
        """동기화된 노이즈 추가"""
        
        # 동일한 노이즈 패턴 생성
        h, w = img_rgb.shape[:2]
        noise_pattern = np.random.normal(0, 10, (h, w, 1)).astype(np.float32)
        
        # RGB에는 색상 노이즈
        rgb_noise = np.repeat(noise_pattern, 3, axis=2)
        img_rgb = np.clip(img_rgb.astype(np.float32) + rgb_noise, 0, 255).astype(np.uint8)
        
        # Thermal에는 강도 노이즈
        thermal_noise = noise_pattern * 0.5  # 더 약한 노이즈
        thermal_noise = np.repeat(thermal_noise, 3, axis=2)
        img_thermal = np.clip(img_thermal.astype(np.float32) + thermal_noise, 0, 255).astype(np.uint8)
        
        return img_rgb, img_thermal
    
    def final_preprocessing(self, img_rgb, img_thermal):
        """최종 전처리"""
        
        # 정규화
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_thermal = img_thermal.astype(np.float32) / 255.0
        
        # Thermal 이미지 특화 정규화
        if self.rgbt_augment_params['thermal_normalization']:
            # Thermal 이미지의 동적 범위 조정
            img_thermal_gray = cv2.cvtColor((img_thermal * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            thermal_min, thermal_max = img_thermal_gray.min(), img_thermal_gray.max()
            if thermal_max > thermal_min:
                img_thermal_normalized = (img_thermal_gray - thermal_min) / (thermal_max - thermal_min)
                img_thermal = cv2.cvtColor((img_thermal_normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        
        # HWC to CHW
        img_rgb = img_rgb.transpose(2, 0, 1)
        img_thermal = img_thermal.transpose(2, 0, 1)
        
        # Numpy to Tensor
        img_rgb = torch.from_numpy(img_rgb).float()
        img_thermal = torch.from_numpy(img_thermal).float()
        
        return img_rgb, img_thermal
    
    def rotate_labels(self, labels, M, w, h):
        """회전 변환 시 라벨 좌표 조정"""
        if labels.size == 0:
            return labels
        
        # xywh를 corner points로 변환
        corners = xywh2corners(labels[:, 1:], w, h)
        
        # 회전 적용
        corners_homogeneous = np.hstack([corners, np.ones((corners.shape[0], 1))])
        corners_rotated = corners_homogeneous @ M.T
        
        # 다시 xywh로 변환
        labels[:, 1:] = corners2xywh(corners_rotated, w, h)
        
        return labels
    
    def resize_and_pad(self, img, target_size):
        """이미지를 목표 크기로 리사이즈하고 패딩"""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        if w > target_w or h > target_h:
            # Crop center
            start_x = (w - target_w) // 2 if w > target_w else 0
            start_y = (h - target_h) // 2 if h > target_h else 0
            img = img[start_y:start_y + target_h, start_x:start_x + target_w]
        else:
            # Pad
            pad_x = (target_w - w) // 2
            pad_y = (target_h - h) // 2
            img = cv2.copyMakeBorder(img, pad_y, target_h - h - pad_y, 
                                   pad_x, target_w - w - pad_x, 
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img

# 헬퍼 함수들
def xywh2corners(xywh, w, h):
    """xywh를 corner points로 변환"""
    corners = np.zeros((xywh.shape[0], 8))
    for i, (x, y, width, height) in enumerate(xywh):
        x1, y1 = (x - width/2) * w, (y - height/2) * h
        x2, y2 = (x + width/2) * w, (y + height/2) * h
        corners[i] = [x1, y1, x2, y1, x2, y2, x1, y2]
    return corners.reshape(-1, 4, 2)

def corners2xywh(corners, w, h):
    """corner points를 xywh로 변환"""
    xywh = np.zeros((corners.shape[0], 4))
    for i, corner in enumerate(corners):
        x_coords, y_coords = corner[:, 0], corner[:, 1]
        x1, x2 = x_coords.min(), x_coords.max()
        y1, y2 = y_coords.min(), y_coords.max()
        
        xywh[i] = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, 
                   (x2 - x1) / w, (y2 - y1) / h]
    return xywh
