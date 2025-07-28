
import cv2
import torch
import numpy as np
import albumentations as A
from PIL import Image
from torchvision.transforms import ToTensor
from albumentations.pytorch import ToTensorV2


def preprocess_pill_image(img: np.ndarray) -> np.ndarray:
    '''
    Preprocesses the pill image using grayscale, CLAHE, and optional adaptive thresholding.

    Args:
        img (np.ndarray): Input BGR image from cv2.imread.

    Returns:
        np.ndarray: Preprocessed grayscale image (still 3-channel for compatibility).
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_3ch


class PillImageTransform:
    def __init__(self, resize=(640, 640)):
        self.resize = resize
        self.to_tensor = ToTensor()

    def __call__(self, img_pil: Image.Image, labels, boxes):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        processed_cv = preprocess_pill_image(img_cv)
        processed_cv = cv2.resize(processed_cv, self.resize)
        # Convert back to PIL for torchvision
        processed_pil = Image.fromarray(cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB))
        return self.to_tensor(processed_pil), boxes


class AlbumentationTransform:
    def __init__(self, resize=(640, 640)):
        h, w = resize
        self.transform = A.Compose(
            [
                A.Resize(height=h, width=w),  # 이미지 + 박스 동시 리사이즈
                A.RandomBrightnessContrast(p=0.3),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Blur(p=0.2),
                A.CLAHE(p=0.2),
                A.Normalize(),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])  # required
        )

    def __call__(self, image, labels, boxes):
        """
        Args:
            image (PIL.Image or np.ndarray): Input image.
            labels (list): Class Ids
            boxes (list of box): Bounding boxes in [x_center, y_center, width, height]

        Returns:
            transformed image tensor, transformed boxes tensor
        """
        # Convert PIL to np.ndarray if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)

        bboxes = transformed['bboxes']
        clamped_boxes: list[list[float]] = []
        for x_c, y_c, bw, bh in bboxes:
            x_c: float = float(np.clip(x_c, 0.0, 1.0))
            y_c: float = float(np.clip(y_c, 0.0, 1.0))
            bw: float = float(np.clip(bw, 0.0, 1.0))
            bh: float = float(np.clip(bh, 0.0, 1.0))
            clamped_boxes.append([x_c, y_c, bw, bh])

        return transformed['image'], clamped_boxes
