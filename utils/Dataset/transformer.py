from torchvision.transforms import ToTensor
from preprocessing import *
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import torch


class AlbumentationTransform:
    def __init__(self):
        self.transform = A.Compose(
            [
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

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image or np.ndarray): Input image.
            boxes (torch.Tensor or list): Bounding boxes in [class_id, x_center, y_center, width, height] format (YOLO).

        Returns:
            transformed image tensor, transformed boxes tensor
        """
        # Convert PIL to np.ndarray if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Prepare bbox list and class labels separately
        if len(boxes) > 0:
            bboxes = boxes[:, 1:].tolist()  # only bbox coords
            class_labels = boxes[:, 0].tolist()
        else:
            bboxes = []
            class_labels = []

        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

        # Recombine boxes: [class_id, x_center, y_center, w, h]
        transformed_boxes = []
        for cls_id, bbox in zip(class_labels, transformed['bboxes']):
            transformed_boxes.append([cls_id] + list(bbox))

        transformed_boxes = torch.tensor(transformed_boxes, dtype=torch.float32)

        return transformed['image'], transformed_boxes

    
# class PillImageTransform:
#     def __init__(self, resize=(640, 640)):
#         self.resize = resize
#         self.to_tensor = ToTensor()


#     def __call__(self, img_pil: Image.Image):
#         # Convert PIL to OpenCV
#         img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

#         # Preprocess
#         processed_cv = preprocess_pill_image(img_cv)

#         # Resize (optional, since YOLO auto-resizes too)
#         processed_cv = cv2.resize(processed_cv, self.resize)

#         # Convert back to PIL for torchvision
#         processed_pil = Image.fromarray(cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB))

#         return self.to_tensor(processed_pil)


