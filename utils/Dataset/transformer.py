from torchvision.transforms import ToTensor
from preprocessing import *
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image

class AlbumentationTransform:
    def __init__(self, resize=(640, 640), apply_clahe=True):
        """
        Args:
            apply_clahe (bool): Whether to apply CLAHE for contrast enhancement.
        """
        transforms = []

        if apply_clahe:
            transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0))

        transforms.append(A.Resize(height=resize[1], width=resize[0]))
        transforms.append(A.ToFloat(max_value=255.0))
        transforms.append(ToTensorV2())

        self.transform = A.Compose(transforms)

    def __call__(self, img_pil: Image.Image):

        img_np = np.array(img_pil)  #  PIL to RGB numpy
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Albumentations expects BGR
        transformed = self.transform(image=img_bgr)
        return transformed['image']

    
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


