
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


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

    def __call__(self, img_pil: Image.Image):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        processed_cv = preprocess_pill_image(img_cv)
        processed_cv = cv2.resize(processed_cv, self.resize)
        # Convert back to PIL for torchvision
        processed_pil = Image.fromarray(cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB))
        return self.to_tensor(processed_pil)
