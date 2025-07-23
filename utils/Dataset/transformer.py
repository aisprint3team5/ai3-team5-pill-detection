from torchvision.transforms import ToTensor
from .preprocessing import *
import numpy as np
class PillImageTransform:
    def __init__(self, resize=(640, 640)):
        self.resize = resize
        self.to_tensor = ToTensor()


    def __call__(self, img_pil: Image.Image):
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Preprocess
        processed_cv = preprocess_pill_image(img_cv)

        # Resize (optional, since YOLO auto-resizes too)
        processed_cv = cv2.resize(processed_cv, self.resize)

        # Convert back to PIL for torchvision
        processed_pil = Image.fromarray(cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB))

        return self.to_tensor(processed_pil)


