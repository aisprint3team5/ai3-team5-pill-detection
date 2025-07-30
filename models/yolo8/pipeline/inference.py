import os
from glob import glob
import config.path as PATH

def run_inference(detector):
    image_paths = sorted(glob(os.path.join(PATH.TEST_IMAGE_DIR, "*.jpg")))
    results = []
    
    for img_path in image_paths:
        result = detector.predict(img_path)
        results.append(result[0])
        
    return results