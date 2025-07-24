import os
from glob import glob

def run_inference(detector, image_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    results = []
    
    for img_path in image_paths:
        result = detector.predict(img_path)
        results.append(result[0])
        
    return results