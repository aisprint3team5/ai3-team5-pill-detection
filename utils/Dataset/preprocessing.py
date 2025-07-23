
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
#from script.db.create_schema import DB_PATH
from utils.common import *

import cv2
import numpy as np
from PIL import Image


def preprocess_pill_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocesses the pill image using grayscale, CLAHE, and optional adaptive thresholding.

    Args:
        img (np.ndarray): Input BGR image from cv2.imread.

    Returns:
        np.ndarray: Preprocessed grayscale image (still 3-channel for compatibility).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)

    # Convert to 3-channel again if needed (YOLOv5 expects 3-channel images)
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced_3ch



def show_img(image, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # 딱 맞는 사각형 좌표
        # bounding_rects.append((x/1280, y/976, (x + w)/1280, (y + h)/976)) # 사각형 그릴떄 쓰는 좌표

    cv2.rectangle(image, (x, y, w, h), (0, 200, 0), 2)  # 사각형 그림
    cv2.drawContours(image, contours, -1, (0, 200, 0))
    cv2.imshow('contour', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def resolve_label_ids_from_filename(filename, category_lookup, class_name_to_idx):
    category_ids = common.extract_category_ids_from_filename(filename)
    label_ids = []

    for cid in category_ids:
        cname = category_lookup.get(cid)
        if cname is None:
            continue
        label_id = class_name_to_idx.get(cname)
        if label_id is not None:
            label_ids.append(label_id)

    return label_ids


def add_bounding_boxes_with_correct_labels(
    new_bbox_dir, output_label_dir, class_name_to_idx, category_lookup
):
    os.makedirs(output_label_dir, exist_ok=True)

    for filename in os.listdir(new_bbox_dir):
        if not filename.endswith(".txt"):
            continue

        bbox_path = os.path.join(new_bbox_dir, filename)
        if not os.path.isfile(bbox_path):
            continue

        # Step 1–4: get correct label_id
        label_ids = resolve_label_ids_from_filename(filename, category_lookup, class_name_to_idx)
        if not label_ids:
            print(f"Skipping {filename}: no matching category name found.")
            continue

        correct_label_id = label_ids[0]  # If multiple IDs, choose the first

        # Step 5: Read bbox values and attach correct label_id
        with open(bbox_path, "r", encoding="utf-8") as f:
            bbox_lines = f.readlines()

        if not bbox_lines:
            continue

        new_label_lines = []
        for line in bbox_lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip invalid line
            _, cx, cy, w, h = parts
            new_label_lines.append(f"{correct_label_id} {cx} {cy} {w} {h}")

        # Step 6: Write to label file
        out_path = os.path.join(output_label_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for line in new_label_lines:
                f.write(line + "\n")

        print(f" Fixed label written for {filename}")

