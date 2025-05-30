#utils/image_processing.py
import os
import cv2
import numpy as np

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captcha_templates")
templates = {}

def preprocess_image(img, size=(30, 50)):
    """
    Resize, blur, and binarize image for consistent template matching
    """
    resized = cv2.resize(img, size)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def load_templates():
    """
    Load all template images from disk into memory
    """
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    templates.clear()
    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            label = filename.split("_")[0]
            path = os.path.join(template_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = preprocess_image(img)
                templates.setdefault(label, []).append(img)
    print(f"Loaded {sum(len(v) for v in templates.values())} templates for {len(templates)} labels.")

def crop_captcha(img, num_chars=4):
    """
    Crop the captcha image into individual characters
    """
    height, width = img.shape
    char_width = width // num_chars
    os.makedirs("cropped_debug", exist_ok=True)
    chars = []
    for i in range(num_chars):
        x_start = i * char_width
        char_img = img[0:height, x_start:x_start + char_width]
        cv2.imwrite(f"cropped_debug/char_{i}.png", char_img)
        chars.append(preprocess_image(char_img))
    return chars

# def match_template(img_char):
#     """
#     Match the character image to the best template using Top-N average scoring
#     """
#     img_char = preprocess_image(img_char)  # <-- เพิ่มตรงนี้
#     best_label = None
#     best_score = float('inf')
#     label_scores = {}

#     for label, template_list in templates.items():
#         scores = []
#         for template_img in template_list:
#             res = cv2.matchTemplate(img_char, template_img, cv2.TM_SQDIFF_NORMED)
#             min_val, _, _, _ = cv2.minMaxLoc(res)
#             scores.append(min_val)

#         if scores:
#             top_n = sorted(scores)[:min(3, len(scores))]
#             avg_score = sum(top_n) / len(top_n)
#             label_scores[label] = avg_score
#             if avg_score < best_score:
#                 best_score = avg_score
#                 best_label = label

#     sorted_scores = sorted(label_scores.items(), key=lambda x: x[1])
#     if len(sorted_scores) >= 2:
#         best_score = sorted_scores[0][1]
#         second_best_score = sorted_scores[1][1]
#     else:
#         second_best_score = float('inf')

#     threshold = 0.4
#     if best_score > threshold or (second_best_score - best_score) < 0.02:
#         return "?"
#     return best_label or "?"

def match_template(img_char):
    """
    Match the character image to the best template using Top-N scoring.
    Returns best label and its confidence (0-100%), calculated from best match only.
    """
    img_char = preprocess_image(img_char)
    best_label = None
    best_score = float('inf')
    label_scores = {}

    for label, template_list in templates.items():
        min_scores = []
        for template_img in template_list:
            res = cv2.matchTemplate(img_char, template_img, cv2.TM_SQDIFF_NORMED)
            min_val, _, _, _ = cv2.minMaxLoc(res)
            min_scores.append(min_val)

        if min_scores:
            best_score_for_label = min(min_scores)  # <-- ใช้ค่าต่ำสุดเท่านั้น
            label_scores[label] = best_score_for_label
            if best_score_for_label < best_score:
                best_score = best_score_for_label
                best_label = label

    sorted_scores = sorted(label_scores.items(), key=lambda x: x[1])
    print("Top 1 match:")
    for label, score in sorted_scores[:1]:
        confidence = max(0.0, min(100.0, (1.0 - score) * 100.0))
        print(f"  {label}: {confidence:.0f}%")

    best_confidence = max(0.0, min(100.0, (1.0 - best_score) * 100.0))
    return best_label if best_label is not None else "?", best_confidence


def save_templates(label, char_images):
    """
    Save new character images into the template folder
    """
    saved_files = []
    for i, char_img in enumerate(char_images):
        char_label = label[i]
        existing = [f for f in os.listdir(template_dir) if f.startswith(char_label + "_")]
        next_index = len(existing)
        filename = f"{char_label}_{next_index}.png"
        filepath = os.path.join(template_dir, filename)
        cv2.imwrite(filepath, preprocess_image(char_img))
        saved_files.append(filename)
    return saved_files
