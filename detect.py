import cv2
import numpy as np
import os
from PIL import Image

# ---------- Your function, but we also return the HSV center for debugging ----------
def get_limits(color_bgr):
    c = np.uint8([[color_bgr]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = int(hsvC[0][0][0])

    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return hsvC[0][0], lowerLimit, upperLimit

def debug_detect_color(image_path, color_name, color_bgr, min_area_px=500, show=True):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hsv_center, lower, upper = get_limits(color_bgr)

    mask = cv2.inRange(hsv, lower, upper)

    # Basic denoise so bbox isn't triggered by single pixels
    kernel = np.ones((5,5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # bbox via PIL (your approach)
    bbox = Image.fromarray(mask_clean).getbbox()  # (left, upper, right, lower) or None

    h, w = mask_clean.shape[:2]
    coverage = float(mask_clean.mean() / 255.0) * 100.0  # percentage of pixels that are 255

    # Also compute largest contour area (more informative than bbox existence)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_rect = None
    if contours:
        cmax = max(contours, key=cv2.contourArea)
        largest_area = int(cv2.contourArea(cmax))
        x, y, ww, hh = cv2.boundingRect(cmax)
        largest_rect = (x, y, ww, hh)

    hair_detected_bbox = bbox is not None
    hair_detected_area = largest_area >= min_area_px

    print("\n==============================")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Color: {color_name}  BGR={color_bgr}")
    print(f"HSV center from BGR: {tuple(int(v) for v in hsv_center)}")
    print(f"Lower HSV: {tuple(int(v) for v in lower)}")
    print(f"Upper HSV: {tuple(int(v) for v in upper)}")
    print(f"Mask coverage: {coverage:.3f}%")
    print(f"PIL bbox: {bbox}  -> detected={hair_detected_bbox}")
    print(f"Largest contour area: {largest_area}px  (min_area_px={min_area_px}) -> detected={hair_detected_area}")
    print("==============================")

    if show:
        # overlay
        overlay = image_bgr.copy()
        overlay[mask_clean > 0] = (0, 255, 0)  # green highlight for matched pixels
        blended = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

        # draw bbox / contour rect
        vis = blended.copy()
        if bbox is not None:
            l, u, r, d = bbox
            cv2.rectangle(vis, (l, u), (r, d), (255, 0, 0), 2)  # blue bbox from PIL
        if largest_rect is not None:
            x, y, ww, hh = largest_rect
            cv2.rectangle(vis, (x, y), (x+ww, y+hh), (0, 0, 255), 2)  # red rect from largest contour

        mask_vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        stack = np.hstack([
            cv2.resize(image_bgr, (0,0), fx=0.5, fy=0.5),
            cv2.resize(mask_vis, (0,0), fx=0.5, fy=0.5),
            cv2.resize(vis, (0,0), fx=0.5, fy=0.5),
        ])

        cv2.imshow(f"DEBUG: {color_name} (orig | mask | overlay)", stack)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        # Controls
        # ESC quits, 's' saves debug images
        if key == 27:
            return "quit"
        if key == ord('s'):
            base = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(f"debug_{base}_{color_name}_mask.png", mask_clean)
            cv2.imwrite(f"debug_{base}_{color_name}_overlay.png", vis)
            print(f"Saved debug_{base}_{color_name}_mask.png and overlay.")

    return "next"

def debug_one_image_all_colors(image_path):
    colors = {
        "black": [0, 0, 0],
        "white": [255, 255, 255],
        "red": [0, 0, 255],
        "green": [0, 255, 0],
        "blue": [255, 0, 0],
        "yellow": [0, 255, 255],
        "cyan": [255, 255, 0],
        "magenta": [255, 0, 255],
        "silver": [192, 192, 192],
        "gray": [128, 128, 128],
        "maroon": [0, 0, 128],
        "olive": [0, 128, 128],
        "purple": [128, 0, 128],
        "teal": [128, 128, 0],
        "navy": [128, 0, 0],
        "orange": [0, 165, 255],
        "pink": [203, 192, 255],
        "lime": [0, 255, 128],
        "brown": [42, 42, 165],
        "gold": [0, 215, 255],
        "beige": [220, 245, 245],
        "tan": [140, 180, 210],
        "chocolate": [30, 105, 210],
        "coral": [80, 127, 255]
    }

    for name, bgr in colors.items():
        action = debug_detect_color(image_path, name, bgr, min_area_px=500, show=True)
        if action == "quit":
            break

if __name__ == "__main__":
    # point to one test image first
    test_image = "./images/ben-p&p.png"
    debug_one_image_all_colors(test_image)