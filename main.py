import os
import json
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from face_parsing.model import BiSeNet


# -----------------------------
# 1) Load model
# -----------------------------
def load_model(weights_path: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = BiSeNet(n_classes=19)  # this matches the common 79999_iter.pth
    state = torch.load(weights_path, map_location=device)
    net.load_state_dict(state)
    net.eval().to(device)
    return net, device


# -----------------------------
# 2) Run face parsing and extract HAIR mask
# -----------------------------
def get_parsing_map(net, device, img_bgr):
    """Returns parsing map (H,W) with class id per pixel."""
    h0, w0 = img_bgr.shape[:2]

    # model expects 512x512 RGB normalized
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)

    x = np.array(img_pil).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std

    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = net(x)[0]  # (1,19,512,512) in most implementations
        out = F.interpolate(out, size=(512, 512), mode="bilinear", align_corners=False)
        parsing_512 = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize back to original size
    parsing = cv2.resize(parsing_512, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return parsing


def get_hair_mask_from_parsing(parsing, img_shape):
    """
    Many CelebAMask-HQ BiSeNet weights use hair label = 17.
    If your mask looks wrong, this number is the first thing to change.
    """
    HAIR_LABEL = 17
    hair_mask = (parsing == HAIR_LABEL).astype(np.uint8) * 255

    # Optional cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, k, iterations=1)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, k, iterations=1)
    return hair_mask


# -----------------------------
# 3) Classify hair: blonde / brown / silver / bald
# -----------------------------
def classify_hair_4(img_bgr, hair_mask):
    hair_ratio = float((hair_mask > 0).mean())
    if hair_ratio < 0.01:
        return "bald", {"hair_ratio": hair_ratio}

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    m = hair_mask > 0
    Hm, Sm, Vm = H[m], S[m], V[m]

    # Use median to reduce highlight glare
    medH = float(np.median(Hm))
    medS = float(np.median(Sm))
    medV = float(np.median(Vm))

    # silver/gray: low saturation + bright
    if medS < 45 and medV > 150:
        return "silver", {"hair_ratio": hair_ratio, "medH": medH, "medS": medS, "medV": medV}

    # brown: darker
    if medV < 115:
        return "brown", {"hair_ratio": hair_ratio, "medH": medH, "medS": medS, "medV": medV}

    # blonde: bright-ish + yellow-ish hue
    if (medV >= 115) and (medS >= 45) and (10 <= medH <= 45):
        return "blonde", {"hair_ratio": hair_ratio, "medH": medH, "medS": medS, "medV": medV}

    # fallback
    return ("blonde" if medV >= 140 else "brown"), {"hair_ratio": hair_ratio, "medH": medH, "medS": medS, "medV": medV}


# -----------------------------
# 4) Process folder
# -----------------------------
def main():
    images_dir = "./images"
    out_dir = "./masks"
    os.makedirs(out_dir, exist_ok=True)

    weights_path = "./face_parsing/weights/79999_iter.pth"
    net, device = load_model(weights_path)

    results = []

    for fname in os.listdir(images_dir):
        path = os.path.join(images_dir, fname)
        if not os.path.isfile(path):
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        parsing = get_parsing_map(net, device, img)
        hair_mask = get_hair_mask_from_parsing(parsing, img.shape)

        # save mask so you can visually verify it
        cv2.imwrite(os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_hair.png"), hair_mask)

        label, dbg = classify_hair_4(img, hair_mask)
        results.append({"name": fname, "hair_color_4": label, "debug": dbg})

    with open("hair_4class_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done. Saved masks to ./masks and results to hair_4class_results.json")


if __name__ == "__main__":
    main()