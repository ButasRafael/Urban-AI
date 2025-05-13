
import albumentations as A
import cv2
import random
import uuid
from pathlib import Path
import numpy as np

ROOT   = Path("../datasets/urban_yolo_final_all")
SPLIT  = "train"
IMG_DIR = ROOT / f"images/{SPLIT}"
LAB_DIR = ROOT / f"labels/{SPLIT}"
OUT_IMG = ROOT / f"images/{SPLIT}_aug"; OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LAB = ROOT / f"labels/{SPLIT}_aug"; OUT_LAB.mkdir(parents=True, exist_ok=True)

CLASSES = [
  "pothole","graffiti","garbage","garbage_bin","overflow",
  "parking_illegal","parking_empty", "parking_legal", 
  "crack", "open_manhole","closed_manhole","fallen_tree",
]

NUM_CLASSES = len(CLASSES)

def uid():
    return uuid.uuid4().hex[:8]

def clip_bbox(b):
    return [min(max(x, 0.0), 1.0) for x in b]

# 1) Gather label stems per class
cls2stems = {c: [] for c in range(NUM_CLASSES)}
for txt in LAB_DIR.glob("*.txt"):
    lines = [l for l in txt.read_text().splitlines() if l.strip()]
    present = {int(l.split()[0]) for l in lines}
    for c in present:
        cls2stems[c].append(txt.stem)

# 2) Count & compute TARGET
counts = {c: len(stems) for c, stems in cls2stems.items()}
pct90 = int(np.percentile(list(counts.values()), 90)) 
TARGET = {cid: max(0, pct90 - counts[cid]) for cid in counts}

max_count = max(counts.values()) if counts else 0

print("Counts:", {CLASSES[c]: counts[c] for c in counts})
print("Targets:", {CLASSES[c]: TARGET[c] for c in TARGET})

#IF YOU WANT TO USE INSTANCE COUNTS with 2x CAP INSTEAD OF CLASS COUNTS, UNCOMMENT THE FOLLOWING LINES

# cls2stems       = {c: [] for c in range(NUM_CLASSES)}
# instance_counts = {c: 0  for c in range(NUM_CLASSES)}

# for txt in LAB_DIR.glob("*.txt"):
#     stem = txt.stem
#     present = set()
#     for ln in txt.read_text().splitlines():
#         if not ln.strip():
#             continue
#         cid = int(ln.split()[0])
#         instance_counts[cid] += 1
#         present.add(cid)
#     for cid in present:
#         cls2stems[cid].append(stem)

# # ───────── 2) compute targets ────────────────────────────────────────────────
# pct90          = int(np.percentile(list(instance_counts.values()), 90))
# TARGET         = {c: max(0, pct90 - instance_counts[c]) for c in range(NUM_CLASSES)}
# for c in TARGET:
#     cap        = int(instance_counts[c] * MAX_RATIO)
#     TARGET[c]  = min(TARGET[c], cap)

# max_count = max(TARGET.values()) if TARGET else 0
# print("Real boxes :", {CLASSES[c]: instance_counts[c] for c in range(NUM_CLASSES)})
# print("Synthetic →", {CLASSES[c]: TARGET[c]           for c in range(NUM_CLASSES)})





# 3) Build grouped pipeline
pipeline = A.Compose([
    # Geometric
    A.OneOf([
        A.RandomResizedCrop(size=(640,640), scale=(0.2,1.0), ratio=(0.75,1.33), p=1.0),
        A.HorizontalFlip(p=1.0),
        A.Affine(translate_percent=0.1, scale=(0.9,1.1), rotate=(-15,15), p=1.0),
        A.Perspective(scale=(0.05,0.1), keep_size=True, p=1.0),
    ], p=0.7),

    # Photometric & Color
    A.OneOf([
        A.AutoContrast(p=1.0),
        A.Equalize(p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        A.Posterize(num_bits=(4,8), p=1.0),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ], p=0.6),

    # Blur / Noise
    A.OneOf([
        A.Blur(blur_limit=3, p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.GlassBlur(sigma=1.0, max_delta=1, iterations=2, p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
        A.UnsharpMask(blur_limit=(3,7), p=1.0),
        A.GaussNoise(std_range=(0.2,0.44), mean_range=(0.0,0.0), per_channel=True, noise_scale_factor=1.0, p=1.0),
        A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.5), p=1.0),
        A.MultiplicativeNoise(multiplier=(0.8,1.2), p=1.0),
        A.ShotNoise(p=1.0),
        A.SaltAndPepper(p=1.0),
    ], p=0.5),

    # Weather & Lighting
    A.OneOf([
        A.RandomFog(fog_coef_range=(0.1,0.3), alpha_coef=0.08, p=1.0),
        A.RandomRain(slant_range=(-10,10), drop_length=20, drop_width=1,
                     drop_color=(200,200,200), blur_value=7,
                     brightness_coefficient=0.7, rain_type="default", p=1.0),
        A.RandomShadow(shadow_roi=(0,0.5,1,1),
                       num_shadows_limit=(1,2),
                       shadow_dimension=5,
                       shadow_intensity_range=(0.5,0.5), p=1.0),
        A.RandomSnow(snow_point_range=(0.1,0.3), brightness_coeff=2.5, method="texture", p=1.0),
        A.RandomSunFlare(src_radius=100, p=1.0),
    ], p=0.3),

    # Compression & Distortion
    A.OneOf([
        A.ImageCompression(quality_range=(75,100), compression_type="jpeg", p=1.0),
        A.Defocus(radius=(3,10), alias_blur=(0.1,0.5), p=1.0),
        A.Downscale(scale_range=(0.5,0.8),
                    interpolation_pair={'downscale': cv2.INTER_NEAREST,
                                        'upscale': cv2.INTER_NEAREST}, p=1.0),
    ], p=0.2),

    # Occlusion
    A.OneOf([
        A.ConstrainedCoarseDropout(num_holes_range=(2,3),
                        hole_height_range=(0.2,0.3),
                        hole_width_range=(0.2,0.3),
                        fill=0.0,
                        fill_mask=None, 
                        p=1.0,
                        bbox_labels=list(range(NUM_CLASSES))),
        A.GridDropout(ratio=0.2, p=1.0),
    ], p=0.3),
],
bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids'], min_visibility=0.2)
)

# 4) Class‑aware augmentation
for cid, stems in cls2stems.items():
    need = TARGET[cid]
    if need <= 0:
        continue

    print(f"Augmenting {CLASSES[cid]}: need {need}")
    added = 0
    attempts = 0
    max_attempts = need * 5
    p_wrap = min(1.0, need / max_count) if max_count else 1.0

    while added < need and attempts < max_attempts:
        attempts += 1
        stem = random.choice(stems)
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        if img is None:
            continue

        # load bboxes & class_ids
        boxes, cls_ids = [], []
        for ln in (LAB_DIR / f"{stem}.txt").read_text().splitlines():
            parts = ln.split()
            cls_ids.append(int(parts[0]))
            boxes.append(list(map(float, parts[1:])))

        # apply with group probability
        if random.random() < p_wrap:
            try:
                out = pipeline(image=img, bboxes=boxes, class_ids=cls_ids)
            except Exception:
                continue
            res_img, res_boxes, res_ids = out['image'], out['bboxes'], out['class_ids']
        else:
            res_img, res_boxes, res_ids = img, boxes, cls_ids

        if not res_boxes:
            continue

        new_id = uid()
        cv2.imwrite(str(OUT_IMG / f"{new_id}.jpg"), res_img)
        with open(OUT_LAB / f"{new_id}.txt", 'w') as f:
            for c, bb in zip(res_ids, res_boxes):
                bb = clip_bbox(bb)
                f.write(f"{c} " + " ".join(f"{v:.6f}" for v in bb) + "\n")

        added += sum(1 for c in res_ids if c == cid)

    if added < need:
        print(f"⚠️ {CLASSES[cid]}: added {added}/{need} after {attempts} attempts")

print("✅ Completed augmentation.")