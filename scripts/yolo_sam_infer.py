import os
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from pycocotools import mask as mask_util

def mask_to_polygons(mask: np.ndarray):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    for cnt in contours:
        pts = cnt.squeeze().tolist()
        if len(pts) >= 6:
            polys.append([coord for pt in pts for coord in pt])
    return polys

def select_best_mask(masks, scores, box, iou_thresh=0.3):
    x1, y1, x2, y2 = map(int, box)
    box_mask = np.zeros(masks[0].shape, dtype=np.uint8)
    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 1, -1)
    best_idx, best_score = None, -1
    for i, (m, s) in enumerate(zip(masks, scores)):
        inter = np.logical_and(m, box_mask).sum()
        union = np.logical_or(m, box_mask).sum()
        iou = inter / union if union > 0 else 0
        if iou >= iou_thresh and s > best_score:
            best_score, best_idx = s, i
    if best_idx is None:
        best_idx = int(np.argmax(scores))
    return masks[best_idx]

def encode_rle(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def overlay_masks(image, masks, class_ids, names, alpha=0.4):
    canvas = image.copy()
    for mask, cid in zip(masks, class_ids):
        mask_bool = mask.astype(bool)
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))
        colored = np.zeros_like(image)
        colored[mask_bool] = color
        canvas = cv2.addWeighted(canvas, 1, colored, alpha, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, color, 2)
        ys, xs = np.where(mask)
        if xs.size and ys.size:
            pos = (int(xs.mean()), int(ys.mean()))
            label = names[cid] if 0 <= cid < len(names) else "unknown"
            cv2.putText(canvas, label, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/detect/yolo11s_urban_final_no_tuning2/weights/best.pt",
                   help="YOLO weights path")
    p.add_argument("--source",  default="../datasets/urban_yolo_final_all/images/test",
                   help="image folder or single image")
    p.add_argument("--conf",    type=float, default=0.2, help="detection confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--batch",   type=int,   default=16,    help="batch size for predict()")
    p.add_argument("--sam_ckpt", default="sam2.1_b.pt", help="SAM 2.1 checkpoint (.pt)")

    p.add_argument("--no‑multimask", dest="multimask", action="store_false", default=True,
                   help="disable multi‑mask proposals")
    p.add_argument("--no‑auto‑mask", dest="auto_mask", action="store_false", default=True,
                   help="disable full‑image SAM fallback")
    p.add_argument("--no‑rle", dest="export_rle", action="store_false", default=True,
                   help="disable saving RLE encodings")
    p.add_argument("--no‑polygons", dest="export_polygons", action="store_false", default=True,
                   help="disable saving polygon annotations")
    p.add_argument("--no‑vis", dest="save_vis", action="store_false", default=True,
                   help="disable overlay visualizations")

    p.add_argument("--out", default="runs/yolo_sam_output_no_tuning", help="output directory")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam_cfg   = r"C:\Users\butas\OneDrive\Documents\AN3\BitStone\bitstone-env\Lib\site-packages\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"
    sam_model = build_sam2(sam_cfg, ckpt_path=args.sam_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    mask_gen  = SAM2AutomaticMaskGenerator(sam_model)

    yolo = YOLO(args.weights).to(device)

    out_dir = Path(args.out)
    (out_dir / "masks_png").mkdir(parents=True, exist_ok=True)
    if args.export_rle:      (out_dir / "masks_rle").mkdir(exist_ok=True)
    if args.export_polygons: (out_dir / "masks_poly").mkdir(exist_ok=True)
    if args.save_vis:        (out_dir / "vis").mkdir(exist_ok=True)

    src  = Path(args.source)
    imgs = [src] if src.is_file() else sorted(src.glob("*.[jp][pn]g"))

    all_annotations = []
    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for batch_paths in chunked(imgs, args.batch):

        batch_bgr = [cv2.imread(str(p)) for p in batch_paths]
        batch_rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in batch_bgr]

        batch_results = yolo.predict(
            source=batch_rgb,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            augment=True,
            verbose=False
        )

        for img_path, img_bgr, det in zip(batch_paths, batch_bgr, batch_results):
            boxes   = det.boxes.xyxy.cpu().numpy()
            classes = det.boxes.cls.cpu().numpy().astype(int)

            masks_out, classes_out = [], []

            if len(boxes):
                predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                for box, cid in zip(boxes, classes):
                    masks, scores, _ = predictor.predict(
                        box=box[None, :],
                        multimask_output=bool(args.multimask)
                    )
                    mask = select_best_mask(masks, scores, box) if args.multimask else masks[0]
                    masks_out.append(mask)
                    classes_out.append(int(cid))

            elif args.auto_mask:
                for m in mask_gen.generate(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)):
                    masks_out.append(m["segmentation"])
                    classes_out.append(-1)

            if not masks_out:
                continue

            for idx, (mask, cid) in enumerate(zip(masks_out, classes_out)):
                base     = f"{img_path.stem}_{idx}"
                png_path = out_dir / "masks_png" / f"{base}.png"
                mask_bool = mask.astype(bool)
                masked = np.zeros_like(img_bgr)
                masked[mask_bool] = img_bgr[mask_bool]
                cv2.imwrite(str(png_path), masked)

                ann = {
                    "image":    img_path.name,
                    "mask_png": png_path.name,
                    "class_id": cid,
                    "class":    yolo.model.names[cid] if cid >= 0 else "unknown",
                }

                if args.export_rle:
                    rle = encode_rle(mask)
                    (out_dir / "masks_rle" / f"{base}.json").write_text(json.dumps(rle))
                    ann["rle"] = rle

                if args.export_polygons:
                    polys = mask_to_polygons(mask)
                    (out_dir / "masks_poly" / f"{base}.json").write_text(json.dumps(polys))
                    ann["polygons"] = polys

                all_annotations.append(ann)

            if args.save_vis:
                vis = overlay_masks(img_bgr, masks_out, classes_out, yolo.model.names)
                cv2.imwrite(str(out_dir / "vis" / f"{img_path.stem}_vis.png"), vis)

            print("Processed", img_path.name)

    (out_dir / "annotations.json").write_text(json.dumps(all_annotations, indent=2))
    print("✅ Done. Outputs in", out_dir)

if __name__ == "__main__":
    main()