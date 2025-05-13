#!/usr/bin/env python3
"""
 yolo_sam2_pipeline_image_only.py
 ────────────────────────────────
 1) YOLO-11 + BoT-SORT detection & tracking pass
 2) Ultralytics SAM2 (image-only) segmentation pass, frame by frame
 Produces:
   • runs/track/YYYYMMDD_HHMMSS/yolo_frames/*.jpg
   • runs/track/YYYYMMDD_HHMMSS/detections.json
   • runs/track/YYYYMMDD_HHMMSS/segmented.mp4
"""
import argparse
import datetime as dt
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
import supervision as sv
from ultralytics import YOLO, SAM

PERF_CFG = {
    "tracker_type":      "botsort",
    "track_high_thresh": 0.6,
    "track_low_thresh":  0.1,
    "new_track_thresh":  0.7,
    "track_buffer":      30,
    "match_thresh":      0.8,
    "fuse_score":        True,
    "gmc_method":        None,
    "with_reid":         False,
    "proximity_thresh":  0.5,
    "appearance_thresh": 0.3,
    "model":             "auto",
}

def get_tracker_yaml(user_path: str | None) -> str:
    if user_path:
        return user_path
    tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    yaml.safe_dump(PERF_CFG, tmp)
    tmp.flush()
    return tmp.name

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    default="runs/detect/yolo11s_urban_final_no_tuning_medium/weights/best.pt")
    p.add_argument("--source",     default="potholes.mp4", help="Path or device index")
    p.add_argument("--tracker",    default=None, help="Path to BoT-SORT YAML config")
    p.add_argument("--imgsz",      type=int,   default=640)
    p.add_argument("--conf",       type=float, default=0.25)
    p.add_argument("--iou",        type=float, default=0.45)
    p.add_argument("--device",     default="0")
    p.add_argument("--half",       action="store_true")
    p.add_argument("--output-dir", default="runs/track")
    return p.parse_args()

def run_detection_and_track(args):
    ts      = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"track_{ts}"
    img_dir = run_dir / "yolo_frames"
    img_dir.mkdir(parents=True)

    model = YOLO(args.weights)
    if args.half:
        model.model.half()

    tracker_yaml = get_tracker_yaml(args.tracker)

    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {args.source!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detections = []
    frame_idx, t0 = -1, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        res = model.track(
            frame,
            persist=True,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            tracker=tracker_yaml,
            verbose=False,
        )[0]

        frame_path = img_dir / f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        if res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids  = res.boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                detections.append({
                    "frame_idx": frame_idx,
                    "track_id":  int(tid),
                    "bbox":      [float(x1), float(y1), float(x2), float(y2)],
                })

    cap.release()
    elapsed = time.perf_counter() - t0
    print(f"→ Detected {frame_idx+1} frames in {elapsed:.1f}s → {(frame_idx+1)/elapsed:.2f} FPS")

    (run_dir / "detections.json").write_text(json.dumps(detections, indent=2))
    return run_dir, fps, (w, h), detections

def run_sam2_image_only(run_dir, fps, vsize, detections, args):
    from collections import defaultdict
    dets_by_frame = defaultdict(list)
    for det in detections:
        dets_by_frame[det["frame_idx"]].append(det)

    sam = SAM("sam2.1_b.pt")
    if args.half:
        sam.model.half()

    out_path = run_dir / "segmented.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        vsize,
    )

    for frame_idx in sorted(dets_by_frame):
        frame = cv2.imread(str(run_dir / "yolo_frames" / f"{frame_idx:05d}.jpg"))
        frame_dets = dets_by_frame[frame_idx]
        bboxes = [d["bbox"] for d in frame_dets]

        res = sam(
            source=[cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)],
            bboxes=[bboxes],
        )[0]

        masks = res.masks.data.cpu().numpy().astype(bool)
        track_ids = [d["track_id"] for d in frame_dets]

        dets = sv.Detections(
            xyxy      = sv.mask_to_xyxy(masks=masks),
            mask      = masks,
            class_id  = np.array(track_ids),
        )
        ann = sv.MaskAnnotator().annotate(scene=frame, detections=dets)
        writer.write(ann)

    writer.release()
    print(f"→ Segmented video saved → {out_path}")

def main():
    args = parse_args()
    run_dir, fps, vsize, dets = run_detection_and_track(args)
    run_sam2_image_only(run_dir, fps, vsize, dets, args)

if __name__ == "__main__":
    main()
