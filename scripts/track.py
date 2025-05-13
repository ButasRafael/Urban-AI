
from __future__ import annotations

import argparse
import datetime as dt
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PERF_CFG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.7,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": None,
    "with_reid": True,
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.3,
    "model": "auto",
}



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--weights",
        type=str,
        default="runs/detect/yolo11s_urban_final_no_tuning2/weights/best.engine",
        help="Path to YOLO-11 .engine/.pt/.onnx weights",
    )
    p.add_argument(
        "--source",
        type=str,
        default="0",
        help="./potholes.mp4",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default=None,
        help="Path to tracker YAML; if omitted, a tuned BoT-SORT file is generated",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Detector confidence")
    p.add_argument("--iou", type=float, default=0.45, help="Detector IoU threshold")
    p.add_argument("--device", type=str, default="0", help="CUDA id or 'cpu'")
    p.add_argument("--half", action="store_true", help="Run FP16 inference")
    p.add_argument("--show", action="store_true", help="Show live window")
    p.add_argument("--save-video", action="store_true", help="Save annotated MP4")
    p.add_argument("--save-txt", action="store_true", help="Save YOLO txt track logs")
    p.add_argument("--trail", type=int, default=20, help="Motion-trail length (0 off)")
    p.add_argument("--output-dir", type=str, default="runs/track", help="Results root")
    return p.parse_args()


def get_tracker_yaml(path_arg: str | None) -> str:
    """Return a tracker YAML path: existing user path or generated temp file."""
    if path_arg:
        return path_arg

    tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    yaml.safe_dump(PERF_CFG, tmp)
    tmp.flush()
    return tmp.name


def main() -> None:
    args = parse_args()
    tracker_yaml = get_tracker_yaml(args.tracker)

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"track_{run_ts}"
    if args.save_video or args.save_txt:
        run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load model
    model = YOLO(args.weights)
    if args.half:
        model.half()

    # 2) Open source
    source = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Unable to open source {args.source!r}")

    # Prepare video writer
    writer, vid_path = None, None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        vid_path = str(run_dir / "tracked.mp4")
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    # 3) Tracking loop
    history: dict[int, list[tuple[float, float]]] = defaultdict(list)
    frame_idx, t0 = -1, time.perf_counter()

    while cap.isOpened():
        frame_idx += 1
        ok, frame = cap.read()
        if not ok:
            break

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

        vis = res.plot()

        if res.boxes.id is not None and args.trail > 0:
            for (x, y, w, h), tid in zip(
                res.boxes.xywh.cpu().numpy(), res.boxes.id.int().cpu().numpy()
            ):
                history[tid].append((float(x), float(y)))
                if len(history[tid]) > args.trail:
                    history[tid].pop(0)
                pts = np.array(history[tid], np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, (255, 255, 255), 2)

        if args.save_txt and res.boxes.id is not None:
            lines = []
            for box, cls, tid, conf in zip(
                res.boxes.xywhn.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
                res.boxes.id.cpu().numpy().astype(int),
                res.boxes.conf.cpu().numpy(),
            ):
                cx, cy, bw, bh = box
                lines.append(f"{tid} {cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}")
            (run_dir / f"{frame_idx:06d}.txt").write_text("\n".join(lines))

        if writer:
            writer.write(vis)
        if args.show:
            cv2.imshow("YOLO-11 Tracking", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    dt_sec = time.perf_counter() - t0
    print(f"✅ {frame_idx+1} frames in {dt_sec:.1f}s → {(frame_idx+1)/dt_sec:.2f} FPS")
    if vid_path:
        print(f"📹 video saved → {vid_path}")
    if args.save_txt:
        print(f"📝 track logs → {run_dir}")


if __name__ == "__main__":
    main()
