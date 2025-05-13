
import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="High‑accuracy YOLO11 prediction pipeline")
    p.add_argument(
        "--weights",
        type=str,
        default="../weights/best_small.engine",
        help="Path to model weights"
    )
    p.add_argument(
        "--source",
        type=str,
        default="../images",
        help="Inference source (image/video/dir/stream)"
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for directory/video inference"
    )
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        batch=args.batch,
        conf=0.05,
        iou=0.45,
        augment=True, 
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        verbose=True
        #the rest of the settings are default and not needed to be specified, see https://docs.ultralytics.com/usage/cfg/
    )

    speeds = [r.speed for r in results]
    avg = {
        "preprocess": sum(s["preprocess"] for s in speeds) / len(speeds),
        "inference":  sum(s["inference"]  for s in speeds) / len(speeds),
        "postprocess":sum(s["postprocess"] for s in speeds) / len(speeds),
    }
    print(f"\nAverage per‑image speeds over {len(speeds)} frames:")
    print(f"  preprocess: {avg['preprocess']:.2f} ms")
    print(f"  inference:  {avg['inference']:.2f} ms")
    print(f"  postprocess:{avg['postprocess']:.2f} ms")


if __name__ == "__main__":
    main()
