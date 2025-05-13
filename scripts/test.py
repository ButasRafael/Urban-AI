import argparse
import multiprocessing as mp
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate YOLO ONNX model with fixed thresholds"
    )
    parser.add_argument(
        "--weights", type=str,
        default="../weights/best_medium.engine",
        help="Path to the trained weights file (.pt)"
    )
    parser.add_argument(
        "--data", type=str,
        default="../datasets/urban_yolo_final_all/urban_yolo_final_all.yaml",
        help="Dataset YAML for validation"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--imgsz", type=int,
        default=640,
        help="Image size (single int for square)"
    )
    parser.add_argument(
        "--device", type=str,
        default="0",
        help="CUDA device index (e.g. '0') or 'cpu'"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Load the YOLO model (ONNX format)
    model = YOLO(args.weights)

    # Fixed thresholds
    conf_threshold = 0.2
    iou_threshold = 0.45

    # Print header
    header = f"{'conf':<8}{'iou':<8}{'mAP50-95':<10}{'mAP50':<10}{'mAP75':<10}"
    print("Running validation with conf=0.2 and IoU=0.45...\n")
    print(header)
    print('-' * len(header))

    # Validate once with fixed thresholds
    metrics = model.val(
        imgsz=args.imgsz,
        data=args.data,
        split=args.split,
        conf=conf_threshold,
        iou=iou_threshold,
        augment=True,
        save_json=True,
        save_txt=True,
        save_conf=True,
        plots=True,
        device=args.device,
        verbose=True,
    )
    #the rest of the settings are default and not needed to be specified, see https://docs.ultralytics.com/usage/cfg/

    # Extract metrics
    mAP5095 = metrics.box.map
    mAP50 = metrics.box.map50
    mAP75 = metrics.box.map75

    # Print results
    print(f"{conf_threshold:<8.3f}{iou_threshold:<8.2f}{mAP5095:<10.3f}{mAP50:<10.3f}{mAP75:<10.3f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()