
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT engine.")
    parser.add_argument(
        "--weights", type=str, default="../weights/best_medium.pt",
        help="Path to the trained weights file (.pt)"
    )
    parser.add_argument(
        "--format", type=str, default="engine",
        choices=["engine"],
        help="Target export format (TensorRT engine)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size (single int for square)"
    )
    parser.add_argument(
        "--dynamic", action='store_true',
        help="Enable dynamic input sizes"
    )
    parser.add_argument(
        "--simplify", action='store_true', default=True,
        help="Simplify graph for TensorRT export (default: True)"
    )
    parser.add_argument(
        "--half", action='store_true',
        help="Enable FP16 precision"
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch size for exported engine"
    )
    parser.add_argument(
        "--workspace", type=float, default=None,
        help="Max workspace size in GiB"
    )
    parser.add_argument(
        "--int8", action='store_true',
        help="Enable INT8 quantization (requires --data)"
    )
    parser.add_argument(
        "--data", type=str, default="../datasets/urban_yolo_final_all/urban_yolo_final_all.yaml",
        help="Dataset YAML for INT8 calibration"
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of dataset for INT8 calibration"
    )
    parser.add_argument(
        "--nms", action='store_true',
        help="Add NMS to the exported engine"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device for export ('0' for GPU)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    export_kwargs = {
        'format': args.format,
        'imgsz': args.imgsz,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'half': args.half,
        'batch': args.batch,
        'workspace': args.workspace,
        'int8': args.int8,
        'data': args.data,
        'fraction': args.fraction,
        'nms': args.nms,
        'device': args.device
    }

    export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None and v is not False}

    print(f"Exporting {args.weights} with options: {export_kwargs}")
    model.export(**export_kwargs)
    print("TensorRT export completed successfully.")

if __name__ == '__main__':  # noqa
    main()