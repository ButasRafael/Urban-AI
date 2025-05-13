
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format.")
    parser.add_argument(
        "--weights", type=str, default="runs/detect/yolo11s_urban_final_no_tuning2/weights/best.pt",
        help="Path to the trained weights file (.pt)"
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
        help="Simplify ONNX graph (default: True)"
    )
    parser.add_argument(
        "--opset", type=int, default=None,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--half", action='store_true',
        help="Enable FP16 precision"
    )
    parser.add_argument(
        "--nms", action='store_true',
        help="Add NMS to the exported model"
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch size for exported model"
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
        'format': 'onnx',
        'imgsz': args.imgsz,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'opset': args.opset,
        'half': args.half,
        'nms': args.nms,
        'batch': args.batch,
        'device': args.device
    }

    export_kwargs = {k: v for k, v in export_kwargs.items() if v is not None and v is not False}

    print(f"Exporting {args.weights} to ONNX with options: {export_kwargs}")
    model.export(**export_kwargs)
    print("ONNX export completed successfully.")