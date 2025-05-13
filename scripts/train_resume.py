
import argparse, yaml
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="runs/detect/yolo11s_urban_final_no_tuning",
                   help="path to previous run folder")
    p.add_argument("--extra_epochs", type=int, default=20,
                   help="how many more epochs to train")
    p.add_argument("--imgsz", type=int, default=640,
                   help="fine-tune resolution (same or higher)")
    p.add_argument("--hyp", default="../datasets/urban_yolo_final/best_hyperparameters.yaml")
    p.add_argument("--name", default="yolo11s_urban_finetune")
    return p.parse_args()

def main():
    args = parse_args()

    ckpt = Path(args.run_dir) / "weights" / "last.pt"
    assert ckpt.is_file(), f"Checkpoint not found: {ckpt}"

    hyp = yaml.safe_load(Path(args.hyp).read_text())
    hyp.update({
        "cls": 0.6,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "lr0": 1e-5,
        "lrf": 0.01
    })
    model = YOLO(str(ckpt))
    model.train(
        resume=True,
        epochs=args.extra_epochs,
        imgsz=args.imgsz,
        batch=16, # or -1 for batch size auto-tuning
        #cache='ram' or 'disk', if enough RAM or disk space
        #multi_scale=True, if enough GPU memory
        cos_lr=True,
        amp=True,
        close_mosaic=0,
        optimizer="AdamW",
        patience=5,
        workers=18,
        name=args.name,
        **hyp
    )

if __name__ == "__main__":
    main()
