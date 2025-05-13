
import argparse, multiprocessing as mp, yaml
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--model",  type=str, default="runs/detect/yolo11s_urban_final_no_tuning2/weights/last.pt") # for resuming training
    p.add_argument("--model",  type=str,
                   default="yolo11s.pt") # for training from scratch
    p.add_argument("--data",   type=str,
                   default="../datasets/urban_yolo_final_all/urban_yolo_final_all.yaml")
    p.add_argument("--hyp",    type=str,
                   default="../datasets/urban_yolo_final_all/best_hyperparameters.yaml")
    p.add_argument("--epochs", type=int, default=80) #125 for medium model, 150 for large model
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--patience", type=int, default=17) # 25 for medium model, 30 for large model
    p.add_argument("--name",   type=str,
                   default="yolo11s_urban_final_no_tuning")
    return p.parse_args()

def main():
    args = parse_args()

    hyp_file = Path(args.hyp)
    if not hyp_file.is_file():
        raise FileNotFoundError(f"⚠️  {hyp_file} not found. "
                                "Make sure tuning finished successfully.")
    best_hyp = yaml.safe_load(hyp_file.read_text())

    # -------- start training ----------
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=16, # 12 for medium model
        #multi_scale=True, if enough GPU memory
        # resume=True, # resume training from last.pt
        cos_lr=True,
        amp=True,
        #cache='ram' or 'disk', if enough RAM or disk space
        close_mosaic=17, # 25 for medium model, 30 for large model
        optimizer="AdamW",
        patience=args.patience,
        device=0,
        workers=18, # more if you have more CPU cores
        name=args.name,
        **best_hyp
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()
