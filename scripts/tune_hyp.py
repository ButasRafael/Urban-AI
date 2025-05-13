
import os, datetime, pathlib, yaml
from ultralytics import YOLO
from ray import tune
from pathlib import Path

try:
    import wandb
    wandb.init(project="urban-yolo11s-tuning", entity="butas-rafael-technical-university-of-cluj-napoca")
except Exception as e:
    print(f"⚠️  W&B disabled ({e}); continuing offline.")
    wandb = None


model = YOLO("yolo11s.pt")

ROOT      = Path(__file__).parent.parent
DATA_YAML = ROOT / "datasets/urban_yolo_final/urban_yolo_final.yaml"

space = {

    "optimizer": tune.choice(["SGD", "AdamW"]),

    "lr0": tune.loguniform(5e-5, 5e-2), 
    "lrf": tune.uniform(0.01, 0.5),
    "momentum": tune.uniform(0.85, 0.98),
    "weight_decay": tune.loguniform(1e-6, 5e-4),


    "warmup_epochs":   tune.uniform(1.0, 5.0),
    "warmup_momentum": tune.uniform(0.7, 0.95),
    "warmup_bias_lr":  tune.loguniform(1e-4, 0.2),

    "box": tune.uniform(0.02, 0.2),
    "cls": tune.uniform(0.2, 3.0),
    "dfl": tune.uniform(0.5, 2.0),


    "hsv_h": tune.uniform(0.0, 0.1),
    "hsv_s": tune.uniform(0.3, 0.8),
    "hsv_v": tune.uniform(0.2, 0.7),

    "degrees":     tune.uniform(-10.0, 10.0),
    "translate":   tune.uniform(0.0, 0.3),
    "scale":       tune.uniform(0.4, 0.9),
    "shear":       tune.uniform(-5.0, 5.0),
    "perspective": tune.uniform(0.0, 0.001),


    "fliplr":   tune.uniform(0.3, 0.8),
    "mosaic":   tune.uniform(0.0, 1.0),
    "mixup":    tune.uniform(0.0, 0.5),
    "copy_paste": tune.uniform(0.0, 0.5),
}


train_args = dict(
    epochs=40,
    batch=16,
    #batch=-1,
    imgsz=640,
    #multi_scale=True,
    fraction=0.5,
    #cache='ram',
    cos_lr=True,
    amp=True,
    workers=18,
    close_mosaic=10,
    save=False,
)

result_grid = model.tune(
    data=str(DATA_YAML),
    use_ray=True,
    space=space,
    iterations=300,
    grace_period=5,
    gpu_per_trial=1, 
    name=f"tune-{datetime.datetime.now():%Y-%m-%d-%H-%M}",
    **train_args
)

best_cfg = result_grid.get_best_result().config
print("\n✅ Best trial hyper-parameters:\n", best_cfg)

dest = pathlib.Path("runs/detect/tune/best_hyperparameters.yaml")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(yaml.dump(best_cfg))
print("Saved to", dest)
