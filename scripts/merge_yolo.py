
import json
import random
import shutil
import uuid
from pathlib import Path

RAW_ROOT = Path("../datasets/raw")
OUT_ROOT = Path("../datasets/urban_yolo_final_all")
SPLITS = ("valid", "test", "train")

CLASSES = [
    "pothole",          # 0
    "graffiti",         # 1
    "garbage",          # 2
    "garbage_bin",      # 3
    "overflow",         # 4
    "parking_illegal",  # 5
    "parking_empty",    # 6
    "parking_legal",    # 7
    "crack",            # 8
    "open_manhole",     # 9
    "closed_manhole",   # 10
    "fallen_tree",      # 11
]

DATASETS = [
    # potholes
    "pothole1","pothole2","pothole3","pothole4","pothole5",
    # graffiti
    "graffiti1","graffiti2","graffiti3","graffiti4",
    # trash / bins
    "trash1","trash2","trash3","trash4","trash5","trash6","trash7",
    # parking
    "parking1","parking2","parking3","parking4","parking5",
    "parking6","parking7","parking8","parking9","parking10",
    # cracks
    "crack1","crack2","crack3","crack4","crack5","crack6",
    # manholes
    "manhole1","manhole2","manhole3","manhole4","manhole5","manhole6",
    # fallen trees
    "trees1","trees2","trees3","trees4","trees5",
]

IDMAP = {

    **{f"pothole{i}": {0: 0} for i in range(1, 6)},

    **{f"graffiti{i}": {0: 1} for i in range(1, 5)},

    "trash1": {0:3, 1:3, 2:3, 3:3, 4:4},
    "trash2": {0:3, 1:4},
    "trash3": {0:4, 1:3},
    "trash4": {0:2, 1:3, 2:4},
    "trash5": {0:2, 1:3, 2:4},
    "trash6": {0:2, 1:3, 2:4, 3:4},
    "trash7": {0:2, 1:3, 2:4},

    "parking1":  {0:5, 1:6, 2:7},
    "parking2":  {0:5},
    "parking3":  {0:5, 1:6, 2:7},
    "parking4":  {0:5, 1:7},
    "parking5":  {0:6, 2:5, 3:7},
    "parking6":  {0:5},
    "parking7":  {0:5, 1:7},
    "parking8":  {0:6, 1:5, 3:7},
    "parking9":  {0:6, 1:5, 3:7},
    "parking10": {0:6, 2:5, 3:7},

    **{f"crack{i}": {0: 8} for i in range(1, 7)},

    "manhole1": {0:10, 1:9},
    "manhole2": {0:10, 1:10, 2:9},
    "manhole3": {3:10, 4:10, 5:9},
    "manhole4": {0:10, 1:9, 2:0},
    "manhole5": {0:10, 1:9, 2:0},
    "manhole6": {0:10, 1:9, 2:0},

    **{f"trees{i}": {0: 11} for i in range(1, 6)},
}

IMG_EXT = {".jpg", ".jpeg", ".png"}
_seen_raw_paths: set[Path] = set()


def make_dirs() -> None:
    for split in SPLITS:
        (OUT_ROOT / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / f"labels/{split}").mkdir(parents=True, exist_ok=True)


def unique_name(img_path: Path) -> str:
    return f"{uuid.uuid4().hex[:8]}{img_path.suffix}"


def copy_and_remap(img_path: Path, lab_dir: Path, split: str, ds_name: str) -> None:
    if img_path in _seen_raw_paths and split == "train":
        return
    if split in ("valid", "test"):
        _seen_raw_paths.add(img_path)

    label_path = lab_dir / img_path.with_suffix(".txt").name
    if not label_path.exists():
        return

    new_fname = unique_name(img_path)
    shutil.copy2(img_path, OUT_ROOT / f"images/{split}" / new_fname)

    remapped: list[str] = []
    for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = ln.split()
        if not parts:
            continue
        src = parts[0]
        try:
            src = int(src)
        except ValueError:
            pass                                       # keep string
        tgt = IDMAP.get(ds_name, {}).get(src)
        if tgt is not None:
            remapped.append(" ".join([str(tgt), *parts[1:]]))

    (OUT_ROOT / f"labels/{split}" / Path(new_fname).with_suffix(".txt")
     ).write_text("\n".join(remapped))


def process_dataset(ds_name: str, split: str) -> None:
    img_dir = RAW_ROOT / ds_name / split / "images"
    lab_dir = RAW_ROOT / ds_name / split / "labels"

    # fallback for missing *train* split – borrow from valid / test
    if split == "train" and not img_dir.exists():
        for alt in ("valid", "test"):
            alt_img = RAW_ROOT / ds_name / alt / "images"
            if alt_img.exists():
                img_dir = alt_img
                lab_dir = RAW_ROOT / ds_name / alt / "labels"
                break

    # if valid / test split missing → sample 10 % from train
    if split in ("valid", "test") and not img_dir.exists():
        train_img = RAW_ROOT / ds_name / "train" / "images"
        train_lab = RAW_ROOT / ds_name / "train" / "labels"
        if not train_img.exists():
            return
        imgs = [p for p in train_img.iterdir() if p.suffix.lower() in IMG_EXT]
        for img_path in random.sample(imgs, max(1, len(imgs) // 10)):
            copy_and_remap(img_path, train_lab, split, ds_name)
            _seen_raw_paths.add(img_path)
        return

    if not img_dir.exists():
        return

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() in IMG_EXT:
            copy_and_remap(img_path, lab_dir, split, ds_name)


def write_yaml() -> None:
    cfg = {
        "path":  str(OUT_ROOT),
        "train": "images/train",
        "val":   "images/valid",
        "test":  "images/test",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    (OUT_ROOT / "urban_yolo_final_all.yaml").write_text(json.dumps(cfg, indent=2))
    print("✅ Wrote", OUT_ROOT / "urban_yolo_final_all.yaml")


def main() -> None:
    make_dirs()
    for ds in DATASETS:
        for split in SPLITS:
            print(f"Merging {ds}/{split} …")
            process_dataset(ds, split)
    write_yaml()
    print("✅ Merge complete →", OUT_ROOT)


if __name__ == "__main__":
    main()
