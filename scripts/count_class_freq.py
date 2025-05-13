
from pathlib import Path
from collections import Counter
import argparse
import pandas as pd

CLASSES = [
    "pothole",         # 0
    "graffiti",        # 1
    "garbage",         # 2
    "garbage_bin",     # 3
    "overflow",        # 4
    "parking_illegal", # 5
    "parking_empty",   # 6
    "parking_legal",   # 7
    "crack",            # 8
    "open_manhole",     # 9
    "closed_manhole",   # 10
    "fallen_tree",      # 11
]


def count_labels(label_dir: Path) -> Counter:
    """Return Counter{class_id: count} for every .txt file in label_dir."""
    counter = Counter()
    for txt in label_dir.glob("*.txt"):
        for ln in txt.read_text().splitlines():
            if ln.strip():
                cls = int(float(ln.split()[0])) 
                counter[cls] += 1
    return counter

def main(root="../datasets/urban_yolo_final_all/labels"):
    root = Path(root)
    splits = ["train", "train_aug"]
    rows = []

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"✖️ {split_dir} not found, skipping")
            continue
        cnt = count_labels(split_dir)
        rows.append([cnt.get(i, 0) for i in range(len(CLASSES))])

    df = pd.DataFrame(rows, index=splits[:len(rows)], columns=CLASSES)
    print("\nObject counts per class:")
    print(df.to_string())
    df.to_csv("class_frequencies.csv")
    print("\n✅  Saved as class_frequencies.csv")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1]) 
    else:
        main()
