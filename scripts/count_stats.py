
import yaml
import os
import glob
import pandas as pd
from collections import Counter


def load_data_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def count_stats(label_dir):
    files = glob.glob(os.path.join(label_dir, '*.txt'))
    img_cnt = Counter()
    inst_cnt = Counter()
    for txt in files:
        with open(txt, 'r') as f:
            lines = f.read().strip().splitlines()
        if not lines:
            continue
        classes = []
        for line in lines:
            parts = line.split()
            try:
                cls_id = int(parts[0])
            except ValueError:
                cls_id = int(float(parts[0]))
            classes.append(cls_id)
        inst_cnt.update(classes)
        img_cnt.update(set(classes))
    rows = []
    for cls_id in sorted(inst_cnt):
        rows.append({
            'class_id':   cls_id,
            'images_cnt': img_cnt[cls_id],
            'instances':  inst_cnt[cls_id]
        })
    return pd.DataFrame(rows)


def main():
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../datasets/urban_yolo_final_all/urban_yolo_final_all.yaml')
    )

    if not os.path.isfile(config_path):
        print(f"❌ ERROR: data YAML not found: {config_path}")
        return

    data = load_data_yaml(config_path)
    base_dir = os.path.dirname(config_path)
    all_dfs = []

    for split in ('train', 'val'):
        raw_paths = data.get(split)
        if not raw_paths:
            continue

        img_dirs = raw_paths if isinstance(raw_paths, list) else [raw_paths]

        for rel_path in img_dirs:
            img_dir = rel_path if os.path.isabs(rel_path) else os.path.normpath(os.path.join(base_dir, rel_path))
            subset_name = os.path.basename(img_dir)
            if not os.path.isdir(img_dir):
                print(f"❌ Missing images folder for {split}/{subset_name}: {img_dir}")
                continue
            label_dir = img_dir.replace(os.path.sep + 'images', os.path.sep + 'labels')
            if not os.path.isdir(label_dir):
                print(f"❌ Missing labels folder for {split}/{subset_name}: {label_dir}")
                continue

            df = count_stats(label_dir)
            df['split'] = split
            df['subset'] = subset_name
            all_dfs.append(df)

    if not all_dfs:
        print("❌ No data processed. Check your YAML and folder structure.")
        return

    result = pd.concat(all_dfs, ignore_index=True)

    print("\n### Dataset class counts by split and subset\n")
    print(result.to_markdown(index=False))

    csv_out = "dataset_stats.csv"
    result.to_csv(csv_out, index=False)
    print(f"\n✅ Saved full breakdown to {csv_out}")


if __name__ == "__main__":
    main()
