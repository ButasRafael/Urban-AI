
from pathlib import Path

lbl_dir_train = Path("../datasets/urban_yolo_final_all/labels/train")
lbl_dir_valid = Path("../datasets/urban_yolo_final_all/labels/valid")
lbl_dir_test = Path("../datasets/urban_yolo_final_all/labels/test")

print("Cleaning txt files in train, valid and test directories...")

for txt_file in lbl_dir_train.glob("*.txt"):
    lines = txt_file.read_text().splitlines()
    cleaned_lines = [line for line in lines if len(line.strip().split()) == 5]
    txt_file.write_text("\n".join(cleaned_lines))



for txt_file in lbl_dir_valid.glob("*.txt"):
    lines = txt_file.read_text().splitlines()
    cleaned_lines = [line for line in lines if len(line.strip().split()) == 5]
    txt_file.write_text("\n".join(cleaned_lines))

for txt_file in lbl_dir_test.glob("*.txt"):
    lines = txt_file.read_text().splitlines()
    cleaned_lines = [line for line in lines if len(line.strip().split()) == 5]
    txt_file.write_text("\n".join(cleaned_lines))


print("Cleaned all txt files in train, valid and test directories.")