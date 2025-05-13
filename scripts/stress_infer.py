import requests
from pathlib import Path
import time

API_BASE  = "http://localhost:8000"
IMAGE_DIR = Path(r"C:\Users\butas\OneDrive\Documents\AN3\BitStone\datasets\urban_yolo_final_all\images\test")
USE_SAM   = True

# 1) Make sure this user is registered already:
#    POST /auth/register  { "username": "Rafa", "password": "Rafa1pass", "role": "user" }

# 2) Log in to get a token
login = requests.post(
    f"{API_BASE}/auth/login",
    data={"username": "Rafa", "password": "Rafapass1"},
)
login.raise_for_status()
token = login.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 3) Iterate over the images
for img_path in IMAGE_DIR.iterdir():
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    start = time.time()
    with img_path.open("rb") as f:
        files  = {"file": (img_path.name, f, "image/jpeg")}
        params = {"use_sam": USE_SAM}
        resp   = requests.post(
            f"{API_BASE}/infer/image",
            files=files,
            params=params,
            headers=headers,
        )
    elapsed = time.time() - start

    if resp.ok:
        out = resp.json()
        print(f"{img_path.name} → OK, saved to {out['annotated_image_url']}  ({elapsed:.2f}s)")
    else:
        print(f"{img_path.name} → ERROR {resp.status_code}: {resp.text}  ({elapsed:.2f}s)")
