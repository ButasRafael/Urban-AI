from __future__ import annotations
import cv2
import numpy as np
import torch
import tempfile
import uuid
from ultralytics import YOLO,SAM

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any
from pycocotools import mask as mask_util
import yaml
import random
import logging
from typing import Tuple, List
import os, json, base64
from openai import AsyncOpenAI, InternalServerError, RateLimitError, APIConnectionError
from dotenv import load_dotenv
from fastapi import HTTPException
from groundingdino.util.inference import Model as _GDINO
import time
import asyncio
import httpx

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)

GDINO_WEIGHTS = "/weights/groundingdino_swinb_cogcoor.pth"
YOLO_WEIGHTS = "/weights/best_medium.engine"
SAM_WEIGHTS  = "/weights/sam2.1_hiera_base_plus.pt"
SAM_CFG      = "configs/sam2.1/sam2.1_hiera_b+.yaml"
GDINO_CFG   = "/weights/configs/GroundingDINO_SwinB_cfg.py"
IMG_SZ = 640
CONF_T = 0.2
IOU_T  = 0.45
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True, parents=True)

PERF_CFG = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.7,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
    "gmc_method": None,
    "with_reid": False,
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.3,
    "model": "auto",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_GPT_SEMAPHORE = asyncio.Semaphore(2)

async def _wait_for_file(path: Path, timeout: float = 5.0, poll: float = 0.05):

    elapsed = 0.0
    while elapsed < timeout:
        try:
            if path.stat().st_size > 0:
                return
        except FileNotFoundError:
            pass
        await asyncio.sleep(poll)
        elapsed += poll
    raise HTTPException(500, f"Image file {path!r} not ready after {timeout}s")


@lru_cache
def _load_grounder():
    return _GDINO(
        model_config_path=GDINO_CFG,
        model_checkpoint_path=GDINO_WEIGHTS,
        device=DEVICE,
    )

@lru_cache
def _load_models():
    logger.info("Loading YOLO & SAM models", extra={"weights": YOLO_WEIGHTS, "device": DEVICE})
    yolo = YOLO(YOLO_WEIGHTS, task="detect")

    sam_model  = build_sam2(SAM_CFG, ckpt_path=SAM_WEIGHTS, device=DEVICE)
    predictor  = SAM2ImagePredictor(sam_model)
    mask_gen   = SAM2AutomaticMaskGenerator(sam_model)
    logger.info("Models loaded", extra={"yolo": str(yolo), "sam_cfg": SAM_CFG})
    return yolo, predictor, mask_gen


def _encode(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _poly(mask: np.ndarray) -> List[List[float]]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        pts = c.squeeze().tolist()
        if len(pts) >= 3:
            out.append([float(v) for p in pts for v in p])
    return out


def _best(masks, scores, box):
    x1, y1, x2, y2 = map(int, box)
    rect = np.zeros(masks[0].shape, np.uint8)
    cv2.rectangle(rect, (x1, y1), (x2, y2), 1, -1)
    return masks[max(range(len(masks)),
                     key=lambda i: (np.logical_and(rect, masks[i]).sum(), scores[i]))]
_label_rects: list[tuple[int,int,int,int]] = []

def clear_label_rects():
    global _label_rects
    _label_rects = []

def rects_overlap(r1, r2):
    x11,y11,x12,y12 = r1
    x21,y21,x22,y22 = r2
    return not (x12 < x21 or x22 < x11 or y12 < y21 or y22 < y11)

def draw_label(
    img: np.ndarray,
    text: str,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    pad: int = 2,
) -> None:
    global _label_rects

    x1, y1, x2, y2 = box
    h_img, w_img = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = float(np.clip((y2-y1) / 200.0, 0.8, 2.0))
    thickness  = int(np.clip((y2-y1) // 120, 1, 5))

    (tw, th), _ = cv2.getTextSize(text, font, base_scale, thickness)
    avail_w = min(w_img, x2 - x1) - 2*pad
    if tw + 2*pad > avail_w and tw > 0:
        scale = base_scale * (avail_w / (tw + 2*pad))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    else:
        scale = base_scale

    x0 = int(np.clip(x1, 0, w_img - tw - 2*pad))
    y0 = y1 - th - 2*pad
    if y0 < 0:
        y0 = y2 + 2

    rect = (x0, y0, x0 + tw + 2*pad, y0 + th + 2*pad)

    for prev in _label_rects:
        if rects_overlap(rect, prev):
            y0 = prev[3] + pad
            y0 = int(min(y0, h_img - th - 2*pad))
            rect = (x0, y0, x0 + tw + 2*pad, y0 + th + 2*pad)

    _label_rects.append(rect)

    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness=-1)
    cv2.putText(
        img,
        text,
        (x0 + pad, y0 + th + pad - 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w  = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h  = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter    = inter_w * inter_h
    area_a   = (xa2 - xa1) * (ya2 - ya1)
    area_b   = (xb2 - xb1) * (yb2 - yb1)
    union    = area_a + area_b - inter + 1e-6
    return inter / union


def _ground_phrase_weights(
    img_bgr: np.ndarray,
    phrase: str,
    existing: List[List[float]] = [],
    box_thresh: float = 0.35,
    text_thresh: float = 0.25,
    iou_filter: float = 0.45,
):

    grounder = _load_grounder()
    detections, _ = grounder.predict_with_caption( 
        image=img_bgr[:, :, ::-1],         
        caption=phrase,
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )

    if detections.xyxy.shape[0] == 0:   
        return None

    keep = []
    for idx, box in enumerate(detections.xyxy):
        if all(iou_xyxy(box, ex) < iou_filter for ex in existing):
            keep.append(idx)

    if not keep:                          
        return None

    idx = int(max(keep, key=lambda i: detections.confidence[i]))
    x1, y1, x2, y2 = detections.xyxy[idx]
    score          = float(detections.confidence[idx])
    return int(x1), int(y1), int(x2), int(y2), score


def _run_yolo(img: np.ndarray, run_id: str | None = None,) -> List[Dict[str, Any]]:

    yolo, _, _ = _load_models()

    results = yolo.predict(
        img,
        imgsz=IMG_SZ,
        conf=CONF_T,
        iou=IOU_T,
        device=DEVICE,
        augment=True,
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        project=str(STATIC_DIR),
        name=run_id,
        exist_ok=True,
    )[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss  = results.boxes.cls.int().cpu().numpy()

    output: List[Dict[str, Any]] = []
    for idx, (box, conf, cls_idx) in enumerate(zip(boxes, confs, clss)):
        x1, y1, x2, y2 = box.tolist()
        name = yolo.names[int(cls_idx)]
        output.append({
            "track_id":   idx,
            "class_id":   int(cls_idx),
            "class_name": f"{name}-yolo",
            "confidence": float(conf),
            "bbox":       [x1, y1, x2, y2],
            "source": "yolo",
        })

    return output


async def generate_comprehensive_summary(
    detections: List[Dict[str, Any]],
) -> Dict[str, str]:

    if not detections:
        return {
            "description": "No infrastructure issues were detected in this image.",
            "solution": "No remediation actions are required."
        }

    issues_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        class_name = det.get("class_name") or "unknown"
        issues_by_type.setdefault(class_name, []).append({
            "description": (det.get("description") or "").strip(),
            "solution": (det.get("solution") or "").strip(),
            "severity": (det.get("severity") or "medium").lower()
        })

    def sev_bucket(items: List[Dict[str, Any]]) -> Dict[str, int]:
        out = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for it in items:
            sev = it.get("severity") or "unknown"
            if sev not in out:
                sev = "unknown"
            out[sev] += 1
        return out

    lines = []
    for issue_type, items in sorted(issues_by_type.items(), key=lambda kv: kv[0]):
        sev = sev_bucket(items)
        total = len(items)
        examples = [i["description"] for i in items if i.get("description")]
        examples = [e for e in examples if e][:2]
        ex_text = ""
        if examples:
            ex_text = "\n    examples: " + " | ".join(f"\"{e}\"" for e in examples)
        lines.append(
            f"- {issue_type}: {total} detections | severity -> "
            f"high:{sev['high']}, medium:{sev['medium']}, low:{sev['low']}, unknown:{sev['unknown']}{ex_text}"
        )

    issues_text = "\n".join(lines)

    prompt = f"""# Role & Objective
You are an urban infrastructure expert. Summarize the issues in a single image and propose an integrated remediation plan.

# Response Rules (follow strictly)
- Output **only** a single JSON object, no prose, no markdown, no code fences.
- The JSON must contain **exactly** these keys: "description" and "solution".
- "description": 2–3 concise sentences covering **all** issues found as a cohesive overview.
- "solution": 3–4 concise sentences with an integrated, **prioritized** action plan that:
  • sequences work by severity and logical dependencies,
  • mentions coordination/safety or access constraints if relevant,
  • avoids redundant steps, and
  • is directly actionable for city operations teams.
- Do **not** invent assets, measurements, or locations not implied by the data.
- Group similar problems; avoid repetitive listing of identical issues.
- Think through prioritization and dependencies **privately**; do **not** reveal your reasoning steps. Only return the final JSON.

# Output Format (must match exactly)
{{ "description": "<2-3 sentences>", "solution": "<3-4 sentences>" }}

# Data: Issue Summary
{issues_text}
"""

    async with _GPT_SEMAPHORE:
        try:
            payload = [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": (
                        "You are an urban infrastructure expert who writes concise, "
                        "operationally useful summaries and action plans. Follow the user's rules exactly. "
                        "Do not include reasoning or meta-commentary; output only the final JSON object."
                    )}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ]

            resp = await client.responses.create(
                model="gpt-4.1",
                input=payload,
                temperature=0.2,
                timeout=30,
            )

            content = getattr(resp, "output_text", "") or ""

            try:
                result = json.loads(content)
            except Exception:
                m = re.search(r"\{.*\}", content, flags=re.DOTALL)
                if not m:
                    raise
                result = json.loads(m.group(0))

            return {
                "description": result.get(
                    "description",
                    "Multiple infrastructure issues were identified in the image."
                ),
                "solution": result.get(
                    "solution",
                    "Coordinate a prioritized remediation plan addressing the most severe hazards first, sequencing dependent repairs efficiently."
                ),
            }

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")

            types_sorted = sorted(issues_by_type.keys())
            top_types = ", ".join(types_sorted[:3]) if types_sorted else "infrastructure issues"
            return {
                "description": f"Issues detected include {top_types}. The image contains multiple occurrences with varying severities.",
                "solution": (
                    "Address high-severity hazards first for public safety, then medium and low. "
                    "Batch similar repairs together (e.g., same crew/equipment) to reduce rework. "
                    "Sequence tasks to respect dependencies (e.g., subsurface or structural fixes before surface restoration). "
                    "Coordinate traffic management and site access to minimize disruption."
                ),
            }


async def _gpt_refine_and_find(
    initial: List[Dict[str, Any]],
    run_id: str,
) -> List[Dict[str, Any]]:
    img_path = STATIC_DIR / run_id / "image0.jpg"
    await _wait_for_file(img_path)

    image_url = f"https://api-tunnel.taileffb4e.ts.net/static/{run_id}/image0.jpg"
    async with httpx.AsyncClient() as http:
        resp = await http.get(image_url, timeout=5.0)
        resp.raise_for_status()

    prompt_text = (
        "You are analyzing an urban-planning scenario based on an input image and its corresponding YOLO detections (bounding boxes)."
        " Perform the following tasks:\n"
        "\n"
        "1. For each YOLO detection **present in the array** (if the YOLO array is empty, skip this section entirely and output **no fields named \"track_id\" or \"keep\"):**\n"
        "   - \"track_id\": integer (the YOLO track-ID provided)\n"
        "   - \"keep\": boolean (true if YOLO correctly identified the object; false if it is a mis-identification)\n"
        "   - \"description\": string (a clear one-sentence description of the detected object or issue)\n"
        "   - \"solution\": string — if the object is an issue, give a concise, practical, one-sentence remediation proposal; "
        "     if the object is NOT an issue, write \"No solution needed\".\n"
        "   - \"severity\": \"low\" | \"medium\" | \"high\" (lowercase only). Use this rubric: high = acute safety/operational hazard or large impact; medium = material degradation or likely to become hazardous soon; low = minor/cosmetic. If uncertain, choose the lower severity.\n"
        "\n"
        "   Do NOT include bounding-box coordinates in your response; YOLO’s geometry will be used directly.\n"
        "\n"
        "2. Only add **new** JSON objects for additional urban issues **when they are BOTH important and realistically solvable**:\n"
        "   - \"new\": true\n"
        "   - \"class_name\": string (the specific type or category of urban issue identified)\n"
        "   - \"confidence\": float between 0 and 1 (your confidence in this additional issue)\n"
        "   - \"description\": string (a clear one-sentence description of the newly identified issue)\n"
        "   - \"solution\": string (a concise, practical, one-sentence remediation proposal)\n"
        "   - \"severity\": \"low\" | \"medium\" | \"high\" (lowercase only; same rubric as above).\n"
        "   - \"dino_prompt\": string\n"
        "       • 1 – 3 **lower-case tokens**, each ≤ 2 words.\n"
        "       • Choose COCO/LVIS/VisualGenome-style nouns whenever possible (e.g. \"traffic light\", \"trash bin\").\n"
        "       • NO commas, punctuation, verbs, prepositions, or numerals.\n"
        "       • Color/material adjectives when they are the *sole* reliable cue (e.g. \"red cone\").\n"
        "       • Put the **most visually distinctive token first**; order the rest by distinctiveness.\n"
        "       • Prefer the canonical dataset label (\"fire hydrant\" not \"hydrant\").\n"
        "       • Use singular form unless plurality is visually obvious.\n"
        "       • Hard cap of three tokens — if unsure, pick ONE high-precision noun.\n"
        "       • Separate tokens with ONE space exactly.\n"
        "\n"
        "   Skip minor, cosmetic, or trivial issues entirely — return ZERO new detections if the photo appears clean.\n"
        "   Do NOT specify exact bounding-box coordinates; these additional detections will be logged as coarse issues.\n"
        "\n"
        "3. Return **ONLY** a JSON-formatted array containing all of the above-described objects, and NOTHING ELSE."
    )

    initial = sorted(initial, key=lambda d: -d["confidence"])[:50]

    payload = [{
        "role": "user",
        "content": [
            {"type": "input_text",  "text": prompt_text},
            {"type": "input_image", "image_url": image_url, "detail": "high"},
            {"type": "input_text",  "text": json.dumps(initial)},
        ],
    }]

    async with _GPT_SEMAPHORE:
        max_retries = 5
        backoff = 0.5

        for attempt in range(max_retries):
            try:
                resp = await client.responses.create(
                    model="gpt-4.1",
                    input=payload,
                    temperature=0.2,
                    timeout=35,
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                await asyncio.sleep(backoff + random.random() * 0.2)
                backoff *= 2
            except InternalServerError as e:
                status = e.response.status_code if e.response else 502
                body   = e.response.text if e.response else "<no body>"
                logger.error("OpenAI 5xx: %s – %s", status, body)
                raise HTTPException(
                    502,
                    f"Upstream service error (OpenAI {status}): {body}"
                )
        else:
            raise HTTPException(502, "Upstream service error (OpenAI). Please try again.")

    raw = resp.output_text or ""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        logger.error("No JSON array in GPT response: %r", raw)
        raise HTTPException(500, f"No JSON array in LLM response: {raw!r}")
    content = raw[start:end+1]
    try:
        return json.loads(content)
    except json.JSONDecodeError as err:
        logger.error("JSON parse failed: %s\nContent was: %r", err, content)
        raise HTTPException(500, f"Invalid JSON from LLM: {content!r}")

DDS_TOKEN = os.getenv("DDS_TOKEN")
DDS_API   = os.getenv("DDS_API", "https://api.deepdataspace.com")

async def _grounding_dino_1p6(
    *,                      
    prompt: str,
    img_bgr:   np.ndarray | None = None,
    image_url: str | None     = None,
    bbox_threshold: float     = 0.25,
    iou_threshold : float     = 0.8,
    poll_every    : float     = 0.35,
    timeout       : float     = 30.0,
) -> list[dict]:        
    if (img_bgr is None) == (image_url is None):
        raise ValueError("Provide either img_bgr or image_url, not both/none.")
    if not DDS_TOKEN:
        raise RuntimeError("Set DDS_TOKEN in your environment!")

    if image_url:
        image_field = image_url                               
    else:  
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _, buf   = cv2.imencode(".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64      = base64.b64encode(buf).decode()
        image_field = f"data:image/jpeg;base64,{b64}"

    payload = {
        "model":   "GroundingDino-1.6-Pro",
        "image":   image_field,
        "prompt":  {"type": "text", "text": prompt},
        "targets": ["bbox"],
        "bbox_threshold": bbox_threshold,
        "iou_threshold":  iou_threshold,
    }
    headers = {"Token": DDS_TOKEN, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=timeout) as http:
        task = await http.post(f"{DDS_API}/v2/task/grounding_dino/detection",
                               json=payload, headers=headers)
        task.raise_for_status()
        task_uuid = task.json()["data"]["task_uuid"]

        start = time.time()
        while True:
            r = await http.get(f"{DDS_API}/v2/task_status/{task_uuid}",
                               headers=headers)
            r.raise_for_status()
            data = r.json()["data"]
            if data["status"] == "success":
                return data["result"]["objects"]
            if data["status"] == "failed":
                raise RuntimeError(f"DINO task failed: {data['error']}")
            if time.time() - start > timeout:
                raise TimeoutError("GroundingDINO request timed out")
            await asyncio.sleep(poll_every)

async def _ground_phrase(
    img_bgr: np.ndarray,
    classes: list[str],
    *,                    
    backend: str = "1.6pro",
    bbox_threshold: float = 0.25,
    iou_threshold : float = 0.8,
    existing: list[list[float]] | None = None,
) -> dict[str, list[dict]]:

    if backend.lower() in {"1.6pro", "pro", "remote"}:
        prompt_text = ".".join(classes)
        objs = await _grounding_dino_1p6(
            img_bgr      = img_bgr,
            prompt       = prompt_text,
            bbox_threshold = bbox_threshold,
            iou_threshold  = iou_threshold,
        )

        by_cat: dict[str, list[dict]] = {c: [] for c in classes}
        for o in objs:
            cat = o["category"]
            if cat in by_cat:
                by_cat[cat].append(o)

        for c, hits in by_cat.items():
            by_cat[c] = [max(hits, key=lambda h: h["score"])] if hits else []
        return by_cat

    elif backend.lower() in {"swinb", "local"}:
        by_cat: dict[str, list[dict]] = {}
        existing: list[list[float]] = existing or []
        for phrase in classes:
            hit = _ground_phrase_weights(
                img_bgr,
                phrase,
                existing      = existing,
                box_thresh    = bbox_threshold,
                text_thresh   = 0.25,
                iou_filter    = iou_threshold,
            )
            if hit:
                x1, y1, x2, y2, score = hit
                by_cat[phrase] = [{
                    "bbox":     [x1, y1, x2, y2],
                    "score":    score,
                    "category": phrase,
                }]
                existing.append([x1, y1, x2, y2])
            else:
                by_cat[phrase] = []
        return by_cat

    else:
        raise ValueError(f"Unknown backend {backend!r}; use '1.6pro' or 'swinb'.")

def overlay_masks(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    label: str,
    alpha: float = 0.5,
) -> None:
    global _label_rects

    # 1) draw the softened mask & glow as before
    mask_uint = (mask.astype(np.uint8) * 255)
    blurred   = cv2.GaussianBlur(mask_uint, (21, 21), 0)
    soft_mask = blurred.astype(bool)

    overlay = image.copy()
    overlay[soft_mask] = color
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

    glow = image.copy()
    contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(glow, contours, -1, color, thickness=15)
    cv2.addWeighted(glow, alpha * 0.3, image, 1 - alpha * 0.3, 0, dst=image)
    cv2.drawContours(image, contours, -1, color, thickness=3)

    # 2) now label
    ys, xs = np.where(mask)
    if not (xs.size and ys.size):
        return

    h_img, w_img = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX


    raw_scale = min(w_img, h_img) * 0.003
    raw_th    = int(min(w_img, h_img) * 0.004)
    base_scale = float(np.clip(raw_scale, 0.8, 2.0))
    thickness = int(np.clip(raw_th, 2, 6))

    (w_txt, h_txt), _ = cv2.getTextSize(label, font, base_scale, thickness)
    pad = int(thickness * 2)
    max_w = w_img - 2*pad
    if w_txt + 2*pad > max_w:
        scale = base_scale * (max_w / (w_txt + 2*pad))
        (w_txt, h_txt), _ = cv2.getTextSize(label, font, scale, thickness)
    else:
        scale = base_scale

    x_c, y_c = int(xs.mean()), int(ys.mean())
    x0 = x_c - w_txt//2 - pad
    y0 = y_c - h_txt - pad - 5
    x0 = int(np.clip(x0, 0, w_img - w_txt - 2*pad))
    if y0 < 0:
        y0 = int(np.clip(y_c + 5, 0, h_img - h_txt - 2*pad))

    rect = (x0, y0, x0 + w_txt + 2*pad, y0 + h_txt + 2*pad)

    for prev in _label_rects:
        if not (rect[2] < prev[0] or prev[2] < rect[0] or rect[3] < prev[1] or prev[3] < rect[1]):
            y0 = prev[3] + pad
            y0 = int(min(y0, h_img - h_txt - 2*pad))
            rect = (x0, y0, x0 + w_txt + 2*pad, y0 + h_txt + 2*pad)

    _label_rects.append(rect)

    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness=-1)
    cv2.putText(
        image,
        label,
        (x0 + pad, y0 + h_txt + pad - 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def process_image(
    img: np.ndarray,
    use_sam: bool = True,
    run_id: str | None = None,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    logger.debug("process_image() start", extra={"img_shape": img.shape, "use_sam": use_sam})
    yolo, predictor, mask_gen = _load_models()

    if run_id is None:
        run_id = uuid.uuid4().hex

    # 1) YOLO detection
    res = yolo.predict(
        img, imgsz=IMG_SZ, conf=CONF_T, iou=IOU_T,
        augment=True, device=DEVICE, save=True,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        project=str(STATIC_DIR),
        name=run_id,
        exist_ok=True,
    )[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss  = res.boxes.cls.int().cpu().numpy()
    logger.info(
        "YOLO detections",
        extra={"num_boxes": len(boxes), "conf_threshold": CONF_T, "iou_threshold": IOU_T}
    )

    unique_cids = set(clss.tolist())
    colors = {cid: tuple(random.randint(0,255) for _ in range(3)) for cid in unique_cids}

    if use_sam:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

    det_out: List[Dict[str, Any]] = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]
    scale = max(0.6, min(w, h) * 0.002)
    thickness = max(2, int(min(w, h) * 0.004))
    
    for box, conf, cid in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        color = colors[cid]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if not use_sam:
            label = yolo.names[cid]
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            pad = int(thickness * 2)
            cv2.rectangle(img, (x1, y1-th-pad), (x1+tw+pad, y1), color, -1)
            cv2.putText(img, label, (x1+pad//2, y1-pad//2), font, scale, (255,255,255), thickness,
                        lineType=cv2.LINE_AA)

        entry: Dict[str, Any] = {
            "track_id": None,
            "class_id": int(cid),
            "class_name": yolo.names[int(cid)],
            "confidence": float(conf),
            "bbox": [float(v) for v in box],
        }

        if use_sam:
            masks, scores, _ = predictor.predict(
                box=box[None, :], multimask_output=True
            )
            mask = _best(masks, scores, box)
            overlay_masks(img, mask, color, yolo.names[int(cid)], alpha=0.5)
            entry["mask"] = {
                "rle": _encode(mask),
                "polygon": _poly(mask),
            }
        else:
             entry["mask"] = {
                "rle": {},
                "polygon": [],
            }

        det_out.append(entry)

    if use_sam and not det_out:
        logger.warning("No YOLO boxes: falling back to full-image SAM masks")
        for mdata in mask_gen.generate(rgb):
            seg = mdata["segmentation"]
            color = tuple(random.randint(0,255) for _ in range(3))
            cv2.rectangle(img, (0,0), (0,0), color, 0)  # no box to draw
            overlay_masks(img, seg, color, "clean", alpha=0.5)
            det_out.append({
                "track_id": None,
                "class_id": -1,
                "class_name": "clean",
                "confidence": float(mdata.get("score", 0)),
                "bbox": [float(x) for x in mdata["bbox"]],
                "mask": {"rle": _encode(seg), "polygon": _poly(seg)},
            })

    logger.info("process_image() complete", extra={"detections": len(det_out)})
    return img, det_out

async def process_image_combined(img_bgr, use_sam=True, run_id=None):
    # 1) YOLO gives us concrete boxes
    clear_label_rects()
    initial = _run_yolo(img_bgr, run_id)
    unique_cids = {d["class_id"] for d in initial}
    colors = {cid: tuple(random.randint(0,255) for _ in range(3)) for cid in unique_cids}

    # 2) GPT tells us KEEP/REMOVE + description/solution (+ any “new” issues)
    refinements = await _gpt_refine_and_find(initial, run_id)

    annotated = img_bgr.copy()
    final     = []

    if use_sam:
        _, predictor, mask_gen = _load_models()
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
    else:
        _, _, mask_gen = _load_models()

    h, w = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, min(w, h) * 0.002)
    thickness = max(2, int(min(w, h) * 0.004))

    # 3) Apply all kept YOLO detections
    for det in initial:
        info = next((r for r in refinements if r.get("track_id") == det["track_id"]), None)
        if not info or not info.get("keep", False):
            continue

        x1, y1, x2, y2 = map(int, det["bbox"])
        color = colors.get(det["class_id"], (0,255,0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        if use_sam:
            masks, scores, _ = predictor.predict(
                box=np.array([[x1, y1, x2, y2]]), multimask_output=True
            )
            mask = masks[np.argmax(scores)]
            overlay_masks(annotated, mask, color, det["class_name"], alpha=0.5)
            det["mask"] = {"rle": _encode(mask), "polygon": _poly(mask)}

        else:
            draw_label(annotated, det["class_name"], (x1, y1, x2, y2), color)
            det["mask"] = {"rle": {}, "polygon": []}

        det["description"] = info["description"]
        det["solution"]    = info["solution"]
        det["severity"]    = (info.get("severity") or "medium").lower()
        det["source"] = "yolo"
        final.append(det)
    
    existing_boxes = [d["bbox"] for d in final]

    # 4) handle any “new” coarse GPT issues
    new_items   = [r for r in refinements if r.get("new")]
    if new_items:
        class_phrases = [
            (r.get("dino_prompt") or r.get("class_name") or r.get("description") or "urban issue").strip()
            for r in new_items
        ]
        dino_hits = await _ground_phrase(img_bgr, class_phrases, backend="local", existing = existing_boxes)
    else:
        dino_hits = {}
        class_phrases = []

    for r, phrase in zip(new_items, class_phrases):
        color = tuple(random.randint(0, 255) for _ in range(3))
        hits  = dino_hits.get(phrase, [])

        if not hits:
            H, W = img_bgr.shape[:2]
            hits = [{"bbox": [0, 0, W, H], "score": r.get("confidence", 0.0)}]

        for h in hits:
            x1, y1, x2, y2 = map(int, h["bbox"])
            score          = float(h["score"])
            existing_boxes.append([x1, y1, x2, y2])

            if use_sam:
                masks, scores, _ = predictor.predict(
                    box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    multimask_output=True,
                )
                mask      = masks[np.argmax(scores)]
                mask_data = {"rle": _encode(mask), "polygon": _poly(mask)}
                overlay_masks(annotated, mask, color, f"{r.get('class_name')}-gpt+dino", 0.5)
            else:
                mask_data = {"rle": {}, "polygon": []}
                draw_label(annotated, f"{r.get('class_name')}-gpt+dino", (x1, y1, x2, y2), color)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            final.append({
                "track_id":   None,
                "class_id":   -1,
                "class_name": f"{r.get('class_name')}-gpt+dino",
                "confidence": score,
                "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                "mask":       mask_data,
                "description": r["description"],
                "solution":    r["solution"],
                "severity": (r.get("severity") or "medium").lower(),
                "source": "gpt_dino",
            })
    
     # 5) full-image SAM fallback if nothing kept
    if use_sam and not final:
        logger.warning("No kept detections: falling back to full-image SAM masks")
        for mdata in mask_gen.generate(rgb):
            seg = mdata["segmentation"]
            color = tuple(random.randint(0,255) for _ in range(3))
            overlay_masks(annotated, seg, color, "clean", alpha=0.5)
            final.append({
                "track_id":   None,
                "class_id":   -1,
                "class_name": "clean",
                "confidence": float(mdata.get("score", 0)),
                "bbox":       [float(x) for x in mdata["bbox"]],
                "mask":       {"rle": _encode(seg), "polygon": _poly(seg)},
                "description": None,
                "solution":    None,
                "severity":    "low",
                "source": "sam_fallback",
            })

    return annotated, final


def _tracker_yaml() -> str:
    fh = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    yaml.safe_dump(PERF_CFG, fh)
    fh.flush()
    return fh.name

_TRACKER_YAML = _tracker_yaml()

def process_video(video_path: Path, use_sam: bool = True):
    logger.info("process_video() start", extra={"video_path": str(video_path), "use_sam": use_sam})
    yolo, predictor, mask_gen = _load_models()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video", extra={"video_path": str(video_path)})
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.debug("Video properties", extra={"fps": fps, "width": W, "height": H})

    dets_by_frame: dict[int, dict] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = yolo.track(
            frame, persist=True, imgsz=IMG_SZ, conf=CONF_T, iou=IOU_T,
            device=DEVICE, tracker=_TRACKER_YAML, verbose=False,
            save=False, save_txt=False, save_conf=False, save_crop=False,
        )[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        if boxes.size:
            dets_by_frame[idx] = {
                "frame": frame.copy(),
                "boxes": boxes,
                "ids": res.boxes.id.cpu().numpy() if res.boxes.id is not None else np.arange(len(boxes)),
                "confs": res.boxes.conf.cpu().numpy(),
                "clss": res.boxes.cls.int().cpu().numpy(),
            }
        idx += 1
    cap.release()
    logger.info("YOLO tracking complete", extra={"frames": len(dets_by_frame)})

    if use_sam:
        sam = SAM("sam2.1_b.pt")

    out_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.mp4"
    vw = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H),
    )

    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.6
    thickness = 2

    COLORS = {}
    frames_meta = []

    for frame_idx in sorted(dets_by_frame):
        data = dets_by_frame[frame_idx]
        frame, boxes, track_ids, confs, clss = (
            data["frame"], data["boxes"], data["ids"], data["confs"], data["clss"],
        )

        if use_sam:
            res = sam(
                source=[cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)],
                bboxes=[boxes.tolist()],
            )[0]
            raw_masks = res.masks.data.cpu().numpy().astype(bool)
        else:
            raw_masks = [None] * len(boxes)

        objects = []
        for box, raw_mask, tid, cid, conf in zip(boxes, raw_masks, track_ids, clss, confs):
            if use_sam and raw_mask is not None:
                mask = cv2.resize(
                    raw_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask = None

            if tid not in COLORS:
                COLORS[tid] = tuple(int(c) for c in np.random.randint(0, 255, 3))
            color = COLORS[tid]

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if use_sam:
                overlay_masks(frame, mask, color, yolo.names[cid], alpha=0.5)
            else:
                label = yolo.names[cid]
                (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
                cv2.rectangle(frame, (x1, y1-th-4), (x1+tw+4, y1), color, -1)
                cv2.putText(frame, label, (x1+2, y1-4),
                            font, scale, (255,255,255), thickness,
                            lineType=cv2.LINE_AA)


            objects.append({
                "track_id": int(tid),
                "class_id": int(cid),
                "class_name": yolo.names[int(cid)],
                "confidence": float(conf),
                "bbox": box.tolist(),
                "mask": {"rle": {}, "polygon": []},
                "source": "yolo",
            })

        vw.write(frame)
        frames_meta.append({
            "frame_index": frame_idx,
            "timestamp_ms": frame_idx * 1000.0 / fps,
            "objects": objects,
        })

    vw.release()
    return out_path, frames_meta



