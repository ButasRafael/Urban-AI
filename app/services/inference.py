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
from typing import List, Dict, Any, Tuple, DefaultDict
from pycocotools import mask as mask_util
import yaml
import random
import logging
import os, json, base64, re
from openai import AsyncOpenAI, InternalServerError, RateLimitError, APIConnectionError
from dotenv import load_dotenv
from fastapi import HTTPException
from groundingdino.util.inference import Model as _GDINO
import time
import asyncio
import httpx
import math
from collections import defaultdict

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
        # Get ALL unique descriptions and solutions, not just 2
        all_descriptions = [i["description"] for i in items if i.get("description")]
        all_solutions = [i["solution"] for i in items if i.get("solution")]
        
        # Remove duplicates while preserving order
        seen_desc = set()
        unique_descriptions = []
        for desc in all_descriptions:
            if desc and desc not in seen_desc:
                seen_desc.add(desc)
                unique_descriptions.append(desc)
        
        seen_sol = set()
        unique_solutions = []
        for sol in all_solutions:
            if sol and sol not in seen_sol:
                seen_sol.add(sol)
                unique_solutions.append(sol)
        
        # Format descriptions and solutions
        desc_text = ""
        if unique_descriptions:
            if len(unique_descriptions) <= 3:
                desc_text = "\n    descriptions: " + " | ".join(f"\"{d}\"" for d in unique_descriptions)
            else:
                # For many descriptions, format them on separate lines for clarity
                desc_text = "\n    descriptions:\n      - " + "\n      - ".join(f"\"{d}\"" for d in unique_descriptions)
        
        sol_text = ""
        if unique_solutions:
            if len(unique_solutions) <= 3:
                sol_text = "\n    solutions: " + " | ".join(f"\"{s}\"" for s in unique_solutions)
            else:
                # For many solutions, format them on separate lines for clarity
                sol_text = "\n    solutions:\n      - " + "\n      - ".join(f"\"{s}\"" for s in unique_solutions)
        
        lines.append(
            f"- {issue_type}: {total} detections | severity -> "
            f"high:{sev['high']}, medium:{sev['medium']}, low:{sev['low']}, unknown:{sev['unknown']}{desc_text}{sol_text}"
        )

    issues_text = "\n".join(lines)

    # Instructions go in the instructions parameter for GPT-4.1
    instructions = """# Role & Objective
You are an urban infrastructure expert. Summarize detected issues and propose an integrated remediation plan.

# Response Rules (follow strictly)
- Output ONLY a single JSON object, no prose, no markdown, no code fences
- The JSON must contain exactly these keys: "description" and "solution"
- "description": 2-3 concise sentences covering all issues found as a cohesive overview
- "solution": 3-4 concise sentences with an integrated, prioritized action plan that:
  • sequences work by severity and logical dependencies
  • mentions coordination/safety or access constraints if relevant
  • avoids redundant steps
  • is directly actionable for city operations teams
- Do not invent assets, measurements, or locations not implied by the data
- Group similar problems; avoid repetitive listing of identical issues
- Think through prioritization and dependencies privately; do not reveal your reasoning

# Output Format
Return exactly: {{"description": "<2-3 sentences>", "solution": "<3-4 sentences>"}}

# Final Instruction
If you cannot comply for any reason, return the smallest valid JSON that satisfies the schema.
"""

    async with _GPT_SEMAPHORE:
        try:
            resp = await client.responses.create(
                model="gpt-4.1",
                instructions=instructions,
                input=f"Issue Summary:\n{issues_text}",
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

    # Instructions moved to instructions parameter per GPT-4.1 best practices
    instructions = """# Role & Objective
You are analyzing an urban-planning scenario based on an input image and its corresponding YOLO detections (bounding boxes).

# Instructions
Perform the following tasks to validate existing detections and identify critical missing issues.

## 1. For Each YOLO Detection Present in the Array
(If the YOLO array is empty, skip this section entirely and output **no fields named "track_id" or "keep"):
- "track_id": integer (the YOLO track-ID provided)
- "keep": boolean (true if YOLO correctly identified the object; false if it is a mis-identification)
- "description": string (a clear one-sentence description of the detected object or issue)
- "solution": string — if the object is an issue, give a concise, practical, one-sentence remediation proposal; if the object is NOT an issue, write "No solution needed".
- "severity": "low" | "medium" | "high" (lowercase only)

### Severity Rubric
Use this rubric: high = acute safety/operational hazard or large impact; medium = material degradation or likely to become hazardous soon; low = minor/cosmetic. If uncertain, choose the lower severity.

Do NOT include bounding-box coordinates in your response; YOLO's geometry will be used directly.

## 2. Only Add New JSON Objects for Additional Urban Issues
When they are BOTH important and realistically solvable:
- "new": true
- "class_name": string (the specific type or category of urban issue identified)
- "confidence": float between 0 and 1 (your confidence in this additional issue)
- "description": string (a clear one-sentence description of the newly identified issue)
- "solution": string (a concise, practical, one-sentence remediation proposal)
- "severity": "low" | "medium" | "high" (lowercase only; same rubric as above)
- "dino_prompt": string

### Dino Prompt Rules
• 1 – 3 **lower-case tokens**, each ≤ 2 words.
• Choose COCO/LVIS/VisualGenome-style nouns whenever possible (e.g. "traffic light", "trash bin").
• NO commas, punctuation, verbs, prepositions, or numerals.
• Color/material adjectives when they are the *sole* reliable cue (e.g. "red cone").
• Put the **most visually distinctive token first**; order the rest by distinctiveness.
• Prefer the canonical dataset label ("fire hydrant" not "hydrant").
• Use singular form unless plurality is visually obvious.
• Hard cap of three tokens — if unsure, pick ONE high-precision noun.
• Separate tokens with ONE space exactly.

Skip minor, cosmetic, or trivial issues entirely — return ZERO new detections if the photo appears clean.
Do NOT specify exact bounding-box coordinates; these additional detections will be logged as coarse issues.

# Internal Reasoning Strategy (apply privately, do not output)
1. First pass: Validate each YOLO detection against visual evidence
2. Second pass: Scan for critical infrastructure issues YOLO may have missed  
3. For each potential new detection: Is it both important AND fixable?
4. For dino_prompt generation: What would a computer vision model need to locate this?
5. Final check: Have I avoided duplicating any YOLO detections?

# Output Format
Return **ONLY** a JSON-formatted array containing all of the above-described objects, and NOTHING ELSE.

# Final Instructions
- Return ONLY a JSON-formatted array, nothing else
- Valid JSON syntax required - no markdown, no backticks, no extra text
- If YOLO array is empty, output empty array []
- Every field must have correct type as specified above
- If you cannot comply for any reason, return the smallest valid JSON array that satisfies the schema"""

    initial = sorted(initial, key=lambda d: -d["confidence"])[:50]

    # Build clean input with data only (no rules)
    content = []
    content.append({"type": "input_text", "text": f"YOLO detections:\n{json.dumps(initial, indent=2)}"})
    content.append({"type": "input_image", "image_url": image_url, "detail": "high"})

    async with _GPT_SEMAPHORE:
        max_retries = 5
        backoff = 0.5

        for attempt in range(max_retries):
            try:
                resp = await client.responses.create(
                    model="gpt-4.1",
                    instructions=instructions,
                    input=[{"role": "user", "content": content}],
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

def _crop_with_margin(img: np.ndarray, box: List[float], margin: float = 0.08) -> np.ndarray:
    """
    Crop image with margin around bounding box.
    Returns a cropped region with additional margin for better context.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    
    # Ensure box coordinates are valid
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Calculate margin based on box dimensions
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    
    # Calculate crop coordinates with margin
    ix1 = int(max(0, math.floor(x1 - mx)))
    iy1 = int(max(0, math.floor(y1 - my)))
    ix2 = int(min(w, math.ceil(x2 + mx)))
    iy2 = int(min(h, math.ceil(y2 + my)))
    
    # Handle edge cases where margin calculation fails
    if ix2 <= ix1 or iy2 <= iy1:
        # Fallback to original box without margin, ensuring valid bounds
        ix1 = int(max(0, min(x1, w-1)))
        iy1 = int(max(0, min(y1, h-1)))
        ix2 = int(max(ix1+1, min(x2, w)))  # Ensure at least 1 pixel width
        iy2 = int(max(iy1+1, min(y2, h)))  # Ensure at least 1 pixel height
    
    return img[iy1:iy2, ix1:ix2]

def _select_track_samples(
    dets_by_frame: Dict[int, Dict],
    max_per_track: int = 1,  # Only keeping 1 sample per track
    diff_threshold: float = 0.975,  # Not used anymore but kept for compatibility
    min_conf_threshold: float = 0.3,  # ignore very low confidence detections
    temporal_spread_factor: float = 0.3,  # Not used anymore but kept for compatibility
) -> Dict[int, Dict]:
    """
    Select the highest confidence sample for each track.
    Returns: { track_id: { 'class_name': str, 'samples': [(frame_idx, crop_bgr), ...], 'best_conf': float } }
    
    Strategy:
    - Keep only the highest confidence detection per track (best quality sample)
    """
    if not dets_by_frame:
        return {}
    
    # Load models once for efficiency
    yolo, _, _ = _load_models()
    
    # Collect all detections per track first
    track_detections: Dict[int, List[Tuple[int, np.ndarray, float, int]]] = defaultdict(list)
    frame_indices = sorted(dets_by_frame.keys())
    total_frames = len(frame_indices)
    
    for frame_idx in frame_indices:
        data = dets_by_frame[frame_idx]
        frame, boxes, ids, clss, confs = data["frame"], data["boxes"], data["ids"], data["clss"], data["confs"]

        for box, tid, cid, conf in zip(boxes, ids, clss, confs):
            tid = int(tid)
            conf = float(conf)
            
            # Skip very low confidence detections
            if conf < min_conf_threshold:
                continue
                
            try:
                crop = _crop_with_margin(frame, box, margin=0.08)
                if crop.size > 0:  # Ensure valid crop
                    track_detections[tid].append((frame_idx, crop, conf, int(cid)))
            except Exception:
                continue  # Skip problematic crops
    
    # Now intelligently select samples for each track
    per_track: Dict[int, Dict] = {}
    
    for tid, detections in track_detections.items():
        if not detections:
            continue
            
        # Sort by frame index for temporal processing
        detections.sort(key=lambda x: x[0])
        
        class_name = yolo.names[detections[0][3]]  # Use class from first detection
        best_conf = max(det[2] for det in detections)
        
        # Keep only the highest confidence detection per track
        best_det = max(detections, key=lambda x: x[2])
        selected_samples = [(best_det[0], best_det[1])]
        
        # Store results
        if selected_samples:
            per_track[tid] = {
                "class_name": class_name,
                "samples": selected_samples,
                "best_conf": best_conf,
                "total_detections": len(detections),
                "frame_span": detections[-1][0] - detections[0][0] if len(detections) > 1 else 0,
            }
    
    return per_track

async def _gpt_describe_tracks(
    tracks: Dict[int, Dict],
    batch_size: int = 3,  # Reduced from 10 to 3 tracks per GPT call
    max_images_per_track: int = 3,
) -> Dict[int, Dict[str, str]]:

    out: Dict[int, Dict[str, str]] = {}

    instructions = """# Role and Objective
You are an urban infrastructure expert analyzing video footage of urban environments. Evaluate tracked objects across multiple video frames and determine whether each represents a legitimate infrastructure issue requiring remediation.

# Instructions

## Analysis Task
- You receive multiple image crops for each track_id showing the same object tracked across different video frames
- Each track represents one detected object that appears consistently throughout the video
- Analyze ALL provided crops for each track to make an informed decision

## Decision Criteria
- Keep (true): Object represents a legitimate urban infrastructure issue that requires attention
- Reject (false): Object is normal urban elements (vehicles, pedestrians, buildings, trees, etc.) or false detections

## Response Requirements
- Provide exactly ONE JSON object per track_id
- Never skip a track_id that appears in the metadata
- Base your analysis on visual evidence from ALL provided crops for each track
- If uncertain about severity level, always choose the lower severity option

## Severity Classification
- high: Acute safety hazard or operational emergency requiring immediate attention
- medium: Infrastructure degradation likely to become hazardous soon, or moderate impact on operations  
- low: Minor cosmetic issues or maintenance needs with minimal operational impact

# Internal Reasoning (do privately)
1. Identify what the tracked object is
2. Assess if it's an infrastructure issue or normal urban element
3. Evaluate severity if it's an issue
4. Formulate description and solution

# Output Format
Output ONLY a JSON array like: [{"track_id": 1, "keep": true, "description": "...", "solution": "...", "severity": "low"}]

## Strict Requirements
- Valid JSON array format, no markdown, no backticks, no extra text
- Each description must be exactly one clear, factual sentence
- Each solution must be exactly one actionable sentence, or exactly "No solution needed" for non-issues
- If an image is too ambiguous, set keep=false and use the description to state the uncertainty briefly

# Final Instruction
Return ONLY a JSON array. No markdown, no backticks, no extra text. If you cannot comply for any reason, return the smallest valid JSON that satisfies the schema."""

    # Chunk tracks to keep image count sane
    tids = list(tracks.keys())
    for i in range(0, len(tids), batch_size):
        chunk_tids = tids[i:i+batch_size]

        # Build input content (data only, no rules)
        content = []

        # Meta for this batch (class hints)
        meta = []
        for tid in chunk_tids:
            meta.append({
                "track_id": int(tid),
                "class_name": tracks[tid]["class_name"],
            })
        content.append({"type": "input_text", "text": f"Track metadata:\n{json.dumps({'meta': meta})}"})

        # Images (≤3 per track)
        for tid in chunk_tids:
            samples = tracks[tid]["samples"][:max_images_per_track]
            for _, crop in samples:
                _, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                b64 = base64.b64encode(buf).decode()
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                })
                # Hint tying the image to track id
                content.append({"type": "input_text", "text": f"track_id={tid}"})

        async with _GPT_SEMAPHORE:
            try:
                resp = await client.responses.create(
                    model="gpt-4.1",
                    instructions=instructions,
                    input=[{"role": "user", "content": content}],
                    temperature=0.2,
                    timeout=40,
                )
            except Exception as e:
                logger.error(f"GPT track description failed: {e}")
                continue
                
        raw = resp.output_text or ""
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            continue
        try:
            arr = json.loads(raw[start:end+1])
        except Exception:
            continue

        for obj in arr:
            try:
                tid = int(obj.get("track_id"))
            except Exception:
                continue
            out[tid] = {
                "keep": bool(obj.get("keep", True)),
                "description": (obj.get("description") or "").strip(),
                "solution": (obj.get("solution") or "").strip(),
                "severity": (obj.get("severity") or "medium").lower(),
            }
    return out

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

    # NEW: Select representative crops per track and get GPT annotations
    track_samples = _select_track_samples(dets_by_frame, max_per_track=3, diff_threshold=0.92)
    
    # Call async GPT function from sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    track_notes = loop.run_until_complete(_gpt_describe_tracks(track_samples))
    
    # Optional filtering: drop non-issues (keep=false)
    drop_track_ids = {tid for tid, note in track_notes.items() if note.get("keep") is False}
    
    # Save track thumbnails for frontend display (best crop per track)
    track_thumbnails = {}
    for tid, track_data in track_samples.items():
        if tid not in drop_track_ids and track_data.get("samples"):
            # Use the best confidence sample (usually the clearest image)
            best_sample = track_data["samples"][0]  # First sample is often best quality
            _, best_crop = best_sample
            track_thumbnails[tid] = best_crop

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
            tid = int(tid)
            
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

            # Pull GPT fields if present
            note = track_notes.get(tid, {})
            if tid in drop_track_ids:
                desc = ""
                sol = "No solution needed"
                sev = "low"
            else:
                desc = (note.get("description") or "").strip()
                sol = (note.get("solution") or "").strip()
                sev = (note.get("severity") or "medium").lower()

            objects.append({
                "track_id": tid,
                "class_id": int(cid),
                "class_name": yolo.names[int(cid)],
                "confidence": float(conf),
                "bbox": box.tolist(),
                "mask": {"rle": {}, "polygon": []},
                "source": "yolo",
                "description": desc if desc else None,
                "solution": sol if sol else None,
                "severity": sev,
            })

        vw.write(frame)
        frames_meta.append({
            "frame_index": frame_idx,
            "timestamp_ms": frame_idx * 1000.0 / fps,
            "objects": objects,
        })

    vw.release()
    return out_path, frames_meta, track_thumbnails



