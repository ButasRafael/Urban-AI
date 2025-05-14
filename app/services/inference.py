
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
from collections import defaultdict
import shutil
import random
import logging
from typing import Tuple, List, Optional
import os, json, base64
from openai import AsyncOpenAI, InternalServerError, RateLimitError, APIConnectionError
from dotenv import load_dotenv
from fastapi import HTTPException
from groundingdino.util.inference import Model as _GDINO


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
    """
    Wait until path exists and is non‐zero‐length (or timeout).
    """
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

    # base font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = float(np.clip((y2-y1) / 200.0, 0.8, 2.0))
    thickness  = int(np.clip((y2-y1) // 120, 1, 5))

    # measure text size at base scale
    (tw, th), _ = cv2.getTextSize(text, font, base_scale, thickness)
    avail_w = min(w_img, x2 - x1) - 2*pad
    # if too wide, shrink the scale proportionally
    if tw + 2*pad > avail_w and tw > 0:
        scale = base_scale * (avail_w / (tw + 2*pad))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    else:
        scale = base_scale

    # initial position: above the box (or inside if too high)
    x0 = int(np.clip(x1, 0, w_img - tw - 2*pad))
    y0 = y1 - th - 2*pad
    if y0 < 0:
        y0 = y2 + 2  # put inside/below box

    # define our new label rect
    rect = (x0, y0, x0 + tw + 2*pad, y0 + th + 2*pad)

    # nudge it down if it overlaps any previous label
    for prev in _label_rects:
        if rects_overlap(rect, prev):
            # bump below the previous label
            y0 = prev[3] + pad
            # clamp inside image
            y0 = int(min(y0, h_img - th - 2*pad))
            rect = (x0, y0, x0 + tw + 2*pad, y0 + th + 2*pad)

    # record this rect for future overlap checks
    _label_rects.append(rect)

    # draw background
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness=-1)
    # draw text
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
    """IoU between two boxes in [x1,y1,x2,y2] format."""
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w  = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h  = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter    = inter_w * inter_h
    area_a   = (xa2 - xa1) * (ya2 - ya1)
    area_b   = (xb2 - xb1) * (yb2 - yb1)
    union    = area_a + area_b - inter + 1e-6
    return inter / union


def _ground_phrase(
    img_bgr: np.ndarray,
    phrase: str,
    existing: List[List[float]] = [],
    box_thresh: float = 0.25,
    text_thresh: float = 0.2,
    iou_filter: float = 0.3,
):
    """
    Returns (x1, y1, x2, y2, score) **in absolute pixels** or None.
    Compatible with GroundingDINO ≥ 0.6 which gives supervision.Detections.
    """
    grounder = _load_grounder()
    detections, _ = grounder.predict_with_caption(   # note the unpack!
        image=img_bgr[:, :, ::-1],                   # BGR ➞ RGB
        caption=phrase,
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )

    if detections.xyxy.shape[0] == 0:      # nothing found
        return None

     # --- NEW: discard boxes that overlap a kept YOLO box ---
    keep = []
    for idx, box in enumerate(detections.xyxy):
        if all(iou_xyxy(box, ex) < iou_filter for ex in existing):
            keep.append(idx)

    if not keep:                                      # nothing genuinely new
        return None

    # pick the best of the remaining boxes
    idx = int(max(keep, key=lambda i: detections.confidence[i]))
    x1, y1, x2, y2 = detections.xyxy[idx]
    score          = float(detections.confidence[idx])
    return int(x1), int(y1), int(x2), int(y2), score




def _run_yolo(img: np.ndarray, run_id: str | None = None,) -> List[Dict[str, Any]]:
    """
    Run YOLO detection on the input BGR image and return a list of
    dicts with keys: track_id, class_name (with '-yolo' suffix),
    confidence, and bbox [x1, y1, x2, y2].
    """
    # load the YOLO model (and ignore SAM predictor/mask_gen)
    yolo, _, _ = _load_models()

    # perform detection
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

    # pull out boxes, confidences, and class indices
    boxes = results.boxes.xyxy.cpu().numpy()      # shape: (N, 4)
    confs = results.boxes.conf.cpu().numpy()     # shape: (N,)
    clss  = results.boxes.cls.int().cpu().numpy()  # shape: (N,)

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
        })

    return output

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
    "You are analyzing an urban planning scenario based on an input image and its corresponding YOLO detections (bounding boxes)."
    " Perform the following tasks:\n"
    "\n"
    "1. For each YOLO detection provided, create exactly one JSON object containing:\n"
    "   - \"track_id\": integer (the YOLO track ID provided)\n"
    "   - \"keep\": boolean (true if the detected issue is valid and relevant, otherwise false)\n"
    "   - \"description\": string (a clear one-sentence description of the detected urban issue)\n"
    "   - \"solution\": string (a concise, practical, one-sentence remediation proposal for this issue)\n"
    "\n"
    "   Do NOT include bounding box coordinates in your response, as YOLO's geometry will be used directly.\n"
    "\n"
    "2. If you identify additional urban issues that were NOT detected by YOLO, add new JSON objects with:\n"
    "   - \"new\": true\n"
    "   - \"class_name\": string (the specific type or category of urban issue identified)\n"
    "   - \"confidence\": float between 0 and 1 (your confidence level for detecting this additional issue)\n"
    "   - \"description\": string (a clear one-sentence description of the newly identified issue)\n"
    "   - \"solution\": string (a concise, practical, one-sentence remediation proposal)\n"
    "\n"
    "   Do NOT specify exact bounding box coordinates; these additional detections will be logged as coarse issues.\n"
    "\n"
    "3. Return ONLY a JSON-formatted array containing all of the above-described JSON objects, and NOTHING ELSE."
)

    
    initial = sorted(initial, key=lambda d: -d["confidence"])[:50]

    payload = [{
        "role": "user",
        "content": [
            {"type": "input_text",  "text": prompt_text},
            {"type": "input_image", "image_url": image_url},
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
                    timeout=35,
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                # exponential back-off for rate / network errors
                await asyncio.sleep(backoff + random.random() * 0.2)
                backoff *= 2
            except InternalServerError as e:
                # 5) bubble up the real status and body to logs and HTTPException
                status = e.response.status_code if e.response else 502
                body   = e.response.text if e.response else "<no body>"
                logger.error("OpenAI 5xx: %s – %s", status, body)
                raise HTTPException(
                    502,
                    f"Upstream service error (OpenAI {status}): {body}"
                )
        else:
            # all retries failed without raising InternalServerError
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

    # base size based on image
    raw_scale = min(w_img, h_img) * 0.003
    raw_th    = int(min(w_img, h_img) * 0.004)
    base_scale = float(np.clip(raw_scale, 0.8, 2.0))
    thickness = int(np.clip(raw_th, 2, 6))

    # measure and possibly shrink to image width
    (w_txt, h_txt), _ = cv2.getTextSize(label, font, base_scale, thickness)
    pad = int(thickness * 2)
    max_w = w_img - 2*pad
    if w_txt + 2*pad > max_w:
        scale = base_scale * (max_w / (w_txt + 2*pad))
        (w_txt, h_txt), _ = cv2.getTextSize(label, font, scale, thickness)
    else:
        scale = base_scale

    # start at mask centroid
    x_c, y_c = int(xs.mean()), int(ys.mean())
    x0 = x_c - w_txt//2 - pad
    y0 = y_c - h_txt - pad - 5
    # keep on‐screen
    x0 = int(np.clip(x0, 0, w_img - w_txt - 2*pad))
    if y0 < 0:
        y0 = int(np.clip(y_c + 5, 0, h_img - h_txt - 2*pad))

    # pack rect
    rect = (x0, y0, x0 + w_txt + 2*pad, y0 + h_txt + 2*pad)

    # bump if overlaps any existing
    for prev in _label_rects:
        if not (rect[2] < prev[0] or prev[2] < rect[0] or rect[3] < prev[1] or prev[3] < rect[1]):
            # move below prev
            y0 = prev[3] + pad
            y0 = int(min(y0, h_img - h_txt - 2*pad))
            rect = (x0, y0, x0 + w_txt + 2*pad, y0 + h_txt + 2*pad)

    _label_rects.append(rect)

    # draw bg & text
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

    # make a copy to draw on
    annotated = img_bgr.copy()
    final     = []

    # prep SAM (and mask generator) if requested
    if use_sam:
        _, predictor, mask_gen = _load_models()
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
    else:
        _, _, mask_gen = _load_models()

    # compute font size & thickness once
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

        # always draw the YOLO box first
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        if use_sam:
            # overlay the best SAM mask (centroid label only)
            masks, scores, _ = predictor.predict(
                box=np.array([[x1, y1, x2, y2]]), multimask_output=True
            )
            mask = masks[np.argmax(scores)]
            overlay_masks(annotated, mask, color, det["class_name"], alpha=0.5)
            det["mask"] = {"rle": _encode(mask), "polygon": _poly(mask)}

        else:
            # YOLO‐only: we already have the box, now add the label
            draw_label(annotated, det["class_name"], (x1, y1, x2, y2), color)
            det["mask"] = {"rle": {}, "polygon": []}

        # attach GPT’s description + solution
        det["description"] = info["description"]
        det["solution"]    = info["solution"]
        final.append(det)
    
    existing_boxes = [d["bbox"] for d in final]

    # 4) handle any “new” coarse GPT issues
    for r in refinements:
        if not r.get("new"):
            continue
        color = tuple(random.randint(0, 255) for _ in range(3))
        phrase = r.get("class_name") or r.get("description") or "urban issue"
        grounded = _ground_phrase(img_bgr, phrase,existing=existing_boxes)
        H, W = img_bgr.shape[:2]
        if grounded is None:
            x1,y1,x2,y2 = 0,0,W,H
            score = 0.0
            mask_data = {"rle": {}, "polygon": []}
        else:
            x1, y1, x2, y2, score = grounded
            if use_sam:
                masks, scores, _ = predictor.predict(
                    box=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    multimask_output=True
                )
                mask = masks[np.argmax(scores)]
                mask_data = {"rle": _encode(mask), "polygon": _poly(mask)}
                overlay_masks(annotated, mask, color, f"{phrase}-gpt+dino", alpha=0.5)
            else:
                mask_data = {"rle": {}, "polygon": []}
                (tw, th), _ = cv2.getTextSize(f"{phrase}-gpt+dino", font, scale, thickness)
                draw_label(
                    annotated, f"{phrase}-gpt+dino", (x1, y1, x2, y2), color,
                )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        final.append({
                "track_id":   None,
                "class_id":   -1,
                "class_name": f"{phrase}-gpt+dino",
                "confidence": score if score != 0 else r.get("confidence", 0),
                "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                "mask":      mask_data,
                "description": r["description"],
                "solution":    r["solution"],
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
            })

        vw.write(frame)
        frames_meta.append({
            "frame_index": frame_idx,
            "timestamp_ms": frame_idx * 1000.0 / fps,
            "objects": objects,
        })

    vw.release()
    return out_path, frames_meta




