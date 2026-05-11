#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageOps

from providers.gemini_vlm import GeminiTargetLocator
from providers.replicate_depth import ReplicateDepthClient

ROOT = Path(__file__).resolve().parent
DEPTH_ANYTHING_SRC = ROOT / "Depth-Anything-3" / "src"

if str(DEPTH_ANYTHING_SRC) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_SRC))

SKIP_LABELS = {"floor", "ceiling", "wall", "ground", "sky", "room", "space", "area"}
MARKER_COLORS = {
    "chair": "#ff4444",
    "table": "#44ff44",
    "door": "#4444ff",
    "person": "#ff8800",
    "plant": "#00cc44",
    "monitor": "#00ccff",
    "lamp": "#ffff00",
    "window": "#88ccff",
    "couch": "#cc44cc",
    "bed": "#ff6688",
    "sink": "#44cccc",
    "toilet": "#cccc44",
    "tv": "#0088ff",
    "book": "#cc8844",
    "bottle": "#44ccaa",
    "cup": "#ffaa44",
    "keyboard": "#aaaaaa",
    "phone": "#88ff88",
    "shelf": "#886644",
    "box": "#ff44aa",
    "cabinet": "#668844",
}
DEFAULT_FOV_DEG = 60.0
DEFAULT_MAX_POINTS = 15000


def load_depth_anything3_class():
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            "Local runtime requires Depth-Anything-3 dependencies. "
            "Run ./setup.sh, or set DEMO_RUNTIME=remote to use Replicate + Gemini."
        ) from exc
    return DepthAnything3


def default_torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def estimate_focal_px(width: int, height: int, fov_deg: float) -> float:
    fov_rad = math.radians(fov_deg)
    if fov_rad <= 0 or fov_rad >= math.pi:
        raise ValueError("fov_deg must be between 0 and 180.")
    sensor_span = float(max(width, height))
    return 0.5 * sensor_span / math.tan(fov_rad / 2.0)


def depth_preview_base64(depth_m: np.ndarray, valid_mask: np.ndarray) -> str:
    if not np.any(valid_mask):
        raise ValueError("No valid depth values available for preview.")

    lo, hi = np.percentile(depth_m[valid_mask], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1e-6

    scaled = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    preview = (scaled * 255.0).astype(np.uint8)
    preview_rgb = np.stack([preview, preview, preview], axis=-1)

    with io.BytesIO() as buffer:
        Image.fromarray(preview_rgb, mode="RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_base64(image: Image.Image) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def base_marker_label(label: str) -> str:
    return re.sub(r"[_ ]\d+$", "", label.strip().lower())


def marker_color_hex(label: str) -> str:
    return MARKER_COLORS.get(base_marker_label(label), "#ff00ff")


def normalize_vlm_url(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def make_point_cloud(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_points: int,
    sky_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    h, w = depth_m.shape
    yy, xx = np.indices((h, w), dtype=np.float32)

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if sky_mask is not None:
        valid &= ~sky_mask.astype(bool)

    if not np.any(valid):
        raise ValueError("No valid depth pixels remained after filtering.")

    near, far = np.percentile(depth_m[valid], [1.0, 99.0])
    valid &= depth_m >= near
    valid &= depth_m <= far

    valid_indices = np.flatnonzero(valid.reshape(-1))
    if valid_indices.size == 0:
        raise ValueError("Point cloud filtering removed every pixel.")

    if valid_indices.size > max_points:
        pick = np.linspace(0, valid_indices.size - 1, num=max_points, dtype=np.int64)
        valid_indices = valid_indices[pick]

    flat_x = xx.reshape(-1)[valid_indices]
    flat_y = yy.reshape(-1)[valid_indices]
    flat_z = depth_m.reshape(-1)[valid_indices]

    x = (flat_x - cx) * flat_z / fx
    y = -(flat_y - cy) * flat_z / fy
    z = -flat_z

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = (rgb.reshape(-1, 3)[valid_indices].astype(np.float32) / 255.0).astype(np.float32)

    bounds = {
        "depth_near_m": float(near),
        "depth_far_m": float(far),
        "point_count": int(points.shape[0]),
    }
    return points, colors, bounds


def custom_prompt_template(user_prompt: str) -> str:
    return f"""You are a vision assistant that localizes user-requested targets in a single image.
Find the visible image locations that best satisfy this request:
{user_prompt}

Return ONLY valid JSON in this exact format:
{{"targets":[{{"label":"short lowercase label","x":320,"y":650,"confidence":0.9}}]}}

Coordinate rules:
- x and y must be integers in the range 0-1000
- (0,0) is the top-left of the image
- (1000,1000) is the bottom-right of the image

Content rules:
- Include only targets that are clearly relevant to the request
- Use short lowercase labels
- Maximum 8 targets
- If nothing relevant is visible, return {{"targets":[]}}
- No markdown, no prose, no code fences, only JSON"""


def parse_vlm_targets(raw: str) -> list[dict]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    else:
        brace = cleaned.find("{")
        if brace >= 0:
            cleaned = cleaned[brace:]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM returned malformed JSON: {exc.msg}") from exc

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM returned a malformed JSON string: {exc.msg}") from exc

    if isinstance(data, list):
        targets = data
    elif isinstance(data, dict):
        targets = data.get("targets", [])
        if isinstance(targets, str):
            try:
                targets = json.loads(targets)
            except json.JSONDecodeError as exc:
                raise ValueError(f"VLM returned malformed targets JSON: {exc.msg}") from exc
    else:
        targets = []

    if not isinstance(targets, list):
        targets = []

    valid = []
    for obj in targets:
        if not all(k in obj for k in ("label", "x", "y")):
            continue
        label = str(obj["label"]).lower().strip()
        if not label or base_marker_label(label) in SKIP_LABELS:
            continue
        valid.append(
            {
                "label": label,
                "x": int(np.clip(int(obj["x"]), 0, 1000)),
                "y": int(np.clip(int(obj["y"]), 0, 1000)),
                "confidence": float(obj.get("confidence", 0.7)),
            }
        )
    return valid[:8]


def annotate_targets(rgb: np.ndarray, targets: list[dict]) -> str:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size

    for target in targets:
        px = int(np.clip(target["x"] / 1000.0 * w, 0, w - 1))
        py = int(np.clip(target["y"] / 1000.0 * h, 0, h - 1))
        color = marker_color_hex(target["label"])
        r = max(6, min(w, h) // 60)
        draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=3)
        draw.line((px, py, px, py - (r * 3)), fill=color, width=3)
        text = f'{target["label"]} {(target["confidence"] * 100):.0f}%'
        tx = min(max(8, px + r + 6), max(8, w - 140))
        ty = max(8, py - (r * 3) - 18)
        draw.rounded_rectangle((tx - 6, ty - 4, tx + 130, ty + 18), radius=6, fill=(0, 0, 0, 190), outline=color)
        draw.text((tx, ty), text, fill="white")

    return image_base64(image)


def project_target_to_3d(
    target: dict,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> dict | None:
    img_h, img_w = depth_map.shape
    px = int(np.clip(target["x"] / 1000.0 * img_w, 0, img_w - 1))
    py = int(np.clip(target["y"] / 1000.0 * img_h, 0, img_h - 1))

    r = 5
    y0, y1 = max(0, py - r), min(img_h, py + r + 1)
    x0, x1 = max(0, px - r), min(img_w, px + r + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.15) & (patch < 25.0)]
    if valid.size == 0:
        return None

    depth_m = float(np.median(valid))
    x_cam = (px - cx) * depth_m / fx
    y_cam = -(py - cy) * depth_m / fy
    z_cam = -depth_m
    return {
        "label": target["label"],
        "confidence": float(target["confidence"]),
        "pixel": {"x": int(px), "y": int(py)},
        "position": {
            "x": float(x_cam),
            "y": float(y_cam),
            "z": float(z_cam),
        },
    }


class DemoRuntime:
    def __init__(self, model_dir: str, device: str, process_res: int):
        self.runtime_name = "local"
        self.model_dir = str(Path(model_dir).expanduser())
        self.device = device
        self.process_res = process_res
        self.vlm_url = normalize_vlm_url(os.environ.get("QWEN_URL", "http://127.0.0.1:8012/v1"))
        self.vlm_model = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self._model: object | None = None
        self._lock = threading.Lock()

    def _is_local_model_ref(self) -> bool:
        raw = self.model_dir
        return raw.startswith("/") or raw.startswith(".") or raw.startswith("~")

    def load_model(self) -> object:
        with self._lock:
            if self._model is None:
                depth_anything3 = load_depth_anything3_class()
                print(f"Loading model: {self.model_dir} on {self.device}")
                model_path = Path(self.model_dir)
                if self._is_local_model_ref() and not model_path.is_dir():
                    raise FileNotFoundError(
                        "Local DA3 model path must be a directory containing Hugging Face-style "
                        f"weights; got: {model_path}"
                    )
                if model_path.is_dir():
                    self._model = depth_anything3.from_pretrained(str(model_path)).to(self.device).eval()
                elif self._is_local_model_ref():
                    raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
                else:
                    self._model = depth_anything3.from_pretrained(self.model_dir).to(self.device).eval()
            return self._model

    def infer_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> dict:
        model = self.load_model()
        np_image = np.asarray(image.convert("RGB"))

        prediction = model.inference(
            [np_image],
            process_res=self.process_res,
            process_res_method="upper_bound_resize",
        )

        rgb = prediction.processed_images[0]
        raw_depth = prediction.depth[0].astype(np.float32)
        sky = prediction.sky[0] if prediction.sky is not None else None

        height, width = raw_depth.shape
        used_focal_px = estimate_focal_px(width, height, DEFAULT_FOV_DEG)
        metric_depth_m = raw_depth * (used_focal_px / 300.0)

        fx = used_focal_px
        fy = used_focal_px
        cx = width * 0.5
        cy = height * 0.5

        points, colors, stats = make_point_cloud(
            rgb=rgb,
            depth_m=metric_depth_m,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            max_points=DEFAULT_MAX_POINTS,
            sky_mask=sky,
        )

        preview_mask = np.isfinite(metric_depth_m) & (metric_depth_m > 0.0)
        if sky is not None:
            preview_mask &= ~sky.astype(bool)

        targets_2d = []
        targets_3d = []
        target_error = None
        overlay_base64 = None
        if prompt and prompt.strip():
            try:
                targets_2d = self.query_vlm_targets(Image.fromarray(rgb, mode="RGB"), prompt.strip())
                overlay_base64 = annotate_targets(rgb, targets_2d)
                for target in targets_2d:
                    projected = project_target_to_3d(
                        target=target,
                        depth_map=metric_depth_m,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                    )
                    if projected is not None:
                        targets_3d.append(projected)
            except Exception as exc:
                target_error = f"Target localization failed: {exc}"

        return {
            "points": np.round(points, 4).tolist(),
            "colors": np.round(colors, 4).tolist(),
            "depth_preview": depth_preview_base64(metric_depth_m, preview_mask),
            "annotated_preview": overlay_base64,
            "targets_2d": targets_2d,
            "targets_3d": targets_3d,
            "meta": {
                "model_dir": self.model_dir,
                "device": self.device,
                "vlm_model": self.vlm_model,
                "input_size": {"width": int(image.width), "height": int(image.height)},
                "processed_size": {"width": int(width), "height": int(height)},
                "focal_px": float(used_focal_px),
                "fov_deg": float(DEFAULT_FOV_DEG),
                "is_metric_model_output": True,
                "prompt_used": prompt.strip() if prompt else None,
                "target_count": len(targets_3d),
                **stats,
            },
        }

    def query_vlm_targets(self, image: Image.Image, prompt: str) -> list[dict]:
        w, h = image.size
        max_dim = 384
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        image_b64 = image_base64(image)
        payload = {
            "model": self.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt_template(prompt)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 400,
            "temperature": 0,
        }
        response = requests.post(self.vlm_url, json=payload, timeout=45)
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return parse_vlm_targets(raw)


class RemoteDemoRuntime:
    def __init__(self):
        self.runtime_name = "remote"
        self.device = "remote-api"
        self.model_dir = os.environ.get(
            "REPLICATE_DEPTH_MODEL",
            "david20321/depth-anything-v3-metric-large",
        )
        self.vlm_url = "https://generativelanguage.googleapis.com/v1beta"
        self.vlm_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self._model = None
        self.depth_client = ReplicateDepthClient.from_env()
        self.vlm_client = GeminiTargetLocator.from_env()

    def load_model(self) -> None:
        # Remote mode has no local model to warm up. API keys are validated in __init__.
        return None

    def infer_image(
        self,
        image: Image.Image,
        prompt: str | None = None,
    ) -> dict:
        depth_result = self.depth_client.infer_depth(image)
        rgb = depth_result.rgb
        metric_depth_m = depth_result.depth_m
        sky = None

        height, width = metric_depth_m.shape
        used_focal_px = float(
            depth_result.meta.get("focal_length_used")
            or estimate_focal_px(width, height, DEFAULT_FOV_DEG)
        )

        fx = used_focal_px
        fy = used_focal_px
        cx = width * 0.5
        cy = height * 0.5

        points, colors, stats = make_point_cloud(
            rgb=rgb,
            depth_m=metric_depth_m,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            max_points=DEFAULT_MAX_POINTS,
            sky_mask=sky,
        )

        preview_mask = np.isfinite(metric_depth_m) & (metric_depth_m > 0.0)

        targets_2d = []
        targets_3d = []
        target_error = None
        overlay_base64 = None
        if prompt and prompt.strip():
            try:
                targets_2d = self.query_vlm_targets(Image.fromarray(rgb, mode="RGB"), prompt.strip())
                overlay_base64 = annotate_targets(rgb, targets_2d)
                for target in targets_2d:
                    projected = project_target_to_3d(
                        target=target,
                        depth_map=metric_depth_m,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                    )
                    if projected is not None:
                        targets_3d.append(projected)
            except Exception as exc:
                target_error = f"Target localization failed: {exc}"

        return {
            "points": np.round(points, 4).tolist(),
            "colors": np.round(colors, 4).tolist(),
            "depth_preview": depth_preview_base64(metric_depth_m, preview_mask),
            "annotated_preview": overlay_base64,
            "targets_2d": targets_2d,
            "targets_3d": targets_3d,
            "meta": {
                "runtime": self.runtime_name,
                "model_dir": self.model_dir,
                "device": self.device,
                "vlm_model": self.vlm_model,
                "input_size": {"width": int(image.width), "height": int(image.height)},
                "processed_size": {"width": int(width), "height": int(height)},
                "focal_px": float(used_focal_px),
                "fov_deg": float(DEFAULT_FOV_DEG),
                "is_metric_model_output": True,
                "prompt_used": prompt.strip() if prompt else None,
                "target_count": len(targets_3d),
                "target_error": target_error,
                **depth_result.meta,
                **stats,
            },
        }

    def query_vlm_targets(self, image: Image.Image, prompt: str) -> list[dict]:
        w, h = image.size
        max_dim = int(os.environ.get("VLM_IMAGE_MAX_SIZE", "768"))
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        return self.vlm_client.locate_targets(
            image=image,
            prompt_text=custom_prompt_template(prompt),
            parse_targets=parse_vlm_targets,
        )


def build_app(runtime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.load_model()
        yield

    app = FastAPI(title="Blinkin VLM", lifespan=lifespan)
    cors_origins = [
        origin.strip()
        for origin in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    def prune_jobs() -> None:
        cutoff = time.time() - 60 * 60
        with jobs_lock:
            stale_ids = [
                job_id
                for job_id, job in jobs.items()
                if job.get("created_at", 0) < cutoff
                and job.get("status") in {"succeeded", "failed"}
            ]
            for job_id in stale_ids:
                jobs.pop(job_id, None)

    def set_job(job_id: str, **updates) -> None:
        with jobs_lock:
            current = jobs.get(job_id, {})
            current.update(updates)
            current["updated_at"] = time.time()
            jobs[job_id] = current

    def run_infer_job(job_id: str, image: Image.Image, prompt: str | None) -> None:
        set_job(job_id, status="running", message="Running Blinkin VLM")
        print(f"[job:{job_id}] inference started", flush=True)
        try:
            result = runtime.infer_image(image=image, prompt=prompt)
        except Exception as exc:
            set_job(job_id, status="failed", error=str(exc), message="Inference failed")
            print(f"[job:{job_id}] inference failed: {exc}", flush=True)
            return

        set_job(
            job_id,
            status="succeeded",
            result=result,
            message="Image processed: mesh and payload ready",
        )
        print(f"[job:{job_id}] inference succeeded", flush=True)

    @app.get("/")
    async def index() -> FileResponse:
        response = FileResponse(ROOT / "index.html")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.get("/index.js")
    async def index_js() -> FileResponse:
        response = FileResponse(ROOT / "index.js", media_type="application/javascript")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": True,
            "service": "Blinkin VLM",
            "status": "ready",
        }

    @app.post("/api/infer")
    async def infer(
        image: UploadFile = File(...),
        prompt: str | None = Form(default=None),
    ) -> dict:
        print(f"[infer] sync request received: filename={image.filename}", flush=True)
        try:
            payload = await image.read()
            pil_image = Image.open(io.BytesIO(payload))
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            pil_image.load()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc

        try:
            return await run_in_threadpool(
                runtime.infer_image,
                image=pil_image,
                prompt=prompt,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/jobs")
    async def create_job(
        image: UploadFile = File(...),
        prompt: str | None = Form(default=None),
    ) -> dict:
        prune_jobs()
        print(f"[jobs] create request received: filename={image.filename}", flush=True)
        try:
            payload = await image.read()
            pil_image = Image.open(io.BytesIO(payload))
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            pil_image.load()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc

        job_id = str(uuid.uuid4())
        with jobs_lock:
            jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "message": "Queued single-image inference",
                "created_at": time.time(),
                "updated_at": time.time(),
            }

        thread = threading.Thread(
            target=run_infer_job,
            args=(job_id, pil_image.copy(), prompt),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found.")
            return dict(job)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Anything 3 metric point-cloud demo server.")
    parser.add_argument(
        "--runtime",
        choices=("local", "remote"),
        default=os.environ.get("DEMO_RUNTIME", "local"),
        help="local uses DA3 + local VLM; remote uses Replicate depth + Gemini VLM.",
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--model-dir", default="depth-anything/DA3METRIC-LARGE")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--device", default=default_torch_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runtime == "remote":
        runtime = RemoteDemoRuntime()
    else:
        runtime = DemoRuntime(
            model_dir=args.model_dir,
            device=args.device,
            process_res=args.process_res,
        )
    app = build_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
