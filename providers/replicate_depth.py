from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image


@dataclass
class RemoteDepthResult:
    rgb: np.ndarray
    depth_m: np.ndarray
    meta: dict[str, Any]


class ReplicateDepthClient:
    """Calls a Replicate-hosted depth model and decodes its depth output."""

    def __init__(
        self,
        api_token: str,
        model: str = "david20321/depth-anything-v3-metric-large",
        timeout_seconds: int = 300,
        wait_seconds: int = 60,
        max_input_size: int = 1280,
    ):
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN is required when DEMO_RUNTIME=remote.")
        if "/" not in model:
            raise ValueError("REPLICATE_DEPTH_MODEL must look like 'owner/model-name'.")

        owner, name = model.split("/", 1)
        self.api_token = api_token
        self.model = model
        self.owner = owner
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.wait_seconds = max(1, min(wait_seconds, 60))
        self.max_input_size = max_input_size
        self.version_id = os.environ.get("REPLICATE_DEPTH_VERSION", "").strip()
        self.model_endpoint = f"https://api.replicate.com/v1/models/{owner}/{name}"
        self.prediction_endpoint = (
            f"{self.model_endpoint}/predictions"
        )

    @classmethod
    def from_env(cls) -> "ReplicateDepthClient":
        return cls(
            api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
            model=os.environ.get(
                "REPLICATE_DEPTH_MODEL",
                "david20321/depth-anything-v3-metric-large",
            ),
            timeout_seconds=int(os.environ.get("REPLICATE_TIMEOUT_SECONDS", "300")),
            wait_seconds=int(os.environ.get("REPLICATE_WAIT_SECONDS", "60")),
            max_input_size=int(os.environ.get("REMOTE_IMAGE_MAX_SIZE", "1280")),
        )

    def infer_depth(self, image: Image.Image) -> RemoteDepthResult:
        prepared = self._prepare_image(image)
        prediction = self._create_prediction(prepared)
        output = self._wait_for_prediction(prediction)
        depth_m, depth_meta = self._decode_depth_output(output)

        height, width = depth_m.shape
        rgb = np.asarray(prepared.resize((width, height), Image.LANCZOS), dtype=np.uint8)
        return RemoteDepthResult(
            rgb=rgb,
            depth_m=depth_m.astype(np.float32),
            meta={
                "depth_provider": "replicate",
                "depth_model": self.model,
                **depth_meta,
            },
        )

    def _create_prediction(self, image: Image.Image) -> dict:
        input_payload: dict[str, Any] = {
            "image": self._image_to_data_url(image),
            "include_base64": False,
            "return_raw_depth": False,
        }
        input_payload.update(self._extra_input())

        if self.version_id:
            response = self._create_version_prediction(input_payload)
            response.raise_for_status()
            return response.json()

        response = requests.post(
            self.prediction_endpoint,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Prefer": f"wait={self.wait_seconds}",
                "Cancel-After": f"{self.timeout_seconds}s",
            },
            json={"input": input_payload},
            timeout=self.wait_seconds + 15,
        )
        if response.status_code == 404:
            response = self._create_version_prediction(input_payload)
        response.raise_for_status()
        return response.json()

    def _create_version_prediction(self, input_payload: dict[str, Any]) -> requests.Response:
        version_id = self.version_id or self._fetch_latest_version_id()
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Prefer": f"wait={self.wait_seconds}",
                "Cancel-After": f"{self.timeout_seconds}s",
            },
            json={"version": version_id, "input": input_payload},
            timeout=self.wait_seconds + 15,
        )
        return response

    def _fetch_latest_version_id(self) -> str:
        response = requests.get(
            self.model_endpoint,
            headers={"Authorization": f"Bearer {self.api_token}"},
            timeout=30,
        )
        response.raise_for_status()
        version_id = (response.json().get("latest_version") or {}).get("id")
        if not version_id:
            raise RuntimeError(f"Replicate model {self.model} did not expose a latest version id.")
        self.version_id = version_id
        return version_id

    def _wait_for_prediction(self, prediction: dict) -> Any:
        deadline = time.monotonic() + self.timeout_seconds
        current = prediction

        while True:
            status = current.get("status")
            if status == "succeeded":
                return current.get("output")
            if status in {"failed", "canceled"}:
                raise RuntimeError(f"Replicate depth prediction {status}: {current.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Replicate depth prediction timed out.")

            get_url = current.get("urls", {}).get("get")
            if not get_url:
                raise RuntimeError("Replicate response did not include a polling URL.")

            time.sleep(2.0)
            response = requests.get(
                get_url,
                headers={"Authorization": f"Bearer {self.api_token}"},
                timeout=30,
            )
            response.raise_for_status()
            current = response.json()

    def _decode_depth_output(self, output: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(output, dict):
            if output.get("depth_png_base64"):
                return self._decode_metric_png_base64(output)
            if output.get("depth_png"):
                return self._decode_metric_png_url(output, output["depth_png"])
            if "image" in output and self._has_metric_scale(output):
                return self._decode_metric_png_url(output, output["image"])
            if output.get("data"):
                return self._decode_npz_url(str(output["data"][0]), output)
            if output.get("depth_images"):
                return self._decode_relative_depth_image(str(output["depth_images"][0]), output)

        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, str):
                return self._decode_relative_depth_image(first, {"output_shape": "list"})

        if isinstance(output, str):
            return self._decode_relative_depth_image(output, {"output_shape": "string"})

        raise ValueError(f"Unsupported Replicate depth output shape: {type(output).__name__}")

    def _decode_metric_png_base64(self, output: dict) -> tuple[np.ndarray, dict[str, Any]]:
        encoded = str(output["depth_png_base64"])
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        return self._metric_png_to_depth(image, output)

    def _decode_metric_png_url(self, output: dict, url: str) -> tuple[np.ndarray, dict[str, Any]]:
        image = Image.open(io.BytesIO(self._download(url)))
        return self._metric_png_to_depth(image, output)

    def _decode_npz_url(self, url: str, output: dict) -> tuple[np.ndarray, dict[str, Any]]:
        with np.load(io.BytesIO(self._download(url))) as archive:
            key = self._pick_depth_key(archive)
            depth = archive[key].astype(np.float32)
        return depth, {
            "depth_source": "npz",
            "depth_key": key,
            "replicate_output": self._compact_output_meta(output),
        }

    def _decode_relative_depth_image(
        self,
        url: str,
        output: dict,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = Image.open(io.BytesIO(self._download(url))).convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        depth = 0.25 + arr * 8.0
        return depth, {
            "depth_source": "relative_image",
            "relative_depth_warning": (
                "No metric scale metadata was returned; visual 3D depth uses an approximate scale."
            ),
            "replicate_output": self._compact_output_meta(output),
        }

    def _metric_png_to_depth(
        self,
        image: Image.Image,
        output: dict,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._has_metric_scale(output):
            raise ValueError("Metric depth PNG output is missing scale_m_per_unit or offset_m.")

        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[..., 0]

        depth = arr.astype(np.float32) * float(output["scale_m_per_unit"]) + float(output["offset_m"])
        return depth, {
            "depth_source": "metric_png",
            "depth_min_m": output.get("depth_min_m"),
            "depth_max_m": output.get("depth_max_m"),
            "scale_m_per_unit": output.get("scale_m_per_unit"),
            "offset_m": output.get("offset_m"),
            "focal_length_used": output.get("focal_length_used"),
            "process_res_used": output.get("process_res_used"),
            "contract_version": output.get("contract_version"),
        }

    def _download(self, url: str) -> bytes:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.api_token}"},
            timeout=60,
        )
        response.raise_for_status()
        return response.content

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        max_side = max(width, height)
        if max_side <= self.max_input_size:
            return rgb

        scale = self.max_input_size / max_side
        return rgb.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=90, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _extra_input() -> dict[str, Any]:
        raw = os.environ.get("REPLICATE_DEPTH_EXTRA_INPUT_JSON", "").strip()
        if not raw:
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("REPLICATE_DEPTH_EXTRA_INPUT_JSON must decode to an object.")
        return parsed

    @staticmethod
    def _has_metric_scale(output: dict) -> bool:
        return "scale_m_per_unit" in output and "offset_m" in output

    @staticmethod
    def _pick_depth_key(archive: np.lib.npyio.NpzFile) -> str:
        for key in archive.files:
            value = archive[key]
            if "depth" in key.lower() and value.ndim == 2 and np.issubdtype(value.dtype, np.number):
                return key
        for key in archive.files:
            value = archive[key]
            if value.ndim == 2 and np.issubdtype(value.dtype, np.number):
                return key
        raise ValueError("Replicate NPZ output did not contain a 2D numeric depth array.")

    @staticmethod
    def _compact_output_meta(output: dict) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key, value in output.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                compact[key] = value
            elif isinstance(value, list):
                compact[key] = f"list[{len(value)}]"
            elif isinstance(value, dict):
                compact[key] = "object"
        return compact
