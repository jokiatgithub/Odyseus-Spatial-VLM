from __future__ import annotations

import base64
import io
import os
from collections.abc import Callable

import requests
from PIL import Image


class GeminiTargetLocator:
    """Calls Gemini with an image and asks it to return target coordinates."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout_seconds: int = 60,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when DEMO_RUNTIME=remote.")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent"
        )

    @classmethod
    def from_env(cls) -> "GeminiTargetLocator":
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            timeout_seconds=int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "60")),
        )

    def locate_targets(
        self,
        image: Image.Image,
        prompt_text: str,
        parse_targets: Callable[[str], list[dict]],
    ) -> list[dict]:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": self._image_to_base64_png(image),
                            }
                        },
                        {"text": prompt_text},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "1200")),
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(
            self.endpoint,
            headers={"x-goog-api-key": self.api_key},
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return parse_targets(self._extract_text(response.json()))

    @staticmethod
    def _image_to_base64_png(image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.convert("RGB").save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _extract_text(payload: dict) -> str:
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini returned no candidates.")

        parts = candidates[0].get("content", {}).get("parts") or []
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        raw = "".join(text_parts).strip()
        if not raw:
            raise ValueError("Gemini returned an empty text response.")
        return raw
