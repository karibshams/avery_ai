import os
import base64
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Ingredient:
    name: str
    quantity: str
    estimated_cost_usd: float


@dataclass
class MealCostResult:
    dish_name: str
    ingredients: list[Ingredient]
    cost_min_usd: float
    cost_max_usd: float
    location: str
    store_type: str

    def to_dict(self) -> dict:
        return {
            "dish_name": self.dish_name,
            "location": self.location,
            "store_type": self.store_type,
            "ingredients": [
                {
                    "name": i.name,
                    "quantity": i.quantity,
                    "estimated_cost_usd": i.estimated_cost_usd,
                }
                for i in self.ingredients
            ],
            "cost_min_usd": self.cost_min_usd,
            "cost_max_usd": self.cost_max_usd,
        }


# ---------------------------------------------------------------------------
# Image Encoder
# ---------------------------------------------------------------------------

class ImageEncoder:
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}

    @staticmethod
    def encode(file_obj) -> tuple[str, str]:
        """Returns (base64_string, media_type)."""
        ext = Path(file_obj.name).suffix.lower()
        if ext not in ImageEncoder.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")
        raw = file_obj.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        media_type = f"image/{ext.lstrip('.')}"
        return b64, media_type


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    SYSTEM_PROMPT = (
        "You are a culinary cost analyst. "
        "Given a food image, user location, and grocery store preference, you must:\n"
        "1. Identify the dish.\n"
        "2. List ingredients with realistic quantities for ONE serving.\n"
        "3. Estimate each ingredient's cost in USD based on the location and store type.\n"
        "4. Return ONLY valid JSON — no markdown, no explanation.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "dish_name": string,\n'
        '  "ingredients": [\n'
        '    {"name": string, "quantity": string, "estimated_cost_usd": number}\n'
        "  ],\n"
        '  "cost_min_usd": number,\n'
        '  "cost_max_usd": number\n'
        "}"
    )

    @staticmethod
    def build_user_message(b64_image: str, media_type: str, location: str, store_type: str) -> list:
        return [
            {
                "type": "text",
                "text": (
                    f"Location: {location}\n"
                    f"Grocery store preference: {store_type}\n"
                    "Identify the meal in this image and estimate the home-cooking cost for one person."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64_image}"},
            },
        ]


# ---------------------------------------------------------------------------
# AI Service
# ---------------------------------------------------------------------------

class MealCostService:
    MODEL = "gpt-4o"
    MAX_TOKENS = 600

    def __init__(self, api_key: Optional[str] = None):
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def predict(self, file_obj, location: str, store_type: str) -> MealCostResult:
        b64, media_type = ImageEncoder.encode(file_obj)
        user_content = PromptBuilder.build_user_message(b64, media_type, location, store_type)

        response = self._client.chat.completions.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=[
                {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )

        raw = response.choices[0].message.content.strip()
        return self._parse(raw, location, store_type)

    def _parse(self, raw: str, location: str, store_type: str) -> MealCostResult:
        data = json.loads(raw)
        ingredients = [
            Ingredient(
                name=i["name"],
                quantity=i["quantity"],
                estimated_cost_usd=float(i["estimated_cost_usd"]),
            )
            for i in data.get("ingredients", [])
        ]
        return MealCostResult(
            dish_name=data["dish_name"],
            ingredients=ingredients,
            cost_min_usd=float(data["cost_min_usd"]),
            cost_max_usd=float(data["cost_max_usd"]),
            location=location,
            store_type=store_type,
        )