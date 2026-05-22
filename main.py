import os
import base64
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _get_api_key() -> str:
    """Reads API key from Streamlit secrets (deployed) or .env (local)."""
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


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
    cost_avg_usd: float
    location: str
    selected_stores: list[str]

    def to_dict(self) -> dict:
        return {
            "dish_name": self.dish_name,
            "location": self.location,
            "selected_stores": self.selected_stores,
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
            "cost_avg_usd": self.cost_avg_usd,
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
        "Given a food image or dish name (or both), user location, and a list of selected grocery stores, you must:\n"
        "1. Identify the dish from the image. If a dish name is also provided, use it to confirm or refine your identification.\n"
        "   If no image is provided, use the dish name directly.\n"
        "2. List ingredients with realistic quantities for ONE serving.\n"
        "3. Estimate each ingredient's cost in USD — use the average price across all selected stores.\n"
        "4. Calculate cost_min_usd (based on the cheapest store in the list), "
        "cost_max_usd (based on the most expensive store), "
        "and cost_avg_usd (average across all selected stores).\n"
        "5. Return ONLY a raw JSON object — no markdown, no ```json fences, no explanation.\n"
        "   Your entire response must start with { and end with }.\n\n"
        "JSON schema (use exactly these keys):\n"
        "{\n"
        '  "dish_name": "string",\n'
        '  "ingredients": [\n'
        '    {"name": "string", "quantity": "string", "estimated_cost_usd": 0.00}\n'
        "  ],\n"
        '  "cost_min_usd": 0.00,\n'
        '  "cost_max_usd": 0.00,\n'
        '  "cost_avg_usd": 0.00\n'
        "}"
    )

    @staticmethod
    def build_user_message(
        location: str,
        selected_stores: list[str],
        b64_image: Optional[str] = None,
        media_type: Optional[str] = None,
        dish_name: Optional[str] = None,
    ) -> list:
        stores_str = ", ".join(selected_stores)
        dish_hint = f"Dish name provided by user: {dish_name}\n" if dish_name else ""

        content = [
            {
                "type": "text",
                "text": (
                    f"{dish_hint}"
                    f"Location: {location}\n"
                    f"Selected grocery stores: {stores_str}\n"
                    "Estimate the home-cooking cost for one person across all selected stores. "
                    "Reply with raw JSON only."
                ),
            }
        ]

        if b64_image and media_type:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64_image}"},
            })

        return content


# ---------------------------------------------------------------------------
# AI Service
# ---------------------------------------------------------------------------

class MealCostService:
    MODEL = "gpt-4o"
    MAX_TOKENS = 900

    def __init__(self, api_key: Optional[str] = None):
        self._client = OpenAI(api_key=api_key or _get_api_key())

    def predict(
        self,
        location: str,
        selected_stores: list[str],
        file_obj=None,
        dish_name: Optional[str] = None,
    ) -> MealCostResult:

        if not file_obj and not dish_name:
            raise ValueError("Please provide a food image, a dish name, or both.")

        if not selected_stores:
            raise ValueError("Please select at least one grocery store.")

        b64, media_type = None, None
        if file_obj:
            b64, media_type = ImageEncoder.encode(file_obj)

        user_content = PromptBuilder.build_user_message(
            location=location,
            selected_stores=selected_stores,
            b64_image=b64,
            media_type=media_type,
            dish_name=dish_name,
        )

        response = self._client.chat.completions.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=[
                {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError("AI returned an empty response. Check your API key and model access.")

        return self._parse(raw, location, selected_stores)

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Strip markdown fences if the model wraps response despite instructions."""
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]
        return raw.strip()

    def _parse(self, raw: str, location: str, selected_stores: list[str]) -> MealCostResult:
        raw = self._clean_json(raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"AI response was not valid JSON.\nRaw: {raw}\nError: {e}")

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
            cost_avg_usd=float(data["cost_avg_usd"]),
            location=location,
            selected_stores=selected_stores,
        )