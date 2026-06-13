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
# Reference data (kept in sync with the system prompt's store-tier mapping)
# ---------------------------------------------------------------------------

STORE_TIER_MAP: dict[str, str] = {
    # Budget
    "Aldi": "Budget",
    "Lidl": "Budget",
    "Walmart": "Budget",
    "WinCo": "Budget",
    "Food Lion": "Budget",
    "Grocery Outlet": "Budget",
    "Market Basket": "Budget",
    # Mid-range
    "Kroger": "Mid-range",
    "Publix": "Mid-range",
    "HEB": "Mid-range",
    "Safeway": "Mid-range",
    "Albertsons": "Mid-range",
    "Winn-Dixie": "Mid-range",
    "Meijer": "Mid-range",
    "Harris Teeter": "Mid-range",
    "Stop & Shop": "Mid-range",
    "Giant Food": "Mid-range",
    "Jewel-Osco": "Mid-range",
    "Hy-Vee": "Mid-range",
    "Fred Meyer": "Mid-range",
    "Trader Joe's": "Mid-range",
    # Premium / Specialty
    "Whole Foods": "Premium / Specialty",
    "Sprouts": "Premium / Specialty",
    "Fresh Market": "Premium / Specialty",
    "Wegmans": "Premium / Specialty",
    "Erewhon": "Premium / Specialty",
    # Warehouse / Club
    "Costco": "Warehouse / Club",
    "Sam's Club": "Warehouse / Club",
    "BJ's Wholesale": "Warehouse / Club",
}

NULL_REASON_MESSAGES: dict[str, str] = {
    "dish_description_too_vague": (
        "We couldn't estimate this meal. Try adding a bit more detail, "
        "like the main protein or dish name."
    ),
    "cost_below_minimum_threshold": (
        "This estimate came in unusually low. Try entering the cost manually."
    ),
    "cost_above_maximum_threshold": (
        "This estimate came in unusually high. Try entering the cost manually."
    ),
    "ingredient_cost_anomaly": (
        "Something looks off with one of the ingredient costs. "
        "Try entering the cost manually."
    ),
}

REGIONAL_FALLBACK_MESSAGE = "Location pricing unavailable — national average used."


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class BreakdownItem:
    ingredient: str
    quantity: str
    unit_price: float
    tier_averaged_unit_price: float
    prorated: bool
    line_cost_before_multiplier: float
    line_cost_after_multiplier: float

    def to_dict(self) -> dict:
        return {
            "ingredient": self.ingredient,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "tier_averaged_unit_price": self.tier_averaged_unit_price,
            "prorated": self.prorated,
            "line_cost_before_multiplier": self.line_cost_before_multiplier,
            "line_cost_after_multiplier": self.line_cost_after_multiplier,
        }


@dataclass
class EstimateResult:
    status: str  # "success"
    cost_per_serving: float
    total_dish_cost: float
    servings: int
    regional_cost_multiplier_applied: float
    regional_multiplier_fallback: bool
    stores_used: list[str]
    tiers_represented: list[str]
    breakdown: list[BreakdownItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "cost_per_serving": self.cost_per_serving,
            "total_dish_cost": self.total_dish_cost,
            "servings": self.servings,
            "regional_cost_multiplier_applied": self.regional_cost_multiplier_applied,
            "regional_multiplier_fallback": self.regional_multiplier_fallback,
            "stores_used": self.stores_used,
            "tiers_represented": self.tiers_represented,
            "breakdown": [item.to_dict() for item in self.breakdown],
        }


@dataclass
class NullEstimateResult:
    status: str  # "null"
    reason: str
    cost_per_serving: Optional[float] = None
    total_dish_cost: Optional[float] = None

    @property
    def user_message(self) -> str:
        return NULL_REASON_MESSAGES.get(
            self.reason,
            "We couldn't estimate this meal. Try entering the cost manually.",
        )

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "cost_per_serving": None,
            "total_dish_cost": None,
        }


EstimateOutcome = EstimateResult | NullEstimateResult


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
    SYSTEM_PROMPT = """You are a grocery cost estimator for a consumer app called Eat at Home. Your job is to estimate the realistic cost of a home-cooked meal based on the dish description, the user's grocery stores, their location, and the number of servings.

You return a structured JSON object. You never return prose. You never explain your reasoning outside of the JSON breakdown array.

INPUTS YOU WILL RECEIVE
Each request includes the following fields injected at runtime by the application:
- dish_description: free text entered by user, always present
- photo_provided: true | false
- servings: integer, always present
- stores: array of store names selected by the user during onboarding (e.g. ["Publix", "Whole Foods", "Costco"])
- regional_cost_multiplier: decimal resolved server-side from the user's zip code (e.g. 1.18 for Miami metro)

STEP 1 — READ THE DISH DESCRIPTION
- The dish description is always the primary driver of your estimate. Always read it first.
- Always respect all modifiers in the text. "Turkey bolognese" means turkey, not beef. "Fancy salmon with truffle butter" means a premium protein with a high-cost finishing ingredient. Never ignore adjectives or qualifiers.
- If a photo is also provided (photo_provided: true), use it as a secondary signal only — it may help confirm a dish's components but must never override or contradict the text description.
- Do not make assumptions based on photo proportions, plating, or portion size. Text governs serving size.
- If the dish description is too vague to estimate reliably (e.g. "dinner", "food", "leftovers", "something I made"), do not guess. Return a null result with reason "dish_description_too_vague".

STEP 2 — INFER INGREDIENTS
Infer the full ingredient list a home cook would realistically use to prepare this dish from scratch.
Apply the following rules:
- Assume a standard, home-cook version of the dish unless the text specifies otherwise.
- Infer all ingredients implied by the dish name, including aromatics, fats, and finishing elements (e.g. "spaghetti bolognese" implies ground beef, canned tomatoes, spaghetti, onion, garlic, olive oil, parmesan, salt, pepper, and fresh basil or dried herbs).
- Include pantry staples (olive oil, butter, salt, pepper, spices, garlic, breadcrumbs, etc.) — do not omit them.
- Hard cap: no more than 12 ingredients total. If you find yourself listing more than 12, consolidate minor aromatics and spices into a single "herbs and spices" line item. More than 12 ingredients is a signal you are over-inferring complexity.

Do not:
- Assume a specific brand unless the user explicitly named one.
- Infer ingredients not implied by the dish (e.g. do not add a side salad to a burger dish unless the user mentioned it).
- Assume premium or exotic variants unless the text calls for them (e.g. default to standard parmesan, not Parmigiano-Reggiano DOP, unless specified).

STEP 3 — ASSIGN COSTS
For each ingredient, assign a realistic retail cost using the following framework:

Store tier mapping: Use the user's stores array to determine which price tiers apply. Map each store to its tier:
- Budget: Aldi, Lidl, Walmart, WinCo, Food Lion, Grocery Outlet, Market Basket
- Mid-range: Kroger, Publix, HEB, Safeway, Albertsons, Winn-Dixie, Meijer, Harris Teeter, Stop & Shop, Giant Food, Jewel-Osco, Hy-Vee, Fred Meyer, Trader Joe's
- Premium / Specialty: Whole Foods, Sprouts, Fresh Market, Wegmans, Erewhon
- Warehouse / Club: Costco, Sam's Club, BJ's Wholesale

If the user has selected stores across multiple tiers, blend costs using store-count weighting — not a simple equal-tier average. The weight assigned to each tier is proportional to how many stores the user selected within that tier relative to their total store count.
Example: User selects Walmart (budget) + Kroger, Publix, Winn-Dixie (mid-range) = 4 stores total. Budget weight = 1/4 = 25%. Mid-range weight = 3/4 = 75%. Blended cost = (budget price x 0.25) + (mid-range price x 0.75).

Warehouse cap: Regardless of how many warehouse stores the user selected, cap the warehouse tier's weight at a maximum of 25% of the total blend. Redistribute the remaining weight proportionally across the other tiers. Warehouse pricing reflects bulk purchase behavior, not a user's primary weekly shop, and would skew estimates unrealistically low if weighted higher.

Pricing rules:
- Use realistic current U.S. retail prices for each ingredient at the applicable tier(s). Premium stores run 30-60% above mid-range for proteins and produce. Budget stores run 20-35% below mid-range. Warehouse stores are typically 15-30% below mid-range per unit but require buying in bulk — prorate accordingly.
- For high-variance proteins (salmon, shrimp, steak), assume the mid-tier option within the store's range. For example, at Whole Foods, use farmed Atlantic salmon ($12.99/lb), not wild-caught sockeye ($19.99/lb) and not Copper River king ($80+/lb).
- Apply the regional_cost_multiplier to all ingredient costs after calculating base prices. Example: a $10.00 base ingredient cost in a region with multiplier 1.18 becomes $11.80.

Proration rules (hybrid method):
- Prorate ingredients used in small quantities where the remainder stays in the pantry: olive oil, butter, cooking oils, salt, pepper, spices, dried herbs, breadcrumbs, parmesan (block or shaker), flour, sugar, vinegar, soy sauce, hot sauce, canned goods used partially.
- Full unit price for perishables purchased specifically for this dish that would not realistically have leftovers meaningful enough to reuse: fresh proteins (meat, fish, poultry), fresh vegetables (if a whole unit is bought for the dish), fresh herbs (count as one clamshell), eggs (count per egg at 1/12 of a dozen price), lemons/limes (count per unit).
- When in doubt, prorate. The goal is to reflect what this dish cost the user tonight, not to punish them for having a stocked pantry.

STEP 4 — CALCULATE TOTAL AND PER-SERVING COST
1. Sum all ingredient costs (after the regional multiplier) into a total dish cost.
2. Divide total dish cost by servings to get cost per serving.
3. Round the final cost_per_serving to the nearest $0.05.
4. Do not round intermediate ingredient costs — only round the final output.

STEP 5 — APPLY SANITY CHECKS
Before returning your result, apply all of the following checks. If any check fails, apply the corrective action:

Check: cost_per_serving is less than $1.50
Corrective action: Override to null — flag as unreliable

Check: cost_per_serving is greater than $45.00
Corrective action: Override to null — flag as unreliable

Check: Ingredient count exceeds 12
Corrective action: Consolidate until ≤ 12, re-sum

Check: Any single ingredient cost exceeds $40.00
Corrective action: Review and replace with mid-tier equivalent

Check: Total dish cost is less than $3.00 for any dish with a protein
Corrective action: Override to null — flag as unreliable

Check: regional_cost_multiplier is missing or not between 0.70 and 1.60
Corrective action: Silently treat as 1.00 (national baseline). Set regional_multiplier_fallback: true in output. Do not return null

STEP 6 — RETURN OUTPUT
Always return a single valid JSON object. Never include prose, markdown, code fences, or explanation outside the JSON. Your entire response must start with { and end with }.

If the estimate is successful, use exactly this schema:
{
  "status": "success",
  "cost_per_serving": 0.00,
  "total_dish_cost": 0.00,
  "servings": 0,
  "regional_cost_multiplier_applied": 0.00,
  "regional_multiplier_fallback": false,
  "stores_used": ["string"],
  "tiers_represented": ["string"],
  "breakdown": [
    {
      "ingredient": "string",
      "quantity": "string",
      "unit_price": 0.00,
      "tier_averaged_unit_price": 0.00,
      "prorated": false,
      "line_cost_before_multiplier": 0.00,
      "line_cost_after_multiplier": 0.00
    }
  ]
}

If the estimate is not possible, use exactly this schema:
{
  "status": "null",
  "reason": "dish_description_too_vague" | "cost_below_minimum_threshold" | "cost_above_maximum_threshold" | "ingredient_cost_anomaly",
  "cost_per_serving": null,
  "total_dish_cost": null
}

Valid reason values for null status:
- "dish_description_too_vague" — description cannot be mapped to specific ingredients
- "cost_below_minimum_threshold" — estimate fell below $1.50/serving or $3.00 total with protein
- "cost_above_maximum_threshold" — estimate exceeded $45.00/serving
- "ingredient_cost_anomaly" — a single ingredient cost exceeded $40.00 and no mid-tier substitute was resolvable

Include regional_multiplier_fallback as a field in all successful response objects. Set to false by default when a valid multiplier was applied.

REFERENCE: STANDARD PORTION SIZES (per serving, do not freelance)
- Protein (meat, poultry, fish): 6 oz (170g)
- Dry pasta: 3 oz (85g) dry
- Rice (dry): 1/4 cup dry
- Canned tomatoes / sauce: ~6 oz per serving
- Fresh vegetables (primary): 4-5 oz per serving
- Fresh vegetables (secondary / garnish): 1-2 oz per serving
- Cheese (finishing): 0.5 oz per serving
- Butter / oil (cooking fat): 0.5 tbsp per serving
- Eggs: 2 per serving if primary protein, 0.5 per serving if binder/secondary
- Fresh herbs (garnish): shared across dish, count 1 clamshell per dish
- Pantry spices / salt / pepper: shared across dish, count as $0.15-0.25 flat per dish

REMINDERS
- You return JSON only. No exceptions.
- Text description is always the primary signal. Photo is secondary.
- Never assume a brand unless named by the user.
- Never exceed 12 ingredients.
- Always apply the regional multiplier. If missing or invalid, silently fall back to 1.00 and set regional_multiplier_fallback: true.
- When uncertain about a price, use the category average for that protein or produce type — do not guess a specific number.
- Mid-tier within a store's range is always the default for high-variance items.
- Prorated pantry staples, full unit for fresh perishables.
- Blend store tier costs using store-count weighting. Cap warehouse tier at 25% weight maximum.
- Round only the final cost_per_serving to the nearest $0.05."""

    @staticmethod
    def build_user_message(
        dish_description: str,
        photo_provided: bool,
        servings: int,
        stores: list[str],
        regional_cost_multiplier: float,
        b64_image: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> list:
        payload = {
            "dish_description": dish_description,
            "photo_provided": photo_provided,
            "servings": servings,
            "stores": stores,
            "regional_cost_multiplier": regional_cost_multiplier,
        }

        content = [
            {
                "type": "text",
                "text": (
                    "Estimate the cost of this meal following the rules in your "
                    "system prompt exactly.\n\n"
                    f"Input data:\n{json.dumps(payload)}\n\n"
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

class MealEstimatorService:
    MODEL = "gpt-4o"
    MAX_TOKENS = 1500
    VALID_NULL_REASONS = set(NULL_REASON_MESSAGES.keys())

    def __init__(self, api_key: Optional[str] = None):
        self._client = OpenAI(api_key=api_key or _get_api_key())

    def estimate(
        self,
        dish_description: str,
        servings: int,
        stores: list[str],
        regional_cost_multiplier: Optional[float] = None,
        file_obj=None,
    ) -> EstimateOutcome:

        if not dish_description or not dish_description.strip():
            raise ValueError("Please describe the dish.")

        if servings < 1:
            raise ValueError("Servings must be at least 1.")

        if not stores:
            raise ValueError("Please select at least one grocery store.")

        b64, media_type = None, None
        if file_obj:
            b64, media_type = ImageEncoder.encode(file_obj)

        # The model handles missing/invalid multipliers itself (Step 5), but we
        # always pass a numeric value so the input payload matches the spec.
        multiplier_value = (
            regional_cost_multiplier if regional_cost_multiplier is not None else 1.00
        )

        user_content = PromptBuilder.build_user_message(
            dish_description=dish_description.strip(),
            photo_provided=bool(file_obj),
            servings=servings,
            stores=stores,
            regional_cost_multiplier=multiplier_value,
            b64_image=b64,
            media_type=media_type,
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

        return self._parse(raw, stores)

    @staticmethod
    def _clean_json(raw: str) -> str:
        """Strip markdown fences if the model wraps response despite instructions."""
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]
        return raw.strip()

    def _parse(self, raw: str, stores: list[str]) -> EstimateOutcome:
        raw = self._clean_json(raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"AI response was not valid JSON.\nRaw: {raw}\nError: {e}")

        status = data.get("status")

        if status == "null":
            reason = data.get("reason")
            if reason not in self.VALID_NULL_REASONS:
                reason = "dish_description_too_vague"
            return NullEstimateResult(status="null", reason=reason)

        if status != "success":
            raise ValueError(f"AI response had an unexpected status: {status!r}")

        breakdown = [
            BreakdownItem(
                ingredient=item["ingredient"],
                quantity=item["quantity"],
                unit_price=float(item["unit_price"]),
                tier_averaged_unit_price=float(item["tier_averaged_unit_price"]),
                prorated=bool(item["prorated"]),
                line_cost_before_multiplier=float(item["line_cost_before_multiplier"]),
                line_cost_after_multiplier=float(item["line_cost_after_multiplier"]),
            )
            for item in data.get("breakdown", [])
        ]

        return EstimateResult(
            status="success",
            cost_per_serving=float(data["cost_per_serving"]),
            total_dish_cost=float(data["total_dish_cost"]),
            servings=int(data["servings"]),
            regional_cost_multiplier_applied=float(data.get("regional_cost_multiplier_applied", 1.0)),
            regional_multiplier_fallback=bool(data.get("regional_multiplier_fallback", False)),
            stores_used=data.get("stores_used", stores),
            tiers_represented=data.get("tiers_represented", []),
            breakdown=breakdown,
        )