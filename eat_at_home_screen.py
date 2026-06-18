import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from eat_at_home_service import (
    MealEstimatorService,
    EstimateResult,
    NullEstimateResult,
    STORE_TIER_MAP,
    REGIONAL_FALLBACK_MESSAGE,
    StaticZipMultiplierResolver,
    AiZipMultiplierResolver,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Store Categories (mirrors STORE_TIER_MAP in eat_at_home_service.py)
# ---------------------------------------------------------------------------

STORE_CATEGORIES: dict[str, list[str]] = {
    "Budget": ["Aldi", "Lidl", "Walmart", "WinCo", "Food Lion", "Grocery Outlet", "Market Basket"],
    "Mid-range": ["Kroger", "Publix", "HEB", "Safeway", "Albertsons", "Winn-Dixie", "Meijer",
                  "Harris Teeter", "Stop & Shop", "Giant Food", "Jewel-Osco", "Hy-Vee",
                  "Fred Meyer", "Trader Joe's"],
    "Premium / Specialty": ["Whole Foods", "Sprouts", "Fresh Market", "Wegmans", "Erewhon"],
    "Warehouse / Club": ["Costco", "Sam's Club", "BJ's Wholesale"],
}

CATEGORY_ICONS = {
    "Budget": "💚",
    "Mid-range": "💛",
    "Premium / Specialty": "💜",
    "Warehouse / Club": "📦",
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_store_selector() -> list[str]:
    """Renders grouped store checkboxes and returns list of selected stores."""
    st.subheader("🛒 Select Grocery Stores")
    st.caption("Pick the stores the user selected during onboarding.")

    selected = []
    for category, stores in STORE_CATEGORIES.items():
        icon = CATEGORY_ICONS.get(category, "🏪")
        with st.expander(f"{icon} {category}", expanded=False):
            cols = st.columns(2)
            for idx, store in enumerate(stores):
                col = cols[idx % 2]
                if col.checkbox(store, key=f"store_{store}"):
                    selected.append(store)

    return selected


def render_selected_tags(selected_stores: list[str]):
    """Shows selected stores as visual tags, grouped by tier for quick sanity-checking."""
    if not selected_stores:
        st.caption("No stores selected yet.")
        return

    tags = "  ".join([f"`{s}`" for s in selected_stores])
    st.markdown(f"**Selected:** {tags}")

    tiers = sorted({STORE_TIER_MAP[s] for s in selected_stores})
    st.caption(f"Tiers represented: {', '.join(tiers)}")


def render_success(result: EstimateResult):
    st.success(
        f"**Cost per serving: ${result.cost_per_serving:.2f}**  "
        f"&nbsp; | &nbsp; Total dish cost: ${result.total_dish_cost:.2f}"
    )

    if result.regional_multiplier_fallback:
        st.info(REGIONAL_FALLBACK_MESSAGE)

    col1, col2, col3 = st.columns(3)
    col1.metric("Servings", result.servings)
    col2.metric("Regional Multiplier", f"{result.regional_cost_multiplier_applied:.2f}x")
    col3.metric("Total Dish Cost", f"${result.total_dish_cost:.2f}")

    st.caption(
        f"🛒 Stores used: {', '.join(result.stores_used)}  •  "
        f"Tiers represented: {', '.join(result.tiers_represented)}"
    )

    st.subheader("🧾 Ingredient Breakdown")
    if result.breakdown:
        df = pd.DataFrame([item.to_dict() for item in result.breakdown])
        df = df.rename(columns={
            "ingredient": "Ingredient",
            "quantity": "Quantity",
            "unit_price": "Unit Price",
            "tier_averaged_unit_price": "Tier-Avg Unit Price",
            "prorated": "Prorated",
            "line_cost_before_multiplier": "Line Cost (before mult.)",
            "line_cost_after_multiplier": "Line Cost (after mult.)",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No breakdown items returned.")

    with st.expander("🔍 Raw JSON response"):
        st.json(result.to_dict())


def render_null(result: NullEstimateResult):
    st.warning(result.user_message)
    with st.expander("🔍 Raw JSON response"):
        st.json(result.to_dict())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    st.set_page_config(page_title="Eat at Home — Cost Estimator", page_icon="🍽️")
    st.title("🍽️ Eat at Home — Meal Cost Estimator")
    st.caption("Internal test harness for the v1.1 meal cost estimator system prompt.")

    # --- Dish Description ---
    dish_description = st.text_area(
        "Dish Description",
        placeholder="e.g. Fancy salmon with truffle butter, or Spaghetti bolognese",
        help="This is the primary signal for the estimate. Be as descriptive as you like.",
    )

    # --- Optional Photo ---
    uploaded_file = st.file_uploader(
        "Food Photo (optional — secondary signal only)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    # --- Servings ---
    servings = st.number_input("Servings", min_value=1, max_value=12, value=2, step=1)

    # --- Location / Regional Multiplier ---
    st.subheader("📍 Location")
    zip_code = st.text_input(
        "Zip Code",
        placeholder="e.g. 10001",
        help="Used to resolve the regional_cost_multiplier server-side. "
             "Unrecognized or out-of-range values fall back to the national baseline (1.00).",
    )

    resolver_mode = st.radio(
        "How should the ZIP code be resolved? (testing toggle)",
        options=["Static Table", "AI Lookup"],
        horizontal=True,
        help="Static Table: instant, free, only works for a small hardcoded list of ZIPs. "
             "AI Lookup: asks GPT-4o to identify the region and estimate a multiplier — "
             "works for any ZIP, costs one extra API call per unique ZIP.",
    )
    st.caption(
        "Leave the ZIP code blank and the request will use the national baseline (1.00)."
    )

    st.divider()

    # --- Stores ---
    selected_stores = render_store_selector()
    render_selected_tags(selected_stores)

    st.divider()

    # --- Submit ---
    if st.button("Estimate Cost", type="primary"):
        if not dish_description.strip():
            st.warning("Please enter a dish description.")
            return
        if not selected_stores:
            st.warning("Please select at least one grocery store.")
            return

        with st.spinner("Estimating cost..."):
            try:
                service = MealEstimatorService()
                cleaned_zip = zip_code.strip() if zip_code.strip() else None

                resolved_multiplier = None
                fallback_used = None
                if resolver_mode == "AI Lookup" and cleaned_zip:
                    ai_resolver = AiZipMultiplierResolver(service._client)
                    resolved_multiplier, fallback_used = ai_resolver.resolve(cleaned_zip)
                    service = MealEstimatorService(multiplier_resolver=ai_resolver)
                elif cleaned_zip:
                    static_resolver = StaticZipMultiplierResolver()
                    resolved_multiplier, fallback_used = static_resolver.resolve(cleaned_zip)

                result = service.estimate(
                    dish_description=dish_description,
                    servings=int(servings),
                    stores=selected_stores,
                    zip_code=cleaned_zip,
                    file_obj=uploaded_file if uploaded_file else None,
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                return

        if cleaned_zip:
            mode_label = "🤖 AI Lookup" if resolver_mode == "AI Lookup" else "📋 Static Table"
            st.caption(
                f"{mode_label} resolved ZIP {cleaned_zip} → multiplier "
                f"{resolved_multiplier:.2f}x (fallback used: {fallback_used})"
            )

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image (secondary signal)", use_container_width=True)

        if isinstance(result, EstimateResult):
            render_success(result)
        else:
            render_null(result)


if __name__ == "__main__":
    run()