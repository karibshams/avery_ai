import streamlit as st
from dotenv import load_dotenv
from main import MealCostService

load_dotenv()

# ---------------------------------------------------------------------------
# Store Categories
# ---------------------------------------------------------------------------

STORE_CATEGORIES: dict[str, list[str]] = {
    "Budget": ["Aldi", "Lidl", "Walmart", "WinCo", "Food Lion", "Grocery Outlet", "Market Basket"],
    "Mid Range": ["Kroger", "Publix", "HEB", "Safeway", "Meijer", "Albertsons",
                  "Harris Teeter", "Winn-Dixie", "Stop & Shop", "Giant Food",
                  "Jewel-Osco", "Hy-Vee", "Fred Meyer", "Trader Joe's"],
    "Premium / Specialty": ["Whole Foods", "Sprouts", "Wegmans", "Fresh Market", "Erewhon"],
    "Warehouse / Club": ["Costco", "Sam's Club", "BJ's Wholesale"],
}

CATEGORY_ICONS = {
    "Budget": "💚",
    "Mid Range": "💛",
    "Premium / Specialty": "💜",
    "Warehouse / Club": "📦",
}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render_store_selector() -> list[str]:
    """Renders grouped store checkboxes and returns list of selected stores."""
    st.subheader("🛒 Select Grocery Stores")
    st.caption("Pick one or more stores from any category to compare prices.")

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
    """Shows selected stores as visual tags."""
    if not selected_stores:
        st.caption("No stores selected yet.")
        return
    tags = "  ".join([f"`{s}`" for s in selected_stores])
    st.markdown(f"**Selected:** {tags}")


def run():
    st.set_page_config(page_title="Meal Cost Predictor", page_icon="🍽️")
    st.title("🍽️ AI Meal Cost Predictor")
    st.caption("Upload a food photo or enter a dish name → AI estimates your home-cooking cost.")

    # --- Image Input ---
    uploaded_file = st.file_uploader("Food Image (optional if dish name provided)",
                                     type=["jpg", "jpeg", "png", "webp"])

    # --- Dish Name Input ---
    dish_name = st.text_input("Dish Name (optional if image provided)",
                              placeholder="e.g. Chicken Curry, Margherita Pizza")

    # --- Location ---
    location = st.text_input("Your Location", placeholder="e.g. New York, Los Angeles")

    # --- Multi Store Selector ---
    selected_stores = render_store_selector()
    render_selected_tags(selected_stores)

    st.divider()

    # --- Submit ---
    if st.button("Estimate Cost", type="primary"):
        if not uploaded_file and not dish_name.strip():
            st.warning("Please upload a food image or enter a dish name.")
            return
        if not location.strip():
            st.warning("Please enter your location.")
            return
        if not selected_stores:
            st.warning("Please select at least one grocery store.")
            return

        with st.spinner("Analyzing and estimating cost across selected stores..."):
            try:
                service = MealCostService()
                result = service.predict(
                    location=location.strip(),
                    selected_stores=selected_stores,
                    file_obj=uploaded_file if uploaded_file else None,
                    dish_name=dish_name.strip() if dish_name.strip() else None,
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                return

        # --- Results ---
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        st.success(f"**Dish Detected:** {result.dish_name}")
        st.caption(f"📍 {result.location}  •  🛒 {', '.join(result.selected_stores)}")

        st.subheader("🧾 Ingredients (1 serving)")
        for ing in result.ingredients:
            st.write(f"• **{ing.name}** — {ing.quantity} → `${ing.estimated_cost_usd:.2f}`")

        st.subheader("💰 Estimated Home-Cooking Cost  ( testing purpose)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum", f"${result.cost_min_usd:.2f}")
        col2.metric("Average", f"${result.cost_avg_usd:.2f}")
        col3.metric("Maximum", f"${result.cost_max_usd:.2f}")

        st.caption("Min = cheapest store · Avg = across all selected · Max = premium store")


if __name__ == "__main__":
    run()