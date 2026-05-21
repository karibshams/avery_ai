import streamlit as st
from dotenv import load_dotenv
from main import MealCostService

load_dotenv()

STORE_CATEGORIES: dict[str, list[str]] = {
    "Budget": ["Aldi", "Lidl", "Walmart", "WinCo", "Food Lion", "Grocery Outlet", "Market Basket"],
    "Mid Range": ["Kroger", "Publix", "HEB", "Safeway", "Meijer", "Albertsons",
                  "Harris Teeter", "Winn-Dixie", "Stop & Shop", "Giant Food",
                  "Jewel-Osco", "Hy-Vee", "Fred Meyer", "Trader Joe's"],
    "Premium / Specialty": ["Whole Foods", "Sprouts", "Wegmans", "Fresh Market", "Erewhon"],
    "Warehouse / Club": ["Costco", "Sam's Club", "BJ's Wholesale"],
}

def run():
    st.set_page_config(page_title="Meal Cost Predictor", page_icon="🍽️")
    st.title("🍽️ AI Meal Cost Predictor")
    st.caption("Upload a food photo → get an estimated home-cooking cost for one person.")

    # --- Inputs ---
    uploaded_file = st.file_uploader("Food Image", type=["jpg", "jpeg", "png", "webp"])
    location = st.text_input("Your Location", placeholder="e.g. New York, Dhaka, Los Angeles")

    category = st.selectbox("Grocery Store Category", list(STORE_CATEGORIES.keys()))
    store = st.selectbox("Preferred Store", STORE_CATEGORIES[category])
    store_label = f"{category} — {store}"

    # --- Submit ---
    if st.button("Estimate Cost", type="primary"):
        if not uploaded_file:
            st.warning("Please upload a food image.")
            return
        if not location.strip():
            st.warning("Please enter your location.")
            return

        with st.spinner("Analyzing image and estimating cost..."):
            try:
                service = MealCostService()
                result = service.predict(uploaded_file, location.strip(), store_label)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                return

        # --- Results ---
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        st.success(f"**Dish:** {result.dish_name}")

        st.subheader("Ingredients (1 serving)")
        for ing in result.ingredients:
            st.write(f"• **{ing.name}** — {ing.quantity} → `${ing.estimated_cost_usd:.2f}`")

        st.subheader("Estimated Home-Cooking Cost")
        st.metric(
            label=f"{result.location} · {result.store_type}",
            value=f"${result.cost_min_usd:.2f} – ${result.cost_max_usd:.2f}",
        )


if __name__ == "__main__":
    run()