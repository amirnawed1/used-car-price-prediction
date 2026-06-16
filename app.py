import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ========== HEADER ==========
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("🚗")
with col2:
    st.title("Used Car Price Predictor")
    st.markdown("### Estimate your car's resale value based on brand, age, and usage")
st.markdown("---")

# ========== LOAD DATA & MODEL ==========
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_car_data.csv")

@st.cache_resource
def load_model():
    return pickle.load(open("LinearRegressionModel.pkl", "rb"))

car = load_data()
model = load_model()

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("🚘 Car Details")
    st.markdown("Enter your car's specifications")

    name = st.selectbox("🚗 Model Name", sorted(car["name"].unique()))
    company = st.selectbox("🏢 Brand / Company", sorted(car["company"].unique()))

    st.markdown("---")

    car_age = st.slider("📅 Car Age (Years)", 0, 30, 5)
    kms_driven = st.slider("🛣️ Kilometers Driven", 0, 200000, 30000, 1000)
    fuel_type = st.selectbox("⛽ Fuel Type", sorted(car["fuel_type"].unique()))

    st.markdown("---")
    predict_btn = st.button("💰 Predict Resale Price", type="primary", use_container_width=True)

# ========== MAIN AREA ==========
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Market Snapshot")
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Price", f"₹{car['Price'].mean():,.0f}")
    m2.metric("Max Price", f"₹{car['Price'].max():,.0f}")
    m3.metric("Cars Listed", f"{len(car):,}")

with col2:
    st.markdown("### 📈 Price Distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(car["Price"], bins=40, kde=True, color="#FF914D", ax=ax)
    ax.axvline(car["Price"].mean(), color="blue", linestyle="--", label="Average")
    ax.set_xlabel("Price (₹)")
    ax.legend()
    st.pyplot(fig)

# ========== PREDICTION ==========
if predict_btn:
    st.markdown("---")
    st.markdown("## 🎯 Estimated Resale Value")

    input_data = pd.DataFrame({
        "name": [name],
        "company": [company],
        "kms_driven": [kms_driven],
        "fuel_type": [fuel_type],
        "car_age": [car_age]
    })

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    # Price bar
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.barh(0, car["Price"].max(), color="#E0E0E0", height=0.4, label="Market Range")
    ax.barh(0, prediction, color="#4ECDC4", height=0.4, label="Your Car")
    ax.scatter(prediction, 0, color="#FF6B6B", s=150, zorder=5)
    ax.set_xlabel("Price (₹)")
    ax.set_yticks([])
    ax.legend(loc="lower right")
    st.pyplot(fig)

    r1, r2 = st.columns(2)
    with r1:
        st.success(f"### 💰 ₹ {prediction:,.0f}")
        st.metric("Estimated Price", f"₹ {prediction:,.0f}")
    with r2:
        st.info("""
        **Factors considered:**
        - Brand & Model
        - Age & Kilometers
        - Fuel Type
        """)

    # Depreciation insight
    st.markdown("#### 📉 Depreciation Insight")
    avg_price_same_age = car[car["car_age"] == car_age]["Price"].mean()
    if not pd.isna(avg_price_same_age):
        diff = ((prediction - avg_price_same_age) / avg_price_same_age) * 100
        if diff > 0:
            st.warning(f"Your car is **{abs(diff):.1f}% above** average for its age group")
        else:
            st.success(f"Your car is **{abs(diff):.1f}% below** average for its age group")
    st.balloons()

# ========== FOOTER ==========
st.markdown("---")
st.markdown("### 👨‍💻 About This Project")
st.markdown("""
**Model:** Linear Regression | **Data:** Used Car Listings Dataset  
**Built by:** Amir Nawed | [GitHub](https://github.com/amirnawed1)  
**Deployed on:** Streamlit Cloud
""")
