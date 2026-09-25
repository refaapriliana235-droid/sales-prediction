import streamlit as st
import pandas as pd
import joblib

model = joblib.load("sales_model.pkl")

st.set_page_config(page_title="Prediksi Sales", page_icon="📊")
st.title("📊 Prediksi Penjualan (Sales)")

categories = {"Furniture": 0, "Office Supplies": 1, "Technology": 2}
subcats = {"Bookcases": 0, "Chairs": 1, "Tables": 2, "Furnishings": 3,
           "Binders": 4, "Appliances": 5, "Paper": 6, "Labels": 7,
           "Storage": 8, "Art": 9, "Phones": 10, "Accessories": 11,
           "Copiers": 12, "Machines": 13}
regions = {"South": 0, "West": 1, "Central": 2, "East": 3}
segments = {"Consumer": 0, "Corporate": 1, "Home Office": 2}
shipmodes = {"Standard Class": 0, "Second Class": 1, "First Class": 2, "Same Day": 3}

col1, col2 = st.columns(2)

with col1:
    cat = st.selectbox("Category", list(categories.keys()))
    subcat = st.selectbox("Sub-Category", list(subcats.keys()))
    region = st.selectbox("Region", list(regions.keys()))
    segment = st.selectbox("Segment", list(segments.keys()))

with col2:
    ship = st.selectbox("Ship Mode", list(shipmodes.keys()))
    qty = st.number_input("Quantity", 1, 20, 1)
    disc = st.slider("Discount", 0.0, 1.0, 0.1, 0.01)
    year = st.number_input("Year", 2015, 2030, 2024)
    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day", 1, 31, 15)
    wday = st.slider("Weekday (0=Senin)", 0, 6, 2)

if st.button("🔍 Prediksi"):
    data = pd.DataFrame([{
        "Category": categories[cat],
        "Sub-Category": subcats[subcat],
        "Region": regions[region],
        "Segment": segments[segment],
        "Ship Mode": shipmodes[ship],
        "Quantity": qty,
        "Discount": disc,
        "Year": year,
        "Month": month,
        "Day": day,
        "Weekday": wday,
    }])
    hasil = model.predict(data)[0]
    st.success(f"💰 Prediksi Sales: **${hasil:,.2f}**")
