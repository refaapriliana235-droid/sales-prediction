import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data_superstore.csv.csv", encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y", errors="coerce")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df

@st.cache_resource
def load_model():
    return joblib.load("sales_model.pkl")

df = load_data()
model = load_model()

with st.sidebar:
    st.markdown("## 📊 Sales Prediction")
    st.markdown("**Machine Learning Dashboard**")
    st.markdown("---")
    menu = st.radio("Navigasi", [
        "🏠 Overview",
        "📊 Visualisasi Data",
        "🎯 Prediksi Sales",
        "ℹ️ Tentang"
    ])
    st.markdown("---")
    st.markdown("### 🟢 Status")
    st.success("Model Active")
    st.caption("Random Forest Regressor")
    st.caption(f"Total data: {len(df):,} baris")

# ============ OVERVIEW ============
if menu == "🏠 Overview":
    st.title("📊 Sales Prediction Dashboard")
    st.markdown("Dashboard analisis data penjualan **Superstore** dengan Machine Learning.")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Sales", f"${df['Sales'].sum():,.0f}")
    with col2:
        st.metric("🛒 Total Orders", f"{df['Order ID'].nunique():,}")
    with col3:
        st.metric("👥 Customers", f"{df['Customer ID'].nunique():,}")
    with col4:
        st.metric("📈 Avg Sales", f"${df['Sales'].mean():,.2f}")
    
    st.markdown("---")
    st.subheader("📋 Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)

# ============ VISUALISASI ============
elif menu == "📊 Visualisasi Data":
    st.title("📊 Visualisasi Data Penjualan")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💼 Sales per Category")
        st.bar_chart(df.groupby("Category")["Sales"].sum().sort_values(ascending=False))
    with col2:
        st.subheader("🌍 Sales per Region")
        st.bar_chart(df.groupby("Region")["Sales"].sum().sort_values(ascending=False))
    
    st.markdown("---")
    st.subheader("📈 Tren Penjualan per Tahun")
    st.line_chart(df.groupby("Year")["Sales"].sum())
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👥 Sales per Segment")
        st.bar_chart(df.groupby("Segment")["Sales"].sum().sort_values(ascending=False))
    with col2:
        st.subheader("🚚 Sales per Ship Mode")
        st.bar_chart(df.groupby("Ship Mode")["Sales"].sum().sort_values(ascending=False))
    
    st.markdown("---")
    st.subheader("🏆 Top 10 Sub-Category by Sales")
    st.bar_chart(df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).head(10))

# ============ PREDIKSI ============
elif menu == "🎯 Prediksi Sales":
    st.title("🎯 Prediksi Penjualan")
    st.markdown("Masukkan data transaksi untuk memprediksi nilai penjualan.")
    st.markdown("---")
    
    categories = {"Furniture": 0, "Office Supplies": 1, "Technology": 2}
    subcats = {"Bookcases": 0, "Chairs": 1, "Tables": 2, "Furnishings": 3,
               "Binders": 4, "Appliances": 5, "Paper": 6, "Labels": 7,
               "Storage": 8, "Art": 9, "Phones": 10, "Accessories": 11,
               "Copiers": 12, "Machines": 13}
    regions = {"South": 0, "West": 1, "Central": 2, "East": 3}
    segments = {"Consumer": 0, "Corporate": 1, "Home Office": 2}
    shipmodes = {"Standard Class": 0, "Second Class": 1, "First Class": 2, "Same Day": 3}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cat = st.selectbox("Category", list(categories.keys()))
        subcat = st.selectbox("Sub-Category", list(subcats.keys()))
        region = st.selectbox("Region", list(regions.keys()))
        segment = st.selectbox("Segment", list(segments.keys()))
    with col2:
        ship = st.selectbox("Ship Mode", list(shipmodes.keys()))
        qty = st.number_input("Quantity", 1, 20, 3)
        disc = st.slider("Discount", 0.0, 1.0, 0.1, 0.01)
    with col3:
        year = st.number_input("Year", 2015, 2030, 2024)
        month = st.slider("Month", 1, 12, 6)
        day = st.slider("Day", 1, 31, 15)
        wday = st.slider("Weekday", 0, 6, 2, help="0=Senin, 6=Minggu")
    
    st.markdown("---")
    if st.button("🔍 PREDIKSI SEKARANG", use_container_width=True):
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
        st.success(f"### 💰 Hasil Prediksi Sales: **${hasil:,.2f}**")
        with st.expander("📋 Lihat detail input"):
            st.dataframe(data)

# ============ TENTANG ============
else:
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("---")
    st.markdown("""
    ### 📊 Sales Prediction App
    
    Aplikasi Machine Learning untuk **memprediksi nilai penjualan** berdasarkan 
    karakteristik transaksi retail Superstore.
    
    ### 🛠️ Teknologi
    - **Bahasa:** Python
    - **ML:** scikit-learn (Random Forest)
    - **Web:** Streamlit
    - **Dataset:** Kaggle Superstore (9.994 baris)
    
    ### 📁 Fitur
    - 🏠 Overview - Statistik ringkasan
    - 📊 Visualisasi - Grafik penjualan
    - 🎯 Prediksi - Form prediksi Sales
    - ℹ️ Tentang - Info aplikasi
    
    ### 👤 Developer
    - **Nama:** Refa Aprilliana
    - **GitHub:** [@refaaprilliana235-droid](https://github.com/refaaprilliana235-droid)
    """)

st.markdown("---")
st.caption("📊 Sales Prediction App | Powered by Random Forest + Streamlit")
