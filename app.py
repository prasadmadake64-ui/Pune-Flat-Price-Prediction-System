import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ================= INR FORMAT FUNCTION =================
def format_inr(amount):
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    rest = ",".join(
        [rest[max(i-2, 0):i] for i in range(len(rest), 0, -2)][::-1]
    )
    return rest + "," + last3

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Pune Flat Price Predictor",
    page_icon="🏙️",
    layout="wide"
)

# ================= LOAD MODELS =================
furnished_model = joblib.load("furnished_XGBRegressor.pkl")
furnished_transformer = joblib.load("furnished_transformer.pkl")

unfurnished_model = joblib.load("unfurnished_randomforest.pkl")
unfurnished_transformer = joblib.load("unfurnished_transformer.pkl")

# ================= CSS =================
st.markdown("""
<style>
.main { background-color: #f6f8fc; }
.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.price-box {
    background: linear-gradient(135deg,#2563eb,#1e40af);
    color: white;
    padding: 25px;
    border-radius: 14px;
    text-align: center;
    font-size: 28px;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-size: 18px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;'>🏙️ Pune Flat Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Estimate your dream flat price using AI&ML</p>", unsafe_allow_html=True)
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.title("🏡 Flat Type")
option = st.sidebar.radio("Select Flat Category", ["Unfurnished", "Furnished"])

# ================= COMMON INPUTS (FLOAT ONLY) =================
def common_inputs():
    col1, col2, col3 = st.columns(3)

    with col1:
        rooms = st.number_input("🛏️ Total Rooms", min_value=0.0, step=1.0)
        balconies = st.number_input("🌿 Balconies", min_value=0.0, step=1.0)
        area = st.number_input("📐 Area (sq.ft)", min_value=100.0)

    with col2:
        bathroom = st.number_input("🚿 Bathrooms", min_value=0.0, step=1.0)
        neworold = st.radio("🏗️ Property Type", ["Old", "New"])
        addi = st.number_input("➕ Additional Rooms", min_value=0.0, step=1.0)

    with col3:
        car = st.radio("🚗 Car Parking", ["Yes", "No"])
        power = st.radio("⚡ Power Backup", ["Yes", "No"])

    return rooms, balconies, bathroom, neworold, addi, area, car, power

# ================= UNFURNISHED =================
if option == "Unfurnished":

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏠 Unfurnished Flat Details")

    rooms, balconies, bathroom, neworold, addi, area, car, power = common_inputs()

    st.markdown("</div>", unsafe_allow_html=True)

    user = {
        "balconies": float(balconies),
        "bathroom": float(bathroom),
        "Neworold": neworold,
        "No of rooms": float(addi),
        "Area": float(area),
        "Total rooms": float(rooms),
        "Car Parking": car,
        "Power Backup": power
    }

    df = pd.DataFrame(user, index=[0])

    if st.button("💰 Predict Price"):
        transformed = unfurnished_transformer.transform(df)
        pred = unfurnished_model.predict(transformed)
        price = np.exp(pred)[0]
        formatted_price = format_inr(price)
        st.markdown( f"""<div class='price-box'> Estimated Flat Price<br>₹ {formatted_price}</div>""",
        unsafe_allow_html=True
        )
        st.balloons()
        

# ================= FURNISHED =================
else:

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🛋️ Furnished Flat Details")

    rooms, balconies, bathroom, neworold, addi, area, car, power = common_inputs()

    st.markdown("### 🧰 Furnishing Amenities")
    col4, col5, col6 = st.columns(3)

    with col4:
        AC = st.radio("❄️ AC", ["Yes", "No"])
        TV = st.radio("📺 TV", ["Yes", "No"])
        Sofa = st.radio("🛋️ Sofa", ["Yes", "No"])

    with col5:
        Refrigerator = st.radio("🧊 Refrigerator", ["Yes", "No"])
        Washing_Machine = st.radio("🧺 Washing Machine", ["Yes", "No"])
        Gas = st.radio("🔥 Gas Connection", ["Yes", "No"])

    with col6:
        Bed = st.radio("🛏️ Bed", ["Yes", "No"])
        Wardrobe = st.radio("🚪 Wardrobe", ["Yes", "No"])

    st.markdown("</div>", unsafe_allow_html=True)

    user = {
        "balconies": float(balconies),
        "bathroom": float(bathroom),
        "Neworold": neworold,
        "No of rooms": float(addi),
        "Area": float(area),
        "Total rooms": float(rooms),
        "Car Parking": car,
        "Power Backup": power,
        "AC": AC,
        "TV": TV,
        "Refrigerator": Refrigerator,
        "Sofa": Sofa,
        "Washing Machine": Washing_Machine,
        "Gas connection": Gas,
        "BED": Bed,
        "Wardrobe": Wardrobe
    }

    df = pd.DataFrame(user, index=[0])

    if st.button("💰 Predict Price"):
        transformed = furnished_transformer.transform(df)
        pred = furnished_model.predict(transformed)
        price = np.exp(pred)[0]
        formatted_price = format_inr(price)
        st.markdown( f"""<div class='price-box'> Estimated Flat Price<br>₹ {formatted_price}</div>""",
        unsafe_allow_html=True
        )
        st.balloons()
