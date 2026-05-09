
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Real Estate Platform",
    page_icon="🏠",
    layout="wide"
)

# ================= DATABASE =================
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

# Users Table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT,
    role TEXT
)
""")

# Prediction History Table
c.execute("""
CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    location TEXT,
    predicted_price REAL
)
""")

# Properties Table
c.execute("""
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seller TEXT,
    title TEXT,
    location TEXT,
    area REAL,
    rooms INTEGER,
    bathrooms INTEGER,
    balconies INTEGER,
    furnished TEXT,
    parking TEXT,
    power_backup TEXT,
    expected_price REAL,
    image TEXT
)
""")


conn.commit()


# ================= CREATE UPLOADS FOLDER =================
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# ================= BACKGROUND IMAGES =================
LOGIN_BG = "https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?q=80&w=2070&auto=format&fit=crop"

SIGNUP_BG = "https://images.unsplash.com/photo-1494526585095-c41746248156?q=80&w=2070&auto=format&fit=crop"

PREDICT_BG = "https://images.unsplash.com/photo-1460317442991-0ec209397118?q=80&w=2070&auto=format&fit=crop"

SIDEBAR_BG = "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?q=80&w=2070&auto=format&fit=crop"

# ================= UI DESIGN =================
def set_bg(bg_url):

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        section[data-testid="stSidebar"] {{
            background-image: url("{SIDEBAR_BG}");
            background-size: cover;
            background-position: center;
        }}

        section[data-testid="stSidebar"] > div {{
            background: rgba(15,23,42,0.90);
        }}

        .glass {{
            background: rgba(0,0,0,0.60);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 20px;
            margin-top: 20px;
        }}

        h1,h2,h3,h4,h5,p,label,span {{
            color: white !important;
        }}

        .stButton>button {{
            width: 100%;
            background: linear-gradient(135deg,#2563eb,#1d4ed8);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
        }}

        .stButton>button:hover {{
            background: linear-gradient(135deg,#1d4ed8,#172554);
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# ================= SESSION =================
if "login" not in st.session_state:
    st.session_state.login = False

if "user" not in st.session_state:
    st.session_state.user = ""

if "role" not in st.session_state:
    st.session_state.role = ""

# ================= SIGNUP =================
def signup():

    set_bg(SIGNUP_BG)

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.title("📝 Signup")

    user = st.text_input("👤 Username")
    pwd = st.text_input("🔒 Password", type="password")

    role = st.selectbox(
        "Select Role",
        ["Buyer", "Seller"]
    )

    if st.button("Create Account"):

        try:

            c.execute(
                "INSERT INTO users VALUES (?,?,?)",
                (user, pwd, role)
            )

            conn.commit()

            st.success("✅ Account Created")

        except:
            st.error("❌ User Already Exists")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= LOGIN =================
def login():

    set_bg(LOGIN_BG)

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.title("🔐 Login")

    user = st.text_input("👤 Username")
    pwd = st.text_input("🔒 Password", type="password")

    if st.button("Login"):

        res = c.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (user, pwd)
        ).fetchone()

        if res:

            st.session_state.login = True
            st.session_state.user = res[0]
            st.session_state.role = res[2]

            st.success("✅ Login Successful")

            st.rerun()

        else:
            st.error("❌ Invalid Credentials")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():

    return (
        joblib.load("models/furnished_XGBRegressor.pkl"),
        joblib.load("models/furnished_transformer.pkl"),
        joblib.load("models/unfurnished_randomforest.pkl"),
        joblib.load("models/unfurnished_transformer.pkl")
    )

# ================= FORMAT =================
def format_inr(x):
    return f"{int(x):,}"

# ================= MAIN APP =================
def app():

    furnished_model, furnished_transformer, unfurnished_model, unfurnished_transformer = load_models()

    # Sidebar
    st.sidebar.title(f"👋 {st.session_state.user}")
    st.sidebar.write(f"Role: {st.session_state.role}")

    # ================= BUYER MENU =================
    if st.session_state.role == "Buyer":

        page = st.sidebar.selectbox(
            "Navigation",
            [
                "Predict",
                "Browse Properties",
                "History",
                "Analytics",
                "About"
            ]
        )

    # ================= SELLER MENU =================
    else:

        page = st.sidebar.selectbox(
            "Navigation",
            [
                "Add Property",
                "My Listings",
                "Analytics",
                "About"
            ]
        )

    # Logout
    if st.sidebar.button("🚪 Logout"):

        st.session_state.login = False
        st.rerun()

    # ================= PREDICTION PAGE =================
    if page == "Predict":

        set_bg(PREDICT_BG)

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.title("🏙️ Flat Price Prediction")

        location = st.selectbox(
            "📍 Location",
            ["Baner", "Wakad", "Hinjewadi", "Kothrud", "Hadapsar"]
        )

        option = st.radio(
            "🏠 Flat Type",
            ["Unfurnished", "Furnished"]
        )

        rooms = st.number_input("🛏 Rooms", 1.0)
        bath = st.number_input("🚿 Bathrooms", 1.0)
        area = st.number_input("📐 Area", 100.0)
        balconies = st.number_input("🌇 Balconies", 0.0)

        new = st.radio("🏗 Property", ["Old", "New"])
        car = st.radio("🚗 Parking", ["Yes", "No"])
        power = st.radio("⚡ Power Backup", ["Yes", "No"])

        # ================= UNFURNISHED =================
        if option == "Unfurnished":

            df = pd.DataFrame({
                "balconies":[balconies],
                "bathroom":[bath],
                "Neworold":[new],
                "No of rooms":[0],
                "Area":[area],
                "Total rooms":[rooms],
                "Car Parking":[car],
                "Power Backup":[power]
            })

            if st.button("🔮 Predict Price"):

                pred = np.exp(
                    unfurnished_model.predict(
                        unfurnished_transformer.transform(df)
                    )
                )[0]

                st.success(f"💰 Predicted Price: ₹ {format_inr(pred)}")

                c.execute(
                    "INSERT INTO history VALUES (?,?,?)",
                    (st.session_state.user, location, pred)
                )

                conn.commit()

        # ================= FURNISHED =================
        else:

            AC = st.radio("❄ AC", ["Yes", "No"])
            TV = st.radio("📺 TV", ["Yes", "No"])

            df = pd.DataFrame({
                "balconies":[balconies],
                "bathroom":[bath],
                "Neworold":[new],
                "No of rooms":[0],
                "Area":[area],
                "Total rooms":[rooms],
                "Car Parking":[car],
                "Power Backup":[power],
                "AC":[AC],
                "TV":[TV],
                "Refrigerator":["Yes"],
                "Sofa":["No"],
                "Washing Machine":["Yes"],
                "Gas connection":["Yes"],
                "BED":["Yes"],
                "Wardrobe":["Yes"]
            })

            if st.button("🔮 Predict Price"):

                pred = np.exp(
                    furnished_model.predict(
                        furnished_transformer.transform(df)
                    )
                )[0]

                st.success(f"💰 Predicted Price: ₹ {format_inr(pred)}")

                c.execute(
                    "INSERT INTO history VALUES (?,?,?)",
                    (st.session_state.user, location, pred)
                )

                conn.commit()

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= BROWSE PROPERTIES =================
    elif page == "Browse Properties":

        set_bg(PREDICT_BG)

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.title("🏠 Available Properties")

        properties = c.execute(
            "SELECT * FROM properties"
        ).fetchall()

        if len(properties) == 0:
            st.warning("No properties available")

        else:

            col1, col2 = st.columns(2)

            for index, p in enumerate(properties):

                with col1 if index % 2 == 0 else col2:

                    st.markdown("""
                    <div style="
                        background: rgba(255,255,255,0.08);
                        padding:15px;
                        border-radius:15px;
                        margin-bottom:20px;
                    ">
                    """, unsafe_allow_html=True)

                    if p[12]:

                        image_path = f"uploads/{p[12]}"

                        if os.path.exists(image_path):
                            st.image(image_path, use_container_width=True)

                    st.subheader(p[2])

                    st.write(f"📍 Location: {p[3]}")
                    st.write(f"📐 Area: {p[4]} sq.ft")
                    st.write(f"🛏 Rooms: {p[5]}")
                    st.write(f"🚿 Bathrooms: {p[6]}")
                    st.write(f"🌇 Balconies: {p[7]}")
                    st.write(f"🛋 Furnished: {p[8]}")
                    st.write(f"🚗 Parking: {p[9]}")
                    st.write(f"⚡ Power Backup: {p[10]}")
                    st.write(f"💰 Price: ₹ {format_inr(p[11])}")
                    st.write(f"👤 Seller: {p[1]}")

                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= HISTORY =================
    elif page == "History":

        set_bg(PREDICT_BG)

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.title("📜 Prediction History")

        history = c.execute(
            "SELECT * FROM history WHERE username=?",
            (st.session_state.user,)
        ).fetchall()

        df = pd.DataFrame(
            history,
            columns=["User", "Location", "Price"]
        )

        st.dataframe(df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= ADD PROPERTY =================
    elif page == "Add Property":

        set_bg(PREDICT_BG)

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.title("🏠 Add Property")

        title = st.text_input("Property Title")

        location = st.selectbox(
            "Location",
            ["Baner", "Wakad", "Hinjewadi", "Kothrud"]
        )

        area = st.number_input("Area")
        rooms = st.number_input("Rooms")
        bathrooms = st.number_input("Bathrooms")
        balconies = st.number_input("Balconies")

        furnished = st.radio(
            "Furnished",
            ["Yes", "No"]
        )

        parking = st.radio(
            "Parking",
            ["Yes", "No"]
        )

        power = st.radio(
            "Power Backup",
            ["Yes", "No"]
        )

        price = st.number_input("Expected Price")

        uploaded_file = st.file_uploader(
            "Upload Property Image",
            type=["jpg", "png", "jpeg"]
        )

        if st.button("Add Property"):

            c.execute(
                """
                INSERT INTO properties (
                    seller,
                    title,
                    location,
                    area,
                    rooms,
                    bathrooms,
                    balconies,
                    furnished,
                    parking,
                    power_backup,
                    expected_price,
                    image
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    st.session_state.user,
                    title,
                    location,
                    area,
                    rooms,
                    bathrooms,
                    balconies,
                    furnished,
                    parking,
                    power,
                    price,
                    image_name
                )
            )

            conn.commit()

            st.success("✅ Property Added Successfully")

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= MY LISTINGS =================
    elif page == "My Listings":

        set_bg(PREDICT_BG)

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.title("📋 My Listings")

        properties = c.execute(
            "SELECT * FROM properties WHERE seller=?",
            (st.session_state.user,)
        ).fetchall()

        if len(properties) == 0:
            st.warning("No listings found")

        for p in properties:

            st.markdown("---")

            if p[12]:

                image_path = f"uploads/{p[12]}"

                if os.path.exists(image_path):
                    st.image(image_path, width=350)

            st.subheader(p[2])

            st.write(f"📍 {p[3]}")
            st.write(f"💰 ₹ {format_inr(p[11])}")

        st.markdown('</div>', unsafe_allow_html=True)

       # ================= ANALYTICS =================
    elif page == "Analytics":

        set_bg(PREDICT_BG)

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.title("📊 Analytics Dashboard")

        data = c.execute(
            "SELECT location, expected_price FROM properties"
        ).fetchall()

        if len(data) > 0:

            df = pd.DataFrame(
                data,
                columns=["Location", "Price"]
            )

            # Average Price
            avg_price = int(df["Price"].mean())

            # Total Listings
            total_properties = len(df)

            # Highest Price
            max_price = int(df["Price"].max())

            # Metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "🏠 Total Listings",
                total_properties
            )

            col2.metric(
                "💰 Avg Price",
                f"₹ {format_inr(avg_price)}"
            )

            col3.metric(
                "🔥 Highest Price",
                f"₹ {format_inr(max_price)}"
            )

            st.markdown("---")

            # Bar Chart
            fig1 = px.bar(
                df,
                x="Location",
                y="Price",
                title="Property Prices by Location"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            # Pie Chart
            pie = px.pie(
                df,
                names="Location",
                title="Property Distribution"
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        else:
            st.warning("No analytics data available")

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= ABOUT =================
    elif page == "About":

        set_bg(PREDICT_BG)

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.title("📖 About Project")

        st.write("""
        ## AI-Based Real Estate Platform

        Features:
        - Machine Learning Prediction
        - Buyer/Seller Modules
        - Property Listings
        - Analytics Dashboard
        - Image Upload
        - SQLite Database
        - Modern UI
        """)

        st.markdown('</div>', unsafe_allow_html=True)

# ================= MAIN FLOW =================
menu = st.sidebar.selectbox(
    "Menu",
    ["Login", "Signup"]
)

if not st.session_state.login:

    if menu == "Login":
        login()

    else:
        signup()

else:
    app()
