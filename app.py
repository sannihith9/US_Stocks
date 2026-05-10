import streamlit as st
import pandas as pd
import hashlib

# ---------------------------
# ⚙️ Page Config
# ---------------------------
st.set_page_config(page_title="Stock Master", layout="wide")

# ---------------------------
# 🔐 Login Functions
# ---------------------------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Users
usernames = ["sanni", "sunny", "Admin"]
passwords = [
    make_hashes("Sunny@123"),
    make_hashes("Data@123"),
    make_hashes("Admin@123")
]

# ---------------------------
# 🚫 Hide Sidebar Before Login
# ---------------------------
if not st.session_state.get("logged_in", False):
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

# ---------------------------
# 🎨 Global Styling + Footer
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}

/* Header */
.header {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    color: white;
    margin-bottom: 10px;
}

/* Footer */
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    font-size: 12px;
    color: white;
    background-color: rgba(0,0,0,0.2);
}

/* Reduce top spacing */
.block-container {
    padding-top: 2rem;
}

/* Button style */
.stButton button {
    border-radius: 8px;
    height: 42px;
    background-color: #1f4037;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🔐 Login Page
# ---------------------------
def login_page():
    st.markdown('<div class="header">📊 Stock Master</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        st.caption("🔐 Login to continue")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login_btn = st.button("Login", use_container_width=True)

        if login_btn:
            if username in usernames:
                user_index = usernames.index(username)
                if check_hashes(password, passwords[user_index]):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
            else:
                st.error("❌ Invalid username")

    # Footer
    st.markdown("""
        <div class="footer">
        © 2026 Stock Master | Built for Learning 📚
        </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 📊 Main App
# ---------------------------
def main_app():
    # Header
    st.markdown('<div class="header">📊 Stock Master Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.success(f"Welcome {st.session_state.get('username')} 👋")

        if st.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state.pop("username", None)
            st.rerun()

        st.header("📂 Upload Symbols")
        uploaded = st.file_uploader("Upload CSV with 'Symbol' column", type=["csv"])

    # Main Content
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            symbols = df["Symbol"].dropna().astype(str).unique().tolist()
            st.session_state["SYMBOLS"] = symbols
            st.success(f"Loaded {len(symbols)} symbols")
            st.dataframe(pd.DataFrame({"Symbol": symbols}).head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Disclaimer ⚠️
    st.warning("""
    ⚠️ **Disclaimer**  
    This application is built strictly for **educational and informational purposes only**.  
    It does **not constitute financial advice, investment recommendation, or trading guidance**.  
    Please consult a certified financial advisor before making any investment decisions.
    """)

    # Footer
    st.markdown("""
        <div class="footer">
        © 2026 Stock Master | Educational Use Only
        </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 🚦 App Flow
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    main_app()
else:
    login_page()