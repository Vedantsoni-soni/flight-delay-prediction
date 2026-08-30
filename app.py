import streamlit as st
import pickle
import datetime

st.set_page_config(page_title="FlyCheck", layout="wide")

# Wallpaper
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05");
    background-size: cover;
}
.stApp::before {
    content: ""; position: fixed; top:0; left:0; width:100%; height:100%;
    background: rgba(0,0,0,0.6); z-index: -1;
}
h1, p, label { color: white!important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>✈️ FlyCheck - Accurate Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ab random nahi, Model se prediction</p>", unsafe_allow_html=True)

# Model Load
model = pickle.load(open('flight_model.pkl', 'rb'))
le_air = pickle.load(open('le_air.pkl', 'rb'))
le_src = pickle.load(open('le_src.pkl', 'rb'))
le_dest = pickle.load(open('le_dest.pkl', 'rb'))

col1, col2, col3 = st.columns(3)
with col1:
    airline = st.selectbox("Airline", ["IndiGo", "Air India", "SpiceJet", "Vistara"])
    source = st.selectbox("Source", ["DEL", "BOM", "BLR", "HYD"])
with col2:
    destination = st.selectbox("Destination", ["BOM", "DEL", "BLR", "HYD"])
    dep_time = st.time_input("Departure Time", value=datetime.time(12, 0))
with col3:
    stops = st.selectbox("Stops", ["Non-Stop", "1 Stop"])
    stops_val = 0 if stops == "Non-Stop" else 1

if st.button("Predict Accurate Delay 🚀", use_container_width=True):
    a = le_air.transform([airline])[0]
    s = le_src.transform([source])[0]
    d = le_dest.transform([destination])[0]
    hour = dep_time.hour

    pred = model.predict([[a, s, d, stops_val, hour]])[0]

    if pred < 15:
        st.success(f"✅ On Time! Estimated Delay: {int(pred)} min")
    elif pred < 45:
        st.warning(f"⚠️ Slight Delay: {int(pred)} min")
    else:
        st.error(f"❌ High Delay: {int(pred)} min")