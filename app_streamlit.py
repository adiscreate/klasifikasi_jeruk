import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_klasifikasi_jeruk.joblib")

st.set_page_config(
	page_title="Klasifikasi Jeruk",
	page_icon=":tangerine:"
)

st.title(":tangerine: Klasifikasi Jeruk")
st.markdown("Aplikasi machine learning untuk klasifikasi jeruk bagus, sedang atau jelek")

diameter = st.slider("Diameter", 5.0, 8.0, 6.0)
berat = st.slider("Berat", 80.0, 225.0, 100.0)
tebal_kulit = st.slider("Tebal Kulit", 0.1, 1.2, 0.4)
kadar_gula = st.slider("Kadar Gula", 5.0, 14.0, 9.0)
asal_daerah = st.pills("Asal Daerah", ["Kalimantan", "Jawa Tengah", "Jawa Barat"], default="Kalimantan")
warna = st.pills("Warna", ["hijau","kuning","oranye"], default="hijau")
musim_panen = st.pills("Musim Panen", ["hujan","kemarau"], default="hujan")

if st.button("Prediksi", type="primary"):
	data = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,warna,musim_panen]], columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen"])
	prediksi = model.predict(data)[0]
	presentase = max(model.predict_proba(data)[0])
	st.success(f"Prediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :tangerine: oleh **Adi Setiawan**")