import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
	page_title="Klasifikasi Jeruk Pintar",
	page_icon="icon.png",
	initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* Background utama */
.stApp {
    background-color: #fff8f0;
}

/* Header (navbar) */
[data-testid="stHeader"] {
    background-color: #4CAF50; /* hijau daun */
    color: white;
}

/* Toolbar (pojok kanan atas) */
[data-testid="stToolbar"] {
    background-color: #388E3C; /* hijau tua */
    color: white;
}

/* Tombol custom */
.stButton>button {
    background-color: #ff6f1a; /* oranye jeruk */
    color: white;
    border-radius: 10px;
    font-weight: bold;
    transition: background-color 0.3s; /* animasi halus */
}

/* Efek hover */
.stButton>button:hover {
    background-color: #388E3C; /* oranye lebih gelap saat hover */
    color: white;
}

</style>
""", unsafe_allow_html=True)


try:
	model = joblib.load("model_klasifikasi_jeruk.joblib")
	scaler = joblib.load("scaler_klasifikasi_jeruk.joblib")
except:
	st.error("Model belum tersedia")
	st.stop()

st.title(":tangerine: Klasifikasi Jeruk Pintar")
st.markdown("Prediksi jenis jeruk **Siam, Pontianak, atau Mandarin** berdasarkan diameter, berat, tebal kulit, dan kadar gula")

diameter = st.slider("Masukkan Diameter Jeruk (cm)", 0.0, 8.0, 7.5)
berat = st.slider("Masukkan Berat Jeruk (gr)", 0.0, 200.0, 145.0)
tebal_kulit = st.slider("Masukkan Tebal Jeruk (cm)", 0.0, 3.5, 3.2)
kadar_gula = st.slider("Masukkan Kadar Gula (%)", 0.0, 20.0, 12.1)

if st.button("Prediksi"):
	data_baru = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula]], columns=["diameter","berat","tebal_kulit","kadar_gula"])
	data_baru_scaled = scaler.transform(data_baru)
	hasil = model.predict(data_baru_scaled)[0]
	presentase = max(model.predict_proba(data_baru_scaled)[0])

	st.write("**Data yang Diprediksi :**")
	st.dataframe(data_baru)

	st.write("**Visualisasi Data :**")
	

	df = pd.read_csv("dataset_jeruk.csv")

	siam = df[df["jenis"]=="siam"]
	pontianak = df[df["jenis"]=="pontianak"]
	mandarin = df[df["jenis"]=="mandarin"]
	
	fig, ax = plt.subplots(figsize=(6,5))
	
	ax.scatter(siam["tebal_kulit"], siam["kadar_gula"], color="blue", s=100, alpha=0.7, label="Siam")
	ax.scatter(pontianak["tebal_kulit"], pontianak["kadar_gula"], color="red", s=100, alpha=0.7, label="Pontianak")
	ax.scatter(mandarin["tebal_kulit"], mandarin["kadar_gula"], color="orange", s=100, alpha=0.7, label="Mandarin")

	ax.scatter(tebal_kulit, kadar_gula, color="black", s=100, marker="x", label="Data Baru")
	
	ax.set_xlabel("Tebal Kulit")
	ax.set_ylabel("Kadar Gula")
	ax.set_title("Tebal Kulit vs Kadar Gula")
	ax.legend()
	ax.grid(True,linestyle="--",alpha=0.5)
	
	st.pyplot(fig)

	st.write("**Hasil Prediksi :**")
	st.success(f"Jeruk **{hasil}** dengan tingkat keyakinan **{presentase*100:.2f}%**")

st.divider()
st.caption("Dibuat dengan penuh :tangerine: oleh **Adi Setiawan**")


with st.sidebar:
	st.title("Did You Know?")

	option = st.selectbox(
		"Kategori Jeruk",
		("Jeruk Siam", "Jeruk Pontianak", "Jeruk Mandarin"),
		index=None,
		placeholder="Pilih Jeruk"
	)
	
	if option == "Jeruk Siam":
		st.success("""
		**Jeruk Siam**
    - **Diameter & Berat:** kecil–sedang (5–7 cm, 80–120 g)
    - **Kulit:** tipis, agak longgar, gampang dikupas
    - **Rasa:** manis segar tapi tidak terlalu pekat (kadar gula 9–11%)
    - **Ciri khas:** sering buat perasan, warnanya agak pucat
		""")	
	elif option == "Jeruk Pontianak":
		st.success("""
		**Jeruk Pontianak (kadang disebut jeruk manis Pontianak)**
    - **Diameter & Berat:** sedang–besar (7–9 cm, 140–190 g)
    - **Kulit:** agak tebal (3–4 mm), melekat kuat
    - **Rasa:** manis kuat (kadar gula 12–13%)
    - **Ciri khas:** sering dijadikan jeruk meja, lebih "berisi" dan berair
		""")
	elif option == "Jeruk Mandarin":
		st.success("""
		**Jeruk Mandarin**
    - **Diameter & Berat:** kecil–sedang (6–7.5 cm, 100–130 g)
    - **Kulit:** sangat tipis & longgar, mudah dikupas (paling gampang di antara ketiga jenis ini)
    - **Rasa:** sangat manis, kadar gula paling tinggi (12–13%), aromanya harum khas
    - **Ciri khas:** buahnya “imut-imut” dan sering untuk jeruk Imlek
		""")
