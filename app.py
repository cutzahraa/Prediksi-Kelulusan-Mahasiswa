import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# Bagian 1: Header Aplikasi
# ===============================
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", page_icon="🎓", layout="wide")
st.title("🎓 Prediksi Kelulusan Mahasiswa Menggunakan Random Forest")

st.markdown("""
Aplikasi ini memprediksi status kelulusan mahasiswa berdasarkan data akademik seperti IPK, jumlah SKS, IPS terakhir, 
jumlah mengulang, cuti, semester, dan lama studi.
""")

# ===============================
# Bagian 2: Pembuatan Dataset
# ===============================
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 220
    data = {
        "IPK": np.round(np.random.uniform(2.0, 4.0, n), 2),
        "Jumlah_SKS": np.random.randint(100, 150, n),
        "IPS_Terakhir": np.round(np.random.uniform(2.0, 4.0, n), 2),
        "Mengulang": np.random.randint(0, 6, n),
        "Cuti": np.random.choice(["Iya", "Tidak"], n, p=[0.3, 0.7]),
        "Semester": np.random.randint(4, 16, n),
        "Lama_Studi": np.random.randint(4, 7, n)
    }

    df = pd.DataFrame(data)
    kondisi = []
    for i in range(n):
        if df.loc[i, "IPK"] < 2.5 or df.loc[i, "Mengulang"] > 4:
            kondisi.append("Tidak Lulus")
        elif df.loc[i, "IPK"] < 3.0 or df.loc[i, "Semester"] > 10:
            kondisi.append("Lulus Tidak Tepat Waktu")
        else:
            kondisi.append("Lulus")
    df["Status"] = kondisi
    return df

df = generate_dataset()

# ===============================
# Bagian 3: Menampilkan Dataset
# ===============================
st.subheader("📊 Data Mahasiswa")
st.dataframe(df.head(10))

st.download_button("📥 Download Dataset (CSV)", df.to_csv(index=False).encode('utf-8'),
                   "dataset_kelulusan.csv", "text/csv")

# ===============================
# Bagian 4: Pemodelan Random Forest
# ===============================
df_encoded = df.copy()
df_encoded["Cuti"] = df_encoded["Cuti"].map({"Iya": 1, "Tidak": 0})
X = df_encoded.drop("Status", axis=1)
y = df_encoded["Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

st.subheader("📈 Hasil Pelatihan Model")
st.write(f"**Akurasi Model:** {akurasi * 100:.2f}%")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

# ===============================
# Bagian 5: Prediksi Manual
# ===============================
st.subheader("🎯 Coba Prediksi Kelulusan Mahasiswa Baru")

col1, col2, col3 = st.columns(3)
with col1:
    ipk = st.slider("IPK", 2.0, 4.0, 3.0, 0.01)
    jumlah_sks = st.number_input("Jumlah SKS", 100, 150, 120)
    ips_terakhir = st.slider("IPS Terakhir", 2.0, 4.0, 3.1, 0.01)
with col2:
    mengulang = st.number_input("Jumlah Mata Kuliah Mengulang", 0, 6, 1)
    cuti = st.selectbox("Pernah Cuti?", ["Tidak", "Iya"])
with col3:
    semester = st.number_input("Semester", 4, 16, 8)
    lama_studi = st.number_input("Lama Studi (tahun)", 4, 7, 4)

if st.button("Prediksi Sekarang 🚀"):
    input_data = pd.DataFrame({
        "IPK": [ipk],
        "Jumlah_SKS": [jumlah_sks],
        "IPS_Terakhir": [ips_terakhir],
        "Mengulang": [mengulang],
        "Cuti": [1 if cuti == "Iya" else 0],
        "Semester": [semester],
        "Lama_Studi": [lama_studi]
    })

    prediksi = model.predict(input_data)[0]
    st.success(f"🎓 Hasil Prediksi: **{prediksi}**")

# ===============================
# Bagian 6: Footer
# ===============================
st.markdown("---")
st.caption("© 2025 - Aplikasi Prediksi Kelulusan Mahasiswa | Dibuat dengan ❤️ menggunakan Streamlit & Random Forest")
