# Sistem Diagnosis Penyakit

**Sistem Diagnosis Penyakit** adalah aplikasi berbasis machine learning yang dirancang untuk membantu mendiagnosis penyakit berdasarkan gejala yang diinput oleh pengguna. Aplikasi ini memanfaatkan model **neural network** yang dilatih dengan **data sintetis**, serta dilengkapi dengan **antarmuka grafis (GUI)** berbasis Tkinter agar mudah digunakan.

> ⚠️ **Peringatan:** Sistem ini hanya untuk tujuan edukasi dan referensi. Diagnosis akhir tetap harus dikonfirmasi oleh dokter profesional.

---

## Fitur

- **Diagnosis Berbasis Gejala:** Prediksi penyakit seperti Flu Biasa, Demam Berdarah, Tifus, COVID-19, dan Bronkitis.
- **Antarmuka Pengguna Interaktif:** GUI dengan slider untuk menginput intensitas gejala dan tampilan hasil diagnosis yang informatif.
- **Penjelasan Berbasis Aturan:** Deskripsi penyakit, penyebab, pengobatan, serta analisis gejala yang cocok atau tidak cocok.
- **Pelatihan Model Otomatis:** Menggunakan neural network dan k-fold cross-validation untuk meningkatkan akurasi.
- **Data Sintetis:** Model dilatih dengan data sintetis berbasis aturan gejala-penyakit.

---

## 🛠️ Prasyarat

- Python 3.8 atau lebih tinggi
- Dependensi yang tercantum dalam `requirements.txt`

---

## 🚀 Instalasi

### 1. Clone Repositori

```bash
git clone https://github.com/Zenn711/disease-diagnosis-system.git
cd disease-diagnosis-system
````

### 2. (Opsional) Buat Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
python main.py
```

---

## 📖 Cara Penggunaan

* Jalankan aplikasi dengan perintah `python main.py`.
* Aplikasi akan melatih model atau memuat model yang sudah ada (`diagnosis_model.h5` dan `scaler.pkl`).
* Di GUI:

  * Gunakan **slider** untuk menentukan intensitas gejala (skala 0.0 hingga 1.0).
  * Klik **Diagnosa** untuk melihat hasil prediksi.
  * Klik **Reset** untuk mengatur ulang input gejala.

**Hasil diagnosis** akan mencakup:

* Prediksi penyakit dan probabilitasnya.
* Deskripsi penyakit, penyebab, dan pengobatan yang disarankan.
* Analisis gejala yang cocok dan tidak cocok dengan diagnosis.

---

## 📁 Struktur Proyek

```
disease-diagnosis-system/
├── main.py              # Kode utama aplikasi
├── requirements.txt     # Daftar dependensi
├── .gitignore           # File yang diabaikan oleh Git
├── README.md            # Dokumentasi proyek
```

---

## 📦 Dependensi

Dependensi utama:

* `numpy`: Operasi numerik
* `pandas`: Manipulasi data
* `tensorflow`: Pembuatan dan pelatihan neural network
* `scikit-learn`: Preprocessing dan evaluasi model
* `matplotlib`: Visualisasi akurasi training
* `joblib`: Penyimpanan dan pemuatan scaler
* `tk`: Antarmuka GUI

---

## 🧠 Catatan Pengembangan

* **Data Sintetis:** Digunakan untuk pelatihan awal; untuk akurasi lebih baik disarankan memakai data klinis.
* **Kustomisasi:** Tambah penyakit atau gejala dengan mengubah daftar di kelas `SistemDiagnosisPenyakit`.
* **Overfitting Alert:** Jika prediksi terlalu yakin (>99%), sistem akan memberi peringatan dan mendorong konsultasi medis.

---

## 🤝 Kontribusi

Kontribusi sangat disambut! Cara berkontribusi:

1. Fork repositori ini
2. Buat branch baru: `git checkout -b fitur-baru`
3. Commit perubahan: `git commit -m "Menambahkan fitur baru"`
4. Push ke branch: `git push origin fitur-baru`
5. Buat Pull Request di GitHub

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License**.

---

## 📬 Kontak

Jika ada pertanyaan atau saran, hubungi **\Harits** melalui email **\haritsnaufal479@gmail.com** atau buka *issue* di repositori ini.

