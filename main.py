import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import joblib
import os

class SistemDiagnosisPenyakit:
    def __init__(self):
        # Definisikan gejala dan penyakit
        self.gejala = [
            "Demam", "Batuk", "Pilek", "Sakit Tenggorokan", "Sakit Kepala", 
            "Nyeri Otot", "Mual", "Muntah", "Diare", "Ruam", 
            "Nyeri Perut", "Lelah", "Kehilangan Nafsu Makan", "Mata Merah",
            "Nyeri Sendi", "Sesak Napas", "Berkeringat Malam", "Berat Badan Turun"
        ]
        
        self.penyakit = ["Flu Biasa", "Demam Berdarah", "Tifus", "COVID-19", "Bronkitis"]
        
        # Siapkan penjelasan berbasis aturan untuk setiap penyakit
        # BARIS 99-100 DI SINI (CONTOH UNTUK "Flu Biasa")
        self.penjelasan_penyakit = {
            "Flu Biasa": {
                "deskripsi": "Infeksi umum pada saluran pernapasan atas yang disebabkan oleh virus influenza.",
                "penyebab": "Virus influenza yang menyebar melalui udara dan kontak dengan penderita.",
                "pengobatan": "Istirahat yang cukup, banyak minum air, obat pereda demam/nyeri, dan dekongestan.",
                "high_symptoms": ["Demam", "Batuk", "Pilek", "Sakit Tenggorokan", "Nyeri Otot"],
                "medium_symptoms": ["Sakit Kepala", "Lelah"],
                "low_symptoms": ["Mual"]
            },
            "Demam Berdarah": {
                "deskripsi": "Penyakit infeksi yang disebabkan oleh virus dengue dan ditularkan melalui gigitan nyamuk Aedes aegypti.",
                "penyebab": "Virus dengue yang ditularkan melalui gigitan nyamuk Aedes aegypti.",
                "pengobatan": "Tidak ada pengobatan spesifik, terapi suportif dengan rehidrasi cairan dan pemantauan trombosit.",
                "high_symptoms": ["Demam", "Nyeri Otot", "Nyeri Sendi", "Sakit Kepala", "Ruam"],
                "medium_symptoms": ["Mual", "Muntah", "Lelah", "Mata Merah"],
                "low_symptoms": ["Nyeri Perut", "Kehilangan Nafsu Makan"]
            },
            "Tifus": {
                "deskripsi": "Infeksi bakteri serius yang menyebar melalui makanan atau air yang terkontaminasi.",
                "penyebab": "Bakteri Salmonella typhi yang menyebar melalui makanan dan minuman yang terkontaminasi.",
                "pengobatan": "Antibiotik seperti ciprofloxacin, ceftriaxone, atau azithromycin, dan terapi suportif.",
                "high_symptoms": ["Demam", "Sakit Kepala", "Nyeri Perut", "Lelah", "Kehilangan Nafsu Makan"],
                "medium_symptoms": ["Mual", "Diare", "Berat Badan Turun"],
                "low_symptoms": ["Ruam", "Batuk"]
            },
            "COVID-19": {
                "deskripsi": "Penyakit pernapasan menular yang disebabkan oleh virus SARS-CoV-2.",
                "penyebab": "Virus SARS-CoV-2 yang menyebar melalui droplet pernapasan atau kontak dengan permukaan yang terkontaminasi.",
                "pengobatan": "Tergantung pada keparahan gejala, dari istirahat total sampai perawatan intensif medis.",
                "high_symptoms": ["Demam", "Batuk", "Sesak Napas", "Lelah", "Sakit Tenggorokan"],
                "medium_symptoms": ["Sakit Kepala", "Nyeri Otot", "Kehilangan Nafsu Makan"],
                "low_symptoms": ["Diare", "Ruam"]
            },
            "Bronkitis": {
                "deskripsi": "Peradangan pada saluran bronkial yang membawa udara dari dan ke paru-paru.",
                "penyebab": "Virus, bakteri, atau paparan iritan seperti asap rokok atau polusi.",
                "pengobatan": "Istirahat, banyak minum air, obat pereda batuk, inhaler untuk memudahkan pernapasan.",
                "high_symptoms": ["Batuk", "Sesak Napas", "Lelah", "Demam ringan"],
                "medium_symptoms": ["Sakit Dada", "Produksi dahak"],
                "low_symptoms": ["Sakit Kepala", "Sakit Tenggorokan"]
            }
        }
        
        # Inisialisasi model dan scaler
        self.model = None
        self.scaler = StandardScaler()
    
    def load_model(self):
        """Memuat model dan scaler dari file jika tersedia"""
        if os.path.exists('diagnosis_model.h5') and os.path.exists('scaler.pkl'):
            try:
                self.model = tf.keras.models.load_model('diagnosis_model.h5')
                self.scaler = joblib.load('scaler.pkl')
                return True
            except Exception as e:
                print(f"Gagal memuat model: {e}")
                return False
        return False
    
    def _generate_synthetic_data(self, n_samples=1000):
        """Membuat data sintetis untuk pelatihan neural network"""
        X = np.zeros((n_samples, len(self.gejala)))
        y = np.zeros((n_samples, len(self.penyakit)))
        
        for i in range(n_samples):
            disease_idx = np.random.randint(0, len(self.penyakit))
            disease = self.penyakit[disease_idx]
            y[i, disease_idx] = 1
            explanation = self.penjelasan_penyakit[disease]
            
            for symptom in explanation["high_symptoms"]:
                if symptom in self.gejala:
                    idx = self.gejala.index(symptom)
                    X[i, idx] = np.random.uniform(0.7, 1.0)
            
            for symptom in explanation["medium_symptoms"]:
                if symptom in self.gejala:
                    idx = self.gejala.index(symptom)
                    X[i, idx] = np.random.uniform(0.3, 0.7)
            
            for symptom in explanation["low_symptoms"]:
                if symptom in self.gejala:
                    idx = self.gejala.index(symptom)
                    X[i, idx] = np.random.uniform(0.0, 0.3)
            
            for j in range(len(self.gejala)):
                if X[i, j] == 0:
                    X[i, j] = np.random.uniform(0.0, 0.1)
        
        return X, y
    
    def train_model(self):
        """Melatih neural network dengan data sintetis menggunakan k-fold cross-validation"""
        X, y = self._generate_synthetic_data(n_samples=5000)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        fold_no = 1
        
        best_accuracy = 0
        best_model = None
        best_scaler = None
        
        for train_idx, test_idx in kfold.split(X, y):
            print(f"\nMelatih pada fold {fold_no}...")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            model = Sequential([
                Dense(64, activation='relu', input_shape=(len(self.gejala),)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(len(self.penyakit), activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            print(f"Fold {fold_no} Classification Report:")
            print(classification_report(y_true_classes, y_pred_classes, target_names=self.penyakit))
            print(f"Fold {fold_no} Confusion Matrix:")
            print(confusion_matrix(y_true_classes, y_pred_classes))
            
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Fold {fold_no} Test accuracy: {accuracy:.4f}")
            accuracies.append(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_scaler = scaler
            
            fold_no += 1
        
        self.model = best_model
        self.scaler = best_scaler
        
        self.model.save('diagnosis_model.h5')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        print("\nRingkasan Cross-Validation:")
        print(f"Rata-rata akurasi: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
        
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
        plt.title('Akurasi per Fold (Cross-Validation)')
        plt.xlabel('Fold')
        plt.ylabel('Akurasi')
        plt.grid(True)
        plt.show()
        
        return np.mean(accuracies)
    
    def predict(self, symptoms_input):
        """Memprediksi penyakit berdasarkan gejala yang diinput"""
        if self.model is None:
            raise ValueError("Model belum dilatih. Panggil train_model() terlebih dahulu.")
        
        X = np.zeros((1, len(self.gejala)))
        for symptom, value in symptoms_input.items():
            if symptom in self.gejala:
                idx = self.gejala.index(symptom)
                X[0, idx] = value
        
        X = self.scaler.transform(X)
        prediction = self.model.predict(X, verbose=0)[0]
        
        # Pastikan probabilitas dinormalisasi
        prediction = prediction / np.sum(prediction) if np.sum(prediction) > 0 else prediction
        
        results = {}
        for i, disease in enumerate(self.penyakit):
            results[disease] = float(prediction[i])
        
        return results
    
    def explain_diagnosis(self, prediction_results, symptoms_input):
        """Memberikan penjelasan berbasis aturan untuk hasil diagnosis dengan analisis gejala"""
        sorted_results = sorted(prediction_results.items(), key=lambda x: x[1], reverse=True)
        
        # Periksa probabilitas ekstrem
        max_prob = sorted_results[0][1]
        if max_prob > 0.99:
            explanation = {
                "diagnosis": "Uncertain",
                "message": f"Model terlalu yakin ({max_prob:.2%}) pada satu penyakit, yang mungkin menunjukkan overfitting. Silakan konsultasikan ke dokter."
            }
            return explanation
        
        if len(sorted_results) < 2 or sorted_results[0][1] - sorted_results[1][1] >= 0.1:
            max_disease = sorted_results[0][0]
            max_prob = sorted_results[0][1]
            
            input_symptoms = [symptom for symptom, value in symptoms_input.items() if value > 0.5]
            matching_symptoms = [
                symptom for symptom in input_symptoms 
                if symptom in self.penjelasan_penyakit[max_disease]["high_symptoms"] or 
                   symptom in self.penjelasan_penyakit[max_disease]["medium_symptoms"]
            ]
            non_matching_symptoms = [
                symptom for symptom in input_symptoms 
                if symptom not in self.penjelasan_penyakit[max_disease]["high_symptoms"] and 
                   symptom not in self.penjelasan_penyakit[max_disease]["medium_symptoms"]
            ]
            
            explanation = {
                "diagnosis": max_disease,
                "probability": max_prob,
                "explanation": self.penjelasan_penyakit[max_disease],
                "matching_symptoms": matching_symptoms,
                "non_matching_symptoms": non_matching_symptoms
            }
        else:
            top_disease, top_prob = sorted_results[0]
            second_disease, second_prob = sorted_results[1]
            explanation = {
                "diagnosis": "Uncertain",
                "message": f"Diagnosis tidak pasti antara {top_disease} ({top_prob:.2%}) dan {second_disease} ({second_prob:.2%}). Silakan konsultasikan ke dokter."
            }
        
        return explanation


class DiagnosisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Diagnosis Penyakit")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        self.sistem = SistemDiagnosisPenyakit()
        self.create_widgets()
        self.train_model()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        title_label = ttk.Label(main_frame, text="Sistem Diagnosis Penyakit", font=("Helvetica", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        symptoms_frame = ttk.LabelFrame(main_frame, text="Gejala", padding=10)
        symptoms_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.symptom_sliders = {}
        for i, symptom in enumerate(self.sistem.gejala):
            label = ttk.Label(symptoms_frame, text=symptom)
            label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
            slider = ttk.Scale(symptoms_frame, from_=0, to=1, orient="horizontal", length=200)
            slider.grid(row=i, column=1, padx=5, pady=2)
            slider.set(0)
            
            value_label = ttk.Label(symptoms_frame, text="0.0")
            value_label.grid(row=i, column=2, padx=5, pady=2)
            
            slider.configure(command=lambda val, lbl=value_label: lbl.configure(text=f"{float(val):.1f}"))
            self.symptom_sliders[symptom] = slider
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        diagnose_button = ttk.Button(button_frame, text="Diagnosa", command=self.diagnose)
        diagnose_button.pack(side="left", padx=5)
        
        reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_symptoms)
        reset_button.pack(side="left", padx=5)
        
        results_frame = ttk.LabelFrame(main_frame, text="Hasil Diagnosa", padding=10)
        results_frame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.result_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=40, height=30)
        self.result_text.pack(fill="both", expand=True)
        self.result_text.config(state="disabled")
        
        disclaimer_frame = ttk.Frame(main_frame)
        disclaimer_frame.grid(row=4, column=0, columnspan=2, pady=10)
        disclaimer_label = ttk.Label(
            disclaimer_frame,
            text="Peringatan: Sistem ini hanya untuk referensi dan bukan pengganti diagnosis dokter profesional.",
            foreground="red",
            wraplength=760
        )
        disclaimer_label.pack()
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def train_model(self):
        """Melatih model atau memuat model yang sudah ada"""
        self.root.config(cursor="wait")
        self.root.update()
        
        if self.sistem.load_model():
            messagebox.showinfo("Model Dimuat", "Model berhasil dimuat dari file.")
        else:
            try:
                accuracy = self.sistem.train_model()
                messagebox.showinfo("Pelatihan Selesai", f"Model berhasil dilatih dengan rata-rata akurasi: {accuracy:.2%}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal melatih model: {str(e)}")
        
        self.root.config(cursor="")
    
    def get_symptom_values(self):
        """Mendapatkan nilai gejala dari slider"""
        symptom_values = {}
        for symptom, slider in self.symptom_sliders.items():
            symptom_values[symptom] = float(slider.get())
        return symptom_values
    
    def reset_symptoms(self):
        """Reset semua slider gejala"""
        for slider in self.symptom_sliders.values():
            slider.set(0)
        
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled")
    
    def validate_symptoms(self, symptom_values):
        """Memvalidasi input gejala"""
        significant_symptoms = sum(1 for value in symptom_values.values() if value > 0.3)
        if significant_symptoms < 2:
            return False, "Harap masukkan setidaknya dua gejala dengan nilai di atas 0.3 untuk diagnosis yang lebih akurat."
        return True, ""
    
    def diagnose(self):
        """Melakukan diagnosa berdasarkan gejala yang diinput"""
        try:
            symptom_values = self.get_symptom_values()
            
            is_valid, error_message = self.validate_symptoms(symptom_values)
            if not is_valid:
                messagebox.showwarning("Input Tidak Valid", error_message)
                return
            
            prediction_results = self.sistem.predict(symptom_values)
            
            # Verifikasi bahwa probabilitas berjumlah mendekati 1
            prob_sum = sum(prediction_results.values())
            if not 0.99 <= prob_sum <= 1.01:
                messagebox.showerror("Error", f"Probabilitas tidak valid (total: {prob_sum:.2f}). Silakan coba lagi atau latih ulang model.")
                return
            
            explanation = self.sistem.explain_diagnosis(prediction_results, symptom_values)
            self.display_results(prediction_results, explanation)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal melakukan diagnosa: {str(e)}")
    
    def display_results(self, prediction_results, explanation):
        """Menampilkan hasil diagnosa"""
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        
        if explanation['diagnosis'] == "Uncertain":
            self.result_text.insert(tk.END, "DIAGNOSA TIDAK PASTI\n\n", "title")
            self.result_text.insert(tk.END, explanation['message'] + "\n\n")
        else:
            self.result_text.insert(tk.END, "HASIL DIAGNOSA\n\n", "title")
            self.result_text.insert(tk.END, f"Diagnosa Utama: {explanation['diagnosis']}\n")
            self.result_text.insert(tk.END, f"Probabilitas: {explanation['probability']:.2%}\n\n")
            
            self.result_text.insert(tk.END, "Kemungkinan Penyakit:\n", "subtitle")
            sorted_results = sorted(prediction_results.items(), key=lambda x: x[1], reverse=True)
            for disease, prob in sorted_results:
                self.result_text.insert(tk.END, f"- {disease}: {prob:.2%}\n")
            
            self.result_text.insert(tk.END, "\nPENJELASAN\n\n", "title")
            self.result_text.insert(tk.END, f"Deskripsi:\n{explanation['explanation']['deskripsi']}\n\n")
            self.result_text.insert(tk.END, f"Penyebab:\n{explanation['explanation']['penyebab']}\n\n")
            self.result_text.insert(tk.END, f"Pengobatan:\n{explanation['explanation']['pengobatan']}\n\n")
            
            self.result_text.insert(tk.END, "Gejala Utama:\n", "subtitle")
            for symptom in explanation['explanation']['high_symptoms']:
                self.result_text.insert(tk.END, f"- {symptom}\n")
            
            self.result_text.insert(tk.END, "\nAnalisis Gejala yang Diinput:\n", "subtitle")
            if explanation['matching_symptoms']:
                self.result_text.insert(tk.END, "Gejala yang Cocok:\n")
                for symptom in explanation['matching_symptoms']:
                    self.result_text.insert(tk.END, f"- {symptom}\n")
            if explanation['non_matching_symptoms']:
                self.result_text.insert(tk.END, "\nGejala yang Tidak Cocok:\n")
                for symptom in explanation['non_matching_symptoms']:
                    self.result_text.insert(tk.END, f"- {symptom}\n")
        
        self.result_text.tag_configure("title", font=("Helvetica", 12, "bold"))
        self.result_text.tag_configure("subtitle", font=("Helvetica", 10, "bold"))
        
        self.result_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = DiagnosisGUI(root)
    root.mainloop()