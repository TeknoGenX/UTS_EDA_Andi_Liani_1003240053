# LAPORAN UJIAN TENGAH SEMESTER (UTS)
**Mata Kuliah:** Machine Learning (IKD-3204)  
**Topik:** Exploratory Data Analysis (EDA) dan Visual Data Mining  
**Nama:** Andi Liani  
**Tanggal:** 23 Mei 2026

---

## BAGIAN A: EDA DATASET KREDIT BANK (PYTHON)

### SOAL 1 — AUDIT AWAL DATASET

#### A. Hasil Audit
- **Dimensi Dataset:** 1020 baris and 21 kolom.
- **Penggunaan Memori:** ~0.17 MB.
- **Tipe Data:** Dominan numerik (float64, int64) dan beberapa kategorikal (object) seperti `kota_domisili` dan `tipe_pekerjaan`.
- **Data Duplikat:** 20 baris (1.96%).
- **Missing Values:** Ditemukan pada kolom `skor_kredit`, `utilisasi_kartu`, `jumlah_kartu_kredit`, `frekuensi_login_bulanan`, dan `pendapatan_pasangan`.

#### B. Pertanyaan Analisis
1. **Mengapa analisis missing values harus dilakukan sebelum tahap modeling?**  
   Analisis missing values sangat krusial karena sebagian besar algoritma machine learning (seperti Linear Regression atau SVM) tidak dapat menangani nilai kosong secara otomatis dan akan menghasilkan error. Selain itu, pola kehilangan data (missingness) dapat memberikan insight apakah data hilang secara acak (MCAR) atau memiliki bias tertentu yang jika diabaikan akan menurunkan akurasi dan validitas generalisasi model.

2. **Apa dampak data duplikat terhadap performa model machine learning?**  
   Data duplikat dapat menyebabkan *overfitting*, di mana model memberikan bobot berlebih pada observasi yang berulang tersebut. Hal ini juga dapat menyebabkan kebocoran data (*data leakage*) jika salinan data yang sama muncul di kedua set training dan testing, sehingga metrik evaluasi (seperti akurasi) terlihat sangat tinggi secara semu padahal model gagal melakukan generalisasi pada data baru.

3. **Berdasarkan tipe data yang ada, kolom mana yang berpotensi membutuhkan encoding?**  
   Kolom kategorikal seperti `kota_domisili`, `tipe_pekerjaan`, `jenis_kelamin`, dan `status_perkawinan` memerlukan encoding (seperti One-Hot Encoding atau Label Encoding). Karena model machine learning berbasis matematika hanya bisa memproses input numerik, transformasi variabel tekstual menjadi representasi angka sangat diperlukan agar fitur tersebut dapat digunakan dalam perhitungan algoritma.

---

### SOAL 2 — DATA CLEANING

#### A. Tugas Pembersihan
1. Duplikat dihapus (1020 -> 1000 baris).
2. `kota_domisili` diseragamkan (e.g., JKT/JAKARTA menjadi JAKARTA).
3. Imputasi median untuk kolom numerik utama.
4. Imputasi nilai 0 untuk `pendapatan_pasangan`.

#### B. Pertanyaan Analisis
1. **Mengapa median lebih cocok dibanding mean untuk data keuangan?**  
   Data keuangan seperti pendapatan atau skor kredit seringkali memiliki pencilan (*outliers*) yang ekstrim atau berdistribusi miring (*skewed*). Nilai mean sangat sensitif terhadap pencilan tersebut, sedangkan median lebih *robust* (tahan) karena mengambil nilai tengah, sehingga lebih merepresentasikan pusat data yang sebenarnya tanpa terdistorsi oleh angka-angka yang sangat besar atau kecil.

2. **Mengapa standardisasi nilai kategorikal sangat penting?**  
   Standardisasi nilai kategorikal memastikan konsistensi data. Jika "JAKARTA" dan "Jakarta" dianggap sebagai dua kategori berbeda oleh model, maka informasi geografis tersebut menjadi terpecah dan melemahkan kekuatan prediktif fitur tersebut. Penyeragaman format sangat penting agar frekuensi setiap kategori terhitung dengan benar dalam analisis statistik.

3. **Apa risiko imputasi sembarangan tanpa konteks bisnis?**  
   Risiko utamanya adalah introduksi bias yang menyesatkan. Misalnya, mengimputasi missing value pada "jumlah keterlambatan" dengan rata-rata bisa membuat nasabah yang sebenarnya bersih terlihat berisiko. Tanpa konteks bisnis, kita mungkin mengisi data dengan nilai yang secara logika tidak mungkin terjadi, yang pada akhirnya merusak integritas model keputusan kredit.

4. **Mengapa pendapatan_pasangan diimputasi dengan 0?**  
   Secara bisnis, kolom yang kosong pada pendapatan pasangan seringkali mengindikasikan bahwa nasabah tersebut tidak memiliki pasangan atau pasangannya tidak bekerja/tidak memiliki pendapatan. Menggunakan median akan memberikan asumsi pendapatan yang tidak realistis bagi mereka yang melajang, sehingga nilai 0 adalah representasi yang lebih aman dan logis secara finansial.

---

### SOAL 3 — FEATURE ENGINEERING

#### A. Fitur Baru
| Fitur | Deskripsi |
|-------|-----------|
| `rasio_cicilan_pendapatan` | Mengukur beban utang bulanan terhadap pemasukan (Clip 0-1). |
| `rasio_utang_aset` | Mengukur solvabilitas atau total kewajiban dibanding kekayaan (Clip 0-5). |
| `pendapatan_total` | Gabungan pendapatan nasabah dan pasangan untuk melihat kapasitas bayar rumah tangga. |
| `saldo_terpakai` | (Nilai Tambah) Nilai nominal penggunaan kartu (`utilisasi` * `limit_total`). |

#### B. Pertanyaan Analisis
1. **Mengapa feature engineering penting?**  
   Feature engineering memungkinkan kita untuk mengekstraksi informasi tersembunyi yang tidak terlihat pada data mentah. Dengan menciptakan fitur yang lebih relevan dengan target (seperti rasio), kita membantu model untuk menangkap hubungan non-linear dan logika bisnis yang lebih dalam, yang seringkali jauh lebih berdampak pada akurasi daripada sekadar optimasi parameter algoritma.

2. **Mana fitur yang paling berpotensi memprediksi gagal bayar?**  
   `rasio_utang_aset` adalah prediktor terkuat berdasarkan korelasi statistik. Secara bisnis, nasabah dengan total utang yang mendekati atau melebihi nilai asetnya memiliki risiko solvabilitas yang sangat tinggi, yang merupakan indikator utama kegagalan bayar.

3. **Keuntungan fitur rasio dibanding nilai absolut?**  
   Fitur rasio memberikan konteks proporsional. Cicilan 5 juta bagi orang berpendapatan 10 juta (rasio 0.5) jauh lebih berisiko daripada cicilan 5 juta bagi orang berpendapatan 100 juta (rasio 0.05). Nilai absolut gagal menangkap tingkat keberatan beban finansial tersebut bagi individu yang berbeda-beda skalanya.

---

### SOAL 4 — VISUALISASI EDA KOMPREHENSIF

#### Interpretasi Visual
1. **1_dist_gagal_bayar.png:** Menunjukkan ketimpangan kelas (*class imbalance*) yang signifikan di mana mayoritas nasabah berstatus lancar. Hal ini menandakan diperlukannya metrik evaluasi seperti F1-Score daripada akurasi biasa.
2. **2_dist_skor_kredit.png:** Distribusi skor kredit untuk nasabah gagal bayar (orange) justru sedikit lebih bergeser ke arah skor yang lebih tinggi. Ini menunjukkan adanya anomali pada data atau skor kredit yang ada belum mampu menangkap risiko gagal bayar secara akurat pada segmen ini.
3. **3_boxplot_usia.png:** Distribusi usia antara kedua kelas relatif serupa, menunjukkan bahwa usia bukan merupakan pembeda risiko yang dominan dalam dataset ini.
4. **4_korelasi_bar.png:** `rasio_utang_aset` dan `total_utang` menunjukkan korelasi positif tertinggi terhadap gagal bayar, memperkuat asumsi bahwa beban utang terhadap kekayaan adalah faktor risiko utama.
5. **5_missing_values.png:** Memvisualisasikan kolom mana saja yang memerlukan perhatian sebelum proses pembersihan; kolom dengan persentase tinggi harus ditangani dengan hati-hati agar tidak menghilangkan informasi penting.
6. **6_heatmap_korelasi.png:** Menunjukkan hubungan antar fitur; terlihat `pendapatan_total` berkorelasi sangat kuat dengan `pendapatan_bulanan`, yang mengindikasikan adanya potensi multikolinearitas jika keduanya digunakan bersamaan.
7. **7_scatter_util_skor.png:** Memperlihatkan pola penggunaan kartu terhadap skor kredit. Nasabah dengan utilisasi tinggi tersebar di kedua kelas, namun konsentrasi risiko terlihat pada kombinasi utang yang tinggi.

---

### SOAL 5 — ANALISIS RISIKO KREDIT

- **Fitur Protektif:** `pendapatan_total` dan `utilisasi_kartu` (yang rendah). Nasabah dengan pendapatan total tinggi memiliki bantalan finansial yang lebih kuat untuk menghadapi guncangan ekonomi.
- **Fitur Risiko:** `rasio_utang_aset`, `total_utang`, dan `jumlah_keterlambatan_12bln`. Nasabah yang memiliki beban utang tinggi terhadap aset adalah kelompok yang paling rentan gagal bayar.
- **Kesimpulan:** Hubungan antara skor kredit dan gagal bayar menunjukkan pola yang tidak konvensional (positif lemah). Nasabah dengan 'rasio_utang_aset' yang tinggi harus menjadi fokus utama pengawasan risiko.

---

### SOAL 6 — EXECUTIVE SUMMARY

(Detail laporan telah disusun secara profesional dalam file terpisah `executive_summary.txt` dan dirangkum di sini untuk manajemen: Fokus pada penanganan class imbalance dan penggunaan metrik AUC-ROC untuk validasi model. Penelitian lanjutan merekomendasikan strategi berbasis segmen untuk akurasi yang lebih tinggi.)

---

## PENELITIAN LANJUTAN (ADDENDUM)

Sebagai nilai tambah, telah dilakukan analisis mendalam menggunakan teknik Advanced Machine Learning:

1. **Anomaly Detection (Isolation Forest):**
   - Menemukan bahwa gagal bayar lebih baik dideteksi sebagai "penyimpangan profil" (anomali) daripada kelas klasifikasi standar. Berhasil meningkatkan *Recall* hingga 20% (4x lipat dari model XGBoost standar).

2. **Customer Segmentation (K-Means Clustering):**
   - **Cluster 0 (High Risk):** Nasabah dengan rasio utang/aset > 0.90. Memiliki default rate tertinggi (17.8%). Rekomendasi: Pengetatan monitoring.
   - **Cluster 3 (Low Risk):** Nasabah berpendapatan tinggi dengan beban utang terkontrol. Memiliki default rate terendah (12.0%). Rekomendasi: Upselling produk premium.

3. **Kesimpulan Teknis:**
   - Data tradisional memiliki *noise* yang tinggi. Bank disarankan menggunakan pendekatan segmentasi dan deteksi anomali untuk mitigasi risiko yang lebih akurat dan efisien.

---

## BAGIAN B: VISUAL DATA MINING DENGAN ORANGE

### SOAL 7 — EKSPLORASI AWAL
- **Keunggulan Orange:** Sangat efisien untuk *Rapid Prototyping*. User dapat melihat distribusi data dan korelasi secara instan tanpa perlu menulis baris kode yang panjang.
- **Keterbatasan:** Kurang fleksibel untuk kustomisasi logika fitur yang sangat kompleks atau otomasi pipeline data skala besar yang membutuhkan integrasi sistem produksi.

### SOAL 8 — DISTRIBUSI & BOX PLOT
- **Pola Bimodal:** Histogram premi tahunan menunjukkan dua puncak karena adanya perbedaan profil risiko yang sangat kontras antara nasabah perokok dan non-perokok.
- **Perbedaan Median:** Nasabah perokok memiliki median premi yang jauh lebih tinggi sebagai kompensasi atas risiko kesehatan yang lebih besar.

### SOAL 9 — SCATTER PLOT & CLUSTERING
- **Faktor Dominan:** Status merokok dan BMI adalah faktor yang paling mempengaruhi besaran premi dalam scatter plot.
- **Karakteristik Cluster:** Cluster 3 (Usia Tua, BMI Tinggi) adalah kelompok risiko tertinggi yang membutuhkan premi paling mahal bagi perusahaan asuransi.

### SOAL 10 — MODELING & EVALUASI
- **AUC-ROC:** Penting karena memberikan gambaran menyeluruh tentang performa model tanpa terpengaruh oleh distribusi kelas yang tidak seimbang.
- **Recall vs Precision:** Recall lebih kritis. Melewatkan nasabah berisiko tinggi (False Negative) bisa menyebabkan kerugian klaim yang tidak terduga bagi perusahaan, yang biayanya jauh lebih besar daripada sekadar salah menargetkan nasabah (False Positive).

---
**Pekerjaan telah diselesaikan sesuai instruksi UTS.**
