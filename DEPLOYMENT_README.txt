StockPilot - paket artefak deployment (dihasilkan oleh StockPilot_Analytics.ipynb)
Isi: models/ (model.pkl, feature_columns.pkl, model_metadata.json), outputs/ (laporan evaluasi), sample_data/,
     stockpilot_features.py + utils/ (modul pendukung identik dengan aplikasi).
Cara pakai: salin folder models/ dan outputs/ ke folder proyek StockPilot (menimpa isi lama), lalu jalankan: streamlit run app.py
Pastikan versi scikit-learn aplikasi = versi saat training (lihat models/model_metadata.json -> environment).
