# import os
# import pandas as pd
# from fpdf import FPDF
# from fastapi.responses import FileResponse
# import csv
# from datetime import datetime
# import os

# def save_log_entry(name: str, session: str, accuracy: float, status: str):
#     os.makedirs("log", exist_ok=True)
#     log_file = "log/log.csv"
#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     with open(log_file, mode='a', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow([now, name, session, f"{accuracy:.2f}%", status])

# def generate_pdf_report():
#     log_path = "log/prediction_log.csv"
#     if not os.path.exists(log_path):
#         return {"status": "error", "message": "File log tidak ditemukan."}

#     df = pd.read_csv(log_path)

#     if not {"label", "prediction"}.issubset(df.columns):
#         return {"status": "error", "message": "Kolom 'label' dan 'prediction' tidak ditemukan dalam log."}

#     # Hitung akurasi
#     total = len(df)
#     benar = (df['label'] == df['prediction']).sum()
#     salah = total - benar
#     akurasi = round((benar / total) * 100, 2)

#     # Buat PDF
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=14)
#     pdf.cell(200, 10, txt="Laporan Evaluasi Otentikasi Wajah", ln=True, align='C')
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt=f"Total data: {total}", ln=True)
#     pdf.cell(200, 10, txt=f"Benar: {benar}", ln=True)
#     pdf.cell(200, 10, txt=f"Salah: {salah}", ln=True)
#     pdf.cell(200, 10, txt=f"Akurasi: {akurasi}%", ln=True)

#     # Simpan PDF
#     os.makedirs("static", exist_ok=True)
#     pdf_path = "static/report.pdf"
#     pdf.output(pdf_path)

#     # Kembalikan sebagai file response
#     return FileResponse(pdf_path, media_type='application/pdf', filename="laporan_evaluasi.pdf")
