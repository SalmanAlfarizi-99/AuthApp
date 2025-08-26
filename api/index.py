import sys
import os
from mangum import Mangum

# Tambahkan path backend supaya bisa import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app  # ambil app dari backend/main.py

handler = Mangum(app)
