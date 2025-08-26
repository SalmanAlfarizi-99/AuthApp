from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router
from fastapi.staticfiles import StaticFiles
from mangum import Mangum

app = FastAPI(title="Face Authentication Backend")
handler = Mangum(app)

# Izinkan akses dari frontend React Native (Expo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain Expo jika sudah ada
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routing utama
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Face Authentication API is running!"}

app.mount("/static", StaticFiles(directory="static"), name="static")
