from fastapi import APIRouter, UploadFile, File, Form, Body, Query, HTTPException
from fastapi.responses import FileResponse
from typing import List
from services.auth import auth_router
from services import auth, evaluations, utils
from services.auth import save_log_entry
import json
import os

router = APIRouter()
router.include_router(auth_router)



@router.post("/api/start-session")
def start_session(user_id: str = Form(...)):
    return auth.start_session(user_id)

@router.post("/api/detect-face")
def detect_face(file: UploadFile = File(...)):
    return auth.detect_face(file)


@router.post("/api/submit-evaluation")
def submit_evaluation():
    return evaluations.run_evaluation()

@router.get("/api/logs/csv")
def download_logs_csv():
    path = os.path.join("log", "log.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Log belum ada.")
    # akan mengirim file log/log.csv dengan nama file downloads: logs.csv
    return FileResponse(path=path, media_type="text/csv", filename="logs.csv")

@router.delete("/api/delete-log")
def delete_log(log_data: dict = Body(...)):
    timestamp = log_data.get("waktu")
    if not timestamp:
        return {"status": "error", "message": "Parameter 'waktu' diperlukan"}
    return auth.delete_log_entry(timestamp)

@router.get("/api/logs")
def get_logs():
    return auth.get_log_data()

@router.post("/api/save-log")
def save_log(log_data: dict = Body(...)):
    return save_log_entry(log_data)
@router.get("/api/evaluation")
def evaluation_stats(nama: str = Query(...)):
    user_eval = evaluations.get_user_evaluation(nama)
    print(json.dumps(user_eval, indent=2)[:500])

    if not user_eval:
        return {"error": "Data evaluasi tidak ditemukan"}

    return {
        "akurasi": user_eval["akurasi"],
        "precision": user_eval["precision"],
        "recall": user_eval["recall"],
        "f1": user_eval["f1"],
        "confusion_matrix": user_eval["confusion_matrix"],
        "confusion_matrix_image": user_eval["confusion_matrix_image"],
        "metrics_chart": user_eval["metrics_chart"],
        "pie_chart": user_eval["pie_chart"],
        "tsne_image": user_eval["tsne_image"],
        "keterangan": user_eval.get("keterangan", "")
    }