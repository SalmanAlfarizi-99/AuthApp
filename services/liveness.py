# import cv2
# import numpy as np
# from fastapi import UploadFile
# from fastapi.responses import JSONResponse
# import mediapipe as mp

# def check_liveness(file: UploadFile):
#     try:
#         image_bytes = file.file.read()
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         if img is None:
#             return JSONResponse(status_code=400, content={"status": "error", "message": "Gambar tidak valid"})

#         mp_face_mesh = mp.solutions.face_mesh
#         with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(img_rgb)

#             if results.multi_face_landmarks:
#                 return JSONResponse(content={"status": "live", "message": "Wajah terdeteksi, kemungkinan hidup (live)"})
#             else:
#                 return JSONResponse(content={"status": "not_live", "message": "Tidak terdeteksi wajah, kemungkinan spoof"})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
