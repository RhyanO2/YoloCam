# face_obj_distance_height.py
from ultralytics import YOLO
import cv2
import json
import os
import time

# ---------- CONFIG ----------
MODEL_FACE = "yolo-Weights/yolov8n-face-lindevs.pt"
MODEL_OBJ  = "yolo-Weights/yolov8n.pt"
CALIB_FILE = "calibration_rect.json"
CAM_INDEX = 0
WIDTH, HEIGHT = 1280, 720

CLASS_NAMES = {
    "face": "Face",
    67: "Cell Phone",
    39: "Bottle"
}

REAL_SIZES_CM = {
    "face": 24,
    67: 15,
    39: 25
}
# ----------------------------

def load_calibration(file):
    return json.load(open(file)) if os.path.exists(file) else {}

def save_calibration(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

# Inicializa câmera e modelos
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

model_face = YOLO(MODEL_FACE)
model_obj  = YOLO(MODEL_OBJ)

focals = load_calibration(CALIB_FILE)
last_dist = {}
prev_time = time.time()

print("Pressione 'c' para calibrar objetos ou face.")
print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []

    # ---- Detecta rosto ----
    for r in model_face(frame, stream=True):
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.6:
                # Reduz bbox 20% nas bordas para evitar cabelo/ombros
                h = y2 - y1
                y_center = y1 + h // 2
                new_h = int(h * 0.8)
                y1_new = max(0, y_center - new_h // 2)
                y2_new = min(HEIGHT, y_center + new_h // 2)
                detections.append({"cls": "face", "conf": conf, "box": (x1, y1_new, x2, y2_new)})

    # ---- Detecta objetos ----
    for r in model_obj(frame, stream=True):
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in CLASS_NAMES and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"cls": cls, "conf": conf, "box": (x1, y1, x2, y2)})

    # ---- Desenha e calcula distâncias e altura real ----
    for det in detections:
        cls = det["cls"]
        name = CLASS_NAMES[cls] if cls in CLASS_NAMES else str(cls)
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        color = (0,255,0) if cls=="face" else (255,0,0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        obj_h_px = max(1, y2 - y1)
        real_h_default = REAL_SIZES_CM[cls] if isinstance(cls,int) else REAL_SIZES_CM["face"]

        # Distância
        if str(cls) in focals:
            focal = focals[str(cls)]
            dist_cm = (real_h_default * focal) / obj_h_px
            dist_cm = last_dist.get(cls, dist_cm) * 0.65 + dist_cm * 0.35
            last_dist[cls] = dist_cm
            # Altura real baseada na distância e bbox
            real_height_cm = (obj_h_px * dist_cm) / focal
            text = f"{name} {conf:.2f} - dist {int(dist_cm)} cm - altura {real_height_cm:.1f} cm"
        else:
            text = f"{name} {conf:.2f} - pressione 'c' p/ calibrar"

        y_text = max(10, y1-10)
        cv2.putText(frame, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ---- FPS ----
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Face + Objetos + Altura e Distancia", frame)
    key = cv2.waitKey(1) & 0xFF

    # ---- Sair ----
    if key==ord('q'):
        break

    # ---- Calibração ----
    if key==ord('c'):
        for det in detections:
            cls = det["cls"]
            x1, y1, x2, y2 = det["box"]
            obj_h_px = max(1, y2 - y1)
            real_h_default = REAL_SIZES_CM[cls] if isinstance(cls,int) else REAL_SIZES_CM["face"]
            known_dist = float(input(f"Digite distância conhecida da câmera para {CLASS_NAMES[cls]} (cm): ").strip())
            focal_px = (obj_h_px * known_dist) / real_h_default
            focals[str(cls)] = focal_px
        save_calibration(CALIB_FILE, focals)
        print("✅ Calibração salva.")

cap.release()
cv2.destroyAllWindows()
