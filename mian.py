# salva como yolo_obj_distance_calib.py
from ultralytics import YOLO
import cv2
import math
import numpy as np
import json
import os
import time

# ---------- CONFIG ----------
MODEL_PATH = "yolo-Weights/yolov8n.pt"
CALIB_FILE = "calibration.json"
CAM_INDEX = 0
WIDTH, HEIGHT = 1920, 1080

# COCO IDS
CLASS_NAMES = {
    0: "person",
    73: "book",
    67: "cell phone",
    64: "mouse",
    56: "chair"
}

# tamanhos reais aproximados (cm) - ajuste conforme seu objeto real
REAL_SIZES_CM = {
    0: 170,  # person (altura média) - ideal calibrar com distância conhecida
    73: 24,  # book (altura típica) - ajuste conforme seu livro
    67: 15,  # cellphone (altura) - ajuste conforme seu aparelho
    64: 11,  # mouse (comprimento) - ajuste
    56: 90   # chair (altura do encosto) - ajuste
}
# ----------------------------

def load_calibration(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return {}

def save_calibration(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)

# Carrega modelo e câmera
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

model = YOLO(MODEL_PATH)

# focal_lengths (pixels) por classe -> armazenado como strings em json
focals = load_calibration(CALIB_FILE)  # ex: {"0": 1234.5, "67": 987.6}
last_dist = {}  # suavização por classe

print("Pressione 'c' para calibrar o objeto maior detectado (precisa informar distância conhecida).")
print("Pressione 'q' para sair.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    detections = []
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in CLASS_NAMES and conf > 0.25:
                detections.append({"cls": cls, "conf": conf, "box": (x1, y1, x2, y2)})

    # desenhar e calcular distância se existir calibração
    for det in detections:
        cls = det["cls"]
        name = CLASS_NAMES[cls]
        x1, y1, x2, y2 = det["box"]
        conf = det["conf"]
        color = (0,255,0) if cls == 0 else (255,0,0)  # pessoa verde, outros azul
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        obj_h_px = max(1, y2 - y1)  # evitar div por zero

        if str(cls) in focals:
            focal = float(focals[str(cls)])
            real_h = REAL_SIZES_CM.get(cls, None)
            if real_h is None:
                text = f"{name} {conf:.2f} - sem tamanho real"
            else:
                dist_cm = (real_h * focal) / obj_h_px
                # suavizar
                prev = last_dist.get(cls, None)
                if prev is None:
                    smooth = dist_cm
                else:
                    alpha = 0.35
                    smooth = prev * (1 - alpha) + dist_cm * alpha
                last_dist[cls] = smooth
                text = f"{name} {conf:.2f} {int(smooth)} cm"
        else:
            text = f"{name} {conf:.2f} - pressione 'c' p/ calibrar"

        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Deteccao + Distancia", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        # pegar maior detecção (por area)
        if not detections:
            print("Nenhum objeto detectado para calibrar. Mostre o objeto e tente novamente.")
            continue
        # escolhe a detecção com maior area
        best = max(detections, key=lambda d: (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]))
        cls = best["cls"]
        name = CLASS_NAMES[cls]
        x1, y1, x2, y2 = best["box"]
        obj_h_px = max(1, y2 - y1)
        print(f"Calibrando classe {cls} -> {name}. Altura do bbox (px): {obj_h_px}")

        # decidir tamanho real
        default_real = REAL_SIZES_CM.get(cls, None)
        if default_real:
            use_default = input(f"Tamanho real padrão para '{name}' é {default_real} cm. Usar esse valor? (y/n): ").strip().lower()
            if use_default == 'y' or use_default == '':
                real_h_cm = float(default_real)
            else:
                real_h_cm = float(input("Digite o tamanho real do objeto em cm (ex: 15): ").strip())
        else:
            real_h_cm = float(input("Digite o tamanho real do objeto em cm (ex: 15): ").strip())

        # distância conhecida
        known_dist = float(input("Digite a distância conhecida (cm) entre câmera e objeto (ex: 100): ").strip())

        # calcular focal
        focal_px = (obj_h_px * known_dist) / real_h_cm
        focals[str(cls)] = float(focal_px)
        save_calibration(CALIB_FILE, focals)
        print(f"Calibração salva: classe {cls} ({name}) -> focal = {focal_px:.2f} px")
        time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
