from ultralytics import YOLO
import cv2

# Cargamos el modelo YOLO
model = YOLO("Unidad III/best.pt")

# Cargamos el video de entrada

cap = cv2.VideoCapture(1)

while cap.isOpened():
    # Leemos el frame del video
    ret, frame = cap.read()
    if not ret:
        break

    # Realizamos la inferencia de YOLO sobre el frame
    results = model(frame)

    # Extraemos los resultados
    annotated_frame = results[0].plot()
    #print(annotated_frame)

    # Visualizamos los resultados
    cv2.imshow("Detector de Objetos", annotated_frame)

    # El ciclo se rompe al presionar "Esc"
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()