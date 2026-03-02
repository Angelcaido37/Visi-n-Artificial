# =============================================================================
#  FRUIT VISION
#  Materia : Visión Artificial
#  Tema    : Detección · Segmentación · Clasificación de Color
# =============================================================================
#
#  TEMARIO
#  ─────────────────────────────────────────────────────────────────────────────
#  1. Cómo funciona YOLOv8 para detectar objetos en tiempo real
#  2. Qué es la segmentación por instancia y cómo obtener la máscara
#  3. Cómo analizar el color de un objeto usando el espacio de color HSV
#  4. Cómo contar objetos detectados y mostrar estadísticas en pantalla
#  5. Cómo adaptar el sistema a TU PROPIO modelo entrenado con tus imágenes
#
#  PIPELINE DEL SISTEMA:
#  ─────────────────────────────────────────────────────────────────────────────
#
#   [Cámara/Video/Imagen]
#          │
#          ▼
#   ┌─────────────────────┐
#   │  1. DETECCIÓN       │  → YOLOv8 encuentra DÓNDE está la fruta
#   │     (YOLO bbox)     │    y le asigna una clase (manzana, plátano...)
#   └────────┬────────────┘
#            │
#            ▼
#   ┌─────────────────────┐
#   │  2. SEGMENTACIÓN    │  → YOLOv8-seg dibuja el contorno EXACTO
#   │     (máscara)       │    píxel por píxel (no solo un rectángulo)
#   └────────┬────────────┘
#            │
#            ▼
#   ┌─────────────────────┐
#   │  3. CLASIFICACIÓN   │  → Analiza el color usando SOLO los píxeles
#   │     de COLOR        │    dentro de la máscara (ignora el fondo)
#   └────────┬────────────┘
#            │
#            ▼
#   ┌─────────────────────┐
#   │  4. CONTEO + UI     │  → Muestra resultados en pantalla y consola
#   └─────────────────────┘
#
#  INSTALACIÓN:
#  ─────────────────────────────────────────────────────────────────────────────
#   pip install ultralytics opencv-python numpy
#
#  USO:
#  ─────────────────────────────────────────────────────────────────────────────
#   python fruit_vision_pro.py                        # Webcam (cámara 0)
#   python fruit_vision_pro.py --source 1             # Cámara 1 (externa)
#   python fruit_vision_pro.py --source foto.jpg      # Imagen estática
#   python fruit_vision_pro.py --source video.mp4     # Archivo de video
#   python fruit_vision_pro.py --model mi_modelo.pt   # TU modelo propio
#   python fruit_vision_pro.py --conf 0.6 --save      # Umbral + guardar
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1: IMPORTACIÓN DE LIBRERÍAS
# ─────────────────────────────────────────────────────────────────────────────

import cv2
# cv2 = OpenCV (Open Computer Vision)
# Es la librería más usada en visión artificial.
# Permite: leer cámaras/videos, dibujar sobre imágenes,
#          convertir espacios de color, aplicar máscaras, etc.

import numpy as np
# numpy = librería de computación numérica
# Las imágenes en OpenCV son ARREGLOS numpy (matrices de píxeles)
# Ejemplo: una imagen 640x480 en color = array de forma (480, 640, 3)
#          donde 3 = canales B, G, R (Azul, Verde, Rojo)

import argparse
# argparse = permite leer argumentos desde la línea de comandos
# Ejemplo: python script.py --source video.mp4 --conf 0.5

import sys
# sys = sistema operativo. Usamos sys.exit() para detener el programa con error.

import time
# time = para medir el tiempo entre frames y calcular FPS (cuadros por segundo)

import os
# os = para manejar rutas de archivos y verificar si existen

from collections import defaultdict
# defaultdict = diccionario que crea automáticamente un valor por defecto
# cuando accedes a una clave que no existe. Muy útil para contadores.
# Ejemplo: contador["pera"] += 1  → no lanza error aunque "pera" no exista

from ultralytics import YOLO
# YOLO = You Only Look Once (Solo miras una vez)
# Es una red neuronal que detecta objetos en una sola pasada,
# lo que la hace muy rápida (tiempo real a 30+ FPS).
# Ultralytics es la empresa que mantiene YOLOv8.


# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN GLOBAL
# =============================================================================
# Aquí defines los parámetros del sistema. Al centralizar la configuración
# aquí, es fácil modificar el comportamiento sin tocar el resto del código.
# =============================================================================

# ─── 2A. CONFIGURACIÓN DEL MODELO ────────────────────────────────────────────
#
# OPCIÓN 1: Usar modelo pre-entrenado de YOLO (COCO dataset)
#   - COCO tiene 80 clases: personas, autos, animales, algunas frutas...
#   - Solo incluye: manzana (47), plátano (46)
#   - NO incluye: pera, naranja, uva, fresa, etc.
#
# OPCIÓN 2: Usar TU PROPIO modelo entrenado (RECOMENDADO para más frutas)
#   - Entrenas con tus propias imágenes usando Roboflow + Google Colab
#   - Puedes detectar CUALQUIER fruta que hayas fotografiado
#   - Ver instrucciones al final del archivo para entrenar tu modelo
#
# MODELOS YOLO DISPONIBLES (de más rápido a más preciso):
#   yolov8n-seg.pt  → nano   (más rápido,  menos preciso, ~7MB)
#   yolov8s-seg.pt  → small  (rápido,      buena precisión, ~22MB)
#   yolov8m-seg.pt  → medium (equilibrado, mejor precisión, ~52MB)
#   yolov8l-seg.pt  → large  (lento,       muy preciso,     ~87MB)
#   yolov8x-seg.pt  → extra  (más lento,   máxima precisión, ~136MB)

DEFAULT_MODEL = "yolov8n-seg.pt"
# ↑ Cambia esto a la ruta de TU modelo: ej. "runs/train/exp/weights/best.pt"

# ─── 2B. CLASES A DETECTAR ───────────────────────────────────────────────────
#
# Si usas el modelo COCO (pre-entrenado), estas son las clases de frutas:
#   ID 46 → "banana"   (plátano)
#   ID 47 → "apple"    (manzana)
#
# IMPORTANTE: Si usas TU PROPIO modelo, cambia estos IDs y nombres.
# En tu modelo custom, las clases empiezan desde 0 y dependen del orden
# en que definiste tus clases al entrenar.
# Ejemplo para modelo propio con 4 frutas:
#   FRUIT_CLASSES = { 0: "manzana", 1: "platano", 2: "pera", 3: "naranja" }

FRUIT_CLASSES = {
    47: "manzana",   # ID 47 en COCO = apple
    46: "platano",   # ID 46 en COCO = banana
    # Agrega más si tu modelo las tiene:
    # 49: "naranja",  # ID 49 en COCO = orange
    # 52: "platano",  # (algunos datasets tienen IDs distintos)
    #https://docs.ultralytics.com/es/datasets/detect/coco/#dataset-structure
    #https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
}

# ─── 2C. COLORES VISUALES PARA CADA FRUTA ────────────────────────────────────
# Define el color BGR que se usará para dibujar la caja y máscara de cada fruta
# BGR = Blue, Green, Red (OpenCV usa este orden, no RGB)
# Ejemplos: (0,255,0)=verde puro, (0,165,255)=naranja, (255,0,0)=azul

FRUIT_COLORS_BGR = {
    "manzana": (40, 200, 40),    # verde brillante
    "platano": (0,  210, 230),   # cyan/turquesa
    # Agrega colores para tus nuevas clases si tienes modelo propio:
    # "pera":     (100, 220, 100),
    # "naranja":  (0,   140, 255),
    # "uva":      (200,  40, 200),
}

# ─── 2D. TRANSPARENCIA DE LA MÁSCARA DE SEGMENTACIÓN ─────────────────────────
# Valor entre 0.0 y 1.0
# 0.0 = completamente transparente (no se ve la máscara)
# 1.0 = completamente opaco (tapa la imagen original)
# 0.45 = 45% de la máscara visible → buen balance visual

MASK_ALPHA = 0.45

# ─── 2E. ÍCONOS DE TEXTO PARA EL PANEL ───────────────────────────────────────
# Como no podemos usar emojis en OpenCV fácilmente,
# usamos etiquetas de texto simples

FRUIT_ICON = {
    "manzana": "[M]",   # [M] de manzana
    "platano": "[P]",   # [P] de plátano
    # Agrega íconos para tus nuevas clases:
    # "pera": "[Pe]",
}


# =============================================================================
# SECCIÓN 3: RANGOS DE COLOR EN ESPACIO HSV
# =============================================================================
# ¿Por qué HSV y no RGB/BGR?
# ─────────────────────────────────────────────────────────────────────────────
# RGB mezcla color e iluminación en los 3 canales, lo que hace difícil
# definir "rango de rojo" porque cambia con la luz.
#
# HSV separa:
#   H = Hue (Matiz)       → EL COLOR PURO  (0-179 en OpenCV)
#   S = Saturation (Sat.) → QUÉ TAN VÍVIDO (0-255, 0=gris, 255=puro)
#   V = Value (Brillo)    → QUÉ TAN CLARO  (0-255, 0=negro, 255=brillante)
#
# Esto permite definir "rojo" con solo el canal H, independiente
# de si la fruta está bien iluminada o en sombra.
#
# RUEDA DE COLORES HSV (valores H en OpenCV, escala 0-179):
#   0-10   → Rojo (inicio)
#   10-25  → Naranja
#   25-35  → Amarillo
#   35-85  → Verde
#   85-125 → Cyan/Azul
#   125-155→ Morado/Violeta
#   155-179→ Rojo (fin, el espectro es circular)
# =============================================================================

COLOR_PROFILES = {
    # Formato: "Nombre": [(lower_hsv, upper_hsv), ...]
    # Se pueden poner múltiples rangos para cubrir variaciones del mismo color.

    "Rojo": [
        # El rojo aparece en DOS extremos del espectro HSV (es circular)
        (np.array([0,   70,  50]), np.array([10,  255, 255])),   # Rojo inicio
        (np.array([160, 70,  50]), np.array([179, 255, 255])),   # Rojo fin
    ],
    "Verde": [
        (np.array([35, 50, 40]),  np.array([85, 255, 255])),     # Todos los verdes
    ],
    "Amarillo": [
        (np.array([18, 80, 80]),  np.array([34, 255, 255])),     # Amarillo puro
    ],
    "Naranja": [
        (np.array([10, 80, 60]),  np.array([20, 255, 255])),     # Naranja
    ],
    "Morado": [
        (np.array([125, 50, 30]), np.array([155, 255, 255])),    # Morado/violeta
    ],
    "Marron": [
        (np.array([8, 40, 20]),   np.array([20, 200, 140])),     # Café/marrón oscuro
    ],
    "Negro": [
        (np.array([0, 0, 0]),     np.array([179, 255, 50])),     # Muy oscuro/negro
    ],
    # Puedes agregar más colores según tus frutas:
    # "Rosa": [(np.array([150, 50, 100]), np.array([175, 255, 255]))],
}


# =============================================================================
# SECCIÓN 4: MÓDULO DE CLASIFICACIÓN DE COLOR
# =============================================================================
# Esta función es la "inteligencia de color" del sistema.
# Recibe la imagen completa y la máscara de la fruta,
# y devuelve el nombre del color dominante.
# =============================================================================

def classify_color_from_mask(frame_bgr: np.ndarray,
                              binary_mask: np.ndarray) -> tuple:
    """
    Determina el color dominante de una fruta usando SOLO sus píxeles.

    ¿Por qué usar la máscara en vez del recorte rectangular (ROI)?
    ──────────────────────────────────────────────────────────────────
    Un recorte rectangular incluye píxeles del FONDO (mesa, pared, etc.)
    que contaminan el análisis de color.
    La máscara de segmentación nos da EXACTAMENTE los píxeles de la fruta,
    lo que hace el análisis de color mucho más preciso.

    Ejemplo visual:
      ROI rectangular:     Máscara de segmentación:
      ┌──────────┐            🟢🟢🟢
      │🔵🟢🟢🔵│          🟢🟢🟢🟢🟢
      │🔵🟢🟢🔵│     →      🟢🟢🟢
      │🔵🟢🟢🔵│
      └──────────┘
      (🔵 = fondo azul que contamina el análisis)

    Args:
        frame_bgr:    Imagen completa en formato BGR (como la da OpenCV).
        binary_mask:  Máscara binaria: 255 donde está la fruta, 0 en el resto.

    Returns:
        Tupla (color_dominante: str, scores: dict)
        scores = {"Rojo": 0.65, "Verde": 0.20, ...} → fracción de píxeles
    """

    # Validación: si no hay máscara o imagen, no podemos analizar
    if binary_mask is None or frame_bgr is None:
        return "Desconocido", {}

    # PASO 1: Aplicar la máscara sobre la imagen
    # bitwise_and pone a 0 (negro) todos los píxeles donde la máscara es 0
    # Solo quedan visibles los píxeles donde la máscara es 255 (la fruta)
    masked_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=binary_mask)

    # PASO 2: Convertir de BGR a HSV
    # COLOR_BGR2HSV = conversión de espacio de color
    # Ahora cada píxel tiene (H, S, V) en lugar de (B, G, R)
    hsv = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2HSV)

    # PASO 3: Contar cuántos píxeles de la fruta tenemos (total)
    # countNonZero cuenta píxeles con valor != 0
    total_px = int(cv2.countNonZero(binary_mask))

    # Si la máscara tiene muy pocos píxeles (<30), la detección es demasiado
    # pequeña para analizar con confianza → retornar "Desconocido"
    if total_px < 30:
        return "Desconocido", {}

    # PASO 4: Para cada color definido, contar cuántos píxeles coinciden
    scores = {}  # Aquí guardaremos la "puntuación" de cada color

    for color_name, ranges in COLOR_PROFILES.items():
        # Crear una máscara vacía (todo negro = 0)
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Recorrer los rangos HSV de este color
        # (el rojo tiene 2 rangos porque aparece en ambos extremos del espectro)
        for (lower_hsv, upper_hsv) in ranges:
            # inRange: crea máscara donde el píxel está DENTRO del rango HSV
            # Resultado: 255 si está en rango, 0 si no
            rango_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            # bitwise_or combina los rangos: si el píxel está en cualquiera
            # de los rangos, se marca como 255
            color_mask = cv2.bitwise_or(color_mask, rango_mask)

        # IMPORTANTE: Intersectar con la máscara de la fruta
        # Esto asegura que solo contamos píxeles que SON de la fruta
        color_mask = cv2.bitwise_and(color_mask, binary_mask)

        # Contar píxeles que coinciden con este color
        count = int(cv2.countNonZero(color_mask))

        # Calcular la fracción: ¿qué % de la fruta es este color?
        scores[color_name] = count / total_px

    # PASO 5: El color dominante es el que tiene mayor fracción de píxeles
    dominant = max(scores, key=scores.get)

    # PASO 6: Verificar que el color dominante cubre al menos el 8% de la fruta
    # Si ningún color cubre ese mínimo, la iluminación es mala o el color
    # no está en nuestros rangos definidos
    if scores[dominant] < 0.08:
        dominant = "Desconocido"

    return dominant, scores


# =============================================================================
# SECCIÓN 5: MÓDULO DE SEGMENTACIÓN Y OVERLAY VISUAL
# =============================================================================
# La segmentación por instancia es una de las capacidades más avanzadas
# de YOLOv8. En vez de solo dar un rectángulo (bounding box), da el
# contorno exacto de cada objeto como una lista de puntos (polígono).
# =============================================================================

def apply_segmentation_overlay(frame: np.ndarray,
                                contour_pts: np.ndarray,
                                fruit_name: str,
                                alpha: float = MASK_ALPHA) -> np.ndarray:
    """
    Dibuja la máscara de segmentación semitransparente sobre el frame.

    ¿Cómo funciona la transparencia?
    ──────────────────────────────────
    Usamos "alpha blending" (mezcla de imágenes):
        resultado = overlay * alpha + original * (1 - alpha)
    Con alpha=0.45:
        resultado = (máscara coloreada * 0.45) + (original * 0.55)
    → La fruta se ve de un color pero todavía se pueden ver sus detalles.

    Args:
        frame:        Frame original (se modifica in-place).
        contour_pts:  Array numpy con los puntos del polígono (x,y).
        fruit_name:   Nombre de la fruta (para elegir el color).
        alpha:        Nivel de transparencia de la máscara (0.0-1.0).
    """

    # Copiar el frame para poder hacer el blend después
    overlay = frame.copy()

    # Obtener el color BGR asignado a esta fruta
    bgr = FRUIT_COLORS_BGR.get(fruit_name, (180, 180, 180))

    # Necesitamos al menos 3 puntos para formar un polígono válido
    if contour_pts is not None and len(contour_pts) >= 3:

        # Reshape: convertir de (N, 2) a (N, 1, 2) que es lo que pide OpenCV
        # astype(np.int32): los píxeles son enteros, no decimales
        pts = contour_pts.reshape((-1, 1, 2)).astype(np.int32)

        # fillPoly: rellena el polígono con el color sólido en la copia
        cv2.fillPoly(overlay, [pts], bgr)

        # addWeighted: mezcla la copia (con polígono) y el original
        # Parámetros: (src1, alpha1, src2, alpha2, gamma, dst)
        # gamma = 0 → sin brillo adicional
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # polylines: dibuja el contorno del polígono (línea sólida)
        # isClosed=True → cierra el polígono uniendo el último punto con el primero
        # thickness=2 → grosor de la línea en píxeles
        cv2.polylines(frame, [pts], isClosed=True, color=bgr, thickness=2)

    return frame  # retornamos el frame modificado (aunque también se modificó in-place)


def polygon_to_mask(contour_pts: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convierte los puntos del polígono YOLOv8-seg en una máscara binaria.

    YOLOv8-seg nos da el contorno de la fruta como lista de puntos (x,y).
    Para analizar el color necesitamos una MÁSCARA BINARIA:
        - 255 (blanco) = píxel pertenece a la fruta
        - 0   (negro)  = píxel es fondo o fuera de la fruta

    Args:
        contour_pts: Array de puntos (x,y) del contorno.
        shape:       Forma del frame: (alto, ancho, canales).

    Returns:
        Máscara binaria de las mismas dimensiones que el frame.
    """

    # Crear imagen negra del mismo tamaño que el frame (solo 1 canal)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    # shape[:2] toma solo (alto, ancho), ignorando el número de canales de color

    if contour_pts is not None and len(contour_pts) >= 3:
        pts = contour_pts.reshape((-1, 1, 2)).astype(np.int32)
        # fillPoly: pinta de blanco (255) el interior del polígono
        cv2.fillPoly(mask, [pts], 255)

    return mask  # imagen binaria: blanco donde está la fruta


# =============================================================================
# SECCIÓN 6: MÓDULO DE ANOTACIONES VISUALES (dibujo sobre el frame)
# =============================================================================

def draw_label(frame: np.ndarray, x1: int, y1: int,
               fruit_name: str, color_name: str,
               conf: float, idx: int) -> None:
    """
    Dibuja la etiqueta de texto encima de la bounding box.

    La etiqueta muestra:
        #1 MANZANA  Verde  89%
        ↑  ↑        ↑      ↑
        ID Clase    Color  Confianza

    La confianza (0.0-1.0) indica qué tan seguro está el modelo
    de su detección. 0.89 = 89% de certeza.

    Args:
        frame:      Frame donde dibujar.
        x1, y1:     Esquina superior izquierda de la bounding box.
        fruit_name: Nombre de la fruta detectada.
        color_name: Color clasificado.
        conf:       Confianza de la detección (0.0-1.0).
        idx:        Número de orden de esta detección en el frame.
    """

    # Obtener color BGR de esta fruta para la etiqueta
    bgr = FRUIT_COLORS_BGR.get(fruit_name, (200, 200, 200))

    # Construir el texto de la etiqueta
    # f-string con formato: {conf:.0%} convierte 0.89 → "89%"
    label = f"#{idx} {fruit_name.upper()}  {color_name}  {conf:.0%}"

    # Parámetros de fuente
    font  = cv2.FONT_HERSHEY_SIMPLEX  # fuente estándar de OpenCV
    scale = 0.52                       # tamaño relativo del texto
    thick = 1                          # grosor de las letras en píxeles

    # getTextSize: calcula cuánto espacio ocupa el texto antes de dibujarlo
    # Retorna ((ancho, alto), baseline)
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thick)

    # Posición Y de la etiqueta: encima de la caja, pero sin salirse del frame
    # max(...) asegura que la etiqueta no quede fuera del borde superior
    ty = max(y1 - 6, text_h + 6)

    # Dibujar fondo sólido para que el texto sea legible sobre cualquier imagen
    # -1 como último parámetro = rellenar el rectángulo (no solo el borde)
    cv2.rectangle(frame,
                  (x1, ty - text_h - baseline - 2),   # esquina superior izq.
                  (x1 + text_w + 6, ty + 2),           # esquina inferior der.
                  bgr, -1)

    # Dibujar el texto sobre el fondo
    # (10, 10, 10) = casi negro → contrasta bien con colores brillantes
    # LINE_AA = anti-aliasing (bordes suaves, mejor calidad visual)
    cv2.putText(frame, label, (x1 + 3, ty),
                font, scale, (10, 10, 10), thick, cv2.LINE_AA)


def draw_bbox(frame: np.ndarray, x1: int, y1: int,
              x2: int, y2: int, fruit_name: str) -> None:
    """
    Dibuja la bounding box (rectángulo delimitador) con esquinas decorativas.

    Una bounding box (caja delimitadora) es el rectángulo más pequeño
    que encierra completamente al objeto detectado.
    Las coordenadas son (x1,y1) = esquina sup-izq, (x2,y2) = esquina inf-der.
    """

    # Obtener color de la fruta
    bgr = FRUIT_COLORS_BGR.get(fruit_name, (200, 200, 200))

    # Dibujar rectángulo: grosor 2 píxeles
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

    # Dibujar "esquinas en L" para un estilo visual más moderno
    L = 14  # longitud de cada segmento de la esquina en píxeles

    # Las 4 esquinas con su dirección de crecimiento:
    # (sx, sy) = punto inicial, (dx, dy) = dirección (+1 o -1)
    corners = [
        (x1, y1,  1,  1),  # Esquina superior izquierda → crece hacia der/abajo
        (x2, y1, -1,  1),  # Esquina superior derecha   → crece hacia izq/abajo
        (x1, y2,  1, -1),  # Esquina inferior izquierda → crece hacia der/arriba
        (x2, y2, -1, -1),  # Esquina inferior derecha   → crece hacia izq/arriba
    ]

    for sx, sy, dx, dy in corners:
        # Línea horizontal de la esquina
        cv2.line(frame, (sx, sy), (sx + dx * L, sy), bgr, 3)
        # Línea vertical de la esquina
        cv2.line(frame, (sx, sy), (sx, sy + dy * L), bgr, 3)


# =============================================================================
# SECCIÓN 7: PANEL DE ESTADÍSTICAS (Dashboard lateral)
# =============================================================================
# El dashboard muestra en tiempo real el conteo y colores detectados.
# Se dibuja como un panel semitransparente en el lado derecho del frame.
# =============================================================================

def draw_dashboard(frame: np.ndarray, counters: dict,
                   color_detail: dict, fps: float) -> None:
    """
    Dibuja el panel de estadísticas en la esquina derecha del frame.

    Muestra:
        - Nombre del sistema
        - Conteo por tipo de fruta
        - Barra proporcional de colores detectados
        - Lista de colores con conteo
        - Total de frutas detectadas
        - FPS actual
    """

    h, w = frame.shape[:2]  # obtener alto y ancho del frame actual
    panel_w = 270            # ancho del panel en píxeles
    margin  = 8              # margen interno del panel

    # ── Fondo semitransparente ────────────────────────────────────────────────
    # Técnica de alpha blending para fondo oscuro semitransparente:
    # 1. Copiar el frame
    overlay = frame.copy()
    # 2. Dibujar rectángulo sólido negro en la copia
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (15, 15, 15), -1)
    # 3. Mezclar: 72% del overlay oscuro + 28% del frame original
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Línea vertical separadora entre el video y el panel
    cv2.line(frame, (w - panel_w, 0), (w - panel_w, h), (60, 60, 60), 1)

    # ── Función auxiliar para escribir texto ─────────────────────────────────
    # Definida aquí dentro para no tener que pasar 'frame' como argumento
    def txt(text, x, y, scale=0.50, color=(220, 220, 220), bold=False):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if bold else 1,  # bold=True → grosor 2
                    cv2.LINE_AA)

    # Posición X de inicio del texto (margen desde el borde del panel)
    x0 = w - panel_w + margin
    y  = 28  # posición Y actual (se va incrementando hacia abajo)

    # ── Título del panel ──────────────────────────────────────────────────────
    txt("FRUIT VISION PRO", x0, y, 0.55, (100, 230, 255), bold=True)
    y += 6
    # Línea horizontal decorativa bajo el título
    cv2.line(frame, (x0, y), (w - margin, y), (60, 60, 60), 1)
    y += 18

    # ── Estadísticas por fruta ────────────────────────────────────────────────
    total = 0  # acumulador del total de frutas en este frame

    for fruit, count in counters.items():
        total += count  # sumar al total

        bgr  = FRUIT_COLORS_BGR.get(fruit, (200, 200, 200))
        icon = FRUIT_ICON.get(fruit, "[ ]")

        # Nombre de la fruta e ícono
        txt(f"{icon} {fruit.upper()}", x0, y, 0.58, bgr, bold=True)
        # Número de detecciones (alineado a la derecha del panel)
        txt(f"{count}", w - margin - 20, y, 0.70, (255, 255, 255), bold=True)
        y += 18

        # ── Barra proporcional de colores ─────────────────────────────────────
        # Muestra visualmente qué colores se detectaron y en qué proporción
        bar_w = panel_w - 2 * margin  # ancho total de la barra
        bar_h = 6                      # alto de la barra en píxeles

        # Fondo gris oscuro de la barra
        cv2.rectangle(frame, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)

        # Rellenar la barra con segmentos de color proporcionales
        fruit_colors = color_detail.get(fruit, {})  # .get() evita KeyError
        x_offset = 0  # posición X dentro de la barra
        for color_name, cnt in sorted(fruit_colors.items(), key=lambda kv: -kv[1]):
            # Calcular ancho proporcional: si count=4 y cnt=3 → 75% del ancho
            seg_w = int(bar_w * cnt / max(count, 1))
            col_bgr = _color_name_to_bgr(color_name)
            cv2.rectangle(frame,
                          (x0 + x_offset, y),
                          (x0 + x_offset + seg_w, y + bar_h),
                          col_bgr, -1)
            x_offset += seg_w  # avanzar la posición

        y += bar_h + 4  # bajar el cursor

        # ── Lista de colores con conteo ───────────────────────────────────────
        for color_name, cnt in sorted(fruit_colors.items(), key=lambda kv: -kv[1]):
            col_bgr = _color_name_to_bgr(color_name)
            # Pequeño círculo de color como viñeta
            cv2.circle(frame, (x0 + 5, y - 1), 4, col_bgr, -1)
            txt(f"  {color_name}: {cnt}", x0 + 4, y + 3, 0.42, (190, 190, 190))
            y += 16  # separación entre líneas

        y += 8  # espacio entre frutas

    # ── Total y FPS ───────────────────────────────────────────────────────────
    cv2.line(frame, (x0, y), (w - margin, y), (60, 60, 60), 1)
    y += 18
    txt("TOTAL", x0, y, 0.54, (200, 200, 200))
    txt(str(total), w - margin - 20, y, 0.72, (255, 255, 100), bold=True)
    y += 24

    # FPS = Frames Per Second (cuadros por segundo procesados)
    # Un buen valor en tiempo real es >15 FPS; webcam normal es 30 FPS
    if fps > 0:
        txt(f"FPS: {fps:.1f}", x0, y, 0.42, (100, 100, 100))

    # Instrucciones al pie del panel
    txt("Q / ESC - Salir", x0, h - 10, 0.38, (80, 80, 80))


def _color_name_to_bgr(name: str) -> tuple:
    """
    Convierte el nombre de un color a su valor BGR para dibujar.
    Esta tabla es SOLO para los colores del dashboard visual,
    no confundir con los rangos HSV de COLOR_PROFILES.
    """
    color_map = {
        "Rojo":     (40,  40,  220),   # rojo en BGR
        "Verde":    (40,  180,  40),   # verde
        "Amarillo": (20,  220, 220),   # amarillo
        "Naranja":  (20,  140, 240),   # naranja
        "Morado":   (200,  40, 180),   # morado
        "Marron":   (30,   80, 130),   # marrón
        "Negro":    (60,   60,  60),   # gris oscuro (negro puro no se ve)
    }
    return color_map.get(name, (160, 160, 160))  # gris si no se encuentra


# =============================================================================
# SECCIÓN 8: PIPELINE PRINCIPAL — PROCESAMIENTO DE UN FRAME
# =============================================================================
# Esta función es el CORAZÓN del sistema.
# Recibe un frame (imagen), lo pasa por todo el pipeline y retorna
# el frame anotado con todos los resultados.
#
# Se llama para CADA FRAME del video o cámara (30+ veces por segundo)
# =============================================================================

def process_frame(frame: np.ndarray, model: YOLO,
                  conf_threshold: float = 0.45):
    """
    Ejecuta el pipeline completo en un solo frame.

    Pipeline:
        Frame entrada
            → Inferencia YOLO (detección + segmentación)
            → Para cada objeto detectado:
                → Obtener máscara de segmentación
                → Clasificar color usando la máscara
                → Contabilizar
                → Dibujar anotaciones
            → Retornar frame anotado + estadísticas

    Args:
        frame:           Imagen numpy BGR (como la da cv2.VideoCapture).
        model:           Modelo YOLO ya cargado en memoria.
        conf_threshold:  Mínima confianza para aceptar una detección (0.0-1.0).

    Returns:
        (frame_anotado, counters, color_detail)
    """

    # Inicializar contadores del frame actual
    # defaultdict(int) → al acceder a clave nueva, crea contador en 0
    counters     = defaultdict(int)
    # defaultdict anidado → al acceder a fruta nueva, crea otro defaultdict(int)
    color_detail = defaultdict(lambda: defaultdict(int))

    # ── PASO 1: INFERENCIA DEL MODELO ─────────────────────────────────────────
    # Aquí ocurre la "magia" de la IA. El modelo procesa el frame y
    # retorna todos los objetos que encontró.
    #
    # Parámetros:
    #   verbose=False    → no imprimir resultados en consola cada frame
    #   conf=threshold   → solo aceptar detecciones con confianza > umbral
    #   classes=[...]    → SOLO buscar las frutas que nos interesan (más rápido)
    #
    # NOTA: Si usas modelo PROPIO, quita el parámetro `classes`
    #       porque tus IDs de clase son diferentes.
    #       En su lugar: results = model(frame, verbose=False, conf=conf_threshold)[0]

    if FRUIT_CLASSES:
        # Modelo COCO: filtrar por IDs de clase conocidos
        results = model(frame,
                        verbose=False,
                        conf=conf_threshold,
                        classes=list(FRUIT_CLASSES.keys()))[0]
    else:
        # Modelo propio sin filtro de clases (detecta todo lo que aprendió)
        results = model(frame, verbose=False, conf=conf_threshold)[0]

    # results[0] → tomamos el primer (y único) resultado
    # results.boxes → lista de objetos detectados con sus coordenadas
    # results.masks → máscaras de segmentación (si el modelo -seg las provee)

    h, w = frame.shape[:2]  # dimensiones del frame para clamp de coordenadas

    # ── PASO 2: ITERAR CADA OBJETO DETECTADO ──────────────────────────────────
    for i, box in enumerate(results.boxes):
        # i    = índice del objeto (0, 1, 2...)
        # box  = objeto con coordenadas, clase, y confianza

        # Obtener el ID de clase de esta detección
        cls_id = int(box.cls[0])
        # box.cls[0] es un tensor de PyTorch → int() lo convierte a entero Python

        # Verificar si es una de las clases que nos interesan
        if cls_id not in FRUIT_CLASSES:
            continue  # saltar este objeto si no es una de nuestras frutas

        # Obtener nombre de la fruta desde el diccionario
        fruit_name = FRUIT_CLASSES[cls_id]

        # Confianza: qué tan seguro está el modelo (0.0 a 1.0)
        confidence = float(box.conf[0])

        # ── BOUNDING BOX (rectángulo delimitador) ──────────────────────────────
        # box.xyxy[0] = [x1, y1, x2, y2] en formato float
        # x1,y1 = esquina superior izquierda
        # x2,y2 = esquina inferior derecha
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Clamp: asegurar que las coordenadas no salgan del frame
        # (puede pasar si el objeto está en el borde de la imagen)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # ── SEGMENTACIÓN (máscara de instancia) ───────────────────────────────
        # results.masks contiene los polígonos de segmentación
        # .xy = lista de arrays con puntos (x,y) del contorno para cada objeto
        seg_pts     = None   # puntos del polígono
        binary_mask = None   # máscara binaria resultante

        if results.masks is not None and i < len(results.masks.xy):
            # Obtener los puntos del polígono para este objeto
            seg_pts = np.array(results.masks.xy[i], dtype=np.float32)

            if len(seg_pts) >= 3:  # mínimo 3 puntos para un polígono válido
                # Dibujar la máscara coloreada semitransparente sobre el frame
                apply_segmentation_overlay(frame, seg_pts, fruit_name)

                # Convertir polígono → máscara binaria para análisis de color
                binary_mask = polygon_to_mask(seg_pts, frame.shape)

        # ── FALLBACK: máscara elíptica si no hay segmentación ─────────────────
        # Si el modelo no provee máscaras (modelo de detección puro, no -seg),
        # creamos una máscara elíptica basada en el bounding box.
        # Es menos precisa pero mejor que nada.
        if binary_mask is None:
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            cx = (x1 + x2) // 2  # centro X de la caja
            cy = (y1 + y2) // 2  # centro Y de la caja
            rx = max(1, (x2 - x1) // 2)  # radio horizontal (mitad del ancho)
            ry = max(1, (y2 - y1) // 2)  # radio vertical   (mitad del alto)
            # Dibujar elipse rellena de blanco (255)
            cv2.ellipse(binary_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        # ── CLASIFICACIÓN DE COLOR ─────────────────────────────────────────────
        # Ahora que tenemos la máscara, podemos clasificar el color
        dominant_color, _scores = classify_color_from_mask(frame, binary_mask)
        # dominant_color = "Rojo", "Verde", etc.
        # _scores = {"Rojo": 0.65, "Verde": 0.20, ...} (no lo usamos aquí)

        # ── CONTABILIZACIÓN ────────────────────────────────────────────────────
        counters[fruit_name] += 1          # +1 al contador de esta fruta
        color_detail[fruit_name][dominant_color] += 1  # +1 al color específico
        local_idx = counters[fruit_name]   # número de orden de esta detección

        # ── DIBUJAR ANOTACIONES EN EL FRAME ───────────────────────────────────
        draw_bbox(frame, x1, y1, x2, y2, fruit_name)             # rectángulo
        draw_label(frame, x1, y1, fruit_name,                     # etiqueta
                   dominant_color, confidence, local_idx)

    # ── PASO 3: GARANTIZAR ENTRADAS PARA TODAS LAS FRUTAS ─────────────────────
    # Asegurar que counters y color_detail tienen una entrada por cada
    # fruta conocida, aunque no se haya detectado ninguna en este frame.
    # Esto evita KeyError en draw_dashboard.
    for f in FRUIT_CLASSES.values():
        if f not in counters:
            counters[f] = 0           # 0 detecciones
        if f not in color_detail:
            color_detail[f] = defaultdict(int)  # sin colores

    return frame, dict(counters), dict(color_detail)


# =============================================================================
# SECCIÓN 9: CARGA DE MODELO (con soporte para modelo propio)
# =============================================================================

def load_model(model_path: str) -> YOLO:
    """
    Carga el modelo YOLO desde disco o lo descarga automáticamente.

    Si model_path es un nombre como "yolov8n-seg.pt", Ultralytics lo
    descarga automáticamente de internet si no existe localmente.

    Si model_path es una ruta como "runs/train/exp/weights/best.pt",
    carga TU modelo entrenado personalmente.

    Args:
        model_path: Nombre del modelo o ruta al archivo .pt

    Returns:
        Modelo YOLO cargado y listo para inferencia.
    """

    print(f"[INFO] Cargando modelo: {model_path}")

    # Verificar si es un modelo local (archivo .pt existente en disco)
    if os.path.isfile(model_path):
        print(f"[INFO] ✔ Modelo local encontrado en: {model_path}")
    else:
        # Si no existe localmente, Ultralytics intentará descargarlo
        print(f"[INFO] Modelo no encontrado localmente. Descargando...")
        print(f"       (Solo la primera vez, se guarda en ~/.cache/ultralytics/)")

    # YOLO() carga el modelo. Si es nombre de YOLO oficial, descarga automático.
    model = YOLO(model_path)

    # Obtener información del modelo
    # model.names es un dict: {0: "clase0", 1: "clase1", ...}
    print(f"[INFO] ✔ Modelo cargado.")
    print(f"[INFO]   Clases del modelo: {model.names}")
    print(f"[INFO]   Número de clases:  {len(model.names)}")

    return model


def configure_classes_from_model(model: YOLO) -> None:
    """
    Actualiza FRUIT_CLASSES automáticamente basándose en el modelo cargado.

    Si usas tu propio modelo entrenado con, por ejemplo:
        Clase 0 = "manzana"
        Clase 1 = "platano"
        Clase 2 = "pera"
        Clase 3 = "naranja"

    Esta función detecta si el modelo tiene clases diferentes a COCO
    y REEMPLAZA FRUIT_CLASSES con las clases del modelo propio.

    Esto hace el código completamente adaptable sin modificar nada más.
    """
    global FRUIT_CLASSES, FRUIT_COLORS_BGR, FRUIT_ICON

    # Clases COCO conocidas que NO son frutas (para filtrar)
    # El modelo COCO tiene 80 clases; las frutas son solo 46, 47, 49, 52
    COCO_NON_FRUIT_IDS = set(range(80)) - {46, 47, 49, 50, 51, 52, 53}

    model_classes = model.names  # {0: "nombre", 1: "nombre", ...}

    # Heurística: si el modelo tiene pocas clases (≤20), probablemente es
    # un modelo CUSTOM enfocado en frutas → usar TODAS sus clases
    if len(model_classes) <= 20:
        print("\n[INFO] ⚡ Modelo CUSTOM detectado (≤20 clases)")
        print("[INFO]    Usando TODAS las clases del modelo:")

        # Paleta de colores BGR para asignar automáticamente
        auto_colors = [
            (40,  200,  40),    # verde
            (0,   210, 230),    # cyan
            (0,   140, 255),    # naranja
            (200,  40, 200),    # morado
            (40,  200, 200),    # amarillo-verde
            (255, 100, 100),    # azul claro
            (100, 255, 200),    # menta
            (200, 100, 255),    # rosa
        ]

        # Reemplazar la configuración global con las clases del modelo
        FRUIT_CLASSES     = {}
        FRUIT_COLORS_BGR  = {}
        FRUIT_ICON        = {}

        for class_id, class_name in model_classes.items():
            FRUIT_CLASSES[class_id]    = class_name
            color_idx = class_id % len(auto_colors)
            FRUIT_COLORS_BGR[class_name] = auto_colors[color_idx]
            FRUIT_ICON[class_name]      = f"[{class_name[:2].upper()}]"
            print(f"         ID {class_id:2d} → {class_name}")

        print()
    else:
        # Modelo COCO estándar → mantener configuración manual
        print(f"[INFO] Modelo COCO estándar ({len(model_classes)} clases).")
        print(f"[INFO] Detectando solo: {list(FRUIT_CLASSES.values())}")


# =============================================================================
# SECCIÓN 10: FLUJO PRINCIPAL DEL PROGRAMA
# =============================================================================

def run(source, model_path: str, conf: float = 0.45,
        save_output: bool = False):
    """
    Función principal que orquesta todo el sistema.

    Decide si la entrada es imagen, video o webcam,
    carga el modelo, y ejecuta el pipeline frame por frame.
    """

    print("\n" + "═" * 65)
    print("  FRUIT VISION PRO  — Sistema de Visión Artificial")
    print("  Pipeline: Detección → Segmentación → Clasificación de Color")
    print("═" * 65)

    # ── Cargar modelo ──────────────────────────────────────────────────────────
    model = load_model(model_path)

    # ── Configurar clases según el modelo cargado ──────────────────────────────
    # Esto adapta automáticamente el sistema al modelo (COCO o propio)
    configure_classes_from_model(model)

    # ── Determinar el tipo de fuente ───────────────────────────────────────────
    # Extensiones de imagen soportadas
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
    is_image = (isinstance(source, str) and
                source.lower().endswith(IMAGE_EXTENSIONS))

    # ── CASO A: IMAGEN ESTÁTICA ────────────────────────────────────────────────
    if is_image:
        print(f"\n[INFO] Modo: IMAGEN → {source}")

        # cv2.imread: carga la imagen como array numpy BGR
        frame = cv2.imread(source)
        if frame is None:
            sys.exit(f"[ERROR] No se pudo abrir la imagen: {source}")

        print(f"[INFO] Imagen cargada: {frame.shape[1]}×{frame.shape[0]} píxeles")

        # Ejecutar el pipeline completo en la imagen
        result_frame, counters, colors = process_frame(frame, model, conf)

        # Dibujar el dashboard (FPS=0 porque es imagen estática)
        draw_dashboard(result_frame, counters, colors, fps=0)

        # Imprimir reporte en consola
        _print_report(counters, colors, frames=1)

        # Guardar si se solicitó
        if save_output:
            filename  = os.path.basename(source)
            out_path  = f"resultado_{filename}"
            cv2.imwrite(out_path, result_frame)
            print(f"[INFO] Imagen guardada: {out_path}")

        # Mostrar ventana
        cv2.imshow("Fruit Vision Pro — [cualquier tecla para cerrar]", result_frame)
        cv2.waitKey(0)  # esperar indefinidamente hasta que se pulse una tecla

    # ── CASO B: VIDEO O WEBCAM ─────────────────────────────────────────────────
    else:
        # Convertir "0", "1", "2"... a entero para webcam
        # Si es ruta de video (mp4, avi, etc.) mantener como string
        src = int(source) if str(source).isdigit() else source
        print(f"\n[INFO] Modo: {'WEBCAM (cámara ' + str(src) + ')' if isinstance(src, int) else 'VIDEO → ' + str(src)}")

        # VideoCapture: abre la fuente de video
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"[ERROR] No se pudo abrir la fuente: {source}")

        # Obtener propiedades del video
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # ancho en píxeles
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # alto en píxeles
        print(f"[INFO] Resolución: {frame_w}×{frame_h}  |  Conf. umbral: {conf}")
        print("[INFO] Presiona  Q  o  ESC  para salir.\n")

        # Configurar grabación de salida (opcional)
        writer = None
        if save_output:
            # mp4v = códec de video MP4
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter("resultado_video.mp4", fourcc,
                                     20, (frame_w, frame_h))
            print("[INFO] Grabando en: resultado_video.mp4")

        # Variables para calcular FPS
        frame_count = 0   # total de frames procesados
        fps         = 0.0 # frames por segundo actual
        t0          = time.time()  # tiempo de referencia

        last_counters = {}  # último conteo (para el reporte final)
        last_colors   = {}  # últimos colores (para el reporte final)

        try:
            while True:
                # cap.read() lee el siguiente frame del video/cámara
                # ret = True si se leyó correctamente, False si terminó el video
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Fin del video.")
                    break

                frame_count += 1

                # ── PROCESAR FRAME ─────────────────────────────────────────────
                result_frame, counters, colors = process_frame(frame, model, conf)
                last_counters, last_colors = counters, colors

                # ── CALCULAR FPS ───────────────────────────────────────────────
                # Calcular FPS cada 10 frames para reducir overhead
                if frame_count % 10 == 0:
                    elapsed = time.time() - t0
                    fps     = 10 / elapsed if elapsed > 0 else 0
                    t0      = time.time()

                # ── DIBUJAR DASHBOARD ──────────────────────────────────────────
                draw_dashboard(result_frame, counters, colors, fps)

                # ── MOSTRAR EN VENTANA ─────────────────────────────────────────
                cv2.imshow("Fruit Vision Pro  [Q/ESC=Salir]", result_frame)

                # Guardar frame si se solicitó
                if writer:
                    writer.write(result_frame)

                # ── DETECTAR TECLAS ────────────────────────────────────────────
                # waitKey(1) = esperar 1ms. Si no se llama, la ventana no se actualiza.
                # & 0xFF = máscara para compatibilidad en 64-bit
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 27 = tecla ESC
                    print("[INFO] Detenido por el usuario.")
                    break

        except KeyboardInterrupt:
            # Ctrl+C en la consola
            print("\n[INFO] Interrumpido con Ctrl+C.")

        finally:
            # SIEMPRE liberar recursos, aunque haya error
            cap.release()      # liberar la cámara/video
            if writer:
                writer.release()  # finalizar el archivo de video

        # Reporte final al cerrar
        _print_report(last_counters, last_colors, frames=frame_count)

    # Cerrar todas las ventanas de OpenCV
    cv2.destroyAllWindows()
    print("[INFO] ✔ Programa finalizado.")


# =============================================================================
# SECCIÓN 11: REPORTE FINAL EN CONSOLA
# =============================================================================

def _print_report(counters: dict, color_detail: dict, frames: int) -> None:
    """
    Imprime un resumen visual en la consola con los resultados finales.
    Las barras de █ representan visualmente la cantidad detectada.
    """

    print("\n" + "═" * 65)
    print("  REPORTE FINAL DE DETECCIÓN")
    print(f"  Frames procesados: {frames}")
    print("═" * 65)

    total = 0

    for fruit, count in counters.items():
        total += count
        # Seleccionar ícono según la fruta
        icon  = "🍎" if "manzana" in fruit else ("🍌" if "plat" in fruit else "🍑")
        print(f"\n  {icon}  {fruit.upper()}  →  {count} unidades detectadas")

        # Desglose de colores con barra visual
        for color, cnt in sorted(color_detail.get(fruit, {}).items(),
                                  key=lambda kv: -kv[1]):
            bar = "█" * min(cnt, 40)  # máximo 40 caracteres de barra
            print(f"       {color:<12}  {bar}  ({cnt})")

    print("\n" + "─" * 65)
    print(f"  TOTAL FRUTAS DETECTADAS: {total}")
    print("═" * 65 + "\n")


# =============================================================================
# SECCIÓN 12: ENTRADA DEL PROGRAMA (argumentos de línea de comandos)
# =============================================================================

if __name__ == "__main__":
    # ArgumentParser: define qué argumentos acepta el programa
    parser = argparse.ArgumentParser(
        description="Fruit Vision Pro — Sistema de Visión Artificial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS DE USO:
  python fruit_vision_pro.py
  python fruit_vision_pro.py --source 1
  python fruit_vision_pro.py --source foto.jpg
  python fruit_vision_pro.py --source video.mp4 --save
  python fruit_vision_pro.py --model mi_modelo.pt --source 0
  python fruit_vision_pro.py --model best.pt --conf 0.6

CÓMO ENTRENAR TU PROPIO MODELO:
  1. Recolecta 100-300 fotos de cada fruta que quieras detectar
  2. Etiqueta las imágenes en https://roboflow.com (gratuito)
     - Crea un proyecto tipo "Instance Segmentation"
     - Sube tus fotos y dibuja los contornos
     - Exporta en formato "YOLOv8"
  3. Entrena en Google Colab (GPU gratis):
     from ultralytics import YOLO
     model = YOLO("yolov8n-seg.pt")  # partir del modelo base
     model.train(data="dataset.yaml", epochs=50, imgsz=640)
  4. El modelo entrenado queda en: runs/segment/train/weights/best.pt
  5. Úsalo con: --model runs/segment/train/weights/best.pt

MODELOS DISPONIBLES (más rápido → más preciso):
  yolov8n-seg.pt  yolov8s-seg.pt  yolov8m-seg.pt  yolov8l-seg.pt
        """)

    # --source: fuente de entrada
    # default="0" → webcam por defecto
    parser.add_argument(
        "--source",
        default="1",
        help="Fuente: 0=webcam, 1=cámara externa, ruta a imagen o video"
    )

    # --model: modelo a usar
    # Por defecto usa el modelo COCO nano de YOLOv8
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ruta al modelo .pt (default: {DEFAULT_MODEL}). "
             "Usa tu propio modelo con: --model ruta/a/best.pt"
    )

    # --conf: umbral de confianza
    # Detecciones con confianza menor a este valor se ignoran
    # Valor bajo (0.3) = detecta más pero con más errores
    # Valor alto (0.7) = detecta menos pero más confiable
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Umbral de confianza 0.0-1.0 (default: 0.45). "
             "Valores más altos = menos falsos positivos"
    )

    # --save: guardar resultado
    parser.add_argument(
        "--save",
        action="store_true",  # si se pone --save, vale True; si no, False
        help="Guardar el resultado en archivo (imagen o video)"
    )

    # Parsear los argumentos recibidos
    args = parser.parse_args()

    # Llamar a la función principal con los argumentos
    run(
        source     = args.source,
        model_path = args.model,
        conf       = args.conf,
        save_output= args.save
    )


# =============================================================================
# FIN DEL ARCHIVO
# =============================================================================
#
# RESUMEN DE LO QUE APRENDISTE:
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. YOLO (You Only Look Once):
#    Red neuronal que detecta objetos en una sola pasada hacia adelante.
#    Divide la imagen en una cuadrícula y predice cajas + clases en paralelo.
#
# 2. Segmentación por instancia:
#    Diferente a segmentación semántica: cada objeto tiene su PROPIA máscara.
#    YOLOv8-seg devuelve un polígono (lista de puntos x,y) por objeto.
#
# 3. Espacio de color HSV:
#    Más robusto que RGB para análisis de color bajo diferentes iluminaciones.
#    H=matiz (el color), S=saturación (viveza), V=valor (brillo).
#
# 4. Alpha blending:
#    Técnica para superponer imágenes con transparencia:
#    resultado = imagen1 * alpha + imagen2 * (1-alpha)
#
# 5. Pipeline de visión artificial:
#    Detección → Segmentación → Clasificación → Conteo → Visualización
#
# RECOMENDACIÓN FINAL PARA TU MODELO PROPIO:
# ─────────────────────────────────────────────────────────────────────────────
# Usa yolov8s-seg.pt como base para entrenar (no nano).
# El modelo "small" tiene mucho mejor precisión para objetos similares
# (como frutas que se parecen entre sí) sin ser demasiado lento.
# Con 150-200 imágenes por clase y 100 épocas en Colab, obtendrás
# un modelo excelente para tus frutas específicas.
# =============================================================================