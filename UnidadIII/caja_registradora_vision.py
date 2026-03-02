#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CAJA REGISTRADORA CON VISIÓN ARTIFICIAL                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ¿QUÉ HACE ESTE PROGRAMA?                                                   ║
║  ─────────────────────────                                                   ║
║  Simula una caja registradora que usa la cámara de tu computadora para       ║
║  *reconocer productos automáticamente* usando Inteligencia Artificial.       ║
║                                                                              ║
║  FLUJO GENERAL DEL PROGRAMA:                                                 ║
║  1. Abre la cámara y captura video en tiempo real.                           ║
║  2. Un modelo de IA (YOLO) analiza cada cuadro del video.                    ║
║  3. Si detecta un producto conocido dentro de la zona de escaneo,            ║
║     lo agrega automáticamente al carrito.                                    ║
║  4. La interfaz gráfica (Tkinter) muestra el video, el carrito y el total.  ║
║  5. Al cobrar, genera un ticket en un archivo .txt.                          ║
║                                                                              ║
║  TECNOLOGÍAS UTILIZADAS:                                                     ║
║  • OpenCV   → captura y procesamiento de imágenes                           ║
║  • YOLO     → red neuronal para detectar objetos                            ║
║  • Tkinter  → interfaz gráfica de escritorio                                ║
║  • Pillow   → convertir imágenes de OpenCV a formato Tkinter                ║
║  • threading → ejecutar la cámara en paralelo a la UI                       ║
║  • JSON     → guardar/cargar el catálogo de productos                       ║
║                                                                              ║
║  INSTALACIÓN:                                                                ║
║    pip install opencv-python ultralytics pillow numpy                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── MÓDULOS DE LA BIBLIOTECA ESTÁNDAR DE PYTHON ─────────────────────────────
# Estos vienen incluidos con Python, no necesitas instalarlos.

import cv2          # OpenCV: leer la cámara, dibujar en imágenes, procesar video
import time         # Funciones de tiempo: time.time(), time.sleep()
import threading    # Ejecutar código en paralelo (hilos / threads)
import json         # Leer y escribir archivos en formato JSON
import os           # Operaciones del sistema operativo (rutas, abrir archivos)
import datetime     # Fecha y hora actual (para el ticket y el reloj en pantalla)
import numpy as np  # NumPy: operaciones matemáticas con arrays (poco usado aquí, pero buena práctica)

# Importamos clases específicas de 'collections':
# - defaultdict: como un diccionario normal, pero si la clave no existe,
#                crea un valor por defecto automáticamente.
# - deque: lista de tamaño fijo que descarta los elementos más antiguos
#          cuando se llena (útil para ventanas de tiempo).
from collections import defaultdict, deque


# ─── IMPORTAR ULTRALYTICS (YOLO) ──────────────────────────────────────────────
# Usamos try/except para que el programa no falle si la librería no está
# instalada; solo muestra un aviso y continúa sin detección.

try:
    from ultralytics import YOLO   # YOLO = You Only Look Once, modelo de detección de objetos
    YOLO_AVAILABLE = True          # Bandera: la librería SÍ está disponible
except ImportError:
    YOLO_AVAILABLE = False         # Bandera: la librería NO está disponible
    print("[AVISO] ultralytics no instalado. Instala con: pip install ultralytics")


# ─── IMPORTAR TKINTER ─────────────────────────────────────────────────────────
# Tkinter es la librería estándar de Python para crear interfaces gráficas.
# También la envolvemos en try/except por si el entorno no la tiene.

try:
    import tkinter as tk                  # El módulo principal de Tkinter
    from tkinter import messagebox        # Ventanas de diálogo (alertas, confirmaciones)
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False
    print("[AVISO] Tkinter no disponible. Este script requiere Tkinter para la UI.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO DE PRODUCTOS
# ═══════════════════════════════════════════════════════════════════════════════
#
# El catálogo se guarda en un archivo "productos.json" en el disco.
# Si el archivo no existe, se crea con los datos por defecto que definimos aquí.
#
# ESTRUCTURA DEL DICCIONARIO:
#   clave   → nombre de la clase YOLO en inglés (ej: "apple")
#   valor   → diccionario con nombre legible, precio, emoji, categoría y código
#
# ¿Por qué en inglés? Porque YOLO fue entrenado con etiquetas en inglés.
# ───────────────────────────────────────────────────────────────────────────────

PRODUCTOS_PATH = "productos.json"   # Nombre del archivo donde se guarda el catálogo

# Catálogo por defecto: se usa cuando NO existe el archivo productos.json
DEFAULT_PRODUCTOS_DB = {
    # clave YOLO       nombre visible         precio   emoji  categoría  código interno
    "toilet paper": {"nombre": "Papel Higiénico",   "precio": 25.00,   "emoji": "🧻", "categoria": "Higiene",     "codigo": "P001"},
    "bottle":       {"nombre": "Botella de Agua",   "precio": 15.00,   "emoji": "🍶", "categoria": "Bebidas",     "codigo": "B001"},
    "cup":          {"nombre": "Taza / Vaso",        "precio": 12.00,   "emoji": "☕", "categoria": "Utensilios", "codigo": "U001"},
    "apple":        {"nombre": "Manzana",            "precio": 8.50,    "emoji": "🍎", "categoria": "Frutas",      "codigo": "F001"},
    "banana":       {"nombre": "Plátano",            "precio": 5.00,    "emoji": "🍌", "categoria": "Frutas",      "codigo": "F002"},
    "orange":       {"nombre": "Naranja",            "precio": 7.00,    "emoji": "🍊", "categoria": "Frutas",      "codigo": "F003"},
    "book":         {"nombre": "Libro",              "precio": 120.00,  "emoji": "📚", "categoria": "Papelería",   "codigo": "L001"},
    "cell phone":   {"nombre": "Teléfono Celular",   "precio": 4500.00, "emoji": "📱", "categoria": "Electrónica", "codigo": "E001"},
    "keyboard":     {"nombre": "Teclado",            "precio": 350.00,  "emoji": "⌨️", "categoria": "Electrónica", "codigo": "E002"},
    "mouse":        {"nombre": "Mouse",              "precio": 200.00,  "emoji": "🖱️", "categoria": "Electrónica", "codigo": "E003"},
    "scissors":     {"nombre": "Tijeras",            "precio": 30.00,   "emoji": "✂️", "categoria": "Papelería",   "codigo": "P002"},
    "pen":          {"nombre": "Bolígrafo",          "precio": 8.00,    "emoji": "🖊️", "categoria": "Papelería",   "codigo": "P003"},
    "remote":       {"nombre": "Control Remoto",     "precio": 80.00,   "emoji": "📡", "categoria": "Electrónica", "codigo": "E004"},
    "clock":        {"nombre": "Reloj",              "precio": 250.00,  "emoji": "🕐", "categoria": "Accesorios",  "codigo": "A001"},
    "vase":         {"nombre": "Jarrón",             "precio": 150.00,  "emoji": "🏺", "categoria": "Hogar",       "codigo": "H001"},
    "sandwich":     {"nombre": "Sándwich",           "precio": 45.00,   "emoji": "🥪", "categoria": "Alimentos",   "codigo": "AL001"},
}


def guardar_productos(db, path=PRODUCTOS_PATH):
    """
    Guarda el diccionario 'db' en un archivo JSON.

    Parámetros:
        db   → el diccionario con los productos
        path → ruta del archivo de destino (por defecto: 'productos.json')

    JSON (JavaScript Object Notation) es un formato de texto que representa
    datos estructurados. Python lo soporta nativamente con el módulo 'json'.
    """
    # Abrimos el archivo en modo escritura ("w"), con codificación UTF-8
    # (importante para emojis y caracteres especiales como la 'ñ').
    with open(path, "w", encoding="utf-8") as f:
        # json.dump convierte el diccionario Python a texto JSON y lo escribe en 'f'.
        # ensure_ascii=False → permite caracteres no-ASCII (emojis, tildes)
        # indent=2          → sangría de 2 espacios para que sea legible al abrirlo
        json.dump(db, f, ensure_ascii=False, indent=2)


def cargar_productos(path=PRODUCTOS_PATH):
    """
    Carga el catálogo desde el archivo JSON.
    Si el archivo no existe, crea uno con el catálogo por defecto.

    Retorna:
        Un diccionario con los productos.
    """
    if os.path.exists(path):           # ¿Ya existe el archivo en el disco?
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)        # json.load lee el JSON y lo convierte a dict Python
    # Si no existe, lo creamos con los datos por defecto y retornamos una copia
    guardar_productos(DEFAULT_PRODUCTOS_DB, path)
    return DEFAULT_PRODUCTOS_DB.copy()  # .copy() para no modificar el original


# Cargamos el catálogo al inicio del programa (variable global)
PRODUCTOS_DB = cargar_productos()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASE: MotorVision
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Esta clase encapsula toda la lógica de Inteligencia Artificial:
#  - Cargar el modelo YOLO
#  - Analizar cada cuadro de video
#  - Filtrar detecciones por confianza, zona y estabilidad temporal
#  - Dibujar los resultados sobre el video
#
#  CONCEPTOS CLAVE:
#  ─────────────────
#  • YOLO: red neuronal que detecta múltiples objetos en una imagen en milisegundos.
#  • Confianza: valor entre 0.0 y 1.0 que indica qué tan seguro está YOLO
#               de que un objeto es lo que dice ser. 0.9 = 90% seguro.
#  • ROI (Region of Interest): subregión del frame donde buscamos objetos.
#               Reducir el área de búsqueda → menos falsos positivos.
#  • Cooldown: tiempo mínimo que debe pasar antes de agregar el mismo
#               producto dos veces seguidas (evita duplicados).
#  • Estabilidad temporal: exigimos que YOLO detecte el mismo objeto
#               varias veces en un intervalo corto antes de aceptarlo.
# ───────────────────────────────────────────────────────────────────────────────

class MotorVision:

    def __init__(self, modelo_path="yolov8s.pt"):
        """
        Constructor: se ejecuta automáticamente al crear un objeto MotorVision.
        Inicializa todos los parámetros y carga el modelo YOLO.

        Parámetro:
            modelo_path → nombre del archivo del modelo YOLO
                          "yolov8s.pt" = versión Small (rápida, menor precisión)
                          "yolov8m.pt" = versión Medium (más lenta, más precisa)
                          Si no existe localmente, Ultralytics lo descarga automáticamente.
        """
        self.modelo = None   # Aquí guardaremos el modelo YOLO una vez cargado

        # ── Umbral de confianza global ────────────────────────────────────────
        # Solo aceptamos detecciones donde YOLO esté al menos 75% seguro.
        # Valores más altos → menos falsos positivos, pero puede perder objetos reales.
        self.confianza_minima = 0.75

        # ── Control de cooldown ───────────────────────────────────────────────
        # Tiempo mínimo (en segundos) entre dos detecciones del mismo objeto.
        # Evita que una manzana se agregue 50 veces en 1 segundo.
        self.cooldown_deteccion = 2.0

        # Diccionario que registra cuándo se detectó cada clase por última vez.
        # Clave: nombre de la clase (ej: "apple")
        # Valor: timestamp (tiempo Unix) de la última detección aceptada
        self.historial_detecciones = {}

        # ── Allowlist (lista blanca de clases) ────────────────────────────────
        # Solo procesamos clases que están en nuestro catálogo.
        # Si YOLO detecta un "cat" (gato) pero no está en el catálogo, lo ignoramos.
        self.clases_permitidas = set(PRODUCTOS_DB.keys())  # set = conjunto sin duplicados

        # ── Umbrales específicos por clase ────────────────────────────────────
        # Algunas clases son más propensas a falsos positivos (ej: un celular
        # confundido con un mando a distancia). Podemos ser más exigentes con ellas.
        self.conf_por_clase = {
            "cell phone": 0.88,   # Necesitamos 88% de confianza para aceptar un celular
            "bottle":     0.88,   # Lo mismo para botellas
            "remote":     0.85,   # Y 85% para controles remotos
        }

        # ── Buffer de estabilidad temporal ───────────────────────────────────
        # Para cada clase, guardamos los últimos 8 timestamps de detección
        # (usando deque con maxlen=8 para que no crezca indefinidamente).
        # Antes de agregar un producto al carrito, verificamos que haya sido
        # detectado al menos 'frames_necesarios' veces en 'ventana_segundos'.
        self.buffer = defaultdict(lambda: deque(maxlen=8))  # clase → [(timestamp, confianza)]
        self.frames_necesarios = 2       # Debe aparecer al menos 2 veces...
        self.ventana_segundos  = 0.8     # ...dentro de los últimos 0.8 segundos

        # ── Zona de escaneo (ROI) ─────────────────────────────────────────────
        # Definimos la subregión del frame donde buscaremos objetos.
        # Los valores son porcentajes del tamaño total del frame:
        #   (x_inicio%, y_inicio%, x_fin%, y_fin%)
        #   (0.30, 0.25, 0.70, 0.75) = un rectángulo centrado que ocupa
        #   el 40% del ancho y el 50% del alto del frame.
        self.roi_rel = (0.30, 0.25, 0.70, 0.75)

        # ── Cargar el modelo YOLO ─────────────────────────────────────────────
        if YOLO_AVAILABLE:   # Solo intentamos si la librería está instalada
            try:
                print(f"[INFO] Cargando modelo YOLO: {modelo_path}")
                self.modelo = YOLO(modelo_path)   # Carga o descarga el modelo
                print("[OK] Modelo YOLO cargado.")
            except Exception as e:
                # Si algo falla (archivo corrupto, sin internet, etc.), no crasheamos
                print(f"[ERROR] No se pudo cargar YOLO: {e}")
                self.modelo = None


    def _clamp(self, v, lo, hi):
        """
        Limita el valor 'v' para que esté entre 'lo' (mínimo) y 'hi' (máximo).
        Ejemplo: _clamp(-5, 0, 100) → 0
                 _clamp(150, 0, 100) → 100
                 _clamp(50, 0, 100) → 50

        Se usa para asegurar que las coordenadas del ROI no salgan del frame.
        """
        return max(lo, min(hi, v))


    def get_roi(self, frame):
        """
        Calcula las coordenadas absolutas (en píxeles) de la zona de escaneo,
        a partir de los porcentajes relativos guardados en self.roi_rel.

        Parámetro:
            frame → imagen BGR de OpenCV (numpy array de forma [alto, ancho, 3])

        Retorna:
            (rx1, ry1, rx2, ry2) → coordenadas de la esquina superior-izquierda
                                   y esquina inferior-derecha del ROI en píxeles.

        ¿Por qué en porcentajes? Porque la resolución de la cámara puede variar
        (720p, 1080p, etc.) y así el ROI siempre queda centrado sin importar el tamaño.
        """
        h, w = frame.shape[:2]   # Extraemos alto (h) y ancho (w) del frame
                                  # frame.shape devuelve (alto, ancho, canales)

        # Convertimos porcentajes a píxeles multiplicando por el tamaño del frame
        rx1 = int(w * self.roi_rel[0])   # 30% del ancho → inicio horizontal
        ry1 = int(h * self.roi_rel[1])   # 25% del alto  → inicio vertical
        rx2 = int(w * self.roi_rel[2])   # 70% del ancho → fin horizontal
        ry2 = int(h * self.roi_rel[3])   # 75% del alto  → fin vertical

        # Usamos _clamp para asegurarnos de que las coordenadas no salgan del frame
        rx1 = self._clamp(rx1, 0, w - 1)
        ry1 = self._clamp(ry1, 0, h - 1)
        rx2 = self._clamp(rx2, rx1 + 1, w)   # rx2 debe ser al menos rx1+1 (ROI de 1px)
        ry2 = self._clamp(ry2, ry1 + 1, h)

        return rx1, ry1, rx2, ry2


    def puede_agregar(self, clase):
        """
        Verifica si ha pasado suficiente tiempo desde la última vez que
        se agregó esta clase al carrito (cooldown anti-duplicados).

        Parámetro:
            clase → nombre de la clase detectada (ej: "apple")

        Retorna:
            True  → ya pasó el tiempo de cooldown, se puede agregar
            False → aún no ha pasado el tiempo mínimo, ignorar

        Efecto secundario: si retorna True, actualiza el timestamp
        de la última detección (para reiniciar el cooldown).
        """
        ahora = time.time()   # Tiempo actual en segundos (número flotante)

        # .get(clase, 0) devuelve 0 si la clase no tiene historial (primera vez)
        ultimo = self.historial_detecciones.get(clase, 0)

        if ahora - ultimo >= self.cooldown_deteccion:
            # Ha pasado suficiente tiempo → actualizamos el historial y permitimos
            self.historial_detecciones[clase] = ahora
            return True

        return False   # Todavía en cooldown


    def deteccion_estable(self, clase, confianza):
        """
        Verifica si el objeto ha sido detectado suficientes veces recientemente,
        es decir, si la detección es "estable" (no un destello de un solo frame).

        Parámetros:
            clase      → nombre de la clase (ej: "bottle")
            confianza  → nivel de confianza de esta detección (0.0 a 1.0)

        Retorna:
            True  → la clase ha aparecido >= frames_necesarios veces en la ventana
            False → no hay suficientes detecciones recientes

        CÓMO FUNCIONA:
            1. Agrega el timestamp actual al buffer de la clase.
            2. Filtra el buffer para quedarse solo con las detecciones
               de los últimos 'ventana_segundos' segundos.
            3. Si hay suficientes, la detección se considera estable.
        """
        ahora = time.time()

        # Agregamos esta detección al buffer: (timestamp, confianza)
        self.buffer[clase].append((ahora, confianza))

        # Filtramos: nos quedamos solo con las detecciones recientes
        recientes = [
            conf                                          # valor de confianza
            for (t, conf) in self.buffer[clase]           # iteramos el buffer
            if (ahora - t) <= self.ventana_segundos       # solo las recientes
        ]

        # ¿Hay suficientes detecciones recientes?
        return len(recientes) >= self.frames_necesarios


    def detectar(self, frame_bgr):
        """
        El método principal de detección. Analiza el frame con YOLO y
        retorna las detecciones válidas (filtradas).

        Parámetro:
            frame_bgr → imagen completa capturada por la cámara (formato BGR de OpenCV)

        Retorna:
            (detecciones, roi_coords) donde:
            - detecciones: lista de dicts con info de cada objeto detectado
              Cada dict tiene: {clase, confianza, bbox, producto}
              • clase     → nombre en inglés (ej: "apple")
              • confianza → float entre 0 y 1
              • bbox      → [x1, y1, x2, y2] en píxeles del frame COMPLETO
              • producto  → dict del catálogo (o None si no está registrado)
            - roi_coords: (rx1, ry1, rx2, ry2) para dibujar el rectángulo verde

        IMPORTANTE SOBRE bbox:
            YOLO devuelve coordenadas relativas al ROI (la subimagen que le pasamos).
            Necesitamos sumarle el offset del ROI para obtener coordenadas
            relativas al frame COMPLETO (que es lo que muestra el canvas).
        """
        if self.modelo is None:
            return [], None   # Sin modelo, no hay detecciones

        # Calculamos las coordenadas del ROI en píxeles
        rx1, ry1, rx2, ry2 = self.get_roi(frame_bgr)

        # Recortamos el frame para quedarnos solo con el ROI
        # Sintaxis de NumPy para slicing de matrices: array[fila_ini:fila_fin, col_ini:col_fin]
        roi = frame_bgr[ry1:ry2, rx1:rx2]

        detecciones = []   # Lista que iremos llenando

        try:
            # ── Ejecutar YOLO sobre el ROI ────────────────────────────────────
            # self.modelo(roi, ...) devuelve una lista de resultados, tomamos [0]
            # imgsz=480 → redimensionar internamente a 480px (más rápido)
            # verbose=False → no imprimir estadísticas en cada inferencia
            res = self.modelo(roi, imgsz=480, verbose=False)[0]

            # res.boxes contiene todas las cajas detectadas en esta imagen
            for b in res.boxes:

                # ── Extraer datos de la caja ──────────────────────────────────
                conf = float(b.conf[0])        # Confianza: tensor de 1 elemento → float
                cls_id = int(b.cls[0])         # ID numérico de la clase
                cls_name = self.modelo.names[cls_id].lower()  # Nombre legible en minúsculas
                # .lower() → "Apple" se convierte en "apple" para comparar consistentemente

                # ── Filtro 1: ¿Está en nuestro catálogo? ─────────────────────
                if cls_name not in self.clases_permitidas:
                    continue   # 'continue' salta al siguiente elemento del for

                # ── Filtro 2: ¿Supera el umbral de confianza? ─────────────────
                # Buscamos el umbral específico; si no existe, usamos el global
                umbral = self.conf_por_clase.get(cls_name, self.confianza_minima)
                if conf < umbral:
                    continue

                # ── Convertir coordenadas ROI → frame completo ─────────────────
                # b.xyxy[0] es un tensor: [x_inicio, y_inicio, x_fin, y_fin]
                # .tolist() lo convierte a una lista Python normal
                x1, y1, x2, y2 = b.xyxy[0].tolist()

                # Sumamos el offset del ROI para que sean coordenadas del frame completo
                x1 += rx1;  x2 += rx1   # Desplazamiento horizontal
                y1 += ry1;  y2 += ry1   # Desplazamiento vertical

                # ── Buscar el producto en el catálogo ─────────────────────────
                # .get() retorna None si la clave no existe (aunque aquí ya filtramos)
                producto = PRODUCTOS_DB.get(cls_name)

                # Agregamos esta detección a la lista de resultados
                detecciones.append({
                    "clase":     cls_name,
                    "confianza": conf,
                    "bbox":      [x1, y1, x2, y2],
                    "producto":  producto
                })

        except Exception as e:
            print(f"[ERROR] Detección fallida: {e}")

        return detecciones, (rx1, ry1, rx2, ry2)


    def dibujar(self, frame, detecciones, roi_coords=None):
        """
        Dibuja sobre el frame: la zona de escaneo y las cajas de detección.

        Parámetros:
            frame       → imagen BGR del frame actual (se modifica en sitio)
            detecciones → lista de dicts retornada por self.detectar()
            roi_coords  → (rx1, ry1, rx2, ry2) o None

        Retorna:
            frame con los dibujos aplicados

        TÉCNICA DE OVERLAY:
            En lugar de dibujar directamente sobre el frame, primero hacemos
            una copia (overlay), dibujamos en el frame, y luego mezclamos ambos.
            Esto crea un efecto de transparencia en los recuadros coloreados.
        """
        overlay = frame.copy()   # Copia para efecto de transparencia

        # ── Dibujar la zona de escaneo (rectángulo verde tenue) ───────────────
        if roi_coords:
            rx1, ry1, rx2, ry2 = roi_coords
            # cv2.rectangle(imagen, punto_superior_izq, punto_inferior_der, color_BGR, grosor)
            # Color verde brillante en BGR: (0, 255, 100)
            # Grosor 1 → línea delgada
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 100), 1)

            # Texto "ZONA DE ESCANEO" encima del rectángulo
            # max(20, ry1-8) → si el rectángulo está muy arriba, el texto no queda fuera del frame
            cv2.putText(frame, "ZONA DE ESCANEO", (rx1, max(20, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,  # Fuente
                        0.55,                       # Escala del texto
                        (0, 255, 100),              # Color (verde) en BGR
                        1,                          # Grosor del trazo
                        cv2.LINE_AA)                # Antialiasing (bordes suaves)

        # ── Dibujar cada detección ────────────────────────────────────────────
        for det in detecciones:
            x1, y1, x2, y2 = map(int, det["bbox"])   # Convertimos a enteros (int) para dibujar
            clase    = det["clase"]
            conf     = det["confianza"]
            producto = det["producto"]

            # Color según si el producto está en el catálogo
            color = (0, 255, 100) if producto else (0, 180, 255)   # Verde : Naranja

            # Dibujamos la caja alrededor del objeto
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Construimos el texto de la etiqueta
            nombre = producto["nombre"] if producto else clase.upper()
            precio = f"${producto['precio']:.2f}" if producto else ""
            # f-string: .0% convierte un float a porcentaje sin decimales (0.87 → "87%")
            label = f"{nombre} {precio} ({conf:.0%})"

            # Medimos el texto para crear un fondo detrás de él
            # getTextSize devuelve ((ancho, alto), baseline)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            # Rectángulo de fondo (relleno, grosor=-1) para que el texto sea legible
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)

            # Texto de la etiqueta sobre el rectángulo de fondo
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0),    # Texto negro (contrasta con el fondo de color)
                        1, cv2.LINE_AA)

        # ── Efecto de transparencia ───────────────────────────────────────────
        # cv2.addWeighted mezcla dos imágenes con pesos:
        #   resultado = overlay*0.25 + frame*0.75
        # Esto hace que los rectángulos rellenos se vean semi-transparentes.
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASE: Carrito
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Representa el carrito de compras. Almacena los productos detectados,
#  sus cantidades y precios.
#
#  PATRÓN OBSERVER (básico):
#  Esta clase implementa una versión simplificada del patrón Observer.
#  Otros objetos se "suscriben" con on_cambio(callback) y son notificados
#  automáticamente cada vez que el carrito cambia.
#  Esto desacopla el carrito de la interfaz gráfica: el carrito no necesita
#  saber que existe Tkinter, solo llama a sus callbacks.
# ───────────────────────────────────────────────────────────────────────────────

class Carrito:

    def __init__(self):
        """
        Inicializa el carrito vacío.

        self.items es un defaultdict con una función lambda como fábrica.
        Cuando accedemos a una clave que no existe (ej: items["apple"]),
        Python crea automáticamente el valor por defecto:
            {"cantidad": 0, "precio": 0, "nombre": "", "emoji": "📦"}
        Así no necesitamos verificar si la clave existe antes de incrementar.

        self.callbacks es una lista de funciones que se llaman cuando
        el carrito cambia (patrón Observer simplificado).
        """
        self.items = defaultdict(lambda: {
            "cantidad": 0,
            "precio":   0,
            "nombre":   "",
            "emoji":    "📦"
        })
        self.callbacks = []   # Lista de funciones que se ejecutan al cambiar el carrito


    def agregar(self, producto_info, clase):
        """
        Agrega una unidad de un producto al carrito.

        Parámetros:
            producto_info → dict del catálogo (con nombre, precio, emoji, etc.)
            clase         → clave del producto (ej: "apple") usada como ID en items

        El operador '+= 1' en items[clase]["cantidad"] funciona gracias a
        defaultdict: si "apple" no existe, crea el dict por defecto con
        cantidad=0 y luego lo incrementa a 1.
        """
        self.items[clase]["cantidad"] += 1                              # Incrementar cantidad
        self.items[clase]["precio"]    = float(producto_info["precio"]) # Precio unitario
        self.items[clase]["nombre"]    = producto_info["nombre"]        # Nombre visible
        self.items[clase]["emoji"]     = producto_info.get("emoji", "📦")  # Emoji (default 📦)
        self.notificar()   # Avisar a los suscriptores que el carrito cambió


    def eliminar(self, clase):
        """
        Elimina completamente un producto del carrito (todas sus unidades).

        Parámetro:
            clase → clave del producto a eliminar (ej: "apple")
        """
        if clase in self.items:   # Verificamos que exista para no dar error
            del self.items[clase]
            self.notificar()


    def vaciar(self):
        """
        Elimina TODOS los productos del carrito.
        .clear() vacía el diccionario sin crearlo de nuevo.
        """
        self.items.clear()
        self.notificar()


    def total(self):
        """
        Calcula el precio total del carrito.

        Retorna:
            float → suma de (cantidad × precio_unitario) de todos los productos

        Usamos sum() con una expresión generadora:
            sum(expresion for elemento in iterable)
        Es equivalente a un for loop que va sumando, pero más conciso.
        """
        return sum(
            v["cantidad"] * v["precio"]
            for v in self.items.values()   # .values() itera solo los valores del dict
        )


    def total_items(self):
        """
        Cuenta el número total de artículos en el carrito (sumando cantidades).
        Ej: 2 manzanas + 1 botella = 3 artículos.
        """
        return sum(v["cantidad"] for v in self.items.values())


    def on_cambio(self, cb):
        """
        Registra una función (callback) que se llamará cada vez que el carrito cambie.

        Parámetro:
            cb → función sin argumentos (callable)

        Ejemplo de uso:
            carrito.on_cambio(actualizar_interfaz)
            # Ahora cada vez que el carrito cambie, se llama actualizar_interfaz()
        """
        self.callbacks.append(cb)


    def notificar(self):
        """
        Llama a todos los callbacks registrados.
        Se ejecuta después de cada operación que modifica el carrito.
        """
        for cb in self.callbacks:
            cb()   # Llamamos cada función registrada


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASE: InterfazCaja
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Maneja toda la interfaz gráfica con Tkinter y coordina el flujo del programa.
#
#  ARQUITECTURA DE HILOS:
#  ─────────────────────────────────────────────────────────────────────────────
#  El programa usa DOS hilos de ejecución paralela:
#
#  Hilo Principal (main thread):
#    → Corre el loop de eventos de Tkinter (root.mainloop())
#    → Actualiza el canvas con el video cada 33ms (≈30fps)
#    → Responde a clicks y eventos del usuario
#    → NO puede hacer operaciones lentas (congela la UI)
#
#  Hilo de Cámara (daemon thread):
#    → Captura frames de la cámara continuamente
#    → Ejecuta YOLO para detectar objetos
#    → Escribe el frame procesado en self.frame_actual
#    → Es "daemon": se destruye automáticamente cuando cierra la ventana
#
#  Comunicación entre hilos:
#    → self.frame_actual = frame  (el hilo de cámara escribe)
#    → canvas muestra self.frame_actual (el hilo principal lee)
#    → Esta comunicación es simplificada (sin locks); funciona porque
#      Python tiene el GIL que previene corrupción de datos básica.
# ───────────────────────────────────────────────────────────────────────────────

class InterfazCaja:

    # ── Paleta de colores (constantes de clase) ───────────────────────────────
    # Definimos los colores como atributos de clase para reutilizarlos fácilmente.
    # Tkinter usa colores en formato hexadecimal "#RRGGBB" o nombres como "red".
    VERDE_NEON = "#00FF64"   # Verde brillante: color principal de la interfaz
    BG_OSCURO  = "#0A0F0A"   # Casi negro con tinte verdoso: fondo general
    BG_PANEL   = "#0F1A0F"   # Gris verdoso oscuro: fondo de paneles
    BG_CARD    = "#152015"   # Un poco más claro: fondo de tarjetas/listas
    TEXTO      = "#E8FFE8"   # Blanco verdoso: texto principal
    TEXTO_DIM  = "#6A8F6A"   # Verde grisáceo: texto secundario/atenuado
    ROJO       = "#FF4455"   # Rojo: botones de acción destructiva


    def __init__(self, cam_index=0):
        """
        Constructor de la interfaz. Crea la ventana, inicializa componentes
        y arranca la cámara.

        Parámetro:
            cam_index → índice de la cámara a usar (0=primera, 1=segunda, etc.)
                        Si tienes webcam integrada y externa, prueba 0 y 1.
        """
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter no está disponible en este entorno.")

        # ── Crear la ventana principal de Tkinter ─────────────────────────────
        self.root = tk.Tk()
        self.root.title("CAJA REGISTRADORA • VISIÓN IA (MEJORADA)")
        self.root.configure(bg=self.BG_OSCURO)  # Color de fondo de la ventana
        self.root.geometry("1280x800")           # Tamaño inicial: 1280×800 píxeles

        # ── Instanciar los módulos principales ───────────────────────────────
        self.carrito = Carrito()                          # El carrito de compras
        self.motor   = MotorVision(modelo_path="yolov8s.pt")  # El motor de IA
        self.cap     = None    # VideoCapture de OpenCV (se crea en _iniciar_camara)
        self.activo  = False   # Bandera: controla el loop del hilo de cámara

        # ── Variables de estado ───────────────────────────────────────────────
        self.fps_actual  = 0      # FPS del hilo de cámara (informativo)
        self.latencia    = 0      # Tiempo de procesamiento por frame en ms
        self.frame_actual = None  # Último frame procesado (compartido entre hilos)

        # ── Notificación temporal ("Manzana añadida") ─────────────────────────
        self.notificacion_texto = ""    # Texto a mostrar
        self.notificacion_timer = 0     # Cuando se disparó (para ocultarla después)

        self.cam_index = cam_index   # Guardamos el índice de cámara

        # ── Construir la interfaz y conectar eventos ──────────────────────────
        self._construir_ui()   # Crea todos los widgets de Tkinter

        # Suscribimos la actualización del carrito al evento on_cambio
        # Cada vez que el carrito cambie, se llamará self._actualizar_carrito_ui
        self.carrito.on_cambio(self._actualizar_carrito_ui)

        self._iniciar_camara()  # Abre la cámara y arranca el hilo de visión


    def _construir_ui(self):
        """
        Construye toda la interfaz gráfica de Tkinter.

        LAYOUT (disposición visual):
        ┌─────────────────────────────────────────────────────┐
        │  HEADER: título + estado + FPS + reloj              │
        ├──────────────────────────────────┬──────────────────┤
        │  VIDEO (canvas)                  │  CARRITO         │
        │                                  │  (listbox)       │
        │  [instrucción]                   │  TOTAL           │
        │  [notificación]                  │  [COBRAR]        │
        │                                  │  [VACIAR]        │
        │                                  │  [EDITAR JSON]   │
        └──────────────────────────────────┴──────────────────┘

        JERARQUÍA DE WIDGETS EN TKINTER:
            root (ventana principal)
            ├── header (Frame horizontal)
            │   ├── lbl_titulo
            │   ├── lbl_estado
            │   ├── lbl_fps
            │   └── lbl_hora
            └── body (Frame principal)
                ├── panel_cam (izquierda)
                │   ├── cam_header
                │   ├── canvas_video
                │   ├── lbl_instruccion
                │   └── lbl_notif
                └── panel_d (derecha)
                    ├── carrito_frame
                    │   ├── ch (encabezado)
                    │   ├── listbox_carrito
                    │   └── btn_eliminar
                    ├── total_frame
                    └── acciones
        """

        # ── HEADER ────────────────────────────────────────────────────────────
        # Frame horizontal en la parte superior. height=55 fija su altura.
        header = tk.Frame(self.root, bg=self.BG_PANEL, height=55)
        header.pack(fill="x")          # fill="x" → ocupa todo el ancho
        header.pack_propagate(False)   # No permite que los hijos cambien su tamaño

        # Título de la aplicación (Label = widget de texto estático)
        tk.Label(header,
                 text="CAJA REGISTRADORA (MEJORADA)",
                 font=("Courier", 18, "bold"),
                 bg=self.BG_PANEL,
                 fg=self.VERDE_NEON).pack(side="left", padx=20, pady=12)
        # side="left" → se apila a la izquierda dentro del header
        # padx/pady   → margen externo en píxeles

        # Indicador de estado ("● ONLINE" / "● OFFLINE")
        # Lo guardamos como atributo para poder cambiar su texto/color más tarde
        self.lbl_estado = tk.Label(header,
                                   text="● ONLINE",
                                   font=("Courier", 11, "bold"),
                                   bg="#1A3020", fg=self.VERDE_NEON,
                                   padx=10, pady=4)
        self.lbl_estado.pack(side="left", padx=10)

        # Contador de FPS y latencia
        self.lbl_fps = tk.Label(header,
                                text="FPS: --  LAT: --ms",
                                font=("Courier", 10),
                                bg=self.BG_PANEL, fg=self.TEXTO_DIM)
        self.lbl_fps.pack(side="left", padx=15)

        # Reloj en tiempo real (actualizado en _actualizar_canvas)
        self.lbl_hora = tk.Label(header, text="",
                                 font=("Courier", 11),
                                 bg=self.BG_PANEL, fg=self.TEXTO_DIM)
        self.lbl_hora.pack(side="right", padx=20)  # side="right" → al lado derecho

        # ── BODY (área principal debajo del header) ───────────────────────────
        body = tk.Frame(self.root, bg=self.BG_OSCURO)
        # fill="both" + expand=True → ocupa todo el espacio restante
        body.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # ── PANEL IZQUIERDO: video de la cámara ───────────────────────────────
        panel_cam = tk.Frame(body, bg=self.BG_PANEL, bd=1, relief="flat")
        # expand=True → crece para ocupar el espacio disponible
        panel_cam.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Encabezado del panel de video
        cam_header = tk.Frame(panel_cam, bg="#0D1A0D", height=30)
        cam_header.pack(fill="x")
        cam_header.pack_propagate(False)
        tk.Label(cam_header, text="FEED EN VIVO",
                 font=("Courier", 9), bg="#0D1A0D", fg=self.TEXTO_DIM).pack(side="left", padx=10)

        # Canvas donde se renderiza el video frame por frame
        # highlightthickness=1 + highlightbackground → borde de color
        self.canvas_video = tk.Canvas(panel_cam, bg="#050A05",
                                      highlightthickness=1,
                                      highlightbackground=self.VERDE_NEON)
        self.canvas_video.pack(fill="both", expand=True, padx=5, pady=5)

        # Instrucción para el usuario
        self.lbl_instruccion = tk.Label(
            panel_cam,
            text="Coloca el producto EN LA ZONA DE ESCANEO (rectángulo verde)",
            font=("Courier", 11, "bold"),
            bg="#0D1A0D", fg=self.VERDE_NEON, pady=6)
        self.lbl_instruccion.pack(fill="x", padx=5, pady=(0, 5))

        # Label de notificación (se muestra/oculta dinámicamente con .pack/.pack_forget)
        self.lbl_notif = tk.Label(panel_cam, text="",
                                  font=("Courier", 13, "bold"),
                                  bg=self.VERDE_NEON, fg="#000000", pady=8)
        # No llamamos .pack() aquí; se hace dinámicamente en _actualizar_canvas

        # ── PANEL DERECHO: carrito + totales + botones ────────────────────────
        panel_d = tk.Frame(body, bg=self.BG_OSCURO, width=360)
        panel_d.pack(side="right", fill="y", padx=(5, 0))
        panel_d.pack_propagate(False)   # Ancho fijo de 360px

        # ── CARRITO ───────────────────────────────────────────────────────────
        carrito_frame = tk.Frame(panel_d, bg=self.BG_PANEL, bd=1)
        carrito_frame.pack(fill="both", expand=True, pady=(0, 5))

        # Encabezado del carrito
        ch = tk.Frame(carrito_frame, bg="#0D1A0D", height=36)
        ch.pack(fill="x")
        ch.pack_propagate(False)
        tk.Label(ch, text="🛒 CARRITO",
                 font=("Courier", 11, "bold"),
                 bg="#0D1A0D", fg=self.VERDE_NEON).pack(side="left", padx=10, pady=8)

        # Badge circular con el conteo de artículos
        self.lbl_contador_badge = tk.Label(ch, text="0",
                                           font=("Courier", 10, "bold"),
                                           bg=self.VERDE_NEON, fg="#000000",
                                           width=3, pady=2)
        self.lbl_contador_badge.pack(side="right", padx=10, pady=6)

        # Contenedor de la lista + scrollbar
        list_container = tk.Frame(carrito_frame, bg=self.BG_PANEL)
        list_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Barra de desplazamiento vertical
        scrollbar = tk.Scrollbar(list_container, bg=self.BG_OSCURO, troughcolor=self.BG_OSCURO)
        scrollbar.pack(side="right", fill="y")

        # Listbox: lista de elementos seleccionables
        # yscrollcommand conecta la listbox con la scrollbar
        self.listbox_carrito = tk.Listbox(
            list_container,
            bg=self.BG_CARD, fg=self.TEXTO,
            selectbackground="#00CC50", selectforeground="#000",
            font=("Courier", 10),
            bd=0, highlightthickness=0,
            yscrollcommand=scrollbar.set)
        self.listbox_carrito.pack(fill="both", expand=True)

        # Conectar la scrollbar con el listbox (bidireccional)
        scrollbar.config(command=self.listbox_carrito.yview)

        # Botón para eliminar el producto seleccionado en el listbox
        btn_eliminar = tk.Button(
            carrito_frame,
            text="⊖ Eliminar seleccionado",
            font=("Courier", 9), bg=self.BG_OSCURO, fg=self.ROJO,
            bd=0, cursor="hand2",    # cursor="hand2" → cursor de mano al pasar el mouse
            pady=4,
            command=self._eliminar_seleccionado)   # Función a llamar al hacer click
        btn_eliminar.pack(fill="x", padx=5, pady=(0, 5))

        # ── TOTAL ─────────────────────────────────────────────────────────────
        total_frame = tk.Frame(panel_d, bg=self.BG_PANEL)
        total_frame.pack(fill="x", pady=(0, 5))

        tk.Label(total_frame, text="SUBTOTAL:",
                 font=("Courier", 10), bg=self.BG_PANEL, fg=self.TEXTO_DIM).pack(side="left", padx=10, pady=8)

        # El total se actualiza automáticamente cada vez que el carrito cambia
        self.lbl_total = tk.Label(total_frame, text="$0.00",
                                  font=("Courier", 18, "bold"),
                                  bg=self.BG_PANEL, fg=self.VERDE_NEON)
        self.lbl_total.pack(side="right", padx=10, pady=8)

        # ── BOTONES DE ACCIÓN ─────────────────────────────────────────────────
        acciones = tk.Frame(panel_d, bg=self.BG_OSCURO)
        acciones.pack(fill="x")

        # Botón COBRAR: verde prominente (acción principal)
        tk.Button(acciones, text="✓  COBRAR",
                  font=("Courier", 12, "bold"),
                  bg=self.VERDE_NEON, fg="#000000", bd=0,
                  cursor="hand2", pady=10,
                  command=self._cobrar).pack(fill="x", pady=(0, 4))

        # Botón VACIAR: texto rojo (acción destructiva)
        tk.Button(acciones, text="⊗  VACIAR CARRITO",
                  font=("Courier", 10),
                  bg=self.BG_PANEL, fg=self.ROJO, bd=0,
                  cursor="hand2", pady=7,
                  command=self._vaciar).pack(fill="x", pady=(0, 4))

        # Botón EDITAR CATÁLOGO: abre el JSON con el editor del sistema
        tk.Button(acciones, text="⊕  EDITAR CATÁLOGO (JSON)",
                  font=("Courier", 10),
                  bg=self.BG_PANEL, fg=self.TEXTO_DIM, bd=0,
                  cursor="hand2", pady=7,
                  command=self._abrir_productos_json).pack(fill="x")


    def _iniciar_camara(self):
        """
        Abre la cámara con OpenCV y arranca el hilo de procesamiento de video.

        cv2.VideoCapture(index):
            - index=0 → primera cámara disponible (webcam integrada usualmente)
            - index=1 → segunda cámara (webcam externa, etc.)
            También acepta rutas de archivo de video: cv2.VideoCapture("video.mp4")

        Si no encuentra cámara, actualiza el indicador de estado a "OFFLINE".
        """
        self.cap = cv2.VideoCapture(self.cam_index)

        if not self.cap.isOpened():   # .isOpened() → False si no pudo abrir la cámara
            self.lbl_estado.config(text="● OFFLINE", fg=self.ROJO, bg="#2A0D10")
            print("[ERROR] No se encontró cámara. Cambia cam_index (0/1/2) o revisa conexión.")
            return

        # Configurar la resolución y FPS de la cámara
        # CAP_PROP_* son constantes que representan propiedades configurables
        # Nota: el hardware puede no soportar estos valores exactos y usar el más cercano
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)  # Ancho en píxeles
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Alto en píxeles
        self.cap.set(cv2.CAP_PROP_FPS, 60)              # Cuadros por segundo

        self.activo = True   # Activamos la bandera que controla el loop del hilo

        # Creamos y arrancamos el hilo de visión (ejecuta _loop_vision en paralelo)
        # daemon=True → el hilo se destruye automáticamente cuando cierra la app
        self.hilo_camara = threading.Thread(target=self._loop_vision, daemon=True)
        self.hilo_camara.start()

        # Iniciamos el ciclo de actualización del canvas (en el hilo principal)
        self._actualizar_canvas()


    def _loop_vision(self):
        """
        Loop principal del HILO DE CÁMARA (se ejecuta en paralelo a la UI).
        Se repite continuamente mientras self.activo sea True.

        CICLO DE CADA ITERACIÓN:
        1. Capturar un frame de la cámara
        2. Voltear horizontalmente (efecto espejo, más natural)
        3. Si ha pasado el intervalo mínimo, ejecutar detección YOLO
        4. Para cada detección válida, verificar estabilidad + cooldown
        5. Si pasa todos los filtros, agregar al carrito
        6. Dibujar las detecciones y el HUD sobre el frame
        7. Guardar el frame en self.frame_actual (el hilo principal lo mostrará)
        8. Dormir el tiempo necesario para no superar 60 FPS
        """
        intervalo_deteccion = 0.12   # Ejecutar YOLO máximo cada 120ms (~8 veces/seg)
        ultimo_det = 0                # Timestamp de la última detección ejecutada

        while self.activo:   # Repetir mientras la app esté activa
            t0 = time.time()   # Marcamos el inicio del ciclo (para medir FPS)

            # ── Capturar frame ────────────────────────────────────────────────
            ret, frame = self.cap.read()
            # ret   → True si capturó correctamente, False si hubo error
            # frame → numpy array BGR de forma (alto, ancho, 3)

            if not ret:   # Si no capturó (cámara desconectada, fin de video, etc.)
                time.sleep(0.05)   # Esperamos 50ms antes de reintentar
                continue           # Saltamos el resto del ciclo

            # Volteamos horizontalmente (efecto espejo)
            # cv2.flip(imagen, 1): 1=horizontal, 0=vertical, -1=ambos
            frame = cv2.flip(frame, 1)

            ahora = time.time()
            detecciones = []   # Iniciamos lista vacía (se llenará si corresponde)
            roi_coords = None

            # ── Detección YOLO (con control de frecuencia) ────────────────────
            if ahora - ultimo_det >= intervalo_deteccion:
                # Ejecutamos detección y obtenemos resultados + coords del ROI
                detecciones, roi_coords = self.motor.detectar(frame)
                ultimo_det = ahora

                # ── Procesar cada detección ───────────────────────────────────
                for det in detecciones:
                    if det["producto"]:   # ¿El objeto detectado está en el catálogo?

                        # Verificamos ESTABILIDAD: ¿ha aparecido suficientes veces?
                        estable = self.motor.deteccion_estable(det["clase"], det["confianza"])

                        # Verificamos COOLDOWN: ¿ha pasado el tiempo mínimo?
                        puede = self.motor.puede_agregar(det["clase"])

                        if estable and puede:
                            # ¡Pasó todos los filtros! Agregamos al carrito
                            self.carrito.agregar(det["producto"], det["clase"])
                            # Mostramos notificación temporal en la UI
                            self._mostrar_notificacion(det["producto"]["nombre"])

            # ── Dibujar resultados sobre el frame ─────────────────────────────
            frame = self.motor.dibujar(frame, detecciones, roi_coords)

            # ── Dibujar HUD (mira de crosshair) ──────────────────────────────
            frame = self._dibujar_hud(frame)

            # ── Compartir el frame con el hilo principal ──────────────────────
            # Esta asignación es atómica en CPython (por el GIL), así que
            # no necesitamos un lock para este caso simple.
            self.frame_actual = frame

            # ── Métricas de rendimiento ───────────────────────────────────────
            t1 = time.time()
            self.latencia  = int((t1 - t0) * 1000)               # ms por frame
            self.fps_actual = int(1.0 / max(t1 - t0, 0.001))    # FPS (evitamos /0)

            # ── Limitar a 60 FPS ──────────────────────────────────────────────
            # Calculamos cuánto tiempo queda del "slot" de 1/60 segundos
            # max(0, ...) evita sleep negativo si el procesamiento tomó más de 1/60s
            sleep_time = max(0, (1.0 / 60) - (time.time() - t0))
            time.sleep(sleep_time)


    def _dibujar_hud(self, frame):
        """
        Dibuja la mira (crosshair) en el centro del frame.

        La mira tiene 4 segmentos que apuntan al centro pero no se tocan,
        creando un espacio vacío en el medio (más estético y legible).

        Parámetro:
            frame → imagen BGR a modificar

        Retorna:
            frame con la mira dibujada
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2   # Centro del frame (división entera)
        color = (0, 255, 100)      # Verde brillante en BGR

        size = 35   # Longitud de cada brazo de la mira en píxeles
        gap  = 12   # Espacio vacío en el centro (mitad del gap en cada lado)

        # Línea horizontal: izquierda del centro
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, 1)
        # Línea horizontal: derecha del centro
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, 1)
        # Línea vertical: arriba del centro
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, 1)
        # Línea vertical: abajo del centro
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, 1)

        return frame


    def _actualizar_canvas(self):
        """
        Actualiza el canvas de Tkinter con el frame más reciente de la cámara.
        También actualiza FPS, reloj y la notificación temporal.

        Se llama a sí misma cada 33ms usando root.after() para crear
        un loop continuo dentro del hilo principal de Tkinter.

        33ms ≈ 30 FPS en la interfaz de usuario.

        CONVERSIÓN DE FORMATOS:
            OpenCV usa BGR (Blue-Green-Red) y numpy arrays.
            Tkinter usa RGB y objetos PhotoImage de Pillow.
            Flujo: frame BGR (numpy) → RGB (numpy) → PIL Image → PhotoImage (Tkinter)
        """
        if self.frame_actual is not None:
            try:
                from PIL import Image, ImageTk   # Importamos aquí para manejar si no están instalados

                cw = self.canvas_video.winfo_width()    # Ancho actual del canvas en píxeles
                ch = self.canvas_video.winfo_height()   # Alto actual del canvas en píxeles

                if cw > 1 and ch > 1:   # El canvas ya tiene tamaño (evita error al inicio)

                    # Convertir BGR → RGB (invertir el orden de los canales de color)
                    rgb = cv2.cvtColor(self.frame_actual, cv2.COLOR_BGR2RGB)

                    # Crear imagen PIL y redimensionar al tamaño actual del canvas
                    # Image.LANCZOS = algoritmo de interpolación de alta calidad
                    img = Image.fromarray(rgb).resize((cw, ch), Image.LANCZOS)

                    # Convertir PIL Image a PhotoImage (formato que entiende Tkinter)
                    # Guardamos en self.photo para evitar que el garbage collector la elimine
                    self.photo = ImageTk.PhotoImage(img)

                    # Dibujar la imagen en el canvas (esquina superior izquierda: 0,0)
                    self.canvas_video.create_image(0, 0, anchor="nw", image=self.photo)

            except Exception:
                pass   # Si algo falla (PIL no instalado, frame inválido), ignoramos

        # ── Actualizar etiquetas de métricas ──────────────────────────────────
        self.lbl_fps.config(text=f"FPS: {self.fps_actual}  LAT: {self.latencia}ms")
        self.lbl_hora.config(text=datetime.datetime.now().strftime("%H:%M:%S"))

        # ── Gestión de la notificación temporal ──────────────────────────────
        tiempo_transcurrido = time.time() - self.notificacion_timer

        if self.notificacion_texto and tiempo_transcurrido < 2.5:
            # Mostrar notificación: actualizamos texto y la hacemos visible con .pack()
            self.lbl_notif.config(text=f"✓  {self.notificacion_texto} añadido")
            self.lbl_notif.pack(fill="x", padx=5, pady=(0, 5))
        else:
            # Ocultar notificación: .pack_forget() la saca del layout pero no la destruye
            self.lbl_notif.pack_forget()
            self.notificacion_texto = ""

        # ── Programar la próxima actualización ────────────────────────────────
        # root.after(ms, función) → llama 'función' después de 'ms' milisegundos
        # Esto crea un loop no-bloqueante dentro del hilo principal de Tkinter.
        if self.activo:
            self.root.after(33, self._actualizar_canvas)


    def _mostrar_notificacion(self, nombre):
        """
        Activa la notificación temporal en el panel de video.

        Parámetros:
            nombre → nombre del producto recién detectado (ej: "Manzana")

        La notificación se ocultará automáticamente después de 2.5 segundos
        (controlado en _actualizar_canvas).
        """
        self.notificacion_texto = nombre
        self.notificacion_timer = time.time()   # Guardamos el momento actual


    def _actualizar_carrito_ui(self):
        """
        Callback que se llama automáticamente cada vez que el carrito cambia.
        Redibuja el listbox y actualiza el total y el badge.

        Este método es llamado desde el hilo de cámara (a través de carrito.notificar()),
        pero modifica widgets de Tkinter que pertenecen al hilo principal.
        En la práctica con Tkinter esto suele funcionar, pero para producción
        sería mejor usar root.after(0, ...) para asegurar thread-safety.
        """
        # Limpiar el listbox completamente
        self.listbox_carrito.delete(0, tk.END)   # 0 = primer elemento, tk.END = último

        # Rellenar con los elementos actuales del carrito
        for clase, item in self.carrito.items.items():
            emoji = item.get("emoji", "📦")

            # Formateamos la línea del ticket:
            # {item['nombre'][:20]:<22} → nombre truncado a 20 chars, alineado a izquierda en 22 cols
            # x{item['cantidad']}       → cantidad
            # ${item['precio'] * item['cantidad']:.2f} → subtotal con 2 decimales
            linea = f"{emoji} {item['nombre'][:20]:<22} x{item['cantidad']}  ${item['precio'] * item['cantidad']:.2f}"
            self.listbox_carrito.insert(tk.END, linea)   # Agrega al final del listbox

        # Actualizar el total y el badge de cantidad
        self.lbl_total.config(text=f"${self.carrito.total():.2f}")
        self.lbl_contador_badge.config(text=str(self.carrito.total_items()))


    def _eliminar_seleccionado(self):
        """
        Elimina del carrito el producto que esté seleccionado en el listbox.

        .curselection() retorna una tupla con los índices seleccionados.
        Como es selección simple, tomamos el primero: sel[0].
        Luego usamos ese índice para obtener la clave correspondiente
        en el diccionario del carrito.
        """
        sel = self.listbox_carrito.curselection()   # Tupla de índices seleccionados
        if not sel:   # Si no hay nada seleccionado, no hacemos nada
            return

        idx = sel[0]   # Índice del elemento seleccionado

        # Obtenemos la clave del carrito en la posición 'idx'
        # list(dict.keys())[i] convierte las claves a lista y toma el elemento i
        clave = list(self.carrito.items.keys())[idx]

        self.carrito.eliminar(clave)   # Eliminamos del carrito (dispara notificar())


    def _vaciar(self):
        """
        Vacía el carrito después de pedir confirmación al usuario.
        messagebox.askyesno abre un diálogo con botones "Sí" / "No" y retorna bool.
        """
        if messagebox.askyesno("Vaciar", "¿Vaciar el carrito?", parent=self.root):
            self.carrito.vaciar()   # Solo vaciamos si el usuario confirmó con "Sí"


    def _cobrar(self):
        """
        Procesa el cobro: genera un ticket, lo guarda en archivo y vacía el carrito.

        Flujo:
        1. Verificar que el carrito no esté vacío
        2. Generar el texto del ticket
        3. Guardarlo en "ticket_ultimo.txt"
        4. Mostrar resumen al usuario
        5. Vaciar el carrito
        """
        if not self.carrito.items:   # Carrito vacío = dict vacío = falsy
            messagebox.showinfo("Caja", "El carrito está vacío.", parent=self.root)
            return

        ticket = self._generar_ticket()   # Genera el texto del ticket

        # Guardar el ticket en archivo de texto
        with open("ticket_ultimo.txt", "w", encoding="utf-8") as f:
            f.write(ticket)

        total       = self.carrito.total()
        items_count = self.carrito.total_items()

        # Mostrar confirmación con resumen de la venta
        messagebox.showinfo(
            "✓ Cobro Completado",
            f"Cobro exitoso!\n\n{items_count} artículos\nTotal: ${total:.2f}\n\nTicket guardado en ticket_ultimo.txt",
            parent=self.root)

        self.carrito.vaciar()   # Reiniciar carrito para la siguiente venta


    def _generar_ticket(self):
        """
        Genera el texto formateado del ticket de compra.

        Retorna:
            str → el ticket completo como una sola cadena de texto
                  con saltos de línea (\n)

        Técnica: construimos una lista de strings y al final los unimos
        con '\n'.join(lista). Es más eficiente que concatenar con +.
        """
        linea = "─" * 40   # Separador horizontal (40 guiones)
        ahora = datetime.datetime.now()   # Fecha y hora actual

        # Construimos las líneas del ticket una por una
        ticket = [
            "=" * 40,
            "      CAJA REGISTRADORA VISION IA",
            f"  {ahora.strftime('%d/%m/%Y  %H:%M:%S')}",  # Fecha formateada
            "=" * 40,
            f"{'PRODUCTO':<25} {'CANT':>4} {'IMPORTE':>9}",  # Encabezados alineados
            linea,
        ]

        # Una línea por cada producto en el carrito
        for _, item in self.carrito.items.items():
            subtotal = item["cantidad"] * item["precio"]
            # Formateo de columnas con f-strings:
            # :<25 → alineado a izquierda en 25 caracteres
            # :>4  → alineado a derecha en 4 caracteres
            # :>8.2f → número con 2 decimales, alineado a derecha en 8 chars
            ticket.append(
                f"{item['nombre'][:24]:<25} {item['cantidad']:>4} ${subtotal:>8.2f}"
            )

        # Pie del ticket
        ticket += [
            linea,
            f"{'TOTAL:':>32} ${self.carrito.total():>7.2f}",
            "=" * 40,
            "    Gracias por su compra!",
            "=" * 40,
        ]

        return "\n".join(ticket)   # Unir todas las líneas con saltos de línea


    def _abrir_productos_json(self):
        """
        Abre el archivo productos.json con el editor de texto predeterminado del sistema.

        Detectamos el sistema operativo con os.name y sys.platform:
        - Windows: os.name == "nt" → usamos os.startfile()
        - macOS: sys.platform contiene "darwin" → usamos subprocess con "open"
        - Linux/otros: usamos subprocess con "xdg-open" (abridor universal)

        subprocess.Popen lanza un proceso externo sin esperar a que termine.
        Esto es diferente de subprocess.run() que sí espera.
        """
        try:
            if not os.path.exists(PRODUCTOS_PATH):
                guardar_productos(DEFAULT_PRODUCTOS_DB)  # Crear si no existe

            if os.name == "nt":
                # Windows: os.startfile abre el archivo con la app asociada
                os.startfile(PRODUCTOS_PATH)   # noqa
            else:
                import subprocess
                if "darwin" in os.sys.platform:
                    # macOS: "open" es el comando de apertura universal
                    subprocess.Popen(["open", PRODUCTOS_PATH])
                else:
                    # Linux: "xdg-open" abre con la app predeterminada del escritorio
                    subprocess.Popen(["xdg-open", PRODUCTOS_PATH])

            messagebox.showinfo(
                "Catálogo",
                f"Se abrió {PRODUCTOS_PATH}.\n\nEdita precios/nombres y reinicia el programa para recargar.",
                parent=self.root)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el archivo: {e}", parent=self.root)


    def iniciar(self):
        """
        Inicia el programa: arranca el loop principal de Tkinter.

        root.mainloop() es un loop bloqueante que:
        1. Espera eventos (clicks, teclas, timers)
        2. Los procesa llamando a los handlers correspondientes
        3. Repinta los widgets que cambiaron
        4. Repite hasta que se cierre la ventana

        El bloque try/finally garantiza que siempre limpiamos recursos
        (cerramos la cámara) incluso si ocurre una excepción.
        """
        try:
            self.root.mainloop()   # BLOQUEA aquí hasta que el usuario cierre la ventana
        finally:
            # Limpieza de recursos al cerrar (se ejecuta siempre)
            self.activo = False       # Señal para que el hilo de cámara termine
            if self.cap:
                self.cap.release()    # Libera la cámara para que otros programas puedan usarla
            cv2.destroyAllWindows()   # Cierra ventanas de OpenCV si hubiera alguna abierta


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA DEL PROGRAMA
# ═══════════════════════════════════════════════════════════════════════════════
#
#  if __name__ == "__main__": es una guarda estándar de Python.
#
#  ¿Para qué sirve?
#  Cuando Python ejecuta un archivo directamente (python caja.py),
#  establece __name__ = "__main__".
#  Cuando el archivo es importado por otro módulo (import caja),
#  __name__ = "caja" (el nombre del módulo).
#
#  Esto nos permite tener código que solo se ejecuta cuando lanzamos
#  el archivo directamente, no cuando lo importamos como librería.
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Mensaje de bienvenida en la consola (informativo para el usuario)
    print("""
╔══════════════════════════════════════════════╗
║   CAJA REGISTRADORA                          ║
║                                              ║
╠══════════════════════════════════════════════╣
║  Requisitos:                                 ║
║    pip install opencv-python ultralytics     ║
║    pip install pillow                        ║
╚══════════════════════════════════════════════╝
""")

    # Crear la aplicación y arrancarla
    # cam_index=1 → cambiar a 0 si tu cámara principal es la integrada
    app = InterfazCaja(cam_index=1)
    app.iniciar()   # Bloquea aquí hasta que el usuario cierre la ventana
