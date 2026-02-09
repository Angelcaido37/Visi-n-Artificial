import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ============================================================
# CONFIGURACIÓN GENERAL (AJUSTABLE)
# ============================================================

CFG = {
    # ---------- Visualización ----------
    "TARGET_VIEW_W": 1200,
    "MAX_VIEW_W": 1600,
    "MAX_VIEW_H": 900,
    "ALLOW_UPSCALE": True,
    "PANEL_H": 95,
    "ALPHA_PANEL": 0.80,
    "FONT": cv2.FONT_HERSHEY_SIMPLEX,

    # ---------- Segmentación ----------
    "KERNEL": 5,
    "OPEN_IT": 2,
    "CLOSE_IT": 3,
    "MIN_AREA": 2500,
    "MIN_EXTENT": 0.35,
    "BORDER_FRAC": 0.06,

    # ---------- Forma: score (Unidad 1) ----------
    "TOMATO_MIN_SOLIDITY": 0.88,
    "TOMATO_MIN_CIRC": 0.42,
    "TOMATO_MAX_ASPECT": 1.65,
    "TOMATO_MIN_EXTENT": 0.55,
    "TOMATO_SHAPE_SCORE_MIN": 0.55,     # mínimo para considerar "buena forma"

    # ---------- Cáliz verde (diferencia jitomate vs manzana) ----------
    # Si el jitomate NO muestra cáliz, puedes bajar umbral, o apagarlo (ver abajo).
    "USE_CALYX_CHECK": True,
    "CALYX_TOP_FRAC": 0.30,            # solo mirar banda superior del fruto
    "CALYX_H_MIN": 35,
    "CALYX_H_MAX": 95,
    "CALYX_MIN_S": 50,
    "CALYX_MIN_V": 40,
    "CALYX_RATIO_MIN": 0.015,          # % mínimo de verde en la parte superior
    "CALYX_WEIGHT": 0.55,              # peso del cáliz en score total (más alto = menos manzanas)

    # Score final "es jitomate"
    "TOMATO_SCORE_MIN": 0.62,

    # ---------- Color (madurez) ----------
    "MIN_S": 35,
    "MIN_V": 35,
    "H_RED1_MAX": 10,
    "H_RED2_MIN": 165,
    "H_ORANGE_MIN": 10,
    "H_ORANGE_MAX": 25,
    "H_YELLOW_MIN": 25,
    "H_YELLOW_MAX": 40,
    "H_GREEN_MIN": 35,
    "H_GREEN_MAX": 95,

    # ---------- Daño (defectos oscuros) ----------
    # Mejorado: ignora la banda superior (tallo/cáliz) para no marcar daño falso.
    "DEFECT_IGNORE_TOP_FRAC": 0.22,
    "DEFECT_RATIO_MIN": 0.030,         # umbral final
    "DEFECT_DARK_Q": 0.12,             # percentil para "oscuro" (adaptativo)

    # ---------- Colores (BGR) ----------
    "C_RED": (0, 0, 255),
    "C_ORANGE": (0, 165, 255),
    "C_GREEN": (0, 255, 0),
    "C_WHITE": (255, 255, 255),
    "C_BLACK": (0, 0, 0),
    "C_YELLOW": (0, 255, 255),
}


# ============================================================
# UTILIDADES
# ============================================================

def seleccionar_imagen():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    root.destroy()
    return path


def resize_for_view(img):
    """Estandariza la vista (agranda o reduce) para que SIEMPRE se vea bien."""
    h, w = img.shape[:2]
    target_w = CFG["TARGET_VIEW_W"]
    scale = target_w / float(w)

    if not CFG["ALLOW_UPSCALE"] and scale > 1.0:
        scale = 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)

    max_w, max_h = CFG["MAX_VIEW_W"], CFG["MAX_VIEW_H"]
    fit_scale = min(max_w / float(new_w), max_h / float(new_h), 1.0)
    new_w = int(new_w * fit_scale)
    new_h = int(new_h * fit_scale)

    if new_w != w or new_h != h:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    return img


def wait_until_close(win_name):
    """Espera hasta cerrar ventana (X) o presionar una tecla."""
    while True:
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            return None
        k = cv2.waitKey(30)
        if k != -1:
            return k


def put_label(img, text, x, y, fg, bg, scale=0.65, thick=2, pad=6):
    """Texto con fondo (legible/estético)."""
    font = CFG["FONT"]
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x2 = x + tw + pad * 2
    y2 = y + th + pad * 2
    cv2.rectangle(img, (x, y), (x2, y2), bg, -1)
    cv2.putText(img, text, (x + pad, y + th + pad - 2),
                font, scale, fg, thick, cv2.LINE_AA)


# ============================================================
# MÉTRICAS DE FORMA (Unidad 1)
# ============================================================

def contour_metrics(cnt):
    area = cv2.contourArea(cnt)
    if area <= 0:
        return None

    peri = cv2.arcLength(cnt, True)
    if peri <= 0:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = max(1, w * h)

    circ = (4.0 * np.pi * area) / (peri * peri)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    extent = area / bbox_area
    aspect = max(w, h) / max(1, min(w, h))

    return {"area": area, "x": x, "y": y, "w": w, "h": h,
            "circ": circ, "solidity": solidity, "extent": extent, "aspect": aspect}


def mask_from_contour(shape, cnt):
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.drawContours(m, [cnt], -1, 255, -1)
    return m


# ============================================================
# SEGMENTACIÓN ROBUSTA (sin ML)
# ============================================================

def build_object_mask(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    A = lab[:, :, 1]
    B = lab[:, :, 2]
    S = hsv[:, :, 1]

    h, w = img_bgr.shape[:2]
    m = max(5, int(min(h, w) * CFG["BORDER_FRAC"]))

    border = np.zeros((h, w), dtype=np.uint8)
    border[:m, :] = 1
    border[-m:, :] = 1
    border[:, :m] = 1
    border[:, -m:] = 1

    bgA = float(np.mean(A[border == 1]))
    bgB = float(np.mean(B[border == 1]))

    dist_ab = np.sqrt((A.astype(np.float32) - bgA) ** 2 + (B.astype(np.float32) - bgB) ** 2)
    dist_u8 = cv2.normalize(dist_ab, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask_ab = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_s = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(mask_ab, mask_s)

    k = CFG["KERNEL"]
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CFG["CLOSE_IT"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=CFG["OPEN_IT"])

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < int(CFG["MIN_AREA"] * 0.35):
            mask[labels == i] = 0

    return mask


def extract_instances(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good = []
    for c in cnts:
        m = contour_metrics(c)
        if m is None:
            continue
        if m["area"] < CFG["MIN_AREA"]:
            continue
        if m["extent"] < CFG["MIN_EXTENT"]:
            continue
        good.append(c)
    good.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    return good


# ============================================================
# SCORE FORMA (Unidad 1)
# ============================================================

def tomato_shape_score(metrics):
    circ = metrics["circ"]
    sol = metrics["solidity"]
    asp = metrics["aspect"]
    ext = metrics["extent"]

    s_circ = np.clip((circ - CFG["TOMATO_MIN_CIRC"]) / (1.0 - CFG["TOMATO_MIN_CIRC"]), 0, 1)
    s_sol  = np.clip((sol - CFG["TOMATO_MIN_SOLIDITY"]) / (1.0 - CFG["TOMATO_MIN_SOLIDITY"]), 0, 1)
    s_asp  = np.clip((CFG["TOMATO_MAX_ASPECT"] - asp) / (CFG["TOMATO_MAX_ASPECT"] - 1.0), 0, 1)
    s_ext  = np.clip((ext - CFG["TOMATO_MIN_EXTENT"]) / (1.0 - CFG["TOMATO_MIN_EXTENT"]), 0, 1)

    return float(0.30 * s_circ + 0.30 * s_sol + 0.20 * s_asp + 0.20 * s_ext)


# ============================================================
# CÁLIZ VERDE (diferencia jitomate vs manzana roja) — SIN ML
# ============================================================

def calyx_green_ratio(img_bgr, fruit_mask, cnt):
    """
    Busca 'verde' (cáliz) en la parte superior del fruto.
    Manzana roja: casi siempre ratio ~ 0.
    Jitomate con cáliz visible: ratio > umbral.
    """
    met = contour_metrics(cnt)
    x, y, w, h = met["x"], met["y"], met["w"], met["h"]

    top_h = int(h * CFG["CALYX_TOP_FRAC"])
    if top_h < 10:
        return 0.0

    # ROI: parte superior del bbox
    roi = img_bgr[y:y+top_h, x:x+w]
    roi_mask = fruit_mask[y:y+top_h, x:x+w]

    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    inside = (roi_mask > 0)
    total = int(np.count_nonzero(inside))
    if total < 120:
        return 0.0

    green = inside & \
            (H >= CFG["CALYX_H_MIN"]) & (H <= CFG["CALYX_H_MAX"]) & \
            (S >= CFG["CALYX_MIN_S"]) & (V >= CFG["CALYX_MIN_V"])

    ratio = float(np.count_nonzero(green)) / float(total)
    return ratio


def tomato_score_total(shape_score, calyx_ratio):
    """
    Score total = mezcla de forma + cáliz.
    Si USE_CALYX_CHECK=False, se usa solo forma.
    """
    if not CFG["USE_CALYX_CHECK"]:
        return float(shape_score)

    # Convertimos calyx_ratio a score [0..1] “suavizado”
    # ratio>=CALYX_RATIO_MIN => calyx_score sube rápido.
    calyx_score = np.clip(calyx_ratio / max(CFG["CALYX_RATIO_MIN"], 1e-6), 0, 1)

    w = CFG["CALYX_WEIGHT"]  # peso del cáliz
    return float((1 - w) * shape_score + w * calyx_score)


# ============================================================
# MADUREZ (color) y DAÑO (mejorado)
# ============================================================

def classify_ripeness(img_bgr, fruit_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    valid = (fruit_mask > 0) & (S > CFG["MIN_S"]) & (V > CFG["MIN_V"])
    total = int(np.count_nonzero(valid))
    if total < 150:
        return "DESCONOCIDO", CFG["C_YELLOW"], 0.0

    h_vals = H[valid]

    p_red = np.mean((h_vals <= CFG["H_RED1_MAX"]) | (h_vals >= CFG["H_RED2_MIN"]))
    p_or  = np.mean((h_vals > CFG["H_ORANGE_MIN"]) & (h_vals <= CFG["H_ORANGE_MAX"]))
    p_ye  = np.mean((h_vals > CFG["H_YELLOW_MIN"]) & (h_vals <= CFG["H_YELLOW_MAX"]))
    p_gr  = np.mean((h_vals >= CFG["H_GREEN_MIN"]) & (h_vals <= CFG["H_GREEN_MAX"]))
    p_trans = p_or + p_ye

    if p_red >= 0.28 and p_red >= p_trans and p_red >= p_gr:
        return "MADURO", CFG["C_RED"], float(p_red)
    if p_gr >= 0.35 and p_gr >= p_red and p_gr >= p_trans:
        return "INMADURO", CFG["C_GREEN"], float(p_gr)
    if p_trans >= 0.22 and p_trans >= p_red and p_trans >= p_gr:
        return "TRANSICION", CFG["C_ORANGE"], float(p_trans)

    mx = max(p_red, p_trans, p_gr)
    if mx == p_red and p_red > 0.18:
        return "MADURO", CFG["C_RED"], float(p_red)
    if mx == p_gr and p_gr > 0.18:
        return "INMADURO", CFG["C_GREEN"], float(p_gr)
    if mx == p_trans and p_trans > 0.18:
        return "TRANSICION", CFG["C_ORANGE"], float(p_trans)

    return "DESCONOCIDO", CFG["C_YELLOW"], float(mx)


def detect_damage(img_bgr, fruit_mask, cnt):
    """
    Daño adaptativo (sin ML), ignorando la zona superior del fruto
    donde suele estar tallo/cáliz (para evitar falsos positivos).
    """
    met = contour_metrics(cnt)
    x, y, w, h = met["x"], met["y"], met["w"], met["h"]

    # ROI del fruto
    roi = img_bgr[y:y+h, x:x+w]
    roi_mask = fruit_mask[y:y+h, x:x+w]
    if roi.size == 0:
        return "OK", 0.0

    # Ignorar banda superior
    ignore_top = int(h * CFG["DEFECT_IGNORE_TOP_FRAC"])
    if ignore_top > 0:
        roi_mask[:ignore_top, :] = 0

    inside = (roi_mask > 0)
    total = int(np.count_nonzero(inside))
    if total < 200:
        return "OK", 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    # Umbral adaptativo con percentil dentro del fruto
    v_vals = V[inside]
    q = np.quantile(v_vals, CFG["DEFECT_DARK_Q"])
    dark = inside & (V <= int(q))

    ratio = float(np.count_nonzero(dark)) / float(total)

    if ratio >= CFG["DEFECT_RATIO_MIN"]:
        return "DAÑADO", ratio
    return "OK", ratio


# ============================================================
# DIBUJO / UI
# ============================================================

def draw_panel(img, total, ok_count, maduros, trans, inmaduros, is_tomato, estado_global):
    panel_h = CFG["PANEL_H"]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], panel_h), (30, 30, 30), -1)
    img[:] = cv2.addWeighted(overlay, CFG["ALPHA_PANEL"], img, 1 - CFG["ALPHA_PANEL"], 0)

    put_label(img, "ANALIZADOR DE CALIDAD (JITOMATE) — MODO FOTO", 12, 10,
              CFG["C_WHITE"], (30, 30, 30), scale=0.75, thick=2)

    put_label(img, f"TOTAL DETECTADO: {total}  (OK: {ok_count})", 12, 48,
              CFG["C_WHITE"], (30, 30, 30), scale=0.70, thick=2)

    txt_j = f"JITOMATE: {'SI' if is_tomato else 'NO'}"
    col_j = CFG["C_GREEN"] if is_tomato else CFG["C_YELLOW"]
    put_label(img, txt_j, 520, 48, col_j, (30, 30, 30), scale=0.70, thick=2)

    estado_col = CFG["C_RED"] if "MADURO" in estado_global else (CFG["C_ORANGE"] if "TRANSICION" in estado_global else CFG["C_GREEN"])
   # put_label(img, f"ESTADO: {estado_global}", 760, 48, estado_col, (30, 30, 30), scale=0.70, thick=2)

    put_label(img, f"MADUROS: {maduros}", 12, 78, CFG["C_RED"], (30, 30, 30), scale=0.70, thick=2)
    put_label(img, f"TRANSICION: {trans}", 210, 78, CFG["C_ORANGE"], (30, 30, 30), scale=0.70, thick=2)
    put_label(img, f"INMADUROS: {inmaduros}", 440, 78, CFG["C_GREEN"], (30, 30, 30), scale=0.70, thick=2)


def draw_detection(img, cnt, label, color, idx, quality, score_total, shape_score, calyx_ratio, rip_score, defect_ratio):
    met = contour_metrics(cnt)
    x, y, w, h = met["x"], met["y"], met["w"], met["h"]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.drawContours(img, [cnt], -1, color, 2)

    # Etiquetas claras (2 renglones)
  #  line1 = f"{idx}. {label} | {quality} | score={score_total:.2f}"
   # line2 = f"shape={shape_score:.2f} calyx={calyx_ratio:.3f} color={rip_score:.2f} defect={defect_ratio:.3f}"

    y0 = max(CFG["PANEL_H"] + 10, y - 70)
    #put_label(img, line1, x, y0, color, (10, 10, 10), scale=0.72, thick=2)
    #put_label(img, line2, x, y0 + 32, (230, 230, 230), (10, 10, 10), scale=0.58, thick=1)


# ============================================================
# ANÁLISIS
# ============================================================

def analyze_image(img_bgr):
    img = resize_for_view(img_bgr)
    out = img.copy()

    mask = build_object_mask(img)
    cnts = extract_instances(mask)

    if len(cnts) == 0:
        draw_panel(out, 0, 0, 0, 0, 0, False, "NO DETECTADO")
        return out, {"total": 0, "ok": 0, "maduros": 0, "trans": 0, "inmaduros": 0,
                     "jitomate": False, "estado": "NO DETECTADO"}

    # Para definir "jitomate global", usamos el objeto principal
    main_cnt = cnts[0]
    main_met = contour_metrics(main_cnt)
    main_mask = mask_from_contour(out.shape, main_cnt)

    shape_sc = tomato_shape_score(main_met)
    calyx_r = calyx_green_ratio(out, main_mask, main_cnt)
    total_sc = tomato_score_total(shape_sc, calyx_r)

    is_tomato = total_sc >= CFG["TOMATO_SCORE_MIN"]

    estado_global = "NO ES JITOMATE"
    if is_tomato:
        estado_global, _, _ = classify_ripeness(out, main_mask)

    maduros = trans = inmaduros = 0
    ok_count = 0

    for i, c in enumerate(cnts, 1):
        met = contour_metrics(c)
        if met is None:
            continue

        fruit_mask = mask_from_contour(out.shape, c)

        shape_score = tomato_shape_score(met)
        calyx_ratio = calyx_green_ratio(out, fruit_mask, c)
        score_total = tomato_score_total(shape_score, calyx_ratio)

        tomato_like = score_total >= CFG["TOMATO_SCORE_MIN"]

        if tomato_like:
            rip_label, color, rip_score = classify_ripeness(out, fruit_mask)
            quality, defect_ratio = detect_damage(out, fruit_mask, c)

            if rip_label == "MADURO":
                maduros += 1
            elif rip_label == "TRANSICION":
                trans += 1
            elif rip_label == "INMADURO":
                inmaduros += 1

            if quality == "OK":
                ok_count += 1

            draw_detection(out, c, rip_label, color, i, quality,
                           score_total, shape_score, calyx_ratio, rip_score, defect_ratio)

        else:
            # Si parece fruta redonda y roja pero sin cáliz => probable manzana
            # (etiqueta clara)
            draw_detection(out, c, "POSIBLE NO-JITOMATE", CFG["C_YELLOW"], i, "N/A",
                           score_total, shape_score, calyx_ratio, 0.0, 0.0)

    draw_panel(out, len(cnts), ok_count, maduros, trans, inmaduros, is_tomato, estado_global)

    return out, {"total": len(cnts), "ok": ok_count, "maduros": maduros, "trans": trans, "inmaduros": inmaduros,
                 "jitomate": is_tomato, "estado": estado_global}


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ANALIZADOR DE CALIDAD (JITOMATE) ")
    print("=" * 70)

    while True:
        path = seleccionar_imagen()
        if not path:
            print("No seleccionaste imagen. Saliendo.")
            break

        img = cv2.imread(path)
        if img is None:
            print("Error: no se pudo leer la imagen.")
            continue

        out, info = analyze_image(img)

        win = "Analizador de Calidad de Frutas"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, out)

        h, w = out.shape[:2]
        cv2.resizeWindow(win, w, h)
        cv2.moveWindow(win, 50, 50)

        _ = wait_until_close(win)
        cv2.destroyAllWindows()

        print("\nResumen:")
        print(f"  Total detectado: {info['total']}   OK: {info['ok']}")
        print(f"  Jitomate: {'SI' if info['jitomate'] else 'NO'}"  )
        print(f"  Maduros: {info['maduros']}  Transición: {info['trans']}  Inmaduros: {info['inmaduros']}")

        cont = input("\n¿Deseas cargar otra imagen? (s/n): ").strip().lower()
        if cont != "s":
            break

    print("\nListo. Programa finalizado.")


if __name__ == "__main__":
    main()
