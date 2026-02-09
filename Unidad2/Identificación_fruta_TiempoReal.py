import cv2
import numpy as np
import time
import os

# ============================================================
# CONFIGURACIÓN GENERAL (AJUSTABLE)
# ============================================================

CFG = {
    # ---------- Visualización ----------
    "TARGET_VIEW_W": 1280,        # ancho objetivo para mostrar
    "MAX_VIEW_W": 1600,
    "MAX_VIEW_H": 900,
    "ALLOW_UPSCALE": True,
    "PANEL_H": 95,
    "ALPHA_PANEL": 0.80,
    "FONT": cv2.FONT_HERSHEY_SIMPLEX,

    # ---------- Webcam ----------
    "WEBCAM_RESIZE_W": 1280,      # estandariza el frame para que no “rebalse”
    "WEBCAM_FPS_LIMIT": 20,       # limita FPS para estabilidad
    "DEBUG": False,               # con tecla 'd' lo alternas
    "SAVE_DIR": "capturas",

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
    "USE_CALYX_CHECK": True,
    "CALYX_TOP_FRAC": 0.40,            # en webcam miramos un poco más zona superior
    "CALYX_H_MIN": 35,
    "CALYX_H_MAX": 95,
    "CALYX_MIN_S": 50,
    "CALYX_MIN_V": 40,
    "CALYX_RATIO_MIN": 0.008,          # más tolerante que en modo foto
    "CALYX_WEIGHT": 0.30,              # en webcam: menos dominante

    # Score final "es jitomate"
    "TOMATO_SCORE_MIN": 0.55,          # más tolerante para webcam

    # Regla extra: si no se ve cáliz, deja pasar por color+forma (webcam)
    "PASS_BY_COLOR_IF_NO_CALYX": True,
    "PASS_SHAPE_MIN": 0.58,
    "PASS_RIP_SCORE_MIN": 0.30,

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
    "DEFECT_IGNORE_TOP_FRAC": 0.22,
    "DEFECT_RATIO_MIN": 0.030,
    "DEFECT_DARK_Q": 0.12,

    # ---------- Colores (BGR) ----------
    "C_RED": (0, 0, 255),
    "C_ORANGE": (0, 165, 255),
    "C_GREEN": (0, 255, 0),
    "C_WHITE": (255, 255, 255),
    "C_BLACK": (0, 0, 0),
    "C_YELLOW": (0, 255, 255),
    "C_GRAY": (30, 30, 30),
}

# ============================================================
# UTILIDADES
# ============================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def resize_to_width(img, target_w, allow_upscale=True):
    h, w = img.shape[:2]
    if w <= 0:
        return img
    scale = target_w / float(w)
    if not allow_upscale and scale > 1.0:
        scale = 1.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return img
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

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
    _, mask_s  = cv2.threshold(S,       0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(mask_ab, mask_s)

    k = CFG["KERNEL"]
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CFG["CLOSE_IT"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=CFG["OPEN_IT"])

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
    sol  = metrics["solidity"]
    asp  = metrics["aspect"]
    ext  = metrics["extent"]

    s_circ = np.clip((circ - CFG["TOMATO_MIN_CIRC"]) / (1.0 - CFG["TOMATO_MIN_CIRC"]), 0, 1)
    s_sol  = np.clip((sol  - CFG["TOMATO_MIN_SOLIDITY"]) / (1.0 - CFG["TOMATO_MIN_SOLIDITY"]), 0, 1)
    s_asp  = np.clip((CFG["TOMATO_MAX_ASPECT"] - asp) / (CFG["TOMATO_MAX_ASPECT"] - 1.0), 0, 1)
    s_ext  = np.clip((ext  - CFG["TOMATO_MIN_EXTENT"]) / (1.0 - CFG["TOMATO_MIN_EXTENT"]), 0, 1)

    return float(0.30 * s_circ + 0.30 * s_sol + 0.20 * s_asp + 0.20 * s_ext)

# ============================================================
# CÁLIZ VERDE (sin ML)
# ============================================================

def calyx_green_ratio(img_bgr, fruit_mask, cnt):
    met = contour_metrics(cnt)
    if met is None:
        return 0.0
    x, y, w, h = met["x"], met["y"], met["w"], met["h"]

    top_h = int(h * CFG["CALYX_TOP_FRAC"])
    if top_h < 10:
        return 0.0

    roi = img_bgr[y:y+top_h, x:x+w]
    roi_mask = fruit_mask[y:y+top_h, x:x+w].copy()

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

    return float(np.count_nonzero(green)) / float(total)

def tomato_score_total(shape_score, calyx_ratio):
    if not CFG["USE_CALYX_CHECK"]:
        return float(shape_score)
    calyx_score = np.clip(calyx_ratio / max(CFG["CALYX_RATIO_MIN"], 1e-6), 0, 1)
    w = CFG["CALYX_WEIGHT"]
    return float((1 - w) * shape_score + w * calyx_score)

# ============================================================
# MADUREZ (color) y DAÑO (sin ML)
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
    met = contour_metrics(cnt)
    if met is None:
        return "OK", 0.0
    x, y, w, h = met["x"], met["y"], met["w"], met["h"]

    roi = img_bgr[y:y+h, x:x+w]
    roi_mask = fruit_mask[y:y+h, x:x+w].copy()
    if roi.size == 0:
        return "OK", 0.0

    ignore_top = int(h * CFG["DEFECT_IGNORE_TOP_FRAC"])
    if ignore_top > 0:
        roi_mask[:ignore_top, :] = 0

    inside = (roi_mask > 0)
    total = int(np.count_nonzero(inside))
    if total < 200:
        return "OK", 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
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
    cv2.rectangle(overlay, (0, 0), (img.shape[1], panel_h), CFG["C_GRAY"], -1)
    img[:] = cv2.addWeighted(overlay, CFG["ALPHA_PANEL"], img, 1 - CFG["ALPHA_PANEL"], 0)

    put_label(img, "ANALIZADOR DE CALIDAD (JITOMATE) — WEBCAM", 12, 10,
              CFG["C_WHITE"], CFG["C_GRAY"], scale=0.78, thick=2)

    put_label(img, f"TOTAL DETECTADO: {total}  (OK: {ok_count})", 12, 48,
              CFG["C_WHITE"], CFG["C_GRAY"], scale=0.72, thick=2)

    txt_j = f"JITOMATE: {'SI' if is_tomato else 'NO'}"
    col_j = CFG["C_GREEN"] if is_tomato else CFG["C_YELLOW"]
    put_label(img, txt_j, 520, 48, col_j, CFG["C_GRAY"], scale=0.72, thick=2)

    # Estado global (solo si jitomate)
    if is_tomato:
        estado_col = CFG["C_RED"] if estado_global == "MADURO" else (CFG["C_ORANGE"] if estado_global == "TRANSICION" else CFG["C_GREEN"])
        put_label(img, f"ESTADO: {estado_global}", 740, 48, estado_col, CFG["C_GRAY"], scale=0.72, thick=2)

    put_label(img, f"MADUROS: {maduros}", 12, 78, CFG["C_RED"], CFG["C_GRAY"], scale=0.72, thick=2)
    put_label(img, f"TRANSICION: {trans}", 220, 78, CFG["C_ORANGE"], CFG["C_GRAY"], scale=0.72, thick=2)
    put_label(img, f"INMADUROS: {inmaduros}", 460, 78, CFG["C_GREEN"], CFG["C_GRAY"], scale=0.72, thick=2)

def draw_detection(img, cnt, label, color, idx, quality, score_total,
                   shape_score, calyx_ratio, rip_score, defect_ratio):
    met = contour_metrics(cnt)
    if met is None:
        return
    x, y, w, h = met["x"], met["y"], met["w"], met["h"]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.drawContours(img, [cnt], -1, color, 2)

    # Etiqueta principal
   # line1 = f"{idx}. {label} | {quality} | score={score_total:.2f}"
    y0 = max(CFG["PANEL_H"] + 10, y - 40)
    #put_label(img, line1, x, y0, color, (10, 10, 10), scale=0.72, thick=2)

    # Debug opcional
    #if CFG["DEBUG"]:
     #   line2 = f"shape={shape_score:.2f} calyx={calyx_ratio:.3f} color={rip_score:.2f} defect={defect_ratio:.3f}"
      #  put_label(img, line2, x, y0 + 32, (230, 230, 230), (10, 10, 10), scale=0.58, thick=1)

# ============================================================
# ANÁLISIS (por frame)
# ============================================================

def analyze_frame(img_bgr):
    # Ajuste de vista (webcam)
    img = resize_to_width(img_bgr, CFG["WEBCAM_RESIZE_W"], allow_upscale=False)
    out = img.copy()

    mask = build_object_mask(img)
    cnts = extract_instances(mask)

    if len(cnts) == 0:
        draw_panel(out, 0, 0, 0, 0, 0, False, "NO DETECTADO")
        return out

    # Jitomate global = basado en objeto principal
    main_cnt = cnts[0]
    main_mask = mask_from_contour(out.shape, main_cnt)
    main_met = contour_metrics(main_cnt)

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

        rip_label, color, rip_score = classify_ripeness(out, fruit_mask)

        # --- decisión jitomate / no jitomate (webcam robusta) ---
        tomato_like = score_total >= CFG["TOMATO_SCORE_MIN"]

        if (not tomato_like) and CFG["PASS_BY_COLOR_IF_NO_CALYX"]:
            # Si no hay cáliz pero el color es fuerte y la forma es decente, lo dejamos pasar
            if (shape_score >= CFG["PASS_SHAPE_MIN"]) and (rip_label in ["MADURO", "TRANSICION", "INMADURO"]) and (rip_score >= CFG["PASS_RIP_SCORE_MIN"]):
                tomato_like = True

        if tomato_like:
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
            draw_detection(out, c, "POSIBLE NO-JITOMATE", CFG["C_YELLOW"], i, "N/A",
                           score_total, shape_score, calyx_ratio, rip_score, 0.0)

    # Si en el frame hay al menos 1 jitomate-like, ponemos jitomate global=SI
    # (más útil para escena con varias frutas)
    is_any_tomato = (maduros + trans + inmaduros) > 0
    if is_any_tomato:
        is_tomato = True
        # Estado global: el del principal si era jitomate, si no, el más “fuerte”
        if estado_global == "NO ES JITOMATE":
            estado_global = "DESCONOCIDO"

    draw_panel(out, len(cnts), ok_count, maduros, trans, inmaduros, is_tomato, estado_global)
    return out

# ============================================================
# MAIN WEBCAM
# ============================================================

def main_webcam():
    print("=" * 70)
    print("ANALIZADOR DE CALIDAD (JITOMATE) — WEBCAM")
    print("Teclas: q=salir | s=guardar captura | d=debug ON/OFF")
    print("=" * 70)

    ensure_dir(CFG["SAVE_DIR"])

    # --- CAM_SOURCE ---
    # 0,1,2... si es webcam local
    # o "http://IP:8080/video" si es cam de celular vía IP webcam/DroidCam/Iriun
    CAM_SOURCE = 2

    cap = cv2.VideoCapture(CAM_SOURCE)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    # intenta fijar resolución (no siempre obedece)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "Analizador de Calidad de Frutas"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_t = 0.0
    delay = 1.0 / max(1, CFG["WEBCAM_FPS_LIMIT"])

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("No se pudo leer frame.")
            break

        # limitador simple de FPS
        now = time.time()
        if now - last_t < delay:
            # igual dejamos pasar eventos de teclado
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('d'):
                CFG["DEBUG"] = not CFG["DEBUG"]
                print("DEBUG:", CFG["DEBUG"])
            elif k == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                fn = os.path.join(CFG["SAVE_DIR"], f"captura_{ts}.png")
                # guardamos el frame procesado actual
                out_tmp = analyze_frame(frame)
                cv2.imwrite(fn, out_tmp)
                print("Guardado:", fn)
            continue

        last_t = now

        out = analyze_frame(frame)
        cv2.imshow(win, out)

        # ajusta ventana al tamaño del output
        h, w = out.shape[:2]
        cv2.resizeWindow(win, w, h)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('d'):
            CFG["DEBUG"] = not CFG["DEBUG"]
            print("DEBUG:", CFG["DEBUG"])
        elif k == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(CFG["SAVE_DIR"], f"captura_{ts}.png")
            cv2.imwrite(fn, out)
            print("Guardado:", fn)

    cap.release()
    cv2.destroyAllWindows()
    print("Listo. Programa finalizado.")

if __name__ == "__main__":
    main_webcam()
