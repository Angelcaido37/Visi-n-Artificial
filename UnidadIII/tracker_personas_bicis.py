# =============================================================================
#  🚶🚲  CONTADOR DE PERSONAS & BICICLETAS — COLORES ÚNICOS POR OBJETO
# =============================================================================
#
#  ¿QUÉ HACE?
#  ─────────────────────────────────────────────────────────────────────────────
#  • Detecta y trackea personas y bicicletas frame a frame
#  • A cada ID único le asigna UN solo color dominante (sin duplicados)
#  • Al finalizar: cuántas personas y bicicletas únicas pasaron, con su color
#
#  SIN línea de conteo — cuenta por IDs únicos que aparecen en el video.
#
#  COLORES ÚNICOS:
#  ─────────────────────────────────────────────────────────────────────────────
#  Cada ID tiene UN color. Se calcula promediando los colores vistos
#  en los primeros frames del objeto (cuando la detección es más estable).
#  Así "Persona #3 = Azul" se cuenta UNA sola vez como Azul, aunque
#  el objeto aparezca en 200 frames.
#
#  INSTALACIÓN:
#    pip install ultralytics opencv-python numpy
#
#  USO:
#    python contador_unico.py --source video.mp4
#    python contador_unico.py --source video.mp4 --no-seg
#    python contador_unico.py --source video.mp4 --save
# =============================================================================

import cv2
import numpy as np
import argparse, sys, time, csv
from collections import defaultdict, Counter
from typing import Dict, Set
from ultralytics import YOLO

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MODEL_DETECT  = "yolov8l.pt"
MODEL_SEG     = "yolov8l-seg.pt"
TARGET        = {0: "Persona", 1: "Bicicleta"}
C_BGR         = {"Persona": (50, 220, 50), "Bicicleta": (50, 180, 255)}
INFER_SZ      = 960  # subido de 416→640: detecta mejor objetos pequeños y borrosos
SEG_EVERY     = 6       # segmentación cada N frames
MASK_ALPHA    = 0.38

# Umbral de confianza diferenciado por clase
# Bicicletas son difíciles: van de frente, parcialmente ocluidas, borrosas
# → necesitan umbral muy bajo para no perderse
CONF_PERSONA   = 0.35
CONF_BICICLETA = 0.12   # muy bajo intencionalmente: mejor falso positivo que no detectar

# Cuántos frames esperar antes de "fijar" el color de un ID
# (más frames = color más estable pero tarda más en aparecer en el panel)
COLOR_STABLE_FRAMES = 5  # bajado de 8→5: bicis aparecen pocos frames

# Rangos HSV para clasificar el color dominante
COLOR_HSV = {
    "Rojo":     [(np.array([0,70,50]),   np.array([10,255,255])),
                 (np.array([160,70,50]), np.array([179,255,255]))],
    "Naranja":  [(np.array([10,80,60]),  np.array([22,255,255]))],
    "Amarillo": [(np.array([22,80,80]),  np.array([34,255,255]))],
    "Verde":    [(np.array([35,50,40]),  np.array([85,255,255]))],
    "Azul":     [(np.array([90,60,40]),  np.array([125,255,255]))],
    "Morado":   [(np.array([125,50,30]), np.array([155,255,255]))],
    "Blanco":   [(np.array([0,0,180]),   np.array([179,40,255]))],
    "Negro":    [(np.array([0,0,0]),     np.array([179,255,50]))],
    "Gris":     [(np.array([0,0,50]),    np.array([179,40,180]))],
}

COLOR_BGR = {
    "Rojo":(40,40,220), "Naranja":(20,130,240), "Amarillo":(20,220,220),
    "Verde":(40,180,40), "Azul":(220,100,40), "Morado":(200,40,180),
    "Blanco":(220,220,220), "Negro":(60,60,60), "Gris":(140,140,140),
}


# =============================================================================
# CLASIFICACIÓN DE COLOR
# =============================================================================

def get_color(frame_bgr: np.ndarray, mask: np.ndarray) -> str:
    """Color dominante del objeto usando solo sus píxeles (máscara)."""
    if mask is None or cv2.countNonZero(mask) < 40:
        return None
    masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    hsv    = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    total  = int(cv2.countNonZero(mask))
    best, bsc = None, 0.0
    for name, ranges in COLOR_HSV.items():
        cm = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            cm = cv2.bitwise_or(cm, cv2.inRange(hsv, lo, hi))
        score = cv2.countNonZero(cv2.bitwise_and(cm, mask)) / total
        if score > bsc:
            bsc, best = score, name
    return best if bsc > 0.08 else None


# =============================================================================
# SEGMENTACIÓN
# =============================================================================

def draw_seg(frame: np.ndarray, pts: np.ndarray, cls: str) -> np.ndarray:
    """Overlay translúcido + retorna máscara binaria."""
    if pts is None or len(pts) < 3:
        return None
    bgr   = C_BGR.get(cls, (180,180,180))
    pts_i = pts.reshape(-1,1,2).astype(np.int32)
    ov    = frame.copy()
    cv2.fillPoly(ov, [pts_i], bgr)
    cv2.addWeighted(ov, MASK_ALPHA, frame, 1-MASK_ALPHA, 0, frame)
    cv2.polylines(frame, [pts_i], True, bgr, 2)
    m = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [pts_i], 255)
    return m


def ellipse_mask(x1,y1,x2,y2, shape) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(m,((x1+x2)//2,(y1+y2)//2),
                (max(1,(x2-x1)//2),max(1,(y2-y1)//2)),0,0,360,255,-1)
    return m


# =============================================================================
# REGISTRO DE OBJETOS ÚNICOS
# =============================================================================

class UniqueRegistry:
    """
    Registro de todos los objetos únicos vistos en el video.

    Por cada ID de ByteTrack almacena:
      - clase (Persona / Bicicleta)
      - votos de color: lista de colores detectados en los primeros frames
      - color_final: el color más votado (se fija tras COLOR_STABLE_FRAMES)

    Al finalizar el video:
      - Conteo de IDs únicos por clase
      - Distribución de colores (1 voto por ID, sin duplicados por frames)
    """

    def __init__(self):
        # {tid: {"cls": str, "votos": [str,...], "color": str|None}}
        self.ids: Dict[int, dict] = {}

    def update(self, tid: int, cls: str, color: str | None) -> None:
        """
        Registra una observación de color para un ID.
        Una vez que el ID tiene COLOR_STABLE_FRAMES votos, el color se fija
        y no se actualiza más (evita que cambie por variaciones de iluminación).
        """
        if tid not in self.ids:
            self.ids[tid] = {"cls": cls, "votos": [], "color": None}

        entry = self.ids[tid]

        # Solo acumular votos hasta que el color esté estabilizado
        if entry["color"] is None and color is not None:
            entry["votos"].append(color)

            # Fijar el color cuando hay suficientes votos
            if len(entry["votos"]) >= COLOR_STABLE_FRAMES:
                # El color más frecuente en los votos = color final del objeto
                entry["color"] = Counter(entry["votos"]).most_common(1)[0][0]

    def counts(self) -> Dict[str, int]:
        """Cuántos IDs únicos hay por clase."""
        result = defaultdict(int)
        for e in self.ids.values():
            result[e["cls"]] += 1
        return dict(result)

    def color_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Distribución de colores únicos por clase.
        Cada ID cuenta UNA sola vez con su color final.
        IDs sin color fijo aún se agrupan como 'Sin clasificar'.
        """
        dist: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for e in self.ids.values():
            cls   = e["cls"]
            color = e["color"] if e["color"] else "Sin clasificar"
            dist[cls][color] += 1
        return {cls: dict(colors) for cls, colors in dist.items()}

    def color_of(self, tid: int) -> str:
        """Color conocido de un ID (para mostrar en la etiqueta)."""
        e = self.ids.get(tid)
        if not e:
            return "?"
        return e["color"] if e["color"] else (
            e["votos"][-1] if e["votos"] else "?")


# =============================================================================
# PANEL DE ESTADÍSTICAS
# =============================================================================

def draw_panel(frame: np.ndarray, reg: UniqueRegistry,
               fps: float, fidx: int, total: int) -> None:
    """
    Panel lateral con conteos únicos y distribución de colores.
    Sin línea de conteo — solo IDs únicos acumulados.
    """
    h, w = frame.shape[:2]
    PW   = 240
    x0   = w - PW + 8
    MG   = w - 8

    ov = frame.copy()
    cv2.rectangle(ov,(w-PW,0),(w,h),(8,8,8),-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    cv2.line(frame,(w-PW,0),(w-PW,h),(50,50,50),1)

    def T(txt, x, y, sc=0.48, col=(210,210,210), bold=False):
        cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX,
                    sc,col,2 if bold else 1,cv2.LINE_AA)

    def NUM(n, x, y, col=(255,255,255)):
        s = str(n)
        (tw,th),_ = cv2.getTextSize(s,cv2.FONT_HERSHEY_SIMPLEX,1.1,2)
        cv2.rectangle(frame,(x-3,y-th-3),(x+tw+3,y+3),(0,0,0),-1)
        cv2.putText(frame,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),2,cv2.LINE_AA)

    y = 26
    T("CONTADOR UNICO", x0, y, 0.54,(100,230,255),True)
    y += 8
    cv2.line(frame,(x0,y),(MG,y),(45,45,45),1)
    y += 18

    counts = reg.counts()
    dist   = reg.color_distribution()
    total_u = sum(counts.values())

    # ── Conteo por clase ──────────────────────────────────────────────────────
    T("OBJETOS UNICOS:", x0, y, 0.42,(150,150,150))
    y += 18

    for cls, bgr in C_BGR.items():
        cnt = counts.get(cls, 0)
        cv2.circle(frame,(x0+7,y-3),6,bgr,-1)
        T(f"  {cls}", x0+6, y, 0.56, bgr, True)
        NUM(cnt, MG-38, y)
        y += 32

    cv2.line(frame,(x0,y),(MG,y),(45,45,45),1)
    y += 14

    T("TOTAL:", x0, y, 0.50,(200,200,200),True)
    NUM(total_u, MG-38, y, (255,220,50))
    y += 28

    # ── Colores únicos por clase ──────────────────────────────────────────────
    cv2.line(frame,(x0,y),(MG,y),(45,45,45),1)
    y += 12
    T("COLOR POR OBJETO", x0, y, 0.42,(130,130,130))
    T("(1 voto/ID)", x0+130, y, 0.36,(80,80,80))
    y += 16

    for cls in TARGET.values():
        colors = dist.get(cls, {})
        if not colors:
            continue
        T(f"{cls}:", x0, y, 0.46, C_BGR.get(cls,(180,180,180)), True)
        y += 16

        # Ordenar por frecuencia
        for cn, cnt in sorted(colors.items(), key=lambda x:-x[1]):
            cb = COLOR_BGR.get(cn,(160,160,160))
            # Barra proporcional al conteo
            bar_max  = 80
            bar_len  = int(bar_max * cnt / max(max(colors.values()),1))
            cv2.rectangle(frame,(x0+4,y-8),(x0+4+bar_len,y-1),cb,-1)
            T(f"  {cn} ({cnt})", x0+90, y, 0.40,(185,185,185))
            y += 14
        y += 6

    # ── FPS y progreso ────────────────────────────────────────────────────────
    cv2.line(frame,(x0,y),(MG,y),(45,45,45),1)
    y += 10
    bw = PW - 16
    cv2.rectangle(frame,(x0,y),(x0+bw,y+6),(30,30,30),-1)
    prog = fidx/max(total,1)
    cv2.rectangle(frame,(x0,y),(x0+int(bw*prog),y+6),(100,230,255),-1)
    y += 14
    T(f"{prog*100:.0f}%  {fidx}/{total}", x0, y, 0.36,(90,90,90))
    y += 16
    cfps = (50,220,50) if fps>=15 else ((50,180,255) if fps>=8 else (40,40,220))
    T(f"FPS: {fps:.1f}", x0, y, 0.46, cfps, fps<15)
    T("Q=Salir  S=Foto", x0, h-10, 0.34,(55,55,55))


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

class Engine:

    def __init__(self, use_seg: bool, seg_every: int):
        print("[INFO] Cargando modelo de detección...")
        self.det = YOLO(MODEL_DETECT)

        self.seg = None
        if use_seg:
            print("[INFO] Cargando modelo de segmentación...")
            self.seg = YOLO(MODEL_SEG)

        self.use_seg   = use_seg
        self.seg_every = seg_every
        self.reg       = UniqueRegistry()
        print("[INFO] ✔ Listo.")

    def run_frame(self, frame: np.ndarray, fidx: int) -> np.ndarray:
        h, w = frame.shape[:2]

        # ── DETECCIÓN + TRACKING ──────────────────────────────────────────────
        det = self.det.track(
            frame,
            imgsz   = INFER_SZ,
            tracker = "bytetrack.yaml",
            persist = True,
            classes = list(TARGET.keys()),
            conf    = 0.15,
            iou     = 0.45,
            verbose = False,
        )[0]

        if det.boxes is None or det.boxes.id is None:
            return frame

        # ── Recopilar todas las detecciones sin filtro de conf aún ─────────────
        raw_persons = []   # [(x1,y1,x2,y2, tid, conf)]
        raw_bicis   = []   # [(x1,y1,x2,y2, tid, conf)]

        for box, tid_t in zip(det.boxes, det.boxes.id):
            tid  = int(tid_t)
            cid  = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w,x2), min(h,y2)
            if cid == 0:   raw_persons.append((x1,y1,x2,y2,tid,conf))
            elif cid == 1: raw_bicis.append((x1,y1,x2,y2,tid,conf))

        # ── LÓGICA DE CICLISTA ────────────────────────────────────────────────
        # Problema: YOLO detecta la bici Y al ciclista como dos objetos separados.
        # Cuando una "Persona" tiene IoU alto con una "Bicicleta" → es el ciclista.
        # En ese caso: contar SOLO como Bicicleta, no como Persona.
        #
        # IoU = Intersection over Union: qué fracción del área se solapa.
        # IoU > 0.25 → la persona está "montada" en la bici (se solapan mucho).

        def iou(a, b):
            """Intersection over Union entre dos cajas (x1,y1,x2,y2)."""
            ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
            ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
            area_a = (a[2]-a[0])*(a[3]-a[1])
            area_b = (b[2]-b[0])*(b[3]-b[1])
            union  = area_a + area_b - inter
            return inter/union if union>0 else 0

        # IDs de personas que son en realidad ciclistas (solapan con bici)
        ciclistas_ids = set()
        for pb in raw_bicis:
            for pp in raw_persons:
                if iou(pb[:4], pp[:4]) > 0.20:
                    ciclistas_ids.add(pp[4])  # pp[4] = tid de la persona

        # Construir mapa final de objetos con conf diferenciada
        objs = {}
        for (x1,y1,x2,y2,tid,conf) in raw_persons:
            if tid in ciclistas_ids: continue          # saltar: es ciclista
            if conf < CONF_PERSONA:  continue          # filtro de confianza
            objs[tid] = (x1,y1,x2,y2,"Persona",conf)

        for (x1,y1,x2,y2,tid,conf) in raw_bicis:
            if conf < CONF_BICICLETA: continue         # umbral muy bajo
            objs[tid] = (x1,y1,x2,y2,"Bicicleta",conf)

        # ── SEGMENTACIÓN (cada SEG_EVERY frames) ─────────────────────────────
        seg_masks: Dict[int, np.ndarray] = {}
        if self.use_seg and self.seg and fidx % self.seg_every == 0:
            sr = self.seg.track(
                frame,
                imgsz   = INFER_SZ,
                tracker = "bytetrack.yaml",
                persist = True,
                classes = list(TARGET.keys()),
                conf    = 0.15,
                iou     = 0.45,
                agnostic_nms = True,
                verbose = False,
            )[0]
            if (sr.boxes is not None and sr.boxes.id is not None
                    and sr.masks is not None):
                for i,(box,tid_t) in enumerate(zip(sr.boxes,sr.boxes.id)):
                    tid = int(tid_t)
                    cls = TARGET.get(int(box.cls[0]),"?")
                    if i < len(sr.masks.xy):
                        pts = np.array(sr.masks.xy[i],dtype=np.float32)
                        m = draw_seg(frame,pts,cls)
                        if m is not None:
                            seg_masks[tid] = m

        # ── COLOR + REGISTRO + ANOTACIÓN ─────────────────────────────────────
        for tid,(x1,y1,x2,y2,cls,conf) in objs.items():

            # Obtener máscara para análisis de color
            seg_m = seg_masks.get(tid)
            mask  = seg_m if seg_m is not None else ellipse_mask(x1,y1,x2,y2,frame.shape)

            # Solo clasificar color si el ID aún no tiene color fijo
            entry = self.reg.ids.get(tid)
            if entry is None or entry["color"] is None:
                color_obs = get_color(frame, mask)
                self.reg.update(tid, cls, color_obs)

            # Color a mostrar en la etiqueta
            display_color = self.reg.color_of(tid)

            # ── Dibujar bbox y etiqueta ───────────────────────────────────────
            bgr = C_BGR.get(cls,(200,200,200))
            cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)

            # Esquinas en L
            L=10
            for sx,sy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(sx,sy),(sx+dx*L,sy),bgr,3)
                cv2.line(frame,(sx,sy),(sx,sy+dy*L),bgr,3)

            # Punto de referencia (pies)
            cv2.circle(frame,((x1+x2)//2,y2),4,bgr,-1)

            lbl = f"#{tid} {cls[:3]} {display_color} {conf:.0%}"
            sc=0.46
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,sc,1)
            ty = max(y1-4,th+4)
            cv2.rectangle(frame,(x1,ty-th-2),(x1+tw+4,ty+1),bgr,-1)
            cv2.putText(frame,lbl,(x1+2,ty),cv2.FONT_HERSHEY_SIMPLEX,
                        sc,(10,10,10),1,cv2.LINE_AA)

        return frame


# =============================================================================
# MAIN
# =============================================================================

def run(source, use_seg, seg_every, save):

    print("\n"+"═"*58)
    print("  🚶🚲  CONTADOR ÚNICO — PERSONAS & BICICLETAS")
    print("="*58)

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] No se puede abrir: {source}")

    fps_v  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video : {fw}×{fh} @ {fps_v:.0f}FPS | {total} frames ({total/fps_v:.1f}s)")
    print(f"[INFO] Seg   : {'cada '+str(seg_every)+' frames' if use_seg else 'DESACTIVADA'}")
    print("[INFO] Teclas: Q/ESC=Salir  S=Snapshot\n")

    engine = Engine(use_seg, seg_every)

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("resultado_contador.mp4",fourcc,fps_v,(fw,fh))

    ms_frame = int(1000/fps_v)
    fidx=0; proc_fps=0.0; t_fps=time.time(); snaps=0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] Fin del video.")
                break

            fidx += 1
            frame = engine.run_frame(frame, fidx)

            if fidx % 15 == 0:
                el=time.time()-t_fps
                proc_fps=15/el if el>0 else 0
                t_fps=time.time()

            draw_panel(frame, engine.reg, proc_fps, fidx, total)

            cv2.imshow("Contador Unico  [Q=Salir | S=Foto]", frame)
            if writer: writer.write(frame)

            wait = max(1, ms_frame - int((time.time()-t0)*1000))
            key  = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"),ord("Q"),27): break
            elif key in (ord("s"),ord("S")):
                snaps+=1
                p=f"snap_{snaps:03d}.jpg"
                cv2.imwrite(p,frame)
                print(f"[INFO] Snapshot: {p}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer: writer.release()

    cv2.destroyAllWindows()

    # ── REPORTE FINAL ──────────────────────────────────────────────────────────
    counts = engine.reg.counts()
    dist   = engine.reg.color_distribution()

    print("\n"+"═"*58)
    print("  REPORTE FINAL — OBJETOS ÚNICOS")
    print("═"*58)
    print(f"\n  Total IDs únicos: {sum(counts.values())}\n")

    for cls in TARGET.values():
        icon = "🚶" if cls=="Persona" else "🚲"
        cnt  = counts.get(cls,0)
        print(f"  {icon}  {cls:<12}  {cnt} objetos únicos")
        for cn,c in sorted(dist.get(cls,{}).items(),key=lambda x:-x[1]):
            bar = "█"*min(c,30)
            print(f"       {bar:<30} {cn} ({c} IDs)")
        print()

    # Exportar CSV con un registro por ID
    csv_path = "reporte_unicos.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        dw = csv.DictWriter(f,fieldnames=["id","clase","color"])
        dw.writeheader()
        for tid,e in sorted(engine.reg.ids.items()):
            dw.writerow({"id":tid,"clase":e["cls"],
                         "color":e["color"] or "Sin clasificar"})
    print(f"  CSV: {csv_path}  ({len(engine.reg.ids)} registros)")
    print("═"*58+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Contador único de personas y bicicletas")
    p.add_argument("--source",    default="video.mp4")
    p.add_argument("--no-seg",    action="store_true", help="Sin segmentación (más rápido)")
    p.add_argument("--seg-every", type=int, default=SEG_EVERY)
    p.add_argument("--save",      action="store_true")
    args = p.parse_args()
    run(args.source, not args.no_seg, args.seg_every, args.save)