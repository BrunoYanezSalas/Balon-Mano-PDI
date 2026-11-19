#!/usr/bin/env python3
"""
Elegir el cuadro de referencia y marcar las 4 esquinas del arco para pipeline_balon.py.

Controles:
  barra "cuadro"  moverse por el video hasta un cuadro donde el arco se vea completo
  click           marcar esquinas INTERIORES: sup-izq, sup-der, inf-der, inf-izq
  z               deshacer el último click
  s               guardar una captura de la ventana (captura_esquinas.png)
  q / ESC         salir

Al marcar la 4ª esquina imprime el comando listo para copiar.
"""
import argparse
import sys

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--cuadro", type=int, default=0, help="cuadro inicial")
ap.add_argument("--esquinas", nargs=4, help="modo revisión: esquinas x,y ya conocidas")
ap.add_argument("--captura", default=None, help="modo revisión: archivo de salida")
a = ap.parse_args()

video = a.video
cap = cv2.VideoCapture(video)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
if n_total <= 0:
    sys.exit(f"No se pudo leer {video}")
nombres = ["superior izquierda", "superior derecha", "inferior derecha", "inferior izquierda"]
pts = []
estado = {"i": 0, "frame": None}


def leer(i):
    """Lee el cuadro i del video (None si no existe)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    return f if ok else None


def cambiar(i):
    """Callback de la barra: cambia el cuadro de referencia y borra las esquinas marcadas."""
    estado["i"] = i
    estado["frame"] = leer(i)
    pts.clear()


def comando():
    """Imprime --ref/--esquinas y el comando listo para pipeline_balon.py."""
    esq = " ".join(f"{px},{py}" for px, py in pts)
    print(f"\n--ref {estado['i']} --esquinas {esq}\n")
    print("Comando:")
    print(f"python src/pipeline_balon.py {video} --ref {estado['i']} --esquinas {esq} "
          f"--hasta {n_total - 3} --salida salida\n")


def click(ev, x, y, *_):
    """Callback del mouse: agrega una esquina (máximo 4)."""
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y))
        if len(pts) == 4:
            comando()


def vista():
    """Imagen a mostrar: el cuadro actual con las esquinas y el contorno del arco marcados."""
    f = estado["frame"]
    vis = f.copy()
    s = max(1.0, f.shape[1] / 1280)                 # textos legibles en video 1080p
    for k, p in enumerate(pts):
        cv2.circle(vis, p, int(4 * s), (0, 0, 255), -1)
        cv2.putText(vis, str(k + 1), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 255), 2)
    if len(pts) > 1:
        for p, q in zip(pts, pts[1:] + (pts[:1] if len(pts) == 4 else [])):
            cv2.line(vis, p, q, (0, 255, 255), max(1, int(s)))
    txt = f"cuadro {estado['i']} ({estado['i'] / fps:.2f}s)  |  "
    txt += f"click: esquina {nombres[len(pts)]}" if len(pts) < 4 else "listo, revisa la terminal"
    cv2.putText(vis, txt, (10, int(25 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 255, 255), 2)
    return vis


if a.esquinas:                                       # modo revisión, sin ventana
    cambiar(a.cuadro)
    pts.extend(tuple(int(float(v)) for v in e.split(",")) for e in a.esquinas)
    comando()
    cv2.imwrite(a.captura or "revision_esquinas.png", vista())
    print("Imagen guardada en", a.captura or "revision_esquinas.png")
    sys.exit(0)

cv2.namedWindow("arco", cv2.WINDOW_NORMAL)
cv2.createTrackbar("cuadro", "arco", 0, max(1, n_total - 1), cambiar)
cv2.setMouseCallback("arco", click)
cv2.setTrackbarPos("cuadro", "arco", a.cuadro)
cambiar(a.cuadro)
print(f"{n_total} cuadros a {fps:.0f} fps. Busca un cuadro con el arco completo y marca las esquinas.")

while True:
    if estado["frame"] is None:
        break
    vis = vista()
    cv2.imshow("arco", vis)
    k = cv2.waitKey(20) & 0xFF
    if k in (ord("q"), 27):
        break
    if k == ord("z") and pts:
        pts.pop()
    if k == ord("s"):
        cv2.imwrite(a.captura or "captura_esquinas.png", vis)
        print("Captura guardada")
cv2.destroyAllWindows()
