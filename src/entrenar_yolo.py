#!/usr/bin/env python3

import argparse
import glob
import os
import shutil

import cv2
import numpy as np

COCO_BALL = 32


def iou(a, b):
    """Intersección sobre unión de dos cajas en formato (cx, cy, w, h)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    i = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def evaluar(modelo, imagenes, clase, imgsz=640, iou_min=0.3):
    """Precisión / recall / F1 (conf 0.25) y AP50 aproximado. IoU 0.3: el balón es chico."""
    from ultralytics import YOLO
    m = YOLO(modelo)
    preds, n_gt = [], 0
    for img in imagenes:
        lab = img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        gt = [list(map(float, l.split()[1:])) for l in open(lab) if l.strip()]
        f = cv2.imread(img)
        H, W = f.shape[:2]
        gt = [(x * W, y * H, w * W, h * H) for x, y, w, h in gt]
        n_gt += len(gt)
        r = m.predict(f, classes=[clase], conf=0.01, imgsz=imgsz, verbose=False)[0]
        usados = set()
        for (x0, y0, x1, y1), c in sorted(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()),
                                          key=lambda z: -z[1]):
            b = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
            ok = False
            for j, g in enumerate(gt):
                # IoU o centro a menos de un diámetro (cajas de balón borroso son imprecisas)
                cerca = np.hypot(b[0] - g[0], b[1] - g[1]) < max(g[2], g[3])
                if j not in usados and (iou(b, g) >= iou_min or cerca):
                    usados.add(j)
                    ok = True
                    break
            preds.append((float(c), ok))
    preds.sort(key=lambda z: -z[0])
    tp = np.cumsum([p[1] for p in preds])
    fp = np.cumsum([not p[1] for p in preds])
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    for r_ in np.linspace(0, 1, 101):                # AP interpolado de 101 puntos (estilo COCO)
        p = prec[rec >= r_]
        ap += (p.max() if len(p) else 0) / 101
    sel = [p for p in preds if p[0] >= 0.25]
    TP = sum(p[1] for p in sel)
    P = TP / max(len(sel), 1)
    R = TP / max(n_gt, 1)
    return dict(precision=round(P, 3), recall=round(R, 3), f1=round(2 * P * R / max(P + R, 1e-9), 3),
                ap50=round(ap, 3), etiquetas=n_gt, predicciones_conf025=len(sel))


def ejemplos(modelos, imagenes, ruta, n=6, imgsz=640):
    """Figura con n imágenes de validación y las detecciones de cada modelo, lado a lado."""
    from ultralytics import YOLO
    ms = [(nombre, YOLO(p), c) for nombre, p, c in modelos]
    con = [i for i in imagenes if os.path.getsize(i.replace("images", "labels").rsplit(".", 1)[0] + ".txt")]
    sel = con[:: max(1, len(con) // n)][:n]
    filas = []
    for img in sel:
        f = cv2.imread(img)
        cols = []
        for nombre, m, c in ms:
            v = f.copy()
            r = m.predict(f, classes=[c], conf=0.25, imgsz=imgsz, verbose=False)[0]
            for (x0, y0, x1, y1), s in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
                cv2.rectangle(v, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(v, f"{s:.2f}", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(v, nombre, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cols.append(cv2.resize(v, (400, 400)))
        filas.append(np.hstack(cols))
    cv2.imwrite(ruta, np.vstack(filas))


def main():
    """Línea de comandos: entrena (o solo evalúa) y compara contra el modelo de COCO."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="dataset/data.yaml")
    ap.add_argument("--base", default="yolo11n.pt", help="pesos pre-entrenados de partida (COCO)")
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="cpu o 0 (GPU CUDA)")
    ap.add_argument("--salida", default="salida")
    ap.add_argument("--solo-evaluar", default=None, help="no entrenar; evaluar estos pesos")
    a = ap.parse_args()
    os.makedirs("modelos", exist_ok=True)
    os.makedirs(a.salida, exist_ok=True)

    final = a.solo_evaluar
    if not final:
        from ultralytics import YOLO
        m = YOLO(a.base)
        m.train(data=a.datos, epochs=a.epocas, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=os.path.abspath(a.salida), name="entrenamiento", exist_ok=True,
                patience=15, workers=2, seed=0, plots=True,
                # aumentaciones: el balón es chico; sin mosaico al final para afinar
                close_mosaic=10, degrees=5, scale=0.3, fliplr=0.5, hsv_v=0.3)
        final = "modelos/handstats_balon.pt"
        shutil.copy(os.path.join(a.salida, "entrenamiento", "weights", "best.pt"), final)
        print("Modelo guardado en", final)

    raiz = os.path.dirname(os.path.abspath(a.datos))
    val = sorted(glob.glob(os.path.join(raiz, "images", "val", "*.jpg")))
    print(f"Evaluando en {len(val)} imágenes de validación...")
    filas = [("COCO pre-entrenado (" + os.path.basename(a.base) + ")", a.base, COCO_BALL),
             ("HandStats fine-tuning", final, 0)]
    with open(os.path.join(a.salida, "comparacion_modelos.csv"), "w") as fh:
        fh.write("modelo,precision,recall,f1,ap50,etiquetas,predicciones_conf025\n")
        for nombre, pesos, clase in filas:
            r = evaluar(pesos, val, clase, a.imgsz)
            print(f"{nombre:40s} {r}")
            fh.write(f"{nombre}," + ",".join(str(v) for v in r.values()) + "\n")
    ejemplos([("COCO", a.base, COCO_BALL), ("fine-tuning", final, 0)], val,
             os.path.join(a.salida, "comparacion_modelos.png"), imgsz=a.imgsz)
    print("Listo ->", os.path.join(a.salida, "comparacion_modelos.csv"))


if __name__ == "__main__":
    main()
