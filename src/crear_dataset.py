#!/usr/bin/env python3
import argparse
import glob
import os
import random

import cv2
import numpy as np

COCO_BALL = 32
TAM = 640


def detectar(modelo, video, conf, imgsz):
    """Corre el YOLO "profesor" sobre todos los cuadros del video.

    Devuelve la lista de detecciones de la clase "sports ball" por cuadro y el número de cuadros."""
    from ultralytics import YOLO
    m = YOLO(modelo)
    cap = cv2.VideoCapture(video)
    dets, n = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = m.predict(f, classes=[COCO_BALL], conf=conf, imgsz=imgsz, verbose=False)[0]
        dets.append([(float((x1 + x2) / 2), float((y1 + y2) / 2), float(x2 - x1), float(y2 - y1), float(c))
                     for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())])
        n += 1
    return dets, cap.get(cv2.CAP_PROP_FPS) or 30


def pistas(dets, gate, max_hueco=3):
    """Une detecciones cuadro a cuadro (vecino más cercano a la posición predicha)."""
    activas, cerradas = [], []
    for t, D in enumerate(dets):
        usados = set()
        for tr in activas:
            tl, x, y, w, h, _ = tr[-1]
            if len(tr) >= 2:
                tp, xp, yp = tr[-2][:3]
                x, y = x + (x - xp) / (tl - tp) * (t - tl), y + (y - yp) / (tl - tp) * (t - tl)
            mejor = None
            for j, d in enumerate(D):
                dist = np.hypot(d[0] - x, d[1] - y)
                if j not in usados and dist < gate * (t - tl) and (mejor is None or dist < mejor[0]):
                    mejor = (dist, j)
            if mejor:
                usados.add(mejor[1])
                tr.append((t, *D[mejor[1]]))
        sig = []
        for tr in activas:
            (cerradas if t - tr[-1][0] > max_hueco else sig).append(tr)
        activas = sig + [[(t, *d)] for j, d in enumerate(D) if j not in usados]
    return cerradas + activas


def etiquetas(dets, w, fps, min_len=5, min_mov=None):
    """Pistas válidas -> {cuadro: (x, y, ancho, alto, interpolado)}."""
    min_mov = min_mov or 0.04 * w
    out = {}
    for tr in pistas(dets, gate=0.06 * w):
        if len(tr) < min_len:
            continue
        p = np.array([q[:5] for q in tr])
        if np.ptp(p[:, 1]) + np.ptp(p[:, 2]) < min_mov:      # pista quieta: falso positivo
            continue
        if np.median(p[:, 4]) > 0.08 * w:                  # demasiado grande para ser el balón
            continue
        for a, b in zip(tr, tr[1:]):
            out[a[0]] = (*a[1:5], False)
            for t in range(a[0] + 1, b[0]):                # relleno de huecos
                s = (t - a[0]) / (b[0] - a[0])
                out[t] = (*(np.array(a[1:5]) * (1 - s) + np.array(b[1:5]) * s), True)
        out[tr[-1][0]] = (*tr[-1][1:5], False)
    return out


def recorte(f, cx, cy, rng):
    """Recorte cuadrado de lado TAM que contiene el punto (cx, cy) en una posición aleatoria.

    Devuelve el recorte y su esquina superior izquierda (x0, y0) y lado t, para mover las etiquetas."""
    h, w = f.shape[:2]
    t = min(TAM, w, h)
    x0 = int(np.clip(cx - rng.uniform(0.2, 0.8) * t, 0, w - t))
    y0 = int(np.clip(cy - rng.uniform(0.2, 0.8) * t, 0, h - t))
    return f[y0:y0 + t, x0:x0 + t], x0, y0, t


def mosaico(carpeta, salida):
    """Mosaicos de revisión: recorte de 96x96 alrededor de cada etiqueta, numerado."""
    filas = []
    for img in sorted(glob.glob(f"{carpeta}/images/*/*.jpg")):
        lab = img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if not os.path.getsize(lab):
            continue
        idx = int(os.path.basename(img).split("_")[0])
        f = cv2.imread(img)
        _, x, y, bw, bh = map(float, open(lab).read().split()[:5])
        x, y = int(x * f.shape[1]), int(y * f.shape[0])
        c = cv2.copyMakeBorder(f, 48, 48, 48, 48, cv2.BORDER_CONSTANT)[y:y + 96, x:x + 96]
        c = cv2.resize(c, (144, 144), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, str(idx), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        filas.append(c)
    for k in range(0, len(filas), 100):
        bloque = filas[k:k + 100] + [np.zeros((144, 144, 3), np.uint8)] * (-len(filas[k:k + 100]) % 10)
        m = np.vstack([np.hstack(bloque[i:i + 10]) for i in range(0, len(bloque), 10)])
        cv2.imwrite(f"{salida}/revision_{k // 100:02d}.jpg", m)
    return len(filas)


def agregar_extra(a, rng):
    """Agrega al dataset las etiquetas de los CSV --extra (pipeline clásico o hechas a mano).

    Cada balón se guarda en --copias recortes distintos; los videos de --val van a validación."""
    import csv
    filas = []
    for ruta in a.extra:
        filas += list(csv.DictReader(open(ruta, encoding="utf-8")))
    previos = glob.glob(f"{a.salida}/images/*/*.jpg")
    idx = max([int(os.path.basename(p).split("_")[0]) for p in previos] + [-1]) + 1
    for s in ("train", "val"):
        os.makedirs(f"{a.salida}/images/{s}", exist_ok=True)
        os.makedirs(f"{a.salida}/labels/{s}", exist_ok=True)
    por_video = {}
    for r in filas:
        por_video.setdefault(r["video"], []).append(r)
    for video, rs in por_video.items():
        nombre = os.path.splitext(video)[0]
        split = "val" if a.val and a.val in video else "train"
        quiero = {int(r["frame"]): r for r in rs}
        cap = cv2.VideoCapture(os.path.join(a.carpeta_videos, video))
        t, n = 0, 0
        while True:
            ok, f = cap.read()
            if not ok or t > max(quiero):
                break
            if t in quiero:
                r = quiero[t]
                x, y = float(r["x"]), float(r["y"])
                bw = float(r.get("w") or 0.8 * float(r["d"]))
                bh = float(r.get("h") or 0.8 * float(r["d"]))
                for c in range(1 if split == "val" else a.copias):
                    rec, x0, y0, T = recorte(f, x, y, rng)
                    base = f"{idx:05d}_{nombre}_{t:05d}_{r.get('fuente', 'extra')}{c}"
                    cv2.imwrite(f"{a.salida}/images/{split}/{base}.jpg", rec, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    open(f"{a.salida}/labels/{split}/{base}.txt", "w").write(
                        f"0 {(x - x0) / T:.6f} {(y - y0) / T:.6f} {bw / T:.6f} {bh / T:.6f}\n")
                    idx += 1
                    n += 1
            t += 1
        print(f"{video}: {n} recortes extra -> {split}")
    mosaico(a.salida, a.salida)


def main():
    """Línea de comandos: crear el dataset, descartar/conservar ids revisados o agregar etiquetas extra."""
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--modelo", default="yolo11n.pt", help="YOLO pre-entrenado que hace de profesor")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--salida", default="dataset")
    ap.add_argument("--val", default=None, help="nombre (o parte) del video que va a validación")
    ap.add_argument("--cada", type=int, default=2, help="guardar 1 de cada N cuadros con balón")
    ap.add_argument("--negativos", type=float, default=0.25, help="fracción de recortes sin balón")
    ap.add_argument("--descartar", nargs="*", type=int, help="ids de recortes mal etiquetados a borrar")
    ap.add_argument("--conservar", nargs="*", type=int,
                    help="ids de recortes BIEN etiquetados; todos los demás con etiqueta se descartan")
    ap.add_argument("--negativos-dificiles", type=int, default=0,
                    help="cuántos de los descartados se guardan como ejemplos negativos (sin etiqueta)")
    ap.add_argument("--extra", nargs="*", help="CSV con etiquetas extra (pipeline / manuales)")
    ap.add_argument("--carpeta-videos", default="videos")
    ap.add_argument("--copias", type=int, default=3, help="recortes por etiqueta extra")
    a = ap.parse_args()
    rng = random.Random(0)

    if a.extra:
        agregar_extra(a, rng)
        return

    if a.descartar is not None or a.conservar is not None:
        ids = set(a.descartar or [])
        if a.conservar is not None:                 # todo lo etiquetado que NO está en la lista
            buenos = set(a.conservar)
            for lab in glob.glob(f"{a.salida}/labels/*/*.txt"):
                i = int(os.path.basename(lab).split("_")[0])
                if os.path.getsize(lab) and i not in buenos:
                    ids.add(i)
        ids = sorted(ids)
        # una parte de los falsos positivos se guarda como ejemplo negativo difícil (sin etiqueta)
        negs = set(rng.sample(ids, min(len(ids), a.negativos_dificiles)))
        n = m = 0
        for i in ids:
            for img in glob.glob(f"{a.salida}/images/*/{i:05d}_*.jpg"):
                lab = img.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
                if i in negs:
                    open(lab, "w").close()
                    m += 1
                else:
                    os.remove(img)
                    os.remove(lab)
                    n += 1
        print(f"{n} recortes descartados, {m} convertidos en negativos difíciles")
        mosaico(a.salida, a.salida)
        return

    for s in ("train", "val"):
        os.makedirs(f"{a.salida}/images/{s}", exist_ok=True)
        os.makedirs(f"{a.salida}/labels/{s}", exist_ok=True)
    val = a.val or os.path.basename(a.videos[-1])
    idx, resumen = 0, []
    for video in a.videos:
        nombre = os.path.splitext(os.path.basename(video))[0]
        split = "val" if val in os.path.basename(video) else "train"
        print(f"{nombre}: detectando con el profesor ({a.modelo})...")
        dets, fps = detectar(a.modelo, video, a.conf, a.imgsz)
        cap = cv2.VideoCapture(video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        lab = etiquetas(dets, w, fps)
        n_det = sum(1 for v in lab.values() if not v[4])
        print(f"   {len(dets)} cuadros, {len(lab)} con balón ({n_det} detectados, "
              f"{len(lab) - n_det} interpolados) -> {split}")
        con = sorted(lab)[::a.cada]
        sin = [t for t in range(len(dets)) if t not in lab and not dets[t]]
        sin = rng.sample(sin, min(len(sin), int(len(con) * a.negativos / (1 - a.negativos)) + 1))
        quiero = set(con) | set(sin)
        t = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if t in quiero:
                h = f.shape[0]
                if t in lab:
                    x, y, bw, bh, _ = lab[t]
                    c, x0, y0, T = recorte(f, x, y, rng)
                    txt = f"0 {(x - x0) / T:.6f} {(y - y0) / T:.6f} {max(bw, 6) / T:.6f} {max(bh, 6) / T:.6f}\n"
                else:
                    c, x0, y0, T = recorte(f, rng.uniform(0, w), rng.uniform(0, h), rng)
                    txt = ""
                base = f"{idx:05d}_{nombre}_{t:05d}"
                cv2.imwrite(f"{a.salida}/images/{split}/{base}.jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 92])
                open(f"{a.salida}/labels/{split}/{base}.txt", "w").write(txt)
                idx += 1
            t += 1
        resumen.append((nombre, split, len(dets), len(lab), n_det, len(con), len(sin)))

    ruta = os.path.abspath(a.salida).replace("\\", "/")
    open(f"{a.salida}/data.yaml", "w").write(
        f"path: {ruta}\ntrain: images/train\nval: images/val\nnames:\n  0: balon\n")
    with open(f"{a.salida}/resumen.csv", "w") as fh:
        fh.write("video,split,cuadros,cuadros_con_balon,detectados_profesor,recortes_balon,recortes_negativos\n")
        for r in resumen:
            fh.write(",".join(map(str, r)) + "\n")
    n = mosaico(a.salida, a.salida)
    print(f"\n{idx} recortes ({n} con balón). Revisa {a.salida}/revision_*.jpg y borra los malos con "
          f"--descartar <ids>.\nDespués: python src/entrenar_yolo.py --datos {a.salida}/data.yaml")


if __name__ == "__main__":
    main()
