"""
Clasificación del resultado de un lanzamiento a partir de la trayectoria del balón.

Resultados posibles:
  GOL       el balón entra al arco y la red lo frena (o desaparece dentro del arco)
  ATAJADA   el balón llega a la zona del arco y rebota / se desvía de golpe
            -> el punto de rebote es donde ataja el arquero (en cm del arco)
  POSTE     igual que la atajada, pero el rebote ocurre pegado al marco
  FUERA     el balón pasa cerca del arco sin entrar ni rebotar
  sin llegada al arco   la pista nunca se acerca al arco

Hay dos variantes según el montaje de la cámara:
  clasificar_lateral()  cámara de lado / en mano (pipeline_balon.py). Se usa la dirección
                        del movimiento en la imagen para detectar el rebote.
  clasificar_frontal()  cámara fija detrás del lanzador (demo_arco.py). El balón se aleja
                        de la cámara, así que su diámetro aparente baja; si vuelve a crecer
                        después de llegar al arco, el balón rebotó (atajada o poste).
"""
import cv2
import numpy as np

ARCO_W, ARCO_H = 300.0, 200.0           # cm, medidas interiores del arco
FILAS, COLS = ["alta", "media", "baja"], ["izq", "centro", "der"]
MARGEN_LLEGADA = 25.0                   # cm alrededor del arco donde se considera "llegada"
MARGEN_POSTE = 12.0                     # cm al marco para decir que pegó en el poste/travesaño


def a_cm(H, pts):
    pts = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def zona(X, Y):
    X = float(np.clip(X, 0, ARCO_W - 1e-3))
    Y = float(np.clip(Y, 0, ARCO_H - 1e-3))
    return f"{FILAS[int(Y // (ARCO_H / 3))]}-{COLS[int(X // (ARCO_W / 3))]}"


def dentro_arco(X, Y, m=0.0):
    return -m <= X <= ARCO_W + m and -m <= Y <= ARCO_H + m


def dist_marco(X, Y):
    d_post = min(abs(X), abs(X - ARCO_W)) if -MARGEN_LLEGADA <= Y <= ARCO_H else 1e9
    d_trav = abs(Y) if -MARGEN_LLEGADA <= X <= ARCO_W + MARGEN_LLEGADA else 1e9
    return min(d_post, d_trav)


def _vel(p):
    p = np.array([q[:3] for q in p], float)
    return np.diff(p[:, 1:3], axis=0) / np.maximum(np.diff(p[:, 0]), 1)[:, None]


def _resultado(tipo, t, X, Y, fps, **extra):
    r = {"resultado": tipo, "zona": zona(X, Y) if tipo in ("GOL", "ATAJADA") else "-",
         "t_s": round(float(t) / fps, 2), "x_cm": round(float(X)), "y_cm": round(float(Y))}
    r.update(extra)
    return r


def clasificar_lateral(balon, H, fps, cos_rebote=0.0, frenado=0.4):
    if len(balon) < 4:
        return {"resultado": "sin llegada al arco", "zona": "-"}
    P = a_cm(H, [(x, y) for _, x, y in balon])
    v = _vel(balon)
    rap = np.hypot(v[:, 0], v[:, 1])

    # 1) llegada: primer punto dentro del arco ampliado por un margen
    lleg = [i for i, (X, Y) in enumerate(P) if dentro_arco(X, Y, MARGEN_LLEGADA)]
    if not lleg:
        k = int(np.argmin([max(0, -X, X - ARCO_W) + max(0, -Y, Y - ARCO_H) for X, Y in P]))
        X, Y = P[k]
        cerca = max(0, -X, X - ARCO_W) + max(0, -Y, Y - ARCO_H) < 80
        if cerca:
            return _resultado("FUERA", balon[k][0], X, Y, fps, i=k)
        return {"resultado": "sin llegada al arco", "zona": "-"}
    i0 = max(1, lleg[0])

    # velocidad de llegada: mediana de los pasos rápidos antes de llegar
    va = v[max(0, i0 - 5):i0]
    va = va[np.hypot(va[:, 0], va[:, 1]) > 0.5 * np.median(rap[:i0] + 1e-9)] if len(va) else va
    if len(va) == 0:
        va = v[:max(1, i0)]
    v_in = np.median(va, axis=0)
    s_in = float(np.hypot(*v_in)) + 1e-9

    # 2) rebote: después de llegar, un paso que vuelve hacia atrás o se desvía de golpe
    fin = min(len(v), i0 + int(0.5 * fps) + 1)
    for j in range(i0 - 1, fin):
        s = rap[j]
        if s < 0.35 * s_in:
            continue
        c = float(v[j] @ v_in) / (s * s_in)
        if c < cos_rebote:
            X, Y = P[j]                                  # punto donde cambió la dirección
            if dist_marco(X, Y) < MARGEN_POSTE:
                return _resultado("POSTE", balon[j][0], X, Y, fps, i=j)
            if dentro_arco(X, Y, MARGEN_POSTE):
                return _resultado("ATAJADA", balon[j][0], X, Y, fps, i=j)
            return _resultado("FUERA", balon[j][0], X, Y, fps, i=j)

    # 3) sin rebote: si entra y se frena (la red) es gol
    ent = [i for i in range(i0, len(P)) if dentro_arco(*P[i])]
    if ent:
        k = ent[0]
        vd = rap[k:k + 8]
        k_in = sum(dentro_arco(*P[q]) for q in range(k, min(len(P), k + 8)))
        # punto de impacto: donde la red frena el balón (primer paso lento después de entrar).
        # Grabando de lado, el punto donde la trayectoria cruza el borde del arco en la imagen
        # es casi siempre el poste más cercano a la cámara, no el lugar real del tiro.
        lentos = [j for j in range(k, len(rap)) if rap[j] < frenado * s_in and dentro_arco(*P[j])]
        k = lentos[0] if lentos else k
        X, Y = P[k]
        if (len(vd) and np.median(vd) < frenado * s_in and k_in >= 3) or k == len(P) - 1:
            return _resultado("GOL", balon[k][0], X, Y, fps, i=k)
        return _resultado("GOL", balon[k][0], X, Y, fps, i=k, confianza="baja")
    X, Y = P[i0]
    return _resultado("FUERA", balon[i0][0], X, Y, fps, i=i0)


def clasificar_frontal(pista, H, d_esp, fps, crece=1.25):
    if len(pista) < 5:
        return None
    ds = np.array([p[3] for p in pista], float)
    ds = np.convolve(np.pad(ds, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    i = int(np.argmin(np.abs(ds - d_esp)))              # llegada al plano del arco
    t, x, y = pista[i][:3]
    X, Y = a_cm(H, [(x, y)])[0]
    # rebote: después de llegar el balón vuelve hacia la cámara (diámetro crece)
    desp = ds[i + 1:i + 1 + int(0.4 * fps)]
    rebote = len(desp) >= 3 and desp.max() > crece * ds[i]
    if rebote:
        if dist_marco(X, Y) < MARGEN_POSTE:
            return _resultado("POSTE", t, X, Y, fps, i=i)
        if dentro_arco(X, Y):
            return _resultado("ATAJADA", t, X, Y, fps, i=i)
    if dentro_arco(X, Y):
        return _resultado("GOL", t, X, Y, fps, i=i)
    return _resultado("FUERA", t, X, Y, fps, i=i)
