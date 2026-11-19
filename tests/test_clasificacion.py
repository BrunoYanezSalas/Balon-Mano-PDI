"""
Pruebas del clasificador de lanzamientos (src/clasificacion.py).

Correr con:   python -m pytest tests -v      (o simplemente: python tests/test_clasificacion.py)

1. Caso real: la trayectoria detectada en el video de la demo de avance (docs/trayectoria_balon.csv)
   tiene que dar GOL, como se vio en el video.
2. Casos construidos sobre esa misma trayectoria real: se conserva la llegada del balón y se cambia
   lo que pasa después (rebote hacia atrás = atajada, rebote en el marco = poste, pasar de largo = fuera).
3. Cámara frontal: diámetro aparente que baja hasta el arco y después sube (rebote) o no.
"""
import csv
import os
import sys

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from clasificacion import clasificar_frontal, clasificar_lateral  # noqa: E402

ESQ = np.float32([[37, 399], [128, 399], [128, 487], [37, 490]])     # arco en el video de la demo
H = cv2.getPerspectiveTransform(ESQ, np.float32([[0, 0], [300, 0], [300, 200], [0, 200]]))
HI = np.linalg.inv(H)
FPS = 60.0


def a_px(X, Y):
    return cv2.perspectiveTransform(np.float32([[[X, Y]]]), HI)[0, 0]


def demo():
    with open(os.path.join(RAIZ, "docs", "trayectoria_balon.csv")) as fh:
        return [(int(r["frame"]), float(r["x_px_ref"]), float(r["y_px_ref"]))
                for r in csv.DictReader(fh)]


def llegada(X, Y, n=8, paso=11.0):
    """Balón que viene desde la derecha (como en la demo) hasta el punto (X, Y) del arco."""
    x, y = a_px(X, Y)
    return [(i, x + paso * (n - i), y - 0.4 * (n - i)) for i in range(n + 1)]


def test_demo_real_es_gol():
    r = clasificar_lateral(demo(), H, FPS)
    assert r["resultado"] == "GOL"
    assert r["zona"].startswith("alta")


def test_atajada_rebote():
    t = llegada(150, 100)                         # llega al centro del arco...
    t0, x, y = t[-1]
    t += [(t0 + k, x + 6 * k, y - 2 * k) for k in range(1, 6)]  # ...y vuelve hacia la cancha
    r = clasificar_lateral(t, H, FPS)
    assert r["resultado"] == "ATAJADA", r
    assert r["zona"] == "media-centro"
    assert abs(r["x_cm"] - 150) < 40 and abs(r["y_cm"] - 100) < 40


def test_poste():
    t = llegada(298, 60)                          # llega pegado al poste derecho
    t0, x, y = t[-1]
    t += [(t0 + k, x + 7 * k, y + 1 * k) for k in range(1, 6)]
    assert clasificar_lateral(t, H, FPS)["resultado"] == "POSTE"


def test_fuera_por_arriba():
    t = llegada(150, -60, n=14)                   # pasa 60 cm por encima del travesaño
    t0, x, y = t[-1]
    t += [(t0 + k, x - 11 * k, y + 0.4 * k) for k in range(1, 6)]
    assert clasificar_lateral(t, H, FPS)["resultado"] == "FUERA"


def test_gol_red_frena():
    t = llegada(60, 160)                          # abajo a la izquierda (zona baja-izq)
    t0, x, y = t[-1]
    t += [(t0 + k, x - 1.0 * k, y + 0.3 * k) for k in range(1, 8)]  # la red lo frena
    r = clasificar_lateral(t, H, FPS)
    assert r["resultado"] == "GOL" and r["zona"] == "baja-izq", r


def test_frontal():
    d_esp = 12.0
    x, y = a_px(150, 100)
    ida = [(i, x, y, 40 - 28 * i / 10) for i in range(11)]         # se aleja: 40 px -> 12 px
    vuelta = [(11 + i, x, y, 12 + 4 * (i + 1)) for i in range(6)]  # rebota hacia la cámara
    quieto = [(11 + i, x, y, 11.5) for i in range(6)]              # se queda en la red
    assert clasificar_frontal(ida + vuelta, H, d_esp, FPS)["resultado"] == "ATAJADA"
    assert clasificar_frontal(ida + quieto, H, d_esp, FPS)["resultado"] == "GOL"


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("OK ", nombre)
