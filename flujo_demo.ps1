# Flujo de la demo de la presentación final (PowerShell, desde la carpeta HandStats)
# Requiere videos/rusia_gol1.webm (ver videos/DESCARGAR_VIDEOS.html)

# 1. (opcional) revisar el arco marcado en el cuadro de referencia, sin abrir ventana
python src\marcar_esquinas.py videos\rusia_gol1.webm --cuadro 96 --esquinas 1205,160 1405,160 1405,740 1205,578 --captura salida\esquinas_rusia_gol1.png

# 2. procesar el lanzamiento con el detector clásico (~1 min)
python src\pipeline_balon.py videos\rusia_gol1.webm --desde 70 --hasta 110 --ref 96 --esquinas 1205,160 1405,160 1405,740 1205,578 --salida salida\demo

# 3. (opcional) el mismo lanzamiento con el YOLO propio
python src\pipeline_balon.py videos\rusia_gol1.webm --desde 70 --hasta 110 --ref 96 --esquinas 1205,160 1405,160 1405,740 1205,578 --detector yolo --salida salida\demo_yolo

# 4. pruebas del clasificador (gol, atajada, poste, fuera)
python tests\test_clasificacion.py

# 5. mostrar resultados
Get-Content salida\demo\resultado.txt
start salida\demo\demo_zoom_lento.mp4
start salida\demo\trayectoria_balon.png
