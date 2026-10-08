# Diagnostico de tasa del vision_node: con esto se demuestra (y se
# vigila en vivo) que la vision entrega datos a frecuencia consistente.
std_msgs/Header header

# Frecuencia de procesamiento real (ventana deslizante) y su jitter
# (desviacion estandar del periodo entre cuadros procesados).
float32 rate_hz
float32 period_ms
float32 jitter_ms
float32 period_max_ms

# Frecuencia a la que la camara entrega cuadros (antes de procesar).
float32 camera_hz

# Tiempo por etapa del ultimo cuadro.
float32 aruco_ms
float32 ball_ms
float32 publish_ms

# Captura -> datos publicados (lo que "ve tarde" el control).
float32 latency_ms

# Cuadros de camara descartados porque el procesamiento iba atrasado
# (acumulado desde el arranque).
uint32 dropped_frames
