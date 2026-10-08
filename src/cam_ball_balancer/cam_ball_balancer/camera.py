"""Captura de camara USB (V4L2) configurada para tener una tasa estable.

Por que existe (diagnostico 2026-09-28): con `cv2.VideoCapture(idx)` sin
configurar, OpenCV abre la webcam en YUYV sin comprimir. Por USB 2.0,
YUYV a 720p no pasa de ~10 FPS, y con auto-exposicion en poca luz la
camara alarga el tiempo de exposicion y baja todavia mas la tasa. El
resultado medido fueron ~11 FPS, sin que el computo fuera el cuello.

Esta clase fija lo que controla la tasa:
  - FOURCC MJPG (la camara comprime -> cabe 640x480@30 en USB 2.0)
  - resolucion y FPS pedidos explicitamente
  - exposicion manual opcional (corta = menos motion blur y tasa fija)
  - balance de blancos fijo opcional (el color de la pelota no deriva)
  - buffer de 1 cuadro + hilo lector: el consumidor siempre recibe el
    cuadro MAS RECIENTE, nunca uno viejo encolado. Si el procesamiento
    es mas lento que la camara, se descartan cuadros (y se cuentan) en
    vez de acumular latencia.

Sin dependencia de ROS: se usa igual desde vision_node y desde
scripts/bench_camera.py.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    device: str | int = 0  # indice (0) o ruta (/dev/video0)
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str = "MJPG"  # "" = no tocar (deja el default del driver)
    # V4L2 exposure_time_absolute, unidades de 100 us (ej. 100 = 10 ms).
    # <= 0 deja la auto-exposicion. Con luz fija conviene manual y corta:
    # debe ser < 1/fps (33 ms a 30 FPS) o la camara no llega a la tasa.
    exposure: float = -1.0
    # Temperatura de color en K para balance de blancos fijo; <= 0 = auto.
    wb_temperature: float = -1.0


@dataclass
class Frame:
    image: np.ndarray
    seq: int  # contador de cuadros leidos de la camara
    t_wall: float  # time.time() al terminar la lectura (para el stamp ROS)
    t_mono: float  # time.monotonic() al terminar la lectura (para medir)


def _open_capture(device: str | int) -> cv2.VideoCapture:
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)  # backend por defecto (no-Linux)
    return cap


def fourcc_to_str(value: float) -> str:
    v = int(value)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


class RateStats:
    """Frecuencia y jitter sobre una ventana deslizante de periodos.
    Thread-safe: el hilo lector de la camara hace tick() mientras otro
    hilo lee summary()."""

    def __init__(self, window: int = 120):
        self._periods = deque(maxlen=window)
        self._last: Optional[float] = None
        self._lock = threading.Lock()

    def tick(self, t: float) -> None:
        with self._lock:
            if self._last is not None:
                self._periods.append(t - self._last)
            self._last = t

    def summary(self) -> dict:
        with self._lock:
            p = np.array(self._periods)
        if p.size == 0:
            return {"hz": 0.0, "period_ms": 0.0, "jitter_ms": 0.0, "period_max_ms": 0.0}
        mean = float(p.mean())
        return {
            "hz": 1.0 / mean if mean > 0 else 0.0,
            "period_ms": mean * 1e3,
            "jitter_ms": float(p.std()) * 1e3,
            "period_max_ms": float(p.max()) * 1e3,
        }


class Camera:
    """Hilo lector que mantiene solo el ultimo cuadro."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = _open_capture(config.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara {config.device!r}")
        self._apply_config()

        self._lock = threading.Condition()
        self._latest: Optional[Frame] = None
        self._seq = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Tasa a la que la camara ENTREGA cuadros (independiente de cuanto
        # tarde el procesamiento).
        self.rate = RateStats()

    def _apply_config(self) -> None:
        cfg = self.config
        cap = self.cap
        # El orden importa en V4L2: formato antes que tamano/FPS.
        if cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cfg.exposure > 0:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual
            cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2: 3 = aperture priority (auto)
        if cfg.wb_temperature > 0:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, cfg.wb_temperature)
        else:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    def actual_settings(self) -> dict:
        """Lo que el driver REALMENTE acepto (puede diferir de lo pedido)."""
        cap = self.cap
        return {
            "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
            "wb_temperature": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        }

    def start(self) -> "Camera":
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="camera_reader", daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        while self._running:
            ok, image = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            frame = Frame(image=image, seq=self._seq, t_wall=time.time(), t_mono=time.monotonic())
            self._seq += 1
            self.rate.tick(frame.t_mono)
            with self._lock:
                self._latest = frame
                self._lock.notify_all()

    def wait_next(self, after_seq: int, timeout: float = 1.0) -> Optional[Frame]:
        """Bloquea hasta que haya un cuadro con seq > after_seq; devuelve
        el mas reciente (los intermedios se descartan)."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._latest is None or self._latest.seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._running:
                    return None
                self._lock.wait(remaining)
            return self._latest

    def stop(self) -> None:
        self._running = False
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
