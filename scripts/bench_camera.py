#!/usr/bin/env python3
"""Mide la frecuencia REAL de la camara y del pipeline de vision, sin ROS.

Dos pruebas:
  1) --sweep: prueba formatos/resoluciones (YUYV vs MJPG) solo capturando,
     para ver cual entrega la tasa pedida con esta camara + este puerto USB.
  2) por defecto: captura con la config dada + pipeline completo (ArUco +
     pelota) y reporta Hz, jitter, tiempo por etapa y overhead de Python
     (tiempo total del cuadro menos lo que se va dentro de OpenCV).

    python3 scripts/bench_camera.py --device 0 --sweep
    python3 scripts/bench_camera.py --device 0 --seconds 20 \
        --params src/cam_ball_balancer/config/params.yaml

Correr en la Pi, headless. El resultado define si hace falta algo mas
(bajar resolucion, aruco_every_n > 1, otra camara o portar a C++).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "cam_ball_balancer"))

from cam_ball_balancer.camera import Camera, CameraConfig, RateStats  # noqa: E402
from cam_ball_balancer.platform_pose import CameraIntrinsics, PlatformPoseConfig  # noqa: E402
from cam_ball_balancer.ball_tracker import BallDetectorConfig  # noqa: E402
from cam_ball_balancer.vision_pipeline import PipelineConfig, VisionPipeline  # noqa: E402

DEFAULT_INTRINSICS = os.path.join(
    os.path.dirname(__file__), "..", "src", "cam_ball_balancer", "config", "camera_intrinsics.npz")


def capture_only(cfg: CameraConfig, seconds: float) -> dict:
    cam = Camera(cfg).start()
    actual = cam.actual_settings()
    time.sleep(1.0)  # descartar arranque (auto-exposicion, primeros cuadros)
    cam.rate = RateStats(window=100000)
    time.sleep(seconds)
    s = cam.rate.summary()
    cam.stop()
    return {**actual, **s}


def sweep(args) -> None:
    combos = [("YUYV", 640, 480), ("MJPG", 640, 480), ("YUYV", 1280, 720),
              ("MJPG", 1280, 720), ("MJPG", 320, 240)]
    print(f"{'pedido':<22}{'obtenido':<24}{'Hz':>7}{'jitter ms':>11}{'max ms':>9}")
    for fourcc, w, h in combos:
        cfg = CameraConfig(device=args.device, width=w, height=h, fps=args.fps, fourcc=fourcc,
                           exposure=args.exposure, wb_temperature=args.wb_temperature)
        try:
            r = capture_only(cfg, args.seconds)
        except RuntimeError as e:
            print(f"{fourcc} {w}x{h}: {e}")
            continue
        got = f"{r['fourcc']} {r['width']}x{r['height']}"
        print(f"{fourcc + f' {w}x{h}@{args.fps:g}':<22}{got:<24}{r['hz']:7.1f}"
              f"{r['jitter_ms']:11.2f}{r['period_max_ms']:9.1f}")


def load_pipeline_config(path: str | None) -> PipelineConfig:
    if not path:
        return PipelineConfig()
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    p = data.get("vision_node", {}).get("ros__parameters", {})
    flat = p.get("corner_positions_m")
    pose = PlatformPoseConfig(
        marker_length_m=p.get("marker_length_m", 0.03),
        corner_marker_ids=tuple(p.get("corner_marker_ids", (0, 1, 2, 3))),
        **({"corner_positions_m": tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))}
           if flat else {}),
    )
    det = BallDetectorConfig(
        hsv_lower=tuple(p.get("hsv_lower", (5, 120, 120))),
        hsv_upper=tuple(p.get("hsv_upper", (18, 255, 255))),
    )
    return PipelineConfig(
        detector=det, pose=pose,
        aruco_every_n=p.get("aruco_every_n", 1),
        ball_radius_m=p.get("ball_radius_m", 0.02),
        use_roi=p.get("use_roi", True),
    )


def full_pipeline(args) -> None:
    cfg = CameraConfig(device=args.device, width=args.width, height=args.height, fps=args.fps,
                       fourcc=args.fourcc, exposure=args.exposure, wb_temperature=args.wb_temperature)
    if os.path.isfile(args.intrinsics):
        intr = CameraIntrinsics.load(args.intrinsics)
    else:
        print(f"(sin {args.intrinsics}, usando identity_guess)")
        intr = CameraIntrinsics.identity_guess(args.width, args.height)
    pipe = VisionPipeline(load_pipeline_config(args.params), intr)

    cam = Camera(cfg).start()
    print("Camara:", cam.actual_settings())
    time.sleep(1.0)
    cam.rate = RateStats(window=100000)
    proc = RateStats(window=100000)

    aruco, ball, total, latency = [], [], [], []
    pose_ok = ball_ok = n = dropped = 0
    last_seq = -1
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        fr = cam.wait_next(last_seq)
        if fr is None:
            continue
        if last_seq >= 0:
            dropped += fr.seq - last_seq - 1
        last_seq = fr.seq
        t0 = time.perf_counter()
        res = pipe.process(fr.image, fr.t_mono)
        total.append((time.perf_counter() - t0) * 1e3)
        latency.append((time.monotonic() - fr.t_mono) * 1e3)
        proc.tick(time.monotonic())
        aruco.append(res.timings_ms["aruco_ms"])
        ball.append(res.timings_ms["ball_ms"])
        pose_ok += res.pose.valid
        ball_ok += res.ball.valid
        n += 1
    cam.stop()

    c, p = cam.rate.summary(), proc.summary()
    pct = lambda a, q: float(np.percentile(a, q)) if a else 0.0  # noqa: E731
    print(f"\nCuadros procesados: {n}  descartados: {dropped}")
    print(f"Camara entrega:     {c['hz']:6.1f} Hz  jitter {c['jitter_ms']:.2f} ms  max {c['period_max_ms']:.1f} ms")
    print(f"Pipeline procesa:   {p['hz']:6.1f} Hz  jitter {p['jitter_ms']:.2f} ms  max {p['period_max_ms']:.1f} ms")
    print(f"ArUco    mediana {pct(aruco, 50):5.1f} ms  p95 {pct(aruco, 95):5.1f} ms")
    print(f"Pelota   mediana {pct(ball, 50):5.1f} ms  p95 {pct(ball, 95):5.1f} ms")
    print(f"Total    mediana {pct(total, 50):5.1f} ms  p95 {pct(total, 95):5.1f} ms")
    print(f"Latencia captura->resultado mediana {pct(latency, 50):5.1f} ms  p95 {pct(latency, 95):5.1f} ms")
    print(f"Pose valida {100 * pose_ok / max(n, 1):.0f}%   pelota valida {100 * ball_ok / max(n, 1):.0f}%")
    budget = 1e3 / args.fps
    print(f"\nPresupuesto a {args.fps:g} FPS: {budget:.1f} ms/cuadro -> "
          + ("CABE" if pct(total, 95) < budget else "NO CABE (p95 excede)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--exposure", type=float, default=-1.0, help="100us units; <=0 auto")
    ap.add_argument("--wb-temperature", type=float, default=-1.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--intrinsics", default=DEFAULT_INTRINSICS)
    ap.add_argument("--params", default=None, help="params.yaml (seccion vision_node)")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    sweep(args) if args.sweep else full_pipeline(args)


if __name__ == "__main__":
    main()
