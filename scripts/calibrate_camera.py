#!/usr/bin/env python3
"""Calibracion de intrinsecos de camara (paso 1 del checklist "When the
hardware arrives" del README) usando un tablero de ajedrez estandar.

No depende de ROS -- corre con cualquier webcam via OpenCV directo, igual
que el resto del "core" del paquete (ver README, "Quickstart -- no ROS").

Uso interactivo (webcam en vivo, recomendado -- capturas mientras mueves
la camara/tablet):

    python3 scripts/calibrate_camera.py \
        --camera-index 0 \
        --square-size-mm 24.5 \
        --cols 10 --rows 7 \
        --out config/camera_intrinsics.npz

    Controles en la ventana de preview:
      ESPACIO  captura el frame actual (solo si se detecto el tablero)
      c        calibra con las capturas juntadas hasta ahora y sale
      q        sale sin calibrar

Uso por lote (si ya tienes fotos guardadas en una carpeta):

    python3 scripts/calibrate_camera.py \
        --images-dir ./calib_photos --square-size-mm 24.5 \
        --cols 10 --rows 7 --out config/camera_intrinsics.npz

El --square-size-mm NO se adivina: mide un cuadrado del tablero mostrado
(ver el Artifact "Tablero de Calibracion") con una regla real y pasa ese
numero. Un valor equivocado aqui escala mal TODA la geometria metrica
corriente abajo (platform_pose.py, homography.py) aunque los angulos de
tilt salgan bien.

El .npz de salida tiene las keys "camera_matrix" y "dist_coeffs", el
formato exacto que espera CameraIntrinsics.load() en platform_pose.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def build_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """Puntos 3D del tablero en su propio plano (z=0), en metros.

    cols/rows son el numero de CUADRADOS; cv2.findChessboardCorners busca
    esquinas INTERNAS, es decir (cols-1) x (rows-1) puntos.
    """
    inner_cols, inner_rows = cols - 1, rows - 1
    objp = np.zeros((inner_rows * inner_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def find_corners(gray: np.ndarray, inner_size: tuple[int, int]):
    found, corners = cv2.findChessboardCorners(
        gray, inner_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def calibrate(object_points_list, image_points_list, image_size):
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points_list, image_size, None, None,
    )
    return ret, camera_matrix, dist_coeffs


def run_interactive(args, inner_size, square_size_m):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara index={args.camera_index}", file=sys.stderr)
        sys.exit(1)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    objp = build_object_points(args.cols, args.rows, square_size_m)
    object_points_list, image_points_list = [], []
    image_size = None

    print("ESPACIO = capturar | c = calibrar y salir | q = salir sin calibrar")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Fallo leyendo un frame de la camara.", file=sys.stderr)
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        corners = find_corners(gray, inner_size)

        display = frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(display, inner_size, corners, True)
        cv2.putText(display, f"capturas: {len(object_points_list)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("calibrate_camera - cam_ball_balancer", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and corners is not None:
            object_points_list.append(objp)
            image_points_list.append(corners)
            print(f"Captura #{len(object_points_list)} guardada.")
        elif key == ord("c"):
            break
        elif key == ord("q"):
            object_points_list = []
            break

    cap.release()
    cv2.destroyAllWindows()
    return object_points_list, image_points_list, image_size


def run_batch(args, inner_size, square_size_m):
    paths = sorted(glob.glob(os.path.join(args.images_dir, "*")))
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not paths:
        print(f"No hay imagenes en {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    objp = build_object_points(args.cols, args.rows, square_size_m)
    object_points_list, image_points_list = [], []
    image_size = None

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        corners = find_corners(gray, inner_size)
        if corners is None:
            print(f"  [saltada] {os.path.basename(path)}: tablero no detectado")
            continue
        object_points_list.append(objp)
        image_points_list.append(corners)
        print(f"  [ok] {os.path.basename(path)}")

    return object_points_list, image_points_list, image_size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cols", type=int, default=10, help="cuadrados por columna del tablero (default: 10, como el Artifact)")
    parser.add_argument("--rows", type=int, default=7, help="cuadrados por fila del tablero (default: 7)")
    parser.add_argument("--square-size-mm", type=float, required=True,
                         help="lado de un cuadrado MEDIDO con regla, en mm (no asumir)")
    parser.add_argument("--out", default="config/camera_intrinsics.npz")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--images-dir", default=None,
                         help="si se da, calibra por lote desde fotos ya tomadas en vez de abrir la camara")
    args = parser.parse_args()

    inner_size = (args.cols - 1, args.rows - 1)
    square_size_m = args.square_size_mm / 1000.0

    if args.images_dir:
        object_points_list, image_points_list, image_size = run_batch(args, inner_size, square_size_m)
    else:
        object_points_list, image_points_list, image_size = run_interactive(args, inner_size, square_size_m)

    if len(object_points_list) < 8:
        print(f"Solo {len(object_points_list)} capturas validas (recomendado >=15, minimo util ~8). "
              "No se calibro.", file=sys.stderr)
        sys.exit(1)

    ret, camera_matrix, dist_coeffs = calibrate(object_points_list, image_points_list, image_size)
    print(f"\nRMS reprojection error: {ret:.4f} px "
          "(bien calibrado suele quedar < 0.5 px; > 1 px revisar capturas/square-size)")
    print("camera_matrix:\n", camera_matrix)
    print("dist_coeffs:\n", dist_coeffs.ravel())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"\nGuardado en {args.out} -- cargalo con CameraIntrinsics.load('{args.out}') "
          "en vez de identity_guess().")


if __name__ == "__main__":
    main()
