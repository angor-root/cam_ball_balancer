#!/usr/bin/env python3
"""Genera la hoja imprimible con los 4 tags ArUco que platform_pose.py
necesita en las esquinas de la plataforma (ver README: "no hay encoder
para el angulo de la plataforma, la unica fuente de verdad es vision").

Usa los MISMOS valores que config/params.yaml (dictionary_name,
marker_length_m, corner_marker_ids) para que el tag impreso coincida
exactamente con lo que el nodo espera detectar -- si cambias esos
parametros, vuelve a correr este script.

    python3 scripts/generate_aruco_tags.py --out ~/Downloads/tags_aruco.png

IMPORTANTE al imprimir: en el dialogo de impresion, usa "Tamano real" /
"Actual size" / escala 100% -- NUNCA "Ajustar a pagina" ("Fit to page"),
que reescala y arruina el marker_length_m real. La hoja trae una barra
de 30mm para que verifiques con una regla que el impreso salio a escala.
"""
from __future__ import annotations

import argparse

import cv2
import cv2.aruco as aruco
import numpy as np

# Mismos defaults que PlatformPoseConfig en platform_pose.py / params.yaml.
DICTIONARY_NAME = "DICT_4X4_50"
MARKER_LENGTH_M = 0.03
# (id, etiqueta de esquina) en el mismo orden que corner_marker_ids /
# corner_positions_m en params.yaml: TL, TR, BR, BL.
CORNERS = [
    (0, "TL - arriba-izquierda"),
    (1, "TR - arriba-derecha"),
    (2, "BR - abajo-derecha"),
    (3, "BL - abajo-izquierda"),
]

PX_PER_MM = 12.0  # 304.8 DPI exactos -> tamanos en mm caen en pixel entero
QUIET_ZONE_MM = 15.0  # margen blanco alrededor de cada marker (>= 1 modulo, generoso para cortar)
LABEL_HEIGHT_MM = 12.0
GAP_MM = 10.0
SCALE_BAR_MM = 30.0


def mm(v_mm: float) -> int:
    return int(round(v_mm * PX_PER_MM))


def make_marker_dict():
    dict_id = getattr(aruco, DICTIONARY_NAME)
    if hasattr(aruco, "ArucoDetector"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def draw_marker(dictionary, marker_id: int, side_px: int) -> np.ndarray:
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker(dictionary, marker_id, side_px)
    return aruco.drawMarker(dictionary, marker_id, side_px)


def put_text_centered(img, text, cx, cy_top, scale=0.6, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(img, text, (cx - tw // 2, cy_top + th), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="tags_aruco.png")
    args = parser.parse_args()

    dictionary = make_marker_dict()
    marker_side_px = mm(MARKER_LENGTH_M * 1000.0)
    quiet_px = mm(QUIET_ZONE_MM)
    label_px = mm(LABEL_HEIGHT_MM)
    gap_px = mm(GAP_MM)

    cell_w = marker_side_px + 2 * quiet_px
    cell_h = marker_side_px + 2 * quiet_px + label_px

    cols, rows = 2, 2
    scale_bar_px = mm(SCALE_BAR_MM)
    header_px = mm(20.0)
    sheet_w = cols * cell_w + (cols + 1) * gap_px
    sheet_h = header_px + rows * cell_h + (rows + 1) * gap_px

    sheet = np.full((sheet_h, sheet_w), 255, dtype=np.uint8)

    # Barra de escala de verificacion (30mm reales) + texto.
    bar_y = mm(6.0)
    cv2.line(sheet, (gap_px, bar_y), (gap_px + scale_bar_px, bar_y), 0, 3)
    cv2.line(sheet, (gap_px, bar_y - 6), (gap_px, bar_y + 6), 0, 2)
    cv2.line(sheet, (gap_px + scale_bar_px, bar_y - 6), (gap_px + scale_bar_px, bar_y + 6), 0, 2)
    cv2.putText(sheet, f"{SCALE_BAR_MM:.0f} mm - medir con regla tras imprimir",
                (gap_px + scale_bar_px + mm(4.0), bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    for i, (marker_id, label) in enumerate(CORNERS):
        r, c = divmod(i, cols)
        x0 = gap_px + c * (cell_w + gap_px)
        y0 = header_px + gap_px + r * (cell_h + gap_px)

        marker_img = draw_marker(dictionary, marker_id, marker_side_px)
        sheet[y0 + quiet_px:y0 + quiet_px + marker_side_px,
              x0 + quiet_px:x0 + quiet_px + marker_side_px] = marker_img

        cv2.rectangle(sheet, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), 180, 1)
        put_text_centered(sheet, f"ID {marker_id}  ({MARKER_LENGTH_M*1000:.0f}mm)",
                           x0 + cell_w // 2, y0 + marker_side_px + 2 * quiet_px, scale=0.55)
        put_text_centered(sheet, label, x0 + cell_w // 2,
                           y0 + marker_side_px + 2 * quiet_px + mm(6.0), scale=0.5)

    cv2.imwrite(args.out, sheet)
    print(f"Guardado {args.out} ({sheet_w}x{sheet_h}px @ {PX_PER_MM*25.4:.1f} DPI)")
    print("Al imprimir: escala 100% / Tamano real (NO 'Ajustar a pagina').")
    print(f"Diccionario: {DICTIONARY_NAME}, marker_length_m: {MARKER_LENGTH_M} "
          f"(cambia esto en config/params.yaml si hace falta, y vuelve a correr el script).")


if __name__ == "__main__":
    main()
